# Speech Commands Support — stylegan2-audio Training Repo Spec

> **Claude Code handoff.** This spec adds a **Speech Commands (GSC v0.02)** data path
> to the audio GAN training repo so it can train a *second* generator (`z → mel`) with
> its own frozen checkpoint + sidecar, exactly parallel to the AudioMNIST generator.
>
> **`GAN_TRAINING_SPEC.md` remains the source of truth.** The mel contract, global-affine
> *mechanism*, canvas/offset layout, single-channel model patch, `NpyMelDataset`, the
> `train.py` audio path, the R1 sweep, and the sidecar writer are **already locked and
> unchanged**. This document only adds a new data source, a new speaker split, a new
> per-dataset affine *value*, and a new eval label provider.
>
> **Verify every assumption against the actual repo before editing.** Function names,
> flag names, and file layout below are described by intent — confirm the real signatures
> (`prepare_audio_data.py`, the AudioMNIST label provider, the convergence-eval module,
> the R1 sweep script) and adapt. Surface any conflict with `GAN_TRAINING_SPEC.md`
> rather than working around it.

---

## 0. Scope & invariants

- **One generator per dataset** (locked, spec §5). Speech Commands gets its own `.pt` +
  `.json` sidecar + `speaker_split.json`. Do **not** merge datasets or build a
  cross-dataset prior in this repo.
- **The frozen mel contract is byte-identical to AudioMNIST** (spec §2): sr 22050,
  n_fft 1024, win 1024, hop 256, n_mels 80, fmin 0, fmax 8000, clip 1.0 s. Same BigVGAN
  `get_mel_spectrogram` extractor. Same 128×128 canvas, same `offset`, same
  `pad_value_normalized = -1`. Because the contract is identical and the clip is 1.0 s,
  the real region `T` **must come out the same (~86)** — derive it empirically and
  **assert it equals the AudioMNIST `T`**; if it differs, something drifted, stop and report.
- **Only these values are per-dataset:** `m_hi` (affine max), the dataset name, the
  train/held-out speaker lists, and `split_seed`. Everything else in the sidecar is
  identical to AudioMNIST.
- **Do NOT use TFDS `speech_commands`.** It exposes only `{audio, label}` with `label`
  collapsed to 12 classes and **speaker ID stripped** — it cannot support the id-disjoint
  speaker split, which is the whole point. Prep from the raw tarball only.

---

## 1. Data source — raw GSC v0.02 tarball

- Download on the Linux/scratch box (no local download needed):
  `http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz` (~2.3 GB).
- Extract to a source dir on **ext4 scratch** (not DrvFs `/mnt/d`).
- Structure: one folder per word (the folder name **is** the content label). 35 words,
  105,829 clips, 2,618 speakers, **16 kHz**, 16-bit WAV, each **≤ 1.0 s**.
- **Filename → speaker id:** `HASH_nohash_N.wav` where `HASH` is an 8-hex speaker id =
  the substring **before the first underscore**. Anything after `_nohash_` is the
  utterance index and is irrelevant to identity. Parse `speaker = fname.split('_')[0]`;
  **verify this parse against a real directory listing on first run** (assert every
  training filename matches `^[0-9a-f]+_nohash_\d+\.wav$`; log any that don't).
- **The 35 words include the spoken digits zero–nine** — this is what makes the SC prior
  usable as an OOD prior for AudioMNIST. Do not drop them.
- **Exclude `_background_noise_/` entirely.** Those are minutes-long noise files; they are
  not 1-s speech and must never enter the mel `.npy` set. Skip the folder in the walk.
- Ignore `validation_list.txt` / `testing_list.txt` — those are KWS splits, not the
  attack split. We build our own speaker split (§3).

---

## 2. Data-prep — add a `speech_commands` branch to `prepare_audio_data.py`

Reuse the **exact ordered pipeline** from `GAN_TRAINING_SPEC.md §3`. The only SC-specific
parts are the source walk, the speaker parse, and the 16 kHz input rate.

Per clip, in this order (unchanged from spec §3 except step 0):

0. **Enumerate** by walking word folders, skipping `_background_noise_`. Record
   `(path, word_label, speaker_hash)` for each clip.
1. **Load + resample** 16 kHz → **22050 Hz**, mono. (AudioMNIST resampled from 48 k; only
   the source rate differs — reuse the same resample call.)
2. **Pad/truncate the WAVEFORM** to exactly 22050 samples. SC clips are frequently *shorter*
   than 1 s, so most will be zero-padded (append silence). Never pad in mel space.
3. **Extract** with BigVGAN `get_mel_spectrogram()` → physical log-mel `(80, T)`.
4. **Clamp + global affine → `[-1,1]`** using the **SC-specific `m_hi`** (§4).
5. **Embed into the 128×128 canvas** at the **same frozen `offset`** as AudioMNIST; fill
   `-1`.
6. **Save float32 `.npy` `(1,128,128)`.** Never PNG.

Carry `word_label` and `speaker_hash` through to the prep manifest so the split (§3) and
the eval label provider (§5) can consume them without re-parsing filenames.

**Expected volume:** holding out 200 speakers (§3) leaves ~2,418 training speakers,
≈ 97k clips → ≈ 97k `.npy` at ~65 KB each ≈ 6 GB. Fine on ext4 scratch. (~4× AudioMNIST.)

---

## 3. id-disjoint speaker split — hash-based, hold out 200

Locked decisions: **hold out 200 speakers**, drawn by seeded shuffle of the unique speaker
hashes. (Cross-repo contract, spec §3/§9.)

- Collect the set of unique speaker hashes (should be ~2,618 — assert count, log it).
- Seeded shuffle; take 200 as `held_out_speakers`, the remaining ~2,418 as
  `train_speakers`. Use the same `--seed` that seeds all RNGs.
- **GAN training data = all clips of `train_speakers` only.** Held-out speakers' clips are
  **not** used for training (they are the attack pool, consumed dlg-sonic-side).
- The **400 attack targets** are drawn from `held_out_speakers` — record that this draw
  exists, but the *target selection itself is attack-side*; this repo only needs the
  train/held partition to (a) exclude held from training and (b) write the lists into the
  sidecar. Do not down-select the held-out clips here.
- Emit `speaker_split.json` (`train_speakers` / `held_out_speakers` / `split_seed`) +
  the prep manifest (`T`, `m_lo`, `m_hi`, `offset`). Assert the two lists are disjoint.

> Do **not** do a naive sample-level split. Split by hash first; every utterance of a held
> speaker stays held. (Same rule as AudioMNIST — mirror that code path.)

---

## 4. Per-dataset affine — recompute `m_hi` over SC train, freeze

- `m_lo = log(1e-5) ≈ -11.5129` (fixed by the extractor clamp, same as AudioMNIST).
- **`m_hi` = recomputed over the SC `train_speakers` set only** (e.g. 99.9th percentile of
  physical log-mel, then clip), **frozen once**. **Do NOT reuse AudioMNIST's `m_hi`.**
- Persist `m_lo`, `m_hi` in the SC sidecar.

> Note for the OOD plan (SC prior → AudioMNIST attack): the affines stay **independent**.
> Cross-dataset representational range is an *attack-side* concern handled in dlg-sonic
> (rung-2). Nothing about OOD changes this repo — do not widen `m_hi` to cover AudioMNIST.

---

## 5. Convergence eval — add an SC label provider (speaker + word)

The convergence monitor (domain-classifier Fréchet distance, mel-native + class-coverage
entropy, `--label-mode both`) already runs for AudioMNIST in speaker-mode + digit-mode.
For SC:

- **Speaker-mode:** class = `speaker_hash` (unchanged mechanism; new label source).
- **Content-mode:** class = **word (all 35)** — the analog of AudioMNIST digit-mode.
  (Decision: cover all words. This is a reporting/coverage signal, not a checkpoint
  selector.)
- **Locate how AudioMNIST wires its label provider into the eval and mirror it** — add an
  SC provider that reads `speaker_hash` / `word_label` from the prep manifest rather than
  re-deriving from filenames. Confirm the eval's expected label interface in the repo
  before wiring.
- **Primary training signal stays speaker-coverage entropy** (established finding: content
  entropy structurally cannot detect speaker-manifold collapse). Word entropy is the
  secondary/content axis and the OOD digit-coverage check.
- **Optional OOD add (cheap, recommended):** a 10-class *digit-subset* content-coverage
  read over the zero–nine words, so you can report that the SC prior's samples cover the
  digit sub-manifold — the credibility hook for the SC→AudioMNIST attack.

---

## 6. Sidecar — SC values only

Identical schema to spec §6. Only these fields differ from AudioMNIST:

- `"dataset": "speech_commands"`
- `"affine": {"m_lo": -11.5129, "m_hi": <SC frozen value>}`
- `"train_speakers": [...]`, `"held_out_speakers": [...]` (hashes), `"split_seed": <seed>`
- `"mel_shape": [80, T]` with the **empirically derived `T` asserted equal to AudioMNIST's**

Everything else (`mel_config`, `vocoder`, `canvas`, `offset`, `pad_value_normalized`,
`num_ws=12`, `w_dim=512`, `w_avg`) is identical. Reuse the existing sidecar writer as-is.

---

## 7. Training + R1 sweep — parametrize, don't rewrite

- `train.py --dataset npy <sc_npy_dir>` is **unchanged** — the float32 `.npy (1,128,128)`
  interface is dataset-agnostic. No flip, no ToTensor-rescale, no Normalize (spec §5).
- **R1 sweep script:** confirm whether it hardcodes the AudioMNIST npy path / eval labels.
  If so, parametrize `--data <npy_dir>` + the eval label source so the *same* sweep runs on
  SC. The sweep logic (2–3 `--r1` values on short runs, pick by mel-FID, then long) is
  identical.
- **Augmentation stays OFF** — ~97k clips is well past the overfitting regime (spec §5;
  SC "does not need it"). Never enable geometric/color/x-flip on mels.
- **Hardware:** full run on AWS g6e / L40S (Ada); laptop RTX 5070 Ti for smoke only.
  Expect longer wall-clock than AudioMNIST (~4× data). Keep the npy dir on ext4.

---

## 8. Validation before handoff (spec §7 ladder — same, on SC)

1. **Overfit-one-sample smoke first** (plumbing only): one SC clip end-to-end
   data-prep → train → checkpoint+sidecar → generate → crop → inverse SC-affine →
   BigVGAN → intelligible.
2. **Round-trip a real SC clip** (no GAN): wav → mel → affine → canvas → crop → inverse
   affine → BigVGAN. Must be intelligible — isolates the mel/affine/vocoder chain.
3. **GT-mel-vocode upper bound** recorded for a few SC clips.
4. **Sample the GAN:** `z → G → crop → inverse affine → BigVGAN`. Confirm plausible speech,
   audible **speaker variety**, and — for the OOD story — audible **digit words** among
   samples.
5. **Sidecar round-trip:** reconstruct crop + inverse affine from sidecar fields alone;
   confirm equal to the prep code's output.

---

## 9. Gotchas / hard "do NOT" list (SC-specific, on top of spec §8)

- Do **not** use TFDS `speech_commands` — no speaker id, 12-class collapse.
- Do **not** include `_background_noise_/` — long noise files, not 1-s speech.
- Do **not** apply per-utterance CMVN / mean-variance normalization (some SC pipelines do;
  it breaks the global-affine contract). Global affine only.
- Do **not** reuse AudioMNIST's `m_hi` — recompute over SC train.
- Do **not** hardcode `T` — derive empirically and assert it equals AudioMNIST's.
- Speaker id = filename prefix before the first `_`; verify the regex on first run.
- Do **not** pad in mel space — pad the waveform to 22050 (most SC clips are short).
- Disjointness from AudioMNIST is automatic (different corpus), but still emit
  `speaker_split.json` for reproducibility and the sidecar verification on the attack side.

---

## 10. Definition of done

- `prepare_audio_data.py --dataset speech_commands <SRC> --out <NPY_DIR> --seed <S>`
  produces ~97k `.npy (1,128,128)` float32 in `[-1,1]` (train speakers only),
  `speaker_split.json` (200 held out, disjoint, seeded), and a prep manifest with
  `T` (== AudioMNIST), `m_lo`, SC `m_hi`, `offset`.
- Convergence eval runs `--label-mode both` on SC (speaker + 35-word), speaker entropy as
  primary signal.
- R1 sweep + `train.py --dataset npy` run on the SC npy dir with no code changes beyond
  path/label parametrization.
- Checkpoint ships with an SC sidecar whose only non-AudioMNIST fields are dataset name,
  SC affine, speaker lists, split seed.
- Section 8 validation items 1–5 pass before handoff.

> Attack-side wiring (SC targets, SC→AudioMNIST OOD, victim model, rung-2/rung-3) lives in
> **dlg-sonic** and is out of scope for this repo — covered by the separate dlg-sonic spec.

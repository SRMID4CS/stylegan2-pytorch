# Speech Commands handoff → dlg-sonic

> **Handoff doc for the attack repo.** The `stylegan2-audio` side of the Speech
> Commands (GSC v0.02) prior is implemented; this describes exactly what ships,
> what dlg-sonic has to build, and which code must be copied byte-identically.
>
> `GAN_TRAINING_SPEC.md` §2/§4/§6/§9 and `SPEECH_COMMANDS_SPEC.md` remain the
> source of truth. **Nothing here changes the attack architecture** — SC is a
> second dataset, not a second design. If you have already wired the AudioMNIST
> prior, §5 is the only genuinely new work.
>
> **Verify every path and field against the actual artifacts before coding.**
> Schemas below are described by intent and were read off the real code on
> 2026-08-04; confirm against the sidecar you are actually handed.

---

## 0. The one thing that will bite you

**The training repo writes `.npy` files for TRAIN speakers only.** Held-out
speakers — your attack targets — are *never* prepared, by design (that is the
id-disjoint contract: their audio must never touch the GAN's data dir).

So dlg-sonic **must build its own target mels from the raw GSC tarball**, using
the frozen affine and canvas geometry read out of the shipped sidecar. That is
the main data-loading task on your side, and it is the same situation as
AudioMNIST — only the source layout and the speaker parse differ.

Corollary: `clip_labels.json` (§3.3) is keyed by *output* `.npy` stem, so it
covers **train clips only**. Its `content_classes` list is complete (all 35
words, derived from the whole corpus before the split), but there is no per-clip
entry for a held-out speaker. Re-walk the tarball for those.

---

## 1. Artifacts you receive

Per checkpoint:

| File | What it is |
|---|---|
| `<iter>.pt` | rosinality StyleGAN2 checkpoint (`g`, `d`, `g_ema`, `g_optim`, `d_optim`, `args`) |
| `<iter>.json` | **the sidecar** — the cross-repo contract (§3.1). Load `g_ema`. |

Once per dataset (from the prep run, also embedded in every sidecar):

| File | What it is |
|---|---|
| `speaker_split.json` | `train_speakers` / `held_out_speakers` / `split_seed` (§3.2) |
| `prep_manifest.json` | mel config, `mel_shape`, `canvas`, `offset`, affine, label summary |
| `clip_labels.json` | per-clip `{speaker, content}` for **train** clips + the class lists (§3.3) |

You also need the raw corpus on the attack box:
`http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz` (~2.3 GB).

---

## 2. Code to copy — byte-identical, not "equivalent"

Copy these from `stylegan2-audio` **verbatim**. Divergence causes silent mel
drift, which surfaces as an unexplained attack-fidelity ceiling, not as an error.

| Copy | Why |
|---|---|
| `audio/meldataset.py` | vendored NVIDIA/BigVGAN, pinned commit `7d2b454564a6c7d014227f635b7423881f14bdac`. **It IS the mel contract.** |
| `audio/env.py` | `AttrDict`, required by the above |
| `audio/contract.py` | `MEL_CONFIG`, `M_LO`, affine forward/inverse, `embed_canvas`/`crop_canvas`, `pad_or_truncate_waveform`, `extract_mel`, `expected_mel_frames` |
| `audio/__init__.py` | puts the package dir on `sys.path` — `meldataset.py` does a top-level `from env import AttrDict` |
| `datasets_audio.py` | **new, and the reason this doc exists**: the GSC walk, the speaker-hash parse, `SC_DIGIT_WORDS`. Stdlib-only, no torch. Copy it so both repos parse speaker identity identically. |

Do **not** copy `audio/bigvgan/` unless you vocode on the attack side (it is
vendored inference code, not part of the contract).

Explicit non-goals, carried over from the AudioMNIST handoff:

- Do **not** substitute torchaudio/librosa mel extraction. BigVGAN's
  `get_mel_spectrogram` only.
- Do **not** "clean up" `meldataset.py`. It caches the mel basis and Hann window
  in a module-level dict keyed by dtype/device/params. That is deliberate.
- Do **not** re-derive the affine, canvas, offset or `T`. Read them from the
  sidecar.
- Do **not** regenerate the speaker split. Read `speaker_split.json`.

**One trap we hit and rejected on this side:** `torchaudio.functional.resample`
rebuilds its sinc kernel every call and looks like an obvious thing to cache with
`torchaudio.transforms.Resample`. The two are **not** bit-identical (verified).
Use `torchaudio.functional.resample` exactly as `prepare_audio_data.load_waveform`
does. The cost is ~2 min over 97k clips — not worth the drift.

---

## 3. Schemas

### 3.1 Sidecar `<iter>.json`

Everything from `prep_manifest.json`, plus everything from `speaker_split.json`,
plus the latent block. Real values from an SC run:

```jsonc
{
  "mel_config": {
    "sample_rate": 22050, "n_fft": 1024, "win_length": 1024, "hop_length": 256,
    "n_mels": 80, "fmin": 0, "fmax": 8000, "clip_seconds": 1.0,
    "extractor": "bigvgan.meldataset.get_mel_spectrogram",
    "bigvgan_commit": "7d2b454564a6c7d014227f635b7423881f14bdac"
  },
  "resampler": "torchaudio.functional.resample",
  "vocoder": "nvidia/bigvgan_v2_22khz_80band_fmax8k_256x",
  "mel_shape": [80, 86],          // real region (n_mels, T) — T derived, IDENTICAL to AudioMNIST
  "canvas":    [128, 128],
  "offset":    [0, 0],            // top-left of the real region in the canvas
  "pad_value_normalized": -1.0,
  "channels": 1,
  "affine": { "m_lo": -11.512925464970229,
              "m_hi": <SC-SPECIFIC frozen value>,
              "m_hi_method": "percentile_99.9_train" },
  "num_clips": 98690,             // train clips actually written
  "num_source_clips": 98690,
  "content_labels": {             // summary only; per-clip map is clip_labels.json
    "type": "word", "num_classes": 35,
    "classes": ["backward", "bed", ...],        // 35, sorted
    "digit_classes": ["zero","one",...,"nine"], // digit order, not alphabetical
    "file": "clip_labels.json"
  },
  "audio_aug": { "enabled": false, ... },       // OFF for SC
  "dataset": "speech_commands",
  "speaker_split": "id_disjoint",
  "split_seed": 0,
  "train_speakers":    ["00176480", ...],       // 2418 hashes
  "held_out_speakers": ["037c445a", ...],       //  200 hashes — YOUR TARGET POOL
  "latent": { "num_ws": 12, "w_dim": 512 },     // 12 = 2*log2(128)-2
  "w_avg": [ ... 512 floats ... ]               // W+ warm-start
}
```

`m_hi` is the **only** affine value that differs from AudioMNIST. `mel_config`,
`mel_shape`, `canvas`, `offset` and `pad_value_normalized` are asserted identical
across the two datasets at prep time (`--reference-manifest`).

`w_avg` is `g_ema.mean_latent(4096)` drawn under a forked, fixed-seed (0) RNG, so
it is stable and comparable across checkpoints.

### 3.2 `speaker_split.json`

```json
{ "dataset": "speech_commands", "speaker_split": "id_disjoint", "split_seed": 0,
  "train_speakers": ["...", ...], "held_out_speakers": ["...", ...] }
```

2618 unique speaker hashes total → **2418 train / 200 held out**, by seeded
shuffle. Disjointness is asserted at prep time. The **400 attack targets are
drawn from `held_out_speakers` on YOUR side** — this repo deliberately does not
down-select them.

### 3.3 `clip_labels.json` (train clips only — see §0)

```json
{ "dataset": "speech_commands", "content_label_type": "word",
  "content_classes": ["backward", ..., "zero"],
  "digit_classes": ["zero", "one", ..., "nine"],
  "clips": { "<npy stem>": { "speaker": "0a7c2a8d", "content": "yes" }, ... } }
```

### 3.4 `.npy` stem naming

`<speaker>__<word>_<source wav stem>` — e.g. `0a7c2a8d__yes_0a7c2a8d_nohash_0`.

The word **must** be in the stem: the same speaker says different words with the
same `_nohash_<n>` index, so `<speaker>__<wav stem>` collides across word folders.
(AudioMNIST keeps its original `<speaker>__<wav stem>`.) The speaker is always the
first `__`-token in both datasets, and survives the `__aug{k}` suffix.

---

## 4. Building target mels (the work on your side)

Per held-out-speaker clip, **in this exact order** — same pipeline as prep, only
the source rate and walk differ:

1. **Walk** the tarball with `datasets_audio.collect_clips(root, "speech_commands")`.
   One folder per word; `_background_noise_/` excluded; speaker =
   `fname.split('_')[0]`. Ignore `validation_list.txt` / `testing_list.txt` —
   those are KWS splits, not this split.
2. **Load + resample** 16 kHz → 22050, mono, via `torchaudio.functional.resample`
   (see the trap in §2).
3. **Pad/truncate the WAVEFORM** to exactly 22050 samples with
   `pad_or_truncate_waveform`. Most SC clips are *shorter* than 1 s, so this is
   mostly zero-append. **Never pad in mel space.**
4. **Extract** with `extract_mel` → physical log-mel `(80, 86)`.
5. **Clamp + affine → `[-1,1]`** with `affine_forward(mel, m_lo, m_hi)` using
   **the sidecar's SC `m_hi`**, not AudioMNIST's, not a recomputed one.
6. **Embed** with `embed_canvas(norm, canvas, offset, pad_value_normalized)` →
   `(1,128,128)` float32.

Attack output goes back the other way: `crop_canvas(canvas, offset, mel_shape)` →
`affine_inverse(norm, m_lo, m_hi)` → BigVGAN. Read `offset`/`mel_shape`/`m_lo`/
`m_hi` from the sidecar every time; never hardcode.

`generate_audio.py` in this repo is the reference implementation of that
consumption path — read it if anything is ambiguous.

Also assert `mel.shape == tuple(sidecar["mel_shape"])` after step 4. If `T` is not
86, something drifted; stop rather than reshaping around it.

---

## 5. What is actually new vs. the AudioMNIST wiring

| | AudioMNIST | Speech Commands |
|---|---|---|
| Source layout | `<speaker>/<digit>_<speaker>_<rec>.wav` | `<word>/<hash>_nohash_<n>.wav` |
| Speaker id | parent dir name | filename prefix before first `_` |
| Source rate | 48 kHz | **16 kHz** |
| Content label | digit `0-9` (10) | **word (35)** |
| Held out | ~10 of 60 | **200 of 2618** |
| Attack targets | 100 | **400** |
| `m_hi` | AudioMNIST's | **its own, independent** |
| Everything else | — | identical |

Unchanged: model lineage, `num_ws=12`, `w_dim=512`, canvas/offset/`T`, mel config,
vocoder, W+ → ILO phases, the io-entry mechanism. If your AudioMNIST loader is
parameterized over the sidecar rather than hardcoded, SC should be a config change
plus the new walk.

### The OOD case (SC prior → AudioMNIST targets)

The two affines stay **independent** — do not widen either to cover the other.
Cross-dataset representational range is an *attack-side* concern (rung-2). When
running the SC prior against AudioMNIST targets you are crossing two different
`[m_lo, m_hi]` maps: normalize the target with the **target dataset's** affine to
build the mel, but note that the prior's output range is defined by the **SC**
affine. Handle that mismatch explicitly and measure it at rung 2 before blaming
the attack.

`digit_classes` in the sidecar exists for exactly this: the SC prior's samples are
scored for zero–nine coverage on the training side (`--digit-coverage`), so you
have a published number backing "this prior covers the digit sub-manifold" before
you point it at AudioMNIST.

---

## 6. Verification gates — wire these before the first attack run

1. **`state_dict` key match**: rosinality `Generator` → GIFD
   `stylegan2_io.Generator`. Raise on any missing/unexpected key.
2. **io-entry correctness**: `G(w, start_layer=0)` ≡
   `G(w, start_layer=k, layer_in=<true activation at k>)` bit-for-bit. If this
   fails, ILO phase 2+ is silently wrong.
3. **Targets ⊆ `held_out_speakers`** — assert on **every** run, from the
   sidecar's own list, not a local copy. This is the claim the whole result rests
   on; make a split mismatch fail loudly.
4. **Split provenance**: assert the sidecar's `split_seed` and speaker lists match
   the `speaker_split.json` you built targets from. Do not regenerate the split.
5. **Sidecar round-trip**: reconstruct crop + inverse affine from sidecar fields
   alone and confirm it matches a known-good prep output.
6. **Rung 1 before anything else**: wav → mel → affine → canvas → crop → inverse →
   BigVGAN on a real held-out SC clip. Must be intelligible. This isolates the
   mel/affine/vocoder chain from the attack.
7. **Rung 2**: embed a held-out real SC mel by direct latent optimization
   (W+ → ILO, image-space L2, **no** gradient matching). This is the ceiling on
   attack fidelity — a "failed attack" is usually a prior-capacity failure caught
   here.
8. **Dataset guard**: assert `sidecar["dataset"] == "speech_commands"` before
   loading SC targets. Two priors now exist and their checkpoints look identical
   on disk; mixing them silently produces garbage.

---

## 7. Status on the training side

Implemented and unit-tested (`unit_test/test_speech_commands.py`, which builds a
miniature corpus rather than needing the 2.3 GB download). Prep has been run on
the real corpus: 35 words, 105,829 clips, 2,618 speakers → 2,418 train / 200 held
out, 98,690 train clips, `T=86`. Training and the §8 validation ladder are still
to run — **the checkpoint is not yours yet**, but the split, the schemas and the
mel contract above are frozen and safe to build against now.

Open item to confirm at handoff time: the final SC `m_hi` value (frozen at prep,
read it from the sidecar you receive — do not copy it from this document).

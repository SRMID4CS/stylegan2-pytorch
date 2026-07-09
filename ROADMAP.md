# stylegan2-audio Roadmap

Persistent project context for humans and LLM agents working on this repo.
Logs what is **done**, what is **in progress**, and what is **to be done**.
Update at the end of each implementation session (see Update Protocol at the bottom).

All technical decisions are **locked in `GAN_TRAINING_SPEC.md`** — this file tracks
execution order and status only. On any conflict, the spec wins; surface conflicts,
do not work around them.

## Project End Goal

Convert this fork of rosinality `stylegan2-pytorch` into a **single-channel audio
GAN training + eval repo**: train an unconditional StyleGAN2 generator `z → mel`
(80×T log-mel embedded in a 128×128 canvas, normalized to `[-1,1]`), and ship a
**frozen checkpoint + JSON sidecar** consumed by the `dlg-sonic` gradient-inversion
repo. The mel config, global affine, canvas layout, and id-disjoint speaker split
are a **cross-repo contract** carried in the sidecar (spec §2, §4, §6, §9).

---

## Status Snapshot

### Completed

- **Fork bring-up on modern PyTorch** — fused leaky-relu contiguous-input fix,
  translation-augmentation fix (pre-existing commits on this branch).
- **Spec** — `GAN_TRAINING_SPEC.md` committed (locked decisions).
- **Phase 0: Docs + environment** *(2026-07-08)* — this `ROADMAP.md`, `USAGE.md`,
  `README_AUDIO.md`, `env.yml` (conda, cu128 wheels for Blackwell laptop + Ada AWS).
  No code changes yet.

### In Progress

- Phase 1 (image smoke test) — next up.

---

## Phase 1 — Image smoke test (repo as-is)

Goal: prove the **unmodified** training loop converges on this machine before
touching anything. ~10 images, overfit, eyeball samples.

1. Create `data/img_smoke/` with ~10 photos (any RGB images).
2. `python prepare_data.py --out data/lmdb_smoke --size 128 data/img_smoke`
   (LMDB is fine here — this phase only validates the loop, not the audio data path).
3. Train: `python train.py --size 128 --batch 8 --iter 2000 data/lmdb_smoke`
   (see USAGE.md §2 for the full command incl. Blackwell env vars).
4. Watch `sample/*.png` every 100 iters; expect recognizable memorization of the
   10 images well before 2k iters at this scale.
5. Confirm a checkpoint loads back into `generate.py`.

**Known gotchas to verify here:**

- `op/` CUDA extensions JIT-compile on first run — on the RTX 5070 Ti (Blackwell)
  set `TORCH_CUDA_ARCH_LIST="12.0"` if the build doesn't pick up `sm_120` (spec §1).
  On Windows this also needs MSVC on PATH + `ninja`.
- `sample/` and `checkpoint/` directories must exist (they do, in-repo).
- Checkpoints save every 10k iters only (plus iter 0) — fine for smoke, made
  configurable in Phase 2.

**Exit criterion:** samples visibly reproduce the training images; no NaNs; a saved
checkpoint round-trips through `generate.py`.

## Phase 2 — Audio GAN modifications (spec §3–§6)

Ordered; each item cites the spec section that locks it.

1. **Shared mel/contract module** (`audio/` package): vendor BigVGAN's
   `get_mel_spectrogram` and implement the global affine (§4) and canvas
   embed/crop (§3.5) as small pure functions. This exact module is later copied
   verbatim into dlg-sonic (spec §6 "best practice") — keep it dependency-light.
2. **Data-prep script** (`prepare_audio_data.py`): waveform → resample 22050 mono →
   pad/truncate waveform to 22050 samples → BigVGAN mel `(80,T)` → clamp + global
   affine → embed in 128×128 canvas at fixed offset → save float32 `.npy`
   `(1,128,128)` (§3, exact order; never PNG, never mel-space padding).
   Derive `T` empirically, don't hardcode (§2). Compute `m_hi` over the TRAIN set
   once and freeze it (§4). Emit `speaker_split.json` + a prep manifest holding
   `T`, `m_lo`, `m_hi`, offset for the sidecar.
3. **id-disjoint speaker split** (§3, §9): split by SPEAKER, seeded;
   AudioMNIST hold out ~10–12/60 speakers, Speech Commands a disjoint hash set.
   Training consumes ONLY train-speaker clips.
4. **Float32 `.npy` Dataset** (`dataset.py` addition, e.g. `NpyMelDataset`):
   bypass LMDB/PIL entirely; returns `(1,128,128)` float32 tensors already in
   `[-1,1]` — so the train transform must apply **no flip, no ToTensor-rescale,
   no Normalize**.
5. **Single-channel model patch** (§5): `ToRGB` output channels 3→1
   ([model.py:376](model.py#L376)) and Discriminator input `ConvLayer(3,…)`→1
   ([model.py:655](model.py#L655)); expose as a `channels`/`img_channels` arg
   rather than hardcoding, default 3 to keep the image path working.
6. **train.py audio path**: `--dataset npy` (or similar) switch; remove
   `RandomHorizontalFlip` + 3-ch `Normalize` for the audio path
   ([train.py:512-518](train.py#L512-L518)) — x-flip is time reversal, banned (§5, §8);
   `--ckpt_every` / `--sample_every` args; seed all RNGs from one `--seed` (§9).
7. **Augmentation policy** (§5): default OFF; add a pruned aug mode
   (translation + cutout only) selectable by flag for later overfitting rescue;
   never enable the geometric/color pipeline on mels.
8. **Checkpoint sidecar writer** (§6): on every checkpoint save, emit
   `<ckpt>.json` with mel_config, vocoder id, mel_shape, canvas, offset,
   pad_value, affine, `num_ws` (= `g.n_latent` = 12 at 128), `w_dim` (512),
   `w_avg` (`g_ema.mean_latent(n)`, stored as a tensor file or embedded list),
   dataset, split seed, train/held-out speaker lists.
9. **Sampling/eval helpers**: `generate_audio.py` — `z → G → crop → inverse
   affine → BigVGAN → .wav`, plus GT-mel-vocode upper-bound mode (§7.2) and
   mel-canvas PNG dumps for quick visual checks (PNG for *viewing only*).

**Decisions (resolved 2026-07-09):**

- **BigVGAN packaging: vendor `meldataset.py` VERBATIM** (it *is* the contract;
  spec §6 "vendor into both repos"); clone/pull the full BigVGAN package only
  where vocoding runs. Caveat: `get_mel_spectrogram` caches mel basis/window in
  a **module-level dict** keyed by dtype/device/params — do NOT "clean it up"
  or refactor it; any divergence from the copy in dlg-sonic risks silent mel
  drift. Byte-identical is the point.
- **FID on mels: relative tripwire only, never a selection criterion or reported
  number.** Channel-repeat mel→3ch through RGB-Inception is meaningless in
  absolute terms for spectrograms; keep it solely as a cheap divergence detector
  during the R1 sweep. **Checkpoint selection for now: D/G loss + listening.**
  Later upgrade (deferred): rung-2 held-out-embedding error (direct latent
  optimization of held-out real mels) as the principled selector — it measures
  representational capacity, which is what the attack needs.

## Phase 3 — Audio overfit test (~10 clips)

The spec's overfit smoke test (§7.5) plus validation ladder rung 1, on the laptop GPU.

1. Prep ~10 one-second clips through `prepare_audio_data.py`.
2. **Round-trip a real clip first, before any training** (§7.1): wav → mel →
   affine → canvas → crop → inverse affine → BigVGAN → wav; must be intelligible.
   This isolates the mel/affine/vocoder chain from the GAN.
3. Record the **GT-mel-vocode upper bound** for these clips (§7.2).
4. Train to overfit (small iter count, no augmentation).
5. Sample: `z → G → crop → inverse affine → BigVGAN`; expect recognizable
   (memorized) speech (§7.3).
6. **Sidecar round-trip** (§7.4): reconstruct crop + inverse affine from sidecar
   fields alone; confirm equal to the data-prep code's output.

**Exit criterion:** an end-to-end chain data-prep → train → checkpoint+sidecar →
generate → vocode → intelligible audio, driven only by contract values read from
the sidecar. (Attack-side load into `stylegan2_io` happens in dlg-sonic — see below.)

## Phase 4 — Unit tests (`unit_test/`)

Tests owned by THIS repo (pytest; CPU-capable where possible so they run anywhere):

- **Mel config**: canonical values (§2) asserted in one place; empirical `T`
  derivation returns a consistent value for a 22050-sample input.
- **Affine**: forward/inverse round-trip identity within float32 eps; clamp
  behavior at `m_lo`/`m_hi`; global (same map for two different samples).
- **Canvas**: embed → crop round-trip identity; fill value is exactly -1;
  offset honored; crop uses sidecar fields, not constants.
- **Waveform padding**: short clip → padded with zeros to 22050; long clip →
  center-cropped; padding lands on mel log-floor after extraction (≈ `m_lo`).
- **Dataset**: `.npy` loader returns `(1,128,128)` float32 in `[-1,1]`; no flip
  or normalize applied.
- **Model**: 1-channel Generator output shape `(B,1,128,128)`; Discriminator
  accepts `(B,1,128,128)`; `n_latent == 12` at size 128; 3-channel default path
  still works (regression).
- **Sidecar**: schema-complete (every §6 key present); values match the prep
  manifest; `w_avg` shape `(1,512)`; train/held-out speaker lists disjoint.
- **Split**: speaker-level (no speaker appears on both sides); deterministic
  under fixed seed.
- **Reproducibility**: two seeded prep runs produce byte-identical `.npy`s.

**Checks that can ONLY be done in dlg-sonic (note for the other repo — do not
implement here):**

- rosinality `Generator` → GIFD `stylegan2_io.Generator` **state_dict key match**
  on first load (§1).
- **io-entry correctness**: `G(w, start_layer=0)` ≡ `G(w, start_layer=k,
  layer_in=captured_activation_k)` bit-for-bit (§7).
- W+ → ILO attack behavior, warm-start from sidecar `w_avg`.
- Attack-side crop + inverse affine consumed purely from the sidecar (mirror of
  our sidecar round-trip test, on their side).
- Validation ladder rung 2 (held-out-sample embedding via direct latent
  optimization) and rung 3 (full gradient inversion).
- Verify attack targets ∈ `held_out_speakers` before every run.

## Phase 5 — Full training + handoff

1. AudioMNIST full prep (~25k clips, train speakers only).
2. **R1 sweep** (spec §5): 2–3 values of `--r1` on short runs, pick by (mel-)FID,
   then train long. Mixed precision on. Aug OFF unless D overfits; then pruned
   translation/cutout only.
3. Full run on **AWS g6e.xlarge (L40S, Ada)** — laptop is for smoke only (§1).
   Expect reasonable ~0.5 day, fuller convergence 2–4 days at 128².
4. Repeat for Speech Commands (one generator per dataset, no cross-dataset prior).
5. Handoff per checkpoint: `.pt` + sidecar `.json` + `speaker_split.json`;
   run spec §7 validation items 1–5 before handing over.
6. Then: inversion testing lives in dlg-sonic.

---

## Deferred / out of scope for this repo

- Anything attack-side (W+/ILO optimization, gradient matching, metrics) — dlg-sonic.
- SWAGAN arch for audio (exists in repo; untested for mels — revisit only if
  training speed becomes the bottleneck).
- Conditional generation (spec locks unconditional).
- Long audio > 1 s.

---

## Update Protocol (after each session)

1. Move finished items to **Completed** (with date).
2. Record **open issues / blockers** and any resolved "open decisions".
3. List **next-session first tasks** (3–5, ordered).
4. Update `USAGE.md` if commands/flags changed; `README_AUDIO.md` only if
   goals/defaults changed. `GAN_TRAINING_SPEC.md` changes only by explicit decision.

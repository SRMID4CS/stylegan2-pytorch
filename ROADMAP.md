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
- **Phase 1: Image smoke test** *(2026-07-09)* — unmodified loop verified locally:
  LMDB dataprep, training convergence on ~10 images, JIT CUDA ops on the RTX 5070 Ti,
  checkpoint round-trip through `generate.py`. All working.
- **Phase 2: Audio GAN modifications** *(2026-07-09)* — implemented (all items below);
  contract module + 1-ch model wiring verified on CPU (`T=86`, affine/canvas round-trip
  ~1e-6, `n_latent==12`, 3-ch regression intact). Training/vocoding validation happens
  in Phase 3 (user-run, WSL).
- **Augmentation (`AUGMENTATION_SPEC.md`)** *(2026-07-15)* — both mechanisms, OFF by
  default (usage: `USAGE.md` §3):
  - **Offline waveform aug** (`prepare_audio_data.py --audio_aug`): per TRAIN clip,
    `--aug_variants` composed copies (`speed → pitch` pre-pad, re-fit to 1.0 s,
    `time_shift` [+ optional `gain`] post-pad) saved as `<stem>__aug{k}.npy`;
    held-out speakers never augmented; `m_hi` recomputed over the full augmented
    set; all params + `--seed` recorded under `audio_aug` in the manifest → sidecar.
    Per-(seed, clip, variant) RNG ⇒ byte-identical outputs for the same seed.
  - **ADA audio mode** (`train.py --augment --augment_mode audio`):
    `non_leaking.augment_audio` = time-axis-only integer translation (replicate
    fill, never reflect/zeros) + single-rectangle cutout; adaptive-p controller
    untouched; default `image` mode and the no-aug npy path unchanged; image-mode
    ADA still hard-blocked for `--dataset npy`.
  - **Phase 4 started early**: `unit_test/` created with `test_audio_aug.py` +
    `test_ada_audio.py` (16 tests, all pass in the WSL env; includes an
    end-to-end double-prep byte-identity run on `data/audio_mnist_test`).
  - **Fixed pre-existing break**: `non_leaking.GridSampleBackward` used the
    torch≤1.12 `_jit_get_operation` API — image-path `--augment` crashed on
    torch 2.13. Now calls `torch.ops.aten.grid_sampler_2d_backward`; verified by
    short GPU runs of both image ADA (lmdb) and audio ADA (npy) paths.

### In Progress

- Phase 3 (audio overfit smoke on `data/audio_mnist_test/`, run in WSL) — commands
  ready in `USAGE.md` §3.

**Next-session first tasks:**

1. Run the Phase 3 smoke in WSL (USAGE §3): prep (`held_out` must log `['02']`),
   rung-1 round-trip BEFORE training, overfit run, sample + listen (`_BVG`/`_GL`).
2. Fix anything the smoke surfaces; record results here.
3. Continue Phase 4 `unit_test/` suite (augmentation tests exist; affine/canvas/
   dataset/model/sidecar/split still to write).
4. Optionally: aug round-trip listen check (AUGMENTATION_SPEC §D.1) — vocode an
   `__aug` variant, confirm intelligible.

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

## Phase 2 — Audio GAN modifications (spec §3–§6) — DONE 2026-07-09

All items implemented; each cites the spec section that locks it.

1. ✅ **Shared mel/contract module** (`audio/` package): `audio/meldataset.py` +
   `audio/env.py` vendored VERBATIM from NVIDIA/BigVGAN (pinned commit in
   `audio/contract.py::BIGVGAN_COMMIT`); `audio/contract.py` holds `MEL_CONFIG`,
   the global affine (§4), canvas embed/crop (§3.5), and waveform pad/truncate as
   pure functions taking all geometry as arguments. This package (minus
   `audio/bigvgan/`) is what gets copied verbatim into dlg-sonic (spec §6).
   `audio/bigvgan/` additionally vendors BigVGAN's inference code (vocoder only,
   weights pulled from HF at first use) — not part of the contract copy.
2. ✅ **Data-prep script** (`prepare_audio_data.py`): waveform → resample 22050 mono →
   pad/truncate waveform → BigVGAN mel `(80,T)` → clamp + global affine → embed in
   canvas → float32 `.npy` `(1,128,128)` (§3, exact order; never PNG, never
   mel-space padding). `T` derived empirically (=86 verified); `m_hi` = 99.9th
   percentile over the TRAIN set, frozen (§4). Emits `speaker_split.json` +
   `prep_manifest.json`; all `[AUDIO]`-tagged logging incl. the split result.
3. ✅ **id-disjoint speaker split** (§3, §9): speaker-level, seeded
   (`--split-seed`, `--holdout N` / `--no-split`); only train-speaker clips are
   processed; held-out speakers logged for seed verification.
4. ✅ **Float32 `.npy` Dataset** (`dataset.py::NpyMelDataset`): no LMDB/PIL,
   returns float32 tensors already in `[-1,1]`, **no flip / ToTensor-rescale /
   Normalize**; shape validated against the prep manifest.
5. ✅ **Single-channel model patch** (§5): `ToRGB(..., out_channel=)` and
   `Generator/Discriminator(..., img_channels=3)` in `model.py` — config-level,
   default 3 keeps the image path bit-identical (regression verified).
6. ✅ **train.py audio path**: `--dataset npy` switch (validates `--size` /
   `--img_channels` against `prep_manifest.json`); npy path applies **no
   transform** (x-flip = time reversal, banned §5/§8); `--ckpt_every` /
   `--sample_every`; one `--seed` for python/numpy/torch/cuda (§9).
7. ✅ **Augmentation policy** (§5): implementation KEPT (this repo is StyleGAN2 —
   the optional ADA-style `--augment` stays available for the image path) but
   **hard-blocked for `--dataset npy`**: train.py exits with an `[AUDIO]` error if
   combined. *Superseded 2026-07-15 by `AUGMENTATION_SPEC.md`*: the block now
   applies only to image-mode ADA; `--augment_mode audio` (mel-valid subset) and
   offline `--audio_aug` are available, both OFF by default.
8. ✅ **Checkpoint sidecar writer** (§6): every checkpoint save on the npy path
   also writes `checkpoint/<iter>.json` = prep manifest + speaker split +
   `latent {num_ws: g.n_latent (=12 at 128), w_dim}` + `w_avg`
   (`g_ema.mean_latent(4096)` under a forked fixed-seed RNG, embedded as a list).
9. ✅ **Sampling/eval helpers**: `generate_audio.py` — `z → G → crop → inverse
   affine → vocoder → .wav`, everything read from the sidecar; `--vocoder
   {bigvgan,gl,both}` (default both; file suffixes `_BVG`/`_GL`; Griffin-Lim =
   librosa inverse of the same mel basis, no vocoder download needed); round-trip
   and GT-mel-vocode upper-bound modes (§7.1–7.2); mel-canvas PNG dumps (viewing only).

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
  ***Resolved 2026-07-21:*** replaced by Route B below — this is now the actual
  tripwire in use, not just a deferred idea.

- **Route B convergence curve — mel-native domain-classifier Frechet distance +
  class-coverage entropy** *(2026-07-21)*, `eval/convergence_curve.py`: trains a
  small mel classifier on the real TRAIN-speaker mels (id-disjoint by
  construction — `prepare_audio_data.py` never writes held-out-speaker clips to
  its `--out` dir), caches `(mu_r, Sigma_r)` + real class-entropy per
  `--dataset` under `convergence_cache/<dataset>/`, then per checkpoint samples
  `g_ema` with a frozen seed (z **and** per-layer noise — `NoiseInjection` in
  `model.py` draws noise from the ambient RNG when `randomize_noise=True`, so
  freezing only z would leave noise uncontrolled run-to-run) and reports the
  Frechet distance to the cached reference plus normalized coverage entropy.
  Never vocodes — stays in mel space throughout, runs in seconds/checkpoint.
  Generator is rebuilt straight from each checkpoint's own saved `args` (same
  trick as `generate_audio.py`) rather than hardcoded size/style_dim/n_mlp, so
  it can't drift from what a checkpoint was actually trained with. This
  **replaces the channel-repeat-Inception-FID idea** as the chosen relative
  curve for the R1 sweep and checkpoint selection. Usage: `USAGE.md` §3.5.

- **Speaker-identity convergence curve — collapse tripwire for the N1
  speaker-diversity claim** *(2026-07-21)*, `eval/convergence_curve.py`
  `--label-mode {content,speaker,both}` (default `both`): the same small CNN is
  trained to predict **speaker ID over the train speakers** and the Frechet
  distance + coverage entropy are computed in that embedding. The digit
  (`content`) classifier saturates (~99.8% acc) and is blind to speaker-manifold
  collapse; the speaker curve is the metric that actually detects it. Speaker
  labels come from `speaker_split.json` `train_speakers` (the id-disjoint
  contract), cross-checked against the `.npy` filenames and remapped to a
  contiguous `0..K-1` head with `K = len(train_speakers)` **derived, never
  hardcoded** (asserted at run start). Caches are namespaced per `(dataset,
  mode)` — `convergence_cache/<dataset>/<content|speaker>/` — so the two never
  clobber. `both` samples each checkpoint **once** (frozen seed) and embeds
  twice, keeping both curves on identical mels at half the generation cost.
  Reports the speaker classifier's train accuracy prominently (speaker ID from
  1 s clips over ~48 classes is genuinely hard; `--clf-epochs` raises training
  length if it underfits toward chance). The PNG now stacks one twin-axis panel
  per mode and omits the untrained iter-0 point by default
  (`--plot-omit-iter0`, PNG only — CSVs keep every checkpoint). Usage:
  `USAGE.md` §3.5.

## Phase 3 — Audio overfit test (~10 clips)

The spec's overfit smoke test (§7.5) plus validation ladder rung 1, run in WSL on the
laptop GPU. Test data: `data/audio_mnist_test/data/` (speaker `01` ×4 clips trains,
speaker `02` held out — the prep log must show `held_out_speakers=['02']`).
Exact commands: `USAGE.md` §3.

1. Prep the clips through `prepare_audio_data.py` (seeded split, held-out logged).
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

Tests owned by THIS repo (pytest; CPU-capable where possible so they run anywhere).
*Started 2026-07-15: `test_audio_aug.py` + `test_ada_audio.py` cover the two
augmentation mechanisms (seeded determinism, train-only application, `.npy`
invariants, re-fit to 1.0 s, ADA audio-mode identity/translation/cutout/gradients).
The items below remain to be written:*

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
2. **R1 sweep** (spec §5): 2–3 values of `--r1` on short runs, then train long.
   Mixed precision on. Start with no augmentation; if D overfits follow the
   rollout order in `AUGMENTATION_SPEC.md` §E (`--audio_aug` first, then
   `--augment --augment_mode audio`).
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

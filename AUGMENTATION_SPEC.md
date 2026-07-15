# Audio Augmentation — Feature Spec (stylegan2-audio)

> Companion to `GAN_TRAINING_SPEC.md`. Read that first: the mel config (§2), global
> affine (§4), canvas layout (§3.5), id-disjoint speaker split (§3/§9), and sidecar
> contract (§6) are all preserved here. The **only** contract change is that `m_hi`
> is recomputed over the augmented train set (see §A.4). Repo details (`prepare_audio_data.py`,
> `train.py`, `non_leaking.py`, `speaker_split.json`, the sidecar writer) are already in place.

## Scope

Two independent mechanisms — different purpose, layer, and leakage behavior. Do **not**
conflate them, and do **not** port anything from NVlabs (rosinality already has the ADA
non-leaking augment + adaptive-p controller).

| Mechanism | Flag | Layer | Applied to | Purpose | Leaks into G? |
|---|---|---|---|---|---|
| Offline waveform data aug | `--audio_aug` (prep) | waveform, offline in `prepare_audio_data.py` | **train speakers only** | expand realistic manifold | yes, by design (realism-preserving only) |
| ADA discriminator aug | `--augment` + `--augment_mode audio` (train) | mel/canvas, in training loop | real **and** fake | prevent D overfitting | no (non-leaking) |

Both default **OFF**. Hard rule: **no waveform augmentation in the training DataLoader** —
all waveform aug is offline in prep; the loader still only reads `.npy`, so training
throughput is identical to the no-aug path.

---

## A. Offline waveform augmentation (`--audio_aug`)

### A.1 CLI (prepare_audio_data.py)

- `--audio_aug` — bool, default `False`. Enable augmentation.
- `--aug_types` — csv, default `time_shift,speed,pitch` (the recommended set). Selects which augs run.
- `--aug_variants` — int, default `2`. Augmented copies per source clip (→ ~3× dataset incl. original).
- `--time_shift_ms` — float, default `100`. Max ± shift (see A.3).
- `--speed_range` — float, default `0.10`. Max ± fractional rate (0.10 → 0.90–1.10).
- `--pitch_semitones` — float, default `2.0`. Max ± semitone shift.
- `--seed` (existing) drives **all** aug RNG.

Passing `--aug_types` with a subset (e.g. `--aug_types time_shift,pitch`) runs only those.

### A.2 Per-variant pipeline order (critical)

For every source clip, emit the **original** plus `aug_variants` augmented copies.
Each augmented copy:

1. Load + resample 22050 mono → `x` (variable length).  *(existing)*
2. Apply length/pitch augs on `x`, in this order: **speed** → **pitch**. Each parameter
   drawn per-variant from its configured range with the seeded RNG.
3. Pad/truncate to exactly **22050 samples**: `rate>1` shortens → zero-pad to 1 s;
   `rate<1` lengthens → center-crop to 1 s.  *(reuse existing pad/truncate)*
4. Apply frame-relative augs on the 1 s clip: **time_shift** (zero-filled linear shift).
5. BigVGAN mel → clamp → global affine → canvas embed → float32 `.npy`.  *(existing §3.3–3.6)*

The original clip runs the existing path unchanged (steps 1, 3, 5). Original keeps its
current filename; variants get a suffix: `<orig_stem>__aug{k}.npy` (k = 1..N).

Each variant composes all enabled augs once (independent seeded param draws). Ranges are
mild, so composition stays realistic.

### A.3 Aug definitions and defaults

| Aug | Transform | Default range | Realistic cap | Notes |
|---|---|---|---|---|
| `time_shift` | linear shift within the 1 s frame, **zero-fill** vacated region (not circular) | ±100 ms (±2205 samp) | ±150 ms | applied **after** pad-to-1 s; content past the edge is dropped (safe given silence margins) |
| `speed` | pitch-preserving time-stretch (`librosa.effects.time_stretch`) | rate 0.90–1.10 | ±15 % | kept orthogonal to `pitch`; `rate>1`→pad, `rate<1`→center-crop |
| `pitch` | pitch shift (`librosa.effects.pitch_shift`) | ±2 semitones | ±2 st | length-preserving; larger = unnatural pseudo-speakers, don't exceed cap |
| `gain` *(optional, OFF)* | scalar amplitude | ±4 dB | ±6 dB | realistic but shifts the log-mel distribution; only meaningful because `m_hi` is recomputed (A.4). Not in default `aug_types`. |

### A.4 Hard requirements (contract)

- **Recompute `m_hi`** over the **full** train set (original + every augmented variant)
  before freezing the affine. Write the new `m_hi` into the prep manifest and the
  checkpoint sidecar. Never freeze `m_hi` on clean data and then add augs.
- **Train speakers only.** Augment only clips whose speaker ∈ `train_speakers` from
  `speaker_split.json`. Never augment held-out target speakers (would blur the id-disjoint claim).
- **Determinism.** Same `--seed` → byte-identical augmented `.npy`. Extend the existing
  reproducibility unit test to cover augmented outputs.
- **Record in manifest + sidecar.** `audio_aug` on/off, `aug_types`, per-aug params,
  `aug_variants`, seed — so the training distribution is reproducible and visible to dlg-sonic.
- **Output invariants unchanged.** Every `.npy` stays `(1,128,128)` float32 in `[-1,1]`;
  speed/pitch outputs are re-fit to exactly 22050 samples **before** mel extraction.

---

## B. ADA discriminator augmentation (`--augment` audio mode)

The adaptive non-leaking augment already exists (`non_leaking.py` + the adaptive-p loop
in `train.py`). This feature prunes its transform set to the mel-valid subset. Keep the
adaptive-p controller untouched.

### B.1 CLI (train.py)

- `--augment` (existing) — enable ADA.
- `--augment_mode {image,audio}` — **new**, default `image`. `audio` restricts the transform
  composition to the mel-valid subset. `image` preserves current behavior (regression-safe).
- `--aug_time_translation` — float, default `0.125`. Max time-axis translation as a fraction of canvas width.
- `--aug_cutout` — float, default `0.4`. Cutout size as a fraction of canvas.
- Adaptive-p args (`ada_target`, `ada_length`, `ada_every`) unchanged: target ~0.6, start p = 0.

### B.2 Pruned transform set (audio mode)

**Keep only:**

- **Time translation** — along the **width (time)** axis only. Frequency-axis (vertical)
  translation MUST be disabled. Padding mode `border` (replicate) — **not** `reflection`
  (a reflected time edge is a short time-reversed segment, which is banned) and **not**
  `zeros` (0 ≠ our normalized −1 floor).
- **Cutout** — single random rectangle, non-leaking (already applied to real + fake).

**Disable everything else:** rotation, x/y flip, isotropic/anisotropic scaling, aspect,
fractional/vertical translation, and all color/brightness/contrast/luma/hue/saturation ops
(also undefined on 1 channel).

### B.3 Requirements

- Change only **which** transforms are active; leave the adaptive-p controller intact.
- Default `--augment_mode image` leaves the image path byte-for-byte unchanged (regression test).
- Shapes preserved `(B,1,128,128)`; no NaNs; p still updates across training.

---

## C. Explicitly excluded — do NOT implement

Leaky-as-corruption or invalid on a time-frequency mel — do not add even if convenient:

- **Noise addition, reverb / room IR** — corrupt a clean prior (generator would learn noisy mels).
- **SpecAugment time/frequency masking** — corrupting; leaks as data aug.
- **x-flip / time reversal, frequency-axis flip, rotation, shear, anisotropic scale,
  color/hue/brightness/contrast** — invalid on mel axes.

---

## D. Acceptance / validation

1. **Aug round-trip.** An augmented clip still passes wav → mel → affine → canvas → crop →
   inverse-affine → BigVGAN and stays intelligible (spec §7.1).
2. **Unit tests.** Seeded determinism (byte-identical); train-only application; `m_hi`
   computed over the augmented set; `.npy` invariants `(1,128,128)` float32 `[-1,1]`;
   speed/pitch outputs re-fit to 22050 samples pre-mel.
3. **ADA audio mode.** Only time-translation + cutout active; default `image` mode
   unchanged (regression); no NaNs; adaptive p still moves.
4. **Throughput.** No-aug and image paths unchanged; ADA-audio per-iter overhead in the
   single-digit-to-~20 % range, not a multiple. (All waveform aug is offline, so training
   kimg throughput is unaffected; offline expansion only grows unique-image count.)

---

## E. Rollout order (recommended)

1. AudioMNIST: train **clean** first; watch for D-overfit (D loss → 0, D(real) ≫ D(fake), FID
   climbing after a min, samples memorizing).
2. If overfitting: enable **`--audio_aug`** with default `time_shift,speed,pitch`, `aug_variants 2`,
   recompute `m_hi`, retrain. (Cheapest win; improves the prior.)
3. Still overfitting: add **`--augment --augment_mode audio`** (time-translation + cutout, adaptive p).
4. Speech Commands (~105k): start clean; use `--audio_aug` only for diversity, ADA likely unnecessary.

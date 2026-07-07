# StyleGAN2 Mel-Prior — Training Repo Spec (z→mel generator for dlg-sonic)

> Context handoff for the **GAN training repo** that produces the frozen mel generator
> consumed by the `dlg-sonic` attack repo. This repo trains a StyleGAN2 generator on
> mel-spectrograms and ships a **frozen checkpoint + sidecar**. It is the other half of a
> cross-repo contract: the mel function, the normalization affine, and the canvas layout
> here MUST be byte-for-byte identical to what dlg-sonic uses.
>
> Read the Glossary first. Everything here is a **locked decision**; surface conflicts
> rather than working around them.

---

## 0. Glossary (read first)

- **The goal in one line:** train an **unconditional** StyleGAN2 generator `G` so that a
  sampled/optimized latent produces a single-channel **128×128** normalized "mel canvas",
  and cropping + un-normalizing that canvas yields a valid **80×T** log-mel that a pretrained
  **BigVGAN v2** vocoder can turn back into speech.
- **Mel-spectrogram:** a 2-D audio feature `(n_mels, time_frames)`. Treated as a
  single-channel image.
- **Vocoder (BigVGAN v2):** pretrained net that maps a mel back to a waveform. We do NOT
  train it. We only need our generated mels to be in *its* expected format so it can
  vocode them. It also defines the mel config (§2).
- **Latent spaces (z / W / W+):** `z` = input noise; `W` = mapping-net output; `W+` = a
  separate `w` per synthesis layer. The **attack side optimizes in W+**, so this repo must
  export `w_avg` (mean latent) and `num_ws` (number of W+ rows) in the sidecar.
- **Global affine:** a single fixed linear map (same for every sample) between physical
  log-mel values and `[-1,1]`. Must be global (not per-sample) so it is invertible at
  attack time. Persisted in the sidecar.
- **Canvas / real region / offset:** StyleGAN needs a square power-of-two, so the real
  `80×T` mel is embedded into a `128×128` canvas at a fixed `offset`; the rest is padding.
  The attack side crops the identical region — so `offset`, canvas size, and real shape are
  a **shared contract** carried in the sidecar.
- **id-disjoint / held-out-speaker split:** `G` is trained ONLY on the training speakers;
  the attack-target speakers are held out. `G` must never see a target speaker.
- **`stylegan2_io` (io generator):** GIFD's generator — a rosinality `Generator` with
  GIFD's added intermediate-layer entry/exit (`start_layer` / `end_layer` / `layer_in`).
  Training uses rosinality's standard forward; the attack loads the **same weights** into
  this io forward to run intermediate-layer optimization. You train with the plain rosinality
  generator and only need `stylegan2_io` on the attack side.
- **Augmentation:** rosinality has an optional ADA-style non-leaking augmentation
  (`--augment`). Its default pipeline is natural-image (geometric/color) augs that are
  invalid for mels, so we default to **no augmentation** and only reach for it (pruned to
  cutout/translation) if the small dataset overfits.

---

## 1. Codebase & environment

- **Train in a clone of rosinality `stylegan2-pytorch`** — it has the actual training loop
  (`train.py`, distributed, R1, EMA) that GIFD's attack repo does not. Train an
  **unconditional** generator here. The attack side then loads these **same weights** into
  **GIFD's `stylegan2_io`** (a light rosinality-based modification that adds
  `start_layer`/`end_layer`/`layer_in`). This works because both share the rosinality
  `Generator` module layout — but **verify the `state_dict` keys match** between rosinality's
  `Generator` and GIFD's `stylegan2_io.Generator` on first load (adapt any renamed keys).
- Do NOT use NVlabs stylegan2-ada-pytorch or the old TF StyleGAN2 — weights are **not**
  portable across lineages (different modulation/resampling internals). Pick the rosinality
  lineage end-to-end.
- **Train on an Ada GPU (AWS g6e / L40S).** rosinality's custom ops (`fused_bias_act`,
  `upfirdn2d`) compile smoothly there. Avoid Blackwell (RTX 50xx) unless you patch
  `TORCH_CUDA_ARCH_LIST` to include `sm_120`.
- **Weights are portable *within this lineage*.** Train with the standard forward; the
  attack repo loads the **same `state_dict`** into the io forward. Training-env and
  attack-env torch versions need not match; the generator lineage must.

---

## 2. Canonical mel config — SINGLE SOURCE OF TRUTH (must equal dlg-sonic)

| Param | Value |
|---|---|
| sample_rate | 22050 |
| n_fft | 1024 |
| win_length | 1024 |
| hop_length | 256 |
| n_mels | 80 |
| fmin | 0 |
| fmax | 8000 |
| clip length | 1.0 s (22050 samples) |

- **Extractor:** BigVGAN's own `get_mel_spectrogram()` (from its `meldataset`). NOT
  torchaudio, NOT librosa — their log/normalization differ silently.
- **Vocoder for validation:** `nvidia/bigvgan_v2_22khz_80band_fmax8k_256x`,
  `use_cuda_kernel=False`, `remove_weight_norm()`, eval, frozen.
- **Derive `T` empirically:** push one 1.0 s clip through `get_mel_spectrogram()` and read
  the exact `(80, T)`. Do NOT hardcode (it is ~86; verify).

---

## 3. Data-prep pipeline (waveform → training sample) — the critical part

For every source clip, in this exact order:

1. **Load + resample** to 22050 Hz, mono.
2. **Pad/truncate the WAVEFORM** to exactly 22050 samples (1.0 s). Pad by appending silence
   (zeros). Center-crop the rare over-length clip. **Never pad in mel space** — silence
   must pass through the extractor so it lands on the mel log-floor.
3. **Extract** with BigVGAN `get_mel_spectrogram()` → physical log-mel `(80, T)`.
4. **Global affine → `[-1,1]`** (see §4). Clamp to `[m_lo, m_hi]` first, then map.
5. **Embed into a 128×128 canvas** at fixed `offset` (e.g. `(0,0)`, top-left). Fill the
   rest with `-1` (the normalized floor). Record `offset` for the sidecar.
6. **Store as float32 `.npy`** (single channel `(1,128,128)`). **NEVER PNG** — 8-bit PNG
   quantizes the mel dynamic range and ruins reconstruction.

**id-disjoint split (speaker-level, NOT sample-level — this is the core control):** split by
SPEAKER first, then draw the fixed attack-target sets from the held-out speakers:

- **AudioMNIST:** hold out ~10–12 of the 60 speakers; draw the **100 attack targets** from
  those held-out speakers. GAN training data = *all clips of the remaining ~48–50 speakers*
  (~25k). The held-out speakers' non-target clips are unused for training (they can serve as
  an unseen-speaker validation pool — see §7 rung 2).
- **Speech Commands:** hold out a disjoint speaker-hash set; draw the **400 attack targets**
  from it. GAN training data = all clips of the training speakers (~105k pool).

> Do NOT do a naive sample-level split ("N random clips for attack, rest for training"). That
> lets the GAN see the target *speakers* and silently breaks the "the GAN never saw this
> identity" claim — the entire point of id-disjoint.

**Do not use rosinality's `prepare_data.py`/LMDB path** (8-bit images). Feed the float32
`(1,128,128)` `.npy` arrays via a small custom `Dataset`.

---

## 4. The global affine (must match dlg-sonic exactly)

```
m_lo = log(1e-5) ≈ -11.5129     # fixed by the extractor's clamp
m_hi = <global data max>        # e.g. 99.9th percentile of physical log-mel over TRAIN set, then clip

# forward (physical -> normalized), same map for EVERY sample:
mel_c      = clamp(mel, m_lo, m_hi)
normalized = 2*(mel_c - m_lo)/(m_hi - m_lo) - 1        # in [-1,1]

# inverse (attack side, before vocoding):
mel        = (normalized + 1)/2 * (m_hi - m_lo) + m_lo
```

- **Global, not per-sample.** A per-image min/max map is not invertible at attack time
  (you don't know the target's per-sample stats). Compute `m_hi` once over the training set
  and freeze it.
- Persist `m_lo`, `m_hi` in the sidecar (§6).

---

## 5. Training

- **Resolution:** `--size 128` (square).
- **Unconditional.** No class conditioning (consistent with vanilla StyleGAN2 and
  rosinality's default). The known label is used on the *attack* side, not baked into the
  prior — keeps the "prior is generic" story clean and the io wiring simple.
- **Single channel:** rosinality's `Generator`/`Discriminator` assume 3 channels. Patch the
  final `ToRGB` to output 1 channel and the discriminator's first `ConvLayer` to accept 1.
  (This is the standard "grayscale rosinality" change.)
- **Custom float32 dataset:** do NOT use `prepare_data.py` → LMDB (8-bit); feed the float32
  `(1,128,128)` `.npy` arrays from §3 via a small custom `Dataset`.
- **ADA-style augmentation is optional — OFF by default.** If AudioMNIST (~25k) shows
  discriminator overfitting (FID climbing back up, D loss diverging), enable rosinality's
  non-leaking `--augment`, but **prune its pipeline to cutout/translation only** — its
  default geometric/color transforms are invalid for mels (rotation/flip = time-frequency
  nonsense; color = undefined on 1 channel). Speech Commands (~105k) does not need it.
- **No horizontal flip, ever.** Whatever aug path you use, disable x-flip — it is time
  reversal, i.e. invalid speech.
- **Mixed precision** on.
- **R1 weight (`--r1`, the "gamma") is the hyperparameter that matters most.** Sweep 2–3
  values on short runs, pick the best by FID, then train long.
- **Checkpointing:** save periodically; results are usually reasonable by ~5000 kimg-
  equivalent — stop when FID plateaus. At 128², single Ada GPU: reasonable ~0.5 day, fuller
  convergence ~2–4 days.

Train **one generator per dataset** (AudioMNIST, Speech Commands), each on its id-disjoint
training speakers. No general/cross-dataset prior for the MVP.

---

## 6. Sidecar — the cross-repo contract (ship with every checkpoint)

Emit a JSON sidecar next to the `.pkl`/`.pt`. dlg-sonic reads this and hardcodes nothing.

```json
{
  "mel_config": {
    "sample_rate": 22050, "n_fft": 1024, "win_length": 1024,
    "hop_length": 256, "n_mels": 80, "fmin": 0, "fmax": 8000,
    "clip_seconds": 1.0, "extractor": "bigvgan.meldataset.get_mel_spectrogram"
  },
  "vocoder": "nvidia/bigvgan_v2_22khz_80band_fmax8k_256x",
  "mel_shape": [80, 86],          // real region (H, W) — from the empirical T
  "canvas": [128, 128],
  "offset": [0, 0],               // top-left of the real region in the canvas
  "pad_value_normalized": -1.0,
  "affine": {"m_lo": -11.5129, "m_hi": 0.0},   // fill m_hi with the frozen data value
  "latent": {"num_ws": 12, "w_dim": 512},      // 12 = 2*log2(128)-2
  "w_avg": "<mean latent vector, stored>",      // W+ warm-start
  "dataset": "audiomnist", "speaker_split": "id_disjoint", "split_seed": 0,
  "train_speakers": [...], "held_out_speakers": [...]
}
```

The attack side needs `w_avg`, `num_ws`, `w_dim` to drive its **W+ → ILO** search
(monotonic search-space expansion — expressiveness vs prior-tightness):

- **Phase 0** — optimize a single `w` broadcast to all layers, warm-started from `w_avg`.
- **Phase 1** — unlock the full independent `W+` (`num_ws × w_dim`).
- **Phase 2+** — unlock intermediate activation tensors of `G` (via the io forward's
  `start_layer`/`layer_in`) with an L2 proximity penalty.

So export `w_avg` (rosinality `G.mean_latent(n)`), `num_ws` (rosinality `n_latent =
2*log2(size)-2` = 12 at 128), and `w_dim` (512). The generator must be the io variant so
Phase 2+ can enter mid-network with the same weights.

**Best practice for the contract:** vendor the *exact same* `get_mel_spectrogram`,
affine, and canvas embed/crop code into both repos (a tiny shared module copied
identically), so the two can never drift.

---

## 7. Validation before handoff (do this, then hand the checkpoint over)

1. **Round-trip a real clip:** waveform → mel → affine → canvas → (crop + inverse affine) →
   BigVGAN → waveform. Must be intelligible. This validates the mel/affine/vocoder chain
   independent of the GAN.
2. **Upper bound:** vocode the **GT mel** directly (no GAN). This is the ceiling the attack
   can reach; dlg-sonic reports it to separate vocoder loss from attack loss.
3. **Sample the GAN:** `z → G → crop → inverse affine → BigVGAN`. Confirm plausible speech
   and (for multi-speaker sets) audible speaker variety.
4. **Sidecar round-trip:** load the sidecar, reconstruct the crop + inverse affine from its
   fields alone, and confirm it matches the data-prep code.
5. **Overfit-one-sample smoke test (do this FIRST, before full training).** Train the GAN to
   memorize a single mel. If that one sample flows end-to-end (data-prep → train →
   checkpoint+sidecar → attack-side load → generate → crop → inverse affine → BigVGAN →
   intelligible audio), the mel config, affine, canvas offset/crop, and vocoder wiring are
   all correct. It validates **plumbing only** — a 1-sample GAN has a degenerate latent
   space, so it says nothing about prior quality or attack fidelity.

**Three-rung sanity ladder (each rung isolates one failure mode):**

1. Overfit 1 sample → validates plumbing (item 5 above).
2. Real GAN + embed a *held-out* real sample by direct latent optimization (W+ → ILO,
   image-space L2, **no** gradient matching) → validates the prior can *represent* held-out
   targets. This is the upper bound on attack fidelity — don't skip it; a "failed attack" is
   often really a prior-capacity failure caught here.
3. Full gradient-inversion attack → any gap below rung 2 is attributable to the attack, not
   the prior.

**io-entry correctness test (when the attack wires `stylegan2_io`):** `G(w, start_layer=0)`
must equal `G(w, start_layer=k, layer_in=<the true activation captured at layer k>)`
bit-for-bit. If entering mid-network with the correct feature map doesn't reproduce the full
forward pass, io entry is broken and ILO Phase 2+ will be silently wrong.

---

## 8. Gotchas / hard "do NOT" list

- Do **not** use torchaudio/librosa — BigVGAN's `get_mel_spectrogram` only.
- Do **not** store mels as PNG — float32 `.npy` only.
- Do **not** normalize per-sample — one **global** affine, frozen, in the sidecar.
- Do **not** pad in mel space — pad the **waveform** to 1.0 s.
- Do **not** enable x-flip or geometric/color augs (default aug pipelines are for photos).
- Do **not** let `G` see held-out target speakers (id-disjoint).
- Do **not** hardcode `T` / `offset` / affine — derive, then persist to the sidecar.
- Do **not** train on Blackwell without the `sm_120` arch patch; prefer g6e (Ada).
- The canvas `offset` and real `mel_shape` here MUST equal the crop dlg-sonic performs —
  they are one contract, shared via the sidecar.

---

## 9. Reproducibility & the split contract

- **Fix seeds** at every relevant location: the speaker split, any target/sample selection,
  latent init, and the torch / numpy / CUDA RNGs — so training runs are repeatable.
- **The id-disjoint speaker split is a cross-repo contract.** This repo trains ONLY on the
  training speakers; the attack repo (dlg-sonic) targets ONLY the held-out speakers; the two
  sets must never overlap. Enforce this by sharing the **same split manifest** (a
  `speaker_split.json` listing `train_speakers` / `held_out_speakers`) *and* the seed that
  generated it — do not regenerate the split independently in each repo.
- **Record it in the sidecar.** Write the actual `train_speakers` / `held_out_speakers`
  (and the split seed) into the checkpoint sidecar (§6). dlg-sonic verifies its targets ∈
  `held_out_speakers` before running, so a split mismatch fails loudly instead of silently
  leaking the prior. (Mirrors the attack repo's reproducibility block / MVP Step 8.)
```

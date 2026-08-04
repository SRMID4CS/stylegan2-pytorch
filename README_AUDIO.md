# stylegan2-audio

A fork of [rosinality/stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch)
being converted into a **single-channel audio GAN training + eval repo**: it trains an
unconditional StyleGAN2 generator `z → mel` on log-mel spectrograms and ships a
**frozen checkpoint + JSON sidecar** consumed by the **dlg-sonic** gradient-inversion
repo as its generative prior.

The original image `README.md` is kept unchanged for reference; this file is the
readme for the audio work.

## What it produces

A generator whose samples are 128×128 single-channel "mel canvases" in `[-1,1]`;
cropping the real 80×T region and inverting a global affine yields a physical
log-mel that the pretrained **BigVGAN v2** vocoder turns back into speech. The mel
config, affine, canvas layout, and id-disjoint speaker split are a **cross-repo
contract** with dlg-sonic, carried in a sidecar JSON next to every checkpoint —
nothing is hardcoded on either side.

All design decisions are locked in **`docs/GAN_TRAINING_SPEC.md`** (read the
Glossary first). Key ones: rosinality lineage end-to-end (weights load into GIFD's
`stylegan2_io` on the attack side), BigVGAN's own mel extractor only, float32
`.npy` data (no LMDB/PNG), waveform-level padding, no flip/color augmentation,
one generator per dataset trained only on non-held-out speakers.

---

## The contract at a glance

*Everything in this section is a locked value, summarized here so the specs in
`docs/` only need opening for rationale. On any conflict the spec wins.*

### Mel config — single source of truth, must equal dlg-sonic

`sample_rate 22050 · n_fft 1024 · win_length 1024 · hop_length 256 · n_mels 80 ·
fmin 0 · fmax 8000 · clip 1.0 s (22050 samples)`

Extractor: **BigVGAN `get_mel_spectrogram` only** (vendored verbatim at
`audio/meldataset.py`, pinned commit in `audio/contract.py::BIGVGAN_COMMIT`) —
never torchaudio/librosa, their log/normalization differ silently. Vocoder:
`nvidia/bigvgan_v2_22khz_80band_fmax8k_256x`, `use_cuda_kernel=False`.
`T` is **derived, never hardcoded** (=86 at these settings; read it from the
manifest/sidecar) and is identical for every dataset.

### Data-prep order (this exact order, per clip)

1. load + resample → 22050 mono (`torchaudio.functional.resample`)
2. pad/truncate the **WAVEFORM** to 22050 samples — zeros appended, over-length
   center-cropped. **Never pad in mel space.**
3. `get_mel_spectrogram` → physical log-mel `(80, T)`
4. clamp to `[m_lo, m_hi]`, then the **global** affine → `[-1,1]`
5. embed into the 128×128 canvas at `offset`, fill `-1`
6. save float32 `.npy (1,128,128)`. **Never PNG.**

### Global affine (global, not per-sample — must be invertible attack-side)

```text
m_lo = log(1e-5) ≈ -11.5129          # fixed by the extractor's clamp
m_hi = 99.9th percentile of physical log-mel over TRAIN, frozen, PER DATASET
normalized = 2*(clamp(mel, m_lo, m_hi) - m_lo)/(m_hi - m_lo) - 1
mel        = (normalized + 1)/2 * (m_hi - m_lo) + m_lo      # inverse
```

### Sidecar (`checkpoint/<iter>.json`, written on every npy-path save)

`mel_config · resampler · vocoder · mel_shape [80,T] · canvas [128,128] ·
offset [0,0] · pad_value_normalized -1.0 · channels 1 · affine {m_lo,m_hi} ·
num_clips · content_labels · audio_aug · dataset · speaker_split · split_seed ·
train_speakers · held_out_speakers · latent {num_ws 12, w_dim 512} · w_avg[512]`

`num_ws = 2*log2(128)-2 = 12`. `w_avg = g_ema.mean_latent(4096)` under a forked,
fixed-seed(0) RNG. dlg-sonic reads all of it and hardcodes nothing.

### Dataset differences — the ONLY things that vary

| | AudioMNIST | Speech Commands v0.02 |
|---|---|---|
| Source layout | `<speaker>/<digit>_<speaker>_<rec>.wav` | `<word>/<hash>_nohash_<n>.wav` |
| Speaker id | parent dir name | filename prefix before first `_` |
| Source rate | 48 kHz | **16 kHz** |
| Content classes | 10 digits | **35 words** (incl. zero–nine) |
| Corpus | ~60 speakers, ~30k clips | 2,618 speakers, 105,829 clips |
| Held out | ~10 | **200** (→ 2,418 train, 98,690 clips) |
| Attack targets | 100 | **400** (drawn attack-side) |
| `.npy` stem | `<speaker>__<wav stem>` | `<speaker>__<word>_<wav stem>` |
| `m_hi` | its own | **its own, independent** |
| Augmentation | optional | **OFF** (past the overfitting regime) |
| Everything else | — | byte-identical (asserted by `--reference-manifest`) |

The SC stem must carry the word: the same speaker says different words with the
same `_nohash_<n>` index, so `<speaker>__<wav stem>` collides across word folders.
The speaker is always the first `__`-token in both, and survives `__aug{k}`.

### Hard "do NOT" list

- No torchaudio/librosa mel extraction; no PNG storage; no per-sample normalization.
- No mel-space padding — pad the **waveform**.
- No x-flip or geometric/color augs (x-flip = time reversal = invalid speech).
- Never let `G` see a held-out speaker; never do a sample-level split.
- Never hardcode `T` / `offset` / affine — derive, then persist to the sidecar.
- Don't reuse one dataset's `m_hi` for another, and don't widen it for OOD.
- Don't use TFDS `speech_commands` (speaker id stripped, 12-class collapse) or
  include `_background_noise_/`.
- Don't "optimize" `torchaudio.functional.resample` into a cached
  `transforms.Resample` — verified **not** bit-identical.
- Don't clean up `audio/meldataset.py`; its module-level basis/window cache is
  deliberate. Byte-identity with dlg-sonic is the point.

### Handoff to dlg-sonic

Ships per checkpoint: `<iter>.pt` + `<iter>.json` sidecar; per dataset:
`speaker_split.json`, `prep_manifest.json`, `clip_labels.json`.
Copied **verbatim** into dlg-sonic: `audio/{meldataset,env,contract,__init__}.py`
and `datasets_audio.py`.

**The asymmetry that matters:** this repo writes `.npy` for TRAIN speakers only —
held-out (target) speakers are never prepared, by design. dlg-sonic builds its own
target mels from the raw corpus using the sidecar's affine and geometry. Full
detail, schemas and verification gates: **`docs/DLG_SONIC_SC_HANDOFF.md`**.

---

## Status

Audio pipeline implemented (Phase 2): vendored BigVGAN mel contract (`audio/`),
`prepare_audio_data.py` (seeded id-disjoint split, float32 `.npy` mel canvases),
1-channel model + `--dataset npy` training path with per-checkpoint sidecars, and
`generate_audio.py` (BigVGAN and/or Griffin-Lim vocoding). Two optional,
independent augmentation mechanisms per **`docs/AUGMENTATION_SPEC.md`** (both OFF by
default): offline waveform aug at prep time (`--audio_aug`: time_shift/speed/pitch,
train speakers only, seeded) and a mel-valid ADA discriminator aug at train time
(`--augment --augment_mode audio`: time translation + cutout only); image-mode ADA
remains hard-blocked for mels.

**Two datasets, two generators** (no cross-dataset prior): AudioMNIST and, per
**`docs/SPEECH_COMMANDS_SPEC.md`**, Speech Commands v0.02 —
`prepare_audio_data.py --dataset speech_commands` walks the raw tarball (one
folder per word, speaker = the filename hash, `_background_noise_` excluded),
holds out 200 speakers, and freezes its own `m_hi`. Everything else — the mel
config, canvas, offset, `T`, model and training path — is byte-identical, and
`--reference-manifest` asserts that. The SC path is implemented and unit-tested
but not yet run on the real corpus. Next: the overfit smoke run — see
**`ROADMAP.md`** Phase 3 and `USAGE.md` §3.

## Quick start

```bash
conda env create -f env.yml
conda activate stylegan2-audio

# Audio pipeline (details + smoke-test walkthrough: USAGE.md §3)
python prepare_audio_data.py --out data/npy_audiomnist --dataset audiomnist \
  --holdout 10 --split-seed 0 <WAV_ROOT>
python train.py --size 128 --batch 8 --img_channels 1 --dataset npy \
  --seed 0 data/npy_audiomnist
python generate_audio.py --ckpt checkpoint/XXXXXX.pt --n 8 --out samples_audio/

# Speech Commands v0.02 (raw tarball, NOT TFDS — it strips the speaker id)
python prepare_audio_data.py --dataset speech_commands --out /scratch/npy_sc \
  --holdout 200 --split-seed 0 \
  --reference-manifest /scratch/npy_audiomnist/prep_manifest.json \
  /scratch/speech_commands_v0.02

# Original image pipeline still works (regression path, USAGE.md §2)
python prepare_data.py --out data/lmdb_smoke --size 128 data/img_smoke
python train.py --size 128 --batch 8 --iter 2000 data/lmdb_smoke
```

GPU notes (Blackwell laptop vs AWS g6e) and the full command/flag reference:
**`USAGE.md`**.

## Documentation map

Day-to-day, this file + `USAGE.md` + `ROADMAP.md` should be enough. The `docs/`
specs hold the **rationale** behind the locked values summarized above — open one
when you need to know *why* a decision is what it is, or before changing it.

| File | Purpose |
|---|---|
| `README_AUDIO.md` | This file — what the fork is, the contract at a glance, quick start |
| `USAGE.md` | Commands, train flags, hardware notes — updated when commands change |
| `ROADMAP.md` | Phased plan, status, decisions log — updated every session |
| `README.md` | Original rosinality readme (image pipeline) — kept as-is |
| `docs/GAN_TRAINING_SPEC.md` | Locked design decisions + cross-repo contract (**the source of truth**) |
| `docs/SPEECH_COMMANDS_SPEC.md` | Why the Speech Commands data path is the way it is (second dataset/generator) |
| `docs/AUGMENTATION_SPEC.md` | The two augmentation mechanisms (offline waveform + ADA audio mode) |
| `docs/DLG_SONIC_SC_HANDOFF.md` | Handoff to the attack repo: what ships, what to copy verbatim, how to build SC target mels |

Code comments cite these by bare filename (`GAN_TRAINING_SPEC.md §3`,
`SPEECH_COMMANDS_SPEC.md §5`, …); they all live in `docs/`.

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

All design decisions are locked in **`GAN_TRAINING_SPEC.md`** (read the Glossary
first). Key ones: rosinality lineage end-to-end (weights load into GIFD's
`stylegan2_io` on the attack side), BigVGAN's own mel extractor only, float32
`.npy` data (no LMDB/PNG), waveform-level padding, no flip/color augmentation,
one generator per dataset trained only on non-held-out speakers.

## Status

Docs + environment done; code changes not started. See **`ROADMAP.md`** for the
phased plan (image smoke test → audio modifications → overfit validation →
unit tests → full training) and current status.

## Quick start

```bash
conda env create -f env.yml
conda activate stylegan2-audio

# Image-pipeline smoke test (works today; validates the training loop as-is)
python prepare_data.py --out data/lmdb_smoke --size 128 data/img_smoke
python train.py --size 128 --batch 8 --iter 2000 data/lmdb_smoke

# Audio pipeline (planned interface — see USAGE.md §3)
python prepare_audio_data.py --out data/npy_audiomnist --dataset audiomnist <WAV_ROOT>
python train.py --size 128 --batch 8 --img_channels 1 --dataset npy data/npy_audiomnist
python generate_audio.py --ckpt checkpoint/XXXXXX.pt --n 8 --out samples_audio/
```

GPU notes (Blackwell laptop vs AWS g6e) and the full command/flag reference:
**`USAGE.md`**.

## Documentation map

| File | Purpose |
|---|---|
| `README_AUDIO.md` | This file — what the fork is, quick start |
| `GAN_TRAINING_SPEC.md` | Locked design decisions + cross-repo contract (the source of truth) |
| `ROADMAP.md` | Phased plan, status, open decisions — updated every session |
| `USAGE.md` | Commands, train flags, hardware notes — updated when commands change |
| `README.md` | Original rosinality readme (image pipeline) — kept as-is |

# Usage Guide — stylegan2-audio

Copy-paste commands and config reference for training, generating, and testing.
Run everything from the repo root with the `stylegan2-audio` conda env active.

Commands are tagged **[works now]** (original image pipeline) or **[planned]**
(audio pipeline, interface per `GAN_TRAINING_SPEC.md`; update the tag when
implemented). Status of everything lives in `ROADMAP.md`.

---

## 1. Environment

```bash
conda env create -f env.yml
conda activate stylegan2-audio
```

- PyTorch wheels are cu128; works on both the RTX 5070 Ti laptop (Blackwell,
  sm_120) and AWS g6/g6e (Ada, sm_89).
- The custom CUDA ops in `op/` **JIT-compile on first run** via
  `torch.utils.cpp_extension.load`. Requirements: `ninja` (in the env) and a
  host C++ compiler — MSVC (VS Build Tools) on Windows, gcc on Linux.
- **Blackwell (RTX 50xx) only:** if the first run fails compiling `fused` /
  `upfirdn2d`, force the arch before launching:

```bash
# bash                                   # PowerShell
export TORCH_CUDA_ARCH_LIST="12.0"       # $env:TORCH_CUDA_ARCH_LIST = "12.0"
```

- First compile takes a couple of minutes; artifacts are cached
  (`%LOCALAPPDATA%/torch_extensions` / `~/.cache/torch_extensions`).

## 2. Image pipeline (original repo) — smoke test **[works now]**

Used only to validate the training loop as-is (ROADMAP Phase 1).

```bash
# 1) Put ~10 RGB images in data/img_smoke/, then build the LMDB at 128px
python prepare_data.py --out data/lmdb_smoke --size 128 data/img_smoke

# 2) Overfit run (single GPU, ~12 GB friendly)
python train.py --size 128 --batch 8 --iter 2000 data/lmdb_smoke

# 3) Inspect progress: sample/<iter>.png every --sample_every (default 100) iters
#    Checkpoints: checkpoint/<iter>.pt at iter 0 and every --ckpt_every (default 10000)

# 4) Generate from a checkpoint
python generate.py --size 128 --ckpt checkpoint/000000.pt --sample 8 --pics 4
```

Multi-GPU (not needed for smoke): see original `README.md`.

### train.py flags that matter

| Flag | Default | Notes |
|---|---|---|
| `path` (positional) | — | LMDB dir (`--dataset lmdb`) or `.npy` dir (`--dataset npy`) |
| `--dataset` | lmdb | `npy` = audio path (float32 mel canvases from `prepare_audio_data.py`) |
| `--img_channels` | 3 | **1** for mel canvases; validated against the prep manifest |
| `--size` | 256 | use **128** for the mel canvas; validated against the manifest |
| `--batch` | 16 | per-GPU; 8 is safe on 12 GB at 128²; must be divisible by 4 (or < 4) for D's minibatch-stddev |
| `--iter` | 800000 | total iterations |
| `--ckpt_every` | 10000 | checkpoint (+ sidecar on npy path) save interval |
| `--sample_every` | 100 | sample-grid PNG interval |
| `--seed` | None | seeds python/numpy/torch/cuda RNGs (spec §9) |
| `--r1` | 10 | **the** hyperparameter to sweep for the real audio runs (spec §5) |
| `--lr` | 0.002 | |
| `--mixing` | 0.9 | style-mixing prob |
| `--channel_multiplier` | 2 | config-f width; keep 2 |
| `--ckpt` | None | resume from checkpoint |
| `--augment` | off | ADA-style non-leaking aug; on the npy path it **requires `--augment_mode audio`** |
| `--augment_mode` | image | `audio` = mel-valid transform subset: time translation + cutout only (`AUGMENTATION_SPEC.md` §B); `image` = original pipeline, unchanged |
| `--aug_time_translation` | 0.125 | audio mode: max time-axis shift as a fraction of canvas width (replicate fill) |
| `--aug_cutout` | 0.4 | audio mode: cutout size as a fraction of the canvas |
| `--augment_p` | 0 | 0 = adaptive p (`--ada_target` 0.6) |
| `--arch` | stylegan2 | `stylegan2` or `swagan` (audio: stylegan2 only) |
| `--wandb` | off | W&B logging |

Hardcoded (change in code, not CLI): `latent=512`, `n_mlp=8`. The image (lmdb)
transform applies `RandomHorizontalFlip` — fine for photos; the npy path applies
**no transform at all** (flip = time reversal, banned for mels).

## 3. Audio pipeline **[works now — validate via the smoke run below]**

Run in WSL with the `stylegan2-audio` conda env
(`cd /mnt/d/repos/stylegan2-pytorch && conda activate stylegan2-audio`).
Audio-related log lines are prefixed `[AUDIO]`.

```bash
# 0) Prepare training data: wav folder -> float32 (1,128,128) .npy canvases
#    - id-disjoint seeded speaker split (train clips only are processed)
#    - resamples to 22050 mono, pads/truncates the WAVEFORM to 1.0 s
#    - BigVGAN get_mel_spectrogram -> global affine -> canvas -> .npy
#    - writes speaker_split.json + prep_manifest.json (T, m_lo, m_hi, offset, ...)
#    Smoke data (data/audio_mnist_test): check the log shows held_out_speakers=['02']
python prepare_audio_data.py --out data/npy_smoke --holdout 1 --split-seed 0 \
  data/audio_mnist_test/data

# 1) Rung-1 chain check BEFORE training (spec §7.1-7.2): writes
#    <stem>_roundtrip_{BVG,GL}.wav and <stem>_gtvocode_{BVG,GL}.wav — must be intelligible
python generate_audio.py --roundtrip data/audio_mnist_test/data/01/0_01_0.wav \
  --manifest data/npy_smoke/prep_manifest.json --out samples_audio/

# 2) Overfit smoke train (ckpt + sidecar at iters 0/500/1000/1500)
python train.py --size 128 --batch 4 --img_channels 1 --dataset npy \
  --iter 2000 --ckpt_every 500 --seed 0 data/npy_smoke

# 3) Sample: z -> G -> crop -> inverse affine -> vocoder(s) -> .wav
#    All contract values are read from the checkpoint's sidecar JSON.
python generate_audio.py --ckpt checkpoint/001500.pt --n 8 --out samples_audio/

# Full dataset variant (AudioMNIST: hold out ~10 of 60 speakers)
python prepare_audio_data.py --out data/npy_audiomnist --dataset audiomnist \
  --holdout 10 --split-seed 0 <WAV_ROOT>
python train.py --size 128 --batch 8 --img_channels 1 --dataset npy \
  --seed 0 data/npy_audiomnist
```

### Optional augmentation (both OFF by default — `AUGMENTATION_SPEC.md`)

Rollout order (spec §E): train **clean** first; if the discriminator overfits,
enable offline `--audio_aug` and re-prep (recompute `m_hi` happens automatically);
only if still overfitting add ADA `--augment --augment_mode audio`.

```bash
# 1) Offline waveform aug at prep time (TRAIN speakers only; m_hi is recomputed
#    over the augmented set; same --seed => byte-identical .npy outputs).
#    Emits <stem>__aug{k}.npy next to the originals (~(1+N)x dataset) and records
#    everything under "audio_aug" in prep_manifest.json -> checkpoint sidecars.
python prepare_audio_data.py --out data/npy_audiomnist_aug --dataset audiomnist \
  --holdout 10 --split-seed 0 --audio_aug --aug_variants 2 --seed 0 <WAV_ROOT>
#    Tuning flags: --aug_types time_shift,speed,pitch[,gain] --time_shift_ms 100
#                  --speed_range 0.10 --pitch_semitones 2.0 --gain_db 4.0
#    Add --save_wavs (default OFF) to also write the exact 1.0 s waveforms fed to
#    the mel extractor (originals + variants) as float32 wav under <out>/wavs/ —
#    listening/inspection only; training never reads them.

# 2) ADA discriminator aug at train time (non-leaking, applied to real+fake,
#    adaptive p; mel-valid subset = time translation + cutout only).
python train.py --size 128 --batch 8 --img_channels 1 --dataset npy --seed 0 \
  --augment --augment_mode audio data/npy_audiomnist_aug
```

- `--vocoder {bigvgan,gl,both}` (default `both`) picks the waveform backend:
  **BigVGAN** (`_BVG.wav`; vendored inference code in `audio/bigvgan/`, weights
  auto-downloaded from Hugging Face on first use) and/or **Griffin-Lim**
  (`_GL.wav`; librosa inverse of the same mel basis — lower quality, no download).
- Every checkpoint save also writes `checkpoint/<iter>.json` — the **sidecar**
  consumed by dlg-sonic (mel config, canvas/offset, affine, `w_avg`/`num_ws`/
  `w_dim`, speaker split). Never edit it by hand; never hardcode its values.

### Canonical mel config (locked, spec §2)

`sample_rate=22050, n_fft=1024, win_length=1024, hop_length=256, n_mels=80,
fmin=0, fmax=8000, clip=1.0 s`. Extractor: **BigVGAN `get_mel_spectrogram` only**
(vendored verbatim at `audio/meldataset.py`, pinned commit in `audio/contract.py`;
never torchaudio/librosa). `T` is derived empirically (=86 at these settings, but
always read it from the manifest/sidecar). Vocoder:
`nvidia/bigvgan_v2_22khz_80band_fmax8k_256x`, `use_cuda_kernel=False`.

## 4. Unit tests **[partial — augmentation suite works now]**

```bash
pytest unit_test/ -v              # full suite (CPU-capable where possible)
pytest unit_test/test_audio_aug.py -v   # single file
```

Implemented (run in WSL with the env active): `test_audio_aug.py` (offline aug —
seeded byte-identical determinism incl. an end-to-end double prep run,
train-speaker-only application, `.npy` invariants, speed/pitch re-fit to 1.0 s,
zero-fill time shift, manifest record) and `test_ada_audio.py` (ADA audio mode —
identity at p=0, time-axis-only translation with replicate fill, single-rectangle
cutout, gradient flow, shape/NaN checks, seeded determinism).

Remaining Phase 4 scope (affine, canvas, waveform padding, dataset, model,
sidecar, split) is listed in `ROADMAP.md` — including which checks belong to
dlg-sonic instead.

## 5. Hardware notes

| Machine | Role | Notes |
|---|---|---|
| RTX 5070 Ti laptop (12 GB, Blackwell) | smoke tests, overfit runs | needs `TORCH_CUDA_ARCH_LIST="12.0"` for the JIT ops if compile fails |
| AWS g6e.xlarge (L40S, Ada) | full training | preferred (spec §1); ops compile cleanly |
| AWS g6.xlarge (L4, Ada) | full training (budget) | works; slower — expect longer than the 0.5–4 day estimates |

At 128², batch 8–16 fits comfortably on all of the above.

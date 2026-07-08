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

# 3) Inspect progress: sample/<iter>.png is written every 100 iters
#    Checkpoints: checkpoint/<iter>.pt at iter 0 and every 10000 iters

# 4) Generate from a checkpoint
python generate.py --size 128 --ckpt checkpoint/000000.pt --sample 8 --pics 4
```

Multi-GPU (not needed for smoke): see original `README.md`.

### train.py flags that matter

| Flag | Default | Notes |
|---|---|---|
| `path` (positional) | — | LMDB dir (image path); audio path will take the `.npy` dir |
| `--size` | 256 | use **128** for the mel canvas |
| `--batch` | 16 | per-GPU; 8 is safe on 12 GB at 128² |
| `--iter` | 800000 | total iterations |
| `--r1` | 10 | **the** hyperparameter to sweep for the real audio runs (spec §5) |
| `--lr` | 0.002 | |
| `--mixing` | 0.9 | style-mixing prob |
| `--channel_multiplier` | 2 | config-f width; keep 2 |
| `--ckpt` | None | resume from checkpoint |
| `--augment` | off | ADA-style non-leaking aug. **Keep OFF for mels** until pruned aug exists |
| `--augment_p` | 0 | 0 = adaptive p |
| `--arch` | stylegan2 | `stylegan2` or `swagan` (audio: stylegan2 only) |
| `--wandb` | off | W&B logging |

Hardcoded (change in code, not CLI): sample grid every 100 iters, checkpoint
every 10000 iters, `latent=512`, `n_mlp=8`. The image transform applies
`RandomHorizontalFlip` — fine for photos, **must not be applied to mels**
(handled by the audio data path in Phase 2).

## 3. Audio pipeline **[planned — Phase 2]**

Interfaces below are the target; update this section as they land.

```bash
# Prepare training data: wav folder -> float32 (1,128,128) .npy canvases
#   - resamples to 22050 mono, pads/truncates waveform to 1.0 s
#   - BigVGAN get_mel_spectrogram -> global affine -> 128x128 canvas
#   - writes speaker_split.json (id-disjoint, seeded) + prep manifest (T, m_lo, m_hi, offset)
python prepare_audio_data.py --out data/npy_audiomnist --dataset audiomnist \
  --split-seed 0 <WAV_ROOT>

# Train (single channel, no flip/normalize, seeded)
python train.py --size 128 --batch 8 --img_channels 1 --dataset npy \
  --seed 0 data/npy_audiomnist

# Overfit smoke (~10 clips)
python prepare_audio_data.py --out data/npy_smoke --no-split <SMALL_WAV_DIR>
python train.py --size 128 --batch 4 --img_channels 1 --dataset npy \
  --iter 2000 --ckpt_every 500 data/npy_smoke

# Generate audio: z -> G -> crop -> inverse affine -> BigVGAN -> .wav
# All contract values (crop, affine) read from the checkpoint's sidecar JSON.
python generate_audio.py --ckpt checkpoint/XXXXXX.pt --n 8 --out samples_audio/

# Vocode the GT mel directly (upper bound / chain validation, spec §7.1–7.2)
python generate_audio.py --roundtrip <some.wav> --out samples_audio/
```

Every checkpoint save also writes `<ckpt>.json` — the **sidecar** consumed by
dlg-sonic (mel config, canvas/offset, affine, `w_avg`/`num_ws`/`w_dim`, speaker
split). Never edit it by hand; never hardcode its values anywhere.

### Canonical mel config (locked, spec §2)

`sample_rate=22050, n_fft=1024, win_length=1024, hop_length=256, n_mels=80,
fmin=0, fmax=8000, clip=1.0 s`. Extractor: **BigVGAN `get_mel_spectrogram` only**
(never torchaudio/librosa). `T` is derived empirically (~86), never hardcoded.
Vocoder: `nvidia/bigvgan_v2_22khz_80band_fmax8k_256x`, `use_cuda_kernel=False`.

## 4. Unit tests **[planned — Phase 4]**

```bash
pytest unit_test/ -v          # full suite (CPU-capable where possible)
pytest unit_test/test_affine.py -v   # single file
```

Suite scope (affine round-trip, canvas embed/crop, waveform padding, dataset,
1-channel model shapes, sidecar schema, split disjointness, seeded
reproducibility) is listed in `ROADMAP.md` Phase 4 — including which checks
belong to dlg-sonic instead.

## 5. Hardware notes

| Machine | Role | Notes |
|---|---|---|
| RTX 5070 Ti laptop (12 GB, Blackwell) | smoke tests, overfit runs | needs `TORCH_CUDA_ARCH_LIST="12.0"` for the JIT ops if compile fails |
| AWS g6e.xlarge (L40S, Ada) | full training | preferred (spec §1); ops compile cleanly |
| AWS g6.xlarge (L4, Ada) | full training (budget) | works; slower — expect longer than the 0.5–4 day estimates |

At 128², batch 8–16 fits comfortably on all of the above.

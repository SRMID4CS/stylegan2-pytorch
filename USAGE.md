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

### Speech Commands v0.02 (`SPEECH_COMMANDS_SPEC.md`)

A **second, independent** generator — one per dataset, no cross-dataset prior.
Everything except `m_hi`, the dataset name, the speaker lists and `split_seed` is
byte-identical to AudioMNIST, so the only new thing is the source walk.

```bash
# 0) Get the RAW tarball (NOT TFDS speech_commands — it strips the speaker id and
#    collapses to 12 classes, so it cannot support the id-disjoint split).
#    Extract on ext4 scratch, not DrvFs /mnt/d.
curl -O http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz   # ~2.3 GB
mkdir -p /scratch/speech_commands_v0.02
tar xzf speech_commands_v0.02.tar.gz -C /scratch/speech_commands_v0.02

# 1) Prep: 35 word folders (speaker = filename hash before the first underscore),
#    _background_noise_/ skipped, 16 kHz -> 22050, 200 speakers held out.
#    --reference-manifest cross-checks the frozen mel contract (mel_shape/T,
#    mel_config, canvas, offset) against the already-prepared AudioMNIST dir.
python prepare_audio_data.py --dataset speech_commands \
  --out /scratch/npy_speech_commands --holdout 200 --split-seed 0 \
  --reference-manifest /scratch/npy_audiomnist/prep_manifest.json \
  /scratch/speech_commands_v0.02
#    -> ~97k .npy (train speakers only) ≈ 6 GB, speaker_split.json (200 held out),
#       prep_manifest.json (T == AudioMNIST's, m_lo, SC m_hi, offset), clip_labels.json

# 2) Train — augmentation stays OFF (~97k clips is past the overfitting regime)
python train.py --size 128 --batch 16 --img_channels 1 --dataset npy --seed 0 \
  --r1 <gamma from the sweep> /scratch/npy_speech_commands

# 3) Convergence curve: speaker (primary) + all 35 words + the digit-subset OOD read
python eval/convergence_curve.py --npy-dir /scratch/npy_speech_commands \
  --ckpt-glob "checkpoint/*.pt" --dataset speech_commands --label-mode both
```

SC-specific prep flags:

| Flag | Default | Notes |
|---|---|---|
| `--dataset speech_commands` | audiomnist | selects the GSC walk *and* names the sidecar |
| `--holdout` | — | **200** for SC (locked, spec §3); the 400 attack targets are drawn from those held-out speakers **attack-side**, in dlg-sonic |
| `--reference-manifest` | None | another dataset's `prep_manifest.json`; the mel contract must match it exactly or prep stops |
| `--expect-speakers` | None | fatal assert on the unique-speaker count (GSC v0.02: **2618**); without it a mismatch is a warning |
| `--mel-cache` / `--mel-cache-max-gb` | auto / 2.0 | pass-1 physical mels: RAM, or a temp memmap next to `--out`. SC (~2.7 GB of mels) auto-spills; AudioMNIST stays in RAM |

- `m_hi` is **recomputed over the SC train speakers and frozen** — never reuse
  AudioMNIST's. The affines stay independent; widening one to cover the other is
  an attack-side concern (dlg-sonic rung-2), not this repo's.
- `clip_labels.json` (`{npy stem: {speaker, content}}` + `content_classes` +
  `digit_classes`) is written for every dataset and is what the convergence eval
  reads, so labels never have to be re-parsed out of filenames.
- SC `.npy` stems are `<speaker>__<word>_<wav stem>`: the same speaker says
  different words with the same `_nohash_<n>` index, so the word has to be in the
  stem or files collide. AudioMNIST naming (`<speaker>__<wav stem>`) is unchanged.
- `_background_noise_/` is skipped, and `validation_list.txt` / `testing_list.txt`
  are ignored — those are KWS splits, not the id-disjoint attack split.

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

### Checkpoint convergence curve (Route B: domain-FD + coverage entropy)

Mel-native, no vocoding — scores every checkpoint in seconds. Reference stats
and the small mel classifier are cached per `(--dataset, --label-mode)` on first
run (later runs hit the cache instead of retraining). Two label modes: `content`
(digit/keyword curve; saturates ~99.8% and is blind to speaker collapse) and
`speaker` (speaker-ID curve — the collapse tripwire for the N1 speaker-diversity
claim). `--label-mode both` (default) emits both from one sampling pass:

```bash
python eval/convergence_curve.py \
  --npy-dir data/npy_audiomnist_aug --ckpt-glob "checkpoint/*.pt" \
  --dataset audiomnist --num-classes 10 --label-mode both
#   -> convergence_cache/audiomnist/{content,speaker}/{classifier.pt,reference_stats.npz} (cached)
#      convergence_curve_content.csv, convergence_curve_speaker.csv  (ALL checkpoints, incl. iter 0)
#      convergence_curve.png  (2-row: content top, speaker bottom; iter-0 omitted from the plot)

# just one curve:
python eval/convergence_curve.py --npy-dir data/npy_audiomnist_aug \
  --dataset audiomnist --num-classes 10 --label-mode speaker

# force classifier/reference retrain (e.g. after re-prepping the data):
python eval/convergence_curve.py --npy-dir data/npy_audiomnist_aug --retrain \
  --dataset audiomnist --num-classes 10 --label-mode both

# AWS full-run layout (matches sweep.sh/train_full.sh):
python eval/convergence_curve.py \
  --npy-dir /scratch/npy_audiomnist_aug --ckpt-glob "runs/full_g2/checkpoint/*.pt" \
  --dataset audiomnist --num-classes 10 --label-mode both

# Speech Commands: 35-word content curve + speaker curve + the digit-subset read.
# --num-classes is unnecessary — the class set comes from clip_labels.json.
python eval/convergence_curve.py \
  --npy-dir /scratch/npy_speech_commands --ckpt-glob "runs/full_g2_sc/checkpoint/*.pt" \
  --dataset speech_commands --label-mode both
#   -> convergence_curve_{content,speaker}.csv + convergence_curve_digits.csv
```

- `--npy-dir` must be TRAIN-speaker mels only (spec §3, §9 id-disjoint contract)
  — any dir written by `prepare_audio_data.py` already satisfies this, since it
  never writes held-out-speaker clips.
- `--label-mode` `content` | `speaker` | `both` (default `both`). Speaker mode
  derives its class set (`K` = number of train speakers, remapped to `0..K-1`)
  from `speaker_split.json` — `--num-classes` is ignored there. Speaker ID from
  1 s clips over ~48 classes (AudioMNIST) or ~2418 (SC) is genuinely hard: watch
  the logged `train_acc`, and if it sits near chance (`1/K`) the speaker embedding
  is uninformative — raise `--clf-epochs` (default 20) to train it longer.
  **Speaker-coverage entropy stays the primary signal** on both datasets: content
  entropy structurally cannot detect speaker-manifold collapse.
- Labels come from `clip_labels.json` in `--npy-dir` (speaker + digit/word,
  straight from the source walk). Content `K` is derived from it, so
  `--num-classes` is only a manual override and warns when it disagrees. A dir
  prepped before that file existed still works via stem parsing. A `labels.json`
  (`{npy_stem: int}`) next to `--npy-dir` overrides everything, as before.
- `--digit-coverage {auto,on,off}` (default `auto`, Speech Commands only): reports
  what fraction of generated mels the 35-word classifier calls a **zero–nine**
  word and how evenly across those ten — the credibility hook for the
  SC-prior → AudioMNIST OOD attack (`SPEECH_COMMANDS_SPEC.md` §5). It rides on the
  content classifier, so it needs no extra sampling, and it is a *reporting*
  signal, never a checkpoint selector. Emits `<out>_digits.csv` + a third panel.
- `--seed` / `--n-samples` are the frozen-comparability contract: keep them
  identical across a whole sweep, or the absolute FD numbers aren't comparable
  checkpoint-to-checkpoint.
- `--plot-omit-iter0` (default `True`) drops the untrained iter-0 point (whose
  blown-out FD would crush the y-axis) from the **PNG only** — both CSVs always
  keep it as the sanity anchor. Pass `--plot-omit-iter0 False` to plot it too.
- `--crop-to-sidecar` (off by default) crops real+generated mels to the real
  80×T region (via `prep_manifest.json` / each checkpoint's sidecar) instead of
  embedding the full padded 128×128 canvas.
- Relative tripwire only — not an absolute FID, never report the number outside
  this repo (ROADMAP.md "FID on mels" decision).

### AWS run scripts (`sweep.sh`, `train_full.sh`)

Both are dataset-agnostic — every knob is an env override, and the defaults
reproduce the original AudioMNIST runs. The R1 sweep logic (2–3 `--r1` values on
short runs, pick by the convergence curve, then train long) is identical for
Speech Commands; only the data dir, the eval namespace and the augmentation
switch differ.

```bash
# R1 sweep (runs the convergence curve after each gamma, frozen --seed/--n-samples)
./sweep.sh                                              # AudioMNIST defaults
DATA=/scratch/npy_speech_commands DATASET=speech_commands \
  AUGMENT=false GAMMAS="10 20" ./sweep.sh               # Speech Commands

# Long run at the chosen gamma
GAMMA=2 ./train_full.sh                                 # AudioMNIST defaults
DATA=/scratch/npy_speech_commands DATASET=speech_commands \
  AUGMENT=false RUN_TAG=sc GAMMA=<winner> ./train_full.sh
```

`RUN_TAG` suffixes `runs/full_g<gamma>` so the two datasets' checkpoints and
sidecars never mix (one generator per dataset — spec §5). `AUGMENT=false` for SC:
~97k clips is well past the overfitting regime, so ADA stays off.

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
zero-fill time shift, manifest record), `test_ada_audio.py` (ADA audio mode —
identity at p=0, time-axis-only translation with replicate fill, single-rectangle
cutout, gradient flow, shape/NaN checks, seeded determinism) and
`test_speech_commands.py` (the SC data path — word-folder walk,
`_background_noise_` exclusion, speaker-hash parse + malformed-filename logging,
collision-free stems, 200-speaker seeded disjoint split, streaming `m_hi`
percentile vs `np.percentile`, memory/disk mel-cache equivalence, derived `T`,
`clip_labels.json`, `--reference-manifest` drift detection, and the eval's SC
label providers + digit-subset coverage).

`test_speech_commands.py` **never needs the real 2.3 GB tarball**: it builds a
miniature corpus with the same layout and runs the real `prepare_audio_data.py`
over it, so the whole SC path is testable on any machine.

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

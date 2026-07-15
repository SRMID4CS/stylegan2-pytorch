"""Pre-save mel canvases for GAN training (GAN_TRAINING_SPEC.md §3).

wav folder -> id-disjoint speaker split -> per TRAIN clip:
resample 22050 mono -> pad/truncate WAVEFORM to 1.0 s -> BigVGAN mel (n_mels, T)
-> global affine [-1,1] -> embed in canvas -> float32 .npy (1, H, W).

Optional offline waveform augmentation (`--audio_aug`, AUGMENTATION_SPEC.md §A):
each TRAIN clip additionally emits `--aug_variants` augmented copies
(`<stem>__aug{k}.npy`), composed per variant as speed -> pitch on the raw
waveform, re-fit to 1.0 s, then time_shift (and optional gain) on the fixed
frame. Held-out speakers are never augmented; `m_hi` is computed over the FULL
augmented train set; all aug RNG is driven by `--seed`.

Also emits `speaker_split.json` and `prep_manifest.json` in --out; train.py copies
both into every checkpoint sidecar. Nothing here is hardcoded — all contract values
come from audio.contract.MEL_CONFIG, CLI flags, or are derived and persisted.

Example (smoke test):
    python prepare_audio_data.py --out data/npy_smoke --holdout 1 --split-seed 0 \
        data/audio_mnist_test/data
"""

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from audio.contract import (
    MEL_CONFIG,
    M_LO,
    RESAMPLER_ID,
    VOCODER_ID,
    affine_forward,
    clip_num_samples,
    embed_canvas,
    extract_mel,
    pad_or_truncate_waveform,
)

# AudioMNIST clip naming: <digit>_<speaker_id>_<recording_id>.wav
FILENAME_RE = re.compile(r"^(?P<digit>\d+)_(?P<speaker>[^_]+)_(?P<rec>\d+)\.wav$")


def log(msg):
    print(f"[AUDIO] {msg}")


def collect_clips(wav_root):
    """Return {speaker_id: [wav paths]}; speaker = parent dir name, cross-checked
    against the filename convention."""
    clips = {}
    for path in sorted(Path(wav_root).rglob("*.wav")):
        speaker = path.parent.name
        m = FILENAME_RE.match(path.name)
        if m and m.group("speaker") != speaker:
            log(
                f"WARNING: {path.name}: filename speaker '{m.group('speaker')}' "
                f"!= parent dir '{speaker}' — using parent dir"
            )
        clips.setdefault(speaker, []).append(path)
    return clips


def split_speakers(speakers, holdout, seed):
    """Seeded id-disjoint speaker split (spec §3, §9)."""
    shuffled = sorted(speakers)
    random.Random(seed).shuffle(shuffled)
    held_out = sorted(shuffled[:holdout])
    train = sorted(shuffled[holdout:])
    return train, held_out


def load_waveform(path, sample_rate):
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0)  # mono
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav


# --- Offline waveform augmentation (AUGMENTATION_SPEC.md §A) -----------------
# Waveform-level, offline-only: the training DataLoader still reads plain .npy,
# so training throughput is identical to the no-aug path.

AUG_CHOICES = ("time_shift", "speed", "pitch", "gain")


def parse_aug_types(csv):
    types = [t.strip() for t in csv.split(",") if t.strip()]
    unknown = [t for t in types if t not in AUG_CHOICES]
    if unknown:
        sys.exit(f"[AUDIO] unknown --aug_types {unknown}; choose from {list(AUG_CHOICES)}")
    if not types:
        sys.exit("[AUDIO] --aug_types is empty")
    return types


def variant_rng(seed, clip_key, variant):
    """RNG unique to (seed, clip, variant): param draws are independent of
    processing order, so the same --seed gives byte-identical outputs (spec A.4)."""
    digest = hashlib.sha256(clip_key.encode("utf-8")).digest()
    return np.random.default_rng(
        np.random.SeedSequence([seed, variant, int.from_bytes(digest[:8], "little")])
    )


def draw_aug_params(rng, aug_types, args, sample_rate):
    """One composed parameter set per variant, each aug drawn from its range (spec A.2/A.3)."""
    params = {}
    if "speed" in aug_types:
        params["speed_rate"] = float(rng.uniform(1.0 - args.speed_range, 1.0 + args.speed_range))
    if "pitch" in aug_types:
        params["pitch_semitones"] = float(rng.uniform(-args.pitch_semitones, args.pitch_semitones))
    if "time_shift" in aug_types:
        max_shift = args.time_shift_ms / 1000.0 * sample_rate
        params["time_shift_samples"] = int(round(rng.uniform(-max_shift, max_shift)))
    if "gain" in aug_types:
        params["gain_db"] = float(rng.uniform(-args.gain_db, args.gain_db))
    return params


def time_shift_waveform(wav, shift):
    """Linear (non-circular) shift within the fixed 1 s frame: vacated region is
    zero-filled, content pushed past the edge is dropped (spec A.3)."""
    out = np.zeros_like(wav)
    if shift == 0:
        out[:] = wav
    elif shift > 0:
        out[shift:] = wav[:-shift]
    else:
        out[:shift] = wav[-shift:]
    return out


def augment_clip(wav, params, num_samples, sample_rate):
    """One augmented 1.0 s variant, spec A.2 order: speed -> pitch on the
    variable-length waveform, re-fit to exactly `num_samples` (rate>1 zero-pads,
    rate<1 center-crops), then frame-relative time_shift (+ optional gain)."""
    x = wav.numpy().astype(np.float32)
    if "speed_rate" in params:
        x = librosa.effects.time_stretch(x, rate=params["speed_rate"])
    if "pitch_semitones" in params:
        x = librosa.effects.pitch_shift(x, sr=sample_rate, n_steps=params["pitch_semitones"])
    x = pad_or_truncate_waveform(
        torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), num_samples
    ).numpy()
    if "time_shift_samples" in params:
        x = time_shift_waveform(x, params["time_shift_samples"])
    if "gain_db" in params:
        x = x * np.float32(10.0 ** (params["gain_db"] / 20.0))
    return torch.from_numpy(x.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description="wav -> float32 mel-canvas .npy")
    parser.add_argument("wav_root", type=str, help="root folder of <speaker>/<clip>.wav")
    parser.add_argument("--out", type=str, required=True, help="output directory")
    parser.add_argument("--dataset", type=str, default="audiomnist", help="dataset name for the sidecar")
    parser.add_argument("--split-seed", type=int, default=0, help="seed for the speaker split")
    parser.add_argument("--holdout", type=int, default=None, help="number of speakers to hold out")
    parser.add_argument("--no-split", action="store_true", help="all speakers train (smoke only)")
    parser.add_argument("--canvas", type=int, nargs=2, default=[128, 128], metavar=("H", "W"))
    parser.add_argument("--offset", type=int, nargs=2, default=[0, 0], metavar=("ROW", "COL"))
    parser.add_argument("--pad-value", type=float, default=-1.0, help="normalized canvas fill value")
    parser.add_argument(
        "--m-hi-percentile", type=float, default=99.9,
        help="percentile of physical log-mel values over the TRAIN set used as m_hi",
    )
    parser.add_argument(
        "--audio_aug", action="store_true",
        help="offline waveform augmentation of TRAIN clips (AUGMENTATION_SPEC.md §A)",
    )
    parser.add_argument(
        "--aug_types", type=str, default="time_shift,speed,pitch",
        help=f"csv subset of {list(AUG_CHOICES)} to compose per variant",
    )
    parser.add_argument(
        "--aug_variants", type=int, default=2,
        help="augmented copies per source clip (dataset grows ~(1+N)x)",
    )
    parser.add_argument(
        "--time_shift_ms", type=float, default=100.0,
        help="max +/- time shift in ms (zero-filled, non-circular)",
    )
    parser.add_argument(
        "--speed_range", type=float, default=0.10,
        help="max +/- fractional time-stretch rate (0.10 -> 0.90-1.10)",
    )
    parser.add_argument(
        "--pitch_semitones", type=float, default=2.0,
        help="max +/- pitch shift in semitones",
    )
    parser.add_argument(
        "--gain_db", type=float, default=4.0,
        help="max +/- gain in dB (only used if 'gain' is in --aug_types)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="seed driving ALL augmentation RNG (same seed -> byte-identical .npy)",
    )
    parser.add_argument(
        "--save_wavs", action="store_true",
        help="also write the exact 1.0 s waveforms fed to the mel extractor "
        "(original + augmented variants) as float32 wav under <out>/wavs/ — "
        "listening/inspection only, not read by training",
    )
    args = parser.parse_args()

    aug_types = parse_aug_types(args.aug_types) if args.audio_aug else []
    if args.audio_aug and args.aug_variants < 1:
        sys.exit("[AUDIO] --aug_variants must be >= 1")

    if args.no_split:
        holdout = 0
    elif args.holdout is None:
        parser.error("pass --holdout N (id-disjoint split, spec §3) or --no-split")
    else:
        holdout = args.holdout

    clips = collect_clips(args.wav_root)
    if not clips:
        sys.exit(f"[AUDIO] no .wav files under {args.wav_root}")
    log(f"found {sum(len(v) for v in clips.values())} clips from {len(clips)} speakers: {sorted(clips)}")

    train_speakers, held_out_speakers = split_speakers(clips, holdout, args.split_seed)
    if not train_speakers:
        sys.exit(
            f"[AUDIO] split left no training speakers: --holdout {holdout} >= "
            f"{len(clips)} speakers found — lower --holdout (seed does not affect this)"
        )
    log(
        f"split_seed={args.split_seed} train_speakers={train_speakers} "
        f"held_out_speakers={held_out_speakers}"
    )

    num_samples = clip_num_samples(MEL_CONFIG)
    log(f"clip length: {MEL_CONFIG['clip_seconds']} s = {num_samples} samples @ {MEL_CONFIG['sample_rate']} Hz")
    if args.audio_aug:
        log(
            f"audio_aug enabled: types={aug_types} variants={args.aug_variants} "
            f"seed={args.seed} (TRAIN speakers only, m_hi over the augmented set)"
        )

    wav_dir = None
    if args.save_wavs:
        wav_dir = Path(args.out) / "wavs"
        wav_dir.mkdir(parents=True, exist_ok=True)
        log(f"save_wavs enabled: 1.0 s float32 waveforms -> {wav_dir}")

    # Pass 1 — extract physical log-mels for TRAIN clips only (original + augmented
    # variants); derive T; m_hi stats come from the FULL augmented set (spec A.4).
    train_clips = [(spk, p) for spk in train_speakers for p in clips[spk]]
    mels = []
    mel_shape = None
    for spk, path in tqdm(train_clips, desc="[AUDIO] extract mels", dynamic_ncols=True):
        wav = load_waveform(path, MEL_CONFIG["sample_rate"])
        variants = [(f"{spk}__{path.stem}", pad_or_truncate_waveform(wav, num_samples))]
        for k in range(1, args.aug_variants + 1 if args.audio_aug else 1):
            rng = variant_rng(args.seed, f"{spk}/{path.stem}", k)
            params = draw_aug_params(rng, aug_types, args, MEL_CONFIG["sample_rate"])
            variants.append(
                (f"{spk}__{path.stem}__aug{k}", augment_clip(wav, params, num_samples, MEL_CONFIG["sample_rate"]))
            )
        for stem, clip in variants:
            if clip.shape[-1] != num_samples:
                sys.exit(f"[AUDIO] {stem}: {clip.shape[-1]} samples after aug (expected {num_samples})")
            if wav_dir is not None:
                # float32 (not PCM16): exact record of the extractor input; gain
                # variants may exceed [-1,1] and would clip in integer formats.
                torchaudio.save(str(wav_dir / f"{stem}.wav"), clip.unsqueeze(0), MEL_CONFIG["sample_rate"])
            mel = extract_mel(clip, MEL_CONFIG).to(torch.float32)
            if mel_shape is None:
                mel_shape = tuple(mel.shape)
                log(f"mel_shape=({mel_shape[0]}, {mel_shape[1]}) (T={mel_shape[1]}, derived empirically)")
            elif tuple(mel.shape) != mel_shape:
                sys.exit(f"[AUDIO] inconsistent mel shape {tuple(mel.shape)} for {stem} (expected {mel_shape})")
            mels.append((stem, mel))

    m_hi = float(np.percentile(np.concatenate([m.numpy().ravel() for _, m in mels]), args.m_hi_percentile))
    if m_hi <= M_LO:
        sys.exit(f"[AUDIO] degenerate affine: m_hi={m_hi} <= m_lo={M_LO}")
    log(f"affine m_lo={M_LO:.6f} m_hi={m_hi:.6f} (method=percentile_{args.m_hi_percentile}_train)")

    # Pass 2 — normalize, embed, save.
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas = tuple(args.canvas)
    offset = tuple(args.offset)
    for stem, mel in tqdm(mels, desc="[AUDIO] write npy", dynamic_ncols=True):
        norm = affine_forward(mel, M_LO, m_hi)
        sample = embed_canvas(norm, canvas, offset, args.pad_value)
        np.save(out_dir / f"{stem}.npy", sample.numpy().astype(np.float32))

    split = {
        "dataset": args.dataset,
        "speaker_split": "id_disjoint",
        "split_seed": args.split_seed,
        "train_speakers": train_speakers,
        "held_out_speakers": held_out_speakers,
    }
    manifest = {
        "mel_config": dict(MEL_CONFIG),
        "resampler": RESAMPLER_ID,
        "vocoder": VOCODER_ID,
        "mel_shape": list(mel_shape),
        "canvas": list(canvas),
        "offset": list(offset),
        "pad_value_normalized": args.pad_value,
        "channels": 1,
        "affine": {
            "m_lo": M_LO,
            "m_hi": m_hi,
            "m_hi_method": f"percentile_{args.m_hi_percentile}_train",
        },
        "num_clips": len(mels),
        "num_source_clips": len(train_clips),
        # Offline waveform augmentation record (AUGMENTATION_SPEC.md A.4): makes the
        # training distribution reproducible and visible to dlg-sonic via the sidecar.
        "audio_aug": {
            "enabled": bool(args.audio_aug),
            "aug_types": aug_types,
            "aug_variants": args.aug_variants if args.audio_aug else 0,
            "seed": args.seed,
            "params": {
                "time_shift_ms": args.time_shift_ms,
                "speed_range": args.speed_range,
                "pitch_semitones": args.pitch_semitones,
                "gain_db": args.gain_db,
            },
        },
    }
    with open(out_dir / "speaker_split.json", "w") as f:
        json.dump(split, f, indent=2)
    with open(out_dir / "prep_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log(f"wrote {len(mels)} .npy canvases + speaker_split.json + prep_manifest.json -> {out_dir}")


if __name__ == "__main__":
    main()

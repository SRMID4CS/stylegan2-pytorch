"""Pre-save mel canvases for GAN training (GAN_TRAINING_SPEC.md §3).

wav folder -> id-disjoint speaker split -> per TRAIN clip:
resample 22050 mono -> pad/truncate WAVEFORM to 1.0 s -> BigVGAN mel (n_mels, T)
-> global affine [-1,1] -> embed in canvas -> float32 .npy (1, H, W).

Also emits `speaker_split.json` and `prep_manifest.json` in --out; train.py copies
both into every checkpoint sidecar. Nothing here is hardcoded — all contract values
come from audio.contract.MEL_CONFIG, CLI flags, or are derived and persisted.

Example (smoke test):
    python prepare_audio_data.py --out data/npy_smoke --holdout 1 --split-seed 0 \
        data/audio_mnist_test/data
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

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
    args = parser.parse_args()

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
        sys.exit("[AUDIO] split left no training speakers")
    log(
        f"split_seed={args.split_seed} train_speakers={train_speakers} "
        f"held_out_speakers={held_out_speakers}"
    )

    num_samples = clip_num_samples(MEL_CONFIG)
    log(f"clip length: {MEL_CONFIG['clip_seconds']} s = {num_samples} samples @ {MEL_CONFIG['sample_rate']} Hz")

    # Pass 1 — extract physical log-mels for TRAIN clips only; derive T; gather m_hi stats.
    train_clips = [(spk, p) for spk in train_speakers for p in clips[spk]]
    mels = []
    mel_shape = None
    for spk, path in tqdm(train_clips, desc="[AUDIO] extract mels", dynamic_ncols=True):
        wav = load_waveform(path, MEL_CONFIG["sample_rate"])
        wav = pad_or_truncate_waveform(wav, num_samples)
        mel = extract_mel(wav, MEL_CONFIG).to(torch.float32)
        if mel_shape is None:
            mel_shape = tuple(mel.shape)
            log(f"mel_shape=({mel_shape[0]}, {mel_shape[1]}) (T={mel_shape[1]}, derived empirically)")
        elif tuple(mel.shape) != mel_shape:
            sys.exit(f"[AUDIO] inconsistent mel shape {tuple(mel.shape)} for {path} (expected {mel_shape})")
        mels.append((spk, path, mel))

    m_hi = float(np.percentile(np.concatenate([m.numpy().ravel() for _, _, m in mels]), args.m_hi_percentile))
    if m_hi <= M_LO:
        sys.exit(f"[AUDIO] degenerate affine: m_hi={m_hi} <= m_lo={M_LO}")
    log(f"affine m_lo={M_LO:.6f} m_hi={m_hi:.6f} (method=percentile_{args.m_hi_percentile}_train)")

    # Pass 2 — normalize, embed, save.
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas = tuple(args.canvas)
    offset = tuple(args.offset)
    for spk, path, mel in tqdm(mels, desc="[AUDIO] write npy", dynamic_ncols=True):
        norm = affine_forward(mel, M_LO, m_hi)
        sample = embed_canvas(norm, canvas, offset, args.pad_value)
        np.save(out_dir / f"{spk}__{path.stem}.npy", sample.numpy().astype(np.float32))

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
    }
    with open(out_dir / "speaker_split.json", "w") as f:
        json.dump(split, f, indent=2)
    with open(out_dir / "prep_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log(f"wrote {len(mels)} .npy canvases + speaker_split.json + prep_manifest.json -> {out_dir}")


if __name__ == "__main__":
    main()

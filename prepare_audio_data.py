"""Pre-save mel canvases for GAN training (GAN_TRAINING_SPEC.md §3).

wav folder -> id-disjoint speaker split -> per TRAIN clip:
resample 22050 mono -> pad/truncate WAVEFORM to 1.0 s -> BigVGAN mel (n_mels, T)
-> global affine [-1,1] -> embed in canvas -> float32 .npy (1, H, W).

Two source layouts (`--dataset`, see datasets_audio.py):
  audiomnist       one folder per speaker; `<digit>_<speaker>_<rec>.wav` (48 kHz)
  speech_commands  raw GSC v0.02 tarball: one folder per WORD, speaker = the hash
                   before the first underscore, `_background_noise_/` skipped,
                   16 kHz in (SPEECH_COMMANDS_SPEC.md §1-§2)
The pipeline itself is byte-identical for both — only the walk, the speaker/content
parse, `m_hi`, the speaker lists and `split_seed` are per-dataset (SC spec §0).

Optional offline waveform augmentation (`--audio_aug`, AUGMENTATION_SPEC.md §A):
each TRAIN clip additionally emits `--aug_variants` augmented copies
(`<stem>__aug{k}.npy`), composed per variant as speed -> pitch on the raw
waveform, re-fit to 1.0 s, then time_shift (and optional gain) on the fixed
frame. Held-out speakers are never augmented; `m_hi` is computed over the FULL
augmented train set; all aug RNG is driven by `--seed`.

Emits `speaker_split.json`, `prep_manifest.json` and `clip_labels.json` in --out;
train.py copies the first two into every checkpoint sidecar and
eval/convergence_curve.py reads the third. Nothing here is hardcoded — all
contract values come from audio.contract.MEL_CONFIG, CLI flags, or are derived
and persisted.

Examples:
    # AudioMNIST smoke test
    python prepare_audio_data.py --out data/npy_smoke --holdout 1 --split-seed 0 \
        data/audio_mnist_test/data
    # Speech Commands v0.02 (200 held-out speakers, SC spec §3)
    python prepare_audio_data.py --dataset speech_commands --out /scratch/npy_sc \
        --holdout 200 --split-seed 0 \
        --reference-manifest /scratch/npy_audiomnist/prep_manifest.json \
        /scratch/speech_commands_v0.02
"""

import argparse
import hashlib
import json
import random
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
    expected_mel_frames,
    extract_mel,
    pad_or_truncate_waveform,
)
from datasets_audio import (
    AUDIOMNIST_FILENAME_RE,
    SC_NUM_SPEAKERS,
    SPEECH_COMMANDS,
    clip_stem,
    collect_clips,
    content_label_type,
    digit_words,
)

# Back-compat alias: the AudioMNIST filename rule now lives in datasets_audio.py.
FILENAME_RE = AUDIOMNIST_FILENAME_RE


def log(msg):
    print(f"[AUDIO] {msg}")


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


# --- Pass-1 mel buffer (scale: Speech Commands is ~4x AudioMNIST) ------------
# `m_hi` is a percentile over the whole TRAIN set, so every physical mel has to
# survive pass 1. AudioMNIST (~25k clips) is 0.7 GB and fits in RAM; Speech
# Commands (~97k) is ~2.7 GB of mels plus a same-sized copy inside
# np.percentile's sort — enough to OOM a 32 GB prep box. MelStore keeps the
# small case in RAM (bit-identical to the original code path) and spills the
# large case to a temp memmap next to --out, computing the percentile from a
# bounded top-K stream instead of a full sort. Re-extracting in pass 2 was the
# alternative but would double the (slow) librosa aug cost.


class StreamingPercentile:
    """Percentile of a value stream in O(K) memory, K = values above the cut.

    Only the largest `total - floor((total-1)*p/100)` values can affect a high
    percentile, so we keep exactly those and drop the rest. Matches
    `np.percentile(all_values, p)` (method="linear") to float32 rounding —
    `unit_test/test_speech_commands.py` checks that against the real thing.
    """

    def __init__(self, total, percentile, chunk=1 << 22):
        if total <= 0:
            raise ValueError("StreamingPercentile needs total > 0")
        self.total = int(total)
        self.percentile = float(percentile)
        idx = (self.total - 1) * self.percentile / 100.0
        self.lo = int(np.floor(idx))
        self.hi = int(np.ceil(idx))
        self.frac = idx - self.lo
        self.k = min(self.total, self.total - self.lo + 1)
        self.chunk = chunk
        self.top = np.empty(0, dtype=np.float32)
        self._buf, self._buf_n, self.seen = [], 0, 0

    def update(self, values):
        values = np.asarray(values, dtype=np.float32).ravel()
        self._buf.append(values)
        self._buf_n += values.size
        self.seen += values.size
        if self._buf_n >= self.chunk:
            self._compact()

    def _compact(self):
        if not self._buf:
            return
        arr = np.concatenate([self.top, *self._buf])
        if arr.size > self.k:
            arr = np.partition(arr, arr.size - self.k)[-self.k:]
        self.top, self._buf, self._buf_n = arr, [], 0

    def value(self):
        if self.seen != self.total:
            raise RuntimeError(f"saw {self.seen} values, expected {self.total}")
        self._compact()
        top = np.sort(self.top)                      # ascending; global ranks base..total-1
        base = self.total - top.size
        a = float(top[self.lo - base])
        b = float(top[self.hi - base])
        # numpy's _lerp, so the two paths agree on the endpoints and the midpoint.
        diff = b - a
        return b - diff * (1.0 - self.frac) if self.frac >= 0.5 else a + diff * self.frac


class MelStore:
    """Ordered (stem, physical log-mel) buffer for pass 1, RAM- or disk-backed."""

    def __init__(self, total, mel_shape, percentile, backend, tmp_path):
        self.total = int(total)
        self.mel_shape = tuple(mel_shape)
        self.backend = backend
        self.tmp_path = Path(tmp_path)
        self.stems = []
        if backend == "disk":
            self.tmp_path.parent.mkdir(parents=True, exist_ok=True)
            self._mm = np.lib.format.open_memmap(
                self.tmp_path, mode="w+", dtype=np.float32,
                shape=(self.total, *self.mel_shape),
            )
            self._acc = StreamingPercentile(self.total * int(np.prod(self.mel_shape)), percentile)
        else:
            self._mels = []
            self._mm = self._acc = None

    def append(self, stem, mel):
        i = len(self.stems)
        if i >= self.total:
            raise RuntimeError(f"MelStore overflow: more than {self.total} clips")
        arr = mel.numpy().astype(np.float32, copy=False)
        if self.backend == "disk":
            self._mm[i] = arr
            self._acc.update(arr)
        else:
            self._mels.append(arr)
        self.stems.append(stem)

    def __len__(self):
        return len(self.stems)

    def percentile(self, p):
        if self.backend == "disk":
            return float(self._acc.value())
        return float(np.percentile(np.concatenate([m.ravel() for m in self._mels]), p))

    def __iter__(self):
        for i, stem in enumerate(self.stems):
            arr = np.asarray(self._mm[i]) if self.backend == "disk" else self._mels[i]
            yield stem, torch.from_numpy(np.ascontiguousarray(arr))

    def close(self):
        if self.backend == "disk":
            self._mm = None
            self.tmp_path.unlink(missing_ok=True)


def choose_mel_backend(mode, total, mel_shape, max_gb):
    est_gb = total * float(np.prod(mel_shape)) * 4 / (1 << 30)
    if mode == "auto":
        backend = "disk" if est_gb > max_gb else "memory"
    else:
        backend = mode
    log(
        f"pass-1 mel buffer: {total} mels x {mel_shape} ~= {est_gb:.2f} GB -> "
        f"{backend} (--mel-cache {mode}, threshold {max_gb} GB)"
    )
    return backend


def check_mel_shape(mel_shape, num_samples, reference_manifest):
    """Assert the empirically derived (n_mels, T) is the frozen one (SC spec §0).

    `T` is never hardcoded: it is compared against the extractor's own framing
    (audio.contract.expected_mel_frames) and, when --reference-manifest is given,
    against the manifest of the already-prepared dataset (i.e. AudioMNIST's T).
    """
    n_mels, T = mel_shape
    expected_T = expected_mel_frames(num_samples, MEL_CONFIG)
    if n_mels != MEL_CONFIG["n_mels"] or T != expected_T:
        sys.exit(
            f"[AUDIO] mel shape drift: extracted {mel_shape}, contract predicts "
            f"({MEL_CONFIG['n_mels']}, {expected_T}) for {num_samples} samples — stop and "
            "investigate (GAN_TRAINING_SPEC.md §2, SPEECH_COMMANDS_SPEC.md §0)"
        )
    if reference_manifest is None:
        return
    ref_path = Path(reference_manifest)
    if not ref_path.is_file():
        sys.exit(f"[AUDIO] --reference-manifest {ref_path} not found")
    ref = json.loads(ref_path.read_text())
    if list(ref["mel_shape"]) != list(mel_shape):
        sys.exit(
            f"[AUDIO] mel_shape {list(mel_shape)} != reference {ref['mel_shape']} from "
            f"{ref_path} — the frozen mel contract must be identical across datasets "
            "(SPEECH_COMMANDS_SPEC.md §0); stop and report"
        )
    for key in ("mel_config", "canvas", "offset", "pad_value_normalized"):
        if key in ref and ref[key] != CONTRACT_ECHO[key]:
            sys.exit(
                f"[AUDIO] {key} {CONTRACT_ECHO[key]!r} != reference {ref[key]!r} from {ref_path} "
                "— only m_hi, the dataset name, the speaker lists and split_seed may differ "
                "between datasets (SPEECH_COMMANDS_SPEC.md §0)"
            )
    log(f"mel contract matches {ref_path} (mel_shape={list(mel_shape)})")


# Filled in by main() before check_mel_shape() runs; keeps the comparison honest
# by echoing exactly what this run is about to persist.
CONTRACT_ECHO = {}


def main():
    parser = argparse.ArgumentParser(description="wav -> float32 mel-canvas .npy")
    parser.add_argument(
        "wav_root", type=str,
        help="source root: <speaker>/<clip>.wav for audiomnist, <word>/<hash>_nohash_<n>.wav "
        "for speech_commands",
    )
    parser.add_argument("--out", type=str, required=True, help="output directory")
    parser.add_argument(
        "--dataset", type=str, default="audiomnist",
        help="dataset name for the sidecar; 'speech_commands' also selects the GSC v0.02 "
        "source walk (word folders, speaker = filename hash, _background_noise_ skipped)",
    )
    parser.add_argument("--split-seed", type=int, default=0, help="seed for the speaker split")
    parser.add_argument(
        "--holdout", type=int, default=None,
        help="number of speakers to hold out (AudioMNIST ~10, Speech Commands 200)",
    )
    parser.add_argument("--no-split", action="store_true", help="all speakers train (smoke only)")
    parser.add_argument(
        "--expect-speakers", type=int, default=None,
        help="assert the source has exactly this many unique speakers (GSC v0.02: 2618); "
        "defaults to a warning-only check for --dataset speech_commands",
    )
    parser.add_argument(
        "--reference-manifest", type=str, default=None,
        help="prep_manifest.json of an already-prepared dataset (e.g. AudioMNIST's): the "
        "derived mel_shape/mel_config/canvas/offset must match it exactly (SC spec §0)",
    )
    parser.add_argument(
        "--mel-cache", choices=("auto", "memory", "disk"), default="auto",
        help="where pass-1 physical mels live: memory (fast, ~2.7 GB at Speech Commands "
        "scale) or a temp memmap next to --out; auto switches at --mel-cache-max-gb",
    )
    parser.add_argument(
        "--mel-cache-max-gb", type=float, default=2.0,
        help="--mel-cache auto threshold in GB of physical mels",
    )
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

    try:
        clips, content = collect_clips(args.wav_root, args.dataset, log=log)
    except (OSError, ValueError) as e:
        sys.exit(f"[AUDIO] {e}")
    if not clips:
        sys.exit(f"[AUDIO] no .wav files under {args.wav_root}")
    n_clips, n_speakers = sum(len(v) for v in clips.values()), len(clips)
    # Only enumerate the speaker ids for small sets — Speech Commands has ~2618.
    roster = sorted(clips) if n_speakers <= 100 else f"{sorted(clips)[:5]} ... (+{n_speakers - 5} more)"
    log(f"dataset={args.dataset} found {n_clips} clips from {n_speakers} speakers: {roster}")

    expect_speakers = args.expect_speakers
    if expect_speakers is None and args.dataset == SPEECH_COMMANDS:
        if n_speakers != SC_NUM_SPEAKERS:
            log(
                f"WARNING: {n_speakers} unique speaker hashes, expected {SC_NUM_SPEAKERS} for "
                "GSC v0.02 — pass --expect-speakers to make this fatal"
            )
    elif expect_speakers is not None and n_speakers != expect_speakers:
        sys.exit(f"[AUDIO] found {n_speakers} speakers, --expect-speakers {expect_speakers}")

    train_speakers, held_out_speakers = split_speakers(clips, holdout, args.split_seed)
    if not train_speakers:
        sys.exit(
            f"[AUDIO] split left no training speakers: --holdout {holdout} >= "
            f"{len(clips)} speakers found — lower --holdout (seed does not affect this)"
        )
    overlap = sorted(set(train_speakers) & set(held_out_speakers))
    if overlap:  # id-disjoint contract (spec §3, §9) — must never fire
        sys.exit(f"[AUDIO] speaker split is not disjoint: {overlap}")
    log(
        f"split_seed={args.split_seed} "
        f"train_speakers={len(train_speakers)} held_out_speakers={len(held_out_speakers)}"
    )
    if max(len(train_speakers), len(held_out_speakers)) <= 100:
        log(f"train_speakers={train_speakers} held_out_speakers={held_out_speakers}")
    else:
        log(f"held_out_speakers (attack pool, first 10 of {len(held_out_speakers)})="
            f"{held_out_speakers[:10]} — full lists in speaker_split.json")

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

    out_dir = Path(args.out)
    canvas = tuple(args.canvas)
    offset = tuple(args.offset)
    CONTRACT_ECHO.update(
        mel_config=dict(MEL_CONFIG), canvas=list(canvas), offset=list(offset),
        pad_value_normalized=args.pad_value,
    )

    # Pass 1 — extract physical log-mels for TRAIN clips only (original + augmented
    # variants); derive T; m_hi stats come from the FULL augmented set (spec A.4).
    train_clips = [(spk, p) for spk in train_speakers for p in clips[spk]]
    variants_per_clip = 1 + (args.aug_variants if args.audio_aug else 0)
    store = None
    mel_shape = None
    clip_labels = {}   # npy stem -> {"speaker", "content"}; consumed by the eval (SC spec §5)
    for spk, path in tqdm(train_clips, desc="[AUDIO] extract mels", dynamic_ncols=True):
        wav = load_waveform(path, MEL_CONFIG["sample_rate"])
        base_stem = clip_stem(args.dataset, spk, path)
        variants = [(base_stem, pad_or_truncate_waveform(wav, num_samples))]
        for k in range(1, args.aug_variants + 1 if args.audio_aug else 1):
            rng = variant_rng(args.seed, f"{spk}/{path.stem}", k)
            params = draw_aug_params(rng, aug_types, args, MEL_CONFIG["sample_rate"])
            variants.append(
                (f"{base_stem}__aug{k}", augment_clip(wav, params, num_samples, MEL_CONFIG["sample_rate"]))
            )
        for stem, clip in variants:
            if stem in clip_labels:
                sys.exit(
                    f"[AUDIO] duplicate output stem {stem!r} (source {path}) — two source clips map "
                    "to the same .npy and one would overwrite the other; fix clip_stem() in "
                    "datasets_audio.py for this dataset"
                )
            if clip.shape[-1] != num_samples:
                sys.exit(f"[AUDIO] {stem}: {clip.shape[-1]} samples after aug (expected {num_samples})")
            if wav_dir is not None:
                # float32 (not PCM16): exact record of the extractor input; gain
                # variants may exceed [-1,1] and would clip in integer formats.
                torchaudio.save(str(wav_dir / f"{stem}.wav"), clip.unsqueeze(0), MEL_CONFIG["sample_rate"])
            mel = extract_mel(clip, MEL_CONFIG).to(torch.float32)
            if store is None:
                mel_shape = tuple(mel.shape)
                log(f"mel_shape=({mel_shape[0]}, {mel_shape[1]}) (T={mel_shape[1]}, derived empirically)")
                check_mel_shape(mel_shape, num_samples, args.reference_manifest)
                total = len(train_clips) * variants_per_clip
                backend = choose_mel_backend(args.mel_cache, total, mel_shape, args.mel_cache_max_gb)
                # Deliberately NOT a *.npy name: NpyMelDataset and the eval both
                # glob *.npy, so a crash-orphaned temp must not look like data.
                store = MelStore(
                    total, mel_shape, args.m_hi_percentile, backend, out_dir / ".mel_cache.tmp"
                )
            elif tuple(mel.shape) != mel_shape:
                sys.exit(f"[AUDIO] inconsistent mel shape {tuple(mel.shape)} for {stem} (expected {mel_shape})")
            store.append(stem, mel)
            clip_labels[stem] = {"speaker": spk, "content": content.get(path)}

    if store is None:
        sys.exit(f"[AUDIO] no clips to process for train speakers under {args.wav_root}")

    try:
        m_hi = store.percentile(args.m_hi_percentile)
        if m_hi <= M_LO:
            sys.exit(f"[AUDIO] degenerate affine: m_hi={m_hi} <= m_lo={M_LO}")
        log(f"affine m_lo={M_LO:.6f} m_hi={m_hi:.6f} (method=percentile_{args.m_hi_percentile}_train)")

        # Pass 2 — normalize, embed, save.
        out_dir.mkdir(parents=True, exist_ok=True)
        for stem, mel in tqdm(store, total=len(store), desc="[AUDIO] write npy", dynamic_ncols=True):
            norm = affine_forward(mel, M_LO, m_hi)
            sample = embed_canvas(norm, canvas, offset, args.pad_value)
            np.save(out_dir / f"{stem}.npy", sample.numpy().astype(np.float32))
    finally:
        store.close()

    # Content classes come from the WHOLE corpus, not just the train split, so the
    # class->index map is a property of the dataset and can't shift with --holdout.
    content_classes = sorted({c for c in content.values() if c is not None})
    sc_digits = [w for w in digit_words(args.dataset) if w in content_classes]
    num_clips = len(clip_labels)

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
        "num_clips": num_clips,
        "num_source_clips": len(train_clips),
        # Content-label summary (SC spec §5). The per-clip map itself lives in
        # clip_labels.json — 97k entries have no business inside a contract sidecar.
        "content_labels": {
            "type": content_label_type(args.dataset),
            "num_classes": len(content_classes),
            "classes": content_classes,
            "digit_classes": sc_digits,
            "file": "clip_labels.json",
        },
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
    # Per-clip speaker + content labels, keyed by .npy stem. eval/convergence_curve.py
    # reads this instead of re-deriving labels from filenames (SC spec §2, §5).
    labels_doc = {
        "dataset": args.dataset,
        "content_label_type": content_label_type(args.dataset),
        "content_classes": content_classes,
        "digit_classes": sc_digits,
        "clips": clip_labels,
    }

    with open(out_dir / "speaker_split.json", "w") as f:
        json.dump(split, f, indent=2)
    with open(out_dir / "prep_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(out_dir / "clip_labels.json", "w") as f:
        json.dump(labels_doc, f)   # no indent: one entry per clip, ~97k at SC scale

    log(
        f"content labels: type={content_label_type(args.dataset)} "
        f"num_classes={len(content_classes)}"
        + (f" digit_subset={sc_digits}" if sc_digits else "")
    )
    log(
        f"wrote {num_clips} .npy canvases + speaker_split.json + prep_manifest.json "
        f"+ clip_labels.json -> {out_dir}"
    )


if __name__ == "__main__":
    main()

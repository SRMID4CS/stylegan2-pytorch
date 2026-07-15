"""Offline waveform augmentation tests (AUGMENTATION_SPEC.md §A, §D.2).

CPU-only. The end-to-end test runs prepare_audio_data.py twice with --audio_aug on
the in-repo smoke data (data/audio_mnist_test) and asserts byte-identical outputs,
train-only application, and the .npy/manifest invariants.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import prepare_audio_data as prep
from audio.contract import MEL_CONFIG, clip_num_samples

SR = MEL_CONFIG["sample_rate"]
NUM_SAMPLES = clip_num_samples(MEL_CONFIG)
SMOKE_WAVS = REPO / "data" / "audio_mnist_test" / "data"


def aug_args(**overrides):
    base = dict(time_shift_ms=100.0, speed_range=0.10, pitch_semitones=2.0, gain_db=4.0)
    base.update(overrides)
    return argparse.Namespace(**base)


def sine_clip(num_samples=NUM_SAMPLES, freq=440.0):
    t = np.arange(num_samples, dtype=np.float32) / SR
    return torch.from_numpy(0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32))


def test_variant_rng_deterministic_and_distinct():
    a = prep.variant_rng(0, "01/0_01_0", 1).uniform(size=8)
    b = prep.variant_rng(0, "01/0_01_0", 1).uniform(size=8)
    assert np.array_equal(a, b)

    other_variant = prep.variant_rng(0, "01/0_01_0", 2).uniform(size=8)
    other_clip = prep.variant_rng(0, "01/0_01_1", 1).uniform(size=8)
    other_seed = prep.variant_rng(1, "01/0_01_0", 1).uniform(size=8)
    for other in (other_variant, other_clip, other_seed):
        assert not np.array_equal(a, other)


def test_draw_params_within_ranges():
    args = aug_args()
    types = ["time_shift", "speed", "pitch", "gain"]
    for k in range(200):
        p = prep.draw_aug_params(prep.variant_rng(0, "x", k), types, args, SR)
        assert 0.90 <= p["speed_rate"] <= 1.10
        assert -2.0 <= p["pitch_semitones"] <= 2.0
        assert abs(p["time_shift_samples"]) <= round(0.1 * SR)
        assert -4.0 <= p["gain_db"] <= 4.0

    subset = prep.draw_aug_params(prep.variant_rng(0, "x", 0), ["time_shift"], args, SR)
    assert set(subset) == {"time_shift_samples"}


def test_time_shift_zero_fill_non_circular():
    x = np.arange(1, 11, dtype=np.float32)

    right = prep.time_shift_waveform(x, 3)
    assert np.array_equal(right[:3], np.zeros(3, dtype=np.float32))
    assert np.array_equal(right[3:], x[:-3])

    left = prep.time_shift_waveform(x, -3)
    assert np.array_equal(left[-3:], np.zeros(3, dtype=np.float32))
    assert np.array_equal(left[:-3], x[3:])

    same = prep.time_shift_waveform(x, 0)
    assert np.array_equal(same, x)


@pytest.mark.parametrize("rate", [0.90, 1.10])
def test_speed_pitch_refit_to_clip_length(rate):
    wav = sine_clip()
    params = {"speed_rate": rate, "pitch_semitones": 1.5}
    out = prep.augment_clip(wav, params, NUM_SAMPLES, SR)
    assert out.shape == (NUM_SAMPLES,)
    assert out.dtype == torch.float32


def test_augment_clip_deterministic_given_params():
    wav = sine_clip()
    params = {"speed_rate": 0.95, "pitch_semitones": -1.0, "time_shift_samples": 500}
    a = prep.augment_clip(wav, params, NUM_SAMPLES, SR)
    b = prep.augment_clip(wav, params, NUM_SAMPLES, SR)
    assert torch.equal(a, b)


def test_gain_is_pure_scaling():
    wav = sine_clip()
    out = prep.augment_clip(wav, {"gain_db": 6.0}, NUM_SAMPLES, SR)
    expected = wav.numpy() * np.float32(10.0 ** (6.0 / 20.0))
    assert np.allclose(out.numpy(), expected, atol=0)


def run_prep(out_dir, extra=()):
    cmd = [
        sys.executable,
        str(REPO / "prepare_audio_data.py"),
        "--out", str(out_dir),
        "--holdout", "1",
        "--split-seed", "0",
        *extra,
        str(SMOKE_WAVS),
    ]
    res = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    assert res.returncode == 0, f"prep failed:\n{res.stdout}\n{res.stderr}"
    return out_dir


@pytest.mark.skipif(not SMOKE_WAVS.is_dir(), reason="smoke wav data not present")
def test_prep_end_to_end_augmented(tmp_path):
    aug_flags = ("--audio_aug", "--aug_variants", "2", "--seed", "0", "--save_wavs")
    run_a = run_prep(tmp_path / "a", aug_flags)
    run_b = run_prep(tmp_path / "b", aug_flags)

    files_a = sorted(p.name for p in run_a.glob("*.npy"))
    files_b = sorted(p.name for p in run_b.glob("*.npy"))
    assert files_a == files_b and files_a

    # Determinism: same --seed => byte-identical .npy (spec A.4).
    for name in files_a:
        assert (run_a / name).read_bytes() == (run_b / name).read_bytes(), name

    manifest = json.loads((run_a / "prep_manifest.json").read_text())
    split = json.loads((run_a / "speaker_split.json").read_text())

    # Train speakers only: no held-out speaker files, augmented or otherwise.
    for name in files_a:
        speaker = name.split("__")[0]
        assert speaker in split["train_speakers"]
        assert speaker not in split["held_out_speakers"]

    # Original + exactly 2 variants per source clip.
    originals = [n for n in files_a if "__aug" not in n]
    assert manifest["num_source_clips"] == len(originals)
    assert manifest["num_clips"] == len(files_a) == 3 * len(originals)
    for n in originals:
        stem = n[: -len(".npy")]
        for k in (1, 2):
            assert f"{stem}__aug{k}.npy" in files_a

    # Aug record in the manifest (spec A.4) — train.py copies it into the sidecar.
    aug = manifest["audio_aug"]
    assert aug["enabled"] is True
    assert aug["aug_types"] == ["time_shift", "speed", "pitch"]
    assert aug["aug_variants"] == 2
    assert aug["seed"] == 0

    # Output invariants unchanged: (1,128,128) float32 in [-1,1] (spec A.4).
    for name in files_a:
        arr = np.load(run_a / name)
        assert arr.shape == (1, 128, 128)
        assert arr.dtype == np.float32
        assert arr.min() >= -1.0 and arr.max() <= 1.0

    # m_hi is recomputed over the FULL augmented set: with time_shift/speed/pitch
    # only (no gain), the augmented percentile stays a valid physical value.
    assert manifest["affine"]["m_hi"] > manifest["affine"]["m_lo"]

    # --save_wavs: one 1.0 s wav per .npy (original + variants), deterministic too.
    wav_names = sorted(p.name for p in (run_a / "wavs").glob("*.wav"))
    assert wav_names == [n.replace(".npy", ".wav") for n in files_a]
    for name in wav_names:
        assert (run_a / "wavs" / name).read_bytes() == (run_b / "wavs" / name).read_bytes(), name


@pytest.mark.skipif(not SMOKE_WAVS.is_dir(), reason="smoke wav data not present")
def test_prep_no_aug_path_unchanged(tmp_path):
    out = run_prep(tmp_path / "clean")
    files = sorted(p.name for p in out.glob("*.npy"))
    assert files and all("__aug" not in n for n in files)

    manifest = json.loads((out / "prep_manifest.json").read_text())
    assert manifest["audio_aug"]["enabled"] is False
    assert manifest["audio_aug"]["aug_variants"] == 0
    assert manifest["num_clips"] == manifest["num_source_clips"] == len(files)

    # --save_wavs defaults OFF: no wavs/ directory.
    assert not (out / "wavs").exists()

"""Mel contract shared byte-identically between stylegan2-audio and dlg-sonic.

Every value that crosses the repo boundary (mel config, global affine, canvas
geometry) either lives here or is derived here and persisted to the prep manifest /
checkpoint sidecar. Nothing downstream may hardcode these.

Extraction goes through the vendored BigVGAN `get_mel_spectrogram` ONLY
(GAN_TRAINING_SPEC.md §2) — never torchaudio/librosa.
"""

import math

import torch

from .env import AttrDict
from .meldataset import get_mel_spectrogram

# Pinned upstream state of the vendored meldataset.py/env.py (NVIDIA/BigVGAN @ main).
BIGVGAN_COMMIT = "7d2b454564a6c7d014227f635b7423881f14bdac"

EXTRACTOR_ID = "bigvgan.meldataset.get_mel_spectrogram"
VOCODER_ID = "nvidia/bigvgan_v2_22khz_80band_fmax8k_256x"
RESAMPLER_ID = "torchaudio.functional.resample"

# Canonical mel config — GAN_TRAINING_SPEC.md §2. T is NOT here: derive it
# empirically via extract_mel() and persist it in the manifest/sidecar.
MEL_CONFIG = {
    "sample_rate": 22050,
    "n_fft": 1024,
    "win_length": 1024,
    "hop_length": 256,
    "n_mels": 80,
    "fmin": 0,
    "fmax": 8000,
    "clip_seconds": 1.0,
    "extractor": EXTRACTOR_ID,
    "bigvgan_commit": BIGVGAN_COMMIT,
}

# Physical log-mel floor, fixed by the extractor's clamp
# (dynamic_range_compression_torch clip_val=1e-5).
M_LO = float(math.log(1e-5))


def clip_num_samples(config=MEL_CONFIG):
    """Number of waveform samples in one training clip."""
    return int(round(config["sample_rate"] * config["clip_seconds"]))


def pad_or_truncate_waveform(wav, num_samples):
    """Force a 1-D waveform to exactly `num_samples` (spec §3.2).

    Short clips get zeros APPENDED (silence passes through the extractor and lands
    on the mel log-floor); over-length clips are center-cropped. Never pad in mel
    space.
    """
    n = wav.shape[-1]
    if n == num_samples:
        return wav
    if n < num_samples:
        pad = torch.zeros(num_samples - n, dtype=wav.dtype, device=wav.device)
        return torch.cat([wav, pad], dim=-1)
    start = (n - num_samples) // 2
    return wav[..., start : start + num_samples]


def extract_mel(wav, config=MEL_CONFIG):
    """Waveform (1-D, [-1,1], config sample rate) -> physical log-mel (n_mels, T).

    Thin adapter over the vendored BigVGAN extractor; the hparams mapping below is
    the only place the contract config meets BigVGAN's argument names.
    """
    h = AttrDict(
        n_fft=config["n_fft"],
        num_mels=config["n_mels"],
        sampling_rate=config["sample_rate"],
        hop_size=config["hop_length"],
        win_size=config["win_length"],
        fmin=config["fmin"],
        fmax=config["fmax"],
    )
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    return get_mel_spectrogram(wav, h).squeeze(0)


def affine_forward(mel, m_lo, m_hi):
    """Physical log-mel -> [-1,1]; ONE global map for every sample (spec §4)."""
    mel_c = torch.clamp(mel, min=m_lo, max=m_hi)
    return 2.0 * (mel_c - m_lo) / (m_hi - m_lo) - 1.0


def affine_inverse(normalized, m_lo, m_hi):
    """[-1,1] -> physical log-mel (attack side runs this before vocoding)."""
    return (normalized + 1.0) / 2.0 * (m_hi - m_lo) + m_lo


def embed_canvas(mel_norm, canvas, offset, fill):
    """Place a normalized (n_mels, T) mel into a (1, H, W) canvas at `offset`
    (row, col), filling the rest with `fill` (the normalized floor, -1)."""
    h, w = mel_norm.shape
    ch, cw = canvas
    r, c = offset
    if r + h > ch or c + w > cw:
        raise ValueError(f"mel {mel_norm.shape} at offset {offset} exceeds canvas {canvas}")
    out = torch.full((1, ch, cw), float(fill), dtype=mel_norm.dtype, device=mel_norm.device)
    out[0, r : r + h, c : c + w] = mel_norm
    return out


def crop_canvas(canvas_tensor, offset, mel_shape):
    """Inverse of embed_canvas: (..., H, W) canvas -> (..., n_mels, T) real region.

    `offset` and `mel_shape` must come from the manifest/sidecar, never constants.
    """
    r, c = offset
    h, w = mel_shape
    return canvas_tensor[..., r : r + h, c : c + w]

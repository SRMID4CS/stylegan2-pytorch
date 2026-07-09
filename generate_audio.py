"""Sample the audio GAN and/or validate the mel chain (GAN_TRAINING_SPEC.md §7).

Sample mode:
    python generate_audio.py --ckpt checkpoint/001500.pt --n 8 --out samples_audio/
    z -> g_ema -> canvas -> crop + inverse affine (ALL values from the checkpoint's
    sidecar JSON) -> vocoder(s) -> .wav @ contract sample rate. Also dumps the mel
    canvases as a PNG grid (viewing only — never a data format).

Round-trip mode (spec §7.1–7.2 — run BEFORE training):
    python generate_audio.py --roundtrip <clip.wav> \
        --manifest data/npy_smoke/prep_manifest.json --out samples_audio/
    Writes <stem>_roundtrip_<SUF>.wav (wav -> pad -> mel -> affine -> canvas ->
    crop -> inverse -> vocode) and <stem>_gtvocode_<SUF>.wav (mel -> vocoder
    directly; the reconstruction upper bound).

Vocoders (--vocoder, default both): BigVGAN (suffix BVG; vendored inference code +
HF weights) and Griffin-Lim (suffix GL; librosa inverse of the same mel basis —
lower quality, but needs no vocoder download).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torchvision import utils

from audio.contract import (
    VOCODER_ID,
    affine_forward,
    affine_inverse,
    clip_num_samples,
    crop_canvas,
    embed_canvas,
    extract_mel,
    pad_or_truncate_waveform,
)
from prepare_audio_data import load_waveform


def log(msg):
    print(f"[AUDIO] {msg}")


def load_bigvgan(device):
    """Vendored BigVGAN inference code (audio/bigvgan, pinned) + HF weights."""
    vendor_dir = str(Path(__file__).resolve().parent / "audio" / "bigvgan")
    if vendor_dir not in sys.path:
        sys.path.insert(0, vendor_dir)
    from bigvgan import BigVGAN

    log(f"loading vocoder {VOCODER_ID} (use_cuda_kernel=False)")
    model = BigVGAN.from_pretrained(VOCODER_ID, use_cuda_kernel=False, proxies=None, resume_download=None)
    model.remove_weight_norm()
    return model.eval().to(device)


def griffin_lim(mel_phys, mel_config, n_iter):
    """Physical log-mel (n_mels, T) numpy -> waveform via librosa Griffin-Lim.

    Inverts the same mel basis BigVGAN's extractor uses (librosa mel filterbank,
    slaney norm, magnitude spectrogram) — an approximation for ear checks only.
    """
    import librosa

    mel_mag = np.exp(mel_phys)  # undo dynamic_range_compression (log, C=1)
    stft_mag = librosa.feature.inverse.mel_to_stft(
        mel_mag,
        sr=mel_config["sample_rate"],
        n_fft=mel_config["n_fft"],
        power=1.0,
        fmin=mel_config["fmin"],
        fmax=mel_config["fmax"],
    )
    return librosa.griffinlim(
        stft_mag,
        n_iter=n_iter,
        hop_length=mel_config["hop_length"],
        win_length=mel_config["win_length"],
        n_fft=mel_config["n_fft"],
    )


class Vocoders:
    """Vocode a physical log-mel with the selected backend(s); returns {suffix: wav}."""

    def __init__(self, which, device, gl_iters, mel_config):
        self.device = device
        self.gl_iters = gl_iters
        self.mel_config = mel_config
        self.bigvgan = load_bigvgan(device) if which in ("bigvgan", "both") else None
        self.use_gl = which in ("gl", "both")

    def __call__(self, mel_phys):
        mel_phys = mel_phys.detach().cpu().float()
        out = {}
        if self.bigvgan is not None:
            with torch.inference_mode():
                wav = self.bigvgan(mel_phys.unsqueeze(0).to(self.device))
            out["BVG"] = wav.squeeze().cpu().numpy()
        if self.use_gl:
            out["GL"] = griffin_lim(mel_phys.numpy(), self.mel_config, self.gl_iters)
        return out


def write_wavs(out_dir, stem, wavs, sample_rate):
    for suffix, wav in wavs.items():
        path = out_dir / f"{stem}_{suffix}.wav"
        sf.write(path, wav, sample_rate)
        log(f"wrote {path} ({len(wav) / sample_rate:.2f} s)")


def run_sample(args, device):
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sidecar_path = Path(args.sidecar) if args.sidecar else Path(args.ckpt).with_suffix(".json")
    if not sidecar_path.is_file():
        sys.exit(f"[AUDIO] sidecar not found: {sidecar_path} (train with --dataset npy writes it)")
    with open(sidecar_path) as f:
        sidecar = json.load(f)
    log(f"sidecar: {sidecar_path}")

    mel_config = sidecar["mel_config"]
    mel_shape = tuple(sidecar["mel_shape"])
    offset = tuple(sidecar["offset"])
    affine = sidecar["affine"]
    latent = sidecar["latent"]
    log(
        f"contract: mel_shape={mel_shape} canvas={tuple(sidecar['canvas'])} offset={offset} "
        f"affine=(m_lo={affine['m_lo']:.4f}, m_hi={affine['m_hi']:.4f}) "
        f"num_ws={latent['num_ws']} w_dim={latent['w_dim']}"
    )

    # Rebuild g_ema from the args stored in the checkpoint — nothing hardcoded.
    train_args = ckpt["args"]
    from model import Generator

    g_ema = Generator(
        train_args.size,
        train_args.latent,
        train_args.n_mlp,
        channel_multiplier=train_args.channel_multiplier,
        img_channels=getattr(train_args, "img_channels", 3),
    ).to(device)
    g_ema.load_state_dict(ckpt["g_ema"])
    g_ema.eval()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        log(f"latent seed={args.seed}")

    truncation_latent = None
    if args.truncation < 1:
        # W+ warm-start vector from the sidecar doubles as the truncation center.
        truncation_latent = torch.tensor(sidecar["w_avg"], device=device).unsqueeze(0)

    with torch.no_grad():
        z = torch.randn(args.n, train_args.latent, device=device)
        canvases, _ = g_ema([z], truncation=args.truncation, truncation_latent=truncation_latent)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_path = out_dir / "samples_mel.png"
    utils.save_image(
        canvases, grid_path, nrow=int(args.n ** 0.5) or 1, normalize=True, value_range=(-1, 1)
    )
    log(f"mel-canvas grid (viewing only): {grid_path}")

    vocoders = Vocoders(args.vocoder, device, args.gl_iters, mel_config)
    canvases = canvases.clamp(-1, 1)  # G output can slightly overshoot the affine range
    for i in range(args.n):
        norm_mel = crop_canvas(canvases[i, 0], offset, mel_shape)
        mel_phys = affine_inverse(norm_mel, affine["m_lo"], affine["m_hi"])
        write_wavs(out_dir, f"sample_{i:02d}", vocoders(mel_phys), mel_config["sample_rate"])


def run_roundtrip(args, device):
    if not args.manifest:
        sys.exit("[AUDIO] --roundtrip needs --manifest <prep dir>/prep_manifest.json")
    with open(args.manifest) as f:
        manifest = json.load(f)
    mel_config = manifest["mel_config"]
    affine = manifest["affine"]
    log(
        f"roundtrip {args.roundtrip} via manifest {args.manifest} "
        f"(m_lo={affine['m_lo']:.4f}, m_hi={affine['m_hi']:.4f})"
    )

    wav = load_waveform(Path(args.roundtrip), mel_config["sample_rate"])
    wav = pad_or_truncate_waveform(wav, clip_num_samples(mel_config))
    mel_phys = extract_mel(wav, mel_config)
    if list(mel_phys.shape) != list(manifest["mel_shape"]):
        sys.exit(f"[AUDIO] mel shape {tuple(mel_phys.shape)} != manifest {manifest['mel_shape']}")

    # Full data-prep chain there and back: affine -> canvas -> crop -> inverse.
    norm = affine_forward(mel_phys, affine["m_lo"], affine["m_hi"])
    canvas = embed_canvas(
        norm, tuple(manifest["canvas"]), tuple(manifest["offset"]), manifest["pad_value_normalized"]
    )
    norm_back = crop_canvas(canvas[0], tuple(manifest["offset"]), tuple(manifest["mel_shape"]))
    mel_back = affine_inverse(norm_back, affine["m_lo"], affine["m_hi"])

    clamped = mel_phys.clamp(affine["m_lo"], affine["m_hi"])
    err = (mel_back - clamped).abs().max().item()
    log(f"canvas/affine round-trip max |err| vs clamped GT mel: {err:.3e}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.roundtrip).stem
    vocoders = Vocoders(args.vocoder, device, args.gl_iters, mel_config)
    write_wavs(out_dir, f"{stem}_gtvocode", vocoders(mel_phys), mel_config["sample_rate"])
    write_wavs(out_dir, f"{stem}_roundtrip", vocoders(mel_back), mel_config["sample_rate"])


def main():
    parser = argparse.ArgumentParser(description="Audio GAN sampling / mel-chain validation")
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint .pt (sample mode)")
    parser.add_argument(
        "--sidecar", type=str, default=None,
        help="sidecar JSON path (default: <ckpt>.json next to the checkpoint)",
    )
    parser.add_argument("--n", type=int, default=8, help="number of samples to generate")
    parser.add_argument("--out", type=str, default="samples_audio", help="output directory")
    parser.add_argument("--truncation", type=float, default=1.0, help="truncation (center = sidecar w_avg)")
    parser.add_argument("--seed", type=int, default=None, help="latent RNG seed")
    parser.add_argument(
        "--vocoder", type=str, choices=["bigvgan", "gl", "both"], default="both",
        help="waveform backend(s); output files are suffixed _BVG / _GL",
    )
    parser.add_argument("--gl-iters", type=int, default=60, help="Griffin-Lim iterations")
    parser.add_argument("--roundtrip", type=str, default=None, help="wav file for chain validation mode")
    parser.add_argument("--manifest", type=str, default=None, help="prep_manifest.json (roundtrip mode)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device} vocoder={args.vocoder}")

    if args.roundtrip:
        run_roundtrip(args, device)
    elif args.ckpt:
        run_sample(args, device)
    else:
        parser.error("pass --ckpt (sample mode) or --roundtrip <wav> (validation mode)")


if __name__ == "__main__":
    main()

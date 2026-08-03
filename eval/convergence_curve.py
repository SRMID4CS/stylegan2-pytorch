"""
Route B convergence curve for stylegan2-audio checkpoints.

Mel-native domain-classifier Frechet distance + class-coverage entropy.
No vocoding, no BigVGAN, no waveform models -> stays entirely in mel space, so
it runs in seconds/checkpoint. Primary target: AWS g6.xlarge (L4, sm_89); also
runs on the Blackwell laptop for quick local checks (pass
TORCH_CUDA_ARCH_LIST=12.0 in the run command on that box -- see USAGE.md).

Two label modes (--label-mode, default both):
  * content  -- classify the digit/keyword; the digit CNN saturates (~99.8% acc)
                 and is insensitive to speaker-manifold collapse.
  * speaker  -- classify speaker ID over the TRAIN speakers; the collapse
                 tripwire that stress-tests the N1 speaker-diversity claim.
Each mode has its own classifier, reference (mu_r, Sigma_r) and real class-entropy.

One-time cost:  train a small mel classifier on the real TRAIN-speaker mels,
                cache reference (mu_r, Sigma_r) and the real class-entropy.
                Cached per (dataset, mode) under convergence_cache/<dataset>/<mode>/
                so content/speaker and AudioMNIST/Speech Commands never collide.
Per checkpoint: sample N mels from g_ema (frozen seed) ONCE -> embed with each
                mode's classifier -> Frechet distance to that mode's cached
                reference + coverage entropy.

Run (from repo root):
    python eval/convergence_curve.py \
        --npy-dir data/npy_audiomnist --ckpt-glob "checkpoint/*.pt" \
        --dataset audiomnist --num-classes 10 --label-mode both
    python eval/convergence_curve.py --retrain ...    # force classifier retrain

Outputs:
    convergence_cache/<dataset>/<mode>/classifier.pt, reference_stats.npz  (cached)
    <--out>_<mode>.csv   (iter, fd, entropy)  -- one per mode, ALL checkpoints
    <--out>.png          (one panel per mode; iter-0 omitted by default)

Frozen contract (GAN_TRAINING_SPEC.md-style): --seed and --n-samples must stay
identical across a full run -- the absolute FD is only comparable across
checkpoints when reference, seed, and N are frozen. --seed also freezes the
per-layer noise (see sample_mels), not just z, so the curve reflects G, not
sampling variance.
"""
import os

# AWS g6.xlarge (L4, sm_89) is the primary target -- do NOT bake in the
# Blackwell laptop's "12.0" here. setdefault() means an activate-hook or
# run-command export always wins; only the un-overridden default changes.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg

# Repo root on sys.path: this file lives in eval/, so `python eval/convergence_curve.py`
# only puts eval/ on sys.path by default (unit_test/ uses the same fix).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from audio.contract import crop_canvas  # noqa: E402  (needs REPO_ROOT on sys.path)

EMB_DIM = 128
DEFAULT_BATCH = 64
LABEL_MODES = ("content", "speaker")


# ---------- filename -> label ----------
# prepare_audio_data.py names every .npy `<speaker>__<orig_wav_stem>[__aug{k}]`
# (verified against prepare_audio_data.py 2026-07-21: original clips are saved as
# f"{spk}__{path.stem}", augmented variants as f"{spk}__{path.stem}__aug{k}" --
# so the speaker is ALWAYS the first "__"-token and survives augmentation).
def speaker_of(stem: str) -> str:
    """Extract the speaker id from a prepared .npy stem (first '__'-token)."""
    parts = stem.split("__")
    if len(parts) < 2 or not parts[0]:
        raise ValueError(
            f"unrecognized .npy stem {stem!r}: expected '<speaker>__<orig_stem>[__aug{{k}}]' "
            "(prepare_audio_data.py naming)"
        )
    return parts[0]


# For AudioMNIST, orig_wav_stem is `<digit>_<speaker>_<rec>` (FILENAME_RE in
# prepare_audio_data.py), so the digit is the first "_"-token of the second
# "__"-separated part. There is no labels.json emitted by prepare_audio_data.py
# today; drop one ({npy_stem: int}) next to --npy-dir to override/extend this.
def content_label(stem: str, dataset: str) -> int:
    parts = stem.split("__")
    if len(parts) < 2:
        raise ValueError(
            f"unrecognized .npy stem {stem!r}: expected '<speaker>__<orig_stem>[__aug{{k}}]' "
            "(prepare_audio_data.py naming) or a labels.json override"
        )
    orig_stem = parts[1]
    if dataset == "audiomnist":
        return int(orig_stem.split("_")[0])
    raise NotImplementedError(
        f"no content-label rule for --dataset {dataset!r}. prepare_audio_data.py only implements "
        "AudioMNIST-style '<digit>_<speaker>_<rec>' source filenames today; add a rule here "
        "for your dataset's naming, or drop a labels.json ({npy_stem: int}) next to --npy-dir."
    )


def load_train_speakers(npy_dir):
    """The id-disjoint contract's authoritative train-speaker set (spec §3, §9)."""
    split_path = Path(npy_dir) / "speaker_split.json"
    if not split_path.is_file():
        sys.exit(
            f"[eval] speaker mode needs {split_path} (prepare_audio_data.py writes it) "
            "to define the train-speaker label set"
        )
    return json.loads(split_path.read_text())["train_speakers"]


def build_label_fn(mode, npy_dir, dataset, num_classes):
    """Return (label_fn(stem)->int, num_classes) for the requested mode.

    content: digit/keyword label (labels.json override honored), K = --num-classes.
    speaker: train-speaker id remapped to a CONTIGUOUS 0..K-1 head (K = number of
             train speakers from speaker_split.json). Non-train speakers can't
             appear -- prepare_audio_data.py only writes train clips -- but we
             cross-check the observed filenames and fail loudly if one does.
    """
    if mode == "content":
        lj = Path(npy_dir) / "labels.json"
        labels_map = json.loads(lj.read_text()) if lj.exists() else None

        def fn(stem):
            return labels_map[stem] if labels_map else content_label(stem, dataset)

        return fn, num_classes

    if mode == "speaker":
        train_speakers = sorted(load_train_speakers(npy_dir))
        speaker_to_idx = {s: i for i, s in enumerate(train_speakers)}
        K = len(train_speakers)

        observed = {speaker_of(Path(f).stem) for f in glob.glob(os.path.join(npy_dir, "*.npy"))}
        unknown = sorted(observed - set(train_speakers))
        if unknown:
            sys.exit(
                f"[eval] speaker mode: {npy_dir} contains clips from speakers {unknown} that are "
                "NOT in speaker_split.json train_speakers -- id-disjoint contract breach (spec §3, §9)"
            )
        missing = sorted(set(train_speakers) - observed)
        if missing:
            print(f"[eval] WARNING: train_speakers {missing} have no .npy clips in {npy_dir} "
                  "(empty classes in the speaker head)")
        # Contract assertion: K derived from the manifest, never hardcoded.
        assert K == len(train_speakers), "speaker K must equal len(train_speakers)"
        print(f"[eval] speaker mode: K={K} train speakers (contiguous remap 0..{K-1}); chance acc ~= {1.0/K:.3f}")

        def fn(stem):
            return speaker_to_idx[speaker_of(stem)]

        return fn, K

    raise ValueError(f"unknown label mode {mode!r}")


# ---------- data ----------
class MelNpy(torch.utils.data.Dataset):
    def __init__(self, npy_dir, label_fn, crop=None):
        self.files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
        assert self.files, f"no .npy under {npy_dir}"
        self.label_fn = label_fn
        self.crop = crop  # (offset, mel_shape) tuple, or None for full-canvas

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        p = self.files[i]
        x = np.load(p).astype(np.float32)          # (1,128,128) in [-1,1]
        x = torch.from_numpy(x)
        if self.crop is not None:
            offset, mel_shape = self.crop
            x = crop_canvas(x, offset, mel_shape)
        y = self.label_fn(Path(p).stem)
        return x, int(y)


# ---------- classifier / embedder ----------
class MelClassifier(nn.Module):
    def __init__(self, num_classes, emb_dim=EMB_DIM, in_ch=1):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1),
                                 nn.BatchNorm2d(o), nn.LeakyReLU(0.2, inplace=True))
        self.features = nn.Sequential(
            blk(in_ch, 32),
            blk(32, 64),
            blk(64, 128),
            blk(128, 128),
            nn.AdaptiveAvgPool2d(1),   # -> works for full 128x128 canvas or a --crop-to-sidecar region alike
        )
        self.embed = nn.Linear(128, emb_dim)
        self.head  = nn.Linear(emb_dim, num_classes)

    def forward(self, x, return_embedding=False):
        e = self.embed(self.features(x).flatten(1))
        if return_embedding:
            return e
        return self.head(F.relu(e)), e


def train_classifier(npy_dir, label_fn, num_classes, crop, cache_dir, device, mode, epochs=20, lr=2e-4):
    ds = MelNpy(npy_dir, label_fn, crop)
    # drop_last=True with a fixed batch_size=128 silently divides-by-zero the
    # running accuracy on small/smoke datasets (len(ds) < 128) -- no GAN-style
    # minibatch-stddev constraint here, so just cap the batch size instead.
    dl = torch.utils.data.DataLoader(ds, batch_size=min(128, len(ds)), shuffle=True,
                                     num_workers=2, drop_last=False)
    net = MelClassifier(num_classes).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    acc = 0.0
    for ep in range(epochs):
        tot = correct = 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits, _ = net(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            correct += (logits.argmax(1) == y).sum().item(); tot += y.numel()
        acc = correct / tot
        print(f"[clf:{mode}] epoch {ep+1}/{epochs}  acc={acc:.3f}")
    # Speaker ID over ~48 classes from 1 s clips is genuinely hard; make the final
    # train acc prominent so a near-chance (uninformative) embedding is obvious.
    print(f"[clf:{mode}] DONE  train_acc={acc:.3f}  (K={num_classes}, chance={1.0/num_classes:.3f})")
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), cache_dir / "classifier.pt")
    return net


def load_classifier(num_classes, cache_dir, device):
    net = MelClassifier(num_classes).to(device)
    net.load_state_dict(torch.load(cache_dir / "classifier.pt", map_location=device))
    net.eval()
    return net


# ---------- embeddings / stats ----------
@torch.no_grad()
def embed_and_predict(net, mels, batch, device):
    """mels: (N,1,H,W) tensor -> (embeddings [N,EMB_DIM], pred_classes [N])."""
    embs, preds = [], []
    for i in range(0, len(mels), batch):
        x = mels[i:i + batch].to(device)
        logits, e = net(x)
        embs.append(e.cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(embs), np.concatenate(preds)


def gaussian_stats(emb):
    return emb.mean(0), np.cov(emb, rowvar=False)


def frechet_distance(mu1, cov1, mu2, cov2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(cov1 @ cov2, disp=False)
    if not np.isfinite(covmean).all():        # numerical fallback
        off = np.eye(cov1.shape[0]) * eps
        covmean = linalg.sqrtm((cov1 + off) @ (cov2 + off))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov1 + cov2 - 2 * covmean))


def norm_entropy(pred_classes, num_classes):
    # A single class has no diversity to measure and log(1)==0 would make the
    # normalizer 0/0 -> nan (hits the degenerate single-train-speaker case).
    if num_classes <= 1:
        return 0.0
    counts = np.bincount(pred_classes, minlength=num_classes).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(num_classes))


def build_reference(net, npy_dir, label_fn, num_classes, crop, cache_dir, device, batch, mode):
    """Cache (mu_r, Sigma_r) and real class-entropy over ALL real TRAIN-speaker
    mels under npy_dir. Frozen once cached -- delete the cache file to rebuild."""
    cache = cache_dir / "reference_stats.npz"
    if cache.exists():
        d = np.load(cache)
        return d["mu"], d["cov"], float(d["entropy"])
    ds = MelNpy(npy_dir, label_fn, crop)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=2)
    embs, preds = [], []
    net.eval()
    with torch.no_grad():
        for x, _ in dl:
            logits, e = net(x.to(device))
            embs.append(e.cpu().numpy()); preds.append(logits.argmax(1).cpu().numpy())
    embs, preds = np.concatenate(embs), np.concatenate(preds)
    mu, cov = gaussian_stats(embs)
    ent = norm_entropy(preds, num_classes)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache, mu=mu, cov=cov, entropy=ent)
    print(f"[ref:{mode}] N={len(embs)}  real class-entropy={ent:.3f}")
    return mu, cov, ent


# ---------- generator sampling ----------
def load_generator(ckpt_path, device):
    """Rebuild g_ema straight from the checkpoint's own saved `args` (train.py
    always stores them, see write of `"args": args` in train.py) -- avoids
    hardcoding size/style_dim/n_mlp/img_channels and can't drift from what the
    checkpoint was actually trained with (mirrors generate_audio.py's approach).
    """
    from model import Generator      # repo's patched single-channel module; not reimplemented

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt["args"]
    G = Generator(
        train_args.size, train_args.latent, train_args.n_mlp,
        channel_multiplier=train_args.channel_multiplier,
        img_channels=getattr(train_args, "img_channels", 3),
    ).to(device)
    G.load_state_dict(ckpt["g_ema"])   # EMA weights = what we sample for eval; strict=True by
                                        # default -- raises loudly on any missing/unexpected key
    G.eval()
    return G


@torch.no_grad()
def sample_mels(G, n, seed, batch, device):
    """Frozen-seed sampling so every checkpoint sees the same z's AND the same
    per-layer noise. NoiseInjection (model.py) draws its noise from the ambient
    default RNG when `noise=None` (randomize_noise=True), not from a `generator=`
    kwarg -- so freezing only z via a private torch.Generator would still leave
    per-layer noise uncontrolled and re-run-to-re-run different. Seeding the
    global RNG once per checkpoint freezes both, which is what "curve reflects
    G, not sampling noise" requires.
    """
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    out = []
    for i in range(0, n, batch):
        b = min(batch, n - i)
        z = torch.randn(b, G.style_dim, device=device)
        img, _ = G([z])                 # rosinality: (image, latent) if return_latents else (image, None)
        out.append(img.cpu())
    return torch.cat(out)               # (n,1,H,W), roughly [-1,1] (no tanh)


def parse_iter(path):
    digits = "".join(c for c in Path(path).stem if c.isdigit())
    return int(digits) if digits else -1


def load_sidecar_crop(ckpt_path):
    sidecar_path = Path(ckpt_path).with_suffix(".json")
    if not sidecar_path.is_file():
        sys.exit(
            f"[eval] --crop-to-sidecar needs {sidecar_path} (train.py --dataset npy writes "
            "one sidecar per checkpoint automatically -- was this checkpoint trained on npy data?)"
        )
    sidecar = json.loads(sidecar_path.read_text())
    return tuple(sidecar["offset"]), tuple(sidecar["mel_shape"])


def warn_if_speaker_mismatch(npy_dir, ckpt_path):
    """Best-effort tripwire for the id-disjoint contract (spec §3, §9): if the
    checkpoint's own sidecar disagrees with --npy-dir about which speakers were
    trained on, the reference distribution may not actually be G's training
    distribution. Warns rather than aborts -- sidecars aren't guaranteed present
    for every caller (e.g. hand-copied checkpoints)."""
    sidecar_path = Path(ckpt_path).with_suffix(".json")
    split_path = Path(npy_dir) / "speaker_split.json"
    if not (sidecar_path.is_file() and split_path.is_file()):
        return
    sidecar_speakers = json.loads(sidecar_path.read_text()).get("train_speakers")
    ref_speakers = json.loads(split_path.read_text()).get("train_speakers")
    if sidecar_speakers is not None and ref_speakers is not None and sorted(sidecar_speakers) != sorted(ref_speakers):
        print(
            f"[eval] WARNING: {sidecar_path.name} train_speakers != {split_path} train_speakers -- "
            "reference set may not be this checkpoint's actual training distribution"
        )


def str2bool(v):
    if isinstance(v, bool):
        return v
    if str(v).lower() in ("1", "true", "yes", "y", "t"):
        return True
    if str(v).lower() in ("0", "false", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {v!r}")


# ---------- outputs ----------
def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iter", "fd", "entropy"]); w.writerows(rows)
    print(f"wrote {path}")


def write_plot(out_png, mode_rows, mode_ent_r, omit_iter0):
    """One twin-axis panel per mode (FD + coverage entropy vs iteration), stacked
    vertically. iter-0 is dropped from the PLOT only (default) so the untrained
    checkpoint's blown-out FD doesn't crush the y-axis -- the CSVs keep it."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot skipped] {e}")
        return

    modes = list(mode_rows.keys())
    fig, axes = plt.subplots(len(modes), 1, figsize=(8, 4.5 * len(modes)), squeeze=False)
    for ax1, mode in zip(axes[:, 0], modes):
        rows = mode_rows[mode]
        if omit_iter0:
            rows = [r for r in rows if r[0] != 0]
        if not rows:
            ax1.set_title(f"{mode}: no rows to plot (all omitted)")
            continue
        it, fd, ent = zip(*rows)
        ax1.plot(it, fd, "o-", color="C0", label="domain-FD")
        ax1.set_xlabel("iteration"); ax1.set_ylabel("Frechet distance", color="C0")
        ax2 = ax1.twinx()
        ax2.plot(it, ent, "s--", color="C1", label="coverage entropy")
        ax2.axhline(mode_ent_r[mode], color="C1", lw=0.8, alpha=0.5)
        ax2.set_ylabel("coverage entropy (norm.)", color="C1"); ax2.set_ylim(0, 1.05)
        ax1.set_title(f"{mode} curve" + ("  (iter-0 omitted)" if omit_iter0 else ""))
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"wrote {out_png}")


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npy-dir", default="data/npy_audiomnist",
                    help="real TRAIN-speaker mel canvases (prepare_audio_data.py output; "
                         "id-disjoint by construction -- held-out speakers are never written here)")
    ap.add_argument("--ckpt-glob", default="checkpoint/*.pt", help="glob of checkpoints to score")
    ap.add_argument("--dataset", default="audiomnist",
                    help="cache namespace (convergence_cache/<dataset>/<mode>/) and content label rule")
    ap.add_argument("--label-mode", choices=["content", "speaker", "both"], default="both",
                    help="content = digit/keyword curve; speaker = speaker-id collapse tripwire; both = both")
    ap.add_argument("--num-classes", type=int, default=10,
                    help="content mode: AudioMNIST=10 digits / Speech Commands=keyword count. "
                         "Ignored in speaker mode (K is derived from speaker_split.json)")
    ap.add_argument("--clf-epochs", type=int, default=20,
                    help="classifier training epochs (raise if speaker acc underfits toward chance)")
    ap.add_argument("--n-samples", type=int, default=2000,
                    help="FROZEN samples/checkpoint -- keep identical across a full run")
    ap.add_argument("--seed", type=int, default=0,
                    help="FROZEN sampling seed (z + per-layer noise) -- keep identical across a full run")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--out", default="convergence_curve",
                    help="output basename -> <out>_<mode>.csv per mode, <out>.png (one panel per mode)")
    ap.add_argument("--retrain", action="store_true", help="force classifier retrain even if cached")
    ap.add_argument("--plot-omit-iter0", type=str2bool, default=True,
                    help="drop iter==0 from the PNG only (default True); the CSVs always keep it")
    ap.add_argument("--crop-to-sidecar", action="store_true",
                    help="[off by default] crop real+generated mels to the real 80xT region "
                         "(per npy_dir's prep_manifest.json / each checkpoint's own sidecar) "
                         "before embedding, instead of the full padded canvas")
    args = ap.parse_args()

    modes = list(LABEL_MODES) if args.label_mode == "both" else [args.label_mode]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] device={device}  modes={modes}  TORCH_CUDA_ARCH_LIST={os.environ.get('TORCH_CUDA_ARCH_LIST')}")

    real_crop = None
    if args.crop_to_sidecar:
        manifest_path = Path(args.npy_dir) / "prep_manifest.json"
        if not manifest_path.is_file():
            sys.exit(f"[eval] --crop-to-sidecar needs {manifest_path} (prepare_audio_data.py output)")
        manifest = json.loads(manifest_path.read_text())
        real_crop = (tuple(manifest["offset"]), tuple(manifest["mel_shape"]))
        print(f"[eval] --crop-to-sidecar: real mels cropped to offset={real_crop[0]} mel_shape={real_crop[1]}")

    # Build/load each mode's classifier + frozen reference. Cached per (dataset, mode).
    ctx = {}  # mode -> dict(net, num_classes, mu_r, cov_r, ent_r)
    for mode in modes:
        cache_dir = Path("convergence_cache") / args.dataset / mode
        label_fn, num_classes = build_label_fn(mode, args.npy_dir, args.dataset, args.num_classes)
        if args.retrain or not (cache_dir / "classifier.pt").exists():
            net = train_classifier(args.npy_dir, label_fn, num_classes, real_crop, cache_dir, device,
                                   mode, epochs=args.clf_epochs)
        else:
            print(f"[eval] classifier cache hit ({mode}): {cache_dir / 'classifier.pt'}")
            net = load_classifier(num_classes, cache_dir, device)
        mu_r, cov_r, ent_r = build_reference(
            net, args.npy_dir, label_fn, num_classes, real_crop, cache_dir, device, args.batch, mode
        )
        ctx[mode] = dict(net=net, num_classes=num_classes, mu_r=mu_r, cov_r=cov_r, ent_r=ent_r)

    ckpts = sorted(glob.glob(args.ckpt_glob), key=parse_iter)
    if not ckpts:
        sys.exit(f"[eval] no checkpoints matched {args.ckpt_glob}")

    mode_rows = {mode: [] for mode in modes}
    for cp in ckpts:
        it = parse_iter(cp)
        warn_if_speaker_mismatch(args.npy_dir, cp)
        G = load_generator(cp, device)
        # Sample ONCE (frozen seed) and embed with every mode's classifier, so
        # all curves score the exact same generated mels at half the gen cost.
        mels = sample_mels(G, args.n_samples, args.seed, args.batch, device)
        if args.crop_to_sidecar:
            offset, mel_shape = load_sidecar_crop(cp)
            mels = crop_canvas(mels, offset, mel_shape)
        for mode in modes:
            c = ctx[mode]
            emb, preds = embed_and_predict(c["net"], mels, args.batch, device)
            mu_g, cov_g = gaussian_stats(emb)
            fd = frechet_distance(c["mu_r"], c["cov_r"], mu_g, cov_g)
            ent = norm_entropy(preds, c["num_classes"])
            mode_rows[mode].append((it, fd, ent))
            print(f"iter {it:>7}  [{mode:>7}]  FD={fd:10.3f}  entropy={ent:.3f}  (real={c['ent_r']:.3f})")
        del G
        if device == "cuda":
            torch.cuda.empty_cache()

    # CSV per mode (ALL checkpoints, incl. iter 0 -- the sanity anchor).
    for mode in modes:
        write_csv(f"{args.out}_{mode}.csv", mode_rows[mode])

    # One PNG: a panel per mode (iter-0 omitted by default).
    write_plot(f"{args.out}.png", mode_rows, {m: ctx[m]["ent_r"] for m in modes}, args.plot_omit_iter0)


if __name__ == "__main__":
    main()

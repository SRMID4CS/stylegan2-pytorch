"""
Route B convergence curve for stylegan2-audio checkpoints.

Mel-native domain-classifier Frechet distance + class-coverage entropy.
No vocoding, no BigVGAN, no waveform models -> stays entirely in mel space, so
it runs in seconds/checkpoint. Primary target: AWS g6.xlarge (L4, sm_89); also
runs on the Blackwell laptop for quick local checks (pass
TORCH_CUDA_ARCH_LIST=12.0 in the run command on that box -- see USAGE.md).

Two label modes (--label-mode, default both):
  * content  -- classify the digit/keyword (AudioMNIST: 10 digits, Speech
                 Commands: all 35 words); the digit CNN saturates (~99.8% acc)
                 and is insensitive to speaker-manifold collapse.
  * speaker  -- classify speaker ID over the TRAIN speakers; the collapse
                 tripwire that stress-tests the N1 speaker-diversity claim.
Each mode has its own classifier, reference (mu_r, Sigma_r) and real class-entropy.
Labels come from prepare_audio_data.py's clip_labels.json when present (no
re-parsing of filenames); it falls back to the naming rules in datasets_audio.py
for .npy dirs prepared before that file existed.

Speech Commands also gets a digit-subset coverage read (--digit-coverage,
SPEECH_COMMANDS_SPEC.md §5): what fraction of generated mels the 35-word content
classifier assigns to a zero-nine word, and how evenly across those ten. That is
the credibility hook for the SC prior -> AudioMNIST OOD attack; it rides on the
content classifier, so it costs nothing extra.

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
from datasets_audio import content_from_stem, speaker_from_stem  # noqa: E402

EMB_DIM = 128
DEFAULT_BATCH = 64
LABEL_MODES = ("content", "speaker")


# ---------- stem -> label ----------
# prepare_audio_data.py names every .npy `<speaker>__<orig_stem>[__aug{k}]`, so the
# speaker is ALWAYS the first "__"-token and survives augmentation. The authoritative
# per-clip labels are clip_labels.json (written by prepare_audio_data.py, carrying
# speaker + content straight from the source walk); the stem parse in
# datasets_audio.py is the fallback for dirs prepared before that file existed.
def load_clip_labels(npy_dir):
    """clip_labels.json as {"content_classes", "digit_classes", "clips"} or None."""
    path = Path(npy_dir) / "clip_labels.json"
    if not path.is_file():
        print(f"[eval] no {path.name} in {npy_dir} -- falling back to stem parsing "
              "(re-run prepare_audio_data.py to emit it)")
        return None
    doc = json.loads(path.read_text())
    print(f"[eval] labels from {path}: {len(doc['clips'])} clips, "
          f"{len(doc['content_classes'])} content classes "
          f"({doc.get('content_label_type', 'content')})")
    return doc


def speaker_of(stem: str, clip_labels=None) -> str:
    """Speaker id of a prepared .npy stem, from clip_labels.json if available."""
    if clip_labels is not None:
        entry = clip_labels["clips"].get(stem)
        if entry is not None:
            return entry["speaker"]
    return speaker_from_stem(stem)


def content_of(stem: str, dataset: str, clip_labels=None) -> str:
    """Content label (digit string / word) of a prepared .npy stem."""
    if clip_labels is not None:
        entry = clip_labels["clips"].get(stem)
        if entry is not None and entry.get("content") is not None:
            return entry["content"]
    return content_from_stem(stem, dataset)


def load_train_speakers(npy_dir):
    """The id-disjoint contract's authoritative train-speaker set (spec §3, §9)."""
    split_path = Path(npy_dir) / "speaker_split.json"
    if not split_path.is_file():
        sys.exit(
            f"[eval] speaker mode needs {split_path} (prepare_audio_data.py writes it) "
            "to define the train-speaker label set"
        )
    return json.loads(split_path.read_text())["train_speakers"]


def build_label_fn(mode, npy_dir, dataset, num_classes, clip_labels=None):
    """Return (label_fn(stem)->int, num_classes, meta) for the requested mode.

    content: digit/word label. Class set comes from clip_labels.json when present
             (AudioMNIST 10 digits / Speech Commands 35 words) so --num-classes
             never has to be right; a labels.json ({npy_stem: int}) next to
             --npy-dir still overrides everything. `meta["digit_indices"]` are the
             class indices of the zero-nine words, for the SC OOD coverage read.
    speaker: train-speaker id remapped to a CONTIGUOUS 0..K-1 head (K = number of
             train speakers from speaker_split.json). Non-train speakers can't
             appear -- prepare_audio_data.py only writes train clips -- but we
             cross-check the observed filenames and fail loudly if one does.
    """
    stems = [Path(f).stem for f in sorted(glob.glob(os.path.join(npy_dir, "*.npy")))]
    if not stems:
        sys.exit(f"[eval] no .npy under {npy_dir}")

    if mode == "content":
        lj = Path(npy_dir) / "labels.json"
        if lj.exists():
            labels_map = json.loads(lj.read_text())
            print(f"[eval] content mode: labels.json override ({len(labels_map)} entries), "
                  f"K={num_classes} from --num-classes")
            return (lambda stem: labels_map[stem]), num_classes, {"classes": None, "digit_indices": []}

        if clip_labels is not None:
            classes = list(clip_labels["content_classes"])
            digit_classes = list(clip_labels.get("digit_classes") or [])
        else:
            # Derive the class set from the stems themselves so a pre-clip_labels
            # dir still works; sorted() matches how prepare_audio_data.py orders it.
            classes = sorted({content_of(s, dataset) for s in stems})
            digit_classes = []
        class_to_idx = {c: i for i, c in enumerate(classes)}
        K = len(classes)
        if num_classes is not None and num_classes != K:
            print(f"[eval] WARNING: --num-classes {num_classes} != {K} derived content classes "
                  f"-- using {K} (from {'clip_labels.json' if clip_labels else 'the .npy stems'})")
        digit_indices = [class_to_idx[w] for w in digit_classes if w in class_to_idx]
        print(f"[eval] content mode: K={K} classes {classes if K <= 40 else f'({K} classes)'}"
              + (f"; digit subset {digit_classes}" if digit_indices else ""))

        def fn(stem):
            return class_to_idx[content_of(stem, dataset, clip_labels)]

        return fn, K, {"classes": classes, "digit_indices": digit_indices}

    if mode == "speaker":
        train_speakers = sorted(load_train_speakers(npy_dir))
        speaker_to_idx = {s: i for i, s in enumerate(train_speakers)}
        K = len(train_speakers)

        observed = {speaker_of(s, clip_labels) for s in stems}
        unknown = sorted(observed - set(train_speakers))
        if unknown:
            sys.exit(
                f"[eval] speaker mode: {npy_dir} contains clips from speakers "
                f"{unknown[:20]}{' ...' if len(unknown) > 20 else ''} that are NOT in "
                "speaker_split.json train_speakers -- id-disjoint contract breach (spec §3, §9)"
            )
        missing = sorted(set(train_speakers) - observed)
        if missing:
            print(f"[eval] WARNING: {len(missing)} train_speakers have no .npy clips in {npy_dir} "
                  f"(empty classes in the speaker head): {missing[:20]}")
        # Contract assertion: K derived from the manifest, never hardcoded.
        assert K == len(train_speakers), "speaker K must equal len(train_speakers)"
        print(f"[eval] speaker mode: K={K} train speakers (contiguous remap 0..{K-1}); chance acc ~= {1.0/K:.4f}")

        def fn(stem):
            return speaker_to_idx[speaker_of(stem, clip_labels)]

        return fn, K, {"classes": train_speakers, "digit_indices": []}

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


def class_counts(pred_classes, num_classes):
    return np.bincount(np.asarray(pred_classes, dtype=np.int64), minlength=num_classes)


def norm_entropy(pred_classes, num_classes):
    return norm_entropy_from_counts(class_counts(pred_classes, num_classes), num_classes)


def norm_entropy_from_counts(counts, num_classes):
    # A single class has no diversity to measure and log(1)==0 would make the
    # normalizer 0/0 -> nan (hits the degenerate single-train-speaker case).
    if num_classes <= 1:
        return 0.0
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(num_classes))


def size_matched_entropy(counts, n_samples, num_classes, seed=0):
    """Reference coverage entropy resampled down to the generated sample count.

    Normalized entropy is capped by the number of samples, not just by coverage:
    at Speech Commands scale (K=2418 train speakers, --n-samples 2000) a PERFECTLY
    covering generator can only reach log(2000)/log(2418) = 0.976, and in
    expectation ~0.914 -- while the real reference, measured over all ~97k clips,
    sits at ~0.998. That ~0.08 gap is pure sample-size artifact and would read as
    speaker-manifold collapse in the one signal the N1 diversity claim rests on.
    Drawing the reference from the same number of samples removes it (the residual
    gap is then ~0.004). At AudioMNIST scale (K=48) the correction is ~0.003, which
    is why this never surfaced there.

    Seeded and deterministic; a no-op when the reference has fewer samples than
    the generated set.
    """
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0 or n_samples >= total:
        return norm_entropy_from_counts(counts, num_classes)
    matched = np.random.default_rng(seed).multinomial(int(n_samples), counts / total)
    return norm_entropy_from_counts(matched, num_classes)


def size_matched_counts(counts, n_samples, seed=0):
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0 or n_samples >= total:
        return counts
    return np.random.default_rng(seed).multinomial(int(n_samples), counts / total)


def subset_coverage(counts, subset_indices):
    """(fraction, normalized entropy) of predictions landing in a class subset.

    The SC digit sub-manifold read (SPEECH_COMMANDS_SPEC.md §5): `fraction` is how
    much of the sample the 35-word classifier calls a zero-nine word, `entropy` is
    how evenly it spreads across those ten. Reporting/coverage only -- never a
    checkpoint selector (the primary signal stays speaker-coverage entropy).
    """
    if not subset_indices:
        return float("nan"), float("nan")
    sub = np.asarray(counts, dtype=np.float64)[list(subset_indices)]
    total = float(np.asarray(counts, dtype=np.float64).sum())
    frac = float(sub.sum() / total) if total > 0 else 0.0
    return frac, norm_entropy_from_counts(sub, len(subset_indices))


def build_reference(net, npy_dir, label_fn, num_classes, crop, cache_dir, device, batch, mode):
    """Cache (mu_r, Sigma_r), real class-entropy and the real class histogram over
    ALL real TRAIN-speaker mels under npy_dir. Frozen once cached -- delete the
    cache file to rebuild."""
    cache = cache_dir / "reference_stats.npz"
    if cache.exists():
        d = np.load(cache)
        # "counts" was added with the SC digit-coverage read; older caches lack it.
        counts = d["counts"] if "counts" in d.files else None
        return d["mu"], d["cov"], float(d["entropy"]), counts
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
    counts = class_counts(preds, num_classes)
    ent = norm_entropy_from_counts(counts, num_classes)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache, mu=mu, cov=cov, entropy=ent, counts=counts)
    print(f"[ref:{mode}] N={len(embs)}  real class-entropy={ent:.3f}")
    return mu, cov, ent, counts


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
def write_csv(path, rows, header):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"wrote {path}")


def write_plot(out_png, panels, omit_iter0):
    """One twin-axis panel per entry (a left-axis series + a right-axis series vs
    iteration), stacked vertically. iter-0 is dropped from the PLOT only (default)
    so the untrained checkpoint's blown-out FD doesn't crush the y-axis -- the CSVs
    keep it.

    panels: [dict(title, rows=[(iter, y1, y2)], y1label, y2label, y2ref, y2lim)]
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot skipped] {e}")
        return

    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 4.5 * len(panels)), squeeze=False)
    for ax1, panel in zip(axes[:, 0], panels):
        rows = [r for r in panel["rows"] if r[0] != 0] if omit_iter0 else panel["rows"]
        if not rows:
            ax1.set_title(f"{panel['title']}: no rows to plot (all omitted)")
            continue
        it, y1, y2 = zip(*rows)
        ax1.plot(it, y1, "o-", color="C0")
        ax1.set_xlabel("iteration"); ax1.set_ylabel(panel["y1label"], color="C0")
        if panel.get("y1lim"):
            ax1.set_ylim(*panel["y1lim"])
        ax2 = ax1.twinx()
        ax2.plot(it, y2, "s--", color="C1")
        if panel.get("y2ref") is not None and np.isfinite(panel["y2ref"]):
            ax2.axhline(panel["y2ref"], color="C1", lw=0.8, alpha=0.5)
        ax2.set_ylabel(panel["y2label"], color="C1")
        ax2.set_ylim(*panel.get("y2lim", (0, 1.05)))
        ax1.set_title(panel["title"] + ("  (iter-0 omitted)" if omit_iter0 else ""))
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
                    help="cache namespace (convergence_cache/<dataset>/<mode>/) and content label "
                         "rule: 'audiomnist' | 'speech_commands'")
    ap.add_argument("--label-mode", choices=["content", "speaker", "both"], default="both",
                    help="content = digit/word curve; speaker = speaker-id collapse tripwire; both = both")
    ap.add_argument("--num-classes", type=int, default=None,
                    help="content mode: override the class count (AudioMNIST=10 digits / Speech "
                         "Commands=35 words). Normally unnecessary -- it is derived from "
                         "clip_labels.json. Ignored in speaker mode (K comes from speaker_split.json)")
    ap.add_argument("--digit-coverage", choices=["auto", "on", "off"], default="auto",
                    help="extra zero-nine content-coverage read for the SC->AudioMNIST OOD story "
                         "(SPEECH_COMMANDS_SPEC.md §5); auto = on when the dataset has a digit "
                         "subset and content mode is active")
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

    clip_labels = load_clip_labels(args.npy_dir)

    # Build/load each mode's classifier + frozen reference. Cached per (dataset, mode).
    ctx = {}  # mode -> dict(net, num_classes, mu_r, cov_r, ent_r, counts_r, digit_indices)
    for mode in modes:
        cache_dir = Path("convergence_cache") / args.dataset / mode
        label_fn, num_classes, meta = build_label_fn(
            mode, args.npy_dir, args.dataset, args.num_classes, clip_labels
        )
        if args.retrain or not (cache_dir / "classifier.pt").exists():
            net = train_classifier(args.npy_dir, label_fn, num_classes, real_crop, cache_dir, device,
                                   mode, epochs=args.clf_epochs)
        else:
            print(f"[eval] classifier cache hit ({mode}): {cache_dir / 'classifier.pt'}")
            net = load_classifier(num_classes, cache_dir, device)
        mu_r, cov_r, ent_r, counts_r = build_reference(
            net, args.npy_dir, label_fn, num_classes, real_crop, cache_dir, device, args.batch, mode
        )
        # Compare like with like: the reference entropy is measured over every real
        # clip (~97k), the per-checkpoint one over --n-samples. See size_matched_entropy.
        ent_r_matched = ent_r
        if counts_r is not None:
            ent_r_matched = size_matched_entropy(counts_r, args.n_samples, num_classes, args.seed)
        if num_classes > args.n_samples:
            ceiling = np.log(args.n_samples) / np.log(num_classes)
            print(f"[eval] NOTE ({mode}): K={num_classes} classes > --n-samples {args.n_samples}, so "
                  f"even perfect coverage caps the entropy at {ceiling:.3f}. The reference below is "
                  f"size-matched; raise --n-samples for a tighter read (keep it frozen across a run).")
        if abs(ent_r_matched - ent_r) > 1e-3:
            print(f"[ref:{mode}] real class-entropy {ent_r:.3f} over all clips -> {ent_r_matched:.3f} "
                  f"size-matched to --n-samples {args.n_samples} (that is the comparable number)")
        ctx[mode] = dict(net=net, num_classes=num_classes, mu_r=mu_r, cov_r=cov_r,
                         ent_r=ent_r, ent_r_matched=ent_r_matched,
                         counts_r=counts_r, digit_indices=meta["digit_indices"])

    # Digit-subset coverage rides on the content classifier -- no extra sampling.
    digit_indices = ctx.get("content", {}).get("digit_indices", [])
    if args.digit_coverage == "on" and not digit_indices:
        sys.exit(
            "[eval] --digit-coverage on needs content mode and a digit subset: pass "
            "--label-mode content|both and an --npy-dir whose clip_labels.json lists "
            "digit_classes (Speech Commands only)"
        )
    do_digits = bool(digit_indices) and args.digit_coverage != "off"
    digit_ref = (float("nan"), float("nan"))
    if do_digits:
        counts_r = ctx["content"]["counts_r"]
        if counts_r is None:
            print("[eval] digit coverage: cached reference predates the class histogram -- real "
                  "reference unavailable, pass --retrain to rebuild it")
        else:
            # Fraction from the full corpus (a proportion -- unbiased, and the best
            # estimate available); entropy size-matched, same reason as above.
            digit_frac_r, _ = subset_coverage(counts_r, digit_indices)
            _, digit_ent_r = subset_coverage(
                size_matched_counts(counts_r, args.n_samples, args.seed), digit_indices
            )
            digit_ref = (digit_frac_r, digit_ent_r)
            print(f"[digits] real: fraction={digit_ref[0]:.3f}  entropy={digit_ref[1]:.3f}")

    ckpts = sorted(glob.glob(args.ckpt_glob), key=parse_iter)
    if not ckpts:
        sys.exit(f"[eval] no checkpoints matched {args.ckpt_glob}")

    mode_rows = {mode: [] for mode in modes}
    digit_rows = []
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
            counts = class_counts(preds, c["num_classes"])
            ent = norm_entropy_from_counts(counts, c["num_classes"])
            mode_rows[mode].append((it, fd, ent))
            print(f"iter {it:>7}  [{mode:>7}]  FD={fd:10.3f}  entropy={ent:.3f}  "
                  f"(real={c['ent_r_matched']:.3f})")
            if mode == "content" and do_digits:
                frac, dent = subset_coverage(counts, digit_indices)
                digit_rows.append((it, frac, dent))
                print(f"iter {it:>7}  [ digits]  fraction={frac:.3f}  entropy={dent:.3f}  "
                      f"(real fraction={digit_ref[0]:.3f} entropy={digit_ref[1]:.3f})")
        del G
        if device == "cuda":
            torch.cuda.empty_cache()

    # CSV per mode (ALL checkpoints, incl. iter 0 -- the sanity anchor).
    panels = []
    for mode in modes:
        write_csv(f"{args.out}_{mode}.csv", mode_rows[mode], ["iter", "fd", "entropy"])
        panels.append(dict(title=f"{mode} curve", rows=mode_rows[mode],
                           y1label="Frechet distance", y2label="coverage entropy (norm.)",
                           y2ref=ctx[mode]["ent_r_matched"]))
    if digit_rows:
        write_csv(f"{args.out}_digits.csv", digit_rows, ["iter", "digit_fraction", "digit_entropy"])
        panels.append(dict(title="digit sub-manifold coverage (OOD read)", rows=digit_rows,
                           y1label="fraction predicted zero-nine", y2label="digit entropy (norm.)",
                           y2ref=digit_ref[1], y1lim=(0, 1.05)))

    # One PNG: a panel per mode (iter-0 omitted by default).
    write_plot(f"{args.out}.png", panels, args.plot_omit_iter0)


if __name__ == "__main__":
    main()

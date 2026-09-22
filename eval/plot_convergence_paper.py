"""
Paper figures for the Route B convergence curves of the two mel priors.

Redraws the CSVs written by eval/convergence_curve.py (domain-classifier Frechet
distance + normalized class-coverage entropy per checkpoint) as print-ready
figures, with the chosen checkpoint marked. Standalone: needs only numpy and
matplotlib -- no torch, no generator, no repo imports -- so it runs in any env.

Inputs (defaults match the repo layout):
    <root>/AM/convergence_curve_{speaker,content}.csv            AudioMNIST prior
    <root>/SC/convergence_curve_{speaker,content,digits}.csv     Speech Commands prior
    <cache>/<dataset>/<mode>/reference_stats.npz                 real-data reference
The reference line is recomputed exactly as convergence_curve.py plotted it: the
real TRAIN-speaker class histogram resampled to --n-samples with --seed (a no-op
for caches that predate the histogram -- the AudioMNIST ones -- where the stored
full-corpus entropy is used, as the eval did). Missing cache -> no reference line.
The legacy convergence_curve.csv (no mode suffix) is not used.

Outputs, next to the CSVs, each as .pdf (vector, fonts embedded as TrueType) and
.png (300 dpi):
    AM/fig1_am_speaker            AudioMNIST, speaker-ID embedding
    AM/fig2_am_content            AudioMNIST, digit embedding
    SC/fig3_sc_speaker            Speech Commands, speaker-ID embedding
    SC/fig4_sc_content            Speech Commands, 35-keyword embedding
    SC/fig5_sc_digits             Speech Commands, zero-nine sub-manifold coverage (OOD read)
    AM/fig6_am_speaker_content    figs 1 + 2 side by side
    SC/fig7_sc_speaker_content    figs 3 + 4 side by side

Every curve is two panels on a shared iteration axis -- FD on top (log scale),
coverage entropy below -- never one panel with two y-scales. Iteration 0 (the
untrained generator, FD in the 1e4-1e6 range) is left off the plots; the CSVs keep it.
Single figures are sized for half the ICLR text width, combined ones for the full
width (5.5 in), so they render at native font size.

Run (from repo root):
    python eval/plot_convergence_paper.py
    python eval/plot_convergence_paper.py --am-ckpt 300000 --sc-ckpt 400000 --no-titles
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import (  # noqa: E402
    FuncFormatter, LogLocator, MaxNLocator, MultipleLocator, NullFormatter, ScalarFormatter,
)

# ---------- style ----------
# Two hues only (validated CVD-safe pair): the generator curve and the selected
# checkpoint. The real-data reference is a reference, not a series -> ink gray.
GEN = "#2a78d6"
SELECT = "#eb6834"
REF = "#52514e"
INK = "#0b0b0b"
INK_2 = "#52514e"
AXIS = "#898781"
GRID = "#e1e0d9"

PAPER_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.6,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "ytick.minor.size": 1.5,
    "ytick.minor.width": 0.4,
    "pdf.fonttype": 42,  # TrueType, not Type 3 -- conference PDF checkers reject Type 3
    "ps.fonttype": 42,
    "savefig.dpi": 300,
}

SINGLE_SIZE = (2.75, 2.7)   # inches: 0.5\linewidth of the 5.5 in ICLR text block
COMBINED_SIZE = (5.5, 2.7)  # inches: \linewidth

# Speech Commands v0.02 content classes in the order the eval's class histogram is
# indexed: sorted(words), exactly as prepare_audio_data.py writes content_classes
# (and as the SC sidecar's content_labels.classes lists them).
SC_WORDS = sorted(
    "backward bed bird cat dog down eight five follow forward four go happy house "
    "learn left marvin nine no off on one right seven sheila six stop three tree two "
    "up visual wow yes zero".split()
)
SC_DIGIT_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


# ---------- data ----------
def read_curve(path):
    """CSV -> (iters, col1, col2) float arrays, sorted by iteration."""
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f)][1:]
    arr = np.array(sorted((float(a), float(b), float(c)) for a, b, c in rows))
    return arr[:, 0].astype(int), arr[:, 1], arr[:, 2]


# Mirrors convergence_curve.py's norm_entropy_from_counts / size_matched_counts --
# that file is the source of truth; copied (not imported) so this stays torch-free.
def norm_entropy(counts, num_classes):
    counts = np.asarray(counts, dtype=np.float64)
    if num_classes <= 1 or counts.sum() <= 0:
        return 0.0
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(num_classes))


def size_matched_counts(counts, n_samples, seed):
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0 or n_samples >= total:
        return counts
    return np.random.default_rng(seed).multinomial(int(n_samples), counts / total)


def load_reference(cache_root, dataset, mode):
    path = Path(cache_root) / dataset / mode / "reference_stats.npz"
    if not path.is_file():
        print(f"[plot] WARNING: no {path} -- {dataset}/{mode} reference line omitted")
        return None
    d = np.load(path)
    return {"entropy": float(d["entropy"]),
            "counts": d["counts"] if "counts" in d.files else None}


def reference_entropy(ref, n_samples, seed):
    """Real-data coverage entropy, as convergence_curve.py plotted it."""
    if ref is None:
        return None
    if ref["counts"] is None:
        return ref["entropy"]
    return norm_entropy(size_matched_counts(ref["counts"], n_samples, seed), len(ref["counts"]))


def reference_digits(ref, n_samples, seed):
    """(real zero-nine fraction, real digit entropy) from the 35-word histogram."""
    if ref is None or ref["counts"] is None:
        return None, None
    counts = np.asarray(ref["counts"], dtype=np.float64)
    if len(counts) != len(SC_WORDS):
        sys.exit(f"[plot] SC content reference has {len(counts)} classes, expected {len(SC_WORDS)}")
    idx = [SC_WORDS.index(w) for w in SC_DIGIT_WORDS]
    frac = float(counts[idx].sum() / counts.sum())  # full corpus: a proportion, unbiased
    ent = norm_entropy(size_matched_counts(counts, n_samples, seed)[idx], len(idx))
    return frac, ent


# ---------- one curve = FD panel over entropy panel ----------
def make_curve(title, csv_path, selected, top_label, bot_label, top_ref=None, bot_ref=None,
               top_log=True, include_iter0=False):
    it, top, bot = read_curve(csv_path)
    if selected not in it:
        sys.exit(f"[plot] selected checkpoint {selected} is not a row of {csv_path}")
    keep = np.ones_like(it, dtype=bool) if include_iter0 else it != 0
    return dict(title=title, path=csv_path, it=it[keep], top=top[keep], bot=bot[keep],
                selected=selected, top_label=top_label, bot_label=bot_label,
                top_ref=top_ref, bot_ref=bot_ref, top_log=top_log)


def fmt_iter(x, _pos=None):
    return "0" if x == 0 else f"{x / 1000:g}k"


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", which="major", color=GRID, lw=0.5, ls="-")
    ax.set_axisbelow(True)


def draw_panel(ax, it, y, selected, ref, log, ylabel):
    """Draw one panel; returns the selected-point value label, placed after layout."""
    style_axes(ax)
    ax.axvline(selected, color=SELECT, lw=0.8, alpha=0.55, zorder=1)
    if ref is not None:
        ax.axhline(ref, color=REF, lw=0.9, ls=(0, (4, 2)), zorder=2)
    ax.plot(it, y, color=GEN, lw=1.2, marker="o", ms=2.6, mec="white", mew=0.4,
            solid_joinstyle="round", solid_capstyle="round", zorder=3)
    y_sel = y[list(it).index(selected)]
    ax.plot([selected], [y_sel], marker="o", ms=5.5, color=SELECT, mec="white", mew=0.9,
            ls="none", zorder=4)
    ax.set_ylabel(ylabel)

    if log:
        ax.set_yscale("log")
        ax.margins(y=0.12)  # headroom so the selected-value label can sit below a minimum
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        lo, hi = float(np.min(y)), float(np.max(y))
        if ref is not None:
            lo, hi = min(lo, ref), max(hi, ref)
        pad = 0.15 * (hi - lo) if hi > lo else 0.05
        ax.set_ylim(lo - pad, hi + pad)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10], min_n_ticks=3))
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

    polylines = [np.column_stack([it, y])]
    if ref is not None:
        polylines.append(np.array([[0, ref], [it.max() + 10_000, ref]]))
    return dict(ax=ax, xy=(selected, y_sel), text=f"{y_sel:.3g}", polylines=polylines)


# Candidate offsets (points) for a selected-value label: beside the marker, then further out.
LABEL_SPOTS = [(-5, 4, "right", "bottom"), (-5, -4, "right", "top"),
               (5, 4, "left", "bottom"), (5, -4, "left", "top"),
               (-5, 11, "right", "bottom"), (-5, -11, "right", "top"),
               (-14, 0, "right", "center")]


def place_value_labels(fig, labels):
    """Put each selected-value label at the candidate spot that hits the fewest
    drawn curve/reference-line pixels and stays inside its axes. Runs after one
    draw so constrained layout has fixed the axes; labels are kept out of layout."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for lab in labels:
        ax = lab["ax"]
        pts = []
        for line in lab["polylines"]:
            d = ax.transData.transform(line)
            for a, b in zip(d[:-1], d[1:]):
                pts.append(a + np.linspace(0, 1, 40)[:, None] * (b - a))
        pts = np.concatenate(pts)
        ax_box = ax.get_window_extent(renderer)
        ann = ax.annotate(lab["text"], lab["xy"], xytext=(0, 0), textcoords="offset points", fontsize=6.5,
                          color=INK, zorder=5, annotation_clip=False,
                          bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85))
        ann.set_in_layout(False)
        best = None
        for rank, (dx, dy, ha, va) in enumerate(LABEL_SPOTS):
            ann.xyann = (dx, dy)
            ann.set_ha(ha)
            ann.set_va(va)
            box = ann.get_window_extent(renderer).expanded(1.1, 1.2)
            hits = int(np.sum((pts[:, 0] >= box.x0) & (pts[:, 0] <= box.x1)
                              & (pts[:, 1] >= box.y0) & (pts[:, 1] <= box.y1)))
            outside = not (box.x0 >= ax_box.x0 and box.x1 <= ax_box.x1
                           and box.y0 >= ax_box.y0 and box.y1 <= ax_box.y1)
            score = (outside, hits, rank)
            if best is None or score < best[0]:
                best = (score, (dx, dy, ha, va))
        dx, dy, ha, va = best[1]
        ann.xyann = (dx, dy)
        ann.set_ha(ha)
        ann.set_va(va)


def draw_curve(ax_top, ax_bot, c, show_title):
    labels = [
        draw_panel(ax_top, c["it"], c["top"], c["selected"], c["top_ref"], c["top_log"], c["top_label"]),
        draw_panel(ax_bot, c["it"], c["bot"], c["selected"], c["bot_ref"], False, c["bot_label"]),
    ]
    if show_title:
        ax_top.set_title(c["title"], loc="left", pad=3)
    ax_bot.set_xlabel("Training iteration")
    ax_bot.set_xlim(0, c["it"].max() + 10_000)
    ax_bot.xaxis.set_major_locator(MultipleLocator(100_000))
    ax_bot.xaxis.set_major_formatter(FuncFormatter(fmt_iter))
    ax_top.tick_params(axis="x", which="both", length=0)
    return labels


def legend_handles(curves):
    sel = curves[0]["selected"]
    handles = [Line2D([], [], color=GEN, lw=1.2, marker="o", ms=2.6, mec="white", mew=0.4)]
    labels = ["Generator"]
    if any(c["top_ref"] is not None or c["bot_ref"] is not None for c in curves):
        handles.append(Line2D([], [], color=REF, lw=0.9, ls=(0, (4, 2))))
        labels.append("Real data")
    handles.append(Line2D([], [], color=SELECT, lw=0.8, marker="o", ms=5.5, mec="white", mew=0.9))
    labels.append(f"Selected ({fmt_iter(sel)})")
    return handles, labels


def save(fig, out_base):
    fig.savefig(out_base.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out_base.with_suffix(".png"), facecolor="white")
    plt.close(fig)
    print(f"[plot] wrote {out_base.with_suffix('.pdf')}  (+ .png)")


def figure_single(c, out_base, show_title):
    fig, (ax_t, ax_b) = plt.subplots(2, 1, sharex=True, figsize=SINGLE_SIZE, layout="constrained")
    labels = draw_curve(ax_t, ax_b, c, show_title)
    fig.align_ylabels([ax_t, ax_b])
    fig.legend(*legend_handles([c]), loc="outside upper center", ncol=3, frameon=False,
               handlelength=1.8, columnspacing=0.9, handletextpad=0.4)
    place_value_labels(fig, labels)
    save(fig, out_base)


def figure_pair(c_left, c_right, out_base, show_title):
    fig, axes = plt.subplots(2, 2, sharex="col", figsize=COMBINED_SIZE, layout="constrained")
    labels = []
    for col, c in enumerate((c_left, c_right)):
        tag = f"({'ab'[col]}) " if show_title else ""
        labels += draw_curve(axes[0, col], axes[1, col], dict(c, title=tag + c["title"]), show_title)
    fig.align_ylabels(axes[:, 0])
    fig.align_ylabels(axes[:, 1])
    fig.legend(*legend_handles([c_left, c_right]), loc="outside upper center", ncol=3,
               frameon=False, handlelength=2.0, columnspacing=1.6, handletextpad=0.5)
    place_value_labels(fig, labels)
    save(fig, out_base)


# ---------- numbers for the paper text ----------
def summarize(name, c, last_k=10):
    it, sel = c["it"], c["selected"]
    for key, label, better in (("top", c["top_label"], "min"), ("bot", c["bot_label"], "max")):
        y = c[key]
        i_best = int(np.argmin(y) if better == "min" else np.argmax(y))
        y_sel = y[list(it).index(sel)]
        tail = y[-last_k:]
        ref = c[f"{key}_ref"]
        print(f"  {name:<16} {label:<17} sel@{fmt_iter(sel)}={y_sel:.4g}  "
              f"{better}={y[i_best]:.4g}@{fmt_iter(it[i_best])}  "
              f"first={y[0]:.4g}@{fmt_iter(it[0])}  "
              f"last{last_k} mean={tail.mean():.4g} sd={tail.std():.3g}"
              + (f"  ref={ref:.4g}" if ref is not None else ""))


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="convergence_final",
                    help="folder holding AM/ and SC/ convergence CSVs; figures are written there")
    ap.add_argument("--cache", default="convergence_cache",
                    help="convergence_curve.py reference cache (real-data reference lines)")
    ap.add_argument("--am-ckpt", type=int, default=300_000, help="selected AudioMNIST checkpoint")
    ap.add_argument("--sc-ckpt", type=int, default=400_000, help="selected Speech Commands checkpoint")
    ap.add_argument("--n-samples", type=int, default=2000,
                    help="the eval's frozen --n-samples (size-matches the reference entropy)")
    ap.add_argument("--seed", type=int, default=0, help="the eval's frozen --seed")
    ap.add_argument("--no-titles", action="store_true",
                    help="drop in-figure panel titles (let the LaTeX caption carry them)")
    ap.add_argument("--include-iter0", action="store_true", help="also plot the untrained iter-0 point")
    args = ap.parse_args()

    root = Path(args.root)
    am_dir, sc_dir = root / "AM", root / "SC"
    for d in (am_dir, sc_dir):
        if not d.is_dir():
            sys.exit(f"[plot] missing {d}")
    titles = not args.no_titles
    # No up/down arrows in axis labels: rotated 90 degrees they read as left/right.
    # The captions say lower FD / higher entropy is better.
    fd_label = "FD"
    ent_label = "Coverage entropy"

    def ref_ent(dataset, mode):
        return reference_entropy(load_reference(args.cache, dataset, mode), args.n_samples, args.seed)

    common = dict(include_iter0=args.include_iter0)
    am_spk = make_curve("AudioMNIST: speaker identity", am_dir / "convergence_curve_speaker.csv",
                        args.am_ckpt, fd_label, ent_label,
                        bot_ref=ref_ent("audiomnist", "speaker"), **common)
    am_cnt = make_curve("AudioMNIST: spoken digit", am_dir / "convergence_curve_content.csv",
                        args.am_ckpt, fd_label, ent_label,
                        bot_ref=ref_ent("audiomnist", "content"), **common)
    sc_spk = make_curve("Speech Commands: speaker identity", sc_dir / "convergence_curve_speaker.csv",
                        args.sc_ckpt, fd_label, ent_label,
                        bot_ref=ref_ent("speech_commands", "speaker"), **common)
    sc_cnt = make_curve("Speech Commands: keyword (35 words)", sc_dir / "convergence_curve_content.csv",
                        args.sc_ckpt, fd_label, ent_label,
                        bot_ref=ref_ent("speech_commands", "content"), **common)
    dig_frac_ref, dig_ent_ref = reference_digits(
        load_reference(args.cache, "speech_commands", "content"), args.n_samples, args.seed)
    sc_dig = make_curve("Speech Commands: digit sub-manifold", sc_dir / "convergence_curve_digits.csv",
                        args.sc_ckpt, "Digit fraction", "Digit entropy",
                        top_ref=dig_frac_ref, bot_ref=dig_ent_ref, top_log=False, **common)

    with plt.rc_context(PAPER_RC):
        figure_single(am_spk, am_dir / "fig1_am_speaker", titles)
        figure_single(am_cnt, am_dir / "fig2_am_content", titles)
        figure_single(sc_spk, sc_dir / "fig3_sc_speaker", titles)
        figure_single(sc_cnt, sc_dir / "fig4_sc_content", titles)
        figure_single(sc_dig, sc_dir / "fig5_sc_digits", titles)
        # Combined panels: short column titles -- the caption names the dataset.
        figure_pair(dict(am_spk, title="Speaker identity"), dict(am_cnt, title="Spoken digit"),
                    am_dir / "fig6_am_speaker_content", titles)
        figure_pair(dict(sc_spk, title="Speaker identity"), dict(sc_cnt, title="Keyword (35 words)"),
                    sc_dir / "fig7_sc_speaker_content", titles)

    print("\n[plot] values behind the figures (iter-0 excluded unless --include-iter0):")
    for name, c in (("AM speaker", am_spk), ("AM digit", am_cnt), ("SC speaker", sc_spk),
                    ("SC keyword", sc_cnt), ("SC digit subset", sc_dig)):
        summarize(name, c)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
visualize_lime.py
=================
Visualize LIME explanations for fMRI voxel predictions.
Styled to match shap_plots.py exactly.

Usage:
    python visualize_lime.py \
        --story sweetaspie \
        --data_dir /path/to/data/shap_lime/sweetaspie_s2 \
        --raw_text_pkl /path/to/raw_text.pkl \
        --voxel_ccs /path/to/cc_bert_lora_subject2.npy \
        --subject 2 \
        --top_n 15 \
        --out_dir ./figs/lime/storyname_s2 (or s3 if for subject 3)

    # add --per_voxel for individual per-voxel figures
    # add --aggregate for a cross-voxel averaged importance plot
"""

import argparse
import json
import os
import pickle
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── style: identical to shap_plots.py ─────────────────────────────────────────
BLUE  = "#2C5F8A"
RED   = "#C0392B"
GRAY  = "#5A5A5A"
LIGHT = "#F4F6F9"

STOPWORDS = {
    "i", "and", "the", "to", "a", "of", "my", "you", "was", "in",
    "that", "it", "we", "she", "he", "they", "for", "on", "be",
    "is", "are", "at", "an", "as", "by", "had", "with", "but",
    # tokenization artifacts
    "t", "s", "re", "ve", "ll", "d", "m",
}

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.facecolor":     LIGHT,
    "figure.facecolor":   "white",
    "axes.titlesize":     13,
    "axes.labelsize":     11,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "axes.titleweight":   "bold",
})


# ── helpers ────────────────────────────────────────────────────────────────────

def load_lime_results(results_dir):
    lime_files = {}
    for fname in os.listdir(results_dir):
        m = re.match(r"lime_voxel(\d+)\.json", fname)
        if m:
            idx = int(m.group(1))
            with open(os.path.join(results_dir, fname)) as f:
                lime_files[idx] = json.load(f)
    return lime_files


def filter_stopwords(lime_dict):
    return {w: v for w, v in lime_dict.items()
            if w.lower().strip(".,!?\"'") not in STOPWORDS}


def aggregate_lime(lime_data, filter_stops=True):
    accum = defaultdict(list)
    for lime_dict in lime_data.values():
        d = filter_stopwords(lime_dict) if filter_stops else lime_dict
        for word, val in d.items():
            accum[word].append(val)
    return {w: np.mean(vs) for w, vs in accum.items()}


def _story_label(story, subject):
    """Format story + subject label exactly like shap_plots.py titles."""
    names = {
        "sweetaspie": "Sweetaspie",
        "thetriangleshirtwaistconnection": "The Triangle Shirtwaist Connection",
    }
    display = names.get(story.lower(), story)
    return f"{display} \u00b7 S{subject}"


# ── LIME bar chart (single panel) ─────────────────────────────────────────────

def plot_lime_bars(lime_dict, voxel_idx, cc, ax, n=15, filter_stops=True):
    d      = filter_stopwords(lime_dict) if filter_stops else lime_dict
    by_abs = sorted(d.items(), key=lambda x: abs(x[1]), reverse=True)[:n]
    items  = sorted(by_abs, key=lambda x: x[1])
    words  = [w for w, _ in items]
    vals   = np.array([v for _, v in items])

    if len(vals) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", transform=ax.transAxes)
        return

    vmax   = np.abs(vals).max()
    # solid blue for positive, solid red for negative — same as shap_plots.py
    colors = [BLUE if v >= 0 else RED for v in vals]

    bars = ax.barh(words, vals, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.7, zorder=2)

    for bar, val in zip(bars, vals):
        xoff = vmax * 0.015 if val >= 0 else -vmax * 0.015
        ha   = "left" if val >= 0 else "right"
        ax.text(val + xoff, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha=ha,
                fontsize=7.5, color=GRAY)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LIME importance  (contribution to activation)", fontsize=10)
    ax.set_title(f"Top {n} words  --  LIME", fontsize=11, fontweight="bold",
                 loc="left")
    ax.text(0.99, 1.01, f"Voxel {voxel_idx}  (CC = {cc:.3f})",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color=GRAY, style="italic")
    ax.set_facecolor(LIGHT)


# ── 2x2 summary grid ─────────────────────────────────────────────────────────

def _summary_2x2(voxels, plot_fn, suptitle, fname_stem, out_dir):
    ncols = 2
    nrows = int(np.ceil(len(voxels) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(11 * ncols, 7 * nrows),
                             facecolor="white", squeeze=False)

    fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=1.01)

    for k, vox_idx in enumerate(voxels):
        r, c = divmod(k, ncols)
        plot_fn(axes[r][c], vox_idx)

    for k in range(len(voxels), nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    for ext, dpi in [(".pdf", 300), (".png", 150)]:
        fig.savefig(os.path.join(out_dir, fname_stem + ext),
                    dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved -> {os.path.join(out_dir, fname_stem + '.pdf')}")


def make_summary_lime(lime_data, voxel_ccs, story, subject, out_dir, n=15):
    voxels = sorted(lime_data.keys())
    label  = _story_label(story, subject)

    def plot_fn(ax, vox_idx):
        cc = float(voxel_ccs[vox_idx]) if voxel_ccs is not None else float("nan")
        plot_lime_bars(lime_data[vox_idx], vox_idx, cc, ax,
                       n=n, filter_stops=True)

    _summary_2x2(
        voxels     = voxels,
        plot_fn    = plot_fn,
        suptitle   = f"Top {n} LIME Words per Voxel  --  {label}",
        fname_stem = "summary_lime_all_voxels",
        out_dir    = out_dir,
    )


# ── aggregate cross-voxel plot ────────────────────────────────────────────────

def make_aggregate_lime(lime_data, story, subject, out_dir, n=20):
    label  = _story_label(story, subject)
    avg    = aggregate_lime(lime_data, filter_stops=True)
    by_abs = sorted(avg.items(), key=lambda x: abs(x[1]), reverse=True)[:n]
    items  = sorted(by_abs, key=lambda x: x[1])
    words  = [w for w, _ in items]
    vals   = np.array([v for _, v in items])

    if len(vals) == 0:
        print("  [aggregate] no words after filtering -- skipping")
        return

    vmax   = np.abs(vals).max()
    colors = [BLUE if v >= 0 else RED for v in vals]

    fig, ax = plt.subplots(figsize=(10, max(6, len(vals) * 0.45)),
                           facecolor="white")

    bars = ax.barh(words, vals, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.7, zorder=2)

    for bar, val in zip(bars, vals):
        xoff = vmax * 0.01 if val >= 0 else -vmax * 0.01
        ha   = "left" if val >= 0 else "right"
        ax.text(val + xoff, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha=ha, fontsize=7.5, color=GRAY)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean LIME importance across voxels", fontsize=10)
    ax.set_title(
        f"Aggregate LIME Word Importance  --  {label}\n"
        f"(averaged across {len(lime_data)} voxels, stopwords removed)",
        fontsize=13, fontweight="bold", loc="left"
    )
    ax.set_facecolor(LIGHT)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    for ext, dpi in [(".pdf", 300), (".png", 150)]:
        fig.savefig(os.path.join(out_dir, f"aggregate_lime{ext}"),
                    dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved -> {os.path.join(out_dir, 'aggregate_lime.pdf')}")


# ── per-voxel individual figure ───────────────────────────────────────────────

def make_voxel_figure_lime(lime_dict, voxel_idx, cc,
                            story, subject, out_dir, n=15):
    label = _story_label(story, subject)
    fig, ax = plt.subplots(figsize=(9, max(5, n * 0.45)), facecolor="white")
    fig.suptitle(
        f"LIME Explanation  --  {label}  --  "
        f"Voxel {voxel_idx}  (CC = {cc:.4f})",
        fontsize=12, fontweight="normal", style="italic", y=1.01
    )
    plot_lime_bars(lime_dict, voxel_idx, cc, ax, n=n, filter_stops=True)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    for ext, dpi in [(".pdf", 300), (".png", 150)]:
        fig.savefig(os.path.join(out_dir, f"lime_voxel_{voxel_idx}{ext}"),
                    dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved -> {os.path.join(out_dir, f'lime_voxel_{voxel_idx}.pdf')}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--story",        required=True)
    p.add_argument("--results_dir",  required=True)
    p.add_argument("--raw_text_pkl", required=True)
    p.add_argument("--voxel_ccs",    default=None)
    p.add_argument("--subject",      default="2")
    p.add_argument("--top_n",        type=int, default=15)
    p.add_argument("--out_dir",      default="./figs")
    p.add_argument("--per_voxel",    action="store_true")
    p.add_argument("--aggregate",    action="store_true")
    args = p.parse_args()

    voxel_ccs_arr = None
    if args.voxel_ccs and os.path.exists(args.voxel_ccs):
        voxel_ccs_arr = np.load(args.voxel_ccs)
        print(f"[ccs]  loaded {voxel_ccs_arr.shape}")

    lime_data = load_lime_results(args.results_dir)
    print(f"[results] {len(lime_data)} voxels: {sorted(lime_data.keys())}")

    os.makedirs(args.out_dir, exist_ok=True)

    print("[summary] LIME 2x2 grid ...")
    make_summary_lime(lime_data, voxel_ccs_arr,
                      args.story, args.subject, args.out_dir, n=args.top_n)

    if args.aggregate:
        print("[aggregate] cross-voxel LIME importance ...")
        make_aggregate_lime(lime_data, args.story, args.subject,
                            args.out_dir, n=args.top_n)

    if args.per_voxel:
        for vox_idx in sorted(lime_data.keys()):
            cc = float(voxel_ccs_arr[vox_idx]) \
                 if voxel_ccs_arr is not None else float("nan")
            print(f"  voxel {vox_idx}  CC = {cc:.4f}")
            make_voxel_figure_lime(
                lime_dict = lime_data[vox_idx],
                voxel_idx = vox_idx,
                cc        = cc,
                story     = args.story,
                subject   = args.subject,
                out_dir   = args.out_dir,
                n         = args.top_n,
            )

    print(f"\ndone -- figures in {args.out_dir}")


if __name__ == "__main__":
    main()

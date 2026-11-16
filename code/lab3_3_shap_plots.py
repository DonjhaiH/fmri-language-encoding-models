"""
SHAP Visualization Script for fMRI Word Importance Analysis
Lab 3.3 — Stat 214, Spring 2026

Generates 4 publication-quality plots:
  1. Global Word Importance       — mean |SHAP| across all voxels
  2. Voxel x Word Heatmap         — word-voxel SHAP matrix
  3. CC Distribution Comparison   — across embedding types & subjects
  4. Positive vs Negative Drivers — signed SHAP analysis per voxel

Usage (run on your server):
    python3 shap_plots.py
"""

import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings("ignore")

# ── Optional: suppress numpy deprecation from ridge_utils pickle ──────────────
import logging
logging.captureWarnings(True)

# =============================================================================
# PATHS — edit these if your layout differs
# =============================================================================
CODE_DIR       = "/path/to/fmri-language-encoding-models/code"
RAW_TEXT_PKL   = "/path/to/data/raw_text.pkl"
SHAP_BASE_DIR  = "/path/to/fmri-language-encoding-models/data/shap_lime"
CC_DIR         = "/path/to/fmri-language-encoding-models/data/ridge_results"
OUTPUT_DIR     = "/path/to/fmri-language-encoding-models/figs/shap"

os.makedirs(OUTPUT_DIR, exist_ok=True)
sys.path.append(CODE_DIR)

# =============================================================================
# STYLE
# =============================================================================
BLUE   = "#2C5F8A"
RED    = "#C0392B"
TEAL   = "#1A7A6E"
GOLD   = "#D4A017"
GRAY   = "#5A5A5A"
LIGHT  = "#F4F6F9"

STOPWORDS = {
    "i", "and", "the", "to", "a", "of", "my", "you", "was", "in",
    "that", "it", "we", "she", "he", "they", "for", "on", "be"
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.facecolor":   LIGHT,
    "figure.facecolor": "white",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "axes.titleweight": "bold",
})

# =============================================================================
# HELPERS
# =============================================================================

def load_vocab(story_name):
    """Load word list for a story from raw_text.pkl DataSequence."""
    import pickle
    with open(RAW_TEXT_PKL, "rb") as f:
        data = pickle.load(f)
    if story_name not in data:
        raise KeyError(f"Story '{story_name}' not in raw_text.pkl. "
                       f"Available: {list(data.keys())}")
    return data[story_name].data   # list of words, len == SHAP feature dim


def load_shap_dir(shap_dir, vocab):
    """
    Load all shap_voxel*.npy files in a directory.
    Returns:
        shap_matrix : (n_voxels, n_words)  — raw SHAP values (shape[1] of each file)
        voxel_ids   : list of int
        labels      : list of str  (word labels, trimmed to actual feature count)
    """
    files = sorted(
        [f for f in os.listdir(shap_dir) if f.startswith("shap_voxel") and f.endswith(".npy")],
        key=lambda x: int(re.search(r"\d+", x).group())
    )
    if not files:
        raise FileNotFoundError(f"No shap_voxel*.npy files in {shap_dir}")

    rows, voxel_ids = [], []
    for fname in files:
        arr = np.load(os.path.join(shap_dir, fname))   # shape (1, n_features)
        rows.append(arr.flatten())
        voxel_ids.append(int(re.search(r"\d+", fname).group()))

    shap_matrix = np.array(rows)   # (n_voxels, n_features)
    n_feat = shap_matrix.shape[1]

    # Align vocab: SHAP features may be a subset if delays were stacked
    # (e.g. 697 base words × 4 delays = 2788 → take first 697 unique words)
    if n_feat <= len(vocab):
        labels = vocab[:n_feat]
    else:
        # Delayed features: repeat vocab across delay blocks
        repeats = (n_feat + len(vocab) - 1) // len(vocab)
        labels = (vocab * repeats)[:n_feat]

    return shap_matrix, voxel_ids, labels


def top_words_global(shap_matrix, labels, n=30):
    """Return top-n words by mean |SHAP| across voxels."""
    mean_abs = np.abs(shap_matrix).mean(axis=0)
    # Deduplicate: if delayed features, aggregate by unique word
    word_scores = {}
    for word, score in zip(labels, mean_abs):
        word = word.strip().lower()
        if word == "" or word in STOPWORDS:
            continue
        word_scores[word] = word_scores.get(word, 0) + score
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    return zip(*sorted_words[:n])   # words, scores


# =============================================================================
# PLOT 1 — Global Word Importance (mean |SHAP| across voxels)
# =============================================================================

def plot_global_importance(shap_dirs, labels_map, top_n=25, save=True, voxel_ccs_map=None):
    n_panels = len(shap_dirs)
    ncols    = 2
    nrows    = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 10 * nrows),
                             facecolor="white", squeeze=False)
    axes = axes.flatten()

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    fig.suptitle("Global Word Importance  ·  Mean |SHAP| Across Voxels",
                 fontsize=15, fontweight="bold", y=1.01)

    for ax, (title, shap_dir) in zip(axes, shap_dirs.items()):
        story_key = labels_map[title]
        try:
            vocab = load_vocab(story_key)
            shap_matrix, voxel_ids, feat_labels = load_shap_dir(shap_dir, vocab)
        except Exception as e:
            ax.text(0.5, 0.5, str(e), ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue

        words, scores = top_words_global(shap_matrix, feat_labels, n=top_n)
        words, scores = list(words)[::-1], list(scores)[::-1]

        colors = [BLUE if s >= 0 else RED for s in scores]
        bars = ax.barh(words, scores, color=colors, edgecolor="white",
                       linewidth=0.5, height=0.7)

        for bar, score in zip(bars, scores):
            ax.text(score + max(scores) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"+{score:.4f}", va="center", ha="left", fontsize=7.5, color=GRAY)

        ax.set_xlabel("Mean |SHAP| (mean activation shift)", fontsize=10)
        ax.set_facecolor(LIGHT)
        ax.axvline(0, color="black", linewidth=0.8)

        # ── CC annotation ──────────────────────────────────────────────────
        cc_info = ""
        if voxel_ccs_map and title in voxel_ccs_map:
            cc_arr     = voxel_ccs_map[title]
            voxel_ccs  = cc_arr[voxel_ids]          # CCs for these voxels only
            median_cc  = np.median(voxel_ccs)
            mean_cc    = np.mean(voxel_ccs)
            cc_info    = f"voxel CCs:  median = {median_cc:.3f},  mean = {mean_cc:.3f}"

        ax.set_title(
            f"{title}  ({len(voxel_ids)} voxels)\n{cc_info}",
            fontsize=11, fontweight="bold"
        )

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "plot1_global_word_importance.pdf")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved: {path}")
    return fig


# =============================================================================
# PLOT 2 — Voxel × Word Heatmap
# =============================================================================

def plot_voxel_word_heatmap(shap_dir, story_key, title, top_n_words=40, save=True):
    """
    Heatmap: rows = voxels, columns = top words by max |SHAP|.
    Color encodes signed SHAP value (blue=positive, red=negative).
    """
    vocab = load_vocab(story_key)
    shap_matrix, voxel_ids, feat_labels = load_shap_dir(shap_dir, vocab)

    # Select top words by max |SHAP| across any voxel
    max_abs = np.abs(shap_matrix).max(axis=0)
    word_score = {}
    for i, (word, score) in enumerate(zip(feat_labels, max_abs)):
        word = word.strip().lower()
        if word == "" or word in STOPWORDS:
            continue
        if word not in word_score or score > word_score[word][1]:
            word_score[word] = (i, score)

    top_items   = sorted(word_score.items(), key=lambda x: x[1][1], reverse=True)[:top_n_words]
    top_indices = [v[0] for _, v in top_items]
    top_words_l = [w for w, _ in top_items]

    sub_matrix = shap_matrix[:, top_indices]   # (n_voxels, top_n_words)

    vmax = np.abs(sub_matrix).max()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(max(12, top_n_words * 0.35),
                                    max(4, len(voxel_ids) * 0.55)))
    im = ax.imshow(sub_matrix, aspect="auto", cmap="RdBu_r", norm=norm)

    ax.set_xticks(range(len(top_words_l)))
    ax.set_xticklabels(top_words_l, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(voxel_ids)))
    ax.set_yticklabels([f"Voxel {v}" for v in voxel_ids], fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("SHAP value (activation shift)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(f"Voxel × Word SHAP Heatmap  ·  {title}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Word", fontsize=10)
    ax.set_ylabel("Voxel", fontsize=10)

    # Vertical separator lines
    for x in range(len(top_words_l)):
        ax.axvline(x - 0.5, color="white", linewidth=0.4)
    for y in range(len(voxel_ids)):
        ax.axhline(y - 0.5, color="white", linewidth=0.4)

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, f"plot2_heatmap_{title.replace(' ', '_')}.pdf")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved: {path}")
    return fig


# =============================================================================
# PLOT 3 — CC Distribution Across Embedding Types
# =============================================================================

def plot_cc_comparison(save=True):
    """
    Violin + strip plot comparing CC distributions across all embedding types
    and both subjects.
    """
    # Map display label -> filename pattern
    embedding_map = {
        "BoW\nS2":        "cc_bow_final_subject2.npy",          # adjust if name differs
        "Word2Vec\nS2":   "cc_w2v_subject2.npy",
        "GloVe\nS2":      "cc_subject2_glove.npy",
        "BERT PT\nS2":    "cc_bert_pt_subject2.npy",
        "BERT FT\nS2":    "cc_bert_ft_subject2.npy",
        "BERT LoRA\nS2":  "cc_bert_lora_subject2.npy",
        "BoW\nS3":        "cc_bow_final_subject3.npy",
        "Word2Vec\nS3":   "cc_w2v_subject3.npy",
        "GloVe\nS3":      "cc_subject3_glove.npy",
        "BERT PT\nS3":    "cc_bert_pt_subject3.npy",
        "BERT FT\nS3":    "cc_bert_ft_subject3.npy",
        "BERT LoRA\nS3":  "cc_bert_lora_subject3.npy",
    }

    labels_found, data_found, colors_found = [], [], []
    color_map = {
        "S2": BLUE,
        "S3": TEAL,
    }

    for label, fname in embedding_map.items():
        fpath = os.path.join(CC_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  [skip] {fname} not found")
            continue
        cc = np.load(fpath)
        cc = cc[np.isfinite(cc)]
        labels_found.append(label)
        data_found.append(cc)
        subj = "S2" if "S2" in label else "S3"
        colors_found.append(color_map[subj])

    if not data_found:
        print("No CC files found — check CC_DIR path.")
        return

    fig, ax = plt.subplots(figsize=(max(10, len(data_found) * 1.1), 5))

    parts = ax.violinplot(data_found, positions=range(len(data_found)),
                          showmedians=True, showextrema=False, widths=0.7)

    for i, (pc, color) in enumerate(zip(parts["bodies"], colors_found)):
        pc.set_facecolor(color)
        pc.set_alpha(0.55)
        pc.set_edgecolor("white")

    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.5)

    # Overlay median dots
    for i, (cc, color) in enumerate(zip(data_found, colors_found)):
        ax.scatter(i, np.median(cc), color=color, s=40, zorder=5, edgecolors="white", linewidth=0.8)
        ax.scatter(i, np.percentile(cc, 95), color=color, marker="^", s=25,
                   zorder=5, edgecolors="white", linewidth=0.5, alpha=0.8)

    ax.set_xticks(range(len(labels_found)))
    ax.set_xticklabels(labels_found, fontsize=8.5)
    ax.set_ylabel("Correlation Coefficient (CC)", fontsize=10)
    ax.set_title("CC Distribution Across Embedding Types & Subjects\n"
                 "● = median,  ▲ = 95th percentile", fontsize=12, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.axhline(0.2, color=RED, linewidth=0.8, linestyle=":", alpha=0.7, label="CC = 0.2 threshold")
    ax.legend(fontsize=8, loc="upper left")

    # Subject divider
    s2_count = sum(1 for l in labels_found if "S2" in l)
    if s2_count > 0 and s2_count < len(labels_found):
        ax.axvline(s2_count - 0.5, color=GRAY, linewidth=1.2, linestyle="--", alpha=0.6)
        ymin, ymax = ax.get_ylim()
        ax.text(s2_count / 2 - 0.5, ymax * 0.88,
                "Subject 2", ha="center", fontsize=9, color=BLUE, fontweight="bold")
        ax.text((s2_count + len(labels_found)) / 2 - 0.5, ymax * 0.88,
                "Subject 3", ha="center", fontsize=9, color=TEAL, fontweight="bold")

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "plot3_cc_comparison.pdf")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved: {path}")
    return fig


# =============================================================================
# PLOT 4 — Positive vs Negative Word Drivers Per Voxel
# =============================================================================

def plot_signed_drivers(shap_dirs, labels_map, top_n=10, save=True):
    """
    For each voxel in each story/subject combo, show top positive and
    negative SHAP words as a diverging bar chart.
    Panels: one row per (story, subject), one column per voxel.
    """
    for title, shap_dir in shap_dirs.items():
        story_key = labels_map[title]
        try:
            vocab = load_vocab(story_key)
            shap_matrix, voxel_ids, feat_labels = load_shap_dir(shap_dir, vocab)
        except Exception as e:
            print(f"[{title}] Error: {e}")
            continue

        n_voxels = len(voxel_ids)
        ncols = 2
        nrows = int(np.ceil(n_voxels / ncols))

        fig, axes = plt.subplots(nrows, ncols,
                                figsize=(11, 5.5 * nrows),
                                sharey=False)

        axes = np.array(axes).flatten()

        fig.suptitle(f"Top Positive & Negative Word Drivers  ·  {title}",
                     fontsize=14, fontweight="bold", y=1.01)

        for ax, voxel_id, shap_row in zip(axes, voxel_ids, shap_matrix):
            # Aggregate by unique word
            word_shap = {}
            for word, val in zip(feat_labels, shap_row):
                word = word.strip().lower()
                if word == "" or word in STOPWORDS:
                    continue
                word_shap[word] = word_shap.get(word, 0) + val

            sorted_items = sorted(word_shap.items(), key=lambda x: x[1])
            neg_words = [w for w, v in sorted_items[:top_n] if v < 0]
            neg_vals  = [v for _, v in sorted_items[:top_n] if v < 0]
            pos_items = sorted(word_shap.items(), key=lambda x: x[1], reverse=True)
            pos_words = [w for w, v in pos_items[:top_n] if v > 0]
            pos_vals  = [v for _, v in pos_items[:top_n] if v > 0]

            all_words = neg_words + pos_words
            all_vals  = neg_vals  + pos_vals
            colors    = [RED] * len(neg_words) + [BLUE] * len(pos_words)

            if not all_words:
                ax.text(0.5, 0.5, "No data", ha="center", transform=ax.transAxes)
                continue

            max_val = max(abs(v) for v in all_vals)

            ax.barh(all_words, all_vals, color=colors, edgecolor="white",
                    linewidth=0.4, height=0.7)
            ax.axvline(0, color="black", linewidth=0.9)
            ax.set_xlim(-max_val, max_val)
            ax.set_title(f"Voxel {voxel_id}", fontsize=10, fontweight="bold")
            ax.set_xlabel("SHAP value", fontsize=9)
            ax.set_facecolor(LIGHT)

            # Annotate values
            for bar_y, val in zip(all_words, all_vals):
                offset = max(abs(v) for v in all_vals) * 0.02
                ha = "left" if val >= 0 else "right"
                ax.text(val + (offset if val >= 0 else -offset), bar_y,
                        f"{val:+.4f}", va="center", ha=ha, fontsize=7, color=GRAY)
        
        for ax in axes[n_voxels:]:
            ax.set_visible(False)

        plt.tight_layout()
        if save:
            fname = f"plot4_signed_drivers_{title.replace(' ', '_')}.pdf"
            path  = os.path.join(OUTPUT_DIR, fname)
            fig.savefig(path, bbox_inches="tight", dpi=150)
            print(f"Saved: {path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    SHAP_DIRS = {
        "Sweetaspie · S2":                         os.path.join(SHAP_BASE_DIR, "sweetaspie_s2"),
        "Sweetaspie · S3":                         os.path.join(SHAP_BASE_DIR, "sweetaspie_s3"),
        "The Triangle Shirtwaist Connection · S2": os.path.join(SHAP_BASE_DIR, "thetriangleshirtwaistconnection_s2"),
        "The Triangle Shirtwaist Connection · S3": os.path.join(SHAP_BASE_DIR, "thetriangleshirtwaistconnection_s3"),
    }

    STORY_KEYS = {
        "Sweetaspie · S2":                         "sweetaspie",
        "Sweetaspie · S3":                         "sweetaspie",
        "The Triangle Shirtwaist Connection · S2": "thetriangleshirtwaistconnection",
        "The Triangle Shirtwaist Connection · S3": "thetriangleshirtwaistconnection",
    }

    voxel_ccs_map = {
        "Sweetaspie · S2":                         np.load(os.path.join(CC_DIR, "cc_bert_lora_subject2.npy")),
        "Sweetaspie · S3":                         np.load(os.path.join(CC_DIR, "cc_bert_lora_subject3.npy")),
        "The Triangle Shirtwaist Connection · S2": np.load(os.path.join(CC_DIR, "cc_bert_lora_subject2.npy")),
        "The Triangle Shirtwaist Connection · S3": np.load(os.path.join(CC_DIR, "cc_bert_lora_subject3.npy")),
    }

    SHAP_DIRS = {k: v for k, v in SHAP_DIRS.items() if os.path.isdir(v)}
    print(f"Found {len(SHAP_DIRS)} SHAP directories: {list(SHAP_DIRS.keys())}")

    print("\n── Plot 1: Global Word Importance ──")
    plot_global_importance(SHAP_DIRS, STORY_KEYS, top_n=25, voxel_ccs_map=voxel_ccs_map)

    print("\n── Plot 2: Voxel × Word Heatmaps ──")
    for title, shap_dir in SHAP_DIRS.items():
        story_key = STORY_KEYS[title]
        plot_voxel_word_heatmap(shap_dir, story_key, title, top_n_words=40)

    print("\n── Plot 3: CC Distribution Comparison ──")
    plot_cc_comparison()

    print("\n── Plot 4: Signed Drivers per Voxel ──")
    plot_signed_drivers(SHAP_DIRS, STORY_KEYS, top_n=10)

    print(f"\n✓ All plots saved to: {OUTPUT_DIR}")

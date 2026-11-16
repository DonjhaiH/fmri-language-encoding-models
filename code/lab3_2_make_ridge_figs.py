"""Generate ridge regression result figures for the lab3.2 report."""
import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = pathlib.Path(__file__).parent.parent
FIG_DIR = ROOT / "docs" / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = ["pt", "ft", "lora"]
LABELS   = {"pt": "Pre-trained", "ft": "MLM FT", "lora": "LoRA FT"}
COLORS   = {"pt": "#4C72B0", "ft": "#DD8452", "lora": "#55A868"}

# ── load data ──────────────────────────────────────────────────────────────
data = {}
for v in VARIANTS:
    p = ROOT / f"ridge_bert_{v}_subject2_summary.json"
    with open(p) as f:
        data[v] = json.load(f)

stories = list(data["pt"]["per_story"].keys())
stories_sorted = sorted(stories)   # alphabetical for consistent ordering

# ── Figure 1: per-story mean r (grouped bar) ───────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4.5))

n = len(stories_sorted)
x = np.arange(n)
width = 0.26

for i, v in enumerate(VARIANTS):
    vals = [data[v]["per_story"][s]["mean_r"] for s in stories_sorted]
    bars = ax.bar(x + (i - 1) * width, vals, width,
                  label=LABELS[v], color=COLORS[v], alpha=0.88, edgecolor="white", linewidth=0.4)

# short story labels
short = [s[:14] + "…" if len(s) > 14 else s for s in stories_sorted]
ax.set_xticks(x)
ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7.5)
ax.set_ylabel("Mean Pearson $r$", fontsize=10)
ax.set_title("Per-story mean Pearson $r$ — Subject 2", fontsize=11)
ax.legend(fontsize=9, framealpha=0.8)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
ax.set_ylim(0, 0.052)
ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "ridge_per_story_mean_r.pdf", dpi=150, bbox_inches="tight")
fig.savefig(FIG_DIR / "ridge_per_story_mean_r.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: ridge_per_story_mean_r")

# ── Figure 2: global mean & median r (dot + error bar style) ──────────────
fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=False)

for ax, metric, ylabel in zip(
    axes,
    ["global_mean_r", "global_median_r"],
    ["Global mean Pearson $r$", "Global median Pearson $r$"],
):
    vals  = [data[v][metric] for v in VARIANTS]
    xlabs = [LABELS[v] for v in VARIANTS]
    cols  = [COLORS[v] for v in VARIANTS]
    bars  = ax.bar(xlabs, vals, color=cols, alpha=0.88, edgecolor="white", width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.0002,
                f"{val:.5f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8.5)

fig.suptitle("Global ridge regression performance — Subject 2", fontsize=10, y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "ridge_global_summary.pdf", dpi=150, bbox_inches="tight")
fig.savefig(FIG_DIR / "ridge_global_summary.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: ridge_global_summary")

# ── Figure 3: frac_positive across stories (box + strip) ──────────────────
fig, ax = plt.subplots(figsize=(5, 3.5))

frac_data = {v: [data[v]["per_story"][s]["frac_positive"] for s in stories_sorted]
             for v in VARIANTS}

positions = [1, 2, 3]
bp = ax.boxplot(
    [frac_data[v] for v in VARIANTS],
    positions=positions,
    widths=0.4,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=1.5),
    whiskerprops=dict(linewidth=1),
    capprops=dict(linewidth=1),
    flierprops=dict(marker="o", markersize=4, alpha=0.5),
)
for patch, v in zip(bp["boxes"], VARIANTS):
    patch.set_facecolor(COLORS[v])
    patch.set_alpha(0.75)

# overlay individual points
rng = np.random.default_rng(42)
for pos, v in zip(positions, VARIANTS):
    jitter = rng.uniform(-0.12, 0.12, size=len(frac_data[v]))
    ax.scatter(pos + jitter, frac_data[v],
               color=COLORS[v], s=22, alpha=0.65, zorder=3, edgecolors="white", linewidths=0.3)

ax.axhline(0.5, color="grey", linewidth=0.9, linestyle="--", alpha=0.6, label="chance (0.5)")
ax.set_xticks(positions)
ax.set_xticklabels([LABELS[v] for v in VARIANTS], fontsize=9)
ax.set_ylabel("Fraction of voxels with $r > 0$", fontsize=9)
ax.set_title("Positive-correlation voxel fraction — Subject 2", fontsize=10)
ax.legend(fontsize=8, framealpha=0.7)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "ridge_frac_positive.pdf", dpi=150, bbox_inches="tight")
fig.savefig(FIG_DIR / "ridge_frac_positive.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: ridge_frac_positive")

print(f"\nAll figures written to {FIG_DIR}")

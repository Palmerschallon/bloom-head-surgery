#!/usr/bin/env python3
"""
Generate publication-quality cross-scale BOS-sink heatmap figure
for the BLOOM model family diagnostic paper.

Shows per-head BOS attention mass across all layers for 4 BLOOM variants,
revealing the characteristic sick band that scales with architecture.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects
from matplotlib.gridspec import GridSpec

# ── Load data ──────────────────────────────────────────────────────────
DATA_PATH = "/ember/00_Koan_Engine/bloom_family_diagnostic/bloom_family_20260217_135922.json"
with open(DATA_PATH) as f:
    family = json.load(f)

# Model display order (small → large)
MODEL_ORDER = ["BLOOM-560m", "BLOOM-1b7", "BLOOM-3b", "BLOOM-7b1"]

# ── Extract BOS mass matrices ─────────────────────────────────────────
matrices = {}
sick_pcts = {}

for model_name in MODEL_ORDER:
    model = family["models"][model_name]
    arch = model["architecture"]
    n_layers = arch["n_layers"]
    n_heads = arch["n_heads"]
    full = model["attention"]["full"]
    sick_pcts[model_name] = model["attention"]["sick_pct"]

    mat = np.zeros((n_layers, n_heads))
    for layer_str, heads in full.items():
        layer_idx = int(layer_str)
        for head_str, head_data in heads.items():
            head_idx = int(head_str)
            mat[layer_idx, head_idx] = head_data["bos_mass"]

    matrices[model_name] = mat

# ── Figure setup ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 11), facecolor='white')

# Use GridSpec for precise layout control with shared colorbar
gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.05],
              wspace=0.28, hspace=0.42,
              left=0.08, right=0.90, top=0.89, bottom=0.06)

# Diverging colormap centered at 0.5: blue (healthy) → red (collapsed)
cmap = plt.cm.RdBu_r
norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)

# Subplot positions in the 2x2 grid
positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

# Parameter counts for subtitle
param_labels = {
    "BLOOM-560m": "560M params, 24L × 16H",
    "BLOOM-1b7": "1.7B params, 24L × 16H",
    "BLOOM-3b": "3B params, 30L × 32H",
    "BLOOM-7b1": "7.1B params, 30L × 32H",
}

axes = []
for idx, model_name in enumerate(MODEL_ORDER):
    row, col = positions[idx]
    ax = fig.add_subplot(gs[row, col])
    axes.append(ax)

    mat = matrices[model_name]
    n_layers, n_heads = mat.shape
    sick_pct = sick_pcts[model_name]

    # Count BOS-sink heads (bos_mass > 0.5)
    n_sick = int((mat > 0.5).sum())
    n_total = n_layers * n_heads

    im = ax.imshow(mat, aspect='auto', cmap=cmap, norm=norm,
                   interpolation='nearest', origin='lower')

    # Titles - use two lines: bold model name, then details in normal weight
    ax.set_title(f"{model_name}\n{param_labels[model_name]}  |  "
                 f"BOS-sink: {sick_pct:.1f}% ({n_sick}/{n_total} heads)",
                 fontsize=9, fontweight='bold', pad=10,
                 linespacing=1.4)

    # Axis labels
    ax.set_xlabel("Head Index", fontsize=9)
    ax.set_ylabel("Layer", fontsize=9)

    # Tick configuration
    if n_heads == 16:
        ax.set_xticks(range(0, 16, 2))
        ax.set_xticks(range(16), minor=True)
    else:
        ax.set_xticks(range(0, 32, 4))
        ax.set_xticks(range(32), minor=True)

    if n_layers == 24:
        ax.set_yticks(range(0, 24, 4))
        ax.set_yticks(range(24), minor=True)
    else:
        ax.set_yticks(range(0, 30, 5))
        ax.set_yticks(range(30), minor=True)

    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', length=0)

    # Draw thin grid lines on minor ticks for readability
    ax.grid(which='minor', color='gray', linewidth=0.15, alpha=0.3)

    # Panel label (a, b, c, d) inside lower-left corner of the heatmap
    panel_label = chr(ord('a') + idx)
    ax.text(0.02, 0.02, f"({panel_label})", transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='bottom', ha='left',
            color='white', path_effects=[
                matplotlib.patheffects.withStroke(linewidth=2.5, foreground='black')])

# ── Shared colorbar ───────────────────────────────────────────────────
cbar_ax = fig.add_subplot(gs[:, 2])
cbar = fig.colorbar(im, cax=cbar_ax, label="BOS Attention Mass")
cbar.ax.tick_params(labelsize=9)
cbar.set_label("BOS Attention Mass", fontsize=10)

# Mark the 0.5 threshold
cbar.ax.axhline(y=0.5, color='black', linewidth=1.0, linestyle='--', alpha=0.7)
cbar.ax.text(1.6, 0.5, "threshold", transform=cbar.ax.get_yaxis_transform(),
             fontsize=7, va='center', ha='left', alpha=0.7)

# ── Suptitle ──────────────────────────────────────────────────────────
fig.suptitle("Cross-Scale BOS-Sink Band Pattern in the BLOOM Model Family",
             fontsize=14, fontweight='bold', y=0.96)

# ── Save ──────────────────────────────────────────────────────────────
out_path = "/ember/00_Koan_Engine/paper/figure_cross_scale_heatmap.png"
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Saved: {out_path}")

# Print summary stats
print("\n=== Summary ===")
for model_name in MODEL_ORDER:
    mat = matrices[model_name]
    n_layers, n_heads = mat.shape
    n_sick = int((mat > 0.5).sum())
    n_total = n_layers * n_heads
    max_bos = mat.max()
    mean_bos = mat.mean()

    # Find the sick band (layers where >50% of heads are BOS-sink)
    sick_layers = []
    for layer in range(n_layers):
        row = mat[layer]
        if (row > 0.5).sum() > n_heads * 0.25:
            sick_layers.append(layer)

    band_str = f"L{min(sick_layers)}-L{max(sick_layers)}" if sick_layers else "none"
    print(f"  {model_name}: {n_sick}/{n_total} sick ({sick_pcts[model_name]:.1f}%), "
          f"max_bos={max_bos:.3f}, mean_bos={mean_bos:.3f}, band={band_str}")

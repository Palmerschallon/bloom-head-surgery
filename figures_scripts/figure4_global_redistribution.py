#!/usr/bin/env python3
"""
Figure 4: Global Attention Redistribution Map

Full 24×16 heatmap showing BOS mass change (signed delta) from stock BLOOM-1b7
to C4 baseline epoch_015 for EVERY head. Surgical zone outlined in black.

This is the figure that shows the redistribution is everywhere, not contained.
"""

import json
import sys
import torch
import gc
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

DIAG_TEXT = (
    "The container holds the boundary. The boundary defines the edge. "
    "The edge separates inside from outside. Inside and outside depend "
    "on the container. Memory persists through structure, not content. "
    "What remains after deletion is the shape of what was deleted."
)

# Surgical zone: layers 5-22, heads 9-15
SURGICAL_LAYERS = set(range(5, 23))  # L5-L22
SURGICAL_HEADS = set(range(9, 16))   # H9-H15

# Reinitialized heads per layer (from surgery map)
REINIT = {
    5:  {14, 15},
    6:  {9, 10, 11, 12, 13, 14, 15},
    7:  {9, 10, 11, 12, 13, 14},
    8:  {11, 12, 13, 14, 15},
    9:  {9, 10, 11, 12, 13, 14, 15},
    10: {9, 10, 11, 12, 13, 14, 15},
    11: {10, 11, 12, 13, 14, 15},
    12: {9, 10, 11, 12, 13, 14, 15},
    13: {9, 10, 11, 12, 13, 14, 15},
    14: {9, 10, 11, 13, 14, 15},
    15: {9, 11, 12, 14, 15},
    16: {10, 11, 12, 13, 14, 15},
    17: {9, 10, 11, 12, 13, 14},
    18: {9, 10, 11, 12, 13, 14, 15},
    19: {9, 10, 11, 12, 13, 14, 15},
    20: {9, 10, 11, 12, 13, 14, 15},
    21: {9, 10, 11, 12, 13},
    22: {9, 10, 11, 13, 14},
}


def measure_all_heads(model_path, device="cuda"):
    """Load checkpoint, measure BOS mass for ALL 24×16 heads."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    tokens = tokenizer(DIAG_TEXT, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**tokens, output_attentions=True)

    results = {}
    for li in range(len(out.attentions)):
        attn = out.attentions[li][0].float()
        for hi in range(attn.shape[0]):
            ha = attn[hi]
            bos = ha[:, 0].mean().item()
            results[(li, hi)] = bos

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np

    base = Path("/ember/progressive_checkpoints/bloom1b7")

    # Find C4 baseline run
    runs = sorted(base.glob("baseline_c4_*"))
    if not runs:
        print("No baseline_c4 runs found.")
        return
    run_dir = runs[-1]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load stock and final epoch
    print("Loading stock BLOOM-1b7...")
    stock = measure_all_heads("bigscience/bloom-1b7", device)

    # Use epoch_015 (the last epoch before final)
    epoch_path = run_dir / "epoch_015"
    if not epoch_path.exists():
        epoch_path = run_dir / "final"
    print(f"Loading {epoch_path.name}...")
    trained = measure_all_heads(str(epoch_path), device)

    n_layers = 24
    n_heads = 16

    # Build delta matrix
    delta_matrix = np.zeros((n_layers, n_heads))
    for li in range(n_layers):
        for hi in range(n_heads):
            s = stock.get((li, hi), 0)
            t = trained.get((li, hi), 0)
            delta_matrix[li, hi] = t - s

    # ── Figure 4: Full 24×16 redistribution map ──
    fig, ax = plt.subplots(figsize=(14, 12))

    # Use diverging colormap, centered at 0
    vmax = max(abs(delta_matrix.min()), abs(delta_matrix.max()))
    vmax = min(vmax, 0.4)  # cap for readability
    im = ax.imshow(delta_matrix, cmap='RdBu_r', aspect='auto',
                   vmin=-vmax, vmax=vmax, interpolation='nearest')

    # Draw surgical zone outline (L5-L22, H9-H15)
    # Rectangle: (x, y) is top-left corner in data coords
    # x = head index (col), y = layer index (row)
    rect = patches.Rectangle(
        (8.5, 4.5),   # x=H9-0.5, y=L5-0.5
        7.0,           # width: H9 to H15 = 7 heads
        18.0,          # height: L5 to L22 = 18 layers
        linewidth=2.5, edgecolor='black', facecolor='none',
        linestyle='-', label='Surgical zone (H9-H15, L5-L22)'
    )
    ax.add_patch(rect)

    # Mark reinitialized heads with small dots
    for li, heads in REINIT.items():
        for hi in heads:
            ax.plot(hi, li, 'k.', markersize=3, alpha=0.4)

    # Annotate cells where |delta| > 0.15 with values
    for li in range(n_layers):
        for hi in range(n_heads):
            d = delta_matrix[li, hi]
            if abs(d) > 0.15:
                color = 'white' if abs(d) > 0.25 else 'black'
                ax.text(hi, li, f"{d:+.2f}", ha='center', va='center',
                        fontsize=5.5, color=color, fontweight='bold')

    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=9)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{l}" for l in range(n_layers)], fontsize=9)
    ax.set_xlabel("Head Index", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(
        "Global Attention Redistribution: BOS Mass Change (Stock → C4 Epoch 15)\n"
        "Black box = surgical zone. Color outside box = untouched heads affected by surgery.",
        fontsize=11, fontweight='bold'
    )

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label="BOS mass delta (epoch 15 − stock)")

    # Add summary stats as text
    n_outside_drifting = 0
    n_outside_total = 0
    worst_outside = ("", 0)
    for li in range(n_layers):
        for hi in range(n_heads):
            in_zone = (li in SURGICAL_LAYERS) and (hi in SURGICAL_HEADS)
            if not in_zone:
                n_outside_total += 1
                d = abs(delta_matrix[li, hi])
                if d > 0.05:
                    n_outside_drifting += 1
                if d > abs(worst_outside[1]):
                    worst_outside = (f"L{li}H{hi}", delta_matrix[li, hi])

    n_inside_total = n_layers * n_heads - n_outside_total
    # Recount inside using frozen-only (non-reinit) heads
    n_inside_drifting = 0
    for li in SURGICAL_LAYERS:
        for hi in SURGICAL_HEADS:
            if li in REINIT and hi in REINIT[li]:
                continue  # skip reinitialized — they're supposed to change
            d = abs(delta_matrix[li, hi])
            if d > 0.05:
                n_inside_drifting += 1

    stats_text = (
        f"Outside surgical zone: {n_outside_drifting}/{n_outside_total} heads drifting (>0.05)\n"
        f"Inside zone (frozen only): {n_inside_drifting} heads drifting\n"
        f"Worst outside-zone: {worst_outside[0]} ({worst_outside[1]:+.3f})"
    )
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    out_path = fig_dir / "global_redistribution_map.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"\nSaved: {out_path}")

    # Also save the raw delta matrix for the paper
    raw_path = fig_dir / "global_redistribution_data.json"
    raw = {
        "description": "BOS mass delta (C4 epoch 15 - stock) for all 24x16 heads",
        "n_layers": n_layers,
        "n_heads": n_heads,
        "surgical_zone": {"layers": "5-22", "heads": "9-15"},
        "outside_drifting": n_outside_drifting,
        "outside_total": n_outside_total,
        "inside_frozen_drifting": n_inside_drifting,
        "worst_outside": {"head": worst_outside[0], "delta": round(worst_outside[1], 6)},
        "delta_matrix": [[round(float(delta_matrix[li, hi]), 6) for hi in range(n_heads)]
                         for li in range(n_layers)],
    }
    with open(raw_path, 'w') as f:
        json.dump(raw, f, indent=2)
    print(f"Saved: {raw_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"GLOBAL REDISTRIBUTION SUMMARY")
    print(f"{'='*60}")
    print(f"  Outside surgical zone: {n_outside_drifting}/{n_outside_total} drifting ({100*n_outside_drifting/n_outside_total:.1f}%)")
    print(f"  Inside zone (frozen):  {n_inside_drifting} drifting")
    print(f"  Worst outside-zone:    {worst_outside[0]} = {worst_outside[1]:+.4f}")
    print(f"  Max |delta| anywhere:  {abs(delta_matrix).max():.4f}")
    print(f"  Mean |delta| outside:  {np.mean([abs(delta_matrix[li,hi]) for li in range(n_layers) for hi in range(n_heads) if not (li in SURGICAL_LAYERS and hi in SURGICAL_HEADS)]):.4f}")
    print(f"  Mean |delta| inside frozen: {np.mean([abs(delta_matrix[li,hi]) for li in SURGICAL_LAYERS for hi in SURGICAL_HEADS if li in REINIT and hi not in REINIT[li]]):.4f}")


if __name__ == "__main__":
    main()

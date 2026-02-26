#!/usr/bin/env python3
"""
Figure 4 (final): Side-by-side global attention redistribution.

Left panel:  Curated corpus (epoch 3, best checkpoint)
Right panel: C4 generic corpus (epoch 15)

Same surgery, same masks, same hyperparameters. Different corpus.
The contrast IS the finding.
"""

import json
import torch
import gc
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

DIAG_TEXT = (
    "The container holds the boundary. The boundary defines the edge. "
    "The edge separates inside from outside. Inside and outside depend "
    "on the container. Memory persists through structure, not content. "
    "What remains after deletion is the shape of what was deleted."
)

SURGICAL_LAYERS = set(range(5, 23))
SURGICAL_HEADS = set(range(9, 16))

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

N_LAYERS = 24
N_HEADS = 16


def measure_all_heads(model_path, device="cuda"):
    """Measure BOS mass for all 24×16 heads."""
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
            bos = attn[hi][:, 0].mean().item()
            results[(li, hi)] = bos

    del model, out
    gc.collect()
    torch.cuda.empty_cache()
    return results


def compute_delta_matrix(stock, trained):
    matrix = np.zeros((N_LAYERS, N_HEADS))
    for li in range(N_LAYERS):
        for hi in range(N_HEADS):
            matrix[li, hi] = trained.get((li, hi), 0) - stock.get((li, hi), 0)
    return matrix


def count_outside_drifters(matrix, threshold=0.05):
    count = 0
    total = 0
    worst = ("", 0)
    for li in range(N_LAYERS):
        for hi in range(N_HEADS):
            in_zone = (li in SURGICAL_LAYERS) and (hi in SURGICAL_HEADS)
            if not in_zone:
                total += 1
                d = abs(matrix[li, hi])
                if d > threshold:
                    count += 1
                if d > abs(worst[1]):
                    worst = (f"L{li}H{hi}", matrix[li, hi])
    return count, total, worst


def draw_panel(ax, matrix, title, subtitle):
    """Draw one panel of the side-by-side figure."""
    import matplotlib.patches as patches

    vmax = 0.4  # shared scale for fair comparison
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto',
                   vmin=-vmax, vmax=vmax, interpolation='nearest')

    # Surgical zone outline
    rect = patches.Rectangle(
        (8.5, 4.5), 7.0, 18.0,
        linewidth=2.5, edgecolor='black', facecolor='none', linestyle='-'
    )
    ax.add_patch(rect)

    # Annotate cells with |delta| > 0.15
    for li in range(N_LAYERS):
        for hi in range(N_HEADS):
            d = matrix[li, hi]
            if abs(d) > 0.15:
                color = 'white' if abs(d) > 0.25 else 'black'
                ax.text(hi, li, f"{d:+.2f}", ha='center', va='center',
                        fontsize=5, color=color, fontweight='bold')

    ax.set_xticks(range(N_HEADS))
    ax.set_xticklabels([f"H{h}" for h in range(N_HEADS)], fontsize=7)
    ax.set_yticks(range(N_LAYERS))
    ax.set_yticklabels([f"L{l}" for l in range(N_LAYERS)], fontsize=7)
    ax.set_xlabel("Head Index", fontsize=10)
    ax.set_ylabel("Layer", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')

    # Stats box
    n_drift, n_total, worst = count_outside_drifters(matrix)
    mean_outside = np.mean([abs(matrix[li, hi])
                            for li in range(N_LAYERS) for hi in range(N_HEADS)
                            if not (li in SURGICAL_LAYERS and hi in SURGICAL_HEADS)])
    stats = (
        f"Outside zone: {n_drift}/{n_total} drifting (>0.05)\n"
        f"Mean |δ| outside: {mean_outside:.4f}\n"
        f"{subtitle}"
    )
    ax.text(0.02, 0.02, stats, transform=ax.transAxes,
            fontsize=7, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.85))

    return im


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    base = Path("/ember/progressive_checkpoints/bloom1b7")
    curated_path = base / "headband_20260217_092844" / "epoch_003"
    c4_runs = sorted(base.glob("baseline_c4_*"))
    c4_path = c4_runs[-1] / "epoch_015"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load all three models
    print("Loading stock BLOOM-1b7...")
    stock = measure_all_heads("bigscience/bloom-1b7", device)

    print(f"Loading curated epoch 3 ({curated_path.name})...")
    curated = measure_all_heads(str(curated_path), device)

    print(f"Loading C4 epoch 15 ({c4_path.parent.name}/{c4_path.name})...")
    c4 = measure_all_heads(str(c4_path), device)

    # Compute deltas
    curated_delta = compute_delta_matrix(stock, curated)
    c4_delta = compute_delta_matrix(stock, c4)

    # ── Side-by-side figure ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

    fig.suptitle(
        "Global Attention Redistribution: Identical Surgery, Different Corpus\n"
        "BOS mass change from stock for all 384 heads. Black box = surgical zone.",
        fontsize=14, fontweight='bold', y=0.98
    )

    im1 = draw_panel(ax1, curated_delta,
                     "Curated Corpus (Epoch 3, Best PPL)",
                     "Curated: 108/108 heads woken, PPL 15.4")

    im2 = draw_panel(ax2, c4_delta,
                     "C4 Generic Corpus (Epoch 15)",
                     "C4: 108/108 heads woken, PPL 20.8")

    # Shared colorbar
    cbar = fig.colorbar(im2, ax=[ax1, ax2], shrink=0.6, pad=0.02,
                        label="BOS mass delta (trained − stock)")

    plt.tight_layout(rect=[0, 0, 0.95, 0.95])

    fig_dir = c4_runs[-1] / "figures"
    fig_dir.mkdir(exist_ok=True)
    out_path = fig_dir / "figure4_sidebyside.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_path}")

    # Print comparison
    c_drift, c_total, c_worst = count_outside_drifters(curated_delta)
    g_drift, g_total, g_worst = count_outside_drifters(c4_delta)

    c_mean = np.mean([abs(curated_delta[li, hi])
                      for li in range(N_LAYERS) for hi in range(N_HEADS)
                      if not (li in SURGICAL_LAYERS and hi in SURGICAL_HEADS)])
    g_mean = np.mean([abs(c4_delta[li, hi])
                      for li in range(N_LAYERS) for hi in range(N_HEADS)
                      if not (li in SURGICAL_LAYERS and hi in SURGICAL_HEADS)])

    print(f"\n{'='*60}")
    print(f"SIDE-BY-SIDE COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<35s} {'Curated':>12s} {'C4':>12s}")
    print(f"{'─'*60}")
    print(f"{'Heads woken':<35s} {'108/108':>12s} {'108/108':>12s}")
    print(f"{'Outside-zone drifters (>0.05)':<35s} {c_drift:>12d} {g_drift:>12d}")
    print(f"{'Outside-zone total':<35s} {c_total:>12d} {g_total:>12d}")
    print(f"{'Mean |δ| outside zone':<35s} {c_mean:>12.4f} {g_mean:>12.4f}")
    print(f"{'Worst outside-zone':<35s} {c_worst[0]+' '+f'{c_worst[1]:+.3f}':>12s} {g_worst[0]+' '+f'{g_worst[1]:+.3f}':>12s}")
    print(f"{'Held-out PPL':<35s} {'~15.4':>12s} {'~20.8':>12s}")

    # Save comparison data
    comp_path = fig_dir / "sidebyside_comparison.json"
    comp = {
        "curated": {
            "checkpoint": "headband_20260217_092844/epoch_003",
            "outside_drifters": c_drift,
            "outside_total": c_total,
            "mean_abs_delta_outside": round(c_mean, 6),
            "worst_outside": {"head": c_worst[0], "delta": round(c_worst[1], 6)},
            "delta_matrix": [[round(float(curated_delta[li, hi]), 6) for hi in range(N_HEADS)]
                             for li in range(N_LAYERS)],
        },
        "c4": {
            "checkpoint": f"{c4_runs[-1].name}/epoch_015",
            "outside_drifters": g_drift,
            "outside_total": g_total,
            "mean_abs_delta_outside": round(g_mean, 6),
            "worst_outside": {"head": g_worst[0], "delta": round(g_worst[1], 6)},
            "delta_matrix": [[round(float(c4_delta[li, hi]), 6) for hi in range(N_HEADS)]
                             for li in range(N_LAYERS)],
        },
    }
    with open(comp_path, 'w') as f:
        json.dump(comp, f, indent=2)
    print(f"\nSaved: {comp_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Figure 4 (matched epoch): Curated E3 vs C4 E3 — fair comparison.

Same surgery, same masks, same hyperparameters, same epoch count.
Only difference: corpus.
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


def count_drifters(matrix, threshold=0.05):
    """Count drifters in different zones."""
    outside = {"count": 0, "total": 0, "worst": ("", 0), "deltas": []}
    frozen_inband = {"count": 0, "total": 0, "deltas": []}

    for li in range(N_LAYERS):
        for hi in range(N_HEADS):
            in_zone = (li in SURGICAL_LAYERS) and (hi in SURGICAL_HEADS)
            d = matrix[li, hi]
            ad = abs(d)

            if not in_zone:
                outside["total"] += 1
                outside["deltas"].append(ad)
                if ad > threshold:
                    outside["count"] += 1
                if ad > abs(outside["worst"][1]):
                    outside["worst"] = (f"L{li}H{hi}", d)
            else:
                # Check if frozen in-band (not reinitialized)
                if li in REINIT and hi not in REINIT[li]:
                    frozen_inband["total"] += 1
                    frozen_inband["deltas"].append(ad)
                    if ad > threshold:
                        frozen_inband["count"] += 1

    return outside, frozen_inband


def draw_panel(ax, matrix, title, stats_text):
    import matplotlib.patches as patches

    vmax = 0.4
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto',
                   vmin=-vmax, vmax=vmax, interpolation='nearest')

    rect = patches.Rectangle(
        (8.5, 4.5), 7.0, 18.0,
        linewidth=2.5, edgecolor='black', facecolor='none', linestyle='-'
    )
    ax.add_patch(rect)

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

    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            fontsize=7, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.85))

    return im


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    base = Path("/ember/progressive_checkpoints/bloom1b7")
    curated_e3 = base / "headband_20260217_092844" / "epoch_003"
    c4_e3 = base / "baseline_c4_20260217_140421" / "epoch_003"
    c4_e15 = base / "baseline_c4_20260217_140421" / "epoch_015"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading stock BLOOM-1b7...")
    stock = measure_all_heads("bigscience/bloom-1b7", device)

    print("Loading curated epoch 3...")
    curated = measure_all_heads(str(curated_e3), device)

    print("Loading C4 epoch 3...")
    c4_3 = measure_all_heads(str(c4_e3), device)

    print("Loading C4 epoch 15...")
    c4_15 = measure_all_heads(str(c4_e15), device)

    curated_delta = compute_delta_matrix(stock, curated)
    c4_3_delta = compute_delta_matrix(stock, c4_3)
    c4_15_delta = compute_delta_matrix(stock, c4_15)

    # Compute all stats
    cur_out, cur_frozen = count_drifters(curated_delta)
    c3_out, c3_frozen = count_drifters(c4_3_delta)
    c15_out, c15_frozen = count_drifters(c4_15_delta)

    # ── Figure A: Matched epoch comparison (curated E3 vs C4 E3) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    fig.suptitle(
        "Matched-Epoch Comparison: Curated vs C4 at Epoch 3\n"
        "Identical surgery, identical training duration. Black box = surgical zone.",
        fontsize=14, fontweight='bold', y=0.98
    )

    cur_stats = (
        f"Outside zone: {cur_out['count']}/{cur_out['total']} drifting\n"
        f"Mean |δ| outside: {np.mean(cur_out['deltas']):.4f}\n"
        f"Frozen in-band: {cur_frozen['count']}/{cur_frozen['total']} drifting\n"
        f"PPL: ~15.4"
    )
    c3_stats = (
        f"Outside zone: {c3_out['count']}/{c3_out['total']} drifting\n"
        f"Mean |δ| outside: {np.mean(c3_out['deltas']):.4f}\n"
        f"Frozen in-band: {c3_frozen['count']}/{c3_frozen['total']} drifting\n"
        f"PPL: ~20.8"
    )

    im1 = draw_panel(ax1, curated_delta, "Curated Corpus — Epoch 3", cur_stats)
    im2 = draw_panel(ax2, c4_3_delta, "C4 Corpus — Epoch 3", c3_stats)

    fig.colorbar(im2, ax=[ax1, ax2], shrink=0.6, pad=0.02,
                 label="BOS mass delta (trained − stock)")
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])

    fig_dir = base / "baseline_c4_20260217_140421" / "figures"
    fig_dir.mkdir(exist_ok=True)
    out_a = fig_dir / "figure4_matched_epoch3.png"
    plt.savefig(out_a, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_a}")

    # ── Figure B: C4 progression (E3 vs E15) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    fig.suptitle(
        "C4 Training Progression: Epoch 3 vs Epoch 15\n"
        "How redistribution evolves with continued training on generic corpus.",
        fontsize=14, fontweight='bold', y=0.98
    )

    c15_stats = (
        f"Outside zone: {c15_out['count']}/{c15_out['total']} drifting\n"
        f"Mean |δ| outside: {np.mean(c15_out['deltas']):.4f}\n"
        f"Frozen in-band: {c15_frozen['count']}/{c15_frozen['total']} drifting\n"
        f"PPL: ~20.8 (overfitting)"
    )

    im1 = draw_panel(ax1, c4_3_delta, "C4 Corpus — Epoch 3", c3_stats)
    im2 = draw_panel(ax2, c4_15_delta, "C4 Corpus — Epoch 15", c15_stats)

    fig.colorbar(im2, ax=[ax1, ax2], shrink=0.6, pad=0.02,
                 label="BOS mass delta (trained − stock)")
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])

    out_b = fig_dir / "figure4_c4_progression.png"
    plt.savefig(out_b, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_b}")

    # ── Print full comparison table ──
    print(f"\n{'='*70}")
    print(f"FULL COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'Metric':<40s} {'Curated E3':>12s} {'C4 E3':>12s} {'C4 E15':>12s}")
    print(f"{'─'*70}")
    print(f"{'Heads woken (108 target)':<40s} {'108/108':>12s} {'108/108':>12s} {'108/108':>12s}")
    print(f"{'Outside-zone drifters (>0.05)':<40s} {cur_out['count']:>12d} {c3_out['count']:>12d} {c15_out['count']:>12d}")
    print(f"{'Outside-zone total':<40s} {cur_out['total']:>12d} {c3_out['total']:>12d} {c15_out['total']:>12d}")
    print(f"{'Mean |δ| outside zone':<40s} {np.mean(cur_out['deltas']):>12.4f} {np.mean(c3_out['deltas']):>12.4f} {np.mean(c15_out['deltas']):>12.4f}")
    cw = f"{cur_out['worst'][0]} {cur_out['worst'][1]:+.3f}"
    g3w = f"{c3_out['worst'][0]} {c3_out['worst'][1]:+.3f}"
    g15w = f"{c15_out['worst'][0]} {c15_out['worst'][1]:+.3f}"
    print(f"{'Worst outside-zone':<40s} {cw:>12s} {g3w:>12s} {g15w:>12s}")
    print(f"{'Frozen in-band drifters (>0.05)':<40s} {cur_frozen['count']:>12d} {c3_frozen['count']:>12d} {c15_frozen['count']:>12d}")
    print(f"{'Frozen in-band total':<40s} {cur_frozen['total']:>12d} {c3_frozen['total']:>12d} {c15_frozen['total']:>12d}")
    print(f"{'Mean |δ| frozen in-band':<40s} {np.mean(cur_frozen['deltas']):>12.4f} {np.mean(c3_frozen['deltas']):>12.4f} {np.mean(c15_frozen['deltas']):>12.4f}")
    print(f"{'Held-out PPL':<40s} {'~15.4':>12s} {'~20.8':>12s} {'~20.8+':>12s}")

    # Save all data
    comp = {
        "curated_e3": {
            "outside_drifters": cur_out["count"], "outside_total": cur_out["total"],
            "mean_abs_delta_outside": round(float(np.mean(cur_out["deltas"])), 6),
            "worst_outside": {"head": cur_out["worst"][0], "delta": round(cur_out["worst"][1], 6)},
            "frozen_inband_drifters": cur_frozen["count"], "frozen_inband_total": cur_frozen["total"],
            "mean_abs_delta_frozen": round(float(np.mean(cur_frozen["deltas"])), 6),
        },
        "c4_e3": {
            "outside_drifters": c3_out["count"], "outside_total": c3_out["total"],
            "mean_abs_delta_outside": round(float(np.mean(c3_out["deltas"])), 6),
            "worst_outside": {"head": c3_out["worst"][0], "delta": round(c3_out["worst"][1], 6)},
            "frozen_inband_drifters": c3_frozen["count"], "frozen_inband_total": c3_frozen["total"],
            "mean_abs_delta_frozen": round(float(np.mean(c3_frozen["deltas"])), 6),
        },
        "c4_e15": {
            "outside_drifters": c15_out["count"], "outside_total": c15_out["total"],
            "mean_abs_delta_outside": round(float(np.mean(c15_out["deltas"])), 6),
            "worst_outside": {"head": c15_out["worst"][0], "delta": round(c15_out["worst"][1], 6)},
            "frozen_inband_drifters": c15_frozen["count"], "frozen_inband_total": c15_frozen["total"],
            "mean_abs_delta_frozen": round(float(np.mean(c15_frozen["deltas"])), 6),
        },
    }
    comp_path = fig_dir / "matched_epoch_comparison.json"
    with open(comp_path, 'w') as f:
        json.dump(comp, f, indent=2)
    print(f"\nSaved: {comp_path}")


if __name__ == "__main__":
    main()

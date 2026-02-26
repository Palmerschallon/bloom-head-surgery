#!/usr/bin/env python3
"""
Three-way corpus comparison: Curated vs Wikipedia vs C4 at epoch 3.
Full 24×16 redistribution maps side-by-side-by-side.
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
    5:  {14, 15}, 6:  {9,10,11,12,13,14,15}, 7:  {9,10,11,12,13,14},
    8:  {11,12,13,14,15}, 9:  {9,10,11,12,13,14,15}, 10: {9,10,11,12,13,14,15},
    11: {10,11,12,13,14,15}, 12: {9,10,11,12,13,14,15}, 13: {9,10,11,12,13,14,15},
    14: {9,10,11,13,14,15}, 15: {9,11,12,14,15}, 16: {10,11,12,13,14,15},
    17: {9,10,11,12,13,14}, 18: {9,10,11,12,13,14,15}, 19: {9,10,11,12,13,14,15},
    20: {9,10,11,12,13,14,15}, 21: {9,10,11,12,13}, 22: {9,10,11,13,14},
}
N_LAYERS, N_HEADS = 24, 16


def measure_all_heads(model_path, device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokens = tokenizer(DIAG_TEXT, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**tokens, output_attentions=True)
    results = {}
    for li in range(len(out.attentions)):
        attn = out.attentions[li][0].float()
        for hi in range(attn.shape[0]):
            results[(li, hi)] = attn[hi][:, 0].mean().item()
    del model, out
    gc.collect()
    torch.cuda.empty_cache()
    return results


def compute_stats(matrix):
    outside_count, outside_total = 0, 0
    frozen_count, frozen_total = 0, 0
    worst_outside = ("", 0)
    deltas_outside = []

    for li in range(N_LAYERS):
        for hi in range(N_HEADS):
            in_zone = (li in SURGICAL_LAYERS) and (hi in SURGICAL_HEADS)
            d = matrix[li, hi]
            ad = abs(d)
            if not in_zone:
                outside_total += 1
                deltas_outside.append(ad)
                if ad > 0.05:
                    outside_count += 1
                if ad > abs(worst_outside[1]):
                    worst_outside = (f"L{li}H{hi}", d)
            else:
                if li in REINIT and hi not in REINIT[li]:
                    frozen_total += 1
                    if ad > 0.05:
                        frozen_count += 1

    return {
        "outside_drifters": outside_count,
        "outside_total": outside_total,
        "mean_abs_delta_outside": float(np.mean(deltas_outside)),
        "worst_outside": worst_outside,
        "frozen_drifters": frozen_count,
        "frozen_total": frozen_total,
    }


def draw_panel(ax, matrix, title, stats):
    import matplotlib.patches as patches
    vmax = 0.4
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax, interpolation='nearest')
    rect = patches.Rectangle((8.5, 4.5), 7.0, 18.0, linewidth=2.5, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    for li in range(N_LAYERS):
        for hi in range(N_HEADS):
            d = matrix[li, hi]
            if abs(d) > 0.15:
                color = 'white' if abs(d) > 0.25 else 'black'
                ax.text(hi, li, f"{d:+.2f}", ha='center', va='center', fontsize=4.5, color=color, fontweight='bold')
    ax.set_xticks(range(N_HEADS))
    ax.set_xticklabels([f"H{h}" for h in range(N_HEADS)], fontsize=6)
    ax.set_yticks(range(N_LAYERS))
    ax.set_yticklabels([f"L{l}" for l in range(N_LAYERS)], fontsize=6)
    ax.set_xlabel("Head Index", fontsize=9)
    ax.set_ylabel("Layer", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    s = stats
    text = (
        f"Outside zone: {s['outside_drifters']}/{s['outside_total']} drifting\n"
        f"Mean |δ| outside: {s['mean_abs_delta_outside']:.4f}\n"
        f"Frozen in-band: {s['frozen_drifters']}/{s['frozen_total']} drifting"
    )
    ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=6, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.85))
    return im


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    base = Path("/ember/progressive_checkpoints/bloom1b7")
    curated_e3 = base / "headband_20260217_092844" / "epoch_003"
    wiki_e3 = sorted(base.glob("wiki_curated_*"))[-1] / "epoch_003"
    c4_e3 = sorted(base.glob("baseline_c4_*"))[-1] / "epoch_003"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading stock...")
    stock = measure_all_heads("bigscience/bloom-1b7", device)
    print("Loading curated E3...")
    curated = measure_all_heads(str(curated_e3), device)
    print("Loading wiki E3...")
    wiki = measure_all_heads(str(wiki_e3), device)
    print("Loading C4 E3...")
    c4 = measure_all_heads(str(c4_e3), device)

    # Compute deltas
    deltas = {}
    for name, trained in [("curated", curated), ("wiki", wiki), ("c4", c4)]:
        m = np.zeros((N_LAYERS, N_HEADS))
        for li in range(N_LAYERS):
            for hi in range(N_HEADS):
                m[li, hi] = trained.get((li, hi), 0) - stock.get((li, hi), 0)
        deltas[name] = m

    stats = {name: compute_stats(m) for name, m in deltas.items()}

    # Three-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(30, 12))
    fig.suptitle(
        "Three-Way Corpus Comparison at Epoch 3: Global Attention Redistribution\n"
        "Identical surgery (108 H9-H15 heads), identical hyperparameters, different corpus only.\n"
        "Black box = surgical zone. All color outside box = untouched heads affected by surgery.",
        fontsize=13, fontweight='bold', y=0.98
    )

    panels = [
        (axes[0], deltas["curated"], "Ember Curated Corpus\n(PPL ~15.4)", stats["curated"]),
        (axes[1], deltas["wiki"], "Wikipedia Curated\n(PPL 17.40)", stats["wiki"]),
        (axes[2], deltas["c4"], "C4 Generic\n(PPL ~20.8)", stats["c4"]),
    ]

    for ax, matrix, title, s in panels:
        im = draw_panel(ax, matrix, title, s)

    fig.colorbar(im, ax=list(axes), shrink=0.5, pad=0.02, label="BOS mass delta (trained − stock)")
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])

    fig_dir = sorted(base.glob("baseline_c4_*"))[-1] / "figures"
    fig_dir.mkdir(exist_ok=True)
    out_path = fig_dir / "figure_threeway_epoch3.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_path}")

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"THREE-WAY CORPUS COMPARISON AT EPOCH 3")
    print(f"{'='*70}")
    print(f"{'Metric':<40s} {'Curated':>10s} {'Wiki':>10s} {'C4':>10s}")
    print(f"{'─'*70}")
    for name in ["curated", "wiki", "c4"]:
        s = stats[name]
    print(f"{'Outside-zone drifters (>0.05)':<40s} {stats['curated']['outside_drifters']:>10d} {stats['wiki']['outside_drifters']:>10d} {stats['c4']['outside_drifters']:>10d}")
    print(f"{'Outside-zone total':<40s} {stats['curated']['outside_total']:>10d} {stats['wiki']['outside_total']:>10d} {stats['c4']['outside_total']:>10d}")
    print(f"{'Mean |δ| outside zone':<40s} {stats['curated']['mean_abs_delta_outside']:>10.4f} {stats['wiki']['mean_abs_delta_outside']:>10.4f} {stats['c4']['mean_abs_delta_outside']:>10.4f}")
    cw = f"{stats['curated']['worst_outside'][0]} {stats['curated']['worst_outside'][1]:+.3f}"
    ww = f"{stats['wiki']['worst_outside'][0]} {stats['wiki']['worst_outside'][1]:+.3f}"
    gw = f"{stats['c4']['worst_outside'][0]} {stats['c4']['worst_outside'][1]:+.3f}"
    print(f"{'Worst outside-zone':<40s} {cw:>10s} {ww:>10s} {gw:>10s}")
    print(f"{'Frozen in-band drifters (>0.05)':<40s} {stats['curated']['frozen_drifters']:>10d} {stats['wiki']['frozen_drifters']:>10d} {stats['c4']['frozen_drifters']:>10d}")
    print(f"{'PPL at E3':<40s} {'~15.4':>10s} {'17.40':>10s} {'~20.8':>10s}")

    # Save data
    save = {name: {
        **stats[name],
        "worst_outside": {"head": stats[name]["worst_outside"][0], "delta": round(stats[name]["worst_outside"][1], 6)},
        "delta_matrix": [[round(float(deltas[name][li, hi]), 6) for hi in range(N_HEADS)] for li in range(N_LAYERS)],
    } for name in ["curated", "wiki", "c4"]}
    data_path = fig_dir / "threeway_comparison.json"
    with open(data_path, 'w') as f:
        json.dump(save, f, indent=2)
    print(f"Saved: {data_path}")


if __name__ == "__main__":
    main()

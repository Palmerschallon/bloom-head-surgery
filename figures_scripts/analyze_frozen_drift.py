#!/usr/bin/env python3
"""
Frozen drift trajectory analysis for the C4 baseline experiment.

Two analyses:

1. PER-HEAD TRACKING: The specific heads that showed drift at E3/E6.
   Determines trajectory shape (cascade vs plateau vs oscillation).

2. COLUMN-WISE ANALYSIS: All frozen heads grouped by head index.
   Tests hypothesis: drift propagates preferentially along head-index
   columns (e.g. all H15 heads drift together, independently of H12).
   If true, this reveals structure in how residual stream perturbations
   propagate — a finding about transformer dynamics, not just BLOOM.

Generates matplotlib figures for the paper.
"""

import json
import sys
import math
import torch
import gc
from pathlib import Path
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

DIAG_TEXT = (
    "The container holds the boundary. The boundary defines the edge. "
    "The edge separates inside from outside. Inside and outside depend "
    "on the container. Memory persists through structure, not content. "
    "What remains after deletion is the shape of what was deleted."
)

# Surgery map from train_1b7_headband.py / train_1b7_baseline.py
# For each layer: which heads were reinitialized (trainable) vs frozen in-band
SURGERY = {
    5:  {'reinit': [14, 15],                    'freeze': [9, 10, 11, 12, 13]},
    6:  {'reinit': [9, 10, 11, 12, 13, 14, 15], 'freeze': []},
    7:  {'reinit': [9, 10, 11, 12, 13, 14],     'freeze': [15]},
    8:  {'reinit': [11, 12, 13, 14, 15],         'freeze': [9, 10]},
    9:  {'reinit': [9, 10, 11, 12, 13, 14, 15], 'freeze': []},
    10: {'reinit': [9, 10, 11, 12, 13, 14, 15], 'freeze': []},
    11: {'reinit': [10, 11, 12, 13, 14, 15],    'freeze': [9]},
    12: {'reinit': [9, 10, 11, 12, 13, 14, 15], 'freeze': []},
    13: {'reinit': [9, 10, 11, 12, 13, 14, 15], 'freeze': []},
    14: {'reinit': [9, 10, 11, 13, 14, 15],     'freeze': [12]},
    15: {'reinit': [9, 11, 12, 14, 15],          'freeze': [10, 13]},
    16: {'reinit': [10, 11, 12, 13, 14, 15],    'freeze': [9]},
    17: {'reinit': [9, 10, 11, 12, 13, 14],     'freeze': [15]},
    18: {'reinit': [9, 10, 11, 12, 13, 14, 15], 'freeze': []},
    19: {'reinit': [9, 10, 11, 12, 13, 14, 15], 'freeze': []},
    20: {'reinit': [9, 10, 11, 12, 13, 14, 15], 'freeze': []},
    21: {'reinit': [9, 10, 11, 12, 13],          'freeze': [14, 15]},
    22: {'reinit': [9, 10, 11, 13, 14],          'freeze': [12, 15]},
}

# All frozen in-band heads
FROZEN_HEADS = []
for li, cfg in sorted(SURGERY.items()):
    for hi in cfg['freeze']:
        FROZEN_HEADS.append((li, hi))

# Also track ALL heads in surgical layers (for column analysis)
ALL_LAYERS = sorted(SURGERY.keys())
N_HEADS = 16


def measure_all_heads(model_path, device="cuda"):
    """Load checkpoint, measure BOS mass + entropy for ALL heads in surgical layers."""
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
            ent = -(ha.clamp(min=1e-10) * ha.clamp(min=1e-10).log()).sum(-1).mean().item()

            if bos > 0.95:
                pat = "DEAD"
            elif bos > 0.5:
                pat = "BOS-sink"
            elif ent < 0.5:
                pat = "low-entropy"
            else:
                pat = "healthy"

            results[(li, hi)] = {
                'bos_mass': round(bos, 6),
                'entropy': round(ent, 4),
                'pattern': pat,
            }

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    if len(sys.argv) < 2:
        base = Path("/ember/progressive_checkpoints/bloom1b7")
        runs = sorted(base.glob("baseline_c4_*"))
        if not runs:
            print("No baseline_c4 runs found.")
            return
        run_dir = runs[-1]
    else:
        run_dir = Path(sys.argv[1])

    print(f"Analyzing frozen drift in: {run_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build checkpoint list
    checkpoints = []
    checkpoints.append(("stock", "bigscience/bloom-1b7"))

    epoch_dirs = sorted(run_dir.glob("epoch_*"))
    for ep_dir in epoch_dirs:
        epoch_num = int(ep_dir.name.split("_")[1])
        checkpoints.append((f"epoch_{epoch_num:03d}", str(ep_dir)))

    final_dir = run_dir / "final"
    if final_dir.exists():
        checkpoints.append(("final", str(final_dir)))

    print(f"  Found {len(checkpoints)} checkpoints: {[l for l,_ in checkpoints]}")

    # ─── Measure every head at every checkpoint ───
    all_data = {}  # label -> {(layer, head): {bos_mass, entropy, pattern}}
    for label, path in checkpoints:
        print(f"\n  Loading {label}...")
        all_data[label] = measure_all_heads(path, device)
        # Quick summary of frozen heads
        n_drifting = 0
        for li, hi in FROZEN_HEADS:
            stock_bos = all_data["stock"].get((li, hi), {}).get('bos_mass', 0)
            curr_bos = all_data[label].get((li, hi), {}).get('bos_mass', 0)
            if abs(curr_bos - stock_bos) > 0.05:
                n_drifting += 1
        if label != "stock":
            print(f"    Frozen heads drifting (>0.05 delta): {n_drifting}/{len(FROZEN_HEADS)}")

    epoch_labels = [l for l, _ in checkpoints if l.startswith("epoch_")]

    # ═══════════════════════════════════════════
    # ANALYSIS 1: Per-head drift trajectories
    # ═══════════════════════════════════════════
    print(f"\n{'='*70}")
    print("ANALYSIS 1: FROZEN HEAD DRIFT TRAJECTORIES")
    print(f"{'='*70}")

    for li, hi in FROZEN_HEADS:
        stock_bos = all_data["stock"][(li, hi)]['bos_mass']
        deltas = []
        for label in epoch_labels:
            curr = all_data[label][(li, hi)]['bos_mass']
            deltas.append(curr - stock_bos)

        max_delta = max(abs(d) for d in deltas) if deltas else 0
        if max_delta < 0.03:
            continue  # skip stable heads

        print(f"\n  L{li} H{hi} (stock BOS={stock_bos:.4f}):")
        print(f"    {'Checkpoint':<14s} {'BOS':>8s} {'Delta':>8s} {'Pattern':<12s}")
        print(f"    {'─'*45}")
        for label in ["stock"] + epoch_labels + (["final"] if "final" in all_data else []):
            d = all_data[label][(li, hi)]
            delta = d['bos_mass'] - stock_bos
            marker = " ←" if abs(delta) > 0.05 else ""
            print(f"    {label:<14s} {d['bos_mass']:>8.4f} {delta:>+8.4f} {d['pattern']:<12s}{marker}")

    # Classify trajectories
    print(f"\n  TRAJECTORY CLASSIFICATION:")
    print(f"  {'Head':<10s} {'Shape':<45s} {'Max delta':>10s}")
    print(f"  {'─'*65}")

    for li, hi in FROZEN_HEADS:
        stock_bos = all_data["stock"][(li, hi)]['bos_mass']
        bos_vals = [all_data[l][(li, hi)]['bos_mass'] for l in epoch_labels]
        deltas = [b - stock_bos for b in bos_vals]
        max_delta = max(abs(d) for d in deltas) if deltas else 0

        if len(bos_vals) < 3:
            shape = "insufficient data"
        elif max_delta < 0.03:
            shape = "STABLE (no significant drift)"
        else:
            diffs = [bos_vals[i+1] - bos_vals[i] for i in range(len(bos_vals)-1)]
            all_inc = all(d > 0.005 for d in diffs)
            all_dec = all(d < -0.005 for d in diffs)
            sign_changes = sum(1 for i in range(len(diffs)-1)
                              if (diffs[i] > 0.005) != (diffs[i+1] > 0.005))

            if all_inc:
                shape = "MONOTONIC INCREASE (cumulative cascade)"
            elif all_dec:
                shape = "MONOTONIC DECREASE (self-correcting)"
            elif max(bos_vals) - min(bos_vals) < 0.03:
                shape = "PLATEAU (adapted to new residual)"
            elif sign_changes >= 2:
                shape = "OSCILLATING (gradient noise)"
            else:
                shape = "MIXED (partial adaptation)"

        print(f"  L{li:>2} H{hi:<2d}    {shape:<45s} {max_delta:>+10.4f}")

    # ═══════════════════════════════════════════
    # ANALYSIS 2: Column-wise drift (THE PAPER FIGURE)
    # ═══════════════════════════════════════════
    print(f"\n{'='*70}")
    print("ANALYSIS 2: COLUMN-WISE DRIFT (head-index grouping)")
    print(f"{'='*70}")
    print("\nHypothesis: drift propagates preferentially along head-index columns.")
    print("If H15 heads across different layers drift together but H12 heads don't,")
    print("the residual stream perturbation has column structure.\n")

    # Group frozen heads by head index
    columns = defaultdict(list)  # head_idx -> [(layer, head_idx), ...]
    for li, hi in FROZEN_HEADS:
        columns[hi].append((li, hi))

    # For each column, compute mean drift across epochs
    print(f"  {'Head idx':<10s} {'N frozen':<10s}", end="")
    for label in epoch_labels:
        ep_num = label.split("_")[1]
        print(f" {'E'+ep_num:>8s}", end="")
    print(f" {'Trend':>12s}")
    print(f"  {'─'*80}")

    column_trajectories = {}
    for hi in sorted(columns.keys()):
        heads = columns[hi]
        mean_deltas = []
        for label in epoch_labels:
            deltas = []
            for li, h in heads:
                stock_bos = all_data["stock"][(li, h)]['bos_mass']
                curr_bos = all_data[label][(li, h)]['bos_mass']
                deltas.append(abs(curr_bos - stock_bos))
            mean_deltas.append(sum(deltas) / len(deltas))

        # Trend
        if len(mean_deltas) >= 2:
            if mean_deltas[-1] > mean_deltas[0] + 0.02:
                trend = "SPREADING"
            elif mean_deltas[-1] < mean_deltas[0] - 0.02:
                trend = "recovering"
            else:
                trend = "stable"
        else:
            trend = "—"

        print(f"  H{hi:<8d} {len(heads):<10d}", end="")
        for md in mean_deltas:
            print(f" {md:>+8.4f}", end="")
        print(f" {trend:>12s}")

        column_trajectories[hi] = {
            'heads': [(li, h) for li, h in heads],
            'mean_abs_deltas': [round(d, 6) for d in mean_deltas],
            'trend': trend,
        }

    # ── Column correlation analysis ──
    print(f"\n  COLUMN CORRELATION MATRIX (Pearson r of drift trajectories):")
    col_indices = sorted(columns.keys())
    if len(col_indices) > 1 and len(epoch_labels) >= 3:
        # Build per-head drift vectors (one value per epoch)
        col_vectors = {}
        for hi in col_indices:
            heads = columns[hi]
            vec = []
            for label in epoch_labels:
                deltas = []
                for li, h in heads:
                    stock_bos = all_data["stock"][(li, h)]['bos_mass']
                    curr_bos = all_data[label][(li, h)]['bos_mass']
                    deltas.append(curr_bos - stock_bos)
                vec.append(sum(deltas) / len(deltas))
            col_vectors[hi] = vec

        # Pearson correlation
        def pearson(a, b):
            n = len(a)
            if n < 3:
                return 0.0
            ma = sum(a) / n
            mb = sum(b) / n
            cov = sum((a[i]-ma)*(b[i]-mb) for i in range(n))
            sa = math.sqrt(sum((x-ma)**2 for x in a))
            sb = math.sqrt(sum((x-mb)**2 for x in b))
            if sa < 1e-10 or sb < 1e-10:
                return 0.0
            return cov / (sa * sb)

        print(f"\n  {'':>6s}", end="")
        for hi in col_indices:
            print(f" {'H'+str(hi):>6s}", end="")
        print()

        for hi_a in col_indices:
            print(f"  {'H'+str(hi_a):>6s}", end="")
            for hi_b in col_indices:
                r = pearson(col_vectors[hi_a], col_vectors[hi_b])
                print(f" {r:>6.2f}", end="")
            print()

    # ── Cascade propagation count ──
    print(f"\n  DRIFT PROPAGATION:")
    for label in epoch_labels:
        n_drift = 0
        for li, hi in FROZEN_HEADS:
            stock_bos = all_data["stock"][(li, hi)]['bos_mass']
            curr_bos = all_data[label][(li, hi)]['bos_mass']
            if abs(curr_bos - stock_bos) > 0.05:
                n_drift += 1
        print(f"    {label}: {n_drift}/{len(FROZEN_HEADS)} frozen heads drifting (>0.05)")

    # ═══════════════════════════════════════════
    # ANALYSIS 3: Non-frozen, non-surgical heads
    # ═══════════════════════════════════════════
    # Do heads OUTSIDE the surgery band (H0-H8) also drift?
    print(f"\n{'='*70}")
    print("ANALYSIS 3: OUTSIDE-BAND DRIFT (H0-H8, untouched heads)")
    print(f"{'='*70}")
    print("\nThese heads had no surgery at all — neither reinitialized nor frozen.")
    print("If they drift, perturbation propagates even beyond the surgical zone.\n")

    outside_drifters = []
    for li in ALL_LAYERS:
        surgical_heads = set(SURGERY[li]['reinit'] + SURGERY[li]['freeze'])
        for hi in range(N_HEADS):
            if hi in surgical_heads:
                continue
            stock_bos = all_data["stock"].get((li, hi), {}).get('bos_mass', 0)
            final_label = epoch_labels[-1] if epoch_labels else "stock"
            final_bos = all_data[final_label].get((li, hi), {}).get('bos_mass', 0)
            delta = abs(final_bos - stock_bos)
            if delta > 0.05:
                outside_drifters.append((li, hi, stock_bos, final_bos, delta))

    if outside_drifters:
        print(f"  {len(outside_drifters)} outside-band heads drifting (>0.05):")
        for li, hi, sb, fb, d in sorted(outside_drifters, key=lambda x: -x[4]):
            print(f"    L{li} H{hi}: stock={sb:.4f} → {final_label}={fb:.4f} (delta={d:+.4f})")
    else:
        print(f"  No outside-band heads drifting. Perturbation contained within surgical zone.")

    # ═══════════════════════════════════════════
    # GENERATE PLOTS
    # ═══════════════════════════════════════════
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        fig_dir = run_dir / "figures"
        fig_dir.mkdir(exist_ok=True)

        # ── Figure 1: Column-wise drift heatmap ──
        fig, ax = plt.subplots(figsize=(14, 8))

        # Build matrix: rows = surgical layers, cols = head indices 9-15
        head_range = list(range(9, 16))
        layer_list = ALL_LAYERS
        matrix = np.zeros((len(layer_list), len(head_range)))
        annotations = np.empty((len(layer_list), len(head_range)), dtype=object)

        final_label = epoch_labels[-1] if epoch_labels else "stock"

        for row, li in enumerate(layer_list):
            for col, hi in enumerate(head_range):
                stock_bos = all_data["stock"].get((li, hi), {}).get('bos_mass', 0)
                curr_bos = all_data[final_label].get((li, hi), {}).get('bos_mass', 0)
                delta = curr_bos - stock_bos

                if hi in SURGERY[li]['reinit']:
                    annotations[row, col] = "R"  # reinitialized
                    matrix[row, col] = 0  # neutral — these were supposed to change
                elif hi in SURGERY[li]['freeze']:
                    annotations[row, col] = f"{delta:+.2f}"
                    matrix[row, col] = delta
                else:
                    annotations[row, col] = ""
                    matrix[row, col] = 0

        im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-0.3, vmax=0.3)
        ax.set_xticks(range(len(head_range)))
        ax.set_xticklabels([f"H{h}" for h in head_range])
        ax.set_yticks(range(len(layer_list)))
        ax.set_yticklabels([f"L{l}" for l in layer_list])
        ax.set_xlabel("Head Index")
        ax.set_ylabel("Layer")
        ax.set_title(f"Frozen Head BOS-Mass Drift (C4 baseline, {final_label})\n"
                      f"R = reinitialized, values = frozen head drift from stock")

        for row in range(len(layer_list)):
            for col in range(len(head_range)):
                txt = annotations[row, col]
                if txt:
                    color = 'white' if abs(matrix[row, col]) > 0.15 else 'black'
                    fontsize = 7 if txt == "R" else 6
                    ax.text(col, row, txt, ha='center', va='center',
                            fontsize=fontsize, color=color, fontweight='bold')

        plt.colorbar(im, ax=ax, label="BOS mass delta from stock")
        plt.tight_layout()
        plt.savefig(fig_dir / "frozen_drift_heatmap.png", dpi=150)
        plt.close()
        print(f"\n  Saved: {fig_dir / 'frozen_drift_heatmap.png'}")

        # ── Figure 2: Column drift trajectories over epochs ──
        fig, ax = plt.subplots(figsize=(10, 6))

        epoch_nums = []
        for l in epoch_labels:
            epoch_nums.append(int(l.split("_")[1]))

        colors = plt.cm.tab10(np.linspace(0, 1, len(col_indices)))
        for idx, hi in enumerate(col_indices):
            heads = columns[hi]
            mean_deltas = []
            for label in epoch_labels:
                deltas = []
                for li, h in heads:
                    stock_bos = all_data["stock"][(li, h)]['bos_mass']
                    curr_bos = all_data[label][(li, h)]['bos_mass']
                    deltas.append(abs(curr_bos - stock_bos))
                mean_deltas.append(sum(deltas) / len(deltas))

            ax.plot(epoch_nums, mean_deltas, 'o-', color=colors[idx],
                    label=f"H{hi} (n={len(heads)})", linewidth=2, markersize=6)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean |BOS mass delta| from stock")
        ax.set_title("Frozen Head Drift by Head-Index Column\n"
                      "(C4 baseline — same surgery, generic corpus)")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='drift threshold')
        plt.tight_layout()
        plt.savefig(fig_dir / "column_drift_trajectories.png", dpi=150)
        plt.close()
        print(f"  Saved: {fig_dir / 'column_drift_trajectories.png'}")

        # ── Figure 3: Cascade propagation count over epochs ──
        fig, ax = plt.subplots(figsize=(8, 5))
        counts = []
        for label in epoch_labels:
            n = 0
            for li, hi in FROZEN_HEADS:
                stock_bos = all_data["stock"][(li, hi)]['bos_mass']
                curr_bos = all_data[label][(li, hi)]['bos_mass']
                if abs(curr_bos - stock_bos) > 0.05:
                    n += 1
            counts.append(n)

        ax.plot(epoch_nums, counts, 'o-', color='darkred', linewidth=2, markersize=8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Number of drifting frozen heads")
        ax.set_title(f"Cascade Propagation: Frozen Head Drift Count\n"
                      f"(out of {len(FROZEN_HEADS)} total frozen heads, threshold=0.05)")
        ax.set_ylim(0, len(FROZEN_HEADS) + 1)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / "cascade_propagation.png", dpi=150)
        plt.close()
        print(f"  Saved: {fig_dir / 'cascade_propagation.png'}")

    except ImportError:
        print("\n  matplotlib not available — skipping plots.")
        print("  Install with: pip install matplotlib")

    # ═══════════════════════════════════════════
    # SAVE ALL RESULTS
    # ═══════════════════════════════════════════
    output = {
        'run_dir': str(run_dir),
        'frozen_heads': [{'layer': li, 'head': hi} for li, hi in FROZEN_HEADS],
        'checkpoints': [l for l, _ in checkpoints],
        'per_head_trajectories': {},
        'column_trajectories': {},
        'cascade_counts': {},
    }

    for li, hi in FROZEN_HEADS:
        key = f"L{li}_H{hi}"
        output['per_head_trajectories'][key] = {}
        for label, _ in checkpoints:
            output['per_head_trajectories'][key][label] = all_data[label][(li, hi)]

    for hi, ct in column_trajectories.items():
        output['column_trajectories'][f"H{hi}"] = ct

    for i, label in enumerate(epoch_labels):
        n = 0
        for li, hi in FROZEN_HEADS:
            stock_bos = all_data["stock"][(li, hi)]['bos_mass']
            curr_bos = all_data[label][(li, hi)]['bos_mass']
            if abs(curr_bos - stock_bos) > 0.05:
                n += 1
        output['cascade_counts'][label] = n

    out_path = run_dir / "frozen_drift_analysis.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Full results: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Four-way comparison: Stock vs Curated E3 vs C4 E3 vs Wiki E3

All at matched epoch 3. Same surgery, same masks, same hyperparameters.
Only difference: corpus.

Generates:
  1. 2x2 redistribution heatmaps (24x16, surgical zone outlined)
  2. Summary comparison table
  3. Completions for the same prompts across all 4 conditions
"""

import json
import torch
import gc
import numpy as np
import time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

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

N_LAYERS = 24
N_HEADS = 16

MODELS = {
    "stock":      ("Stock BLOOM-1b7", "bigscience/bloom-1b7"),
    "curated_e3": ("Curated E3",      "/ember/progressive_checkpoints/bloom1b7/headband_20260217_092844/epoch_003"),
    "c4_e3":      ("C4 E3",           "/ember/progressive_checkpoints/bloom1b7/baseline_c4_20260217_140421/epoch_003"),
    "wiki_e3":    ("Wiki E3",         "/ember/progressive_checkpoints/bloom1b7/wiki_curated_20260217_160224/epoch_003"),
}

COMPLETION_PROMPTS = [
    ("conceptual",   "The difference between structure and pattern is"),
    ("conceptual",   "Emptiness is not the absence of content but"),
    ("technical",    "In a transformer, the attention mechanism computes"),
    ("technical",    "A recursive function terminates when"),
    ("narrative",    "She opened the door and found"),
    ("narrative",    "The library had one book that no one was allowed to read because"),
    ("code",         "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n    if n <= 1:\n        return n\n    "),
    ("code",         "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    "),
    ("french",       "La structure du langage reflète la structure de la pensée, mais"),
    ("spanish",      "La diferencia entre conocer y comprender es que"),
]


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


def generate_completions(model_path, prompts, device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    results = []
    for cat, prompt in prompts:
        set_seed(42 + hash(prompt) % 10000)
        tokens = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = tokens["input_ids"].shape[1]
        with torch.no_grad():
            output = model.generate(
                **tokens, max_new_tokens=100, temperature=0.7,
                top_p=0.92, do_sample=True, repetition_penalty=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
        results.append({"category": cat, "prompt": prompt, "completion": completion})
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def count_drifters(matrix, threshold=0.05):
    outside = {"count": 0, "total": 0, "worst": ("", 0), "deltas": []}
    frozen = {"count": 0, "total": 0, "deltas": []}
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
                if li in REINIT and hi not in REINIT[li]:
                    frozen["total"] += 1
                    frozen["deltas"].append(ad)
                    if ad > threshold:
                        frozen["count"] += 1
    return outside, frozen


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = Path("/ember/progressive_checkpoints/bloom1b7")

    # ── Measure all heads for all 4 models ──
    all_bos = {}
    for key, (label, path) in MODELS.items():
        print(f"Loading {label}...")
        all_bos[key] = measure_all_heads(path, device)

    # ── Compute delta matrices (relative to stock) ──
    deltas = {}
    stats = {}
    for key in ["curated_e3", "c4_e3", "wiki_e3"]:
        matrix = np.zeros((N_LAYERS, N_HEADS))
        for li in range(N_LAYERS):
            for hi in range(N_HEADS):
                matrix[li, hi] = all_bos[key].get((li, hi), 0) - all_bos["stock"].get((li, hi), 0)
        deltas[key] = matrix
        outside, frozen = count_drifters(matrix)
        stats[key] = {"outside": outside, "frozen": frozen}

    # ── Figure: 1×3 side-by-side heatmaps ──
    fig, axes = plt.subplots(1, 3, figsize=(30, 12))
    fig.suptitle(
        "Three-Way Corpus Comparison: Global Attention Redistribution at Epoch 3\n"
        "Identical surgery (108 heads), identical hyperparameters, identical seed. Black box = surgical zone.",
        fontsize=14, fontweight='bold', y=0.98
    )

    vmax = 0.4
    panels = [
        ("curated_e3", "Curated Corpus (PPL ~15.4)"),
        ("wiki_e3",    "Wikipedia Corpus (PPL 17.4)"),
        ("c4_e3",      "C4 Generic (PPL ~20.8)"),
    ]

    for ax, (key, title) in zip(axes, panels):
        matrix = deltas[key]
        im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax, interpolation='nearest')
        rect = patches.Rectangle((8.5, 4.5), 7.0, 18.0, linewidth=2.5,
                                  edgecolor='black', facecolor='none')
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
        ax.set_title(title, fontsize=12, fontweight='bold')

        o = stats[key]["outside"]
        f = stats[key]["frozen"]
        mean_o = np.mean(o["deltas"]) if o["deltas"] else 0
        text = (f"Outside: {o['count']}/{o['total']} drifting\n"
                f"Mean |δ| outside: {mean_o:.4f}\n"
                f"Frozen in-band: {f['count']}/{f['total']}")
        ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=7,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.85))

    fig.colorbar(im, ax=axes.tolist(), shrink=0.6, pad=0.02,
                 label="BOS mass delta (trained − stock)")
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])

    fig_dir = base / "figures_paper"
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / "threeway_corpus_comparison.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out}")

    # ── Generate completions for all 4 models ──
    print("\nGenerating completions...")
    all_completions = {}
    for key, (label, path) in MODELS.items():
        print(f"  {label}...")
        all_completions[key] = generate_completions(path, COMPLETION_PROMPTS, device)

    # ── Print comparison table ──
    print(f"\n{'='*80}")
    print("FOUR-WAY COMPARISON TABLE (all at Epoch 3)")
    print(f"{'='*80}")
    print(f"{'Metric':<40s} {'Curated':>12s} {'Wiki':>12s} {'C4':>12s}")
    print(f"{'─'*80}")
    print(f"{'Heads woken':<40s} {'108/108':>12s} {'108/108':>12s} {'108/108':>12s}")

    for key, label in [("curated_e3","Curated"), ("wiki_e3","Wiki"), ("c4_e3","C4")]:
        o = stats[key]["outside"]
        f = stats[key]["frozen"]

    labels = ["curated_e3", "wiki_e3", "c4_e3"]
    for metric_fn, name in [
        (lambda k: stats[k]["outside"]["count"], "Outside-zone drifters (>0.05)"),
        (lambda k: f"{np.mean(stats[k]['outside']['deltas']):.4f}", "Mean |δ| outside zone"),
        (lambda k: stats[k]["frozen"]["count"], "Frozen in-band drifters (>0.05)"),
        (lambda k: f"{np.mean(stats[k]['frozen']['deltas']):.4f}" if stats[k]['frozen']['deltas'] else "N/A", "Mean |δ| frozen in-band"),
    ]:
        vals = [str(metric_fn(k)) for k in labels]
        print(f"{name:<40s} {vals[0]:>12s} {vals[1]:>12s} {vals[2]:>12s}")

    print(f"{'Best PPL':<40s} {'~15.4':>12s} {'17.40':>12s} {'~20.8':>12s}")

    # ── Print completion samples ──
    print(f"\n{'='*80}")
    print("COMPLETION SAMPLES")
    print(f"{'='*80}")

    for i, (cat, prompt) in enumerate(COMPLETION_PROMPTS):
        short = prompt[:70].replace('\n', '\\n')
        print(f"\n[{cat}] {short}...")
        for key in ["stock", "curated_e3", "wiki_e3", "c4_e3"]:
            label = MODELS[key][0]
            comp = all_completions[key][i]["completion"][:150].replace('\n', '\\n')
            print(f"  {label:>15s}: {comp}")

    # ── Save everything ──
    output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "description": "Four-way comparison: stock, curated E3, C4 E3, wiki E3",
            "models": {k: v[1] for k, v in MODELS.items()},
        },
        "stats": {},
        "completions": {},
    }
    for key in labels:
        o = stats[key]["outside"]
        f = stats[key]["frozen"]
        output["stats"][key] = {
            "outside_drifters": o["count"],
            "outside_total": o["total"],
            "mean_abs_delta_outside": round(float(np.mean(o["deltas"])), 6) if o["deltas"] else None,
            "worst_outside": {"head": o["worst"][0], "delta": round(o["worst"][1], 6)},
            "frozen_drifters": f["count"],
            "frozen_total": f["total"],
            "mean_abs_delta_frozen": round(float(np.mean(f["deltas"])), 6) if f["deltas"] else None,
        }
    for key in MODELS:
        output["completions"][key] = all_completions[key]

    data_path = fig_dir / "fourway_comparison.json"
    with open(data_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {data_path}")


if __name__ == "__main__":
    main()

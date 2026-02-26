#!/usr/bin/env python3
"""
BOS-sink diagnostic across the BLOOM model family.

Runs the same attention head health analysis on stock weights for:
  - BLOOM-560m  (24 layers, 16 heads, 1024 hidden)
  - BLOOM-1b7   (24 layers, 16 heads, 2048 hidden)
  - BLOOM-3b    (30 layers, 32 heads, 2560 hidden)
  - BLOOM-7b1   (30 layers, 32 heads, 4096 hidden)

For each model, classifies every attention head as:
  - healthy (content-attending)
  - BOS-sink (>50% mass on position 0)
  - DEAD (>95% mass on position 0)
  - low-entropy (<0.5 entropy)

Uses device_map="auto" for models that don't fit in VRAM.
"""

import json
import math
import time
import torch
import gc
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [
    ("bigscience/bloom-560m", "BLOOM-560m"),
    ("bigscience/bloom-1b7",  "BLOOM-1b7"),
    ("bigscience/bloom-3b",   "BLOOM-3b"),
    ("bigscience/bloom-7b1",  "BLOOM-7b1"),
]

OUTPUT_DIR = Path("/ember/00_Koan_Engine/bloom_family_diagnostic")

DIAG_TEXT = (
    "The container holds the boundary. The boundary defines the edge. "
    "The edge separates inside from outside. Inside and outside depend "
    "on the container. Memory persists through structure, not content. "
    "What remains after deletion is the shape of what was deleted."
)

# Held-out perplexity prompts (same as evaluation scripts)
PPL_PROMPTS = [
    "The river carried sediment downstream, depositing layers that would eventually become stone.",
    "Because the experiment failed three times, the researcher redesigned the protocol entirely.",
    "A symphony orchestra tunes to the oboe because its pitch is the most stable and penetrating.",
    "async function fetchData(url) {\n  const response = await fetch(url);\n  return response.json();\n}",
    "SELECT users.name, orders.total FROM users JOIN orders ON users.id = orders.user_id WHERE orders.total > 100;",
    "fn fibonacci(n: u64) -> u64 {\n    match n {\n        0 => 0,\n        1 => 1,\n        _ => fibonacci(n-1) + fibonacci(n-2),\n    }\n}",
    "The map is not the territory, but without maps we cannot navigate territories we haven't visited.",
    "Every recursive function must have a base case; every recursive argument must have a ground truth.",
    "La structure du langage reflète la structure de la pensée, mais la pensée dépasse toujours le langage.",
    "Das Ganze ist mehr als die Summe seiner Teile, aber die Teile definieren das Ganze.",
    "The attention mechanism computes a weighted sum over value vectors, where weights are derived from query-key dot products.",
    "In distributed systems, the CAP theorem states that consistency, availability, and partition tolerance cannot all be simultaneously guaranteed.",
]


def analyze_all_heads(model, tokenizer, device):
    """Architecture-agnostic attention head analysis."""
    tokens = tokenizer(DIAG_TEXT, return_tensors="pt")
    # Move to same device as model's first parameter
    first_device = next(model.parameters()).device
    tokens = {k: v.to(first_device) for k, v in tokens.items()}

    with torch.no_grad():
        out = model(**tokens, output_attentions=True)

    n_layers = len(out.attentions)
    results = {}
    for li in range(n_layers):
        attn = out.attentions[li][0].float().cpu()  # move to CPU for analysis
        n_heads = attn.shape[0]
        seq_len = attn.shape[1]
        layer_data = {}

        for h in range(n_heads):
            ha = attn[h]
            ent = -(ha.clamp(min=1e-10) * ha.clamp(min=1e-10).log()).sum(-1).mean().item()
            bos = ha[:, 0].mean().item()

            if bos > 0.95:
                pat = "DEAD"
            elif bos > 0.5:
                pat = "BOS-sink"
            elif ent < 0.5:
                pat = "low-entropy"
            else:
                pat = "healthy"

            layer_data[h] = {
                'entropy': round(ent, 4),
                'bos_mass': round(bos, 4),
                'pattern': pat,
            }

        results[li] = layer_data

    return results


def count_patterns(analysis):
    """Summarize patterns across all layers."""
    counts = {'healthy': 0, 'BOS-sink': 0, 'DEAD': 0, 'low-entropy': 0}
    per_layer = {}
    for li in sorted(analysis.keys(), key=int):
        lc = {'healthy': 0, 'BOS-sink': 0, 'DEAD': 0, 'low-entropy': 0}
        for h in analysis[li].values():
            counts[h['pattern']] += 1
            lc[h['pattern']] += 1
        sick = lc['BOS-sink'] + lc['DEAD']
        if sick > 0:
            per_layer[int(li)] = {
                'healthy': lc['healthy'],
                'bos_sink': lc['BOS-sink'],
                'dead': lc['DEAD'],
                'sick_total': sick,
            }
    return counts, per_layer


def compute_ppl(model, tokenizer, device):
    """Perplexity on held-out prompts."""
    first_device = next(model.parameters()).device
    total_loss, total_tokens = 0, 0
    per_prompt = []
    for text in PPL_PROMPTS:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        tokens = {k: v.to(first_device) for k, v in tokens.items()}
        with torch.no_grad():
            out = model(**tokens, labels=tokens["input_ids"])
        n = tokens["input_ids"].shape[1]
        total_loss += out.loss.item() * n
        total_tokens += n
        per_prompt.append({
            'text': text[:80],
            'ppl': round(math.exp(out.loss.item()), 2),
            'tokens': n,
        })
    overall = math.exp(total_loss / total_tokens)
    return overall, per_prompt


def measure_dormancy(model, threshold=0.01):
    """MLP neuron dormancy."""
    total_d, total_n = 0, 0
    per_layer = {}
    for i, block in enumerate(model.transformer.h):
        up = block.mlp.dense_h_to_4h.weight.data.float().cpu().norm(dim=1)
        down = block.mlp.dense_4h_to_h.weight.data.float().cpu().norm(dim=0)
        imp = up * down
        nd = (imp < threshold).sum().item()
        nt = imp.numel()
        total_d += nd
        total_n += nt
        if nd > 0:
            per_layer[i] = round(nd / nt * 100, 2)
    return round(total_d / total_n * 100, 2), per_layer


def diagnose_model(model_id, label, device):
    """Full diagnostic for one model."""
    print(f"\n{'='*60}")
    print(f"DIAGNOSING: {label} ({model_id})")
    print(f"{'='*60}")

    t0 = time.time()

    # Decide loading strategy based on model size
    print("  Loading model...")
    try:
        # Try GPU first
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        ).to(device)
        load_strategy = "gpu"
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            print(f"  GPU OOM — using device_map='auto' (CPU offload)...")
            gc.collect()
            torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            load_strategy = "auto"
        else:
            raise

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Architecture info
    config = model.config
    n_layers = config.n_layer
    n_heads = config.n_head
    hidden = config.hidden_size
    head_dim = hidden // n_heads
    total_params = sum(p.numel() for p in model.parameters())
    total_heads = n_layers * n_heads

    print(f"  Architecture: {n_layers} layers × {n_heads} heads = {total_heads} total")
    print(f"  Hidden: {hidden}, head_dim: {head_dim}")
    print(f"  Parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"  Load strategy: {load_strategy}")

    # Attention analysis
    print("  Analyzing attention heads...")
    attn = analyze_all_heads(model, tokenizer, device)
    counts, sick_layers = count_patterns(attn)

    print(f"    Healthy: {counts['healthy']}/{total_heads} ({counts['healthy']/total_heads*100:.1f}%)")
    print(f"    BOS-sink: {counts['BOS-sink']} ({counts['BOS-sink']/total_heads*100:.1f}%)")
    print(f"    DEAD: {counts['DEAD']} ({counts['DEAD']/total_heads*100:.1f}%)")
    print(f"    Low-entropy: {counts['low-entropy']}")

    # Show sick layer distribution
    if sick_layers:
        print(f"    Sick layers ({len(sick_layers)} of {n_layers}):")
        for li in sorted(sick_layers.keys()):
            s = sick_layers[li]
            print(f"      L{li:>2}: {s['bos_sink']} BOS-sink, {s['dead']} dead "
                  f"({s['sick_total']}/{n_heads} = {s['sick_total']/n_heads*100:.0f}%)")

    # Identify BOS-sink band (contiguous head indices with high sick rates)
    head_sick_counts = {}
    for li in attn:
        for h, d in attn[li].items():
            if d['pattern'] in ('BOS-sink', 'DEAD'):
                head_sick_counts[h] = head_sick_counts.get(h, 0) + 1

    if head_sick_counts:
        print(f"    Per-head-index sick frequency:")
        for h in sorted(head_sick_counts.keys()):
            bar = "█" * head_sick_counts[h]
            print(f"      H{h:>2}: {head_sick_counts[h]:>3}/{n_layers} layers  {bar}")

    # Perplexity
    print("  Computing held-out perplexity...")
    ppl, ppl_detail = compute_ppl(model, tokenizer, device)
    print(f"    Overall PPL: {ppl:.2f}")

    # Dormancy
    print("  Measuring MLP dormancy...")
    dorm_pct, dorm_layers = measure_dormancy(model)
    print(f"    Overall dormancy: {dorm_pct}%")

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.0f}s")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        'model_id': model_id,
        'label': label,
        'architecture': {
            'n_layers': n_layers,
            'n_heads': n_heads,
            'hidden_size': hidden,
            'head_dim': head_dim,
            'total_params': total_params,
            'total_heads': total_heads,
        },
        'load_strategy': load_strategy,
        'attention': {
            'counts': counts,
            'sick_layers': sick_layers,
            'sick_pct': round((counts['BOS-sink'] + counts['DEAD']) / total_heads * 100, 1),
            'head_sick_frequency': {str(k): v for k, v in head_sick_counts.items()},
            'full': {str(k): v for k, v in attn.items()},
        },
        'perplexity': {
            'overall': round(ppl, 2),
            'per_prompt': ppl_detail,
        },
        'dormancy': {
            'overall_pct': dorm_pct,
            'per_layer': dorm_layers,
        },
        'elapsed_seconds': round(elapsed, 1),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for model_id, label in MODELS:
        result = diagnose_model(model_id, label, device)
        all_results[label] = result

        # Save individual result immediately (in case later models crash)
        safe_name = label.lower().replace("-", "_").replace(".", "_")
        with open(OUTPUT_DIR / f"{safe_name}.json", 'w') as f:
            json.dump(result, f, indent=2, default=str)

    # ── Cross-model comparison ──
    print(f"\n{'='*70}")
    print("BLOOM FAMILY BOS-SINK DIAGNOSTIC")
    print(f"{'='*70}")

    print(f"\n  {'Model':<14s} {'Params':>8s} {'Layers':>7s} {'Heads':>7s} "
          f"{'Healthy':>9s} {'BOS-sink':>10s} {'DEAD':>6s} {'Sick%':>7s} {'PPL':>8s}")
    print(f"  {'─'*85}")

    for label in [m[1] for m in MODELS]:
        r = all_results[label]
        a = r['architecture']
        c = r['attention']['counts']
        th = a['total_heads']
        sick = c['BOS-sink'] + c['DEAD']
        print(f"  {label:<14s} {a['total_params']/1e9:>7.2f}B {a['n_layers']:>7d} {th:>7d} "
              f"{c['healthy']:>9d} {c['BOS-sink']:>10d} {c['DEAD']:>6d} "
              f"{r['attention']['sick_pct']:>6.1f}% {r['perplexity']['overall']:>8.2f}")

    # Band analysis
    print(f"\n  BOS-sink band analysis (which head indices are most affected):")
    for label in [m[1] for m in MODELS]:
        r = all_results[label]
        freq = r['attention']['head_sick_frequency']
        if freq:
            n_layers = r['architecture']['n_layers']
            # Find the "band" — contiguous head indices with >30% sick rate
            sorted_heads = sorted(freq.items(), key=lambda x: int(x[0]))
            band_heads = [(int(h), c) for h, c in sorted_heads if c / n_layers > 0.3]
            if band_heads:
                band_range = f"H{band_heads[0][0]}-H{band_heads[-1][0]}"
                avg_rate = sum(c for _, c in band_heads) / len(band_heads) / n_layers * 100
                print(f"    {label:<14s}: band={band_range} ({len(band_heads)} heads, "
                      f"avg {avg_rate:.0f}% sick across layers)")
            else:
                print(f"    {label:<14s}: no clear band (scattered)")
        else:
            print(f"    {label:<14s}: no sick heads")

    # Save combined results
    combined = {
        'timestamp': ts,
        'diagnostic_text': DIAG_TEXT,
        'models': all_results,
    }
    with open(OUTPUT_DIR / f"bloom_family_{ts}.json", 'w') as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\n  Results saved to: {OUTPUT_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Evaluate epoch 3 checkpoint vs stock BLOOM-1b7.

Compares:
  1. Attention patterns across all 24 layers (sick/healthy/dead counts)
  2. Perplexity on diverse held-out prompts (not in training corpus)
  3. Generation quality on structural, code, and creative prompts
  4. Dormancy (MLP dead neurons)
"""

import json
import math
import torch
import gc
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

STOCK = "bigscience/bloom-1b7"
EPOCH3 = "/ember/progressive_checkpoints/bloom1b7/headband_20260217_092844/epoch_003"
OUTPUT = Path("/ember/progressive_checkpoints/bloom1b7/headband_20260217_092844/evaluation.json")

DIAG_TEXT = (
    "The container holds the boundary. The boundary defines the edge. "
    "The edge separates inside from outside. Inside and outside depend "
    "on the container. Memory persists through structure, not content. "
    "What remains after deletion is the shape of what was deleted."
)

# ── Held-out perplexity prompts (NOT in training corpus) ──
PPL_PROMPTS = [
    # General prose
    "The river carried sediment downstream, depositing layers that would eventually become stone.",
    "Because the experiment failed three times, the researcher redesigned the protocol entirely.",
    "A symphony orchestra tunes to the oboe because its pitch is the most stable and penetrating.",
    # Code (different from training corpus style)
    "async function fetchData(url) {\n  const response = await fetch(url);\n  return response.json();\n}",
    "SELECT users.name, orders.total FROM users JOIN orders ON users.id = orders.user_id WHERE orders.total > 100;",
    "fn fibonacci(n: u64) -> u64 {\n    match n {\n        0 => 0,\n        1 => 1,\n        _ => fibonacci(n-1) + fibonacci(n-2),\n    }\n}",
    # Structural / philosophical
    "The map is not the territory, but without maps we cannot navigate territories we haven't visited.",
    "Every recursive function must have a base case; every recursive argument must have a ground truth.",
    # Multilingual
    "La structure du langage reflète la structure de la pensée, mais la pensée dépasse toujours le langage.",
    "Das Ganze ist mehr als die Summe seiner Teile, aber die Teile definieren das Ganze.",
    # Technical
    "The attention mechanism computes a weighted sum over value vectors, where weights are derived from query-key dot products.",
    "In distributed systems, the CAP theorem states that consistency, availability, and partition tolerance cannot all be simultaneously guaranteed.",
]

# ── Generation prompts ──
GEN_PROMPTS = [
    # Structural / Ember-style
    "The container holds",
    "What the dormant neurons taught me is",
    "The boundary between structure and",
    # Code
    "def transform(x):\n    \"\"\"Apply a structural transformation.\"\"\"",
    "class Observer:\n    def __init__(self, subject):\n        self.subject = subject\n    \n    def watch(self",
    # Creative
    "In the space between the last token and the next, there is",
    "The first thing an attention head learns is",
    # Long-range
    "A function that calls itself is recursive. A story that tells itself is narrative. A model that models itself is",
    # Cross-domain
    "If a neural network were a city, the attention heads would be",
    "The difference between forgetting and compression is",
]


def analyze_all_heads(model, tokenizer, device):
    """Full 24-layer attention analysis."""
    tokens = tokenizer(DIAG_TEXT, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**tokens, output_attentions=True)

    results = {}
    for li in range(24):
        attn = out.attentions[li][0].float()
        seq_len = attn.shape[1]
        layer_data = {}

        for h in range(attn.shape[0]):
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
                'pattern': pat
            }

        results[li] = layer_data
    return results


def count_patterns(analysis):
    """Summarize pattern counts."""
    counts = {'healthy': 0, 'BOS-sink': 0, 'DEAD': 0, 'low-entropy': 0}
    per_layer = {}
    for li in sorted(analysis.keys(), key=int):
        lc = {'healthy': 0, 'BOS-sink': 0, 'DEAD': 0, 'low-entropy': 0}
        for h in analysis[li].values():
            counts[h['pattern']] += 1
            lc[h['pattern']] += 1
        sick = lc['BOS-sink'] + lc['DEAD']
        if sick > 0:
            per_layer[int(li)] = {'healthy': lc['healthy'], 'sick': sick, 'dead': lc['DEAD']}
    return counts, per_layer


def compute_ppl(model, tokenizer, device, prompts):
    """Perplexity on held-out prompts."""
    total_loss, total_tokens = 0, 0
    per_prompt = []
    for text in prompts:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model(**tokens, labels=tokens.input_ids)
        n = tokens.input_ids.shape[1]
        total_loss += out.loss.item() * n
        total_tokens += n
        per_prompt.append({
            'text': text[:80],
            'ppl': round(math.exp(out.loss.item()), 2),
            'tokens': n
        })
    overall = math.exp(total_loss / total_tokens)
    return overall, per_prompt


def generate(model, tokenizer, device, prompts, max_new=80):
    """Generate completions."""
    model.eval()
    results = []
    for prompt in prompts:
        toks = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **toks, max_new_tokens=max_new,
                temperature=0.85, top_p=0.92,
                repetition_penalty=1.15, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        full = tokenizer.decode(out[0], skip_special_tokens=True)
        completion = full[len(prompt):].strip()
        results.append({'prompt': prompt, 'completion': completion})
    return results


def measure_dormancy(model, threshold=0.01):
    """MLP neuron dormancy."""
    total_d, total_n = 0, 0
    per_layer = {}
    for i, block in enumerate(model.transformer.h):
        up = block.mlp.dense_h_to_4h.weight.data.float().norm(dim=1)
        down = block.mlp.dense_4h_to_h.weight.data.float().norm(dim=0)
        imp = up * down
        nd = (imp < threshold).sum().item()
        nt = imp.numel()
        total_d += nd
        total_n += nt
        if nd > 0:
            per_layer[i] = round(nd / nt * 100, 1)
    return round(total_d / total_n * 100, 2), per_layer


def evaluate_model(name, model_path, device):
    """Run full evaluation on one model."""
    print(f"\n{'='*60}")
    print(f"EVALUATING: {name}")
    print(f"{'='*60}")

    print("  Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Attention analysis
    print("  Analyzing attention heads (all 24 layers)...")
    attn = analyze_all_heads(model, tokenizer, device)
    counts, sick_layers = count_patterns(attn)
    print(f"    Healthy: {counts['healthy']}/384")
    print(f"    BOS-sink: {counts['BOS-sink']}")
    print(f"    DEAD: {counts['DEAD']}")
    for li in sorted(sick_layers.keys()):
        s = sick_layers[li]
        print(f"    L{li}: {s['sick']} sick ({s['dead']} dead)")

    # Perplexity
    print("  Computing held-out perplexity...")
    ppl, ppl_detail = compute_ppl(model, tokenizer, device, PPL_PROMPTS)
    print(f"    Overall PPL: {ppl:.2f}")
    for p in ppl_detail:
        print(f"    {p['ppl']:>8.1f}  {p['text'][:60]}")

    # Generation
    print("  Generating completions...")
    gens = generate(model, tokenizer, device, GEN_PROMPTS)
    for g in gens:
        print(f"    \"{g['prompt'][:50]}\"")
        print(f"      → {g['completion'][:120]}")

    # Dormancy
    print("  Measuring MLP dormancy...")
    dorm_pct, dorm_layers = measure_dormancy(model)
    print(f"    Overall dormancy: {dorm_pct}%")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        'name': name,
        'attention': {
            'counts': counts,
            'sick_layers': sick_layers,
            'full': {str(k): v for k, v in attn.items()},
        },
        'perplexity': {
            'overall': round(ppl, 2),
            'per_prompt': ppl_detail,
        },
        'generations': gens,
        'dormancy': {
            'overall_pct': dorm_pct,
            'per_layer': dorm_layers,
        },
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Evaluate stock
    stock_results = evaluate_model("Stock BLOOM-1b7", STOCK, device)

    # Evaluate epoch 3
    epoch3_results = evaluate_model("Epoch 3 (Best)", EPOCH3, device)

    # ── Comparison ──
    print(f"\n{'='*60}")
    print("COMPARISON: Stock vs Epoch 3")
    print(f"{'='*60}")

    sc = stock_results['attention']['counts']
    ec = epoch3_results['attention']['counts']
    print(f"\n  Attention Heads:")
    print(f"  {'':20s} {'Stock':>10s} {'Epoch 3':>10s} {'Delta':>10s}")
    print(f"  {'─'*52}")
    for pat in ['healthy', 'BOS-sink', 'DEAD', 'low-entropy']:
        delta = ec[pat] - sc[pat]
        sign = "+" if delta > 0 else ""
        print(f"  {pat:20s} {sc[pat]:>10d} {ec[pat]:>10d} {sign}{delta:>9d}")

    sp = stock_results['perplexity']['overall']
    ep = epoch3_results['perplexity']['overall']
    print(f"\n  Held-out Perplexity:")
    print(f"    Stock:   {sp:.2f}")
    print(f"    Epoch 3: {ep:.2f}")
    print(f"    Delta:   {ep - sp:+.2f} ({'better' if ep < sp else 'worse'})")

    sd = stock_results['dormancy']['overall_pct']
    ed = epoch3_results['dormancy']['overall_pct']
    print(f"\n  MLP Dormancy:")
    print(f"    Stock:   {sd}%")
    print(f"    Epoch 3: {ed}%")
    print(f"    Delta:   {ed - sd:+.2f}%")

    print(f"\n  Generation Comparison (side by side):")
    for sg, eg in zip(stock_results['generations'], epoch3_results['generations']):
        print(f"\n  Prompt: \"{sg['prompt'][:60]}\"")
        print(f"    Stock:   {sg['completion'][:100]}")
        print(f"    Epoch3:  {eg['completion'][:100]}")

    # Save full results
    output = {
        'stock': stock_results,
        'epoch3': epoch3_results,
        'comparison': {
            'attention_delta': {pat: ec[pat] - sc[pat] for pat in sc},
            'ppl_delta': round(ep - sp, 2),
            'dormancy_delta': round(ed - sd, 2),
        }
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Full results saved to: {OUTPUT}")
    print("\nDone.")


if __name__ == "__main__":
    main()

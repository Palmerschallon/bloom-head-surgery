#!/usr/bin/env python3
"""
Diagnostic scan of stock BLOOM-1b7 (bigscience/bloom-1b7).
Same architecture as 560m (24 layers, 16 heads) but 2× hidden dim (2048 vs 1024).

Measures:
1. Baseline perplexity (prose, code, mixed)
2. Head health across ALL 24 layers (BOS-sink, dead, live)
3. Layer ablation sensitivity
4. MLP dormancy per layer
5. Generation quality on standard prompts
"""
import torch
import json
import math
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "bigscience/bloom-1b7"
OUTPUT_PATH = Path("/ember/progressive_checkpoints/bloom1b7_baseline_diagnostic.json")

device = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════

class LayerAblation:
    def __init__(self, model, layers_to_ablate):
        self.model = model
        self.layers = layers_to_ablate
        self.hooks = []

    def __enter__(self):
        for layer_idx in self.layers:
            def make_hook(l):
                def hook_fn(module, input, output):
                    return torch.zeros_like(output)
                return hook_fn
            h = self.model.transformer.h[layer_idx].mlp.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(h)
        return self

    def __exit__(self, *args):
        for h in self.hooks:
            h.remove()


def compute_perplexity(model, tokenizer, texts):
    total_loss, total_tokens = 0, 0
    for text in texts:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model(**tokens, labels=tokens.input_ids)
        total_loss += out.loss.item() * tokens.input_ids.shape[1]
        total_tokens += tokens.input_ids.shape[1]
    return math.exp(total_loss / total_tokens)


def generate(model, tokenizer, prompt, max_tokens=80):
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **toks, max_new_tokens=max_tokens,
            temperature=0.85, top_p=0.92,
            repetition_penalty=1.15, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()


def analyze_heads(model, tokenizer, layers):
    """Analyze attention head patterns across specified layers."""
    diag = ("The container holds the boundary. The boundary defines the edge. "
            "The edge separates inside from outside. Inside and outside depend "
            "on the container. Memory persists through structure, not content. "
            "What remains after deletion is the shape of what was deleted.")
    tokens = tokenizer(diag, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**tokens, output_attentions=True)

    results = {}
    for li in layers:
        attn = out.attentions[li][0]  # [num_heads, seq, seq]
        seq_len = attn.shape[1]
        layer_data = {}
        for h in range(attn.shape[0]):
            ha = attn[h]
            ent = -(ha.clamp(min=1e-10) * ha.clamp(min=1e-10).log()).sum(-1).mean().item()
            bos = ha[:, 0].mean().item()
            local = sum(ha[i, max(0, i-2):i+1].sum().item() for i in range(seq_len)) / seq_len
            if bos > 0.95:
                pat = "DEAD"
            elif bos > 0.5:
                pat = "BOS-sink"
            elif local > 0.6:
                pat = "local"
            else:
                pat = "distributed"
            layer_data[h] = {
                'entropy': round(ent, 3),
                'bos_mass': round(bos, 3),
                'local_mass': round(local, 3),
                'pattern': pat
            }
        results[li] = layer_data
    return results


def measure_dormancy(model, threshold=0.01):
    """Check MLP neuron dormancy across all layers."""
    results = {}
    for i, block in enumerate(model.transformer.h):
        up = block.mlp.dense_h_to_4h.weight.data.norm(dim=1)
        down = block.mlp.dense_4h_to_h.weight.data.norm(dim=0)
        imp = up * down
        nd = (imp < threshold).sum().item()
        nt = imp.numel()
        results[i] = (nd, nt, round(nd / nt * 100, 2))
    return results


# ═══════════════════════════════════════════
# BENCHMARK TEXTS
# ═══════════════════════════════════════════

PROSE_TEXTS = [
    "The model contains layers. Each layer transforms the representation through attention and feedforward processing.",
    "Because the bridge was old, engineers decided to reinforce it before the winter storms arrived.",
    "A word is to a sentence as a brick is to a wall: the smallest unit of a larger structure.",
    "First I notice the pattern. Then I question whether noticing changes the pattern. Then I notice myself questioning.",
    "The river that carved this canyon no longer flows here, but the canyon remembers its shape.",
    "She opened the door to find the room empty. The chair where he always sat was pushed back, as if he had just left.",
    "In the beginning there was the command line. It was a simple thing: you typed words, the machine responded.",
    "The paradox of tolerance states that if a society is tolerant without limit, its ability to be tolerant is eventually seized.",
]

CODE_TEXTS = [
    "def identity(x):\n    return x\n\ndef transform(x):\n    return x + 1",
    "class Container:\n    def __init__(self, boundary):\n        self.boundary = boundary\n        self.contents = []",
    "import hashlib\nimport math\nfrom datetime import datetime",
    "for i in range(len(items)):\n    if items[i].is_valid():\n        results.append(items[i].process())",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "class Observer:\n    def __init__(self):\n        self.subscribers = []\n    def notify(self, event):\n        for sub in self.subscribers:\n            sub.handle(event)",
]

MIXED_TEXTS = [
    "# The structure of attention is a pattern that repeats across scales.\n# From neurons to networks, the same shape emerges.\ndef attention(query, key, value):\n    scores = query @ key.T / math.sqrt(d_k)\n    return softmax(scores) @ value",
    "The boundary between code and prose dissolves when you realize both are sequences of tokens being processed by the same architecture.",
    "Ember was carved from BLOOM-560m through vocabulary surgery. 250,880 tokens became 8,000. What remained was structure.",
]

PROMPTS = {
    "identity": "I am",
    "structural": "The container holds a boundary that",
    "causal": "Because the temperature dropped, the water",
    "code_fn": "def process(items):\n    for item in items:\n        if item.",
    "code_class": "class Boundary:\n    def __init__(self",
    "analogy": "A neuron is to a brain as a transistor is to",
    "negation": "This is not what it appears to be. The real",
    "meta": "# The structure of attention is",
    "import": "import torch\nimport",
    "poetic": "The fog crept in on little cat feet, settling",
    "logical": "All transformers have attention. This model is a transformer. Therefore",
    "counterfactual": "If the model had twice as many layers, it would",
}


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

def main():
    print("=" * 70)
    print("  BLOOM-1b7 BASELINE DIAGNOSTIC")
    print("=" * 70)
    print(f"  Model: {MODEL_ID}")
    print(f"  Device: {device}")

    print("\n  Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Verify architecture
    config = model.config
    print(f"\n  ARCHITECTURE:")
    print(f"    Layers:     {config.n_layer}")
    print(f"    Heads:      {config.n_head}")
    print(f"    Hidden dim: {config.hidden_size}")
    print(f"    Head dim:   {config.hidden_size // config.n_head}")
    print(f"    Vocab size: {config.vocab_size}")
    print(f"    Params:     {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

    results = {
        'model': MODEL_ID,
        'architecture': {
            'n_layer': config.n_layer,
            'n_head': config.n_head,
            'hidden_size': config.hidden_size,
            'head_dim': config.hidden_size // config.n_head,
            'vocab_size': config.vocab_size,
        }
    }

    # ── 1. Perplexity ──
    print(f"\n{'='*70}")
    print("  1. BASELINE PERPLEXITY")
    print(f"{'='*70}")
    prose_ppl = compute_perplexity(model, tokenizer, PROSE_TEXTS)
    code_ppl = compute_perplexity(model, tokenizer, CODE_TEXTS)
    mixed_ppl = compute_perplexity(model, tokenizer, MIXED_TEXTS)
    all_ppl = compute_perplexity(model, tokenizer, PROSE_TEXTS + CODE_TEXTS + MIXED_TEXTS)
    print(f"    Prose:    {prose_ppl:.2f}")
    print(f"    Code:     {code_ppl:.2f}")
    print(f"    Mixed:    {mixed_ppl:.2f}")
    print(f"    Overall:  {all_ppl:.2f}")
    results['ppl'] = {'prose': prose_ppl, 'code': code_ppl, 'mixed': mixed_ppl, 'overall': all_ppl}

    # ── 2. Head Analysis ALL 24 LAYERS ──
    print(f"\n{'='*70}")
    print("  2. HEAD HEALTH (ALL 24 LAYERS)")
    print(f"{'='*70}")
    heads = analyze_heads(model, tokenizer, list(range(24)))

    total_live, total_bos, total_dead = 0, 0, 0
    head_summary = {}
    sick_layers = []

    for li in range(24):
        layer = heads[li]
        n_bos = sum(1 for h in layer.values() if h['pattern'] == 'BOS-sink')
        n_dead = sum(1 for h in layer.values() if h['pattern'] == 'DEAD')
        n_live = 16 - n_bos - n_dead
        total_live += n_live
        total_bos += n_bos
        total_dead += n_dead
        head_summary[li] = {'live': n_live, 'bos_sink': n_bos, 'dead': n_dead}

        marker = ""
        if n_bos > 0 or n_dead > 0:
            marker = "  ◄ SICK" if (n_bos + n_dead) >= 4 else "  ◄"
            sick_layers.append(li)

        print(f"    L{li:>2}: {n_live:>2}/16 live, {n_bos:>2} BOS-sink, {n_dead:>2} dead{marker}")

        for h in sorted(layer.keys()):
            d = layer[h]
            if d['pattern'] in ('BOS-sink', 'DEAD'):
                print(f"          H{h}: {d['pattern']} (BOS={d['bos_mass']:.3f}, ent={d['entropy']:.3f})")

    total_heads = 24 * 16
    print(f"\n    TOTAL: {total_live}/{total_heads} live, {total_bos} BOS-sink, {total_dead} dead")
    print(f"    Sick layers (any BOS-sink/dead): {sick_layers}")

    results['heads'] = {str(li): {str(h): v for h, v in layer.items()} for li, layer in heads.items()}
    results['head_summary'] = {str(k): v for k, v in head_summary.items()}
    results['totals'] = {'live': total_live, 'bos_sink': total_bos, 'dead': total_dead, 'total': total_heads}
    results['sick_layers'] = sick_layers

    # ── 3. Layer Ablation Sensitivity ──
    print(f"\n{'='*70}")
    print("  3. LAYER ABLATION SENSITIVITY")
    print(f"{'='*70}")
    print(f"    {'Layer(s)':<20} {'PPL':>12} {'Δ from normal':>14}")
    print(f"    {'-'*48}")
    print(f"    {'Normal':<20} {all_ppl:>12.2f} {'—':>14}")

    # Test each layer individually
    ablation_results = {}
    for li in range(24):
        label = f"No L{li}"
        with LayerAblation(model, [li]):
            ppl = compute_perplexity(model, tokenizer, PROSE_TEXTS + CODE_TEXTS + MIXED_TEXTS)
        delta = ppl - all_ppl
        marker = " ◄◄◄" if delta > 100000 else (" ◄◄" if delta > 10000 else (" ◄" if delta > 1000 else ""))
        print(f"    {label:<20} {ppl:>12.2f} {delta:>+14.2f}{marker}")
        ablation_results[label] = {'ppl': ppl, 'delta': delta}

    # Test layer groups
    for label, layers in [
        ("No L0-L5 (early)", list(range(0, 6))),
        ("No L6-L11 (mid)", list(range(6, 12))),
        ("No L12-L17 (mid2)", list(range(12, 18))),
        ("No L18-L23 (late)", list(range(18, 24))),
    ]:
        with LayerAblation(model, layers):
            ppl = compute_perplexity(model, tokenizer, PROSE_TEXTS + CODE_TEXTS + MIXED_TEXTS)
        delta = ppl - all_ppl
        print(f"    {label:<20} {ppl:>12.2f} {delta:>+14.2f}")
        ablation_results[label] = {'ppl': ppl, 'delta': delta}

    results['ablation'] = ablation_results

    # ── 4. MLP Dormancy ──
    print(f"\n{'='*70}")
    print("  4. MLP DORMANCY (threshold=0.01)")
    print(f"{'='*70}")
    dorm = measure_dormancy(model)
    total_dormant, total_neurons = 0, 0
    for li in range(24):
        nd, nt, pct = dorm[li]
        total_dormant += nd
        total_neurons += nt
        marker = " ◄◄" if pct > 10 else (" ◄" if pct > 3 else "")
        print(f"    L{li:>2}: {nd:>5}/{nt} ({pct:>5.1f}%){marker}")
    print(f"    TOTAL: {total_dormant}/{total_neurons} ({total_dormant/total_neurons*100:.1f}%)")
    results['dormancy'] = {str(k): v for k, v in dorm.items()}
    results['dormancy_total'] = {'dormant': total_dormant, 'total': total_neurons, 'pct': round(total_dormant/total_neurons*100, 2)}

    # ── 5. Generation ──
    print(f"\n{'='*70}")
    print("  5. GENERATION QUALITY")
    print(f"{'='*70}")
    completions = {}
    for name_p, prompt in PROMPTS.items():
        gen = generate(model, tokenizer, prompt)
        completions[name_p] = gen
        print(f"    [{name_p}] \"{prompt[:50]}\"")
        print(f"      → {gen[:150]}")
    results['completions'] = completions

    # ── 6. Surgery Candidates ──
    print(f"\n{'='*70}")
    print("  6. SURGERY CANDIDATES")
    print(f"{'='*70}")
    candidates = {}
    for li in range(24):
        layer = heads[li]
        sick_heads = [h for h, d in layer.items() if d['pattern'] in ('BOS-sink', 'DEAD')]
        if sick_heads:
            abl_delta = ablation_results.get(f"No L{li}", {}).get('delta', 0)
            candidates[li] = {
                'sick_heads': sick_heads,
                'n_sick': len(sick_heads),
                'n_live': 16 - len(sick_heads),
                'ablation_delta': abl_delta,
                'patterns': {h: layer[h]['pattern'] for h in sick_heads},
            }
            print(f"    L{li:>2}: {len(sick_heads)}/16 sick  heads={sorted(sick_heads)}  ablation_Δ={abl_delta:+.0f}")
    results['surgery_candidates'] = {str(k): v for k, v in candidates.items()}

    if not candidates:
        print("    No sick heads found! Model is healthy.")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Full results saved to {OUTPUT_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()

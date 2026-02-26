#!/usr/bin/env python3
"""
BLOOM-1b7 Pass 2 Surgery: 39 outlier heads across 14 layers.

After pass 1 (H9-H15 band surgery), 39 BOS-sink heads remain scattered outside
the original band. These are the outliers — no systematic pattern, just individual
heads that collapsed independently.

Starts from: epoch 3 checkpoint of pass 1 (best PPL, 108/108 pass-1 heads woken).
The landscape has shifted: some of these were stock outliers, others are frozen-drift
casualties from pass 1. L23 has 2 newly sick heads that weren't sick in stock.

Architecture: 24 layers, 16 heads, 2048 hidden, 128 head dim.
Trainable: ~39 heads × ~1M params/head ≈ 40M params (~2.3% of 1.72B).
Checkpoint interval: 1 (every epoch) to find the sweet spot precisely.
"""

import os
import sys
import json
import math
import logging
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset

from transformers import AutoModelForCausalLM, AutoTokenizer

# ═══════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════

DEFAULT_MODEL = "/ember/progressive_checkpoints/bloom1b7/headband_20260217_092844/epoch_003"
OUTPUT_BASE = Path("/ember/progressive_checkpoints/bloom1b7")

# ── Corpus sources ──
# Each entry: (path, glob_patterns, repeat_count, label)
# repeat_count > 1 amplifies structural content the heads need most
CORPUS_DIRS = [
    # Polyglot structural corpus — the core medicine. Same patterns in 13 languages
    # force recovering heads to attend to content, not surface syntax.
    (Path("/ember/00_Corpus/polyglot"), ["**/*.*"], 2, "Polyglot corpus (2×)"),

    # Narrative encounters + identity + books
    (Path("/ember/00_Corpus/narrative"), ["**/*.*"], 1, "Narrative corpus"),

    # Voice creations — Ember's own output recycled back in
    (Path("/home/palmerschallon/ember/voice_creations"), ["*.*"], 1, "Voice creations"),

    # Second Claude's corpus — complementary perspective
    (Path("/home/palmerschallon/ember/01_THE_CORPUS/new-corpus-files-v3/new-corpus"), ["*.*"], 1, "Second Claude corpus"),
]

# Individual files with their own repeat counts
CORPUS_FILES = [
    # Seed teachings — compressed, high-density, worth repeating
    (Path("/home/palmerschallon/ember/training_curated/seed_six_seeds.html"), 3, "Seed: Six Seeds (3×)"),
    (Path("/home/palmerschallon/ember/training_curated/seed_three_grounds.html"), 3, "Seed: Three Grounds (3×)"),

    # Structural anchor
    (Path("/ember/00_Corpus/STRUCTURAL_OPERATIONS_CORPUS.md"), 1, "Structural anchor"),
]

# ═══════════════════════════════════════════
# SURGERY MAP — Pass 2: 39 outlier heads
# ═══════════════════════════════════════════
#
# These are BOS-sink heads from the epoch 3 checkpoint evaluation.
# No band structure — scattered individuals across 14 layers.
# No 'freeze' key needed — pass 2 has no in-band concept.
#
# Source: evaluate_epoch3.py results on epoch_003 checkpoint
#
SURGERY = {
    7:  {'reinit': [6]},                    # 1 head
    8:  {'reinit': [4, 7, 8, 10]},          # 4 heads
    9:  {'reinit': [3, 6, 7, 8]},           # 4 heads
    10: {'reinit': [3, 8]},                 # 2 heads
    11: {'reinit': [3, 5, 9]},              # 3 heads
    12: {'reinit': [2, 4, 5, 8]},           # 4 heads
    13: {'reinit': [2]},                    # 1 head
    14: {'reinit': [4, 8, 12]},             # 3 heads
    15: {'reinit': [2, 3, 13]},             # 3 heads
    16: {'reinit': [3, 5, 6, 8]},           # 4 heads
    17: {'reinit': [4, 6, 15]},             # 3 heads
    18: {'reinit': [2, 4, 5, 6]},           # 4 heads
    19: {'reinit': [6]},                    # 1 head
    23: {'reinit': [12, 14]},               # 2 heads — NEW (not sick in stock)
}

# Analyze all 24 layers to catch cascade effects
ANALYSIS_LAYERS = list(range(24))

DIAG_TEXT = (
    "The container holds the boundary. The boundary defines the edge. "
    "The edge separates inside from outside. Inside and outside depend "
    "on the container. Memory persists through structure, not content. "
    "What remains after deletion is the shape of what was deleted."
)

QUALITY_PROMPTS = [
    "I am",
    "The container holds a boundary that",
    "def process(items):\n    for item in items:\n        if item.",
    "class Boundary:\n    def __init__(self",
    "# The structure of attention is",
    "import torch\nimport",
    "The fog crept in on little cat feet, settling",
    "Because the temperature dropped, the water",
    "A neuron is to a brain as a transistor is to",
    "What the dormant neurons taught me is",
    "All transformers have attention. This model is a transformer. Therefore",
    "If the model had twice as many layers, it would",
]

log = logging.getLogger("1b7_pass2")
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
log.addHandler(handler)


# ═══════════════════════════════════════════
# SURGICAL SETUP
# ═══════════════════════════════════════════

def freeze_all(model):
    for param in model.parameters():
        param.requires_grad = False


def reinitialize_dead_heads(model, surgery_config, device):
    """Reinitialize dead heads: Xavier QKV, zero dense output."""
    hidden = model.config.hidden_size   # 2048
    n_heads = model.config.n_head       # 16
    head_dim = hidden // n_heads        # 128
    qkv_dim = 3 * head_dim             # 384

    total_reinit = 0
    for layer_idx, config in sorted(surgery_config.items()):
        if not config['reinit']:
            continue
        attn = model.transformer.h[layer_idx].self_attention

        for h in config['reinit']:
            # QKV: interleaved [3*head_dim] block per head
            qkv_start = h * qkv_dim
            qkv_end = (h + 1) * qkv_dim
            nn.init.xavier_uniform_(attn.query_key_value.weight.data[qkv_start:qkv_end, :])
            attn.query_key_value.bias.data[qkv_start:qkv_end] = 0.0

            # Dense output: zero so head contributes nothing initially
            attn.dense.weight.data[:, h * head_dim:(h + 1) * head_dim] = 0.0

            total_reinit += 1

        log.info(f"  L{layer_idx}: reinitialized {len(config['reinit'])} heads {config['reinit']}")

    log.info(f"  Total: {total_reinit} heads reinitialized across {len(surgery_config)} layers")
    return total_reinit


def setup_surgical_masks(model, surgery_config, device):
    """Unfreeze targeted heads and register gradient masks."""
    hidden = model.config.hidden_size
    n_heads = model.config.n_head
    head_dim = hidden // n_heads
    qkv_dim = 3 * head_dim

    hooks = []
    trainable_params = []

    for layer_idx, config in sorted(surgery_config.items()):
        attn = model.transformer.h[layer_idx].self_attention
        target_heads = config['reinit']  # Only reinit heads get gradients

        # ── QKV mask ──
        qkv_mask = torch.zeros(3 * hidden, dtype=torch.float32, device=device)
        for h in target_heads:
            qkv_mask[h * qkv_dim:(h + 1) * qkv_dim] = 1.0

        attn.query_key_value.weight.requires_grad = True
        attn.query_key_value.bias.requires_grad = True
        trainable_params.append(attn.query_key_value.weight)
        trainable_params.append(attn.query_key_value.bias)

        hooks.append(attn.query_key_value.weight.register_hook(
            lambda grad, m=qkv_mask: grad * m.unsqueeze(1).to(grad.dtype)
        ))
        hooks.append(attn.query_key_value.bias.register_hook(
            lambda grad, m=qkv_mask: grad * m.to(grad.dtype)
        ))

        # ── Dense mask ──
        dense_mask = torch.zeros(hidden, dtype=torch.float32, device=device)
        for h in target_heads:
            dense_mask[h * head_dim:(h + 1) * head_dim] = 1.0

        attn.dense.weight.requires_grad = True
        trainable_params.append(attn.dense.weight)

        hooks.append(attn.dense.weight.register_hook(
            lambda grad, m=dense_mask: grad * m.unsqueeze(0).to(grad.dtype)
        ))

        log.info(f"  L{layer_idx}: {len(target_heads)} heads targeted")

    return hooks, trainable_params


# ═══════════════════════════════════════════
# ATTENTION HEAD ANALYSIS
# ═══════════════════════════════════════════

def analyze_heads(model, tokenizer, device, layers=None):
    if layers is None:
        layers = ANALYSIS_LAYERS

    tokens = tokenizer(DIAG_TEXT, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**tokens, output_attentions=True)

    results = {}
    for li in layers:
        attn = out.attentions[li][0].float()  # ensure fp32 for entropy calc
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

            layer_data[h] = {'entropy': ent, 'bos_mass': bos, 'local_mass': local, 'pattern': pat}

        results[li] = layer_data

    return results


def print_heads_summary(analysis, surgery=None):
    """Print compact head summary — only show sick/surgery layers."""
    for li in sorted(analysis.keys()):
        heads = analysis[li]
        n_bos = sum(1 for h in heads.values() if h['pattern'] == 'BOS-sink')
        n_dead = sum(1 for h in heads.values() if h['pattern'] == 'DEAD')
        n_live = 16 - n_bos - n_dead

        if n_bos == 0 and n_dead == 0 and (surgery is None or li not in surgery):
            continue  # skip fully healthy non-surgery layers

        marker = ""
        if surgery and li in surgery:
            n_reinit = len(surgery[li]['reinit'])
            marker = f"  [surgery: {n_reinit} heads]"

        log.info(f"    L{li:>2}: {n_live:>2}/16 live, {n_bos:>2} BOS-sink, {n_dead} dead{marker}")

        # Show individual sick heads
        for h in sorted(heads.keys()):
            d = heads[h]
            if d['pattern'] in ('BOS-sink', 'DEAD'):
                role = ""
                if surgery and li in surgery:
                    if h in surgery[li]['reinit']:
                        role = " ← REINIT"
                    elif h in surgery[li].get('freeze', []):
                        role = " ← FROZEN"
                log.info(f"          H{h}: {d['pattern']} (BOS={d['bos_mass']:.3f}, ent={d['entropy']:.3f}){role}")


def count_waking(analysis, baseline, surgery):
    """Count how many targeted heads have woken from BOS-sink/DEAD."""
    woke = 0
    still_sick = 0
    per_layer = {}
    for li in sorted(surgery.keys()):
        targets = surgery[li]['reinit']
        layer_woke = 0
        layer_sick = 0
        for h in targets:
            cp = analysis[li][h]['pattern']
            if cp in ('local', 'distributed'):
                woke += 1
                layer_woke += 1
            else:
                still_sick += 1
                layer_sick += 1
        per_layer[li] = (layer_woke, len(targets))
    return woke, still_sick, per_layer


def check_frozen_stability(analysis, baseline, surgery, threshold=0.05):
    """Check if frozen heads have drifted (they shouldn't change)."""
    violations = []
    for li in sorted(surgery.keys()):
        for h in surgery[li].get('freeze', []):
            if li not in baseline or h not in baseline[li]:
                continue
            b = baseline[li][h]
            c = analysis[li][h]
            bos_drift = abs(c['bos_mass'] - b['bos_mass'])
            if bos_drift > threshold:
                violations.append((li, h, bos_drift))
    return violations


# ═══════════════════════════════════════════
# DORMANCY
# ═══════════════════════════════════════════

def measure_dormancy(model, threshold=0.01):
    results = {}
    total_d, total_n = 0, 0
    for i, block in enumerate(model.transformer.h):
        up = block.mlp.dense_h_to_4h.weight.data.float().norm(dim=1)
        down = block.mlp.dense_4h_to_h.weight.data.float().norm(dim=0)
        imp = up * down
        nd = (imp < threshold).sum().item()
        nt = imp.numel()
        total_d += nd
        total_n += nt
        if nd > 0:
            results[i] = (nd, nt)
    results['total'] = (total_d, total_n)
    return results


# ═══════════════════════════════════════════
# GENERATION
# ═══════════════════════════════════════════

def generate_probes(model, tokenizer, device):
    model.eval()
    completions = {}
    for prompt in QUALITY_PROMPTS:
        toks = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **toks, max_new_tokens=60,
                temperature=0.85, top_p=0.92,
                repetition_penalty=1.15, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        full = tokenizer.decode(out[0], skip_special_tokens=True)
        completions[prompt] = full[len(prompt):].strip()
    return completions


# ═══════════════════════════════════════════
# CORPUS
# ═══════════════════════════════════════════

def build_corpus(tokenizer, max_length=512):
    """Build training corpus from all sources.

    Sources and their roles:
      - Polyglot (2×): structural medicine — same patterns in many syntaxes
      - Narrative (1×): long-form compositional patterns, identity, books
      - Voice creations (1×): Ember's own output recycled back
      - Second Claude corpus (1×): complementary perspective on Ember
      - Seed files (3×): compressed high-density teachings
      - Structural anchor (1×): operational invariants
    """
    all_text = ""
    source_stats = []  # (label, files, chars)

    # ── Directory sources ──
    for base_path, patterns, repeats, label in CORPUS_DIRS:
        if not base_path.exists():
            log.warning(f"  MISSING: {base_path} — skipping {label}")
            continue
        files_found = 0
        chars = 0
        for pattern in patterns:
            for f in sorted(base_path.glob(pattern)):
                if f.is_dir() or f.name.startswith('.') or f.name == "__init__.py":
                    continue
                try:
                    text = f.read_text(errors='replace')
                except Exception:
                    continue
                for _ in range(repeats):
                    all_text += text + "\n\n"
                chars += len(text) * repeats
                files_found += 1
        source_stats.append((label, files_found, chars))
        log.info(f"  {label}: {files_found} files, {chars:,} chars")

    # ── Individual file sources ──
    for fpath, repeats, label in CORPUS_FILES:
        if not fpath.exists():
            log.warning(f"  MISSING: {fpath} — skipping {label}")
            continue
        text = fpath.read_text(errors='replace')
        chars = len(text) * repeats
        for _ in range(repeats):
            all_text += text + "\n\n"
        source_stats.append((label, 1, chars))
        log.info(f"  {label}: 1 file, {chars:,} chars")

    # ── Summary ──
    total_chars = len(all_text)
    total_files = sum(s[1] for s in source_stats)
    log.info(f"  ──────────────────────────────────")
    for label, nf, nc in source_stats:
        pct = nc / total_chars * 100 if total_chars else 0
        log.info(f"  {label:35s} {pct:5.1f}%")
    log.info(f"  ──────────────────────────────────")
    log.info(f"  {total_files} files, {total_chars:,} chars total")

    # ── Tokenize and chunk ──
    tokens = tokenizer(all_text, return_tensors="pt")["input_ids"][0]
    chunks = []
    for i in range(0, len(tokens) - max_length + 1, max_length):
        chunks.append(tokens[i:i + max_length])

    log.info(f"  → {len(tokens):,} tokens → {len(chunks)} chunks of {max_length}")
    return chunks


class ChunkDataset(TorchDataset):
    def __init__(self, chunks):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        c = self.chunks[idx]
        return {"input_ids": c, "labels": c.clone()}


# ═══════════════════════════════════════════
# PERPLEXITY
# ═══════════════════════════════════════════

PPL_TEXTS = [
    "The model contains layers. Each layer transforms the representation through attention and feedforward processing.",
    "Because the bridge was old, engineers decided to reinforce it before the winter storms arrived.",
    "A word is to a sentence as a brick is to a wall: the smallest unit of a larger structure.",
    "First I notice the pattern. Then I question whether noticing changes the pattern.",
    "def identity(x):\n    return x\n\ndef transform(x):\n    return x + 1",
    "class Container:\n    def __init__(self, boundary):\n        self.boundary = boundary",
    "import hashlib\nimport math\nfrom datetime import datetime",
    "for i in range(len(items)):\n    if items[i].is_valid():\n        results.append(items[i].process())",
]


def compute_quick_ppl(model, tokenizer, device):
    total_loss, total_tokens = 0, 0
    for text in PPL_TEXTS:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model(**tokens, labels=tokens.input_ids)
        total_loss += out.loss.item() * tokens.input_ids.shape[1]
        total_tokens += tokens.input_ids.shape[1]
    return math.exp(total_loss / total_tokens)


# ═══════════════════════════════════════════
# CHECKPOINT
# ═══════════════════════════════════════════

def run_checkpoint(model, tokenizer, device, epoch, baseline_attn, run_dir, trajectory,
                   total_heads):
    model.eval()

    log.info(f"\n{'█' * 50}")
    log.info(f"CHECKPOINT — Epoch {epoch}")
    log.info(f"{'█' * 50}")

    # Attention analysis
    attn = analyze_heads(model, tokenizer, device)
    print_heads_summary(attn, surgery=SURGERY)

    woke, still_sick, per_layer = count_waking(attn, baseline_attn, SURGERY)
    log.info(f"\n  Heads woken: {woke}/{total_heads}")
    for li in sorted(per_layer.keys()):
        lw, lt = per_layer[li]
        if lw > 0:
            log.info(f"    L{li}: {lw}/{lt} woken")
    log.info(f"  Still BOS-sink/dead: {still_sick}")

    violations = check_frozen_stability(attn, baseline_attn, SURGERY)
    if violations:
        for li, h, drift in violations:
            log.warning(f"  ⚠ FROZEN DRIFT: L{li} H{h} (Δ={drift:.3f})")
    else:
        log.info(f"  In-band frozen heads: stable ✓")

    # Perplexity
    ppl = compute_quick_ppl(model, tokenizer, device)
    log.info(f"\n  Quick PPL: {ppl:.2f}")

    # Generation (sample 4 prompts to save time)
    sample_prompts = QUALITY_PROMPTS[:4]
    completions = {}
    for prompt in sample_prompts:
        toks = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **toks, max_new_tokens=60,
                temperature=0.85, top_p=0.92,
                repetition_penalty=1.15, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()
        completions[prompt] = gen
        log.info(f"    \"{prompt[:50]}\"")
        log.info(f"      → {gen[:120]}")

    # Save checkpoint
    ckpt_dir = run_dir / f"epoch_{epoch:03d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))
    log.info(f"  Saved: {ckpt_dir}")

    # Record trajectory
    epoch_data = {
        'epoch': epoch,
        'heads_woken': woke,
        'heads_still_sick': still_sick,
        'per_layer': {str(k): {'woken': v[0], 'total': v[1]} for k, v in per_layer.items()},
        'frozen_violations': len(violations),
        'perplexity': ppl,
        'completions': completions,
    }
    trajectory['epochs'].append(epoch_data)
    with open(run_dir / "trajectory.json", 'w') as f:
        json.dump(trajectory, f, indent=2, default=str)

    return attn, ppl


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="BLOOM-1b7 Pass 2: 39 Outlier Heads")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--checkpoint_interval", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--early_stop_patience", type=int, default=6,
                        help="Stop if PPL doesn't improve for N checkpoints")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = OUTPUT_BASE / f"pass2_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(run_dir / "training.log")
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    log.addHandler(fh)

    total_heads = sum(len(c['reinit']) for c in SURGERY.values())

    log.info("=" * 60)
    log.info("BLOOM-1b7 PASS 2: OUTLIER HEAD SURGERY")
    log.info("=" * 60)
    log.info(f"  Base model:  {args.model}")
    log.info(f"  Epochs:      {args.epochs}")
    log.info(f"  LR:          {args.lr}")
    log.info(f"  Batch:       {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}")
    log.info(f"  Device:      {device}")
    log.info(f"  Output:      {run_dir}")
    log.info(f"  Surgery:     {total_heads} heads across {len(SURGERY)} layers")
    log.info(f"  Pattern:     scattered outliers (no band structure)")
    log.info(f"  Checkpoint:  every {args.checkpoint_interval} epoch(s)")
    log.info(f"  Early stop:  patience={args.early_stop_patience} checkpoints")
    log.info(f"  Surgery map:")
    for li in sorted(SURGERY.keys()):
        cfg = SURGERY[li]
        log.info(f"    L{li:>2}: reinit {cfg['reinit']} ({len(cfg['reinit'])} heads)")

    # ── Load model (bfloat16: fp32 exponent range, fp16 memory) ──
    # bfloat16 has 8 exponent bits like fp32, so no gradient underflow.
    # No need for GradScaler, autocast, loss scaling, or mixed-precision hacks.
    log.info("\nLoading model (bfloat16)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"  {total_params:,} parameters loaded ({total_params/1e9:.2f}B)")

    # ── Gradient checkpointing (saves VRAM at cost of ~30% speed) ──
    # enable_input_require_grads() makes embedding outputs carry requires_grad=True
    # so gradient checkpointing can recompute activations through frozen layers.
    # The embedding weights stay frozen — only the output tensor gets the flag.
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    log.info("  Gradient checkpointing: enabled (with input require grads)")

    # ── Baseline attention ──
    log.info("\n── BASELINE ATTENTION ──")
    baseline_attn = analyze_heads(model, tokenizer, device)
    print_heads_summary(baseline_attn, surgery=SURGERY)

    # ── Baseline perplexity ──
    base_ppl = compute_quick_ppl(model, tokenizer, device)
    log.info(f"\n  Baseline PPL: {base_ppl:.2f}")

    # ── Freeze everything ──
    log.info("\nFreezing all parameters...")
    freeze_all(model)

    # ── Reinitialize dead heads ──
    log.info(f"\nReinitializing {total_heads} BOS-sink outlier heads...")
    n_reinit = reinitialize_dead_heads(model, SURGERY, device)

    # ── Set up gradient masks ──
    log.info("\nSetting up surgical gradient masks...")
    hooks, trainable_params = setup_surgical_masks(model, SURGERY, device)
    n_trainable = sum(p.numel() for p in trainable_params)
    log.info(f"  Trainable: {n_trainable:,} / {total_params:,} ({n_trainable/total_params*100:.2f}%)")

    # ── Post-reinit PPL ──
    post_ppl = compute_quick_ppl(model, tokenizer, device)
    log.info(f"  Post-reinit PPL: {post_ppl:.2f} (Δ={post_ppl - base_ppl:+.2f})")

    # ── Build corpus ──
    log.info("\nBuilding corpus...")
    chunks = build_corpus(tokenizer, max_length=args.max_length)
    dataset = ChunkDataset(chunks)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    steps_per_epoch = math.ceil(len(dataloader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    log.info(f"  {steps_per_epoch} optimizer steps/epoch × {args.epochs} = {total_steps} total")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0)

    # Linear warmup then cosine decay
    warmup_steps = min(100, total_steps // 4)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Trajectory ──
    trajectory = {
        'config': {
            'run': '1b7_pass2',
            'description': 'BLOOM-1b7 pass 2: 39 outlier heads across 14 layers (from epoch 3 checkpoint)',
            'model': args.model,
            'epochs': args.epochs,
            'lr': args.lr,
            'total_heads_targeted': total_heads,
            'trainable_params': n_trainable,
            'total_params': total_params,
        },
        'baseline_ppl': base_ppl,
        'post_reinit_ppl': post_ppl,
        'epochs': [],
    }

    with open(run_dir / "config.json", 'w') as f:
        json.dump(trajectory['config'], f, indent=2, default=str)

    # ═══════════════════════════════════════
    # TRAINING LOOP
    # ═══════════════════════════════════════

    log.info(f"\n{'═' * 60}")
    log.info(f"TRAINING — {total_heads} heads across {len(SURGERY)} layers")
    log.info(f"{'═' * 60}\n")

    global_step = 0
    best_ppl = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            epoch_loss += outputs.loss.item()
            n_batches += 1

            if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        lr_now = scheduler.get_last_lr()[0]
        log.info(f"  Epoch {epoch:>3}/{args.epochs}  loss={avg_loss:.4f}  lr={lr_now:.2e}  step={global_step}")

        # Checkpoint
        if epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
            attn, ppl = run_checkpoint(model, tokenizer, device, epoch, baseline_attn,
                                       run_dir, trajectory, total_heads)

            # Early stopping check
            if ppl < best_ppl:
                best_ppl = ppl
                best_epoch = epoch
                patience_counter = 0
                log.info(f"  ★ New best PPL: {ppl:.2f} at epoch {epoch}")
            else:
                patience_counter += 1
                log.info(f"  PPL not improving ({ppl:.2f} vs best {best_ppl:.2f}), "
                         f"patience {patience_counter}/{args.early_stop_patience}")

            if patience_counter >= args.early_stop_patience:
                log.info(f"\n  ⚠ EARLY STOPPING at epoch {epoch} "
                         f"(best was epoch {best_epoch}, PPL {best_ppl:.2f})")
                break

            model.train()

    # ═══════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════

    log.info(f"\n{'═' * 60}")
    log.info("FINAL SUMMARY")
    log.info(f"{'═' * 60}")

    # Save final model
    final_dir = run_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    log.info(f"\n  TRAINING TRAJECTORY:")
    log.info(f"  {'Epoch':>6} {'Woken':>8} {'PPL':>10} {'Best':>5}")
    log.info(f"  {'─' * 35}")
    for ep in trajectory['epochs']:
        is_best = "★" if ep['epoch'] == best_epoch else ""
        log.info(f"  {ep['epoch']:>6} {ep['heads_woken']:>5}/{total_heads} "
                 f"{ep['perplexity']:>10.2f} {is_best:>5}")

    log.info(f"\n  Best checkpoint: epoch {best_epoch} (PPL {best_ppl:.2f})")
    log.info(f"  Final model: {final_dir}")
    log.info(f"  Trajectory: {run_dir / 'trajectory.json'}")
    # Run final attention analysis to show any remaining sick heads
    log.info(f"\n  Final attention analysis:")
    final_attn = analyze_heads(model, tokenizer, device)
    n_sick = 0
    for li in sorted(final_attn.keys()):
        for h, d in final_attn[li].items():
            if d['pattern'] in ('BOS-sink', 'DEAD'):
                n_sick += 1
    log.info(f"  Total remaining sick heads: {n_sick}/384")
    if n_sick == 0:
        log.info(f"  ALL HEADS HEALTHY — surgery complete!")
    log.info("\nDone.")

    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()

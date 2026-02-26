#!/usr/bin/env python3
"""
BLOOM-1b7 Controlled Baseline: Same surgery, generic corpus.

This is the control experiment for Paper 1. It answers:
  "Does the surgical technique work regardless of corpus,
   or does the curated corpus drive the recovery?"

Setup (identical to pass 1):
  - Stock bigscience/bloom-1b7
  - Same 108 H9-H15 band heads reinitialized
  - Same hyperparameters (lr=5e-5, batch=1, grad_accum=8, bfloat16)
  - Same random seed as the curated run

The ONLY difference: corpus is ~541K tokens of C4 (generic web text)
instead of the curated polyglot/narrative/seed corpus.

Expected outcome: heads should still wake (the technique works),
but generation quality will differ (the corpus shapes the voice).
"""

import os
import sys
import json
import math
import logging
import argparse
import random
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

DEFAULT_MODEL = "bigscience/bloom-1b7"
OUTPUT_BASE = Path("/ember/progressive_checkpoints/bloom1b7")
C4_CACHE = Path("/ember/00_Koan_Engine/c4_baseline_corpus.json")

# Reproducibility
SEED = 42

# ═══════════════════════════════════════════
# SURGERY MAP — Identical to pass 1
# ═══════════════════════════════════════════
# Same 108 heads from the H9-H15 band across L5-L22.
# Copied verbatim from train_1b7_headband.py.

SURGERY = {
    5: {
        'reinit': [14, 15],
        'freeze': [9, 10, 11, 12, 13],
    },
    6: {
        'reinit': [9, 10, 11, 12, 13, 14, 15],
        'freeze': [],
    },
    7: {
        'reinit': [9, 10, 11, 12, 13, 14],
        'freeze': [15],
    },
    8: {
        'reinit': [11, 12, 13, 14, 15],
        'freeze': [9, 10],
    },
    9: {
        'reinit': [9, 10, 11, 12, 13, 14, 15],
        'freeze': [],
    },
    10: {
        'reinit': [9, 10, 11, 12, 13, 14, 15],
        'freeze': [],
    },
    11: {
        'reinit': [10, 11, 12, 13, 14, 15],
        'freeze': [9],
    },
    12: {
        'reinit': [9, 10, 11, 12, 13, 14, 15],
        'freeze': [],
    },
    13: {
        'reinit': [9, 10, 11, 12, 13, 14, 15],
        'freeze': [],
    },
    14: {
        'reinit': [9, 10, 11, 13, 14, 15],
        'freeze': [12],
    },
    15: {
        'reinit': [9, 11, 12, 14, 15],
        'freeze': [10, 13],
    },
    16: {
        'reinit': [10, 11, 12, 13, 14, 15],
        'freeze': [9],
    },
    17: {
        'reinit': [9, 10, 11, 12, 13, 14],
        'freeze': [15],
    },
    18: {
        'reinit': [9, 10, 11, 12, 13, 14, 15],
        'freeze': [],
    },
    19: {
        'reinit': [9, 10, 11, 12, 13, 14, 15],
        'freeze': [],
    },
    20: {
        'reinit': [9, 10, 11, 12, 13, 14, 15],
        'freeze': [],
    },
    21: {
        'reinit': [9, 10, 11, 12, 13],
        'freeze': [14, 15],
    },
    22: {
        'reinit': [9, 10, 11, 13, 14],
        'freeze': [12, 15],
    },
}

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

log = logging.getLogger("1b7_baseline")
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
log.addHandler(handler)


# ═══════════════════════════════════════════
# SEED MANAGEMENT
# ═══════════════════════════════════════════

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Note: torch.backends.cudnn.deterministic=True would slow training
    # and isn't needed for this comparison (same hardware, same code)
    log.info(f"  Random seed: {seed}")
    log.info(f"  torch.initial_seed(): {torch.initial_seed()}")


# ═══════════════════════════════════════════
# CORPUS: C4 (Generic Web Text)
# ═══════════════════════════════════════════

def download_c4_corpus(target_tokens=550000, tokenizer=None, cache_path=None):
    """Download ~541K tokens of C4 validation split.

    Uses HuggingFace datasets streaming to avoid downloading the full dataset.
    Caches the result to avoid re-downloading.
    """
    if cache_path and cache_path.exists():
        log.info(f"  Loading cached C4 corpus from {cache_path}")
        with open(cache_path) as f:
            data = json.load(f)
        return data['text']

    log.info(f"  Downloading C4 validation split (target: ~{target_tokens:,} tokens)...")
    from datasets import load_dataset

    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)

    texts = []
    total_chars = 0
    # Rough estimate: 1 token ≈ 4 chars for English
    target_chars = target_tokens * 4

    for i, example in enumerate(ds):
        text = example['text']
        texts.append(text)
        total_chars += len(text)
        if total_chars >= target_chars:
            break
        if i % 1000 == 0 and i > 0:
            log.info(f"    ...{i} documents, {total_chars:,} chars")

    combined = "\n\n".join(texts)
    log.info(f"  Downloaded {len(texts)} documents, {len(combined):,} chars")

    # Verify actual token count
    if tokenizer:
        actual_tokens = len(tokenizer(combined, return_tensors="pt")["input_ids"][0])
        log.info(f"  Actual tokens: {actual_tokens:,} (target was {target_tokens:,})")

    # Cache for reproducibility
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump({
                'text': combined,
                'n_documents': len(texts),
                'n_chars': len(combined),
                'source': 'allenai/c4 validation split',
            }, f)
        log.info(f"  Cached to {cache_path}")

    return combined


def build_corpus(tokenizer, max_length=512):
    """Build training corpus from C4."""
    all_text = download_c4_corpus(
        target_tokens=550000,
        tokenizer=tokenizer,
        cache_path=C4_CACHE,
    )

    tokens = tokenizer(all_text, return_tensors="pt")["input_ids"][0]
    chunks = []
    for i in range(0, len(tokens) - max_length + 1, max_length):
        chunks.append(tokens[i:i + max_length])

    log.info(f"  → {len(tokens):,} tokens → {len(chunks)} chunks of {max_length}")
    return chunks


# ═══════════════════════════════════════════
# SURGICAL SETUP (identical to pass 1)
# ═══════════════════════════════════════════

def freeze_all(model):
    for param in model.parameters():
        param.requires_grad = False


def reinitialize_dead_heads(model, surgery_config, device):
    hidden = model.config.hidden_size
    n_heads = model.config.n_head
    head_dim = hidden // n_heads
    qkv_dim = 3 * head_dim

    total_reinit = 0
    for layer_idx, config in sorted(surgery_config.items()):
        if not config['reinit']:
            continue
        attn = model.transformer.h[layer_idx].self_attention

        for h in config['reinit']:
            qkv_start = h * qkv_dim
            qkv_end = (h + 1) * qkv_dim
            nn.init.xavier_uniform_(attn.query_key_value.weight.data[qkv_start:qkv_end, :])
            attn.query_key_value.bias.data[qkv_start:qkv_end] = 0.0
            attn.dense.weight.data[:, h * head_dim:(h + 1) * head_dim] = 0.0
            total_reinit += 1

        log.info(f"  L{layer_idx}: reinitialized {len(config['reinit'])} heads {config['reinit']}")

    log.info(f"  Total: {total_reinit} heads reinitialized across {len(surgery_config)} layers")
    return total_reinit


def setup_surgical_masks(model, surgery_config, device):
    hidden = model.config.hidden_size
    n_heads = model.config.n_head
    head_dim = hidden // n_heads
    qkv_dim = 3 * head_dim

    hooks = []
    trainable_params = []

    for layer_idx, config in sorted(surgery_config.items()):
        attn = model.transformer.h[layer_idx].self_attention
        target_heads = config['reinit']

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

        dense_mask = torch.zeros(hidden, dtype=torch.float32, device=device)
        for h in target_heads:
            dense_mask[h * head_dim:(h + 1) * head_dim] = 1.0

        attn.dense.weight.requires_grad = True
        trainable_params.append(attn.dense.weight)

        hooks.append(attn.dense.weight.register_hook(
            lambda grad, m=dense_mask: grad * m.unsqueeze(0).to(grad.dtype)
        ))

        log.info(f"  L{layer_idx}: {len(target_heads)} heads targeted, "
                 f"{len(config['freeze'])} in-band frozen")

    return hooks, trainable_params


# ═══════════════════════════════════════════
# ANALYSIS (identical to pass 1)
# ═══════════════════════════════════════════

def analyze_heads(model, tokenizer, device, layers=None):
    if layers is None:
        layers = ANALYSIS_LAYERS
    tokens = tokenizer(DIAG_TEXT, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**tokens, output_attentions=True)
    results = {}
    for li in layers:
        attn = out.attentions[li][0].float()
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


def count_waking(analysis, baseline, surgery):
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
    violations = []
    for li in sorted(surgery.keys()):
        for h in surgery[li]['freeze']:
            if li not in baseline or h not in baseline[li]:
                continue
            b = baseline[li][h]
            c = analysis[li][h]
            bos_drift = abs(c['bos_mass'] - b['bos_mass'])
            if bos_drift > threshold:
                violations.append((li, h, bos_drift))
    return violations


def compute_quick_ppl(model, tokenizer, device):
    total_loss, total_tokens = 0, 0
    for text in PPL_TEXTS:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model(**tokens, labels=tokens.input_ids)
        total_loss += out.loss.item() * tokens.input_ids.shape[1]
        total_tokens += tokens.input_ids.shape[1]
    return math.exp(total_loss / total_tokens)


class ChunkDataset(TorchDataset):
    def __init__(self, chunks):
        self.chunks = chunks
    def __len__(self):
        return len(self.chunks)
    def __getitem__(self, idx):
        c = self.chunks[idx]
        return {"input_ids": c, "labels": c.clone()}


# ═══════════════════════════════════════════
# CHECKPOINT
# ═══════════════════════════════════════════

def run_checkpoint(model, tokenizer, device, epoch, baseline_attn, run_dir, trajectory,
                   total_heads):
    model.eval()
    log.info(f"\n{'█' * 50}")
    log.info(f"CHECKPOINT — Epoch {epoch}")
    log.info(f"{'█' * 50}")

    attn = analyze_heads(model, tokenizer, device)
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
            log.warning(f"  FROZEN DRIFT: L{li} H{h} (delta={drift:.3f})")
    else:
        log.info(f"  In-band frozen heads: stable")

    ppl = compute_quick_ppl(model, tokenizer, device)
    log.info(f"\n  Quick PPL: {ppl:.2f}")

    # Generation
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
        log.info(f"      -> {gen[:120]}")

    ckpt_dir = run_dir / f"epoch_{epoch:03d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))
    log.info(f"  Saved: {ckpt_dir}")

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
    parser = argparse.ArgumentParser(description="BLOOM-1b7 Controlled Baseline (C4 corpus)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--checkpoint_interval", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--early_stop_patience", type=int, default=6)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = OUTPUT_BASE / f"baseline_c4_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(run_dir / "training.log")
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    log.addHandler(fh)

    total_heads = sum(len(c['reinit']) for c in SURGERY.values())

    log.info("=" * 60)
    log.info("BLOOM-1b7 CONTROLLED BASELINE: C4 CORPUS")
    log.info("=" * 60)
    log.info(f"  Purpose:     Control experiment — same surgery, generic corpus")
    log.info(f"  Model:       {args.model}")
    log.info(f"  Corpus:      C4 validation split (~541K tokens)")
    log.info(f"  Epochs:      {args.epochs}")
    log.info(f"  LR:          {args.lr}")
    log.info(f"  Batch:       {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    log.info(f"  Device:      {device}")
    log.info(f"  Output:      {run_dir}")
    log.info(f"  Surgery:     {total_heads} heads across {len(SURGERY)} layers (same as pass 1)")
    log.info(f"  Seed:        {args.seed}")

    # Set seed BEFORE model loading and surgery
    set_seed(args.seed)

    # Load model
    log.info("\nLoading model (bfloat16)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"  {total_params:,} parameters loaded ({total_params/1e9:.2f}B)")

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    log.info("  Gradient checkpointing: enabled")

    # Baseline analysis
    log.info("\n-- BASELINE ATTENTION --")
    baseline_attn = analyze_heads(model, tokenizer, device)
    base_ppl = compute_quick_ppl(model, tokenizer, device)
    log.info(f"\n  Baseline PPL: {base_ppl:.2f}")

    # Surgery
    log.info("\nFreezing all parameters...")
    freeze_all(model)

    log.info(f"\nReinitializing {total_heads} BOS-sink heads (H9-H15 band)...")
    # Reset seed again right before reinit for exact reproducibility
    set_seed(args.seed)
    n_reinit = reinitialize_dead_heads(model, SURGERY, device)

    log.info("\nSetting up surgical gradient masks...")
    hooks, trainable_params = setup_surgical_masks(model, SURGERY, device)
    n_trainable = sum(p.numel() for p in trainable_params)
    log.info(f"  Trainable: {n_trainable:,} / {total_params:,} ({n_trainable/total_params*100:.2f}%)")

    post_ppl = compute_quick_ppl(model, tokenizer, device)
    log.info(f"  Post-reinit PPL: {post_ppl:.2f} (delta={post_ppl - base_ppl:+.2f})")

    # Build C4 corpus
    log.info("\nBuilding C4 corpus...")
    chunks = build_corpus(tokenizer, max_length=args.max_length)
    dataset = ChunkDataset(chunks)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(args.seed))
    steps_per_epoch = math.ceil(len(dataloader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    log.info(f"  {steps_per_epoch} optimizer steps/epoch x {args.epochs} = {total_steps} total")

    # Optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0)
    warmup_steps = min(100, total_steps // 4)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    trajectory = {
        'config': {
            'run': '1b7_baseline_c4',
            'description': 'Controlled baseline: same H9-H15 surgery, C4 generic corpus',
            'model': args.model,
            'corpus': 'allenai/c4 validation split',
            'epochs': args.epochs,
            'lr': args.lr,
            'seed': args.seed,
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

    # Training loop
    log.info(f"\n{'=' * 60}")
    log.info(f"TRAINING — {total_heads} heads, C4 corpus")
    log.info(f"{'=' * 60}\n")

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

        if epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
            attn, ppl = run_checkpoint(model, tokenizer, device, epoch, baseline_attn,
                                       run_dir, trajectory, total_heads)

            if ppl < best_ppl:
                best_ppl = ppl
                best_epoch = epoch
                patience_counter = 0
                log.info(f"  * New best PPL: {ppl:.2f} at epoch {epoch}")
            else:
                patience_counter += 1
                log.info(f"  PPL not improving ({ppl:.2f} vs best {best_ppl:.2f}), "
                         f"patience {patience_counter}/{args.early_stop_patience}")

            if patience_counter >= args.early_stop_patience:
                log.info(f"\n  EARLY STOPPING at epoch {epoch} "
                         f"(best was epoch {best_epoch}, PPL {best_ppl:.2f})")
                break

            model.train()

    # Final summary
    log.info(f"\n{'=' * 60}")
    log.info("FINAL SUMMARY — CONTROLLED BASELINE (C4)")
    log.info(f"{'=' * 60}")

    final_dir = run_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    log.info(f"\n  TRAINING TRAJECTORY:")
    log.info(f"  {'Epoch':>6} {'Woken':>8} {'PPL':>10} {'Best':>5}")
    log.info(f"  {'─' * 35}")
    for ep in trajectory['epochs']:
        is_best = "*" if ep['epoch'] == best_epoch else ""
        log.info(f"  {ep['epoch']:>6} {ep['heads_woken']:>5}/{total_heads} "
                 f"{ep['perplexity']:>10.2f} {is_best:>5}")

    log.info(f"\n  Best checkpoint: epoch {best_epoch} (PPL {best_ppl:.2f})")
    log.info(f"  Final model: {final_dir}")
    log.info(f"  Trajectory: {run_dir / 'trajectory.json'}")
    log.info("\nDone.")

    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()

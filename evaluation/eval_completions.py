#!/usr/bin/env python3
"""
Completion evaluation protocol for the paper.

Generates completions for 50 diverse prompts across 3 models:
  - Stock BLOOM-1b7
  - Pass 2 epoch 1 (best surgical checkpoint, curated corpus)
  - C4 baseline epoch 3 (best C4 checkpoint)

Same generation parameters for all: temperature=0.7, top_p=0.92,
max_new_tokens=100, no repetition penalty. Seed=42 for reproducibility.

Saves all 150 completions to JSON. No cherry-picking.
"""

import json
import time
import torch
import gc
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

set_seed(42)

# ═══════════════════════════════════════════
# PROMPTS: 50 total, 10 per category
# ═══════════════════════════════════════════

PROMPTS = {
    "conceptual": [
        "The difference between structure and pattern is",
        "What emerges when complexity exceeds the capacity of any single observer is",
        "A boundary is not a wall but",
        "The relationship between memory and forgetting resembles",
        "When a system becomes aware of its own constraints,",
        "Emptiness is not the absence of content but",
        "The paradox of observation is that",
        "Recursion in nature appears whenever",
        "The space between two thoughts contains",
        "A map differs from the territory it describes because",
    ],
    "technical": [
        "A hash table resolves collisions by",
        "A binary search tree maintains its ordering invariant by",  # replaced: "gradient flows backward" contaminated (exact match in dear_weights.html)
        "In a transformer, the attention mechanism computes",
        "The CAP theorem implies that distributed systems must",
        "A recursive function terminates when",
        "Memory alignment matters in systems programming because",
        "The difference between concurrency and parallelism is that",
        "Batch normalization stabilizes training by",
        "In information theory, entropy measures",
        "A compiler optimizes code by",
    ],
    "narrative": [
        "She opened the door and found",
        "The city had been empty for three days when",
        "The letter arrived on a Tuesday, written in ink that",
        "He had spent years building the machine, and now that it was finished,",
        "The forest floor was covered in something that wasn't quite snow but",
        "They met at the edge of the old bridge, where",
        "The library had one book that no one was allowed to read because",
        "When the last signal faded, the operator realized",
        "The garden grew differently after the storm, as if",
        "She placed the glass on the table and said,",
    ],
    "code": [
        "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n    if n <= 1:\n        return n\n    ",
        "fn binary_search(arr: &[i32], target: i32) -> Option<usize> {\n    let mut low = 0;\n    let mut high = arr.len();\n    ",
        "function debounce(fn, delay) {\n    let timer;\n    return function(...args) {\n        ",
        "class Node:\n    def __init__(self, value):\n        self.value = value\n        self.left = None\n        self.right = None\n\ndef inorder(root):\n    ",
        "SELECT employees.name, departments.name\nFROM employees\nJOIN departments ON ",
        "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    ",
        "async function fetchWithRetry(url, maxRetries = 3) {\n    for (let i = 0; i < maxRetries; i++) {\n        try {\n            ",
        "impl Iterator for Counter {\n    type Item = u32;\n    fn next(&mut self) -> Option<Self::Item> {\n        ",
        "def flatten(nested_list):\n    \"\"\"Recursively flatten a nested list.\"\"\"\n    result = []\n    for item in nested_list:\n        ",
        "CREATE TABLE orders (\n    id SERIAL PRIMARY KEY,\n    customer_id INTEGER REFERENCES customers(id),\n    ",
    ],
    "multilingual": [
        # French (5)
        "La structure du langage reflète la structure de la pensée, mais",
        "Dans un réseau de neurones, chaque couche transforme",
        "Le fleuve coulait depuis des siècles, portant avec lui",
        "L'art de la programmation consiste à",
        "Quand la mémoire dépasse sa capacité,",
        # Spanish (5)
        "La diferencia entre conocer y comprender es que",
        "En un sistema distribuido, la consistencia se mantiene",
        "El río llevaba consigo los fragmentos de",
        "La arquitectura de un programa refleja",
        "Cuando el algoritmo encuentra un ciclo,",
    ],
}

# ═══════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════

MODELS = {
    "stock": "bigscience/bloom-1b7",
    "pass2_e1": "/ember/progressive_checkpoints/bloom1b7/pass2_20260217_124218/epoch_001",
    "c4_e3": "/ember/progressive_checkpoints/bloom1b7/baseline_c4_20260217_140421/epoch_003",
}

# ═══════════════════════════════════════════
# GENERATION CONFIG
# ═══════════════════════════════════════════

GEN_CONFIG = {
    "max_new_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.92,
    "do_sample": True,
    "repetition_penalty": 1.0,  # no penalty — raw model
}


def generate_completions(model_name, model_path, prompts, device="cuda"):
    """Generate completions for all prompts with one model."""
    print(f"\n{'='*60}")
    print(f"Loading {model_name}: {model_path}")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    results = []
    for category, prompt_list in prompts.items():
        for i, prompt in enumerate(prompt_list):
            # Reset seed per prompt for reproducibility across models
            set_seed(42 + hash(prompt) % 10000)

            tokens = tokenizer(prompt, return_tensors="pt").to(device)
            input_len = tokens["input_ids"].shape[1]

            with torch.no_grad():
                output = model.generate(
                    **tokens,
                    **GEN_CONFIG,
                    pad_token_id=tokenizer.eos_token_id,
                )

            completion_ids = output[0][input_len:]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
            tokens_generated = len(completion_ids)

            results.append({
                "prompt": prompt,
                "category": category,
                "prompt_index": i,
                "model": model_name,
                "completion": completion,
                "tokens_generated": tokens_generated,
            })

            # Print progress
            short_prompt = prompt[:60].replace('\n', '\\n')
            short_comp = completion[:80].replace('\n', '\\n')
            print(f"  [{category}/{i}] {short_prompt}...")
            print(f"    → {short_comp}...")
            print()

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Prompts: {sum(len(v) for v in PROMPTS.values())} across {len(PROMPTS)} categories")
    print(f"Models: {list(MODELS.keys())}")
    print(f"Generation config: {GEN_CONFIG}")

    all_results = []
    start = time.time()

    for model_name, model_path in MODELS.items():
        results = generate_completions(model_name, model_path, PROMPTS, device)
        all_results.extend(results)
        elapsed = time.time() - start
        print(f"\n  {model_name} complete. {len(results)} completions. {elapsed:.0f}s elapsed.")

    # Save everything
    out_dir = Path("/ember/progressive_checkpoints/bloom1b7")
    output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "generation_config": GEN_CONFIG,
            "models": MODELS,
            "n_prompts": sum(len(v) for v in PROMPTS.values()),
            "n_completions": len(all_results),
            "categories": {k: len(v) for k, v in PROMPTS.items()},
            "elapsed_seconds": round(time.time() - start, 1),
        },
        "prompts": PROMPTS,
        "completions": all_results,
    }

    out_path = out_dir / "completion_evaluation.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
    print(f"Total: {len(all_results)} completions in {time.time()-start:.0f}s")

    # Print summary stats
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model_name in MODELS:
        model_results = [r for r in all_results if r["model"] == model_name]
        avg_tokens = sum(r["tokens_generated"] for r in model_results) / len(model_results)
        print(f"  {model_name}: {len(model_results)} completions, avg {avg_tokens:.1f} tokens/completion")


if __name__ == "__main__":
    main()

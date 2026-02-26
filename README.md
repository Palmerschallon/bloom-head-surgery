# Surgical Repair of Collapsed Attention Heads in ALiBi Transformers

Surgical reinitialization recovers 98.7% of collapsed attention heads in BLOOM transformers. Includes cross-scale diagnosis (560M-7.1B), two-phenomenon framework (functional redistribution vs local degradation), and evidence that healthy heads also improve when reinitialized. Pretrained attention is a local minimum, not a global one.

## Key Findings

- **31-44% of BLOOM attention heads are collapsed** due to ALiBi positional encoding, attending almost entirely to BOS
- **Surgical reinitialization** (Xavier Q/K/V reinit + zeroed output + gradient masks) recovers 379/384 heads in two passes
- **Corpus content determines specialization, not recovery**: both curated and C4 corpora produce identical head recovery (108/108)
- **Two distinct post-surgical phenomena**: early functional redistribution (beneficial) vs late local degradation (pathological)
- **Healthy heads improve too**: reinitializing mostly-healthy H5 column produces 25% better training PPL than stock (12.70 vs 16.99)

## Repository Structure

```
paper/
  paper.md                    # Full paper draft
  figures/                    # All publication figures (7 main + supplementary)
diagnosis/
  diagnose_bloom1b7.py        # Single-model diagnostic (384 heads, BOS mass + entropy)
  diagnose_bloom_family.py    # Cross-scale diagnostic (560m, 1b7, 3b, 7b1)
surgery/
  train_1b7_headband.py       # Pass 1: H9-H15 band surgery (108 heads, curated corpus)
  train_1b7_pass2.py          # Pass 2: 39 outlier heads
  train_1b7_baseline.py       # Control: same surgery, C4 corpus
evaluation/
  evaluate_epoch3.py          # Stock vs Pass 1 E3 comparison
  eval_completions.py         # 50-prompt generation evaluation (3 conditions)
  figure_threeway.py          # 3-way comparison (stock, curated, C4)
  figure_fourway.py           # 4-way comparison at matched epochs
figures_scripts/
  figure4_matched_epoch.py    # Two-phenomenon figure (curated E3 vs C4 E3)
  figure4_global_redistribution.py  # Full 24x16 redistribution heatmap
  figure4_sidebyside.py       # Side-by-side redistribution comparison
  analyze_frozen_drift.py     # Frozen head drift + column propagation analysis
  gen_figure_cross_scale_heatmap.py # Cross-scale BOS-sink band figure
```

## Requirements

- Python 3.10+
- PyTorch 2.9+ (bfloat16 support required)
- HuggingFace Transformers 4.57+
- NVIDIA GPU with 16GB+ VRAM (tested on RTX 5070 Ti)
- CUDA 12.8+

```bash
pip install torch transformers accelerate matplotlib numpy scipy
```

## Note on Paths

Scripts contain hardcoded paths from the development environment (`/ember/...`). You will need to update corpus paths, checkpoint directories, and output paths for your setup. The scripts are research code released for reproducibility, not a polished library.

## Quick Start

### Diagnose a BLOOM model

```bash
python diagnosis/diagnose_bloom1b7.py
```

Produces `bloom1b7_baseline_diagnostic.json` with per-head BOS mass, entropy, and classification.

### Run surgery (Pass 1)

```bash
python surgery/train_1b7_headband.py
```

Targets H9-H15 band across layers 5-22. Trains ~118M parameters (6.9%) with gradient masks freezing the rest. Checkpoints saved per epoch.

### Run surgery (Pass 2)

```bash
python surgery/train_1b7_pass2.py
```

Starts from Pass 1 best checkpoint. Targets remaining 39 collapsed outlier heads.

## Hardware

All experiments ran on a single NVIDIA RTX 5070 Ti (16GB VRAM) using bfloat16 precision with gradient checkpointing. The technique's reliance on freezing most parameters makes it feasible on modest hardware.

## Citation

```
@misc{schallon2026surgical,
  title={Surgical Repair of Collapsed Attention Heads in ALiBi Transformers},
  author={Schallon, Palmer},
  year={2026},
  howpublished={\url{https://github.com/Palmerschallon/bloom-head-surgery}}
}
```

## License

MIT

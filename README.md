# SWE Speedrun Scaffold

This repository packages a reproducible workflow for exploring SWE-agent speedrun style datasets and fine-tuning [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) with LoRA adapters. It contains:

- A **read-only SWE-agent configuration** for guided repository walkthroughs
- Dataset utilities that curate prompt/response pairs from SWE-bench–style corpora
- Training scripts that apply PEFT/LoRA adapters to Qwen
- Evaluation helpers and lightweight CLI scripts for dataset QA

## Dataset Splits

This scaffold uses **dev** and **test** splits:
- **dev**: Training split (by default uses all data when `train_only=True`)
- **test**: Evaluation/holdout split (empty by default; use `--holdout-fraction` to enable)

The holdout ratio can be configured via the `holdout_fraction` parameter in configuration files or the `--holdout-fraction` CLI flag.

## Project Layout

```
configs/
  scaffold_readonly.yaml    # Read-only SWE-agent session template
  speedrun.json              # Training and dataset configuration
swe_scaffold/
  config.py                 # Dataclasses for dataset and training hyperparameters
  dataset.py                # Dataset builders and HF dataset adapters
  evaluation.py             # Perplexity + bootstrap utilities
  labels.py                 # Conversation labelling heuristics
  training.py               # LoRA-friendly Trainer glue
scripts/
  build_speedrun_dataset.py # Convert upstream HF dataset into JSONL cache
  summarize_dataset.py      # Quick statistics for JSONL payloads
training/
  train_qwen_speedrun.py    # LoRA fine-tuning entrypoint
  evaluate_qwen_speedrun.py # Validation script for trained checkpoints
tests/
  test_dataset_splits.py    # Tests for dev/test split handling
  test_training_config.py   # Tests for training configuration
```

## Quickstart

1. **Install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. **Prepare dataset**

Build the full SWE-bench train split:

```bash
python scripts/build_speedrun_dataset.py --dataset SWE-bench/SWE-bench --output data/processed/swe-speedrun.jsonl
python scripts/summarize_dataset.py data/processed/swe-speedrun.jsonl
```

Or with a limit and holdout for testing:

```bash
python scripts/build_speedrun_dataset.py --dataset SWE-bench/SWE-bench --limit 500 --holdout-fraction 0.1 --output data/processed/swe-speedrun.jsonl --emit-test
```

3. **Fine-tune Qwen with LoRA**

```bash
python training/train_qwen_speedrun.py configs/speedrun.json data/processed/swe-speedrun.jsonl
```

4. **Evaluate checkpoint**

```bash
python training/evaluate_qwen_speedrun.py artifacts/checkpoints/qwen-speedrun data/processed/swe-speedrun.jsonl
```

5. **Run tests**

```bash
pytest tests/ -v
```

## Configuration

Edit `configs/speedrun.json` to customize training parameters. Key settings include:

- `dataset.source_dataset`: Dataset identifier (default: "SWE-bench/SWE-bench")
- `dataset.train_only`: Use all data for training without holdout (default: True)
- `dataset.holdout_fraction`: Fraction of data for test/holdout (default: None)
- `dataset.include_empty`: Include examples with empty responses (default: False)
- `dataset.min_patch_chars`: Minimum patch length filter (default: None)
- `dataset.max_examples`: Limit on examples (default: None, no limit)
- `training.eval_strategy`: When to evaluate ("no", "steps", or "epoch")
- `training.eval_steps`: Number of steps between evaluations
- `training.save_strategy`: When to save checkpoints

Training hyperparameters live in `swe_scaffold/config.py` and can be overridden through JSON passed to `train_qwen_speedrun.py`.

## Dataset Filtering and Auditing

The build script supports comprehensive filtering:

### CLI Flags

- `--train-only` / `--no-holdout`: Use all data for training (default behavior)
- `--holdout-fraction FLOAT`: Fraction to hold out for test split
- `--include-empty`: Include examples with empty responses
- `--min-patch-chars INT`: Filter examples with patches shorter than INT characters
- `--limit INT`: Maximum examples to process (<=0 means no limit)
- `--skipped-report PATH`: Write skipped examples to PATH
- `--force-skipped-report`: Write skipped report even if empty

### Response Fallback Order

When building prompt/response pairs, the system uses this fallback order:
1. `change_summary` (preferred)
2. `patch` (if no change_summary)
3. `test_patch` (if no patch)
4. Empty string (skipped by default unless `--include-empty`)

### Labeling Heuristic

Examples are labeled as success/failure using:
1. The `resolved` field if present (primary)
2. Presence of a `patch` field (fallback)

### Skipped Report Format

The skipped report is a JSONL file with entries like:

```json
{"instance_id": "django__django-12345", "reason": "empty_response"}
{"instance_id": "sympy__sympy-67890", "reason": "patch_too_short", "patch_length": "42"}
```

## License

MIT

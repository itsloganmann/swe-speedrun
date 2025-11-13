# SWE Speedrun Scaffold

This repository packages a reproducible workflow for exploring SWE-agent speedrun style datasets and fine-tuning [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) with LoRA adapters. It contains:

- A **read-only SWE-agent configuration** for guided repository walkthroughs
- Dataset utilities that curate prompt/response pairs from SWE Bench–style corpora
- Training scripts that apply PEFT/LoRA adapters to Qwen
- Evaluation helpers and lightweight CLI scripts for dataset QA

## Dataset Splits

This scaffold uses **dev** and **test** splits to align with SWE-bench Lite naming conventions:
- **dev**: Training split (default: 100% of data)
- **test**: Evaluation/holdout split (default: none, can be configured via `holdout_fraction`)

By default, no holdout is used and all data is assigned to the dev split. The holdout can be configured via the `holdout_fraction` parameter in configuration files.

**Response Filtering**: Examples lacking both a `change_summary` and `patch`/`test_patch` fields are automatically removed during dataset loading. The fallback chain tries `change_summary` → `patch` → `test_patch` to find a valid response.

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

```bash
# Default: no holdout, all data for training
python scripts/build_speedrun_dataset.py --dataset anchen-li/swe-bench-lite --limit 500 --output data/processed/swe-speedrun.jsonl

# With holdout for evaluation (not default)
python scripts/build_speedrun_dataset.py --dataset anchen-li/swe-bench-lite --limit 500 --output data/processed/swe-speedrun.jsonl --emit-test

# View dataset statistics
python scripts/summarize_dataset.py data/processed/swe-speedrun.jsonl
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

- `dataset.holdout_fraction`: Fraction of data to hold out for testing (default: None for no holdout)
- `dataset.dev_split`: Legacy parameter (deprecated, use `holdout_fraction` instead)
- `training.eval_strategy`: When to evaluate ("no", "steps", or "epoch")
- `training.eval_steps`: Number of steps between evaluations
- `training.save_strategy`: When to save checkpoints

Training hyperparameters live in `swe_scaffold/config.py` and can be overridden through JSON passed to `train_qwen_speedrun.py`.

## License

MIT

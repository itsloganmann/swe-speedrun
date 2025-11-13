# SWE Speedrun Scaffold

This repository packages a reproducible workflow for exploring SWE-agent speedrun style datasets and fine-tuning [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) with LoRA adapters. It contains:

- A **read-only SWE-agent configuration** for guided repository walkthroughs
- Dataset utilities that curate prompt/response pairs from SWE-bench corpora
- Training scripts that apply PEFT/LoRA adapters to Qwen
- Evaluation helpers and lightweight CLI scripts for dataset QA

## Dataset Splits

This scaffold uses **dev** and **test** splits to align with standard machine learning naming conventions:
- **dev**: Training split (default 90% of data)
- **test**: Evaluation/holdout split (default 10% of data)

The split ratio can be configured via the `dev_split` parameter in configuration files.

By default, the scaffold loads only the **train** split from the full SWE-bench dataset (https://huggingface.co/datasets/SWE-bench/SWE-bench/viewer/default/train) and creates an internal holdout split. This can be configured using the `train_only` and `holdout_fraction` parameters.

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
python scripts/build_speedrun_dataset.py --dataset SWE-bench/SWE-bench --train-only --limit 1000 --output data/processed/swe-speedrun.jsonl
python scripts/summarize_dataset.py data/processed/swe-speedrun.jsonl
```

The build script now loads only the **train** split from SWE-bench by default and creates an internal 90/10 split. Responses are derived from the `patch` field (code diff) when no `change_summary` is available. 

Additional flags:
- `--train-only` (default): Load only the train split from HF dataset
- `--no-train-only`: Disable train-only mode and load all available splits
- `--no-holdout`: Skip creating a test split (all data goes to dev)
- `--emit-test`: Write both dev and test splits to separate JSONL files

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

- `dataset.source_dataset`: Hugging Face dataset identifier (default: "SWE-bench/SWE-bench")
- `dataset.train_only`: Load only the train split from HF dataset (default: true)
- `dataset.holdout_fraction`: Fraction for test split when train_only is true (default: null, uses 1 - dev_split)
- `dataset.dev_split`: Fraction of data used for training (default: 0.9)
- `training.eval_strategy`: When to evaluate ("no", "steps", or "epoch")
- `training.eval_steps`: Number of steps between evaluations
- `training.save_strategy`: When to save checkpoints

Training hyperparameters live in `swe_scaffold/config.py` and can be overridden through JSON passed to `train_qwen_speedrun.py`.

### Response Field Selection

The dataset builder uses a fallback hierarchy for response content:
1. `change_summary` (if available)
2. `patch` (code diff)
3. `test_patch` (test changes)
4. Empty string (if none available)

### Label Heuristics

Success/failure labels are determined by:
- If `resolved` field exists: use its boolean value
- Otherwise: SUCCESS if non-empty `patch` exists, else FAILURE

## License

MIT

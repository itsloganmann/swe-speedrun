# SWE Speedrun Scaffold

This repository packages a reproducible workflow for exploring SWE-agent speedrun style datasets and fine-tuning [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) with LoRA adapters. It contains:

- A **read-only SWE-agent configuration** for guided repository walkthroughs
- Dataset utilities that curate prompt/response pairs from SWE-Bench–style corpora
- Training scripts that apply PEFT/LoRA adapters to Qwen
- Evaluation helpers and lightweight CLI scripts for dataset QA

## Dataset Splits

This scaffold uses **dev** and **test** splits to organize training and evaluation data:
- **dev**: Training split (default 90% of data when using holdout)
- **test**: Evaluation/holdout split (default 10% of data, or empty if `--no-holdout` is used)

The holdout fraction can be configured via the `holdout_fraction` parameter in configuration files or disabled entirely using the `--no-holdout` flag.

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

Build the full SWE-bench dataset (train split only, no holdout):

```bash
python scripts/build_speedrun_dataset.py --dataset SWE-bench/SWE-bench --train-only --no-holdout --output data/processed/swe-speedrun.jsonl
python scripts/summarize_dataset.py data/processed/swe-speedrun.jsonl
```

Or with a holdout split for evaluation:

```bash
python scripts/build_speedrun_dataset.py --dataset SWE-bench/SWE-bench --train-only --output data/processed/swe-speedrun.jsonl
python scripts/summarize_dataset.py data/processed/swe-speedrun.jsonl
```

To limit the number of examples (e.g., for testing):

```bash
python scripts/build_speedrun_dataset.py --dataset SWE-bench/SWE-bench --train-only --limit 500 --output data/processed/swe-speedrun.jsonl
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
- `dataset.train_only`: Whether to load only the train split (default: true)
- `dataset.holdout_fraction`: Fraction of data held out for test split (default: 0.1, set to null for no holdout)
- `dataset.max_examples`: Maximum examples to load (default: null for unlimited)
- `training.eval_strategy`: When to evaluate ("no", "steps", or "epoch")
- `training.eval_steps`: Number of steps between evaluations
- `training.save_strategy`: When to save checkpoints

Training hyperparameters live in `swe_scaffold/config.py` and can be overridden through JSON passed to `train_qwen_speedrun.py`.

### Dataset Loading Behavior

The dataset loader uses the following logic:

**Response Field Fallback Order:**
1. `change_summary` (if present and non-empty)
2. `patch` (if present and non-empty)
3. `test_patch` (if present and non-empty)
4. Empty string (fallback)

**Label Assignment:**
- If `resolved` field is present: use its value (SUCCESS if true, FAILURE if false)
- Otherwise: SUCCESS if `patch` field is non-empty, FAILURE if empty

### CLI Flags

`scripts/build_speedrun_dataset.py` supports:
- `--dataset`: Hugging Face dataset identifier (default: "SWE-bench/SWE-bench")
- `--train-only`: Load only the train split from the dataset
- `--no-holdout`: Do not create a test split; all data becomes dev
- `--limit N`: Cap number of examples; omit or use <=0 for unlimited
- `--output PATH`: Destination JSONL file path
- `--emit-test`: Also write a separate .test.jsonl file

## License

MIT

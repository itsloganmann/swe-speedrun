# SWE Speedrun Scaffold

This repository packages a reproducible workflow for exploring SWE-agent speedrun style datasets and fine-tuning [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) with LoRA adapters. It contains:

- A **read-only SWE-agent configuration** for guided repository walkthroughs
- Dataset utilities that curate prompt/response pairs from SWE Bench–style corpora
- Training scripts that apply PEFT/LoRA adapters to Qwen
- Evaluation helpers and lightweight CLI scripts for dataset QA

## Dataset Splits

This scaffold uses **dev** and **test** splits to align with SWE-bench Lite naming conventions:
- **dev**: Training split (default 90% of data)
- **test**: Evaluation/holdout split (default 10% of data)

The split ratio can be configured via the `dev_split` parameter in configuration files.

## Dataset Filtering and Auditing

The dataset builder supports optional filtering and auditing features:

### Response Fallback
The dataset builder uses a fallback strategy when selecting responses from source data:
1. **change_summary** (preferred)
2. **patch** (fallback if change_summary is empty)
3. **test_patch** (fallback if both change_summary and patch are empty)

### Minimum Patch Length Filter
You can filter out examples where the selected response comes from `patch` or `test_patch` fields and is too short:
- Configure via `min_patch_chars` in `DatasetConfig`
- Use CLI flag `--min-patch-chars` to set threshold (e.g., `--min-patch-chars 200`)
- Only applies to patch-based responses, not `change_summary`
- Useful for filtering trivial diffs

### Empty Response Handling
By default, examples with empty responses (after fallback) are filtered out:
- Use `include_empty=True` in config or `--include-empty` CLI flag to retain them
- Empty examples are labeled as FAILURE when included
- Default behavior filters empty responses for cleaner training data

### Skipped Examples Report
Track which examples were filtered and why:
- Automatically generated at `data/processed/swe-speedrun.skipped.jsonl` (or custom path via `--skipped-report`)
- Only created if examples were actually skipped (unless `--force-skipped-report` is used)
- Each line contains:
  - `instance_id`: identifier for the example
  - `reason`: either `empty_response` or `short_patch`
  - `source_fields`: which fields were available (has_change_summary, has_patch, has_test_patch)
  - `patch_len`: character count of the selected response

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
# Basic usage
python scripts/build_speedrun_dataset.py --dataset anchen-li/swe-bench-lite --limit 500 --output data/processed/swe-speedrun.jsonl

# With patch length filter (exclude patches shorter than 200 characters)
python scripts/build_speedrun_dataset.py --dataset anchen-li/swe-bench-lite --limit 500 --min-patch-chars 200 --output data/processed/swe-speedrun.jsonl

# Include empty responses (retain examples with no response content)
python scripts/build_speedrun_dataset.py --dataset anchen-li/swe-bench-lite --limit 500 --include-empty --output data/processed/swe-speedrun.jsonl

# Generate test split and skipped report
python scripts/build_speedrun_dataset.py --dataset anchen-li/swe-bench-lite --limit 500 --emit-test --skipped-report data/processed/skipped.jsonl --output data/processed/swe-speedrun.jsonl

# Summarize the generated dataset
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

### Dataset Configuration
- `dataset.dev_split`: Fraction of data used for training (default: 0.9)
- `dataset.min_patch_chars`: Minimum character count for patch-based responses (default: None)
- `dataset.include_empty`: Include examples with empty responses (default: False)
- `dataset.max_examples`: Maximum number of examples to process (default: 500)

### Training Configuration
- `training.eval_strategy`: When to evaluate ("no", "steps", or "epoch")
- `training.eval_steps`: Number of steps between evaluations
- `training.save_strategy`: When to save checkpoints

Training hyperparameters live in `swe_scaffold/config.py` and can be overridden through JSON passed to `train_qwen_speedrun.py`.

### CLI Flags for build_speedrun_dataset.py

- `--dataset`: HuggingFace dataset identifier (default: SWE-bench/SWE-bench)
- `--limit`: Maximum number of dev examples (default: 500)
- `--min-patch-chars`: Minimum character length for patch responses (filters short patches)
- `--include-empty`: Include examples with empty responses (labeled as FAILURE)
- `--emit-test`: Also write test split to separate file
- `--skipped-report`: Path for skipped examples report (default: data/processed/swe-speedrun.skipped.jsonl)
- `--force-skipped-report`: Create skipped report even when no examples skipped
- `--output`: Destination path for dev split JSONL (default: data/processed/swe-speedrun.jsonl)

## License

MIT

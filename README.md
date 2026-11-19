# SWE Speedrun Scaffold

This repository packages a reproducible workflow for exploring SWE-agent speedrun style datasets and fine-tuning [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) with LoRA adapters. It contains:

- A **read-only SWE-agent configuration** for guided repository walkthroughs
- Dataset utilities that curate prompt/response pairs from SWE-bench corpora
- Training scripts that apply PEFT/LoRA adapters to Qwen
- Evaluation helpers and lightweight CLI scripts for dataset QA

## Dataset Splits

This scaffold uses **dev** for training examples and **test** for an optional internal holdout:
- **dev**: All retained training examples by default
- **test**: Empty by default; populated with `holdout_fraction` or `--holdout-fraction`

The default configuration requests `train` from [SWE-bench/SWE-bench](https://huggingface.co/datasets/SWE-bench/SWE-bench), with no example cap and no holdout. It never folds the official test split into training. Set a positive limit for a smaller run, or a holdout fraction between zero and one for evaluation.

`None` and zero disable the internal holdout. Out-of-range or nonfinite holdout fractions emit a warning and use no holdout.

The default source is pinned to [revision `ac1995e`](https://huggingface.co/datasets/SWE-bench/SWE-bench/tree/ac1995e464b07cbd69702e8490a07f6d557b69b5), which contains 19,008 training rows. Upstream `main` removed `train` in August 2026. The pin keeps the same official corpus and its original split; it does not substitute evaluation rows. Override it with `--revision` or `dataset.source_revision` when selecting another verified revision. Custom datasets use their own default revision unless one is specified.

Legacy JSON configurations with `dev_split` still map that training fraction to an internal holdout. Legacy `split` and converted `dev_split` inputs emit a `FutureWarning` explaining the no-holdout default. An explicit `holdout_fraction` takes precedence. `train_only` controls which upstream split is loaded; it does not disable an explicit holdout. In legacy `train_only=false` mode, an existing dev/test pair is preserved unless a split override is supplied.

## Dataset Filtering and Auditing

The dataset builder supports optional filtering and auditing features:

### Response Fallback
The dataset builder uses a fallback strategy when selecting responses from source data:
1. **change_summary** (preferred)
2. **patch** (fallback if change_summary is empty)
3. **test_patch** (fallback if both change_summary and patch are empty)

Whitespace-only fields count as empty during selection. Stored responses retain their original text, including patch line endings.

### Minimum Patch Length Filter
You can filter out examples where the selected response comes from `patch` or `test_patch` fields and is too short:
- Configure via `min_patch_chars` in `DatasetConfig`
- Use CLI flag `--min-patch-chars` to set threshold (e.g., `--min-patch-chars 200`)
- Only applies to patch-based responses, not `change_summary`
- Useful for filtering trivial diffs

### Empty Response Handling
By default, examples with empty responses (after fallback) are filtered out:
- Use `include_empty=True` in config or `--include-empty` CLI flag to retain them
- Included empty examples follow the existing resolved-field and patch-presence label rules
- Default behavior filters empty responses for cleaner training data

### Skipped Examples Report
Track which examples were filtered and why:
- Automatically generated beside the output as `<output-stem>.skipped.jsonl` (or custom path via `--skipped-report`)
- Only created if examples were actually skipped (unless `--force-skipped-report` is used)
- An existing report is cleared on a zero-skip rerun so it describes the current output.
- Each line contains:
  - `instance_id`: identifier for the example
  - `reason`: either `empty_response` or `short_patch`
  - `source_fields`: which fields were available (has_change_summary, has_patch, has_test_patch)
  - `patch_len`: stripped character count used by the length filter

The skipped-report path must differ from the dataset output and any emitted test output.

The loader returns a `DatasetSplit`-compatible `ProjectionResult` with `dev`, `test`, `skipped`, `total_kept`, and `total_skipped`. The `.split` alias supports existing callers. Disk caches support empty dev and test splits.

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
pip install -e .
```

2. **Prepare dataset**

```bash
# Full training dataset, no cap or holdout
python scripts/build_speedrun_dataset.py --dataset SWE-bench/SWE-bench --output data/processed/swe-speedrun.jsonl

# Read dataset settings from JSON or YAML; explicit flags override them
python scripts/build_speedrun_dataset.py --config configs/speedrun.json

# With patch length filter (exclude patches shorter than 200 characters)
python scripts/build_speedrun_dataset.py --dataset SWE-bench/SWE-bench --limit 500 --min-patch-chars 200 --output data/processed/swe-speedrun.jsonl

# Include empty responses (retain examples with no response content)
python scripts/build_speedrun_dataset.py --dataset SWE-bench/SWE-bench --limit 500 --include-empty --output data/processed/swe-speedrun.jsonl

# Generate an internal test split and skipped report
python scripts/build_speedrun_dataset.py --holdout-fraction 0.1 --emit-test --skipped-report data/processed/skipped.jsonl

# Summarize the generated dataset
python scripts/summarize_dataset.py data/processed/swe-speedrun.jsonl
```

Responses come from `change_summary`, then `patch`, then `test_patch`. Empty responses are excluded by default. Omitted, zero, and negative limits keep all examples. With a positive limit, dev is capped at that limit and a nonempty test split is capped at `max(limit // 10, 1)`.

Additional flags:
- `--train-only` (default): Load only the train split from HF dataset
- `--no-train-only`: Load the upstream split dictionary; preserve an existing dev/test pair without split overrides, otherwise require a train or dev source
- `--no-holdout`: Explicitly disable a configured holdout
- `--holdout-fraction`: Create an internal holdout, such as `0.1` for 10%
- `--emit-test`: Write a separate test JSONL file, empty when there is no holdout

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
- `dataset.source_revision`: Optional HF revision; null uses the pinned revision above for the default source and the upstream default for custom sources
- `dataset.train_only`: Load only the train split from HF dataset (default: true)
- `dataset.holdout_fraction`: Internal holdout fraction (default: null, no holdout)

### Dataset Configuration
- `dataset.dev_split`: Legacy training-fraction input, mapped to `1 - dev_split` holdout when no new setting is present
- `dataset.min_patch_chars`: Minimum character count for patch-based responses (default: None)
- `dataset.include_empty`: Include examples with empty responses (default: False)
- `dataset.max_examples`: Dev example cap; null or nonpositive means unlimited (default: null)
- `dataset.local_cache`: Default JSONL output path
- `dataset.skipped_report_path`: Optional audit path; null puts the report beside the output

The build CLI accepts JSON or YAML objects with `dataset` and/or `training` sections. `configs/scaffold_readonly.yaml` is a separate SWE-agent session configuration, not a dataset build configuration.

### Training Configuration
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

### CLI Flags for build_speedrun_dataset.py

- `--dataset`: HuggingFace dataset identifier (default: SWE-bench/SWE-bench)
- `--revision`: Override the source revision; changing `--dataset` also clears a different source's configured revision unless `--revision` is supplied
- `--config`: Dataset/training JSON or YAML file (default: built-in configuration)
- `--limit`: Maximum dev examples; omit or use a nonpositive value for no cap by default
- `--min-patch-chars`: Minimum character length for patch responses (filters short patches)
- `--include-empty` / `--no-include-empty`: Override whether empty responses are retained
- `--emit-test`: Also write test split to separate file
- `--skipped-report`: Audit path (default: `<output-stem>.skipped.jsonl` beside the output)
- `--force-skipped-report`: Create skipped report even when no examples skipped
- `--output`: Destination path for dev split JSONL (default: data/processed/swe-speedrun.jsonl)

## License

MIT

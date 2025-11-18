#!/usr/bin/env python3
"""CLI utility to assemble the SWE speedrun dataset."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from swe_scaffold.config import SpeedrunConfig
from swe_scaffold.dataset import load_conversation_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the SWE speedrun dataset cache.")
    parser.add_argument("--config", type=Path, default=Path("configs/scaffold_readonly.yaml"), help="Path to config file (YAML or JSON)")
    parser.add_argument("--output", type=Path, default=Path("data/processed/swe-speedrun.jsonl"), help="Destination JSONL path (dev split)")
    parser.add_argument("--dataset", type=str, default="SWE-bench/SWE-bench", help="Hugging Face dataset identifier")
    parser.add_argument("--limit", type=int, default=500, help="Optional limit on number of dev examples")
    parser.add_argument("--train-only", action="store_true", default=True, help="Load only the train split from HF dataset (default: True)")
    parser.add_argument("--no-train-only", dest="train_only", action="store_false", help="Disable train-only mode (load all splits)")
    parser.add_argument("--no-holdout", action="store_true", help="Skip creating a test split (all data goes to dev)")
    parser.add_argument("--emit-test", action="store_true", help="Also write a .test.jsonl next to output")
    return parser.parse_args()


def _dump_jsonl(ds, path: Path) -> int:
    """Write a dataset to JSONL with ensure_ascii to escape all special characters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for prompt, response, label in zip(ds["prompt"], ds["response"], ds["label"]):
            record = {
                "prompt": str(prompt),
                "response": str(response),
                "label": label
            }
            
            # Use ensure_ascii=True to escape all problematic characters
            try:
                line = json.dumps(record, ensure_ascii=True)
                handle.write(line + "\n")
                handle.flush()  # Force immediate write to prevent buffer issues
                count += 1
            except (TypeError, ValueError, UnicodeEncodeError) as e:
                print(f"Warning: Failed to serialize record {count + 1}: {e}")
                continue
    
    return count


def main() -> None:
    # Suppress TensorFlow warnings if TF is installed
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    
    args = parse_args()
    # Loaded but not used directly here; kept for parity with scaffold flow
    _ = SpeedrunConfig()

    # Determine holdout_fraction based on flags
    holdout_fraction = 0.0 if args.no_holdout else None

    split = load_conversation_dataset(
        args.dataset,
        limit=args.limit,
        train_only=args.train_only,
        holdout_fraction=holdout_fraction,
    )

    # Write dev split to the requested output
    n_dev = _dump_jsonl(split.dev, args.output)
    print(f"Wrote {n_dev} dev examples to {args.output}")

    if args.emit_test:
        test_path = args.output.with_name(args.output.stem + ".test.jsonl")
        n_test = _dump_jsonl(split.test, test_path)
        print(f"Wrote {n_test} test examples to {test_path}")


if __name__ == "__main__":
    main()

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
    parser.add_argument("--limit", type=int, default=None, help="Cap number of training examples; omit or <=0 for all")
    parser.add_argument("--train-only", action="store_true", help="Load only the train split from the dataset")
    parser.add_argument("--no-holdout", action="store_true", help="Do not create a holdout/test split (all data becomes dev)")
    parser.add_argument("--emit-test", action="store_true", help="Also write a .test.jsonl next to output")
    return parser.parse_args()


def _dump_jsonl(ds, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for prompt, response, label in zip(ds["prompt"], ds["response"], ds["label"]):
            handle.write(json.dumps({"prompt": prompt, "response": response, "label": label}, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    # Suppress TensorFlow warnings if TF is installed
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    
    args = parse_args()
    # Loaded but not used directly here; kept for parity with scaffold flow
    _ = SpeedrunConfig()

    # Determine holdout_fraction based on flags
    holdout_fraction = None if args.no_holdout else 0.1
    
    # Load dataset with new parameters
    split = load_conversation_dataset(
        args.dataset,
        train_only=args.train_only,
        holdout_fraction=holdout_fraction,
        limit=args.limit if args.limit and args.limit > 0 else None,
    )
    
    # Log total examples loaded
    print(f"Loaded {len(split.dev) + len(split.test)} total examples (dev={len(split.dev)}, test={len(split.test)})")

    # Write dev split to the requested output
    n_dev = _dump_jsonl(split.dev, args.output)
    print(f"Wrote {n_dev} dev examples to {args.output}")

    if args.emit_test:
        test_path = args.output.with_name(args.output.stem + ".test.jsonl")
        n_test = _dump_jsonl(split.test, test_path)
        print(f"Wrote {n_test} test examples to {test_path}")


if __name__ == "__main__":
    main()

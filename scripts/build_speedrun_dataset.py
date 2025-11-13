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
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of examples (<=0 means no cap)")
    parser.add_argument("--emit-test", action="store_true", help="Also write a .test.jsonl next to output")
    parser.add_argument("--train-only", action="store_true", default=True, help="No holdout/test split (default: True)")
    parser.add_argument("--no-holdout", action="store_true", help="Alias for --train-only")
    parser.add_argument("--include-empty", action="store_true", help="Include examples with empty responses")
    parser.add_argument("--min-patch-chars", type=int, default=None, help="Minimum patch length in characters")
    parser.add_argument("--skipped-report", type=Path, default=None, help="Path to write skipped examples report")
    parser.add_argument("--force-skipped-report", action="store_true", help="Write skipped report even if no examples were skipped")
    parser.add_argument("--holdout-fraction", type=float, default=None, help="Fraction of data to hold out for test (overrides train-only)")
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

    # Resolve train_only flag
    train_only = args.train_only or args.no_holdout
    if args.holdout_fraction is not None:
        train_only = False  # Explicit holdout overrides train-only
    
    # Resolve limit (<=0 means no cap)
    limit = args.limit if args.limit > 0 else None

    result = load_conversation_dataset(
        args.dataset,
        limit=limit,
        train_only=train_only,
        holdout_fraction=args.holdout_fraction,
        include_empty=args.include_empty,
        min_patch_chars=args.min_patch_chars,
    )

    # Write dev split to the requested output
    n_dev = _dump_jsonl(result.split.dev, args.output)
    print(f"Wrote {n_dev} dev examples to {args.output}")

    if args.emit_test:
        test_path = args.output.with_name(args.output.stem + ".test.jsonl")
        n_test = _dump_jsonl(result.split.test, test_path)
        print(f"Wrote {n_test} test examples to {test_path}")
    
    # Summary logging
    print(f"\n=== Summary ===")
    print(f"Total examples kept: {result.total_kept}")
    print(f"Total examples skipped: {result.total_skipped}")
    
    # Write skipped report if needed
    if result.skipped and (args.skipped_report or args.force_skipped_report):
        report_path = args.skipped_report or args.output.with_name(args.output.stem + ".skipped.jsonl")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            for record in result.skipped:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Wrote skipped report to {report_path}")
    elif args.force_skipped_report and not result.skipped:
        # Force write empty report
        report_path = args.skipped_report or args.output.with_name(args.output.stem + ".skipped.jsonl")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("")
        print(f"Wrote empty skipped report to {report_path}")


if __name__ == "__main__":
    main()

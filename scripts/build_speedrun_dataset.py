#!/usr/bin/env python3
"""CLI utility to assemble the SWE speedrun dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml
from datasets import Dataset

from swe_scaffold.config import SpeedrunConfig
from swe_scaffold.dataset import load_conversation_dataset


def _load_config(path: Path | None) -> SpeedrunConfig:
    if path is None:
        return SpeedrunConfig()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    elif path.suffix.lower() in (".yaml", ".yml"):
        payload = yaml.safe_load(text)
    else:
        raise ValueError("--config must be a JSON or YAML file")
    if not isinstance(payload, dict) or not payload.keys() <= {"dataset", "training"}:
        raise ValueError(
            "Expected a configuration object with dataset/training sections"
        )
    return SpeedrunConfig.from_dict(payload)


def _apply_config(
    args: argparse.Namespace, config: SpeedrunConfig
) -> argparse.Namespace:
    dataset = config.dataset
    output = args.output if args.output is not None else dataset.local_cache
    report = (
        args.skipped_report
        if args.skipped_report is not None
        else dataset.skipped_report_path
    )
    return argparse.Namespace(
        **{
            **vars(args),
            "dataset": (
                args.dataset if args.dataset is not None else dataset.source_dataset
            ),
            "output": Path(output),
            "limit": args.limit if args.limit is not None else dataset.max_examples,
            "train_only": (
                args.train_only if args.train_only is not None else dataset.train_only
            ),
            "holdout_fraction": (
                0.0
                if args.no_holdout
                else (
                    args.holdout_fraction
                    if args.holdout_fraction is not None
                    else dataset.holdout_fraction
                )
            ),
            "min_patch_chars": (
                args.min_patch_chars
                if args.min_patch_chars is not None
                else dataset.min_patch_chars
            ),
            "include_empty": (
                args.include_empty
                if args.include_empty is not None
                else dataset.include_empty
            ),
            "skipped_report": (
                Path(report)
                if report is not None
                else Path(output).with_name(Path(output).stem + ".skipped.jsonl")
            ),
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the SWE speedrun dataset cache."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional dataset/training configuration (JSON or YAML)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination JSONL path (defaults to configured local_cache)",
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="Hugging Face dataset identifier"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap dev examples; omit or use <=0 for all by default",
    )
    parser.add_argument(
        "--train-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Load only the train split from HF (default: true)",
    )
    holdout = parser.add_mutually_exclusive_group()
    holdout.add_argument(
        "--no-holdout", action="store_true", help="Skip creating a test split"
    )
    holdout.add_argument(
        "--holdout-fraction",
        type=float,
        default=None,
        help="Create an explicit internal holdout",
    )
    parser.add_argument(
        "--emit-test",
        action="store_true",
        help="Also write a .test.jsonl next to output",
    )
    parser.add_argument(
        "--min-patch-chars",
        type=int,
        default=None,
        help="Minimum stripped length for patch-based responses",
    )
    parser.add_argument(
        "--include-empty",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Retain examples with empty responses",
    )
    parser.add_argument(
        "--skipped-report",
        type=Path,
        default=None,
        help="Audit path (default: next to dataset output)",
    )
    parser.add_argument(
        "--force-skipped-report",
        action="store_true",
        help="Create an audit even when nothing was skipped",
    )
    args = parser.parse_args()
    try:
        args = _apply_config(args, _load_config(args.config))
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    outputs = (args.output,)
    if args.emit_test:
        outputs += (args.output.with_name(args.output.stem + ".test.jsonl"),)
    for output in outputs:
        same_file = (
            args.skipped_report.exists()
            and output.exists()
            and args.skipped_report.samefile(output)
        )
        if args.skipped_report.resolve() == output.resolve() or same_file:
            parser.error("--skipped-report must differ from dataset output paths")
    return args


def _dump_jsonl(ds: Dataset, path: Path) -> int:
    """Write a dataset to JSONL with ensure_ascii to escape all special characters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for prompt, response, label in zip(ds["prompt"], ds["response"], ds["label"]):
            record = {"prompt": str(prompt), "response": str(response), "label": label}

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
    result = load_conversation_dataset(
        args.dataset,
        limit=args.limit,
        train_only=args.train_only,
        holdout_fraction=args.holdout_fraction,
        min_patch_chars=args.min_patch_chars,
        include_empty=args.include_empty,
    )

    # Write dev split to the requested output
    n_dev = _dump_jsonl(result.split.dev, args.output)

    # Count skipped examples by reason
    empty_count = sum(1 for s in result.skipped if s["reason"] == "empty_response")
    short_count = sum(1 for s in result.skipped if s["reason"] == "short_patch")
    total_skipped = len(result.skipped)

    print(
        f"Loaded {result.total_kept} examples (dev={n_dev}, test={len(result.test)}, skipped {total_skipped}: empty={empty_count}, short={short_count})"
    )

    if args.emit_test:
        test_path = args.output.with_name(args.output.stem + ".test.jsonl")
        n_test = _dump_jsonl(result.split.test, test_path)
        print(f"Wrote {n_test} test examples to {test_path}")

    # Write skipped report if there are skipped examples or if forced
    if result.skipped or args.force_skipped_report:
        args.skipped_report.parent.mkdir(parents=True, exist_ok=True)
        with args.skipped_report.open("w", encoding="utf-8") as handle:
            for skip_record in result.skipped:
                handle.write(json.dumps(skip_record, ensure_ascii=False) + "\n")
        if result.skipped:
            print(f"Wrote skipped examples report to {args.skipped_report}")
    elif args.skipped_report.exists():
        args.skipped_report.write_text("", encoding="utf-8")


if __name__ == "__main__":
    main()

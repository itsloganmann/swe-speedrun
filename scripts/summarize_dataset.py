#!/usr/bin/env python3
"""Summarise dataset statistics for sanity checks."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise a SWE speedrun dataset JSONL file.")
    parser.add_argument("dataset", type=Path, help="Path to the JSONL dataset file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.dataset.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    label_counts = Counter(record.get("label", "unknown") for record in records)
    avg_prompt_length = sum(len(record.get("prompt", "")) for record in records) / max(len(records), 1)
    avg_response_length = sum(len(record.get("response", "")) for record in records) / max(len(records), 1)

    print(f"Examples: {len(records)}")
    print("Label distribution:")
    for label, count in label_counts.items():
        print(f"  - {label}: {count}")
    print(f"Average prompt characters: {avg_prompt_length:.1f}")
    print(f"Average response characters: {avg_response_length:.1f}")


if __name__ == "__main__":
    main()

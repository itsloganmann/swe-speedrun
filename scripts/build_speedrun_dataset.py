#!/usr/bin/env python3
"""CLI utility to assemble the SWE speedrun dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from swe_scaffold.config import SpeedrunConfig
from swe_scaffold.dataset import SpeedrunDatasetBuilder, load_conversation_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the SWE speedrun dataset cache.")
    parser.add_argument("--config", type=Path, default=Path("configs/scaffold_readonly.yaml"), help="Path to config file (YAML or JSON)")
    parser.add_argument("--output", type=Path, default=Path("data/processed/swe-speedrun.jsonl"), help="Destination JSONL path")
    parser.add_argument("--dataset", type=str, default="anchen-li/swe-bench-lite", help="Hugging Face dataset identifier")
    parser.add_argument("--limit", type=int, default=500, help="Optional limit on number of examples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SpeedrunConfig()
    split = load_conversation_dataset(args.dataset, limit=args.limit)

    records = []
    for prompt, response, label in zip(split.train["prompt"], split.train["response"], split.train["label"]):
        records.append({"prompt": prompt, "response": response, "label": label})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} training examples to {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate a fine-tuned Qwen checkpoint on held-out data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset

from swe_scaffold.evaluation import evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Qwen checkpoint on validation data.")
    parser.add_argument("model", type=Path, help="Path to the trained model directory")
    parser.add_argument("dataset", type=Path, help="Path to the JSONL validation dataset")
    return parser.parse_args()


def load_dataset(path: Path) -> Dataset:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    texts = [record["prompt"] + "\n\n" + record["response"] for record in records]
    labels = [record.get("label", "unknown") for record in records]
    return Dataset.from_dict({"text": texts, "label": labels})


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset)
    perplexity, label_counts = evaluate_model(str(args.model), dataset)
    print(f"Perplexity: {perplexity:.3f}")
    print("Label distribution:")
    for label, count in label_counts.items():
        print(f"  - {label}: {count}")


if __name__ == "__main__":
    main()

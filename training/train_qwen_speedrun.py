#!/usr/bin/env python3
"""Fine-tune Qwen on the SWE speedrun dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict

from swe_scaffold.config import SpeedrunConfig
from swe_scaffold.dataset import SpeedrunDatasetBuilder
from swe_scaffold.training import train_lora_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen with LoRA on the SWE speedrun dataset.")
    parser.add_argument("config", type=Path, help="Path to a JSON configuration file")
    parser.add_argument("dataset", type=Path, help="Path to the processed JSONL dataset")
    return parser.parse_args()


def load_split(path: Path) -> DatasetDict:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    dataset = Dataset.from_dict(records)
    split_point = int(len(dataset) * 0.9)
    return DatasetDict(
        {
            "train": dataset.select(range(split_point)),
            "validation": dataset.select(range(split_point, len(dataset))),
        }
    )


def main() -> None:
    args = parse_args()
    config = SpeedrunConfig.from_dict(json.loads(args.config.read_text()))
    builder = SpeedrunDatasetBuilder(config.dataset.local_cache)

    dataset_split = builder.from_jsonl(args.dataset)
    trainer, tokenizer = train_lora_model(dataset_split, config.training)
    print("Training complete. Model saved to:", config.training.output_dir)


if __name__ == "__main__":
    main()

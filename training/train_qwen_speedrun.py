#!/usr/bin/env python3
"""Fine-tune Qwen on the SWE speedrun dataset."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from swe_scaffold.config import SpeedrunConfig
from swe_scaffold.dataset import SpeedrunDatasetBuilder
from swe_scaffold.training import train_lora_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen with LoRA on the SWE speedrun dataset.")
    parser.add_argument("config", type=Path, help="Path to a JSON configuration file")
    parser.add_argument("dataset", type=Path, help="Path to the processed JSONL dataset")
    return parser.parse_args()


def main() -> None:
    # Suppress TensorFlow warnings if TF is installed
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    
    args = parse_args()
    config = SpeedrunConfig.from_dict(json.loads(args.config.read_text()))
    builder = SpeedrunDatasetBuilder(config.dataset.local_cache)

    dataset_split = builder.from_jsonl(args.dataset)
    trainer, tokenizer = train_lora_model(dataset_split, config.training)
    print("Training complete. Model saved to:", config.training.output_dir)


if __name__ == "__main__":
    main()

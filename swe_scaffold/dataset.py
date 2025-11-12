"""Dataset helpers for SWE speedrun fine-tuning."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from datasets import Dataset, DatasetDict, load_dataset

from .labels import SpeedrunLabel, batch_label_conversations


@dataclass(slots=True)
class DatasetSplit:
    dev: Dataset
    test: Dataset


class SpeedrunDatasetBuilder:
    """Utility for constructing instruction-tuning corpora from SWE-agent transcripts."""

    def __init__(self, cache_path: Path, seed: int = 42) -> None:
        self.cache_path = cache_path
        self.seed = seed

    def from_jsonl(self, path: Path) -> DatasetSplit:
        """Load a pre-tokenised JSONL file into dev/test splits."""
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        labels = batch_label_conversations(records)
        prompts = [record.get("prompt", "") for record in records]
        responses = [record.get("response", "") for record in records]

        dataset = Dataset.from_dict(
            {
                "prompt": prompts,
                "response": responses,
                "label": [label.value for label in labels],
            }
        )
        dataset = dataset.shuffle(seed=self.seed)
        split_idx = int(len(dataset) * 0.9)
        return DatasetSplit(
            dev=dataset.select(range(split_idx)),
            test=dataset.select(range(split_idx, len(dataset))),
        )

    def hydrate_cache(self, split: DatasetSplit) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = DatasetDict({"dev": split.dev, "test": split.test})
        payload.save_to_disk(str(self.cache_path))

    def load_or_build(self, builder: Callable[[], DatasetSplit]) -> DatasetSplit:
        if self.cache_path.exists():
            dataset_dict = DatasetDict.load_from_disk(str(self.cache_path))
            # enforce presence of dev/test
            return DatasetSplit(dev=dataset_dict["dev"], test=dataset_dict["test"])
        split = builder()
        self.hydrate_cache(split)
        return split


def load_conversation_dataset(dataset_name: str, split: float = 0.9, limit: Optional[int] = None) -> DatasetSplit:
    """Load SWE-bench_Lite and project into prompt/response pairs using dev/test splits."""

    dataset_any = load_dataset(dataset_name)
    # Resolve to dev/test sources
    if isinstance(dataset_any, DatasetDict):
        if "dev" in dataset_any and "test" in dataset_any:
            dev_source = dataset_any["dev"]
            test_source = dataset_any["test"]
        else:
            # Fallback: derive dev/test from a single split
            base_key = "train" if "train" in dataset_any else sorted(dataset_any.keys())[0]
            derived = dataset_any[base_key].train_test_split(test_size=1 - split, seed=42)
            dev_source, test_source = derived["train"], derived["test"]
    elif isinstance(dataset_any, Dataset):
        derived = dataset_any.train_test_split(test_size=1 - split, seed=42)
        dev_source, test_source = derived["train"], derived["test"]
    else:
        raise TypeError("Only map-style datasets are supported for speedrun training")

    # Optional limits
    if limit is not None:
        dev_source = dev_source.select(range(min(limit, len(dev_source))))
        test_cap = max(limit // 10, 1)
        test_source = test_source.select(range(min(test_cap, len(test_source))))

    # Project fields to prompt/response/label
    def _project(src: Dataset) -> Dataset:
        prompts: List[str] = []
        responses: List[str] = []
        labels: List[SpeedrunLabel] = []
        for row in src:
            row_dict = dict(row)
            prompt = row_dict.get("problem_statement", "") or ""
            response = row_dict.get("change_summary", "") or ""
            resolved = row_dict.get("resolved", False)
            label = SpeedrunLabel.SUCCESS if resolved else SpeedrunLabel.FAILURE
            prompts.append(prompt)
            responses.append(response)
            labels.append(label)
        return Dataset.from_dict(
            {
                "prompt": prompts,
                "response": responses,
                "label": [label.value for label in labels],
            }
        )

    dev_dataset = _project(dev_source)
    test_dataset = _project(test_source)

    return DatasetSplit(dev=dev_dataset, test=test_dataset)


__all__ = [
    "DatasetSplit",
    "SpeedrunDatasetBuilder",
    "load_conversation_dataset",
]

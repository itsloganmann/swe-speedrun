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
    train: Dataset
    validation: Dataset


class SpeedrunDatasetBuilder:
    """Utility for constructing instruction-tuning corpora from SWE-agent transcripts."""

    def __init__(self, cache_path: Path, seed: int = 42) -> None:
        self.cache_path = cache_path
        self.seed = seed

    def from_jsonl(self, path: Path) -> DatasetSplit:
        """Load a pre-tokenised JSONL file into train/validation splits."""

        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        labels = batch_label_conversations(records)
        prompts = [record["prompt"] for record in records]
        responses = [record["response"] for record in records]

        dataset = Dataset.from_dict(
            {
                "prompt": prompts,
                "response": responses,
                "label": [label.value for label in labels],
            }
        )
        dataset = dataset.shuffle(seed=self.seed)
        split_idx = int(len(dataset) * 0.9)
        return DatasetSplit(train=dataset.select(range(split_idx)), validation=dataset.select(range(split_idx, len(dataset))))

    def hydrate_cache(self, split: DatasetSplit) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = DatasetDict({"train": split.train, "validation": split.validation})
        payload.save_to_disk(str(self.cache_path))

    def load_or_build(self, builder: Callable[[], DatasetSplit]) -> DatasetSplit:
        if self.cache_path.exists():
            dataset_dict = DatasetDict.load_from_disk(str(self.cache_path))
            return DatasetSplit(train=dataset_dict["train"], validation=dataset_dict["validation"])
        split = builder()
        self.hydrate_cache(split)
        return split


def load_conversation_dataset(dataset_name: str, split: float = 0.9, limit: Optional[int] = None) -> DatasetSplit:
    """Load SWE-Bench-lite (or compatible) dataset and project into prompt/response pairs."""

    dataset_any = load_dataset(dataset_name)
    if isinstance(dataset_any, DatasetDict):
        dataset_dict = dataset_any
    elif isinstance(dataset_any, Dataset):
        dataset_dict = DatasetDict({"train": dataset_any})
    else:
        raise TypeError("Only map-style datasets are supported for speedrun training")

    train_dataset: Dataset = dataset_dict["train"]
    train_split = train_dataset.train_test_split(test_size=1 - split, seed=42)
    if limit is not None:
        train_split["train"] = train_split["train"].select(range(min(limit, len(train_split["train"]))))
        train_split["test"] = train_split["test"].select(range(min(max(limit // 10, 1), len(train_split["test"]))))

    train_records: List[str] = []
    train_responses: List[str] = []
    val_records: List[str] = []
    val_responses: List[str] = []
    train_labels: List[SpeedrunLabel] = []
    val_labels: List[SpeedrunLabel] = []

    def _project(batch: Dataset, records: List[str], responses: List[str], labels: List[SpeedrunLabel]) -> None:
        for row in batch:
            row_dict = dict(row)
            prompt = row_dict.get("problem_statement", "") or ""
            response = row_dict.get("change_summary", "") or ""
            label = SpeedrunLabel.SUCCESS if row_dict.get("resolved", True) else SpeedrunLabel.FAILURE
            records.append(prompt)
            responses.append(response)
            labels.append(label)

    _project(train_split["train"], train_records, train_responses, train_labels)
    _project(train_split["test"], val_records, val_responses, val_labels)

    train_dataset = Dataset.from_dict(
        {
            "prompt": train_records,
            "response": train_responses,
            "label": [label.value for label in train_labels],
        }
    )
    val_dataset = Dataset.from_dict(
        {
            "prompt": val_records,
            "response": val_responses,
            "label": [label.value for label in val_labels],
        }
    )

    return DatasetSplit(train=train_dataset, validation=val_dataset)


__all__ = [
    "DatasetSplit",
    "SpeedrunDatasetBuilder",
    "load_conversation_dataset",
]

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
        """Load a pre tokenised JSONL file into dev and test splits.

        If `path` is a directory created by `DatasetDict.save_to_disk`, it will
        be loaded via `DatasetDict.load_from_disk` instead of JSONL parsing.
        """
        # If the path is a directory, treat it as a saved DatasetDict
        if path.is_dir():
            dataset_dict = DatasetDict.load_from_disk(str(path))
            if "dev" in dataset_dict and "test" in dataset_dict:
                return DatasetSplit(dev=dataset_dict["dev"], test=dataset_dict["test"])
            # Fallback: derive dev and test from a single split
            base_key = "train" if "train" in dataset_dict else sorted(dataset_dict.keys())[0]
            derived = dataset_dict[base_key].train_test_split(test_size=0.1, seed=self.seed)
            return DatasetSplit(dev=derived["train"], test=derived["test"])

        # Otherwise treat it as a JSONL file - read line by line to handle large files
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for lineno, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    preview = raw_line[:120].replace("\n", " ")
                    raise ValueError(
                        f"Invalid JSON on line {lineno} of {path}: {preview!r}"
                    ) from e
                records.append(obj)

        if not records:
            raise ValueError(f"No records loaded from {path}; is the file empty or non JSONL?")

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
            # enforce presence of dev and test
            return DatasetSplit(dev=dataset_dict["dev"], test=dataset_dict["test"])
        split = builder()
        self.hydrate_cache(split)
        return split


def load_conversation_dataset(
    dataset_name: str,
    split: float = 0.9,
    limit: Optional[int] = None,
    train_only: bool = True,
    holdout_fraction: Optional[float] = None,
) -> DatasetSplit:
    """Load SWE-bench and project into prompt and response pairs using dev and test splits.

    Parameters
    ----------
    dataset_name:
        Hugging Face dataset identifier (for example "SWE-bench/SWE-bench").
    split:
        Fraction of data to use for training (dev) split. Default: 0.9.
    limit:
        Optional limit on number of examples to load from dev split.
    train_only:
        If True, load only the "train" split from HF dataset. Default: True.
    holdout_fraction:
        Optional fraction for test split when train_only is True.
        If None, uses (1 - split). If 0.0, no test split is created.
    """

    # Determine how to load the dataset
    if train_only:
        # Load only the train split
        dataset_any = load_dataset(dataset_name, split="train")
        if not isinstance(dataset_any, Dataset):
            raise TypeError("Expected Dataset when using train_only mode")
        
        # Determine if we should create a holdout
        if holdout_fraction is None:
            holdout_fraction = 1 - split
        
        if holdout_fraction > 0:
            derived = dataset_any.train_test_split(test_size=holdout_fraction, seed=42)
            dev_source, test_source = derived["train"], derived["test"]
        else:
            # No holdout: all data goes to dev, empty test
            dev_source = dataset_any
            test_source = Dataset.from_dict({"prompt": [], "response": [], "label": []})
    else:
        # Legacy behavior: try to load from existing splits
        dataset_any = load_dataset(dataset_name)
        # Resolve to dev and test sources
        if isinstance(dataset_any, DatasetDict):
            if "dev" in dataset_any and "test" in dataset_any:
                dev_source = dataset_any["dev"]
                test_source = dataset_any["test"]
            else:
                # Fallback: derive dev and test from a single split
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
        if holdout_fraction is None or holdout_fraction > 0:
            test_cap = max(limit // 10, 1)
            test_source = test_source.select(range(min(test_cap, len(test_source))))

    # Project fields to prompt, response, and label
    def _project(src: Dataset) -> Dataset:
        prompts: List[str] = []
        responses: List[str] = []
        labels: List[SpeedrunLabel] = []
        for row in src:
            row_dict = dict(row)
            prompt = row_dict.get("problem_statement", "") or ""
            
            # Fallback response selection: change_summary -> patch -> test_patch -> ""
            response = (
                row_dict.get("change_summary", "")
                or row_dict.get("patch", "")
                or row_dict.get("test_patch", "")
                or ""
            )
            
            # Label heuristic: If resolved field exists, use it; otherwise check patch
            if "resolved" in row_dict:
                resolved = row_dict.get("resolved", False)
                label = SpeedrunLabel.SUCCESS if resolved else SpeedrunLabel.FAILURE
            else:
                # If non-empty patch exists, label as SUCCESS
                patch = row_dict.get("patch", "")
                label = SpeedrunLabel.SUCCESS if patch else SpeedrunLabel.FAILURE
            
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

    dev_dataset = _project(dev_source) if len(dev_source) > 0 else dev_source
    test_dataset = _project(test_source) if len(test_source) > 0 else test_source

    return DatasetSplit(dev=dev_dataset, test=test_dataset)


__all__ = [
    "DatasetSplit",
    "SpeedrunDatasetBuilder",
    "load_conversation_dataset",
]

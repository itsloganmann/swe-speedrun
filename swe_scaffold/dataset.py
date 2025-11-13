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


def load_conversation_dataset(
    dataset_name: str, 
    split: Optional[float] = None, 
    holdout_fraction: Optional[float] = None,
    limit: Optional[int] = None
) -> DatasetSplit:
    """Load SWE-bench_Lite and project into prompt/response pairs using dev/test splits.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        split: Legacy parameter (deprecated, use holdout_fraction instead)
        holdout_fraction: Fraction of data to hold out for test. If None, all data goes to dev.
        limit: Optional limit on number of examples to load
    
    Returns:
        DatasetSplit with dev and test datasets
    """
    # Handle legacy dev_split parameter
    if split is not None and holdout_fraction is None:
        print(f"Warning: 'split' parameter is deprecated. Use 'holdout_fraction' instead. Default changed to no holdout.")
        holdout_fraction = 1.0 - split if split > 0 and split < 1 else None
    
    # Determine if we should create a holdout set
    use_holdout = holdout_fraction is not None and holdout_fraction > 0
    
    dataset_any = load_dataset(dataset_name)
    
    # Get the base dataset to work with
    if isinstance(dataset_any, DatasetDict):
        if "dev" in dataset_any and "test" in dataset_any:
            # Use existing dev split
            base_source = dataset_any["dev"]
        else:
            # Fallback: use train or first available split
            base_key = "train" if "train" in dataset_any else sorted(dataset_any.keys())[0]
            base_source = dataset_any[base_key]
    elif isinstance(dataset_any, Dataset):
        base_source = dataset_any
    else:
        raise TypeError("Only map-style datasets are supported for speedrun training")
    
    # Apply limit before splitting
    if limit is not None:
        base_source = base_source.select(range(min(limit, len(base_source))))
    
    # Split into dev/test if holdout requested
    if use_holdout:
        derived = base_source.train_test_split(test_size=holdout_fraction, seed=42)
        dev_source, test_source = derived["train"], derived["test"]
    else:
        # No holdout: all data goes to dev
        dev_source = base_source
        test_source = None

    # Project fields to prompt/response/label with fallback chain and filtering
    def _project(src: Dataset, skip_empty: bool = True) -> tuple[Dataset, int]:
        prompts: List[str] = []
        responses: List[str] = []
        labels: List[SpeedrunLabel] = []
        skipped = 0
        
        for row in src:
            row_dict = dict(row)
            prompt = row_dict.get("problem_statement", "") or ""
            
            # Response fallback chain: change_summary -> patch -> test_patch
            response = row_dict.get("change_summary", "") or ""
            if not response.strip():
                response = row_dict.get("patch", "") or ""
            if not response.strip():
                response = row_dict.get("test_patch", "") or ""
            
            # Skip if empty after fallback chain
            if skip_empty and not response.strip():
                skipped += 1
                continue
            
            resolved = row_dict.get("resolved", False)
            label = SpeedrunLabel.SUCCESS if resolved else SpeedrunLabel.FAILURE
            prompts.append(prompt)
            responses.append(response)
            labels.append(label)
        
        dataset = Dataset.from_dict(
            {
                "prompt": prompts,
                "response": responses,
                "label": [label.value for label in labels],
            }
        )
        return dataset, skipped

    dev_dataset, skipped_dev = _project(dev_source, skip_empty=True)
    
    if test_source is not None:
        test_dataset, skipped_test = _project(test_source, skip_empty=True)
        total_skipped = skipped_dev + skipped_test
    else:
        # Create empty test dataset with required columns
        test_dataset = Dataset.from_dict({
            "prompt": [],
            "response": [],
            "label": [],
        })
        total_skipped = skipped_dev
    
    if total_skipped > 0:
        print(f"Skipped {total_skipped} empty-response examples")

    return DatasetSplit(dev=dev_dataset, test=test_dataset)


__all__ = [
    "DatasetSplit",
    "SpeedrunDatasetBuilder",
    "load_conversation_dataset",
]

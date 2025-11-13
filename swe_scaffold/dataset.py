"""Dataset helpers for SWE speedrun fine-tuning."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from datasets import Dataset, DatasetDict, load_dataset

from .labels import SpeedrunLabel, batch_label_conversations


@dataclass(slots=True)
class DatasetSplit:
    dev: Dataset
    test: Dataset


@dataclass(slots=True)
class ProjectionResult:
    """Result of projecting a dataset with filtering and skip tracking."""
    split: DatasetSplit
    skipped: List[Dict[str, str]] = field(default_factory=list)
    total_kept: int = 0
    total_skipped: int = 0


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
    split: float = 0.9, 
    limit: Optional[int] = None,
    train_only: bool = True,
    holdout_fraction: Optional[float] = None,
    include_empty: bool = False,
    min_patch_chars: Optional[int] = None,
) -> ProjectionResult:
    """Load SWE-bench and project into prompt/response pairs using dev/test splits.
    
    Parameters
    ----------
    dataset_name : str
        HuggingFace dataset identifier
    split : float
        Legacy parameter for backward compatibility (default: 0.9)
    limit : Optional[int]
        Maximum number of examples to use. If None or <=0, no limit is applied.
    train_only : bool
        If True, no holdout/test split is created (default: True)
    holdout_fraction : Optional[float]
        Fraction of data to hold out for test. Overrides split parameter.
        If None and train_only is False, uses (1 - split).
    include_empty : bool
        If True, include examples with empty responses (default: False)
    min_patch_chars : Optional[int]
        Minimum patch length in characters. Examples below this are skipped.
    
    Returns
    -------
    ProjectionResult
        Contains the dataset split and skipped example metadata
    """

    dataset_any = load_dataset(dataset_name)
    skipped_records: List[Dict[str, str]] = []
    
    # Resolve holdout fraction
    if holdout_fraction is not None:
        effective_holdout = holdout_fraction
    elif train_only:
        effective_holdout = 0.0
    else:
        effective_holdout = 1.0 - split
    
    # Resolve to train source
    if isinstance(dataset_any, DatasetDict):
        if "train" in dataset_any:
            base_source = dataset_any["train"]
        elif "dev" in dataset_any:
            base_source = dataset_any["dev"]
        else:
            base_source = dataset_any[sorted(dataset_any.keys())[0]]
    elif isinstance(dataset_any, Dataset):
        base_source = dataset_any
    else:
        raise TypeError("Only map-style datasets are supported for speedrun training")

    # Apply limit if specified and > 0
    if limit is not None and limit > 0:
        base_source = base_source.select(range(min(limit, len(base_source))))

    # Project fields to prompt/response/label with filtering
    prompts: List[str] = []
    responses: List[str] = []
    labels: List[SpeedrunLabel] = []
    
    for idx, row in enumerate(base_source):
        row_dict = dict(row)
        
        # Build prompt
        prompt = row_dict.get("problem_statement", "") or ""
        
        # Build response with fallback order: change_summary -> patch -> test_patch -> ""
        response = ""
        response_source = None  # Track which field was used
        if row_dict.get("change_summary"):
            response = row_dict["change_summary"]
            response_source = "change_summary"
        elif row_dict.get("patch"):
            response = row_dict["patch"]
            response_source = "patch"
        elif row_dict.get("test_patch"):
            response = row_dict["test_patch"]
            response_source = "test_patch"
        
        # Filter: skip empty responses unless include_empty
        if not response and not include_empty:
            instance_id = row_dict.get("instance_id", f"row_{idx}")
            skipped_records.append({
                "instance_id": instance_id,
                "reason": "empty_response",
            })
            continue
        
        # Filter: skip if patch is below min_patch_chars threshold
        # Only applies when we're actually using the patch as the response
        if min_patch_chars is not None and response_source == "patch":
            if len(response) < min_patch_chars:
                instance_id = row_dict.get("instance_id", f"row_{idx}")
                skipped_records.append({
                    "instance_id": instance_id,
                    "reason": "patch_too_short",
                    "patch_length": str(len(response)),
                })
                continue
        
        # Labeling heuristic: resolved flag if present, else patch presence
        if "resolved" in row_dict:
            resolved = row_dict.get("resolved", False)
            label = SpeedrunLabel.SUCCESS if resolved else SpeedrunLabel.FAILURE
        else:
            # Fallback: label as success if patch exists, failure otherwise
            has_patch = bool(row_dict.get("patch", ""))
            label = SpeedrunLabel.SUCCESS if has_patch else SpeedrunLabel.FAILURE
        
        prompts.append(prompt)
        responses.append(response)
        labels.append(label)
    
    # Create full dataset
    full_dataset = Dataset.from_dict(
        {
            "prompt": prompts,
            "response": responses,
            "label": [label.value for label in labels],
        }
    )
    
    # Split into dev/test based on holdout fraction
    if effective_holdout > 0.0:
        split_result = full_dataset.train_test_split(test_size=effective_holdout, seed=42)
        dev_dataset = split_result["train"]
        test_dataset = split_result["test"]
    else:
        # No holdout: all data goes to dev, test is empty
        dev_dataset = full_dataset
        test_dataset = Dataset.from_dict({
            "prompt": [],
            "response": [],
            "label": [],
        })

    split_obj = DatasetSplit(dev=dev_dataset, test=test_dataset)
    
    return ProjectionResult(
        split=split_obj,
        skipped=skipped_records,
        total_kept=len(full_dataset),
        total_skipped=len(skipped_records),
    )


__all__ = [
    "DatasetSplit",
    "ProjectionResult",
    "SpeedrunDatasetBuilder",
    "load_conversation_dataset",
]

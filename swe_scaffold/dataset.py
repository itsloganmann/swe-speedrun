"""Dataset helpers for SWE speedrun fine-tuning."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from datasets import Dataset, DatasetDict, load_dataset

from .config import DEFAULT_SOURCE_DATASET, DEFAULT_SOURCE_REVISION
from .labels import SpeedrunLabel, batch_label_conversations


@dataclass(slots=True)
class DatasetSplit:
    dev: Dataset
    test: Dataset


@dataclass(slots=True)
class ProjectionResult(DatasetSplit):
    """Result of dataset projection including skipped examples."""

    skipped: List[dict]

    @property
    def split(self) -> DatasetSplit:
        return self

    @property
    def total_kept(self) -> int:
        return len(self.dev) + len(self.test)

    @property
    def total_skipped(self) -> int:
        return len(self.skipped)


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
            base_key = (
                "train" if "train" in dataset_dict else sorted(dataset_dict.keys())[0]
            )
            derived = dataset_dict[base_key].train_test_split(
                test_size=0.1, seed=self.seed
            )
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
            raise ValueError(
                f"No records loaded from {path}; is the file empty or non JSONL?"
            )

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
        # Empty splits need an Arrow shard so load_from_disk can reopen them.
        empty_shards = {
            str(key): 1 for key, dataset in payload.items() if len(dataset) == 0
        }
        payload.save_to_disk(str(self.cache_path), num_shards=empty_shards)

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
    split: Optional[float] = None,
    limit: Optional[int] = None,
    train_only: bool = True,
    holdout_fraction: Optional[float] = None,
    *,
    min_patch_chars: Optional[int] = None,
    include_empty: bool = False,
    revision: str | None = None,
) -> ProjectionResult:
    """Load SWE-bench and project into prompt and response pairs using dev and test splits.

    Parameters
    ----------
    dataset_name:
        Hugging Face dataset identifier (for example "SWE-bench/SWE-bench").
    split:
        Optional legacy training fraction. When provided, derives a holdout unless explicitly overridden.
    limit:
        Optional positive cap on dev examples. None and nonpositive values are unlimited.
    train_only:
        If True, load only the "train" split from HF dataset. Default: True.
    holdout_fraction:
        Explicit holdout fraction. Defaults to no holdout when split is omitted.
        An existing dev/test pair is preserved in legacy no-train-only mode without overrides.
    min_patch_chars:
        Minimum stripped response length for patch and test_patch sources.
    include_empty:
        Retain examples with no response after field fallback.
    revision:
        Optional Hugging Face revision. The default SWE-bench source uses a pinned
        revision that still contains train; other sources use their own default.
    """

    if split is not None and (not math.isfinite(split) or not 0 < split <= 1):
        raise ValueError("split must be greater than zero and at most one")
    fraction = (
        holdout_fraction
        if holdout_fraction is not None
        else (1 - split if split is not None else 0.0)
    )
    if not math.isfinite(fraction) or not 0 <= fraction < 1:
        warnings.warn(
            "holdout_fraction must be finite and between zero and one; using no holdout.",
            UserWarning,
            stacklevel=2,
        )
        fraction = 0.0
    if limit is not None and not isinstance(limit, int):
        raise ValueError("limit must be an integer or None")
    if min_patch_chars is not None and (
        not isinstance(min_patch_chars, int) or min_patch_chars < 0
    ):
        raise ValueError("min_patch_chars must be a nonnegative integer or None")
    if not isinstance(train_only, bool) or not isinstance(include_empty, bool):
        raise ValueError("train_only and include_empty must be booleans")

    if split is not None:
        warnings.warn(
            "split is deprecated; the default is now no holdout. Use holdout_fraction.",
            FutureWarning,
            stacklevel=2,
        )

    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise ValueError("revision must be a nonempty string or None")
    source_revision = revision
    if source_revision is None and dataset_name == DEFAULT_SOURCE_DATASET:
        source_revision = DEFAULT_SOURCE_REVISION
    if source_revision is None:
        dataset_any = (
            load_dataset(dataset_name, split="train")
            if train_only
            else load_dataset(dataset_name)
        )
    else:
        dataset_any = (
            load_dataset(dataset_name, split="train", revision=source_revision)
            if train_only
            else load_dataset(dataset_name, revision=source_revision)
        )
    if (
        isinstance(dataset_any, DatasetDict)
        and not train_only
        and "dev" in dataset_any
        and "test" in dataset_any
        and split is None
        and holdout_fraction is None
    ):
        dev_source, test_source = dataset_any["dev"], dataset_any["test"]
    else:
        if isinstance(dataset_any, DatasetDict):
            if not dataset_any:
                raise ValueError("Dataset contains no splits")
            if "train" in dataset_any:
                base_key = "train"
            elif "dev" in dataset_any:
                base_key = "dev"
            else:
                raise ValueError("Dataset must contain a train or dev source split")
            dataset_any = dataset_any[base_key]
        if not isinstance(dataset_any, Dataset):
            raise TypeError(
                "Only map-style datasets are supported for speedrun training"
            )
        if fraction > 0:
            derived = dataset_any.train_test_split(test_size=fraction, seed=42)
            dev_source, test_source = derived["train"], derived["test"]
        else:
            dev_source = dataset_any
            test_source = Dataset.from_dict({"prompt": [], "response": [], "label": []})

    # Optional limits
    if limit is not None and limit > 0:
        dev_source = dev_source.select(range(min(limit, len(dev_source))))
        if len(test_source) > 0:
            test_cap = max(limit // 10, 1)
            test_source = test_source.select(range(min(test_cap, len(test_source))))

    # Project fields to prompt/response/label
    def _project(src: Dataset) -> tuple[Dataset, List[dict]]:
        prompts: List[str] = []
        responses: List[str] = []
        labels: List[SpeedrunLabel] = []
        skipped: List[dict] = []

        for row in src:
            row_dict = dict(row)
            instance_id = row_dict.get("instance_id", "unknown")
            prompt = row_dict.get("problem_statement", "") or ""

            # Response fallback: change_summary → patch → test_patch
            change_summary = row_dict.get("change_summary", "") or ""
            patch = row_dict.get("patch", "") or ""
            test_patch = row_dict.get("test_patch", "") or ""

            # Track which fields are available
            has_change_summary = bool(change_summary.strip())
            has_patch = bool(patch.strip())
            has_test_patch = bool(test_patch.strip())

            # Select response with fallback
            response = ""
            response_source = None
            if has_change_summary:
                response = change_summary
                response_source = "change_summary"
            elif has_patch:
                response = patch
                response_source = "patch"
            elif has_test_patch:
                response = test_patch
                response_source = "test_patch"

            response_length = len(response.strip())

            # Check if we should skip this example
            skip_reason = None

            # Check for empty response
            if response_length == 0 and not include_empty:
                skip_reason = "empty_response"

            # Check patch length threshold (only for patch/test_patch sources)
            if (
                skip_reason is None
                and min_patch_chars is not None
                and response_source in ("patch", "test_patch")
                and response_length < min_patch_chars
            ):
                skip_reason = "short_patch"

            # Record skip or add to dataset
            if skip_reason:
                skipped.append(
                    {
                        "instance_id": instance_id,
                        "reason": skip_reason,
                        "source_fields": {
                            "has_change_summary": has_change_summary,
                            "has_patch": has_patch,
                            "has_test_patch": has_test_patch,
                        },
                        "patch_len": response_length,
                    }
                )
            else:
                if "resolved" in row_dict:
                    resolved = row_dict["resolved"]
                else:
                    resolved = bool(row_dict.get("patch", ""))
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

    dev_dataset, dev_skipped = _project(dev_source)
    test_dataset, test_skipped = _project(test_source)

    all_skipped = dev_skipped + test_skipped

    return ProjectionResult(
        dev=dev_dataset,
        test=test_dataset,
        skipped=all_skipped,
    )


__all__ = [
    "DatasetSplit",
    "ProjectionResult",
    "SpeedrunDatasetBuilder",
    "load_conversation_dataset",
]

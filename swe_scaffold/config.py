"""Configuration models for the SWE speedrun training pipeline."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

DEFAULT_SOURCE_DATASET = "SWE-bench/SWE-bench"
DEFAULT_SOURCE_REVISION = "ac1995e464b07cbd69702e8490a07f6d557b69b5"


@dataclass(slots=True)
class LoRAConfig:
    """Hyperparameters for Low-Rank Adaptation fine-tuning."""

    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]
    )


@dataclass(slots=True)
class TrainingConfig:
    """Training hyperparameters shared across scripts."""

    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    output_dir: Path = Path("artifacts/checkpoints/qwen-speedrun")
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    max_steps: Optional[int] = None
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 100
    max_grad_norm: float = 1.0
    fp16: bool = True
    seed: int = 42
    lora: LoRAConfig = field(default_factory=LoRAConfig)


@dataclass(slots=True, init=False)
class DatasetConfig:
    """Configuration for dataset assembly and caching."""

    source_dataset: str
    local_cache: Path
    train_only: bool
    holdout_fraction: Optional[float]
    text_fields: List[str]
    max_examples: Optional[int]
    min_patch_chars: Optional[int]
    include_empty: bool
    skipped_report_path: Optional[Path]
    source_revision: str | None

    def __init__(
        self,
        source_dataset: str = DEFAULT_SOURCE_DATASET,
        local_cache: Path = Path("data/processed/swe-speedrun.jsonl"),
        dev_split: Optional[float] = None,
        train_only: bool = True,
        holdout_fraction: Optional[float] = None,
        text_fields: Optional[List[str]] = None,
        max_examples: Optional[int] = None,
        min_patch_chars: Optional[int] = None,
        include_empty: bool = False,
        skipped_report_path: Optional[Path] = None,
        source_revision: str | None = None,
    ) -> None:
        self.source_dataset = source_dataset
        self.source_revision = source_revision
        self.local_cache = local_cache
        self.train_only = train_only
        self.holdout_fraction = holdout_fraction
        if dev_split is not None and holdout_fraction is None:
            self.dev_split = dev_split
            warnings.warn(
                "dev_split is deprecated; the default is now no holdout. Use holdout_fraction.",
                FutureWarning,
                stacklevel=2,
            )
        self.text_fields = (
            ["problem_statement", "change_summary"]
            if text_fields is None
            else list(text_fields)
        )
        self.max_examples = max_examples
        self.min_patch_chars = min_patch_chars
        self.include_empty = include_empty
        self.skipped_report_path = skipped_report_path

    @property
    def dev_split(self) -> float:
        return 1.0 - (self.holdout_fraction or 0.0)

    @dev_split.setter
    def dev_split(self, value: float) -> None:
        if not 0 < value <= 1:
            raise ValueError("dev_split must be greater than zero and at most one")
        self.holdout_fraction = 1.0 - value


@dataclass(slots=True)
class SpeedrunConfig:
    """Top-level configuration aggregator used by CLI scripts."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, payload: dict) -> "SpeedrunConfig":
        """Instantiate a configuration object from a nested dictionary."""

        dataset_payload = dict(payload.get("dataset", {}))
        if "dev_split" in dataset_payload:
            legacy_split = dataset_payload.pop("dev_split")
            if "holdout_fraction" not in dataset_payload:
                if not 0 < legacy_split <= 1:
                    raise ValueError(
                        "dev_split must be greater than zero and at most one"
                    )
                dataset_payload["holdout_fraction"] = 1.0 - legacy_split
                warnings.warn(
                    "dev_split is deprecated; the default is now no holdout. Use holdout_fraction.",
                    FutureWarning,
                    stacklevel=2,
                )
        for name in ("local_cache", "skipped_report_path"):
            if dataset_payload.get(name) is not None:
                dataset_payload[name] = Path(dataset_payload[name])
        dataset = DatasetConfig(**dataset_payload)
        training_payload = dict(payload.get("training", {}))
        lora_payload = training_payload.pop("lora", {})
        training = TrainingConfig(**training_payload, lora=LoRAConfig(**lora_payload))
        return cls(dataset=dataset, training=training)


__all__ = [
    "LoRAConfig",
    "TrainingConfig",
    "DatasetConfig",
    "SpeedrunConfig",
]

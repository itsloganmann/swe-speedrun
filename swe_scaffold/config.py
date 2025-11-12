"""Configuration models for the SWE speedrun training pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


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


@dataclass(slots=True)
class DatasetConfig:
    """Configuration for dataset assembly and caching."""

    source_dataset: str = "anchen-li/swe-bench-lite"
    local_cache: Path = Path("data/processed/swe-speedrun.jsonl")
    dev_split: float = 0.9
    text_fields: List[str] = field(
        default_factory=lambda: [
            "problem_statement",
            "change_summary",
        ]
    )
    max_examples: Optional[int] = 500


@dataclass(slots=True)
class SpeedrunConfig:
    """Top-level configuration aggregator used by CLI scripts."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, payload: dict) -> "SpeedrunConfig":
        """Instantiate a configuration object from a nested dictionary."""

        dataset = DatasetConfig(**payload.get("dataset", {}))
        training_payload = payload.get("training", {})
        lora_payload = training_payload.pop("lora", {})
        training = TrainingConfig(**training_payload, lora=LoRAConfig(**lora_payload))
        return cls(dataset=dataset, training=training)


__all__ = [
    "LoRAConfig",
    "TrainingConfig",
    "DatasetConfig",
    "SpeedrunConfig",
]

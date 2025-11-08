"""Top-level package for the SWE speedrun scaffold helpers."""

from .config import SpeedrunConfig, TrainingConfig, DatasetConfig, LoRAConfig
from .dataset import SpeedrunDatasetBuilder, load_conversation_dataset
from .labels import label_conversation, SpeedrunLabel
from .training import train_lora_model, load_tokenizer_and_model
from .evaluation import evaluate_model, bootstrap_confidence_interval

__all__ = [
    "SpeedrunConfig",
    "TrainingConfig",
    "DatasetConfig",
    "LoRAConfig",
    "SpeedrunDatasetBuilder",
    "load_conversation_dataset",
    "label_conversation",
    "SpeedrunLabel",
    "train_lora_model",
    "load_tokenizer_and_model",
    "evaluate_model",
    "bootstrap_confidence_interval",
]

"""Evaluation helpers for SWE speedrun fine-tuning."""
from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Iterable, Sequence, Tuple

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluate_model(model_path: str, dataset: Dataset, batch_size: int = 2) -> Tuple[float, Counter]:
    """Compute perplexity over the dataset and return basic label counts."""

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()

    losses = []
    label_counter: Counter = Counter(dataset["label"])

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        encoded = tokenizer(batch["text"], return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**encoded, labels=encoded["input_ids"])
        losses.append(outputs.loss.item())

    perplexity = float(torch.exp(torch.tensor(mean(losses)))) if losses else float("inf")
    return perplexity, label_counter


def bootstrap_confidence_interval(values: Sequence[float], iterations: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    """Return a bootstrap confidence interval for the provided metric."""

    if not values:
        raise ValueError("Cannot compute interval for empty sequence")

    tensor = torch.tensor(values)
    samples = []
    for _ in range(iterations):
        idx = torch.randint(0, len(values), (len(values),))
        samples.append(tensor[idx].mean().item())

    lower = torch.quantile(torch.tensor(samples), alpha / 2).item()
    upper = torch.quantile(torch.tensor(samples), 1 - alpha / 2).item()
    return lower, upper


__all__ = [
    "evaluate_model",
    "bootstrap_confidence_interval",
]

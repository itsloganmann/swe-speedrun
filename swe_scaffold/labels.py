"""Label helpers for SWE speedrun dataset curation."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence
import numbers


class SpeedrunLabel(str, Enum):
    """Binary label for SWE speedrun prompt curation."""

    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(slots=True)
class LabelThresholds:
    """Threshold container for heuristics when assigning labels."""

    max_response_tokens: int = 2048
    min_pass_rate: float = 0.6
    max_turns: int = 18


DEFAULT_THRESHOLDS = LabelThresholds()


def _to_int(value: object, default: int) -> int:
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return int(round(float(value)))
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return int(float(value))
            except ValueError:
                return default
    return default if value is None else default


def _to_float(value: object, default: float) -> float:
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default if value is None else default


def label_conversation(conversation: Mapping[str, object], thresholds: LabelThresholds = DEFAULT_THRESHOLDS) -> SpeedrunLabel:
    """Return a label for a single SWE-agent conversation transcript.

    Parameters
    ----------
    conversation:
        Dictionary with keys like ``turns``, ``pass_rate`` and ``response_tokens``.
    thresholds:
        Heuristic boundaries for determining if the run qualifies as a "success".
    """

    turns = _to_int(conversation.get("turns"), 0)
    pass_rate = _to_float(conversation.get("pass_rate"), 0.0)
    response_tokens = _to_int(conversation.get("response_tokens"), thresholds.max_response_tokens)

    if pass_rate >= thresholds.min_pass_rate and turns <= thresholds.max_turns and response_tokens <= thresholds.max_response_tokens:
        return SpeedrunLabel.SUCCESS
    return SpeedrunLabel.FAILURE


def batch_label_conversations(conversations: Iterable[Mapping[str, object]], thresholds: LabelThresholds = DEFAULT_THRESHOLDS) -> Sequence[SpeedrunLabel]:
    """Assign labels to a batch of conversations."""

    return [label_conversation(conv, thresholds) for conv in conversations]


__all__ = [
    "SpeedrunLabel",
    "LabelThresholds",
    "DEFAULT_THRESHOLDS",
    "label_conversation",
    "batch_label_conversations",
]

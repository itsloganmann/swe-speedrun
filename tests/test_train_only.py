"""Tests for train_only and holdout_fraction behavior."""
import pytest
from datasets import Dataset, DatasetDict

from swe_scaffold.dataset import load_conversation_dataset


def test_train_only_produces_empty_test_split(monkeypatch):
    """Test that train_only=True produces an empty test split."""
    mock_data = DatasetDict({
        "train": Dataset.from_dict({
            "instance_id": ["test-1", "test-2", "test-3"],
            "problem_statement": ["P1", "P2", "P3"],
            "change_summary": ["S1", "S2", "S3"],
            "patch": ["", "", ""],
        })
    })
    
    def mock_load_dataset(name):
        return mock_data
    
    monkeypatch.setattr("swe_scaffold.dataset.load_dataset", mock_load_dataset)
    
    result = load_conversation_dataset(
        "test-dataset",
        train_only=True,
    )
    
    assert len(result.split.dev) == 3
    assert len(result.split.test) == 0


def test_holdout_fraction_creates_test_split(monkeypatch):
    """Test that holdout_fraction creates appropriate test split."""
    mock_data = DatasetDict({
        "train": Dataset.from_dict({
            "instance_id": [f"test-{i}" for i in range(100)],
            "problem_statement": [f"P{i}" for i in range(100)],
            "change_summary": [f"S{i}" for i in range(100)],
            "patch": [""] * 100,
        })
    })
    
    def mock_load_dataset(name):
        return mock_data
    
    monkeypatch.setattr("swe_scaffold.dataset.load_dataset", mock_load_dataset)
    
    result = load_conversation_dataset(
        "test-dataset",
        train_only=False,
        holdout_fraction=0.2,
    )
    
    # Should split approximately 80/20
    assert len(result.split.dev) == 80
    assert len(result.split.test) == 20


def test_holdout_fraction_overrides_train_only(monkeypatch):
    """Test that explicit holdout_fraction overrides train_only."""
    mock_data = DatasetDict({
        "train": Dataset.from_dict({
            "instance_id": [f"test-{i}" for i in range(10)],
            "problem_statement": [f"P{i}" for i in range(10)],
            "change_summary": [f"S{i}" for i in range(10)],
            "patch": [""] * 10,
        })
    })
    
    def mock_load_dataset(name):
        return mock_data
    
    monkeypatch.setattr("swe_scaffold.dataset.load_dataset", mock_load_dataset)
    
    # Even though train_only would suggest no split, holdout_fraction takes precedence
    result = load_conversation_dataset(
        "test-dataset",
        train_only=True,  # This should be overridden
        holdout_fraction=0.3,
    )
    
    # Should have a 70/30 split
    assert len(result.split.dev) == 7
    assert len(result.split.test) == 3


def test_legacy_split_parameter_backward_compatibility(monkeypatch):
    """Test that legacy split parameter still works."""
    mock_data = DatasetDict({
        "train": Dataset.from_dict({
            "instance_id": [f"test-{i}" for i in range(10)],
            "problem_statement": [f"P{i}" for i in range(10)],
            "change_summary": [f"S{i}" for i in range(10)],
            "patch": [""] * 10,
        })
    })
    
    def mock_load_dataset(name):
        return mock_data
    
    monkeypatch.setattr("swe_scaffold.dataset.load_dataset", mock_load_dataset)
    
    result = load_conversation_dataset(
        "test-dataset",
        split=0.8,
        train_only=False,
    )
    
    # Should split 80/20
    assert len(result.split.dev) == 8
    assert len(result.split.test) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

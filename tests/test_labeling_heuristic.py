"""Tests for the labeling heuristic in dataset loading."""
import pytest
from datasets import Dataset, DatasetDict

from swe_scaffold.dataset import load_conversation_dataset
from swe_scaffold.labels import SpeedrunLabel


def test_labeling_uses_resolved_field_when_present(monkeypatch):
    """Test that labeling uses resolved field as primary heuristic."""
    mock_data = DatasetDict({
        "train": Dataset.from_dict({
            "instance_id": ["test-1", "test-2"],
            "problem_statement": ["P1", "P2"],
            "change_summary": ["S1", "S2"],
            "patch": ["Patch1", "Patch2"],
            "resolved": [True, False],
        })
    })
    
    def mock_load_dataset(name):
        return mock_data
    
    monkeypatch.setattr("swe_scaffold.dataset.load_dataset", mock_load_dataset)
    
    result = load_conversation_dataset(
        "test-dataset",
        train_only=True,
    )
    
    labels = result.split.dev["label"]
    assert labels[0] == SpeedrunLabel.SUCCESS.value
    assert labels[1] == SpeedrunLabel.FAILURE.value


def test_labeling_fallback_to_patch_presence(monkeypatch):
    """Test that labeling falls back to patch presence when resolved is absent."""
    mock_data = DatasetDict({
        "train": Dataset.from_dict({
            "instance_id": ["test-1", "test-2", "test-3"],
            "problem_statement": ["P1", "P2", "P3"],
            "change_summary": ["S1", "", "S3"],
            "patch": ["Patch1", "", ""],
        })
    })
    
    def mock_load_dataset(name):
        return mock_data
    
    monkeypatch.setattr("swe_scaffold.dataset.load_dataset", mock_load_dataset)
    
    result = load_conversation_dataset(
        "test-dataset",
        train_only=True,
        include_empty=True,
    )
    
    labels = result.split.dev["label"]
    # test-1 has patch -> success
    assert labels[0] == SpeedrunLabel.SUCCESS.value
    # test-2 no patch -> failure
    assert labels[1] == SpeedrunLabel.FAILURE.value
    # test-3 no patch -> failure
    assert labels[2] == SpeedrunLabel.FAILURE.value


def test_labeling_resolved_takes_precedence_over_patch(monkeypatch):
    """Test that resolved field takes precedence over patch presence."""
    mock_data = DatasetDict({
        "train": Dataset.from_dict({
            "instance_id": ["test-1", "test-2"],
            "problem_statement": ["P1", "P2"],
            "change_summary": ["S1", "S2"],
            "patch": ["Patch1", ""],  # test-1 has patch, test-2 doesn't
            "resolved": [False, True],  # But resolved says opposite
        })
    })
    
    def mock_load_dataset(name):
        return mock_data
    
    monkeypatch.setattr("swe_scaffold.dataset.load_dataset", mock_load_dataset)
    
    result = load_conversation_dataset(
        "test-dataset",
        train_only=True,
        include_empty=True,
    )
    
    labels = result.split.dev["label"]
    # Resolved field takes precedence
    assert labels[0] == SpeedrunLabel.FAILURE.value  # resolved=False
    assert labels[1] == SpeedrunLabel.SUCCESS.value  # resolved=True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

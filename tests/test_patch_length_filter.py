"""Tests for patch length filtering in dataset loading."""
import tempfile
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

from swe_scaffold.dataset import load_conversation_dataset


def test_patch_length_filter_excludes_short_patches(monkeypatch):
    """Test that min_patch_chars filters out patches below threshold."""
    # Create a mock dataset with varying patch lengths
    mock_data = DatasetDict({
        "train": Dataset.from_dict({
            "instance_id": ["test-1", "test-2", "test-3"],
            "problem_statement": ["Problem 1", "Problem 2", "Problem 3"],
            "patch": ["short", "this is a much longer patch content here", "mid"],
            "change_summary": ["", "", ""],
        })
    })
    
    def mock_load_dataset(name):
        return mock_data
    
    monkeypatch.setattr("swe_scaffold.dataset.load_dataset", mock_load_dataset)
    
    # Load with min_patch_chars=20
    result = load_conversation_dataset(
        "test-dataset",
        min_patch_chars=20,
        train_only=True,
    )
    
    # Should only keep the example with patch length >= 20
    assert result.total_kept == 1
    assert result.total_skipped == 2
    
    # Check skip reasons
    skipped_reasons = [r["reason"] for r in result.skipped]
    assert skipped_reasons.count("patch_too_short") == 2


def test_patch_length_filter_metadata_includes_length(monkeypatch):
    """Test that skipped records include patch length in metadata."""
    mock_data = DatasetDict({
        "train": Dataset.from_dict({
            "instance_id": ["test-1"],
            "problem_statement": ["Problem 1"],
            "patch": ["short"],
            "change_summary": [""],
        })
    })
    
    def mock_load_dataset(name):
        return mock_data
    
    monkeypatch.setattr("swe_scaffold.dataset.load_dataset", mock_load_dataset)
    
    result = load_conversation_dataset(
        "test-dataset",
        min_patch_chars=100,
        train_only=True,
    )
    
    assert result.total_skipped == 1
    assert result.skipped[0]["reason"] == "patch_too_short"
    assert "patch_length" in result.skipped[0]
    assert result.skipped[0]["patch_length"] == "5"


def test_no_patch_length_filter_keeps_all(monkeypatch):
    """Test that without min_patch_chars, all examples are kept."""
    mock_data = DatasetDict({
        "train": Dataset.from_dict({
            "instance_id": ["test-1", "test-2"],
            "problem_statement": ["Problem 1", "Problem 2"],
            "patch": ["x", "very long patch content"],
            "change_summary": ["", ""],
        })
    })
    
    def mock_load_dataset(name):
        return mock_data
    
    monkeypatch.setattr("swe_scaffold.dataset.load_dataset", mock_load_dataset)
    
    result = load_conversation_dataset(
        "test-dataset",
        min_patch_chars=None,
        train_only=True,
    )
    
    # All examples should be kept (no filter)
    assert result.total_kept == 2
    assert result.total_skipped == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

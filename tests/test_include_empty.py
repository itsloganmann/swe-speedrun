"""Tests for include_empty flag in dataset loading."""
import pytest
from datasets import Dataset, DatasetDict

from swe_scaffold.dataset import load_conversation_dataset


def test_include_empty_false_skips_empty_responses(monkeypatch):
    """Test that include_empty=False skips examples with empty responses."""
    mock_data = DatasetDict({
        "train": Dataset.from_dict({
            "instance_id": ["test-1", "test-2", "test-3"],
            "problem_statement": ["Problem 1", "Problem 2", "Problem 3"],
            "change_summary": ["", "Valid summary", ""],
            "patch": ["", "", "Valid patch"],
        })
    })
    
    def mock_load_dataset(name):
        return mock_data
    
    monkeypatch.setattr("swe_scaffold.dataset.load_dataset", mock_load_dataset)
    
    result = load_conversation_dataset(
        "test-dataset",
        include_empty=False,
        train_only=True,
    )
    
    # Should only keep examples with non-empty responses
    assert result.total_kept == 2
    assert result.total_skipped == 1
    assert result.skipped[0]["reason"] == "empty_response"


def test_include_empty_true_keeps_empty_responses(monkeypatch):
    """Test that include_empty=True keeps examples with empty responses."""
    mock_data = DatasetDict({
        "train": Dataset.from_dict({
            "instance_id": ["test-1", "test-2"],
            "problem_statement": ["Problem 1", "Problem 2"],
            "change_summary": ["", "Valid summary"],
            "patch": ["", ""],
        })
    })
    
    def mock_load_dataset(name):
        return mock_data
    
    monkeypatch.setattr("swe_scaffold.dataset.load_dataset", mock_load_dataset)
    
    result = load_conversation_dataset(
        "test-dataset",
        include_empty=True,
        train_only=True,
    )
    
    # Should keep all examples including empty
    assert result.total_kept == 2
    assert result.total_skipped == 0


def test_response_fallback_order(monkeypatch):
    """Test the response fallback order: change_summary -> patch -> test_patch."""
    mock_data = DatasetDict({
        "train": Dataset.from_dict({
            "instance_id": ["test-1", "test-2", "test-3", "test-4"],
            "problem_statement": ["P1", "P2", "P3", "P4"],
            "change_summary": ["Summary 1", "", "", ""],
            "patch": ["Patch 1", "Patch 2", "", ""],
            "test_patch": ["Test 1", "Test 2", "Test 3", ""],
        })
    })
    
    def mock_load_dataset(name):
        return mock_data
    
    monkeypatch.setattr("swe_scaffold.dataset.load_dataset", mock_load_dataset)
    
    result = load_conversation_dataset(
        "test-dataset",
        include_empty=False,
        train_only=True,
    )
    
    # First three should be kept (have responses via fallback chain)
    # Fourth is empty and should be skipped
    assert result.total_kept == 3
    assert result.total_skipped == 1
    
    # Verify responses follow fallback order
    responses = result.split.dev["response"]
    assert responses[0] == "Summary 1"  # change_summary preferred
    assert responses[1] == "Patch 2"    # patch fallback
    assert responses[2] == "Test 3"     # test_patch fallback


def test_empty_response_skip_reason_includes_instance_id(monkeypatch):
    """Test that skipped empty responses include instance_id."""
    mock_data = DatasetDict({
        "train": Dataset.from_dict({
            "instance_id": ["django-12345"],
            "problem_statement": ["Problem"],
            "change_summary": [""],
            "patch": [""],
        })
    })
    
    def mock_load_dataset(name):
        return mock_data
    
    monkeypatch.setattr("swe_scaffold.dataset.load_dataset", mock_load_dataset)
    
    result = load_conversation_dataset(
        "test-dataset",
        include_empty=False,
        train_only=True,
    )
    
    assert result.total_skipped == 1
    assert result.skipped[0]["instance_id"] == "django-12345"
    assert result.skipped[0]["reason"] == "empty_response"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for dataset filtering and auditing features."""
import json
import tempfile
from pathlib import Path

import pytest
from datasets import Dataset

from swe_scaffold.dataset import load_conversation_dataset, ProjectionResult, DatasetSplit


def create_mock_dataset_dict(records):
    """Helper to create a mock dataset from records."""
    from datasets import DatasetDict
    dataset = Dataset.from_dict({
        key: [r.get(key, "") for r in records] 
        for key in ["instance_id", "problem_statement", "change_summary", "patch", "test_patch", "resolved"]
    })
    return DatasetDict({"train": dataset})


def test_patch_length_filter_excludes_short_patches(monkeypatch):
    """Test that patches below min_patch_chars are excluded."""
    # Create mock dataset with varying patch lengths
    records = [
        {
            "instance_id": "test-1",
            "problem_statement": "Problem 1",
            "change_summary": "",
            "patch": "short",  # 5 chars
            "test_patch": "",
            "resolved": False,
        },
        {
            "instance_id": "test-2",
            "problem_statement": "Problem 2",
            "change_summary": "",
            "patch": "a" * 300,  # 300 chars
            "test_patch": "",
            "resolved": False,
        },
        {
            "instance_id": "test-3",
            "problem_statement": "Problem 3",
            "change_summary": "",
            "patch": "medium",  # 6 chars
            "test_patch": "",
            "resolved": False,
        },
    ]
    
    mock_ds = create_mock_dataset_dict(records)
    
    def mock_load_dataset(name):
        return mock_ds
    
    monkeypatch.setattr("swe_scaffold.dataset.load_dataset", mock_load_dataset)
    
    # Load with min_patch_chars=200
    result = load_conversation_dataset(
        "mock-dataset",
        min_patch_chars=200,
        include_empty=False,
    )
    
    assert isinstance(result, ProjectionResult)
    # Only test-2 should pass (300 chars > 200)
    assert len(result.split.dev) + len(result.split.test) == 1
    
    # Check skipped records
    short_skips = [s for s in result.skipped if s["reason"] == "short_patch"]
    assert len(short_skips) == 2
    assert set(s["instance_id"] for s in short_skips) == {"test-1", "test-3"}


def test_patch_length_filter_only_applies_to_patches(monkeypatch):
    """Test that min_patch_chars only applies to patch/test_patch sources, not change_summary."""
    records = [
        {
            "instance_id": "test-1",
            "problem_statement": "Problem 1",
            "change_summary": "short",  # 5 chars, but should not be filtered
            "patch": "",
            "test_patch": "",
            "resolved": False,
        },
        {
            "instance_id": "test-2",
            "problem_statement": "Problem 2",
            "change_summary": "",
            "patch": "short",  # 5 chars, should be filtered
            "test_patch": "",
            "resolved": False,
        },
    ]
    
    mock_ds = create_mock_dataset_dict(records)
    
    def mock_load_dataset(name):
        return mock_ds
    
    monkeypatch.setattr("swe_scaffold.dataset.load_dataset", mock_load_dataset)
    
    result = load_conversation_dataset(
        "mock-dataset",
        min_patch_chars=200,
        include_empty=False,
    )
    
    # test-1 should pass (change_summary not subject to length filter)
    # test-2 should be skipped (patch too short)
    assert len(result.split.dev) + len(result.split.test) == 1
    assert len(result.skipped) == 1
    assert result.skipped[0]["instance_id"] == "test-2"


def test_include_empty_flag_retains_empty_responses(monkeypatch):
    """Test that include_empty=True retains examples with empty responses."""
    records = [
        {
            "instance_id": "test-1",
            "problem_statement": "Problem 1",
            "change_summary": "",
            "patch": "",
            "test_patch": "",
            "resolved": False,
        },
        {
            "instance_id": "test-2",
            "problem_statement": "Problem 2",
            "change_summary": "Valid response",
            "patch": "",
            "test_patch": "",
            "resolved": False,
        },
    ]
    
    mock_ds = create_mock_dataset_dict(records)
    
    def mock_load_dataset(name):
        return mock_ds
    
    monkeypatch.setattr("swe_scaffold.dataset.load_dataset", mock_load_dataset)
    
    # With include_empty=True
    result = load_conversation_dataset(
        "mock-dataset",
        include_empty=True,
    )
    
    # Both examples should be included
    total = len(result.split.dev) + len(result.split.test)
    assert total == 2
    assert len(result.skipped) == 0
    
    # Check that empty response is indeed empty
    dev_responses = result.split.dev["response"]
    test_responses = result.split.test["response"]
    all_responses = list(dev_responses) + list(test_responses)
    assert "" in all_responses
    assert "Valid response" in all_responses


def test_include_empty_false_filters_empty_responses(monkeypatch):
    """Test that include_empty=False (default) filters out empty responses."""
    records = [
        {
            "instance_id": "test-1",
            "problem_statement": "Problem 1",
            "change_summary": "",
            "patch": "",
            "test_patch": "",
            "resolved": False,
        },
        {
            "instance_id": "test-2",
            "problem_statement": "Problem 2",
            "change_summary": "Valid response",
            "patch": "",
            "test_patch": "",
            "resolved": False,
        },
    ]
    
    mock_ds = create_mock_dataset_dict(records)
    
    def mock_load_dataset(name):
        return mock_ds
    
    monkeypatch.setattr("swe_scaffold.dataset.load_dataset", mock_load_dataset)
    
    # With include_empty=False (default)
    result = load_conversation_dataset(
        "mock-dataset",
        include_empty=False,
    )
    
    # Only test-2 should be included
    total = len(result.split.dev) + len(result.split.test)
    assert total == 1
    assert len(result.skipped) == 1
    assert result.skipped[0]["reason"] == "empty_response"
    assert result.skipped[0]["instance_id"] == "test-1"


def test_response_fallback_order(monkeypatch):
    """Test that response fallback follows change_summary → patch → test_patch."""
    records = [
        {
            "instance_id": "test-1",
            "problem_statement": "Problem 1",
            "change_summary": "summary",
            "patch": "patch",
            "test_patch": "test_patch",
            "resolved": False,
        },
        {
            "instance_id": "test-2",
            "problem_statement": "Problem 2",
            "change_summary": "",
            "patch": "patch",
            "test_patch": "test_patch",
            "resolved": False,
        },
        {
            "instance_id": "test-3",
            "problem_statement": "Problem 3",
            "change_summary": "",
            "patch": "",
            "test_patch": "test_patch",
            "resolved": False,
        },
    ]
    
    mock_ds = create_mock_dataset_dict(records)
    
    def mock_load_dataset(name):
        return mock_ds
    
    monkeypatch.setattr("swe_scaffold.dataset.load_dataset", mock_load_dataset)
    
    result = load_conversation_dataset("mock-dataset", include_empty=False)
    
    # Collect all responses
    dev_responses = list(result.split.dev["response"])
    test_responses = list(result.split.test["response"])
    all_responses = dev_responses + test_responses
    
    # test-1 should use change_summary
    assert "summary" in all_responses
    # test-2 should use patch (not test_patch)
    assert "patch" in all_responses
    # test-3 should use test_patch
    assert "test_patch" in all_responses


def test_skipped_report_structure(monkeypatch):
    """Test that skipped report has the correct structure."""
    records = [
        {
            "instance_id": "test-empty",
            "problem_statement": "Problem 1",
            "change_summary": "",
            "patch": "",
            "test_patch": "",
            "resolved": False,
        },
        {
            "instance_id": "test-short",
            "problem_statement": "Problem 2",
            "change_summary": "",
            "patch": "short",
            "test_patch": "",
            "resolved": False,
        },
    ]
    
    mock_ds = create_mock_dataset_dict(records)
    
    def mock_load_dataset(name):
        return mock_ds
    
    monkeypatch.setattr("swe_scaffold.dataset.load_dataset", mock_load_dataset)
    
    result = load_conversation_dataset(
        "mock-dataset",
        min_patch_chars=100,
        include_empty=False,
    )
    
    assert len(result.skipped) == 2
    
    # Check structure of skipped records
    for skip_record in result.skipped:
        assert "instance_id" in skip_record
        assert "reason" in skip_record
        assert skip_record["reason"] in ["empty_response", "short_patch"]
        assert "source_fields" in skip_record
        assert "has_change_summary" in skip_record["source_fields"]
        assert "has_patch" in skip_record["source_fields"]
        assert "has_test_patch" in skip_record["source_fields"]
        assert "patch_len" in skip_record
        assert isinstance(skip_record["patch_len"], int)


def test_skipped_report_counts_by_reason(monkeypatch):
    """Test that we can count skipped examples by reason."""
    records = [
        {"instance_id": f"empty-{i}", "problem_statement": f"P{i}", 
         "change_summary": "", "patch": "", "test_patch": "", "resolved": False}
        for i in range(5)
    ] + [
        {"instance_id": f"short-{i}", "problem_statement": f"P{i}", 
         "change_summary": "", "patch": "x", "test_patch": "", "resolved": False}
        for i in range(3)
    ]
    
    mock_ds = create_mock_dataset_dict(records)
    
    def mock_load_dataset(name):
        return mock_ds
    
    monkeypatch.setattr("swe_scaffold.dataset.load_dataset", mock_load_dataset)
    
    result = load_conversation_dataset(
        "mock-dataset",
        min_patch_chars=10,
        include_empty=False,
    )
    
    empty_count = sum(1 for s in result.skipped if s["reason"] == "empty_response")
    short_count = sum(1 for s in result.skipped if s["reason"] == "short_patch")
    
    assert empty_count == 5
    assert short_count == 3
    assert len(result.skipped) == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

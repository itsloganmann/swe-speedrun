"""Integration test for build script with mock data."""
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from datasets import Dataset, DatasetDict

from swe_scaffold.dataset import load_conversation_dataset, DatasetSplit


def test_build_script_jsonl_format():
    """Test that the build script creates proper JSONL with dev/test splits."""
    # Create a mock dataset to test the flow
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "mock.jsonl"
        
        # Write mock JSONL data
        mock_records = []
        for i in range(10):
            record = {
                "prompt": f"Problem statement {i}",
                "response": f"Solution {i}",
                "label": "success" if i % 2 == 0 else "failure"
            }
            mock_records.append(record)
        
        with jsonl_path.open("w") as f:
            for record in mock_records:
                f.write(json.dumps(record) + "\n")
        
        # Load and verify splits
        from swe_scaffold.dataset import SpeedrunDatasetBuilder
        builder = SpeedrunDatasetBuilder(Path(tmpdir) / "cache", seed=42)
        split = builder.from_jsonl(jsonl_path)
        
        # Verify split structure
        assert isinstance(split, DatasetSplit)
        assert len(split.dev) + len(split.test) == 10
        
        # Verify dev is larger than test (90/10 split)
        assert len(split.dev) > len(split.test)
        
        # Verify all required fields are present
        assert "prompt" in split.dev.column_names
        assert "response" in split.dev.column_names
        assert "label" in split.dev.column_names


def test_load_conversation_dataset_with_no_limit():
    """Test that when limit is None, the full dataset is loaded without truncation."""
    # Create a mock dataset with a known size
    mock_train_data = []
    for i in range(100):
        mock_train_data.append({
            "problem_statement": f"Problem {i}",
            "change_summary": f"Solution {i}",
            "resolved": i % 2 == 0
        })
    
    mock_train_dataset = Dataset.from_dict({
        "problem_statement": [d["problem_statement"] for d in mock_train_data],
        "change_summary": [d["change_summary"] for d in mock_train_data],
        "resolved": [d["resolved"] for d in mock_train_data]
    })
    
    # Mock load_dataset to return our test data
    with patch('swe_scaffold.dataset.load_dataset') as mock_load:
        mock_load.return_value = DatasetDict({"train": mock_train_dataset})
        
        # Test with limit=None (default)
        split = load_conversation_dataset("mock-dataset", limit=None)
        
        # With 90/10 split, we expect ~90 dev and ~10 test
        # Total should match the original dataset size
        total_loaded = len(split.dev) + len(split.test)
        assert total_loaded == 100, f"Expected 100 total examples, got {total_loaded}"
        
        # Verify dev is approximately 90% and test is approximately 10%
        assert len(split.dev) == 90, f"Expected 90 dev examples, got {len(split.dev)}"
        assert len(split.test) == 10, f"Expected 10 test examples, got {len(split.test)}"


def test_load_conversation_dataset_with_zero_limit():
    """Test that when limit is 0, the full dataset is loaded."""
    # Create a mock dataset with a known size
    mock_train_data = []
    for i in range(50):
        mock_train_data.append({
            "problem_statement": f"Problem {i}",
            "change_summary": f"Solution {i}",
            "resolved": True
        })
    
    mock_train_dataset = Dataset.from_dict({
        "problem_statement": [d["problem_statement"] for d in mock_train_data],
        "change_summary": [d["change_summary"] for d in mock_train_data],
        "resolved": [d["resolved"] for d in mock_train_data]
    })
    
    # Mock load_dataset to return our test data
    with patch('swe_scaffold.dataset.load_dataset') as mock_load:
        mock_load.return_value = DatasetDict({"train": mock_train_dataset})
        
        # Test with limit=0 (should load full dataset)
        split = load_conversation_dataset("mock-dataset", limit=0)
        
        # Total should match the original dataset size
        total_loaded = len(split.dev) + len(split.test)
        assert total_loaded == 50, f"Expected 50 total examples, got {total_loaded}"


def test_load_conversation_dataset_with_positive_limit():
    """Test that when limit is positive, it caps the dev portion accordingly."""
    # Create a mock dataset with a known size
    mock_train_data = []
    for i in range(100):
        mock_train_data.append({
            "problem_statement": f"Problem {i}",
            "change_summary": f"Solution {i}",
            "resolved": True
        })
    
    mock_train_dataset = Dataset.from_dict({
        "problem_statement": [d["problem_statement"] for d in mock_train_data],
        "change_summary": [d["change_summary"] for d in mock_train_data],
        "resolved": [d["resolved"] for d in mock_train_data]
    })
    
    # Mock load_dataset to return our test data
    with patch('swe_scaffold.dataset.load_dataset') as mock_load:
        mock_load.return_value = DatasetDict({"train": mock_train_dataset})
        
        # Test with limit=20 (should cap dev to 20 and test to 2)
        split = load_conversation_dataset("mock-dataset", limit=20)
        
        # Dev should be capped at the limit
        assert len(split.dev) == 20, f"Expected 20 dev examples, got {len(split.dev)}"
        # Test should be limit // 10 = 2
        assert len(split.test) == 2, f"Expected 2 test examples, got {len(split.test)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

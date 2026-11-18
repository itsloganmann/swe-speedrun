"""Smoke tests for dataset split naming and loading."""
import json
import tempfile
from pathlib import Path

import pytest
from datasets import Dataset

from swe_scaffold.dataset import DatasetSplit, SpeedrunDatasetBuilder


def test_dataset_split_has_dev_and_test():
    """Verify DatasetSplit dataclass has dev and test attributes."""
    dev_data = Dataset.from_dict({"prompt": ["test"], "response": ["answer"], "label": ["success"]})
    test_data = Dataset.from_dict({"prompt": ["test2"], "response": ["answer2"], "label": ["failure"]})
    
    split = DatasetSplit(dev=dev_data, test=test_data)
    
    assert hasattr(split, "dev")
    assert hasattr(split, "test")
    assert len(split.dev) == 1
    assert len(split.test) == 1


def test_builder_from_jsonl_creates_dev_test_splits():
    """Verify SpeedrunDatasetBuilder.from_jsonl creates dev/test splits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "cache"
        jsonl_path = Path(tmpdir) / "test.jsonl"
        
        # Create a minimal JSONL file
        records = [
            {"prompt": f"problem {i}", "response": f"solution {i}", "label": "success"}
            for i in range(10)
        ]
        with jsonl_path.open("w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        
        builder = SpeedrunDatasetBuilder(cache_path, seed=42)
        split = builder.from_jsonl(jsonl_path)
        
        assert isinstance(split, DatasetSplit)
        assert hasattr(split, "dev")
        assert hasattr(split, "test")
        assert len(split.dev) > 0
        assert len(split.test) > 0
        assert len(split.dev) + len(split.test) == 10


def test_builder_hydrate_cache_uses_dev_test():
    """Verify hydrate_cache saves dev and test splits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "cache"
        
        dev_data = Dataset.from_dict({
            "prompt": ["dev1", "dev2"],
            "response": ["ans1", "ans2"],
            "label": ["success", "failure"]
        })
        test_data = Dataset.from_dict({
            "prompt": ["test1"],
            "response": ["ans1"],
            "label": ["success"]
        })
        
        split = DatasetSplit(dev=dev_data, test=test_data)
        builder = SpeedrunDatasetBuilder(cache_path)
        builder.hydrate_cache(split)
        
        # Load and verify
        from datasets import DatasetDict
        loaded = DatasetDict.load_from_disk(str(cache_path))
        
        assert "dev" in loaded
        assert "test" in loaded
        assert "train" not in loaded
        assert "validation" not in loaded


def test_load_conversation_dataset_train_only_mode():
    """Verify load_conversation_dataset works with train_only mode."""
    # Create mock dataset
    mock_data = {
        "problem_statement": [f"Problem {i}" for i in range(20)],
        "patch": [f"diff patch {i}" if i % 2 == 0 else "" for i in range(20)],
        "test_patch": [f"test patch {i}" for i in range(20)],
    }
    mock_dataset = Dataset.from_dict(mock_data)
    
    # Test with train_only and holdout
    from swe_scaffold.dataset import load_conversation_dataset
    from unittest.mock import patch
    
    with patch("swe_scaffold.dataset.load_dataset") as mock_load:
        mock_load.return_value = mock_dataset
        
        split = load_conversation_dataset(
            "mock/dataset",
            train_only=True,
            holdout_fraction=0.1,
            limit=20
        )
        
        assert isinstance(split, DatasetSplit)
        assert len(split.dev) > 0
        assert len(split.test) > 0
        assert len(split.dev) + len(split.test) == 20


def test_load_conversation_dataset_no_holdout():
    """Verify load_conversation_dataset works with no holdout (test split empty)."""
    # Create mock dataset
    mock_data = {
        "problem_statement": [f"Problem {i}" for i in range(10)],
        "patch": [f"patch {i}" for i in range(10)],
    }
    mock_dataset = Dataset.from_dict(mock_data)
    
    from swe_scaffold.dataset import load_conversation_dataset
    from unittest.mock import patch
    
    with patch("swe_scaffold.dataset.load_dataset") as mock_load:
        mock_load.return_value = mock_dataset
        
        split = load_conversation_dataset(
            "mock/dataset",
            train_only=True,
            holdout_fraction=0.0,
            limit=10
        )
        
        assert isinstance(split, DatasetSplit)
        assert len(split.dev) == 10
        assert len(split.test) == 0


def test_response_field_fallback():
    """Verify response field fallback: change_summary -> patch -> test_patch -> empty."""
    # Create mock dataset with different field combinations
    mock_data = {
        "problem_statement": ["P1", "P2", "P3", "P4"],
        "change_summary": ["summary1", "", "", ""],
        "patch": ["", "patch2", "", ""],
        "test_patch": ["", "", "test3", ""],
    }
    mock_dataset = Dataset.from_dict(mock_data)
    
    from swe_scaffold.dataset import load_conversation_dataset
    from unittest.mock import patch
    
    with patch("swe_scaffold.dataset.load_dataset") as mock_load:
        mock_load.return_value = mock_dataset
        
        split = load_conversation_dataset(
            "mock/dataset",
            train_only=True,
            holdout_fraction=0.0,
            limit=4
        )
        
        responses = split.dev["response"]
        assert responses[0] == "summary1"  # change_summary preferred
        assert responses[1] == "patch2"    # patch fallback
        assert responses[2] == "test3"     # test_patch fallback
        assert responses[3] == ""          # empty fallback


def test_patch_based_success_labeling():
    """Verify success labeling based on patch when resolved field absent."""
    # Create mock dataset without resolved field
    mock_data = {
        "problem_statement": ["P1", "P2"],
        "patch": ["diff content", ""],  # First has patch, second doesn't
    }
    mock_dataset = Dataset.from_dict(mock_data)
    
    from swe_scaffold.dataset import load_conversation_dataset
    from unittest.mock import patch
    
    with patch("swe_scaffold.dataset.load_dataset") as mock_load:
        mock_load.return_value = mock_dataset
        
        split = load_conversation_dataset(
            "mock/dataset",
            train_only=True,
            holdout_fraction=0.0,
            limit=2
        )
        
        labels = split.dev["label"]
        assert labels[0] == "success"  # Has patch
        assert labels[1] == "failure"  # No patch


def test_resolved_field_takes_precedence():
    """Verify resolved field takes precedence over patch for labeling."""
    # Create mock dataset with both resolved and patch fields
    mock_data = {
        "problem_statement": ["P1", "P2"],
        "patch": ["diff1", "diff2"],
        "resolved": [True, False],
    }
    mock_dataset = Dataset.from_dict(mock_data)
    
    from swe_scaffold.dataset import load_conversation_dataset
    from unittest.mock import patch
    
    with patch("swe_scaffold.dataset.load_dataset") as mock_load:
        mock_load.return_value = mock_dataset
        
        split = load_conversation_dataset(
            "mock/dataset",
            train_only=True,
            holdout_fraction=0.0,
            limit=2
        )
        
        labels = split.dev["label"]
        assert labels[0] == "success"  # resolved=True
        assert labels[1] == "failure"  # resolved=False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

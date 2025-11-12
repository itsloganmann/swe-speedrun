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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

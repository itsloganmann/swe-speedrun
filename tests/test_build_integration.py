"""Integration test for build script with mock data."""
import json
import tempfile
from pathlib import Path

import pytest
from datasets import Dataset

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

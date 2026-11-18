"""Integration test for build script with mock data."""
import json
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from datasets import Dataset

from scripts import build_speedrun_dataset
from swe_scaffold.config import DatasetConfig, SpeedrunConfig
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


@pytest.fixture
def train_data():
    return Dataset.from_dict({
        "problem_statement": [f"Problem {i}" for i in range(600)],
        "change_summary": [f"Solution {i}" for i in range(600)],
        "resolved": [True] * 600,
    })


@pytest.mark.parametrize("limit", [None, 0, -1])
@pytest.mark.parametrize("holdout_fraction,expected", [(None, (540, 60)), (0.0, (600, 0))])
def test_full_dataset_limits(train_data, limit, holdout_fraction, expected):
    with patch("swe_scaffold.dataset.load_dataset", return_value=train_data) as load:
        split = load_conversation_dataset(
            "mock-dataset", limit=limit, holdout_fraction=holdout_fraction
        )
    load.assert_called_once_with("mock-dataset", split="train")
    assert (len(split.dev), len(split.test)) == expected
    assert set(split.dev["prompt"]) | set(split.test["prompt"]) == set(train_data["problem_statement"])


@pytest.mark.parametrize("holdout_fraction,expected", [(None, (20, 2)), (0.0, (20, 0))])
def test_positive_limit_preserves_caps(train_data, holdout_fraction, expected):
    with patch("swe_scaffold.dataset.load_dataset", return_value=train_data):
        split = load_conversation_dataset(
            "mock-dataset", limit=20, holdout_fraction=holdout_fraction
        )
    assert (len(split.dev), len(split.test)) == expected


def test_config_defaults_load_all_examples():
    assert DatasetConfig().max_examples is None
    config_path = Path(__file__).parents[1] / "configs" / "speedrun.json"
    config = SpeedrunConfig.from_dict(json.loads(config_path.read_text()))
    assert config.dataset.max_examples is None


@pytest.mark.parametrize("arguments,expected", [([], None), (["--limit", "0"], 0), (["--limit", "-1"], -1), (["--limit", "20"], 20)])
def test_cli_limit_arguments(monkeypatch, arguments, expected):
    monkeypatch.setattr(sys, "argv", ["build_speedrun_dataset.py", *arguments])
    assert build_speedrun_dataset.parse_args().limit == expected


@pytest.mark.parametrize("arguments,expected", [([], (540, 60)), (["--limit", "20"], (20, 2)), (["--no-holdout"], (600, 0))])
def test_build_script_keeps_all_examples_by_default(tmp_path, monkeypatch, train_data, arguments, expected):
    output = tmp_path / "data.jsonl"
    monkeypatch.setattr(sys, "argv", ["build_speedrun_dataset.py", "--output", str(output), "--emit-test", *arguments])
    with patch("swe_scaffold.dataset.load_dataset", return_value=train_data):
        build_speedrun_dataset.main()
    dev = [json.loads(line) for line in output.read_text().splitlines()]
    test = [json.loads(line) for line in output.with_suffix(".test.jsonl").read_text().splitlines()]
    assert (len(dev), len(test)) == expected
    assert all(record["response"] for record in dev + test)

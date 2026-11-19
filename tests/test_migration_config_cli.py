import copy
import json
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from datasets import Dataset, DatasetDict

from scripts import build_speedrun_dataset
from swe_scaffold.config import DatasetConfig, SpeedrunConfig
from swe_scaffold.dataset import (
    DatasetSplit,
    SpeedrunDatasetBuilder,
    load_conversation_dataset,
)


@pytest.fixture
def training_source() -> Dataset:
    return Dataset.from_dict(
        {
            "instance_id": [str(i) for i in range(1201)],
            "problem_statement": [f"problem {i}" for i in range(1201)],
            "change_summary": [f"answer {i}" for i in range(1201)],
        }
    )


def test_legacy_config_maps_holdout_without_mutating_input() -> None:
    payload = {"dataset": {"dev_split": 0.8}, "training": {"lora": {"r": 4}}}
    original = copy.deepcopy(payload)
    config = SpeedrunConfig.from_dict(payload)
    assert config.dataset.holdout_fraction == pytest.approx(0.2)
    assert config.training.lora.r == 4
    assert payload == original


@pytest.mark.parametrize("route", ["constructor", "json", "loader"])
def test_legacy_split_settings_warn_about_new_default(
    route: str, training_source: Dataset
) -> None:
    with (
        patch("swe_scaffold.dataset.load_dataset", return_value=training_source),
        pytest.warns(FutureWarning, match="no holdout"),
    ):
        if route == "constructor":
            config = DatasetConfig(dev_split=0.8)
            assert config.holdout_fraction == pytest.approx(0.2)
        elif route == "json":
            config = SpeedrunConfig.from_dict({"dataset": {"dev_split": 0.8}}).dataset
            assert config.holdout_fraction == pytest.approx(0.2)
        else:
            result = load_conversation_dataset("example", split=0.8)
            assert len(result.test) > 0
            assert result.total_kept == len(training_source)


def test_current_split_settings_do_not_emit_migration_warning(
    training_source: Dataset,
) -> None:
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always", FutureWarning)
        assert DatasetConfig().holdout_fraction is None
        config = SpeedrunConfig.from_dict({"dataset": {"holdout_fraction": 0.2}})
        with patch("swe_scaffold.dataset.load_dataset", return_value=training_source):
            result = load_conversation_dataset(
                "example", holdout_fraction=config.dataset.holdout_fraction
            )
        assert len(result.test) > 0
    assert not any(issubclass(item.category, FutureWarning) for item in emitted)


@pytest.mark.parametrize("dev_rows,test_rows", [(1, 0), (0, 1), (0, 0)])
def test_empty_split_cache_roundtrip(
    tmp_path: Path, dev_rows: int, test_rows: int
) -> None:
    row = {"prompt": "problem", "response": "answer\n", "label": "SUCCESS"}
    source = Dataset.from_list([row])
    split = DatasetSplit(
        dev=source.select(range(dev_rows)), test=source.select(range(test_rows))
    )
    builder = SpeedrunDatasetBuilder(tmp_path / "cache")
    builder.hydrate_cache(split)
    cached = builder.load_or_build(lambda: pytest.fail("cache should be reused"))
    reopened = builder.from_jsonl(builder.cache_path)
    for result in (cached, reopened):
        assert result.dev.to_dict() == split.dev.to_dict()
        assert result.test.to_dict() == split.test.to_dict()


def test_explicit_new_holdout_setting_overrides_legacy_split() -> None:
    config = SpeedrunConfig.from_dict(
        {"dataset": {"dev_split": 0.8, "holdout_fraction": None}}
    )
    assert config.dataset.holdout_fraction is None
    assert config.dataset.dev_split == 1.0
    assert DatasetConfig().max_examples is None


@pytest.mark.parametrize("positional", [False, True])
def test_legacy_dataset_constructor_maps_split(positional: bool) -> None:
    config = (
        DatasetConfig("example", Path("cache"), 0.8)
        if positional
        else DatasetConfig(dev_split=0.8)
    )
    assert config.train_only is True
    assert config.dev_split == pytest.approx(0.8)
    assert config.holdout_fraction == pytest.approx(0.2)
    config.dev_split = 0.7
    assert config.holdout_fraction == pytest.approx(0.3)


def test_test_only_dataset_is_never_projected_into_training(
    training_source: Dataset,
) -> None:
    source = DatasetDict({"test": training_source})
    with patch("swe_scaffold.dataset.load_dataset", return_value=source):
        with pytest.raises(ValueError, match="train or dev"):
            load_conversation_dataset("example", train_only=False)


@pytest.mark.parametrize("limit", [None, 0, -2])
def test_unlimited_loader_defaults_keep_full_training_source(
    training_source: Dataset, limit: int | None
) -> None:
    with patch(
        "swe_scaffold.dataset.load_dataset", return_value=training_source
    ) as load:
        result = load_conversation_dataset("example", limit=limit)
    load.assert_called_once_with("example", split="train")
    assert (len(result.dev), len(result.test)) == (1201, 0)
    assert (result.total_kept, result.total_skipped) == (1201, 0)


def test_explicit_holdout_retains_positive_dev_and_test_caps(
    training_source: Dataset,
) -> None:
    with patch("swe_scaffold.dataset.load_dataset", return_value=training_source):
        result = load_conversation_dataset("example", limit=20, holdout_fraction=0.2)
    assert (len(result.dev), len(result.test)) == (20, 2)


def test_legacy_positional_split_and_limit_still_work(training_source: Dataset) -> None:
    with patch("swe_scaffold.dataset.load_dataset", return_value=training_source):
        result = load_conversation_dataset("example", 0.8, 5)
    assert (len(result.dev), len(result.test)) == (5, 1)


@pytest.mark.parametrize("fraction", [-0.1, 1.0, float("nan"), float("inf")])
def test_invalid_holdout_warns_and_uses_no_holdout(
    training_source: Dataset, fraction: float
) -> None:
    with patch(
        "swe_scaffold.dataset.load_dataset", return_value=training_source
    ) as load:
        with pytest.warns(UserWarning, match="using no holdout"):
            result = load_conversation_dataset("example", holdout_fraction=fraction)
    load.assert_called_once_with("example", split="train")
    assert (len(result.dev), len(result.test)) == (len(training_source), 0)


@pytest.mark.parametrize("extension", ["json", "yaml"])
def test_cli_uses_dataset_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extension: str
) -> None:
    source = Dataset.from_list(
        [
            {
                "instance_id": "summary",
                "problem_statement": "p1",
                "change_summary": "answer",
                "patch": "",
            },
            {
                "instance_id": "empty",
                "problem_statement": "p2",
                "change_summary": "",
                "patch": "",
            },
            {
                "instance_id": "short",
                "problem_statement": "p3",
                "change_summary": "",
                "patch": "x",
            },
            {
                "instance_id": "patch",
                "problem_statement": "p4",
                "change_summary": "",
                "patch": "long patch",
            },
            {
                "instance_id": "capped",
                "problem_statement": "p5",
                "change_summary": "omitted",
                "patch": "",
            },
        ]
    )
    config = {
        "dataset": {
            "source_dataset": "config-source",
            "max_examples": 4,
            "holdout_fraction": None,
            "include_empty": True,
            "min_patch_chars": 5,
        }
    }
    config_path, output = tmp_path / f"config.{extension}", tmp_path / "data.jsonl"
    config_path.write_text(
        json.dumps(config) if extension == "json" else yaml.safe_dump(config)
    )
    monkeypatch.setattr(
        "sys.argv", ["build", "--config", str(config_path), "--output", str(output)]
    )
    with patch("swe_scaffold.dataset.load_dataset", return_value=source) as load:
        build_speedrun_dataset.main()
    load.assert_called_once_with("config-source", split="train")
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["response"] for row in rows] == ["answer", "", "long patch"]
    skipped = [
        json.loads(line)
        for line in output.with_suffix(".skipped.jsonl").read_text().splitlines()
    ]
    assert [(row["instance_id"], row["reason"]) for row in skipped] == [
        ("short", "short_patch")
    ]


def test_cli_flags_override_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, training_source: Dataset
) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "dataset": {
                    "source_dataset": "config-source",
                    "max_examples": 20,
                    "holdout_fraction": 0.25,
                }
            }
        )
    )
    output = tmp_path / "data.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build",
            "--config",
            str(config),
            "--output",
            str(output),
            "--dataset",
            "override-source",
            "--limit",
            "1",
            "--no-holdout",
            "--emit-test",
        ],
    )
    with patch(
        "swe_scaffold.dataset.load_dataset", return_value=training_source
    ) as load:
        build_speedrun_dataset.main()
    load.assert_called_once_with("override-source", split="train")
    assert len(output.read_text().splitlines()) == 1
    assert output.with_suffix(".test.jsonl").read_text() == ""


def test_cli_without_limit_keeps_all_rows_and_no_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, training_source: Dataset
) -> None:
    output = tmp_path / "data.jsonl"
    monkeypatch.setattr("sys.argv", ["build", "--output", str(output), "--emit-test"])
    with patch("swe_scaffold.dataset.load_dataset", return_value=training_source):
        build_speedrun_dataset.main()
    assert len(output.read_text().splitlines()) == 1201
    assert output.with_suffix(".test.jsonl").read_text() == ""


def test_default_audit_is_next_to_custom_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = Dataset.from_dict(
        {"problem_statement": ["p"] * 10, "change_summary": ["answer"] * 9 + [""]}
    )
    output = tmp_path / "nested" / "custom.jsonl"
    monkeypatch.setattr("sys.argv", ["build", "--output", str(output)])
    with patch("swe_scaffold.dataset.load_dataset", return_value=source):
        build_speedrun_dataset.main()
    report = output.with_suffix(".skipped.jsonl")
    assert json.loads(report.read_text())["reason"] == "empty_response"

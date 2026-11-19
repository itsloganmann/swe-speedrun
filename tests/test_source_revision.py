import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from datasets import Dataset

from scripts import build_speedrun_dataset
from swe_scaffold.dataset import load_conversation_dataset


@pytest.fixture
def training_source() -> Dataset:
    return Dataset.from_dict(
        {"problem_statement": ["problem"], "patch": ["diff --git a/a b/a"]}
    )


@pytest.mark.parametrize(
    "dataset,revision,expected",
    [
        ("SWE-bench/SWE-bench", None, "ac1995e464b07cbd69702e8490a07f6d557b69b5"),
        ("SWE-bench/SWE-bench", "main", "main"),
        ("custom/source", None, None),
        ("custom/source", "v1", "v1"),
    ],
)
@pytest.mark.parametrize("train_only", [True, False])
def test_loader_selects_training_revision_without_retargeting_custom_sources(
    dataset: str,
    revision: str | None,
    expected: str | None,
    training_source: Dataset,
    train_only: bool,
) -> None:
    with patch(
        "swe_scaffold.dataset.load_dataset", return_value=training_source
    ) as load:
        result = load_conversation_dataset(
            dataset, train_only=train_only, revision=revision
        )
    expected_kwargs = {"split": "train"} if train_only else {}
    if expected is not None:
        expected_kwargs["revision"] = expected
    load.assert_called_once_with(dataset, **expected_kwargs)
    assert result.dev["response"] == training_source["patch"]
    assert len(result.test) == 0


@pytest.mark.parametrize("revision", ["", "  "])
def test_blank_revision_fails_before_any_download(revision: str) -> None:
    with (
        patch("swe_scaffold.dataset.load_dataset") as load,
        pytest.raises(ValueError, match="revision must be a nonempty string"),
    ):
        load_conversation_dataset("SWE-bench/SWE-bench", revision=revision)
    load.assert_not_called()


def test_cli_default_uses_the_pinned_training_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, training_source: Dataset
) -> None:
    output = tmp_path / "data.jsonl"
    monkeypatch.setattr("sys.argv", ["build", "--output", str(output)])
    with patch(
        "swe_scaffold.dataset.load_dataset", return_value=training_source
    ) as load:
        build_speedrun_dataset.main()
    load.assert_called_once_with(
        "SWE-bench/SWE-bench",
        split="train",
        revision="ac1995e464b07cbd69702e8490a07f6d557b69b5",
    )
    assert json.loads(output.read_text())["response"] == training_source[0]["patch"]


@pytest.mark.parametrize("extension", ["json", "yaml"])
@pytest.mark.parametrize(
    "overrides,expected_source,expected_revision",
    [
        ([], "configured/source", "configured-revision"),
        (["--revision", "cli-revision"], "configured/source", "cli-revision"),
        (["--dataset", "other/source"], "other/source", None),
        (
            ["--dataset", "configured/source"],
            "configured/source",
            "configured-revision",
        ),
        (
            ["--dataset", "other/source", "--revision", "cli-revision"],
            "other/source",
            "cli-revision",
        ),
    ],
)
def test_cli_revision_precedence_keeps_revisions_with_their_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    training_source: Dataset,
    extension: str,
    overrides: list[str],
    expected_source: str,
    expected_revision: str | None,
) -> None:
    config = {
        "dataset": {
            "source_dataset": "configured/source",
            "source_revision": "configured-revision",
        }
    }
    config_path = tmp_path / f"config.{extension}"
    config_path.write_text(
        json.dumps(config) if extension == "json" else yaml.safe_dump(config)
    )
    output = tmp_path / "data.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        ["build", "--config", str(config_path), "--output", str(output), *overrides],
    )
    with patch(
        "swe_scaffold.dataset.load_dataset", return_value=training_source
    ) as load:
        build_speedrun_dataset.main()
    expected_kwargs = {"split": "train"}
    if expected_revision is not None:
        expected_kwargs["revision"] = expected_revision
    load.assert_called_once_with(expected_source, **expected_kwargs)
    assert json.loads(output.read_text())["response"] == training_source[0]["patch"]

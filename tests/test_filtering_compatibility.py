import json
from pathlib import Path
from unittest.mock import patch

import pytest
from datasets import Dataset, DatasetDict

from scripts import build_speedrun_dataset
from swe_scaffold.dataset import (
    DatasetSplit,
    SpeedrunDatasetBuilder,
    load_conversation_dataset,
)


@pytest.fixture
def source_data() -> Dataset:
    return Dataset.from_dict(
        {
            "instance_id": [str(i) for i in range(10)],
            "problem_statement": [f"problem {i}" for i in range(10)],
            "change_summary": [f"answer {i}" for i in range(10)],
            "resolved": [True] * 10,
        }
    )


def test_filtered_loader_preserves_dataset_split_contract(source_data: Dataset) -> None:
    with patch("swe_scaffold.dataset.load_dataset", return_value=source_data) as load:
        result = load_conversation_dataset("example", 0.9, 5)
    load.assert_called_once_with("example", split="train")
    assert isinstance(result, DatasetSplit)
    assert (len(result.dev), len(result.test)) == (5, 1)
    assert result.split.dev is result.dev
    assert result.skipped == []


def test_filtered_loader_can_build_and_reuse_cache(
    tmp_path: Path, source_data: Dataset
) -> None:
    builder = SpeedrunDatasetBuilder(tmp_path / "cache")
    with patch("swe_scaffold.dataset.load_dataset", return_value=source_data):
        fresh = builder.load_or_build(lambda: load_conversation_dataset("example"))
    cached = builder.load_or_build(lambda: pytest.fail("cache should be reused"))
    assert isinstance(fresh, DatasetSplit)
    assert isinstance(cached, DatasetSplit)
    assert fresh.dev.to_dict() == cached.dev.to_dict()
    assert fresh.test.to_dict() == cached.test.to_dict()


def test_legacy_explicit_splits_are_preserved(source_data: Dataset) -> None:
    sources = DatasetDict(
        {"dev": source_data.select([0, 1]), "test": source_data.select([2])}
    )
    with patch("swe_scaffold.dataset.load_dataset", return_value=sources) as load:
        result = load_conversation_dataset("example", train_only=False)
    load.assert_called_once_with("example")
    assert result.dev["prompt"] == ["problem 0", "problem 1"]
    assert result.test["prompt"] == ["problem 2"]


@pytest.mark.parametrize("response_field", ["change_summary", "patch", "test_patch"])
def test_response_payload_is_preserved(response_field: str) -> None:
    payload = "--- a/example.txt\n+++ b/example.txt\n@@ -1 +1 @@\n-old\n+new\n"
    source = Dataset.from_list(
        [{"problem_statement": "Update the file", response_field: payload}]
    )
    with patch("swe_scaffold.dataset.load_dataset", return_value=source):
        result = load_conversation_dataset("example", holdout_fraction=0.0)
    assert result.dev["response"] == [payload]


def test_patch_threshold_uses_stripped_length() -> None:
    payload = "  short\n"
    source = Dataset.from_list([{"problem_statement": "p", "patch": payload}])
    with patch("swe_scaffold.dataset.load_dataset", return_value=source):
        result = load_conversation_dataset(
            "example", holdout_fraction=0.0, min_patch_chars=6
        )
    assert len(result.dev) == 0
    assert result.skipped[0]["reason"] == "short_patch"
    assert result.skipped[0]["patch_len"] == 5


def test_cli_refreshes_existing_audit_after_zero_skip_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, report = tmp_path / "data.jsonl", tmp_path / "skipped.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build",
            "--output",
            str(output),
            "--skipped-report",
            str(report),
            "--no-holdout",
        ],
    )
    empty = Dataset.from_list(
        [{"instance_id": "same", "problem_statement": "p", "change_summary": ""}]
    )
    with patch("swe_scaffold.dataset.load_dataset", return_value=empty):
        build_speedrun_dataset.main()
    assert json.loads(report.read_text())["instance_id"] == "same"
    answered = Dataset.from_list(
        [{"instance_id": "same", "problem_statement": "p", "change_summary": "answer"}]
    )
    with patch("swe_scaffold.dataset.load_dataset", return_value=answered):
        build_speedrun_dataset.main()
    assert json.loads(output.read_text())["response"] == "answer"
    assert report.read_text() == ""


@pytest.mark.parametrize("alias", ["primary", "normalized", "test", "hardlink"])
def test_cli_rejects_audit_alias_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_data: Dataset, alias: str
) -> None:
    output = tmp_path / "data.jsonl"
    test_output = tmp_path / "data.test.jsonl"
    output.write_text("existing data\n")
    test_output.write_text("existing test data\n")
    if alias == "test":
        report = test_output
    elif alias == "normalized":
        report = tmp_path / "nested" / ".." / "data.jsonl"
    elif alias == "hardlink":
        report = tmp_path / "audit.jsonl"
        report.hardlink_to(output)
    else:
        report = output
    monkeypatch.setattr(
        "sys.argv",
        [
            "build",
            "--output",
            str(output),
            "--skipped-report",
            str(report),
            "--no-holdout",
            "--emit-test",
        ],
    )
    with patch("swe_scaffold.dataset.load_dataset", return_value=source_data) as load:
        with pytest.raises(SystemExit, match="2"):
            build_speedrun_dataset.main()
    load.assert_not_called()
    assert output.read_text() == "existing data\n"
    assert test_output.read_text() == "existing test data\n"


def test_cli_accounts_for_filters_and_writes_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
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
                "instance_id": "patch",
                "problem_statement": "p2",
                "change_summary": "",
                "patch": "long enough patch",
            },
            {
                "instance_id": "short",
                "problem_statement": "p3",
                "change_summary": "",
                "patch": "x",
            },
            {
                "instance_id": "empty",
                "problem_statement": "p4",
                "change_summary": "",
                "patch": "",
            },
        ]
    )
    output, report = tmp_path / "data.jsonl", tmp_path / "skipped.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build",
            "--output",
            str(output),
            "--skipped-report",
            str(report),
            "--min-patch-chars",
            "10",
            "--no-holdout",
            "--emit-test",
        ],
    )
    with patch("swe_scaffold.dataset.load_dataset", return_value=source):
        build_speedrun_dataset.main()
    records = [json.loads(line) for line in output.read_text().splitlines()]
    skipped = [json.loads(line) for line in report.read_text().splitlines()]
    assert [(row["response"], row["label"]) for row in records] == [
        ("answer", "failure"),
        ("long enough patch", "success"),
    ]
    assert {(row["instance_id"], row["reason"]) for row in skipped} == {
        ("short", "short_patch"),
        ("empty", "empty_response"),
    }
    assert len(records) + len(skipped) == len(source)
    assert output.with_suffix(".test.jsonl").read_text() == ""
    assert "skipped 2: empty=1, short=1" in capsys.readouterr().out


@pytest.mark.parametrize("force", [False, True])
def test_cli_only_creates_empty_audit_when_forced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_data: Dataset, force: bool
) -> None:
    output, report = tmp_path / "data.jsonl", tmp_path / "skipped.jsonl"
    args = [
        "build",
        "--output",
        str(output),
        "--skipped-report",
        str(report),
        "--no-holdout",
    ]
    if force:
        args.append("--force-skipped-report")
    monkeypatch.setattr("sys.argv", args)
    with patch("swe_scaffold.dataset.load_dataset", return_value=source_data):
        build_speedrun_dataset.main()
    assert report.exists() is force
    if force:
        assert report.read_text() == ""

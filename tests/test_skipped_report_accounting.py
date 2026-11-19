"""Tests for skipped report accounting and aggregation."""

import pytest
from datasets import Dataset, DatasetDict

from swe_scaffold.dataset import ProjectionResult, load_conversation_dataset


def test_skipped_report_tracks_multiple_reasons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that skipped report correctly tracks different skip reasons."""
    mock_data = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "instance_id": ["test-1", "test-2", "test-3", "test-4"],
                    "problem_statement": ["P1", "P2", "P3", "P4"],
                    "change_summary": ["", "", "Summary 3", ""],
                    "patch": ["", "short", "", "longer patch"],
                }
            )
        }
    )

    def mock_load_dataset(name: str, **kwargs: object) -> Dataset | DatasetDict:
        return mock_data["train"] if kwargs.get("split") == "train" else mock_data

    monkeypatch.setattr("swe_scaffold.dataset.load_dataset", mock_load_dataset)

    result = load_conversation_dataset(
        "test-dataset",
        include_empty=False,
        min_patch_chars=10,
        train_only=True,
    )

    # test-1: empty_response
    # test-2: short_patch
    # test-3: kept (has change_summary)
    # test-4: kept (patch is long enough)
    assert result.total_kept == 2
    assert result.total_skipped == 2

    reasons = [r["reason"] for r in result.skipped]
    assert "empty_response" in reasons
    assert "short_patch" in reasons


def test_audit_metrics_match_returned_split() -> None:
    """Audit totals describe the returned examples and skipped records."""
    from swe_scaffold.dataset import DatasetSplit

    dev_data = Dataset.from_dict(
        {"prompt": ["p1"], "response": ["r1"], "label": ["success"]}
    )
    test_data = Dataset.from_dict({"prompt": [], "response": [], "label": []})

    result = ProjectionResult(
        dev=dev_data,
        test=test_data,
        skipped=[{"instance_id": "test", "reason": "empty_response"}],
    )

    assert isinstance(result, DatasetSplit)
    assert result.total_kept == 1
    assert result.total_skipped == 1
    assert len(result.skipped) == 1


def test_no_filters_produces_empty_skipped_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that with no filters, skipped list is empty."""
    mock_data = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "instance_id": ["test-1"],
                    "problem_statement": ["Problem 1"],
                    "change_summary": ["Summary 1"],
                    "patch": [""],
                }
            )
        }
    )

    def mock_load_dataset(name: str, **kwargs: object) -> Dataset | DatasetDict:
        return mock_data["train"] if kwargs.get("split") == "train" else mock_data

    monkeypatch.setattr("swe_scaffold.dataset.load_dataset", mock_load_dataset)

    result = load_conversation_dataset(
        "test-dataset",
        include_empty=True,
        min_patch_chars=None,
        train_only=True,
    )

    assert result.total_skipped == 0
    assert len(result.skipped) == 0


def test_combined_filters_accounting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that multiple filters work together and accounting is correct."""
    mock_data = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "instance_id": ["t1", "t2", "t3", "t4", "t5"],
                    "problem_statement": ["P1", "P2", "P3", "P4", "P5"],
                    "change_summary": ["", "", "OK", "", "OK"],
                    "patch": ["", "xy", "", "longer patch here", ""],
                }
            )
        }
    )

    def mock_load_dataset(name: str, **kwargs: object) -> Dataset | DatasetDict:
        return mock_data["train"] if kwargs.get("split") == "train" else mock_data

    monkeypatch.setattr("swe_scaffold.dataset.load_dataset", mock_load_dataset)

    result = load_conversation_dataset(
        "test-dataset",
        include_empty=False,
        min_patch_chars=10,
        train_only=True,
    )

    # t1: empty_response (no change_summary, no patch)
    # t2: short_patch (patch='xy')
    # t3: kept (change_summary='OK')
    # t4: kept (patch long enough)
    # t5: kept (change_summary='OK')

    assert result.total_kept == 3
    assert result.total_skipped == 2
    assert result.total_kept + result.total_skipped == 5


def test_skipped_records_have_instance_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that all skipped records include instance_id."""
    mock_data = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "instance_id": ["id-1", "id-2"],
                    "problem_statement": ["P1", "P2"],
                    "change_summary": ["", ""],
                    "patch": ["x", ""],
                }
            )
        }
    )

    def mock_load_dataset(name: str, **kwargs: object) -> Dataset | DatasetDict:
        return mock_data["train"] if kwargs.get("split") == "train" else mock_data

    monkeypatch.setattr("swe_scaffold.dataset.load_dataset", mock_load_dataset)

    result = load_conversation_dataset(
        "test-dataset",
        include_empty=False,
        min_patch_chars=5,
        train_only=True,
    )

    assert result.total_skipped == 2
    for record in result.skipped:
        assert "instance_id" in record
        assert record["instance_id"] in ["id-1", "id-2"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

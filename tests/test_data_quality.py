"""Tests for the data quality gate."""

import pandas as pd

from src.data.quality import check_data_quality


def test_quality_gate_passes_on_cleaned_data(cleaned_data):
    """
    The cleaned dataset should pass the quality gate with no failures.
    Warnings are okay (e.g., the class imbalance warning we expect).
    """
    result = check_data_quality(cleaned_data)

    assert isinstance(result, dict)
    assert "success" in result
    assert "failures" in result
    assert "warnings" in result
    assert "statistics" in result

    # The cleaned data should have zero failures
    assert result["success"] is True, (
        f"Quality gate failed unexpectedly. Failures: {result['failures']}"
    )
    assert len(result["failures"]) == 0


def test_quality_gate_returns_expected_statistics(cleaned_data):
    """The quality gate should return statistics with row count and target counts."""
    result = check_data_quality(cleaned_data)
    stats = result["statistics"]

    assert "total_rows" in stats
    assert stats["total_rows"] == len(cleaned_data)
    assert "target_distribution" in stats
    # Binary target should have exactly two classes
    assert len(stats["target_distribution"]) == 2


def test_quality_gate_catches_broken_dataframe(broken_dataframe):
    """A deliberately broken dataset should fail the quality gate."""
    result = check_data_quality(broken_dataframe)

    # We expect failures here — that's the whole point
    assert result["success"] is False
    assert len(result["failures"]) > 0


def test_quality_gate_catches_missing_required_columns(broken_dataframe):
    """Specifically: the broken dataset is missing V1-V28 and Amount."""
    result = check_data_quality(broken_dataframe)

    # At least one failure should mention missing columns
    failure_text = " ".join(result["failures"]).lower()
    assert "missing" in failure_text or "required" in failure_text, (
        f"Expected a 'missing column' failure. Got: {result['failures']}"
    )


def test_quality_gate_catches_low_row_count():
    """A dataset with fewer than 100 rows should fail the row-count check."""
    tiny_df = pd.DataFrame({
        "Time": [0.0, 1.0, 2.0],
        "Amount": [10.0, 20.0, 30.0],
        "Class": [0, 0, 1],
    })
    # Add the V columns so the schema check passes
    for i in range(1, 29):
        tiny_df[f"V{i}"] = [0.0, 0.0, 0.0]

    result = check_data_quality(tiny_df)
    assert result["success"] is False
    # At least one failure mentions row count
    failure_text = " ".join(result["failures"]).lower()
    assert "row" in failure_text or "few" in failure_text
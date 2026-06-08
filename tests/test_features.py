"""Tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import create_features, select_features


@pytest.fixture(scope="session")
def engineered_data(cleaned_data) -> pd.DataFrame:
    """Run feature engineering once per session."""
    return create_features(cleaned_data)


def test_create_features_adds_expected_number_of_columns(cleaned_data, engineered_data):
    """create_features should add at least 10 new columns to the input."""
    new_cols = set(engineered_data.columns) - set(cleaned_data.columns)
    assert len(new_cols) >= 10, (
        f"Expected at least 10 new features, got {len(new_cols)}: {new_cols}"
    )


def test_create_features_preserves_row_count(cleaned_data, engineered_data):
    """Feature engineering should not add or remove rows."""
    assert len(engineered_data) == len(cleaned_data)


def test_create_features_preserves_original_columns(cleaned_data, engineered_data):
    """All original columns should still exist after feature engineering."""
    for col in cleaned_data.columns:
        assert col in engineered_data.columns, f"Original column '{col}' was dropped"


def test_no_nan_values_in_engineered_features(engineered_data):
    """Engineered features should not introduce any NaN values."""
    nan_counts = engineered_data.isnull().sum()
    columns_with_nans = nan_counts[nan_counts > 0]
    assert len(columns_with_nans) == 0, (
        f"Columns with NaN values: {dict(columns_with_nans)}"
    )


def test_hour_of_day_is_in_valid_range(engineered_data):
    """hour_of_day should always be between 0 and 23."""
    assert engineered_data["hour_of_day"].min() >= 0
    assert engineered_data["hour_of_day"].max() <= 23


def test_is_night_is_binary(engineered_data):
    """is_night should only contain 0 or 1."""
    unique_values = set(engineered_data["is_night"].unique())
    assert unique_values.issubset({0, 1}), (
        f"is_night contained unexpected values: {unique_values}"
    )


def test_log_amount_is_nonnegative(engineered_data):
    """log_amount = log(1 + Amount) — should always be >= 0 because Amount >= 0."""
    assert engineered_data["log_amount"].min() >= 0


def test_amount_bucket_is_in_valid_range(engineered_data):
    """amount_bucket is a discrete bucket from 0 to 4."""
    unique_values = set(engineered_data["amount_bucket"].unique())
    assert unique_values.issubset({0, 1, 2, 3, 4}), (
        f"amount_bucket contained unexpected values: {unique_values}"
    )


def test_v_negative_count_is_in_valid_range(engineered_data):
    """v_negative_count counts how many of V1-V28 are negative — must be in [0, 28]."""
    assert engineered_data["v_negative_count"].min() >= 0
    assert engineered_data["v_negative_count"].max() <= 28


def test_v_magnitude_is_positive(engineered_data):
    """v_magnitude is an L2 norm — must be non-negative."""
    assert engineered_data["v_magnitude"].min() >= 0


def test_select_features_returns_correct_shape(engineered_data):
    """select_features should return (list, DataFrame) and include target."""
    selected, df_reduced = select_features(engineered_data)

    assert isinstance(selected, list)
    assert len(selected) > 0
    assert isinstance(df_reduced, pd.DataFrame)
    # Target should be present in the reduced DataFrame
    assert "Class" in df_reduced.columns
    # Reduced df should have len(selected) features + the target
    assert df_reduced.shape[1] == len(selected) + 1
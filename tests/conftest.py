"""
Shared pytest fixtures.
Anything defined here is automatically available to all test files in this directory.
"""

from pathlib import Path

import pandas as pd
import pytest


# Project root path — useful for locating data and model files
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def cleaned_data() -> pd.DataFrame:
    """Load the cleaned dataset once per test session."""
    path = PROJECT_ROOT / "data" / "cleaned.csv"
    if not path.exists():
        pytest.skip(f"Cleaned data not found at {path}. Run the pipeline first.")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def features_data() -> pd.DataFrame:
    """Load the feature-engineered dataset once per test session."""
    path = PROJECT_ROOT / "data" / "features.csv"
    if not path.exists():
        pytest.skip(f"Features data not found at {path}. Run the pipeline first.")
    return pd.read_csv(path)


@pytest.fixture
def broken_dataframe() -> pd.DataFrame:
    """A deliberately broken dataset to verify the quality gate catches issues."""
    return pd.DataFrame({
        # Only a few rows — should fail the row-count check
        # Missing required columns (V1-V28, Amount)
        # Negative Time should fail the value-range check
        "Time": [-1, 100, 200, 300, 400],
        "Class": [0, 0, 0, 0, 0],   # Only one class — should fail target check
    })
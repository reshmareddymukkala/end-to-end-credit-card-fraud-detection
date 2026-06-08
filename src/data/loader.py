"""
Data loader: inspects a CSV file and prints key information about it.
"""

import pandas as pd
from pathlib import Path


def load_data(csv_path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    df = pd.read_csv(csv_path)
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Print a full inspection report of the DataFrame."""

    # 1. Shape — rows and columns
    print("=" * 60)
    print("DATASET SHAPE")
    print("=" * 60)
    print(f"Rows:    {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    print()

    # 2. Column names and data types
    print("=" * 60)
    print("COLUMNS AND DATA TYPES")
    print("=" * 60)
    print(df.dtypes)
    print()

    # 3. Summary statistics for numeric columns
    print("=" * 60)
    print("SUMMARY STATISTICS (numeric columns)")
    print("=" * 60)
    print(df.describe())
    print()

    # 4. Missing values: count and percentage
    print("=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    missing_count = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100

    missing_report = pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct": missing_pct.round(2)
    })
    # Only show columns that actually have missing values
    missing_report = missing_report[missing_report["missing_count"] > 0]

    if missing_report.empty:
        print("No missing values found.")
    else:
        print(missing_report.sort_values("missing_count", ascending=False))
    print()


if __name__ == "__main__":
    # Path to the data folder, relative to the project root
    data_folder = Path("data")

    # Find the first CSV file in the data folder
    csv_files = list(data_folder.glob("*.csv"))

    if not csv_files:
        print("No CSV files found in the data/ folder.")
        print("Please add a CSV file to data/ and try again.")
    else:
        csv_path = csv_files[0]
        print(f"Loading: {csv_path}\n")
        df = load_data(csv_path)
        inspect_data(df)
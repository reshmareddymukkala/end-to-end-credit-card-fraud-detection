"""
Data cleaner: handles nulls, duplicates, dtypes; saves cleaned CSV.
Run from the project root with: python src/data/cleaner.py
"""

import pandas as pd
from pathlib import Path

# Import the quality gate from the sibling module
from src.data.quality import check_data_quality, print_report


# ---------- Configuration ----------

TARGET_COLUMN = "Class"
IS_TIME_SERIES = False   # Set True if rows depend on previous rows (e.g. sensor data)

# Columns we expect to be numeric (will be coerced to float/int)
NUMERIC_COLUMNS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
INTEGER_COLUMNS = ["Class"]

# Treated as text/categorical (none for this dataset)
CATEGORICAL_COLUMNS = []


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean a DataFrame:
      1. Handle nulls
      2. Remove duplicates
      3. Convert dtypes
      4. Save to data/cleaned.csv
      5. Re-run quality gate
    Returns (cleaned_df, quality_result).
    """
    print(f"Starting clean: {len(df):,} rows, {df.shape[1]} columns")

    # ---------- Step 1: Handle nulls ----------

    # 1a. Drop rows where the target is null — we can't train on these
    if TARGET_COLUMN in df.columns:
        before = len(df)
        df = df.dropna(subset=[TARGET_COLUMN])
        dropped = before - len(df)
        if dropped > 0:
            print(f"  Dropped {dropped:,} rows with null target")

    # 1b. Drop columns with more than 50% nulls — they're too damaged to use
    null_pcts = (df.isnull().sum() / len(df)) * 100
    cols_to_drop = null_pcts[null_pcts > 50].index.tolist()
    if cols_to_drop:
        print(f"  Dropping columns with >50% nulls: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # 1c. Handle remaining nulls in other columns
    if IS_TIME_SERIES:
        # Forward-fill: copy the last known value into the gap
        before_nulls = df.isnull().sum().sum()
        df = df.ffill()
        # Some leading nulls have nothing to fill from — drop those rows
        df = df.dropna()
        after_nulls = df.isnull().sum().sum()
        print(f"  Forward-filled nulls: {before_nulls} → {after_nulls}")
    else:
        # Non-time-series: drop any row with any remaining null
        before = len(df)
        df = df.dropna()
        dropped = before - len(df)
        if dropped > 0:
            print(f"  Dropped {dropped:,} rows with nulls in other columns")

    # ---------- Step 2: Remove duplicates ----------

    before = len(df)
    df = df.drop_duplicates(keep="first")
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Removed {dropped:,} duplicate rows")
    else:
        print("  No duplicates found")

    # ---------- Step 3: Convert dtypes ----------

    # Numeric columns → float
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            # errors='coerce' turns un-convertible values into NaN, which is
            # safer than crashing on one bad value
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Integer columns → int (after confirming no nulls)
    for col in INTEGER_COLUMNS:
        if col in df.columns and not df[col].isnull().any():
            df[col] = df[col].astype("int64")

    # Categorical columns → string
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # If dtype coercion introduced nulls, drop those rows now
    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped:,} rows with nulls from dtype coercion")

    # Reset row indices since we dropped rows
    df = df.reset_index(drop=True)

    print(f"Finished clean: {len(df):,} rows, {df.shape[1]} columns")

    # ---------- Step 4: Save cleaned CSV ----------

    output_path = Path("data") / "cleaned.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to: {output_path}")

    # ---------- Step 5: Re-run quality gate ----------

    print("\nRe-running quality gate on cleaned data...\n")
    quality_result = check_data_quality(df)

    return df, quality_result


if __name__ == "__main__":
    data_folder = Path("data")

    # Find the source CSV — anything except cleaned.csv
    csv_files = [
        f for f in data_folder.glob("*.csv") if f.name != "cleaned.csv"
    ]

    if not csv_files:
        print("No source CSV files found in data/ folder.")
    else:
        csv_path = csv_files[0]
        print(f"Loading raw data from: {csv_path}\n")
        df_raw = pd.read_csv(csv_path)
        rows_before = len(df_raw)

        df_clean, quality_result = clean_data(df_raw)
        rows_after = len(df_clean)

        # Before/after summary
        print("\n" + "=" * 60)
        print("BEFORE / AFTER SUMMARY")
        print("=" * 60)
        print(f"Rows before: {rows_before:,}")
        print(f"Rows after:  {rows_after:,}")
        print(f"Removed:     {rows_before - rows_after:,} "
              f"({(rows_before - rows_after) / rows_before * 100:.2f}%)")

        print_report(quality_result)
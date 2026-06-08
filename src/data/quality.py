"""
Data quality gate: runs 5 automated checks on a DataFrame.
Run from the project root with: python src/data/quality.py
"""

import pandas as pd
from pathlib import Path


# ---------- Configuration ----------
# These define what the dataset SHOULD look like.
# For the credit card fraud dataset.

REQUIRED_COLUMNS = {
    "Time": "float64",
    "Amount": "float64",
    "Class": "int64",
}
# V1..V28 are also expected, all float64 — added programmatically below
for i in range(1, 29):
    REQUIRED_COLUMNS[f"V{i}"] = "float64"

TARGET_COLUMN = "Class"
IS_CLASSIFICATION = True

# Sensible bounds for specific columns: (min_allowed, max_allowed)
# None means no bound on that side.
VALUE_RANGES = {
    "Amount": (0, None),    # Amount can't be negative
    "Time":   (0, None),    # Time can't be negative
    "Class":  (0, 1),       # Binary target
}


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Run 5 quality checks on the DataFrame.
    Returns a dict with success flag, failures, warnings, and statistics.
    """
    failures = []
    warnings = []
    statistics = {}

    # ---------- Check 1: Schema validation ----------
    # Required columns exist + correct dtypes
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        failures.append(f"Missing required columns: {missing_cols}")

    wrong_dtypes = []
    for col, expected_dtype in REQUIRED_COLUMNS.items():
        if col in df.columns:
            actual_dtype = str(df[col].dtype)
            if actual_dtype != expected_dtype:
                wrong_dtypes.append(
                    f"{col}: expected {expected_dtype}, got {actual_dtype}"
                )
    if wrong_dtypes:
        failures.append(f"Wrong dtypes: {wrong_dtypes}")

    # ---------- Check 2: Row count ----------
    total_rows = len(df)
    statistics["total_rows"] = total_rows

    if total_rows < 100:
        failures.append(
            f"Too few rows: {total_rows} (minimum 100 required)"
        )
    elif total_rows < 1000:
        warnings.append(
            f"Low row count: {total_rows} (recommended at least 1000)"
        )

    # ---------- Check 3: Null rates ----------
    null_counts = df.isnull().sum()
    null_pcts = (null_counts / len(df)) * 100
    statistics["total_nulls_by_column"] = null_counts.to_dict()

    for col in df.columns:
        pct = null_pcts[col]
        if pct > 50:
            failures.append(
                f"Column '{col}' has {pct:.1f}% nulls (limit: 50%)"
            )
        elif pct > 20:
            warnings.append(
                f"Column '{col}' has {pct:.1f}% nulls (recommended < 20%)"
            )

    # ---------- Check 4: Value ranges ----------
    # Catches obvious data corruption: negative counts, percentages > 100, etc.
    for col, (min_allowed, max_allowed) in VALUE_RANGES.items():
        if col not in df.columns:
            continue

        col_min = df[col].min()
        col_max = df[col].max()

        if min_allowed is not None and col_min < min_allowed:
            failures.append(
                f"Column '{col}' has values below {min_allowed} "
                f"(min found: {col_min})"
            )
        if max_allowed is not None and col_max > max_allowed:
            failures.append(
                f"Column '{col}' has values above {max_allowed} "
                f"(max found: {col_max})"
            )

    # ---------- Check 5: Target distribution ----------
    if IS_CLASSIFICATION and TARGET_COLUMN in df.columns:
        target_counts = df[TARGET_COLUMN].value_counts()
        target_pcts = df[TARGET_COLUMN].value_counts(normalize=True) * 100

        statistics["target_distribution"] = target_counts.to_dict()
        statistics["target_distribution_pct"] = target_pcts.round(4).to_dict()

        n_classes = len(target_counts)
        if n_classes < 2:
            failures.append(
                f"Target has only {n_classes} class — need at least 2 "
                f"for classification"
            )
        else:
            min_class_pct = target_pcts.min()
            if min_class_pct < 5:
                warnings.append(
                    f"Target is imbalanced: smallest class is "
                    f"{min_class_pct:.2f}% of data (recommended >= 5%)"
                )

    # ---------- Final result ----------
    success = len(failures) == 0
    return {
        "success": success,
        "failures": failures,
        "warnings": warnings,
        "statistics": statistics,
    }


def print_report(result: dict) -> None:
    """Pretty-print the quality gate result."""
    print("=" * 60)
    print("DATA QUALITY GATE REPORT")
    print("=" * 60)

    status = "PASSED" if result["success"] else "FAILED"
    print(f"\nStatus: {status}\n")

    print(f"Failures: {len(result['failures'])}")
    for f in result["failures"]:
        print(f"  [FAIL] {f}")

    print(f"\nWarnings: {len(result['warnings'])}")
    for w in result["warnings"]:
        print(f"  [WARN] {w}")

    print("\nStatistics:")
    for key, value in result["statistics"].items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                # Skip columns with zero nulls to keep output clean
                if key == "total_nulls_by_column" and v == 0:
                    continue
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    print()


if __name__ == "__main__":
    data_folder = Path("data")
    csv_files = list(data_folder.glob("*.csv"))

    if not csv_files:
        print("No CSV files found in the data/ folder.")
    else:
        csv_path = csv_files[0]
        print(f"Loading: {csv_path}\n")
        df = pd.read_csv(csv_path)
        result = check_data_quality(df)
        print_report(result)

        # Exit with error code if quality gate failed — useful for CI/CD later
        import sys
        if not result["success"]:
            sys.exit(1)
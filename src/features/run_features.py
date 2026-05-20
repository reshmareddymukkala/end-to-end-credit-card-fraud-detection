"""
Feature engineering pipeline runner.

Orchestrates the feature engineering and selection steps:
  1. Load cleaned data
  2. Create new features (create_features)
  3. Select useful features (select_features)
  4. Save result to data/features.csv

This is the reproducible entry point for the feature stage.
Run from the project root with: python src/features/run_features.py
"""

import time
from pathlib import Path

import pandas as pd

from src.features.engineering import create_features, select_features


def run_feature_pipeline(
    input_path: Path = Path("data") / "cleaned.csv",
    output_path: Path = Path("data") / "features.csv",
) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline.
    Returns the final DataFrame with engineered + selected features.
    """
    # ---------- Load ----------
    if not input_path.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at {input_path}. "
            f"Run src/data/cleaner.py first."
        )

    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)
    rows_before, cols_before = df.shape
    print(f"  Loaded: {rows_before:,} rows × {cols_before} columns")

    # ---------- Step 1: Create features ----------
    print("\n" + "=" * 60)
    print("STEP 1: CREATING FEATURES")
    print("=" * 60)
    t1 = time.time()
    df_featured = create_features(df)
    t1_elapsed = time.time() - t1

    new_features = [c for c in df_featured.columns if c not in df.columns]
    print(f"\nCreated {len(new_features)} new features in "
          f"{t1_elapsed:.2f}s:")
    for col in new_features:
        print(f"  + {col}")

    # ---------- Step 2: Select features ----------
    print("\n" + "=" * 60)
    print("STEP 2: SELECTING FEATURES")
    print("=" * 60)
    t2 = time.time()
    selected, df_selected = select_features(df_featured)
    t2_elapsed = time.time() - t2
    print(f"\nFeature selection completed in {t2_elapsed:.2f}s")

    # ---------- Save ----------
    df_selected.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # ---------- Summary ----------
    rows_after, cols_after = df_selected.shape
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Input shape:       {rows_before:,} rows × {cols_before} columns")
    print(f"After creation:    {df_featured.shape[0]:,} rows × "
          f"{df_featured.shape[1]} columns")
    print(f"After selection:   {rows_after:,} rows × {cols_after} columns "
          f"({len(selected)} features + target)")

    print(f"\nFinal features kept ({len(selected)}):")
    for col in selected:
        marker = "★" if col in new_features else " "
        # ★ marks features we engineered (vs. original dataset features)
        print(f"  {marker} {col}")

    return df_selected


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("FEATURE PIPELINE STARTED")
    print("=" * 60 + "\n")

    df_final = run_feature_pipeline()

    total_elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETE in {total_elapsed:.2f}s")
    print("=" * 60)
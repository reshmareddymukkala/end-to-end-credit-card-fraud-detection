"""
Feature engineering for credit card fraud detection.

Constraint: V1-V28 are PCA-anonymized — we don't know their original meaning,
so most domain-specific engineering uses the two interpretable features
(Time and Amount). The V features are combined statistically rather than
semantically.

Run from the project root with: python src/features/engineering.py
"""

import numpy as np
import pandas as pd
from pathlib import Path


# Top features by absolute correlation with Class, identified in EDA
# (negative correlations: fraud tends to have lower values on these)
TOP_FRAUD_V_FEATURES = ["V17", "V14", "V12", "V10", "V16"]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer new features for fraud detection.
    Returns a new DataFrame with original columns plus engineered features.
    """
    df = df.copy()  # Avoid mutating the caller's DataFrame

    # =================================================================
    # CATEGORY 1: DOMAIN-SPECIFIC FEATURES (Time and Amount)
    # =================================================================
    # The dataset's Time column is "seconds since first transaction"
    # spanning ~2 days. We can derive hour-of-day patterns because
    # fraud often happens at unusual hours when victims are asleep.

    # Feature 1: hour_of_day
    # WHY: Fraud is often more common at night when cardholders won't
    # notice the transaction immediately. The original Time is just
    # seconds since the dataset start, which is hard for a model to use.
    df["hour_of_day"] = (df["Time"] // 3600) % 24

    # Feature 2: is_night
    # WHY: A binary flag for "unusual hour" makes the night-time pattern
    # explicit for the model. Fraud rates often spike between midnight and 6 AM.
    df["is_night"] = ((df["hour_of_day"] >= 0) &
                      (df["hour_of_day"] < 6)).astype(int)

    # Feature 3: log_amount
    # WHY: EDA showed Amount is extremely right-skewed ($0 to $25,000
    # with most values under $100). Linear models and distance-based
    # algorithms struggle with skewed features. log1p(x) = log(1+x)
    # handles zeros gracefully.
    df["log_amount"] = np.log1p(df["Amount"])

    # Feature 4: amount_zero_flag
    # WHY: A surprising number of transactions have Amount = $0
    # (card verification, pre-auth checks). These are mechanically
    # different from a $50 purchase and the model should know.
    df["amount_zero_flag"] = (df["Amount"] == 0).astype(int)

    # Feature 5: amount_bucket
    # WHY: Bucketing into discrete tiers (small/medium/large/huge)
    # gives tree-based models cleaner split points and captures
    # threshold-based fraud patterns (e.g., many fraud transactions
    # cluster in specific dollar ranges to stay under detection limits).
    df["amount_bucket"] = pd.cut(
        df["Amount"],
        bins=[-0.01, 1, 10, 100, 1000, np.inf],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)

    # =================================================================
    # CATEGORY 2: STATISTICAL / AGGREGATE FEATURES
    # =================================================================
    '''Since the V features are PCA components, we can summarize them
    statistically. Fraud transactions tend to be "outliers" across
    multiple PCA dimensions simultaneously, these aggregates capture
    that without needing to know what each V means.'''

    # Feature 6: v_magnitude
    '''WHY: The L2 (Euclidean) norm of the V vector measures how far
    a transaction sits from the "center" of the PCA space. Fraud
    transactions identified in EDA had extreme negative values on
    multiple V features simultaneously, so they should have large
    overall magnitude.'''
    v_cols = [f"V{i}" for i in range(1, 29)]
    df["v_magnitude"] = np.sqrt((df[v_cols] ** 2).sum(axis=1))

    # Feature 7: v_negative_count
    # WHY: EDA showed top fraud-correlated features (V10, V12, V14, V17)
    # all have NEGATIVE correlation with Class, fraud sits at lower
    # values. Counting how many V features are negative for a row gives
    # a simple "fraud-likeness" score.
    df["v_negative_count"] = (df[v_cols] < 0).sum(axis=1)

    # Feature 8: top_fraud_v_mean
    # WHY: Average of the V features most correlated with fraud.
    # If a transaction is fraud, all of these tend to be very negative,
    # so the mean will be strongly negative. A single combined feature
    # is easier for linear models than five separate ones.
    df["top_fraud_v_mean"] = df[TOP_FRAUD_V_FEATURES].mean(axis=1)

    # Feature 9: top_fraud_v_min
    # WHY: For fraud, multiple V features hit extreme negative values.
    # The minimum captures how extreme the most-negative one gets,
    # which is often a stronger fraud signal than the mean.
    df["top_fraud_v_min"] = df[TOP_FRAUD_V_FEATURES].min(axis=1)

    # =================================================================
    # CATEGORY 3: INTERACTION FEATURES
    # =================================================================
    # Cases where two features together carry more information than
    # either does alone.

    # Feature 10: amount_v17_interaction  (INTERACTION 1)
    # WHY: V17 is the strongest single fraud predictor. Combining it
    # with transaction amount captures "extreme V17 AND large amount" —
    # high-value fraud transactions that are riskier than low-value ones.
    df["amount_v17_interaction"] = df["log_amount"] * df["V17"]

    # Feature 11: night_amount_ratio  (INTERACTION 2)
    # WHY: A large transaction at night is more suspicious than a large
    # transaction during the day. Multiplying log_amount by is_night
    # zeros out daytime transactions and amplifies night-time ones.
    df["night_amount_ratio"] = df["log_amount"] * df["is_night"]

    # Feature 12: fraud_score_proxy  (INTERACTION 3)
    # WHY: Combine the strongest fraud signals into one composite.
    # Negative top_fraud_v_mean * positive v_magnitude captures "extreme
    # outlier in the fraud direction." Negate the result so larger
    # numbers mean more fraud-like.
    df["fraud_score_proxy"] = -df["top_fraud_v_mean"] * df["v_magnitude"]

    # Feature 13: v14_v17_interaction  (INTERACTION 4)
    # WHY: V14 and V17 are the #1 and #2 fraud-correlated features
    # individually. Their product captures cases where BOTH are extreme,
    # which is a much stronger signal than either being extreme alone.
    df["v14_v17_interaction"] = df["V14"] * df["V17"]

    return df
def select_features(
    df: pd.DataFrame,
    target_column: str = "Class",
    correlation_threshold: float = 0.95,
    variance_threshold_ratio: float = 0.01,
) -> tuple[list[str], pd.DataFrame]:
    """
    Select features by removing:
      1. Features highly correlated with another feature (|corr| > 0.95)
      2. Features with very low variance (< 1% of overall mean variance)

    The target column is excluded from these checks (we never drop the target).

    Returns:
        selected_features: list of feature names kept
        df_reduced: DataFrame containing only the selected features + target
    """
    # Separate the target — we never want to drop or modify it
    if target_column in df.columns:
        target = df[target_column]
        features_df = df.drop(columns=[target_column])
    else:
        target = None
        features_df = df.copy()

    # Only consider numeric features for variance/correlation checks
    numeric_df = features_df.select_dtypes(include=[np.number])
    original_features = list(numeric_df.columns)

    print(f"Starting feature selection with {len(original_features)} features")
    dropped_features = {}  # feature_name -> reason it was dropped

    # ---------- Check 1: High correlation pairs ----------
    print(f"\nChecking for correlations > {correlation_threshold}...")
    corr_matrix = numeric_df.corr().abs()

    # We only want to look at each pair once. Take the upper triangle of the
    # correlation matrix (above the diagonal) so we don't double-count pairs
    # like (A, B) and (B, A), and so we don't flag a feature against itself.
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # For each column in the upper triangle, if any value exceeds the
    # threshold, that column is redundant with an earlier column → drop it.
    # (Earlier columns are the "first" we keep; later ones get dropped.)
    cols_to_drop_corr = [
        col for col in upper_triangle.columns
        if any(upper_triangle[col] > correlation_threshold)
    ]

    if cols_to_drop_corr:
        for col in cols_to_drop_corr:
            # Find which earlier feature it's correlated with (for logging)
            correlated_with = upper_triangle[col][
                upper_triangle[col] > correlation_threshold
            ].idxmax()
            corr_value = upper_triangle.loc[correlated_with, col]
            reason = (f"corr={corr_value:.3f} with '{correlated_with}' "
                      f"(>{correlation_threshold})")
            dropped_features[col] = reason
            print(f"  Drop '{col}' — {reason}")
    else:
        print("  No highly-correlated features found.")

    # Remove the high-correlation drops before doing variance check,
    # so we don't compute variance on features we already plan to drop
    remaining = [c for c in original_features if c not in cols_to_drop_corr]
    numeric_df = numeric_df[remaining]

    # ---------- Check 2: Low variance features ----------
    print("\nChecking for low-variance features...")
    variances = numeric_df.var()
    # Use median instead of mean — robust to outlier features like 'Time'
    # whose variance is orders of magnitude larger than everything else.
    median_variance = variances.median()
    threshold = variance_threshold_ratio * median_variance
    print(f"  Median variance across features: {median_variance:.4f}")
    print(f"  Variance threshold ({variance_threshold_ratio*100:.0f}% "
          f"of median): {threshold:.4f}")

    cols_to_drop_var = variances[variances < threshold].index.tolist()
    if cols_to_drop_var:
        for col in cols_to_drop_var:
            reason = (f"variance={variances[col]:.6f} "
                      f"(below threshold {threshold:.4f})")
            dropped_features[col] = reason
            print(f"  Drop '{col}' — {reason}")
    else:
        print("  No low-variance features found.")

    # ---------- Build final list and reduced DataFrame ----------
    selected_features = [
        c for c in original_features if c not in dropped_features
    ]

    # Reduced DataFrame: selected features + target (if present)
    df_reduced = df[selected_features].copy()
    if target is not None:
        df_reduced[target_column] = target

    # Summary
    print(f"\n{'=' * 60}")
    print("FEATURE SELECTION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Started with: {len(original_features)} features")
    print(f"Dropped:      {len(dropped_features)}")
    print(f"  - High correlation: {len(cols_to_drop_corr)}")
    print(f"  - Low variance:     {len(cols_to_drop_var)}")
    print(f"Selected:     {len(selected_features)} features")

    return selected_features, df_reduced

if __name__ == "__main__":
    data_folder = Path("data")
    input_path = data_folder / "cleaned.csv"

    if not input_path.exists():
        print(f"Cleaned data not found at {input_path}.")
        print("Run src/data/cleaner.py first.")
    else:
        print(f"Loading: {input_path}\n")
        df = pd.read_csv(input_path)
        print(f"Before: {df.shape[0]:,} rows × {df.shape[1]} columns")

        df_featured = create_features(df)
        print(f"After:  {df_featured.shape[0]:,} rows × "
              f"{df_featured.shape[1]} columns")

        new_columns = [c for c in df_featured.columns if c not in df.columns]
        print(f"\nEngineered {len(new_columns)} new features:")
        for col in new_columns:
            print(f"  - {col}")

        # Save the result for downstream steps
        output_path = data_folder / "featured.csv"
        df_featured.to_csv(output_path, index=False)
        print(f"\nSaved featured data to: {output_path}")

        # Quick sanity check: any nulls introduced?
        new_nulls = df_featured[new_columns].isnull().sum().sum()
        if new_nulls > 0:
            print(f"\nWARNING: {new_nulls} nulls introduced by engineering.")
        else:
            print("\nNo nulls introduced. Engineering successful.")

        # Show summary statistics for engineered features
        print("\nSummary of engineered features:")
        print(df_featured[new_columns].describe().T[
            ["mean", "std", "min", "max"]
        ].round(3))

        # ---- Step 2: Select features ----
        print("\n" + "=" * 60)
        print("RUNNING FEATURE SELECTION")
        print("=" * 60 + "\n")
        selected, df_selected = select_features(df_featured)

        # Save the selected version
        selected_path = data_folder / "selected.csv"
        df_selected.to_csv(selected_path, index=False)
        print(f"\nSaved selected data to: {selected_path}")
        print(f"Final shape: {df_selected.shape[0]:,} rows × "
              f"{df_selected.shape[1]} columns")
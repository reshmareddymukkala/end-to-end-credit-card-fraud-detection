"""
Demo data generators for when real artifacts haven't been produced yet.

These let the app run immediately after a fresh `git clone`, even before
the training pipeline has been executed. When the real files exist,
the app uses them; when they don't, these generators provide stand-ins.

Synthetic data here is INTENTIONALLY similar in structure to the real
data so plots and tables look plausible, but it's NOT representative
of model performance — that's why we banner the app when running in
demo mode.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# Same constants the real pipeline uses
TARGET_COLUMN = "Class"
RANDOM_STATE = 42

# What the real feature set looks like — used to fake the columns
ORIGINAL_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
ENGINEERED_FEATURES = [
    "hour_of_day", "is_night", "log_amount", "amount_bucket",
    "v_magnitude", "v_negative_count", "top_fraud_v_mean", "top_fraud_v_min",
    "amount_v17_interaction", "night_amount_ratio",
    "fraud_score_proxy", "v14_v17_interaction",
]


def is_demo_mode() -> bool:
    """
    Return True if any of the real data files are missing.
    Used to decide whether to show the demo-mode banner.
    """
    required_files = [
        Path("data") / "features.csv",
        Path("data") / "predictions.csv",
        Path("data") / "model_results.json",
    ]
    return any(not f.exists() for f in required_files)


def generate_demo_features(n_rows: int = 10_000, fraud_rate: float = 0.0017):
    """
    Build a synthetic feature dataset that LOOKS like the real fraud dataset:
      - 10K rows (smaller than real 283K for speed)
      - Same column names
      - Same approximate class imbalance (~0.17%)
      - Fraud cases sit at extreme negative V values, matching real patterns
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n_fraud = max(1, int(n_rows * fraud_rate))
    n_legit = n_rows - n_fraud

    # ----- Legit transactions: V features near zero, normal amounts -----
    legit = pd.DataFrame({
        "Time": rng.uniform(0, 172000, n_legit),
        "Amount": rng.exponential(80, n_legit).clip(0, 5000),
    })
    for i in range(1, 29):
        # PCA-like: roughly N(0, 1)
        legit[f"V{i}"] = rng.normal(0, 1, n_legit)

    # ----- Fraud transactions: V features at extreme negatives -----
    fraud = pd.DataFrame({
        "Time": rng.uniform(0, 172000, n_fraud),
        "Amount": rng.exponential(120, n_fraud).clip(0, 5000),
    })
    for i in range(1, 29):
        # Most V features near zero; top fraud features at extreme negatives
        if i in [10, 12, 14, 17]:
            fraud[f"V{i}"] = rng.normal(-5, 2, n_fraud)
        else:
            fraud[f"V{i}"] = rng.normal(0, 1.5, n_fraud)

    df = pd.concat([legit, fraud], ignore_index=True)
    df["Class"] = [0] * n_legit + [1] * n_fraud

    # ----- Engineered features (computed like the real pipeline) -----
    df["hour_of_day"] = (df["Time"] // 3600) % 24
    df["is_night"] = ((df["hour_of_day"] >= 0) & (df["hour_of_day"] < 6)).astype(int)
    df["log_amount"] = np.log1p(df["Amount"])
    df["amount_bucket"] = pd.cut(
        df["Amount"], bins=[-0.01, 1, 10, 100, 1000, np.inf], labels=[0, 1, 2, 3, 4]
    ).astype(int)

    v_cols = [f"V{i}" for i in range(1, 29)]
    df["v_magnitude"] = np.sqrt((df[v_cols] ** 2).sum(axis=1))
    df["v_negative_count"] = (df[v_cols] < 0).sum(axis=1)

    top_fraud_v = ["V17", "V14", "V12", "V10", "V16"]
    df["top_fraud_v_mean"] = df[top_fraud_v].mean(axis=1)
    df["top_fraud_v_min"] = df[top_fraud_v].min(axis=1)

    df["amount_v17_interaction"] = df["log_amount"] * df["V17"]
    df["night_amount_ratio"] = df["log_amount"] * df["is_night"]
    df["fraud_score_proxy"] = -df["top_fraud_v_mean"] * df["v_magnitude"]
    df["v14_v17_interaction"] = df["V14"] * df["V17"]

    # Shuffle so the fraud cases aren't all at the bottom
    return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def generate_demo_predictions(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Take the demo feature DataFrame and add fake "predicted_class" and
    "fraud_probability" columns. We use a simple heuristic — assign higher
    fraud probability to rows that already have fraud-like feature values —
    so confusion matrices look plausible.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    # Use the same logic the real model relies on: extreme negative V14/V17
    # are strong fraud signals. Synthesize probabilities based on that.
    base_score = (-features_df["V14"] - features_df["V17"]) / 2

    # Add noise so the predictions aren't perfectly aligned with truth
    score = base_score + rng.normal(0, 1, len(features_df))

    # Map to [0, 1] with a sigmoid
    proba = 1.0 / (1.0 + np.exp(-score + 2))

    df = features_df.copy()
    df["true_class"] = df["Class"]
    df["fraud_probability"] = proba
    df["predicted_class"] = (proba >= 0.5).astype(int)
    df = df.drop(columns=["Class"])
    return df


def generate_demo_results() -> dict:
    """
    Plausible model comparison results — these numbers are made up but
    structurally similar to what the real pipeline produces.
    """
    feature_importance = [
        {"feature": "V14", "importance": 0.183},
        {"feature": "V17", "importance": 0.142},
        {"feature": "V12", "importance": 0.119},
        {"feature": "V10", "importance": 0.095},
        {"feature": "top_fraud_v_min", "importance": 0.071},
        {"feature": "v14_v17_interaction", "importance": 0.054},
        {"feature": "fraud_score_proxy", "importance": 0.048},
        {"feature": "V11", "importance": 0.040},
        {"feature": "V4", "importance": 0.036},
        {"feature": "top_fraud_v_mean", "importance": 0.032},
        {"feature": "v_magnitude", "importance": 0.028},
        {"feature": "V3", "importance": 0.024},
        {"feature": "Amount", "importance": 0.019},
        {"feature": "log_amount", "importance": 0.017},
        {"feature": "V16", "importance": 0.014},
    ]

    return {
        "baseline": {
            "name": "Logistic Regression (Baseline)",
            "pr_auc": 0.673, "roc_auc": 0.959,
            "precision": 0.055, "recall": 0.863, "f1": 0.104,
            "tn": 55254, "fp": 1397, "fn": 13, "tp": 82,
        },
        "random_forest": {
            "name": "Random Forest",
            "pr_auc": 0.811, "roc_auc": 0.970,
            "precision": 0.972, "recall": 0.726, "f1": 0.831,
            "tn": 56649, "fp": 2, "fn": 26, "tp": 69,
        },
        "xgboost_untuned": {
            "name": "XGBoost (Untuned)",
            "pr_auc": 0.806, "roc_auc": 0.972,
            "precision": 0.923, "recall": 0.758, "f1": 0.832,
            "tn": 56645, "fp": 6, "fn": 23, "tp": 72,
        },
        "xgboost_tuned": {
            "name": "XGBoost (Tuned) ★ WINNER",
            "pr_auc": 0.810, "roc_auc": 0.973,
            "precision": 0.936, "recall": 0.768, "f1": 0.844,
            "tn": 56646, "fp": 5, "fn": 22, "tp": 73,
        },
        "feature_importance": feature_importance,
    }
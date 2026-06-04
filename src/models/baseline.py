"""
Baseline model for credit card fraud detection.

This script trains a logistic regression baseline. Because the dataset is
severely imbalanced (~600:1), we use:
  - Stratified train/test split (preserves class ratio in both sets)
  - StandardScaler inside a Pipeline (prevents test-set leakage)
  - class_weight="balanced" (penalizes minority-class mistakes more)
  - PR-AUC as the headline metric (more honest than accuracy or ROC-AUC
    for heavily imbalanced data)

Run from the project root with: python src/models/baseline.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Identify this as a classification task (vs. regression)
TASK_TYPE = "classification"
TARGET_COLUMN = "Class"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_features(path: Path = Path("data") / "features.csv") -> pd.DataFrame:
    """Load the feature-engineered dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found at {path}. "
            f"Run src/features/run_features.py first."
        )
    return pd.read_csv(path)


def split_data(df: pd.DataFrame):
    """
    Split into train/test using stratified sampling on the target.
    Stratification ensures both splits have the same class ratio.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,           # Critical for imbalanced data
        random_state=RANDOM_STATE,
    )

    print(f"Train set: {X_train.shape[0]:,} rows "
          f"({y_train.sum()} fraud, "
          f"{y_train.mean() * 100:.3f}% fraud rate)")
    print(f"Test set:  {X_test.shape[0]:,} rows "
          f"({y_test.sum()} fraud, "
          f"{y_test.mean() * 100:.3f}% fraud rate)")

    return X_train, X_test, y_train, y_test


def train_baseline(X_train, y_train) -> Pipeline:
    """
    Train a logistic regression baseline inside a Pipeline.

    The Pipeline handles scaling automatically:
      - During fit: scaler.fit_transform on training data
      - During predict: scaler.transform on new data
    This prevents test-set leakage that would happen if we scaled first.

    class_weight='balanced' tells the model to weight rare-class mistakes
    more heavily, which is essential for imbalanced data.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,        # Default 100 may not converge
            random_state=RANDOM_STATE,
        )),
    ])
    print("\nTraining logistic regression...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")
    return pipeline


def evaluate(model: Pipeline, X_test, y_test) -> dict:
    """Evaluate the model on the test set and print results."""
    y_pred = model.predict(X_test)
    # predict_proba gives probability of each class; we want probability of class 1
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_proba),
        "pr_auc":    average_precision_score(y_test, y_proba),
    }

    print("\n" + "=" * 60)
    print("BASELINE MODEL EVALUATION")
    print("=" * 60)
    print(f"Accuracy:    {metrics['accuracy']:.4f}  "
          f"<-- misleading on imbalanced data, ignore")
    print(f"Precision:   {metrics['precision']:.4f}  "
          f"<-- of flagged fraud, how many were real")
    print(f"Recall:      {metrics['recall']:.4f}  "
          f"<-- of real fraud, how much we caught")
    print(f"F1 score:    {metrics['f1']:.4f}  "
          f"<-- balance of precision and recall")
    print(f"ROC-AUC:     {metrics['roc_auc']:.4f}  "
          f"<-- inflated by class imbalance")
    print(f"PR-AUC:      {metrics['pr_auc']:.4f}  "
          f"<-- KEY METRIC for fraud detection")

    # Confusion matrix gives a much more concrete picture than scalar metrics
    print("\nConfusion matrix:")
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"  True Negatives  (legit correctly):  {tn:,}")
    print(f"  False Positives (legit flagged):    {fp:,}")
    print(f"  False Negatives (fraud missed):     {fn:,}")
    print(f"  True Positives  (fraud caught):     {tp:,}")

    return metrics


def save_model(model: Pipeline, path: Path = Path("models") / "baseline.pkl") -> None:
    """Save the trained pipeline to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"\nSaved baseline model to: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("BASELINE MODEL TRAINING")
    print("=" * 60 + "\n")

    df = load_features()
    print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns\n")

    X_train, X_test, y_train, y_test = split_data(df)
    model = train_baseline(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    save_model(model)

    print("\n" + "=" * 60)
    print("BASELINE COMPLETE")
    print("=" * 60)
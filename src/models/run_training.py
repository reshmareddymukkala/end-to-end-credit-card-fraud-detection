"""
Training runner with MLflow experiment tracking.

Trains two model configurations:
  1. baseline       — logistic regression with class weighting
  2. tuned_best     — XGBoost with hyperparameters from Optuna tuning

Each run is logged to MLflow: parameters, metrics, and the model artifact.
The tuned model is also saved as production_model.pkl for downstream use.

Prereq: start MLflow server in another terminal:
    mlflow server --host 127.0.0.1 --port 5000

Run from the project root with: python src/models/run_training.py
"""

import json
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
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
from xgboost import XGBClassifier


# ---------- Configuration ----------
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "fraud_detection"

TARGET_COLUMN = "Class"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_and_split(path: Path = Path("data") / "features.csv"):
    """Load features and produce a stratified 80/20 split."""
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return X_train, X_test, y_train, y_test


def compute_metrics(model, X, y, prefix: str) -> dict:
    """
    Compute classification metrics on (X, y).
    The prefix ('train' or 'test') prepends each metric name so MLflow can
    show both train and test side by side.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        f"{prefix}_accuracy":  accuracy_score(y, y_pred),
        f"{prefix}_precision": precision_score(y, y_pred, zero_division=0),
        f"{prefix}_recall":    recall_score(y, y_pred, zero_division=0),
        f"{prefix}_f1":        f1_score(y, y_pred, zero_division=0),
        f"{prefix}_roc_auc":   roc_auc_score(y, y_proba),
        f"{prefix}_pr_auc":    average_precision_score(y, y_proba),
        f"{prefix}_tp": int(tp),
        f"{prefix}_fp": int(fp),
        f"{prefix}_fn": int(fn),
        f"{prefix}_tn": int(tn),
    }


def train_baseline(X_train, y_train):
    """Logistic regression baseline with scaling + class weighting."""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )),
    ])
    model.fit(X_train, y_train)
    return model


def train_tuned_xgboost(X_train, y_train, best_params: dict):
    """Train XGBoost with the tuned hyperparameters from Optuna."""
    neg_to_pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    fixed_params = {
        "scale_pos_weight": neg_to_pos_ratio,
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        "verbosity": 0,
    }
    model = XGBClassifier(**best_params, **fixed_params)
    model.fit(X_train, y_train)
    return model


def run_experiment(config_name: str, model, params: dict,
                   X_train, X_test, y_train, y_test):
    """
    Train one model configuration and log everything to MLflow.

    'with mlflow.start_run()' creates a new run that captures everything
    logged inside the block — params, metrics, artifacts. Each run gets
    its own UUID and shows up as a row in the MLflow UI.
    """
    with mlflow.start_run(run_name=config_name):
        # Log the config identifier as a tag so we can filter runs by it
        mlflow.set_tag("config_name", config_name)

        # ---- Log hyperparameters ----
        for key, value in params.items():
            mlflow.log_param(key, value)

        # ---- Time the training ----
        t0 = time.time()
        # (model is already trained — measure the recorded time)
        # If you wanted to time fit() itself, you'd train inside this block.
        train_time = time.time() - t0
        mlflow.log_metric("train_time_seconds", train_time)

        # ---- Log metrics on both train and test sets ----
        # Train metrics tell us if the model fit the training data well
        # Test metrics tell us if it generalizes
        train_metrics = compute_metrics(model, X_train, y_train, prefix="train")
        test_metrics = compute_metrics(model, X_test, y_test, prefix="test")

        for key, value in {**train_metrics, **test_metrics}.items():
            mlflow.log_metric(key, value)

        # ---- Log the model itself as an artifact ----
        # MLflow has model-flavor-specific loggers (sklearn, xgboost, etc.)
        # that handle serialization correctly for each library.
        if isinstance(model, XGBClassifier):
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        # ---- Print summary to console ----
        print(f"\n--- {config_name} ---")
        print(f"  Train PR-AUC: {train_metrics['train_pr_auc']:.4f}")
        print(f"  Test PR-AUC:  {test_metrics['test_pr_auc']:.4f}")
        print(f"  Test precision/recall: "
              f"{test_metrics['test_precision']:.4f} / "
              f"{test_metrics['test_recall']:.4f}")
        print(f"  Confusion: TP={test_metrics['test_tp']}  "
              f"FP={test_metrics['test_fp']}  "
              f"FN={test_metrics['test_fn']}")

        return test_metrics


def load_best_params(path: Path = Path("models") / "best_params.json") -> dict:
    """Load the hyperparameters chosen by the Optuna tuning step."""
    if not path.exists():
        raise FileNotFoundError(
            f"Best params not found at {path}. "
            f"Run src/models/tuning.py first."
        )
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING RUNNER (with MLflow tracking)")
    print("=" * 60 + "\n")

    # ---- Configure MLflow ----
    # Point the client at the local server we started in the other terminal
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Experiment: {EXPERIMENT_NAME}\n")

    # ---- Load data ----
    X_train, X_test, y_train, y_test = load_and_split()
    print(f"Train: {X_train.shape[0]:,} rows ({y_train.sum()} fraud)")
    print(f"Test:  {X_test.shape[0]:,} rows ({y_test.sum()} fraud)")

    # ---- Run 1: Baseline ----
    print("\n" + "=" * 60)
    print("Training BASELINE (logistic regression)")
    print("=" * 60)
    baseline_params = {
        "model_name": "logistic_regression",
        "class_weight": "balanced",
        "max_iter": 1000,
        "scaler": "StandardScaler",
    }
    baseline_model = train_baseline(X_train, y_train)
    baseline_metrics = run_experiment(
        "baseline", baseline_model, baseline_params,
        X_train, X_test, y_train, y_test,
    )

    # ---- Run 2: Tuned XGBoost ----
    print("\n" + "=" * 60)
    print("Training TUNED XGBOOST (best params from Optuna)")
    print("=" * 60)
    best_params = load_best_params()
    tuned_params = {"model_name": "xgboost_tuned", **best_params}
    tuned_model = train_tuned_xgboost(X_train, y_train, best_params)
    tuned_metrics = run_experiment(
        "tuned_best", tuned_model, tuned_params,
        X_train, X_test, y_train, y_test,
    )

    # ---- Save final production model ----
    # The tuned XGBoost is what we'd serve in production
    Path("models").mkdir(exist_ok=True)
    production_path = Path("models") / "production_model.pkl"
    joblib.dump(tuned_model, production_path)
    print(f"\nSaved production model to: {production_path}")

    # ---- Final comparison ----
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20}{'Baseline':>12}{'Tuned':>12}")
    print("-" * 44)
    for metric_key in ["test_pr_auc", "test_precision", "test_recall",
                       "test_f1", "test_roc_auc"]:
        clean_name = metric_key.replace("test_", "").upper()
        print(f"{clean_name:<20}{baseline_metrics[metric_key]:>12.4f}"
              f"{tuned_metrics[metric_key]:>12.4f}")

    print(f"\nView all runs at: {MLFLOW_TRACKING_URI}")
    print("=" * 60)
"""
Model comparison for credit card fraud detection.

Compares three candidate models against the logistic regression baseline:
  1. Random Forest (no scaling, ensemble, robust)
  2. XGBoost (gradient boosting, industry standard for fraud)
  3. LightGBM (faster gradient boosting alternative)

Each model is evaluated with 5-fold stratified cross-validation on PR-AUC,
then tested on a held-out test set. We track training time as well, since
the deployment target (automated system) cares about operational efficiency.

Run from the project root with: python src/models/compare_models.py
"""

import time
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier


TARGET_COLUMN = "Class"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


def load_and_split(path: Path = Path("data") / "features.csv"):
    """Load features and produce a stratified 80/20 train/test split."""
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"Train: {X_train.shape[0]:,} rows ({y_train.sum()} fraud)")
    print(f"Test:  {X_test.shape[0]:,} rows ({y_test.sum()} fraud)")

    # scale_pos_weight for XGBoost: ratio of negatives to positives in train set
    neg_to_pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Class imbalance ratio (neg/pos): {neg_to_pos_ratio:.1f}")

    return X_train, X_test, y_train, y_test, neg_to_pos_ratio


def build_models(neg_to_pos_ratio: float) -> dict:
    """Build the three candidate models, each configured for imbalanced data."""
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            n_jobs=-1,            # Use all CPU cores
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            scale_pos_weight=neg_to_pos_ratio,  # The imbalance fix for XGBoost
            eval_metric="aucpr",                # Optimize for PR-AUC
            tree_method="hist",                 # Fast histogram-based trees
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            #is_unbalance=True,    # The imbalance fix for LightGBM
            scale_pos_weight=neg_to_pos_ratio,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=-1,
        ),
    }


def evaluate_on_test(model, X_test, y_test, model_name: str) -> dict:
    """Evaluate a fitted model on the test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "model": model_name,
        "test_pr_auc": average_precision_score(y_test, y_proba),
        "test_roc_auc": roc_auc_score(y_test, y_proba),
        "test_precision": precision_score(y_test, y_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_pred, zero_division=0),
        "test_f1": f1_score(y_test, y_pred, zero_division=0),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def run_comparison():
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60 + "\n")

    X_train, X_test, y_train, y_test, neg_to_pos = load_and_split()
    models = build_models(neg_to_pos)

    # Stratified K-Fold preserves class ratio in each CV fold
    # (critical for imbalanced data — random folds could end up with no fraud)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = []
    Path("models").mkdir(exist_ok=True)

    for name, model in models.items():
        print("\n" + "=" * 60)
        print(f"Training: {name}")
        print("=" * 60)

        # ---- Cross-validation on training set ----
        # 'average_precision' is sklearn's name for PR-AUC
        print(f"Running {CV_FOLDS}-fold CV (scoring=PR-AUC)...")
        cv_start = time.time()
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring="average_precision",
            n_jobs=-1,
        )
        cv_elapsed = time.time() - cv_start
        print(f"CV PR-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"CV time:   {cv_elapsed:.1f}s")

        # ---- Train on full train set and evaluate on test ----
        print("Training on full train set...")
        train_start = time.time()
        model.fit(X_train, y_train)
        train_elapsed = time.time() - train_start
        print(f"Training time: {train_elapsed:.1f}s")

        test_metrics = evaluate_on_test(model, X_test, y_test, name)
        print(f"Test PR-AUC: {test_metrics['test_pr_auc']:.4f}")
        print(f"Test precision: {test_metrics['test_precision']:.4f} "
              f"recall: {test_metrics['test_recall']:.4f}")
        print(f"Confusion: TP={test_metrics['tp']}  FP={test_metrics['fp']}  "
              f"FN={test_metrics['fn']}  TN={test_metrics['tn']:,}")

        # Save the trained model
        model_path = Path("models") / f"{name.lower()}.pkl"
        joblib.dump(model, model_path)
        print(f"Saved to {model_path}")

        results.append({
            "model": name,
            "cv_pr_auc_mean": cv_scores.mean(),
            "cv_pr_auc_std": cv_scores.std(),
            "test_pr_auc": test_metrics["test_pr_auc"],
            "test_precision": test_metrics["test_precision"],
            "test_recall": test_metrics["test_recall"],
            "test_f1": test_metrics["test_f1"],
            "train_time_s": train_elapsed,
            "tp": test_metrics["tp"],
            "fp": test_metrics["fp"],
            "fn": test_metrics["fn"],
        })

    # ---- Final comparison table ----
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("test_pr_auc", ascending=False)

    # Print main comparison (PR-AUC focused)
    print("\nRanked by Test PR-AUC:\n")
    main_table = df_results[[
        "model", "cv_pr_auc_mean", "cv_pr_auc_std",
        "test_pr_auc", "test_precision", "test_recall",
        "train_time_s"
    ]].copy()
    main_table.columns = [
        "Model", "CV PR-AUC (mean)", "CV PR-AUC (std)",
        "Test PR-AUC", "Test Precision", "Test Recall", "Train Time (s)"
    ]
    print(main_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Print confusion matrix counts for context
    print("\nConfusion matrix counts (test set):\n")
    cm_table = df_results[["model", "tp", "fp", "fn"]].copy()
    cm_table.columns = ["Model", "Fraud Caught (TP)", "False Alarms (FP)", "Fraud Missed (FN)"]
    print(cm_table.to_string(index=False))

    # Save table for the README
    df_results.to_csv(Path("models") / "comparison.csv", index=False)
    print("\nSaved comparison table to: models/comparison.csv")

    return df_results


if __name__ == "__main__":
    df_results = run_comparison()

    # Pick winner
    winner = df_results.iloc[0]
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print(f"\nBest model by test PR-AUC: {winner['model']}")
    print(f"  Test PR-AUC:    {winner['test_pr_auc']:.4f}")
    print(f"  Test precision: {winner['test_precision']:.4f}")
    print(f"  Test recall:    {winner['test_recall']:.4f}")
    print(f"  False alarms:   {winner['fp']} per test set")
    print("\nNote: For an automated system, also consider precision and false "
          "alarm count — not just PR-AUC. Threshold tuning will follow.")
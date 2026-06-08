"""
Hyperparameter tuning for XGBoost on credit card fraud detection.

Uses Optuna with TPE (Tree-structured Parzen Estimator) for Bayesian search.
Optimizes for PR-AUC via 5-fold stratified cross-validation. Each trial
proposes a new hyperparameter combination, and Optuna learns from results
to focus on promising regions of the search space.

Run from the project root with: python src/models/tuning.py
"""

import json
import time
from pathlib import Path

import joblib
import optuna
import pandas as pd
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
N_TRIALS = 30


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
    neg_to_pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    return X_train, X_test, y_train, y_test, neg_to_pos_ratio


def make_objective(X_train, y_train, neg_to_pos_ratio):
    """
    Create the Optuna objective function.

    The objective is what Optuna minimizes/maximizes. Each trial samples a
    set of hyperparameters from the search space, trains XGBoost with them
    using 5-fold CV, and returns the mean PR-AUC score.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        # Define the search space — these ranges are reasonable for fraud detection
        # but you could narrow or widen them based on domain knowledge.
        params = {
            # n_estimators: more trees = stronger model but slower; 100-500 is typical
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),

            # max_depth: deeper trees capture more interactions but risk overfitting
            "max_depth": trial.suggest_int("max_depth", 3, 10),

            # learning_rate: smaller = slower learning, often better generalization
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),

            # subsample: fraction of rows per tree — adds randomness, reduces overfit
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),

            # colsample_bytree: fraction of features per tree — similar regularization
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),

            # min_child_weight: minimum samples per leaf — higher = more regularized
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),

            # gamma: minimum loss reduction to split — higher = more conservative
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),

            # reg_alpha (L1): sparsity regularization on weights
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),

            # reg_lambda (L2): standard weight regularization
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        # Fixed parameters that should not be tuned
        fixed_params = {
            "scale_pos_weight": neg_to_pos_ratio,
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
            "verbosity": 0,
        }

        model = XGBClassifier(**params, **fixed_params)

        # Run 5-fold CV using PR-AUC as the score
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring="average_precision",
            n_jobs=-1,
        )
        return scores.mean()  # Maximize this

    return objective


def log_trial_callback(study: optuna.Study, trial: optuna.Trial) -> None:
    """Print a one-line summary after each trial."""
    print(f"Trial {trial.number:2d}: PR-AUC={trial.value:.4f} | "
          f"best so far={study.best_value:.4f}")


def train_final_model(X_train, y_train, best_params, neg_to_pos_ratio):
    """Retrain XGBoost with the winning hyperparameters on the full train set."""
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


def evaluate_final_model(model, X_test, y_test) -> dict:
    """Evaluate the tuned model on the held-out test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "pr_auc":    average_precision_score(y_test, y_proba),
        "roc_auc":   roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    return metrics


if __name__ == "__main__":
    print("=" * 60)
    print("HYPERPARAMETER TUNING (XGBoost via Optuna)")
    print("=" * 60 + "\n")

    # ---- Load data ----
    X_train, X_test, y_train, y_test, neg_to_pos = load_and_split()
    print(f"Train: {X_train.shape[0]:,} rows ({y_train.sum()} fraud)")
    print(f"Test:  {X_test.shape[0]:,} rows ({y_test.sum()} fraud)")
    print(f"Imbalance ratio: {neg_to_pos:.1f}\n")

    # ---- Run Optuna study ----
    print(f"Running {N_TRIALS} trials with {CV_FOLDS}-fold CV...\n")

    # Suppress Optuna's verbose default logging — we use our callback instead
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # 'maximize' because higher PR-AUC is better
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )

    t0 = time.time()
    objective = make_objective(X_train, y_train, neg_to_pos)
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[log_trial_callback])
    tuning_elapsed = time.time() - t0

    # ---- Report best trial ----
    print("\n" + "=" * 60)
    print("BEST TRIAL")
    print("=" * 60)
    print(f"Trial #{study.best_trial.number}")
    print(f"CV PR-AUC: {study.best_value:.4f}")
    print(f"Total tuning time: {tuning_elapsed:.1f}s "
          f"({tuning_elapsed/60:.1f} minutes)")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key:20s} = {value:.6f}")
        else:
            print(f"  {key:20s} = {value}")

    # ---- Save best params to JSON ----
    Path("models").mkdir(exist_ok=True)
    params_path = Path("models") / "best_params.json"
    with open(params_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\nSaved best params to: {params_path}")

    # ---- Train final model with best params ----
    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL")
    print("=" * 60)
    t_final = time.time()
    final_model = train_final_model(X_train, y_train, study.best_params, neg_to_pos)
    final_train_time = time.time() - t_final
    print(f"Final training time: {final_train_time:.2f}s")

    # ---- Evaluate on test set ----
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)
    metrics = evaluate_final_model(final_model, X_test, y_test)
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 score:  {metrics['f1']:.4f}")
    print("\nConfusion matrix:")
    print(f"  Fraud caught (TP):     {metrics['tp']}")
    print(f"  False alarms (FP):     {metrics['fp']}")
    print(f"  Fraud missed (FN):     {metrics['fn']}")
    print(f"  Legit correct (TN):    {metrics['tn']:,}")

    # ---- Compare to untuned XGBoost ----
    print("\n" + "=" * 60)
    print("IMPROVEMENT VS. UNTUNED XGBOOST")
    print("=" * 60)
    print("Untuned XGBoost test PR-AUC: 0.8055")
    print(f"Tuned XGBoost test PR-AUC:   {metrics['pr_auc']:.4f}")
    delta = metrics['pr_auc'] - 0.8055
    print(f"Change: {delta:+.4f}")
    if delta > 0:
        print("  → Tuning helped.")
    elif delta < -0.02:
        print("  → Tuning hurt — possible overfitting on CV folds.")
    else:
        print("  → Tuning made no meaningful difference.")

    # ---- Save tuned model ----
    model_path = Path("models") / "tuned_model.pkl"
    joblib.dump(final_model, model_path)
    print(f"\nSaved tuned model to: {model_path}")
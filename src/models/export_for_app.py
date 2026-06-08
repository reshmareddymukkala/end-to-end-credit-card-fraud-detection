"""
Export model artifacts that the Streamlit app needs to display:
  1. data/predictions.csv      — sample test-set predictions + probabilities
  2. data/model_results.json   — comparison metrics across models

Run from the project root with: python src/models/export_for_app.py
"""

import json

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "Class"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def main():
    # ---- Load data and recreate the same split used during training ----
    df = pd.read_csv("data/features.csv")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # ---- Load the production model and make predictions on test set ----
    production_model = joblib.load("models/production_model.pkl")
    y_pred = production_model.predict(X_test)
    y_proba = production_model.predict_proba(X_test)[:, 1]

    # ---- Save predictions ----
    predictions = X_test.copy()
    predictions["true_class"] = y_test.values
    predictions["predicted_class"] = y_pred
    predictions["fraud_probability"] = y_proba
    predictions.to_csv("data/predictions.csv", index=False)
    print(f"Saved data/predictions.csv ({len(predictions)} rows)")

    # ---- Recompute metrics for each saved model for the results file ----
    results = {}

    # Baseline
    baseline = joblib.load("models/baseline.pkl")
    y_pred_b = baseline.predict(X_test)
    y_proba_b = baseline.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred_b).ravel()
    results["baseline"] = {
        "name": "Logistic Regression (Baseline)",
        "pr_auc": float(average_precision_score(y_test, y_proba_b)),
        "roc_auc": float(roc_auc_score(y_test, y_proba_b)),
        "precision": float(precision_score(y_test, y_pred_b, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_b, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred_b, zero_division=0)),
        "tn": int(cm[0]), "fp": int(cm[1]), "fn": int(cm[2]), "tp": int(cm[3]),
    }

    # Random Forest
    rf = joblib.load("models/randomforest.pkl")
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred_rf).ravel()
    results["random_forest"] = {
        "name": "Random Forest",
        "pr_auc": float(average_precision_score(y_test, y_proba_rf)),
        "roc_auc": float(roc_auc_score(y_test, y_proba_rf)),
        "precision": float(precision_score(y_test, y_pred_rf, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_rf, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred_rf, zero_division=0)),
        "tn": int(cm[0]), "fp": int(cm[1]), "fn": int(cm[2]), "tp": int(cm[3]),
    }

    # XGBoost (untuned)
    xgb = joblib.load("models/xgboost.pkl")
    y_pred_xgb = xgb.predict(X_test)
    y_proba_xgb = xgb.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred_xgb).ravel()
    results["xgboost_untuned"] = {
        "name": "XGBoost (Untuned)",
        "pr_auc": float(average_precision_score(y_test, y_proba_xgb)),
        "roc_auc": float(roc_auc_score(y_test, y_proba_xgb)),
        "precision": float(precision_score(y_test, y_pred_xgb, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_xgb, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred_xgb, zero_division=0)),
        "tn": int(cm[0]), "fp": int(cm[1]), "fn": int(cm[2]), "tp": int(cm[3]),
    }

    # Tuned XGBoost (production winner)
    cm = confusion_matrix(y_test, y_pred).ravel()
    results["xgboost_tuned"] = {
        "name": "XGBoost (Tuned) ★ WINNER",
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "tn": int(cm[0]), "fp": int(cm[1]), "fn": int(cm[2]), "tp": int(cm[3]),
    }

    # ---- Feature importance from the production model ----
    importances = production_model.feature_importances_
    feature_names = list(X_test.columns)
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    results["feature_importance"] = fi_df.to_dict(orient="records")

    # ---- Save the results file ----
    with open("data/model_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved data/model_results.json")


if __name__ == "__main__":
    main()
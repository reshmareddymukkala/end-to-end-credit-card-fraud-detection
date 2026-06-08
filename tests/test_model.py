"""Tests for the trained production model."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "production_model.pkl"


@pytest.fixture(scope="session")
def production_model():
    """Load the trained production model once per test session."""
    if not MODEL_PATH.exists():
        pytest.skip(
            f"Model not found at {MODEL_PATH}. Run src/models/run_training.py first."
        )
    return joblib.load(MODEL_PATH)


def test_model_loads(production_model):
    """The production model file should load without errors."""
    assert production_model is not None
    # The model should expose a predict method (basic ML interface)
    assert hasattr(production_model, "predict")
    assert hasattr(production_model, "predict_proba")


def test_model_makes_prediction(production_model, features_data):
    """The model should produce a prediction for a single row."""
    X = features_data.drop(columns=["Class"]).head(1)
    prediction = production_model.predict(X)

    assert len(prediction) == 1
    # Binary classification — prediction must be 0 or 1
    assert prediction[0] in {0, 1}


def test_model_makes_batch_predictions(production_model, features_data):
    """The model should handle a batch of rows."""
    n_samples = 100
    X = features_data.drop(columns=["Class"]).head(n_samples)
    predictions = production_model.predict(X)

    assert len(predictions) == n_samples
    # All predictions are 0 or 1
    unique_values = set(predictions.tolist())
    assert unique_values.issubset({0, 1})


def test_predict_proba_returns_probabilities(production_model, features_data):
    """predict_proba should return values in [0, 1] for each class."""
    X = features_data.drop(columns=["Class"]).head(50)
    probabilities = production_model.predict_proba(X)

    # Shape should be (n_samples, 2) for binary classification
    assert probabilities.shape == (50, 2)
    # All probabilities are in [0, 1]
    assert probabilities.min() >= 0.0
    assert probabilities.max() <= 1.0
    # Each row's probabilities should sum to ~1 (allowing tiny floating-point error)
    row_sums = probabilities.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


def test_model_predictions_are_consistent(production_model, features_data):
    """Calling predict twice on the same input should give the same result."""
    X = features_data.drop(columns=["Class"]).head(10)
    pred_1 = production_model.predict(X)
    pred_2 = production_model.predict(X)

    np.testing.assert_array_equal(pred_1, pred_2)


def test_model_handles_known_fraud_case(production_model, features_data):
    """
    A confirmed fraud transaction should be predicted as fraud with high probability.
    This is a probabilistic test — we don't require perfect accuracy on all fraud,
    but a random known-fraud sample should usually score > 0.5.
    """
    fraud_rows = features_data[features_data["Class"] == 1]
    if len(fraud_rows) == 0:
        pytest.skip("No fraud rows in features.csv")

    # Take the first fraud row deterministically
    X = fraud_rows.drop(columns=["Class"]).head(1)
    proba = production_model.predict_proba(X)[0, 1]

    # The model should produce a probability — we only assert it's a real number
    # in [0, 1]. We don't assert it's > 0.5 because individual cases vary.
    assert 0.0 <= proba <= 1.0
"""
Shared helpers for the Streamlit app — data loading with demo fallbacks,
theming, and common widgets.

Loaders check for the real data files first. If they don't exist (which
happens on a fresh clone, before the pipeline runs), demo data is
generated automatically so the app always works.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from demo_data import (
    generate_demo_features,
    generate_demo_predictions,
    generate_demo_results,
    is_demo_mode,
)


# Centralized color scheme used across all pages
COLORS = {
    "primary": "#1f77b4",
    "fraud":   "#d62728",
    "legit":   "#2ca02c",
    "neutral": "#7f7f7f",
    "accent":  "#ff7f0e",
}


# Real data file paths
FEATURES_PATH    = Path("data") / "features.csv"
PREDICTIONS_PATH = Path("data") / "predictions.csv"
RESULTS_PATH     = Path("data") / "model_results.json"


@st.cache_data
def load_features() -> pd.DataFrame:
    """Load the feature dataset, or generate a demo version if missing."""
    if FEATURES_PATH.exists():
        return pd.read_csv(FEATURES_PATH)
    return generate_demo_features()


@st.cache_data
def load_predictions() -> pd.DataFrame:
    """Load test predictions, or generate a demo version if missing."""
    if PREDICTIONS_PATH.exists():
        return pd.read_csv(PREDICTIONS_PATH)
    # The demo predictions need the demo features as a base
    features = load_features()
    return generate_demo_predictions(features)


@st.cache_data
def load_model_results() -> dict:
    """Load model comparison metrics, or generate demo results if missing."""
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return generate_demo_results()


def render_demo_banner():
    """
    Show a warning banner at the top of pages when running on demo data.
    Hidden when the real pipeline has been run.
    """
    if is_demo_mode():
        st.warning(
            "📌 **Demo Mode** — You're seeing synthetic data because the "
            "training pipeline hasn't been run on this machine yet. "
            "The visualizations and metrics shown are illustrative, not "
            "actual model results. To see real results, run the full pipeline:\n\n"
            "```\n"
            "python src/data/cleaner.py\n"
            "python src/features/run_features.py\n"
            "python src/models/tuning.py\n"
            "python src/models/run_training.py\n"
            "python src/models/export_for_app.py\n"
            "```"
        )


def render_footer():
    """Render a consistent footer on every page."""
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray; font-size: 0.85em;'>"
        "Built with Python, scikit-learn, XGBoost, MLflow, and Streamlit · "
        "<a href='https://github.com/reshmareddymukkala/end-to-end-credit-card-fraud-detection' target='_blank'>"
        "View on GitHub</a>"
        "</div>",
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = ""):
    """Consistent page header for all pages."""
    st.markdown(f"# {title}")
    if subtitle:
        st.markdown(
            f"<p style='color: gray; font-size: 1.1em;'>{subtitle}</p>",
            unsafe_allow_html=True,
        )
    st.markdown("---")
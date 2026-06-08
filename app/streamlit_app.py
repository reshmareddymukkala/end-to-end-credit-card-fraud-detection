"""
Credit Card Fraud Detection Portfolio Showcase
Main landing page (Project Overview).
"""

import streamlit as st

from utils import (
    COLORS,
    load_features,
    load_model_results,
    render_demo_banner,
    render_footer,
)


# ---------- Page configuration ----------
st.set_page_config(
    page_title="Fraud Detection Portfolio",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Sidebar branding ----------
with st.sidebar:
    st.markdown("# Fraud Detection")
    st.markdown("*An end-to-end ML portfolio project*")
    st.markdown("---")
    st.markdown(
        "**Author:** Reshma Reddy Mukkala  \n"
        "**Stack:** Python, scikit-learn, XGBoost, MLflow, Streamlit"
    )

render_demo_banner()
# ---------- Hero section ----------
st.markdown(
    """
    <div style="
        padding: 2rem 0 1rem 0;
        text-align: center;
    ">
        <h1 style="font-size: 3em; margin-bottom: 0.2em;">
            Credit Card Fraud Detection
        </h1>
        <p style="font-size: 1.3em; color: #555; margin-top: 0;">
            An end-to-end machine learning system for detecting
            fraudulent transactions in real time.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- What this project does ----------
st.markdown("## What this project does")
st.markdown(
    """
    This project builds a production grade machine learning pipeline that
    identifies fraudulent credit card transactions in a highly imbalanced
    dataset (only **0.17% of transactions are fraud**). The system covers
    every stage of the ML lifecycle: data validation, exploratory analysis,
    feature engineering, model selection, hyperparameter tuning and
    deployment via an interactive web application.
    """
)


# ---------- KPI cards ----------
st.markdown("## Project at a Glance")

results = load_model_results()
features_df = load_features()

baseline_pr = results["baseline"]["pr_auc"]
winner_pr = results["xgboost_tuned"]["pr_auc"]
improvement = ((winner_pr - baseline_pr) / baseline_pr) * 100

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label="Transactions Analyzed",
        value=f"{len(features_df):,}",
        help="Total credit card transactions in the cleaned dataset",
    )
with col2:
    n_engineered = sum(1 for c in features_df.columns if c not in [
        "Time", "Amount", "Class"
    ] and not c.startswith("V"))
    st.metric(
        label="Engineered Features",
        value=f"{n_engineered}",
        help="Custom features created from raw inputs across 3 categories",
    )
with col3:
    st.metric(
        label="Final Model PR-AUC",
        value=f"{winner_pr:.3f}",
        help="Precision-Recall AUC the key metric for imbalanced data",
    )
with col4:
    st.metric(
        label="Improvement over Baseline",
        value=f"+{improvement:.0f}%",
        help="Relative improvement from logistic regression to tuned XGBoost",
    )


# ---------- The Winner: One-line summary ----------
st.markdown("## The Outcome")
winner = results["xgboost_tuned"]
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(
        f"""
        The final model is a **hyperparameter-tuned XGBoost classifier** It
        catches **{winner["tp"]} of {winner["tp"] + winner["fn"]}**
        fraudulent transactions in a held-out test set, with just
        **{winner["fp"]} false alarms** out of 56,746 transactions reviewed.
        This represents a precision of **{winner["precision"]:.1%}** -
        meaning when the system flags a transaction, it's right
        **{winner["precision"]:.1%}** of the time.
        """
    )
with col2:
    st.success(
        f"**PR-AUC: {winner['pr_auc']:.3f}**  \n"
        f"Precision: {winner['precision']:.1%}  \n"
        f"Recall: {winner['recall']:.1%}"
    )


# ---------- Tech stack ----------
st.markdown("## Tech Stack")

tech_stack = [
    ("Python 3.13", "Language"),
    ("pandas, NumPy", "Data manipulation"),
    ("scikit-learn", "ML pipeline & baseline"),
    ("XGBoost", "Production model"),
    ("Optuna", "Hyperparameter tuning"),
    ("MLflow", "Experiment tracking"),
    ("Plotly", "Interactive visualizations"),
    ("Streamlit", "Web deployment"),
    ("Git", "Version control"),
]

cols = st.columns(3)
for i, (tech, role) in enumerate(tech_stack):
    with cols[i % 3]:
        st.markdown(
            f"""
            <div style='
                background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
                padding: 1rem;
                border-radius: 8px;
                border-left: 4px solid {COLORS["primary"]};
                margin-bottom: 0.5rem;
            '>
                <strong>{tech}</strong><br>
                <span style='color: #666; font-size: 0.9em;'>{role}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------- Call to action ----------
st.markdown("## Explore the Project")
st.info(
    "Use the sidebar to navigate through the project: "
    "explore the data, see model comparisons and try the live prediction tool."
)


render_footer()
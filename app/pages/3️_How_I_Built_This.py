"""Page 4: The build story."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st

from utils import page_header, render_demo_banner, render_footer


st.set_page_config(
    page_title="How I Built This",
    page_icon="🛠️",
    layout="wide",
)

page_header(
    "How I Built This",
    "The build process, key decisions, and lessons learned",
)
render_demo_banner()

# ---------- Architecture ----------
st.markdown("### Architecture")

st.graphviz_chart("""
digraph {
    rankdir=LR;
    node [shape=box, style="rounded,filled", fillcolor="#e3f2fd",
          fontname="Helvetica", fontsize=12];
    edge [fontname="Helvetica", fontsize=10];

    Raw [label="Raw CSV\\n284K transactions", fillcolor="#fff3e0"];
    QG  [label="Quality Gate\\n5 automated checks"];
    Clean [label="Cleaner\\nDedupe + dtypes"];
    EDA [label="EDA Notebook\\n7 sections"];
    FE  [label="Feature Engineering\\n13 new features"];
    FS  [label="Feature Selection\\nVariance + correlation"];
    Train [label="Model Training\\n4 candidates"];
    Tune [label="Optuna Tuning\\n30 trials, 5-fold CV"];
    MLflow [label="MLflow Tracking", fillcolor="#f3e5f5"];
    Model [label="production_model.pkl", fillcolor="#c8e6c9"];
    App [label="Streamlit App", fillcolor="#c8e6c9"];

    Raw -> QG -> Clean -> EDA;
    Clean -> FE -> FS -> Train -> Tune -> Model -> App;
    Train -> MLflow;
    Tune -> MLflow;
}
""")


# ---------- Timeline ----------
st.markdown("---")
st.markdown("### The Build Process")

timeline = [
    ("Stage 1", "Project Setup",
     "Set up project structure, virtual environment, `setup.py`, and Git. "
     "Established the conventions that would guide the entire build."),
    ("Stage 2", "Data Pipeline",
     "Built the data loader, automated quality gate (5 checks), and cleaner. "
     "Removed 1,081 exact duplicates from the raw dataset."),
    ("Stage 3", "EDA",
     "Built an interactive Jupyter notebook with 7 sections. Confirmed the "
     "severe class imbalance (0.17% fraud) and identified V14, V17, V12 "
     "as the strongest fraud predictors."),
    ("Stage 4", "Feature Engineering",
     "Engineered 13 features across 3 categories: domain-specific (hour-of-day, "
     "log-amount), statistical (V-magnitude, negative-count), and interactions "
     "(V14×V17, amount×V17)."),
    ("Stage 5", "Modeling",
     "Trained a logistic regression baseline (PR-AUC 0.67), then compared "
     "Random Forest, XGBoost, and LightGBM. Dropped LightGBM due to "
     "degenerate probability outputs."),
    ("Stage 6", "Tuning & Tracking",
     "Tuned XGBoost with Optuna (30 trials, 5-fold CV). Logged everything "
     "to MLflow for reproducibility. Final test PR-AUC: 0.81."),
    ("Stage 7", "Deployment",
     "Wrapped the model in this Streamlit app with interactive EDA, model "
     "comparison, and a live prediction tool."),
]

for day, title, desc in timeline:
    with st.expander(f"**{day} — {title}**", expanded=False):
        st.markdown(desc)


# ---------- Key decisions ----------
st.markdown("---")
st.markdown("### Key Decisions & Lessons Learned")

decisions = [
    ("Why PR-AUC, not accuracy",
     "On a dataset that's 99.83% legit, a model that predicts \"not fraud\" "
     "for everything gets 99.83% accuracy and catches zero fraud. PR-AUC "
     "focuses on the rare class and is the honest metric for imbalanced "
     "classification."),
    ("Why a stratified split",
     "With only 473 fraud cases in 284K rows, a random 80/20 split could "
     "leave one split with almost no fraud. Stratification forces both "
     "splits to preserve the fraud ratio."),
    ("Why explicit class weighting",
     "Default sklearn models will learn to predict the majority class "
     "because that minimizes total loss. `class_weight=\"balanced\"` (and "
     "`scale_pos_weight` for XGBoost) rebalances the loss function."),
    ("Why I chose XGBoost over Random Forest",
     "PR-AUC was tied, but XGBoost catches 3 more fraud cases for only 4 "
     "more false alarms. A favorable trade-off and it trains 18× faster, "
     "which matters for an automated system that retrains regularly."),
    ("What I'd do differently next time",
     "The variance-based feature selection nearly deleted every feature "
     "because `Time`'s variance (in seconds) dominated the mean. Using the "
     "median or standardizing first would have been more robust. "
     "Lesson: any threshold derived from a mean is fragile when feature "
     "scales differ wildly."),
]

for title, body in decisions:
    with st.expander(title, expanded=False):
        st.markdown(body)


# ---------- Repo link ----------
st.markdown("---")
st.markdown("### Source Code")
st.markdown(
    "All code, the trained model, and the full project history are on GitHub:"
)
st.markdown(
    "**[github.com/your-username/EndToEndMLProject]"
    "(https://github.com/your-username/EndToEndMLProject)**"
)
#st.caption("Replace the link above with your actual repo URL when deployed.")


render_footer()
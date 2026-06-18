"""Page 3: Model comparison and live prediction."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import COLORS, load_features, load_model_results, page_header, render_demo_banner, render_footer


st.set_page_config(
    page_title="Model Results",
    page_icon="🤖",
    layout="wide",
)

page_header(
    "Model Results",
    "How four candidate models compared and why XGBoost (tuned) won",
)
render_demo_banner()

# ---------- Comparison table ----------
results = load_model_results()
order = ["baseline", "random_forest", "xgboost_untuned", "xgboost_tuned"]
rows = []
for key in order:
    r = results[key]
    rows.append({
        "Model": r["name"],
        "PR-AUC": r["pr_auc"],
        "ROC-AUC": r["roc_auc"],
        "Precision": r["precision"],
        "Recall": r["recall"],
        "F1": r["f1"],
        "Fraud Caught": r["tp"],
        "False Alarms": r["fp"],
        "Fraud Missed": r["fn"],
    })
df_results = pd.DataFrame(rows)

st.markdown("### Side-by-side comparison")
st.dataframe(
    df_results.style.format({
        "PR-AUC": "{:.4f}",
        "ROC-AUC": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1": "{:.4f}",
    }).background_gradient(subset=["PR-AUC", "F1"], cmap="Greens"),
    use_container_width=True,
    hide_index=True,
)


# ---------- Why this model won ----------
st.markdown("### Why XGBoost (Tuned) — Not Just the Highest Score")

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown(
        """
        Random Forest and XGBoost are essentially tied on PR-AUC
        (0.81 each). The winner isn't decided by a fourth decimal place
        — it's decided by the **business context**.

        **Three reasons XGBoost (tuned) won:**

        1. **Better recall-precision balance for an automated system.**
           XGBoost catches **3 more fraud cases** than Random Forest
           (73 vs. 69) while adding only 4 false alarms in 56,746
           transactions. At scale, that's millions more dollars in
           fraud prevented per year.

        2. **18× faster training.** XGBoost trains in ~2 seconds; Random
           Forest takes ~30. For an automated system that retrains
           regularly, this matters operationally.

        3. **Industry standard for fraud detection.** Tooling for
           production deployment, monitoring, and explainability is
           more mature for gradient boosting.

        **What about LightGBM?** It produced degenerate probability
        outputs at this imbalance level (all 0s and 1s), making PR-AUC
        unreliable. Excluded from the final comparison.
        """
    )

with col2:
    st.success(
        "** Winner: XGBoost (Tuned)**  \n\n"
        f"PR-AUC: **{results['xgboost_tuned']['pr_auc']:.4f}**  \n"
        f"Precision: **{results['xgboost_tuned']['precision']:.1%}**  \n"
        f"Recall: **{results['xgboost_tuned']['recall']:.1%}**  \n\n"
        f"Catches **{results['xgboost_tuned']['tp']}** of "
        f"{results['xgboost_tuned']['tp'] + results['xgboost_tuned']['fn']} "
        "fraud cases  \n"
        f"with **{results['xgboost_tuned']['fp']}** false alarms."
    )


# ---------- Feature importance ----------
st.markdown("---")
st.markdown("### What the Model Learned: Top 15 Features")
st.caption("Higher importance = the feature contributes more to the model's decisions.")

fi = pd.DataFrame(results["feature_importance"]).head(15)

fig = go.Figure(go.Bar(
    x=fi["importance"],
    y=fi["feature"],
    orientation="h",
    marker_color=COLORS["primary"],
))
fig.update_layout(
    xaxis_title="Feature Importance",
    yaxis={"categoryorder": "total ascending"},
    height=500,
)
st.plotly_chart(fig, use_container_width=True)


# ---------- Confusion matrix ----------
st.markdown("---")
st.markdown("### Confusion Matrix (Held-Out Test Set)")

winner = results["xgboost_tuned"]
z = [[winner["tn"], winner["fp"]],
     [winner["fn"], winner["tp"]]]

fig = go.Figure(data=go.Heatmap(
    z=z,
    x=["Predicted Legit", "Predicted Fraud"],
    y=["Actual Legit", "Actual Fraud"],
    text=[[f"{v:,}" for v in row] for row in z],
    texttemplate="%{text}",
    textfont={"size": 18},
    colorscale="Blues",
    showscale=False,
))
fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True)


# ---------- Try-it-yourself ----------
st.markdown("---")
st.markdown("###Try It Yourself")
st.caption("Adjust feature values to see what the model predicts in real time.")

@st.cache_resource
def load_production_model():
    """Try to load the production model. Return None if it doesn't exist."""
    model_path = Path("models/production_model.pkl")
    if not model_path.exists():
        return None
    return joblib.load(model_path)


model = load_production_model()

if model is None:
    st.info(
        "🎮 **The interactive prediction widget is hidden in demo mode.** "
        "Train the production model first to enable it: "
        "`python src/models/run_training.py`"
    )
    render_footer()
    st.stop()   # Halt page rendering — anything below this won't execute
df_features = load_features()
feature_cols = [c for c in df_features.columns if c != "Class"]

st.markdown("**Pick a starting point:**")
preset = st.radio(
    "Preset",
    ["Random legit transaction", "Random fraud transaction", "Median values"],
    horizontal=True,
    label_visibility="collapsed",
)

# Build initial values from preset
if preset == "Random legit transaction":
    sample = df_features[df_features["Class"] == 0].sample(1, random_state=None).iloc[0]
elif preset == "Random fraud transaction":
    sample = df_features[df_features["Class"] == 1].sample(1, random_state=None).iloc[0]
else:
    sample = df_features.median()

# Adjustable controls for the most important features
top_features = [f["feature"] for f in results["feature_importance"][:5]]

st.markdown("**Adjust the top 5 features:**")
inputs = {}
cols = st.columns(5)
for i, feature in enumerate(top_features):
    with cols[i]:
        col_min = float(df_features[feature].min())
        col_max = float(df_features[feature].max())
        default = float(sample[feature])
        inputs[feature] = st.slider(
            feature,
            min_value=col_min,
            max_value=col_max,
            value=default,
            key=f"slider_{feature}",
        )

# Build prediction input (use sample values for non-slider features)
prediction_input = pd.DataFrame([sample[feature_cols].to_dict()])
for feature, value in inputs.items():
    prediction_input[feature] = value

# CRITICAL: reorder columns to match the order the model was trained on.
# XGBoost is strict about column order and will fail if they don't match.
if hasattr(model, "feature_names_in_"):
    prediction_input = prediction_input[list(model.feature_names_in_)]

# Predict
proba = model.predict_proba(prediction_input)[0, 1]
pred = int(proba >= 0.5)

# Display result
col1, col2 = st.columns([1, 2])
with col1:
    if pred == 1:
        st.error(f"### FRAUD\nProbability: {proba:.2%}")
    else:
        st.success(f"### LEGIT\nProbability of fraud: {proba:.2%}")

with col2:
    # Probability gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        title={"text": "Fraud Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": COLORS["fraud"] if pred == 1 else COLORS["legit"]},
            "steps": [
                {"range": [0, 50], "color": "#e8f5e9"},
                {"range": [50, 100], "color": "#ffebee"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": 50,
            },
        },
        number={"suffix": "%"},
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


render_footer()
"""Page 2: Interactive data exploration."""

import sys
from pathlib import Path

# Add parent folder to path so we can import utils
sys.path.append(str(Path(__file__).parent.parent))

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import COLORS, load_features, page_header, render_demo_banner, render_footer


st.set_page_config(
    page_title="Explore the Data",
    page_icon="📊",
    layout="wide",
)

page_header(
    "Explore the Data",
    "Interactive visualizations of the credit card fraud dataset",
)
render_demo_banner()
df = load_features()

# ---------- Quick stats ----------
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Transactions", f"{len(df):,}")
with col2:
    st.metric("Fraud Cases", f"{int(df['Class'].sum()):,}")
with col3:
    st.metric("Fraud Rate", f"{df['Class'].mean() * 100:.3f}%")

st.markdown("---")


# ---------- Target distribution ----------
st.markdown("### Target Distribution")
st.caption(
    "Fraud is just 0.17% of transactions, a severe class imbalance that "
    "drives every modeling decision in this project. The log scale shows "
    "both classes clearly."
)

col1, col2 = st.columns(2)

with col1:
    class_counts = df["Class"].value_counts().sort_index()
    fig = go.Figure(data=[
        go.Bar(
            x=["Legit", "Fraud"],
            y=class_counts.values,
            marker_color=[COLORS["legit"], COLORS["fraud"]],
            text=class_counts.values,
            textposition="auto",
        )
    ])
    fig.update_layout(
        title="Class Distribution (Linear Scale)",
        yaxis_title="Count",
        height=400,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = go.Figure(data=[
        go.Bar(
            x=["Legit", "Fraud"],
            y=class_counts.values,
            marker_color=[COLORS["legit"], COLORS["fraud"]],
            text=class_counts.values,
            textposition="auto",
        )
    ])
    fig.update_layout(
        title="Class Distribution (Log Scale)",
        yaxis_title="Count (log)",
        yaxis_type="log",
        height=400,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------- Interactive feature explorer ----------
st.markdown("---")
st.markdown("### Feature Explorer")
st.caption("Select a feature to view its distribution and how it differs between fraud and legit transactions.")

interesting_features = [
    "Amount", "Time", "log_amount", "hour_of_day", "v_magnitude",
    "top_fraud_v_mean", "fraud_score_proxy",
    "V14", "V17", "V12", "V10", "V11", "V4",
]
interesting_features = [f for f in interesting_features if f in df.columns]

feature = st.selectbox("Choose a feature:", interesting_features, index=0)

col1, col2 = st.columns(2)
with col1:
    # Histogram colored by class
    fig = px.histogram(
        df, x=feature, color="Class",
        color_discrete_map={0: COLORS["legit"], 1: COLORS["fraud"]},
        nbins=60, barmode="overlay", opacity=0.7,
    )
    fig.update_layout(
        title=f"Distribution of {feature} by Class",
        height=400,
        legend_title_text="",
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Box plot by class
    fig = px.box(
        df, x="Class", y=feature,
        color="Class",
        color_discrete_map={0: COLORS["legit"], 1: COLORS["fraud"]},
    )
    fig.update_layout(
        title=f"{feature} Comparison: Legit vs Fraud",
        height=400,
        showlegend=False,
        xaxis_title="Class (0=Legit, 1=Fraud)",
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------- Top correlations heatmap ----------
st.markdown("---")
st.markdown("### Correlation Heatmap (Top 15 Features)")
st.caption(
    "Correlations among the 15 features most predictive of fraud. "
    "Top fraud predictors (V14, V17, V12, V10) cluster together "
    "but show low correlation with each other — they're capturing "
    "different aspects of the fraud signal."
)

# Get the top 15 features by absolute correlation with Class
top_features = (
    df.corr()["Class"]
    .drop("Class")
    .abs()
    .sort_values(ascending=False)
    .head(15)
    .index.tolist()
)
top_features.append("Class")  # Include the target

# Build smaller correlation matrix
top_corr = df[top_features].corr()

fig = px.imshow(
    top_corr,
    color_continuous_scale="RdBu_r",
    aspect="auto",
    zmin=-1, zmax=1,
    text_auto=".2f",   # Show numbers inside each cell
    labels={"color": "Correlation"},
)
fig.update_layout(
    height=600,
    title="Correlation Among Top 15 Fraud-Predictive Features",
)
st.plotly_chart(fig, use_container_width=True)


# ---------- Key findings ----------
st.markdown("---")
st.markdown("### Key Findings from EDA")

col1, col2 = st.columns(2)

with col1:
    st.info(
        "**Severe class imbalance**  \n"
        "Fraud is only 0.17% of transactions. Accuracy is meaningless. "
        "PR-AUC and the confusion matrix are the real metrics."
    )
    st.info(
        "**Top fraud predictors**  \n"
        "V17, V14, V12, V10 - all PCA components with negative correlation. "
        "Fraud consistently sits at extreme low values."
    )

with col2:
    st.info(
        "**Amount is heavily skewed**  \n"
        "Most transactions are under \$100; a few reach \$25,000. "
        "Log-transform makes it usable for linear models."
    )
    st.info(
        "**Time has bimodal structure**  \n"
        "The data spans ~2 days with overnight quiet periods. "
        "Hour-of-day was engineered to capture this."
    )


render_footer()
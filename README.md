# Credit Card Fraud Detection — End-to-End ML Project

An end-to-end machine learning project for detecting fraudulent credit card
transactions. Covers data loading, quality validation, cleaning, exploratory
analysis, feature engineering, modeling, and deployment via FastAPI and
Streamlit.

## Project Structure

EndToEndMLProject/
├── src/
│   ├── data/         # Data loading, quality gates, cleaning
│   ├── features/     # Feature engineering
│   └── models/       # Model training and prediction
├── app/              # FastAPI API and Streamlit dashboard
├── tests/            # Unit tests
├── notebooks/        # EDA and experimentation
├── data/             # Datasets (gitignored)
└── models/           # Trained model files (gitignored)

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install -e .
```

## Exploratory Data Analysis

**Dataset:** 283,726 credit card transactions × 31 features after cleaning
(1,081 duplicates removed). All numeric; binary target `Class` (0 = legit,
1 = fraud). Features are `Time`, `Amount`, and 28 PCA-anonymized components
(`V1`–`V28`).

**Key findings:**

- **Severe class imbalance (~600:1)** - fraud is 0.17% of transactions.
  Accuracy is meaningless.
- **Top fraud predictors are V17, V14, V12, V10** (correlations -0.31,
  -0.29, -0.25, -0.21). Box plots show clear separation - fraud sits at
  -5 to -6 while legit clusters near 0.
- **`Amount` is heavily right-skewed** - most transactions under $100,
  tail to $25,000. Needs a log transform or robust scaler.
- **No missing values, V1-V28 are orthogonal by construction.** No
  multicollinearity concerns among PCA features.

  **Implications for modeling:** Use stratified train/test split, scale
`Amount` (V features already comparable), apply class weights
and evaluate with PR-AUC rather than accuracy or ROC-AUC.
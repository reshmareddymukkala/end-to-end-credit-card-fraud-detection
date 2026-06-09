# 💳 Credit Card Fraud Detection — End-to-End ML Project

> An end-to-end machine learning system for identifying fraudulent credit card transactions in a heavily imbalanced dataset (0.17% fraud rate). Built as a full production pipeline: data validation, EDA, feature engineering, model selection, hyperparameter tuning, experiment tracking, containerization, automated testing and a public Streamlit dashboard.

**Live Demo:** [link coming soon — deployment in progress]
**Repo:** https://github.com/reshmareddymukkala/end-to-end-credit-card-fraud-detection
**MLflow Runs:** Tracked locally with full experiment history (see `models/` artifacts)

[![CI](https://github.com/reshmareddymukkala/end-to-end-credit-card-fraud-detection/workflows/CI/badge.svg)](https://github.com/reshmareddymukkala/end-to-end-credit-card-fraud-detection/actions)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 1. Project Overview

**The problem.** Credit card fraud causes billions of dollars in losses annually. Detection is difficult because fraud is rare and typically less than 0.2% of transactions, so naive classifiers achieve high accuracy by never flagging anything. The challenge is to catch real fraud while keeping false alarms low enough that the system is operationally usable.

**The end user.** An automated transaction-monitoring system that flags suspicious transactions in real time. This shapes the design: false positives have a real cost (a customer's card gets blocked), so the model must achieve **high precision**, not just high recall.

**The data.** [Credit Card Fraud Detection dataset] 284,807 European cardholder transactions over two days in September 2013. Features V1–V28 are anonymized via PCA; only `Time` and `Amount` remain in their original form. The target is binary: `Class = 1` (fraud) or `Class = 0` (legit).

**What the model outputs.** Given a transaction, a probability between 0 and 1 that it is fraudulent. By default, transactions with probability ≥ 0.5 are flagged, but the threshold is tunable based on operational priorities.

**Key design decision.** Because the dataset is 99.83% legit and 0.17% fraud, accuracy is meaningless. A model that always predicts "not fraud" achieves 99.83% accuracy and catches zero fraud. The headline metric throughout this project is **PR-AUC (Precision-Recall AUC)**, which focuses on the minority class and is the industry standard for imbalanced classification. ROC-AUC and accuracy are reported but never used as the primary metric.

---

## 2. Architecture

The pipeline flows from raw data through cleaning, feature engineering, model training and deployment:
┌─────────────────┐
│     CSV         │  284,807 transactions
│  (raw data)     │
└────────┬────────┘
│
▼
┌─────────────────┐
│  Quality Gate   │  5 automated checks (schema, rows,
│  (validation)   │   nulls, ranges, target distribution)
└────────┬────────┘
│
▼
┌─────────────────┐
│  Cleaner        │  Dedupe (-1,081 rows), dtype coercion
└────────┬────────┘
│
▼
┌─────────────────┐
│  Feature        │  13 engineered features across 3 categories:
│  Engineering    │   domain, statistical, interaction
└────────┬────────┘
│
▼
┌─────────────────┐
│  Feature        │  Drop low-variance / highly-correlated features
│  Selection      │   (1 dropped: amount_zero_flag)
└────────┬────────┘
│
▼
┌─────────────────┐      ┌──────────────────┐
│  Model          │─────▶  MLflow          
│  Training       │      │  Experiment      │
│  + Tuning       │      │  Tracking        │
└────────┬────────┘      └──────────────────┘
│
▼
┌─────────────────┐
│  production_    │  Tuned XGBoost (Optuna, 30 trials)
│  model.pkl      │
└────────┬────────┘
│
▼
┌─────────────────┐
│  Streamlit App  │  Multi-page portfolio dashboard
│  (Dockerized)   │   with live prediction widget
└─────────────────┘

---

## 3. Results

Four models were trained and compared on a held-out 20% stratified test set (56,746 transactions, 95 fraud cases). The tuned XGBoost was selected as the production model.

| Model                          | PR-AUC | Precision | Recall | F1     | Fraud Caught | False Alarms | Train Time |
| ------------------------------ | ------ | --------- | ------ | ------ | ------------ | ------------ | ---------- |
| Logistic Regression (Baseline) | 0.673  | 0.055     | 0.863  | 0.104  | 82 / 95      | **1,397**    | < 1s       |
| Random Forest                  | 0.811  | 0.972     | 0.726  | 0.831  | 69 / 95      | 2            | 30.8s      |
| XGBoost (Untuned)              | 0.806  | 0.923     | 0.758  | 0.832  | 72 / 95      | 6            | 1.7s       |
| **XGBoost (Tuned) ★ WINNER**   | **0.810** | **0.936** | **0.768** | **0.844** | **73 / 95** | **5** | 2.5s   |

### Improvement vs. Baseline

The tuned XGBoost reduces false alarms from **1,397 → 5** (a 99.6% reduction) while maintaining comparable recall to the baseline. Precision increased from 5.5% to 93.6% — a 17× improvement that makes the model viable for automated use. PR-AUC improved from 0.673 to 0.810 (+20.4%).

### Why XGBoost (Tuned), Not Random Forest

Random Forest and XGBoost achieved nearly identical PR-AUC (0.811 vs 0.810). The choice was made on operational grounds, not metrics:

- **XGBoost catches 4 more fraud cases (73 vs 69)** for only 3 additional false alarms — a favorable trade-off at the scale of millions of daily transactions.
- **XGBoost trains 18× faster** (2.5s vs 30.8s) — important for retraining workflows.
- **XGBoost is the industry standard** for tabular fraud detection; tooling for monitoring, explainability, and deployment is more mature.

LightGBM was also tested but produced degenerate probability outputs (all 0s and 1s) at this imbalance level, making PR-AUC unreliable. Excluded from the final comparison and documented as a known issue.

### Honest Caveat on Overfitting

The tuned XGBoost achieves training PR-AUC of 1.0 vs. test PR-AUC of 0.81 — significant memorization of the 378 training fraud cases. This is typical for tree-based models on imbalanced data and is the practical ceiling without additional data sources (e.g., per-customer transaction history, which this dataset doesn't provide).

---

## 4. Tech Stack

| Tool                  | Purpose                                                              |
| --------------------- | -------------------------------------------------------------------- |
| **Python 3.11**       | Implementation language                                              |
| **pandas, NumPy**     | Data loading and manipulation                                        |
| **scikit-learn**      | Baseline model, train/test split, preprocessing pipeline             |
| **XGBoost**           | Production model (gradient boosting)                                 |
| **LightGBM**          | Comparison model (excluded — see results)                            |
| **Optuna**            | Hyperparameter tuning via Bayesian optimization (TPE sampler)        |
| **MLflow**            | Experiment tracking — params, metrics, model artifacts               |
| **Streamlit**         | Multi-page portfolio dashboard with live prediction widget           |
| **Plotly**            | Interactive visualizations in the dashboard                          |
| **Matplotlib / seaborn** | Static plots in the EDA notebook                                  |
| **Jupyter**           | EDA notebook environment                                             |
| **pytest**            | Test suite (22 tests across data, features, and model layers)        |
| **ruff**              | Linting and import sorting                                           |
| **Docker / docker-compose** | Containerization for reproducible deployment                   |
| **GitHub Actions**    | CI pipeline — auto-runs tests and lint on every push                 |
| **Git**               | Version control                                                      |

---

## 5. Setup & Installation

Tested on Python 3.11+, Windows / macOS / Linux.

```bash
# Clone the repo
git clone https://github.com/reshmareddymukkala/end-to-end-credit-card-fraud-detection.git
cd end-to-end-credit-card-fraud-detection

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
pip install -e .                  # Make `src/` importable as a package
```

You'll also need the Credit Card Fraud dataset:

1. The dataset is widely available from public ML education resources.
2. Place it in the `data/` folder

---

## 6. How to Run

### Run the full ML pipeline

Each step depends on the previous one's output:

```bash
# 1. Validate raw data
python src/data/quality.py

# 2. Clean the data (writes data/cleaned.csv)
python src/data/cleaner.py

# 3. Engineer and select features (writes data/features.csv)
python src/features/run_features.py

# 4. Tune hyperparameters via Optuna (writes models/best_params.json)
python src/models/tuning.py

# 5. Train final model with MLflow tracking
python src/models/run_training.py

# 6. Export artifacts for the Streamlit app
python src/models/export_for_app.py
```

### Run the Streamlit app locally

```bash
streamlit run app/streamlit_app.py
```

Opens at http://localhost:8501. The app falls back to synthetic demo data if the pipeline hasn't been run, so it works immediately on a fresh clone.

### Run with Docker

```bash
docker compose up
```

Same URL: http://localhost:8501. No Python setup needed on the host beyond Docker.

### View MLflow experiment history

```bash
mlflow server --host 127.0.0.1 --port 5000 --workers 1
```

Opens at http://localhost:5000.

### Run the test suite

```bash
pytest tests/ -v
```

22 tests across data quality, feature engineering, and model behavior. Tests run in ~5 seconds and gracefully skip if data files are missing.

---

## 7. Feature Engineering

Thirteen features were engineered across three categories. The PCA-anonymized V1–V28 features limited domain-specific engineering to the interpretable columns (`Time`, `Amount`) and statistical aggregates of the V components.

| Feature                  | Category    | Rationale                                                              |
| ------------------------ | ----------- | ---------------------------------------------------------------------- |
| `hour_of_day`            | Domain      | Fraud often spikes at unusual hours (e.g., overnight)                  |
| `is_night`               | Domain      | Binary flag for the 0–6 AM window                                      |
| `log_amount`             | Domain      | `Amount` is heavily right-skewed; log transform improves linear models |
| `amount_zero_flag`       | Domain      | Captures the 0.6% of transactions at exactly $0 (often pre-auth pings) |
| `amount_bucket`          | Domain      | Discretizes amount into 5 tiers for tree models                        |
| `v_magnitude`            | Statistical | L2 norm of V1–V28 — fraud sits at extreme PCA-space distances          |
| `v_negative_count`       | Statistical | Count of V features < 0; top fraud predictors all skew negative        |
| `top_fraud_v_mean`       | Statistical | Mean of V10, V12, V14, V16, V17 — the strongest fraud predictors      |
| `top_fraud_v_min`        | Statistical | Min of the same — captures the most extreme negative value             |
| `amount_v17_interaction` | Interaction | High-value transactions × strongest fraud signal                       |
| `night_amount_ratio`     | Interaction | Amplifies suspicious large transactions during off-hours              |
| `fraud_score_proxy`      | Interaction | Composite signal: `-top_fraud_v_mean × v_magnitude`                    |
| `v14_v17_interaction`    | Interaction | Product of the two strongest individual fraud predictors               |

One feature (`amount_zero_flag`) was dropped during selection due to low variance (0.6% of rows have value 1). The remaining 12 plus the 30 original columns made up the final 42-feature training set.

---

## 8. Key Decisions & Lessons Learned

**Choosing PR-AUC over accuracy as the headline metric.** With 99.83% legit transactions, accuracy is a vanity metric — a model that predicts "not fraud" for everything scores 99.83% accuracy and catches zero fraud. PR-AUC focuses on the minority class and is the industry standard for imbalanced classification.

**A variance-based feature selection bug taught me to never trust mean-based thresholds.** My first pass at feature selection used "drop features with variance < 1% of the mean variance." This deleted **42 of 43 features** because `Time` (in seconds, range 0–172,000) has a variance ~2 billion times larger than any other feature. Its variance dominated the mean and made the threshold useless. The fix was to use the **median variance** instead, which is robust to outlier features. **Lesson: any threshold derived from a mean is fragile when feature scales differ.**

**Picking XGBoost over Random Forest despite a PR-AUC tie was the right operational call.** Random Forest and XGBoost scored 0.811 vs 0.810 — effectively identical. Choosing XGBoost based on 18× faster training and better tooling for production deployment was a judgment call, not a metric call. Worth knowing that real model selection often happens on factors beyond pure score.

**LightGBM produced degenerate outputs that I chose to document rather than fix.** At a 600:1 imbalance ratio, LightGBM consistently output probabilities of exactly 0 or 1 even after several configuration attempts (`is_unbalance=True`, then `scale_pos_weight`, then deeper regularization). After several rounds of debugging, I concluded that LightGBM didn't fit this specific imbalance and excluded it from the final comparison. **Lesson: knowing when to stop debugging a non-essential component is a real skill.**

**Tuning produced a smaller improvement than I expected — and that was useful information.** Optuna with 30 trials moved CV PR-AUC from 0.849 to 0.855 (+0.6%) and test PR-AUC from 0.806 to 0.810 (+0.5%). This narrow gain suggested the achievable ceiling for this dataset with these features is around 0.81 — further improvement would require additional data sources, not better tuning. **Lesson: tuning has diminishing returns; recognize the plateau and stop.**

---

## 9. Project Structure

end-to-end-credit-card-fraud-detection/
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI pipeline
├── app/
│   ├── pages/
│   │   ├── 1_Explore_the_Data.py     # Interactive EDA page
│   │   ├── 2_Model_Results.py        # Model comparison + live prediction widget
│   │   └── 3_How_I_Built_This.py     # Architecture + build story
│   ├── demo_data.py                  # Demo data fallback for fresh clones
│   ├── streamlit_app.py              # Main landing page
│   └── utils.py                      # Shared helpers + cached data loaders
├── data/                             # (gitignored) — CSVs from pipeline
├── models/                           # (gitignored) — trained .pkl files
├── notebooks/
│   └── eda.ipynb                     # 7-section EDA with matplotlib + seaborn
├── src/
│   ├── data/
│   │   ├── loader.py                 # Initial CSV inspection
│   │   ├── quality.py                # 5-check automated quality gate
│   │   └── cleaner.py                # Dedupe + dtype coercion
│   ├── features/
│   │   ├── engineering.py            # create_features + select_features
│   │   └── run_features.py           # Reproducible feature pipeline
│   └── models/
│       ├── baseline.py               # Logistic regression baseline
│       ├── compare_models.py         # Multi-model comparison
│       ├── tuning.py                 # Optuna hyperparameter search
│       ├── run_training.py           # Final training with MLflow
│       └── export_for_app.py         # Generate artifacts for Streamlit
├── tests/
│   ├── conftest.py                   # Shared pytest fixtures
│   ├── test_data_quality.py          # 5 quality gate tests
│   ├── test_features.py              # 11 feature engineering tests
│   └── test_model.py                 # 6 production model tests
├── .dockerignore
├── .gitignore
├── Dockerfile                        # Python 3.11-slim container
├── docker-compose.yml                # One-command deployment
├── pyproject.toml                    # ruff configuration
├── README.md                         # ← you are here
├── requirements.txt
└── setup.py                          # Makes src/ an importable package

---

## 10. Acknowledgments

- Built as a portfolio project demonstrating end-to-end ML engineering  from raw data to deployed app, with testing and CI

## Contact

**Reshma Reddy Mukkala** · [LinkedIn](linkedin.com/in/reshma-reddy-mukkala-8956b5209) · [GitHub](https://github.com/reshmareddymukkala)
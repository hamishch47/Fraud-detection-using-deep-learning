# Comparative Analysis of Machine Learning Models for Credit Card Fraud Detection (IEEE‑CIS)

This repository contains the code and artifacts for my BCA major project / thesis:

**“Comparative Analysis of Machine Learning Models for Credit Card Fraud Detection on Imbalanced Datasets”**

It evaluates multiple machine learning and deep learning models on the **IEEE‑CIS Fraud Detection** dataset (Kaggle) using **time‑ordered (chronological) splitting** to reflect real-world deployment.

## Thesis overview

Credit card fraud detection is challenging because:

- **Severe class imbalance** (fraud is rare)
- **High cost of false negatives** (missing fraud)
- **Changing fraud patterns over time (concept drift)**

This project compares models using metrics that are appropriate for imbalanced classification:

- **ROC‑AUC**
- **PR‑AUC** (primary metric)
- **Recall@5% FPR** (operational metric)

## Dataset

- **Dataset:** IEEE‑CIS Fraud Detection (Kaggle, 2019)
- **Transactions:** 590,540
- **Fraud rate:** ~3.5% (20,663 fraud / 569,877 legitimate)
- **Features:** 431 raw features (transaction + identity tables)

## Data splitting strategy

A **chronological split** is used:

- **Train:** first 60% of transactions
- **Test/stream:** last 40% of transactions

This avoids “future leakage” that can happen with random shuffling and better simulates production scoring (train on past → predict on future).

## Preprocessing pipeline (high level)

The preprocessing steps implemented in the notebook/scripts follow the thesis:

- Missing values: numeric → **median**, categorical → **"missing"**
- Categorical encoding: **target encoding** (fit on training only)
- Feature engineering:
  - Time features from `TransactionDT` (hour/day/week/month)
  - Amount features (log amount, decimal part, round-number flag)
  - Card-level aggregations (transaction count and **per-card fraud rate**)
- Dimensionality reduction: **PCA** on V1–V339 → 50 components (~95% variance)
- Scaling: StandardScaler fit on training only
- Class imbalance handling:
  - Tree models: class weights / `scale_pos_weight`
  - Neural network: sampling + loss weighting

## Models compared (8 total)

The thesis compares the following models:

1. Logistic Regression (L1, class-weight balanced)
2. SGD Logistic Regression
3. Random Forest
4. XGBoost
5. LightGBM
6. MLP (Static)
7. MLP (Adaptive, with ADWIN drift detection + replay buffer updates)
8. Stacked Hybrid (Random Forest + XGBoost + MLP combined via Logistic Regression)

## Results summary (from thesis)

| Model | ROC‑AUC | PR‑AUC | Recall@5%FPR |
|---|---:|---:|---:|
| Stacked Hybrid | 86.2% | 43.0% | 52.2% |
| Random Forest | 86.3% | 41.3% | 52.6% |
| XGBoost | 80.4% | 40.8% | 51.4% |
| LightGBM | 81.9% | 41.3% | 51.6% |
| MLP Adaptive | 82.7% | 23.5% | 41.3% |
| MLP Static | 79.7% | 27.2% | 39.2% |
| Logistic Regression | 81.7% | 18.6% | 37.1% |
| SGD Logistic Regression | 69.7% | 11.9% | 29.5% |

Key takeaways:

- **Tree-based ensembles** (Random Forest, XGBoost, LightGBM) are consistently strong on this tabular dataset.
- **Stacking** provides the best overall PR‑AUC.
- **PR‑AUC is more informative than ROC‑AUC** under heavy imbalance.
- The **adaptive MLP** improves Recall@5%FPR over the static MLP, supporting the importance of adapting to drift.

## Repository contents (typical)

- `Fraud_detection.ipynb` — main notebook with preprocessing, training, and evaluation.
- `app.py` — Streamlit demo/dashboard for local scoring.
- `scoring_service.py` — scoring utilities.
- `sample_transactions.csv` — sample transactions for demo/testing.
- `requirements.txt` — dependencies.

## Running locally

1) Install dependencies

```bash
pip install -r requirements.txt
```

2) Run the Streamlit dashboard

```bash
streamlit run app.py
```

Open http://localhost:8501

## Citation

If you use or adapt this work, please cite the IEEE‑CIS dataset and the key references listed in the thesis.

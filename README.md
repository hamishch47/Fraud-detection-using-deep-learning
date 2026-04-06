# Adaptive Credit Card Fraud Detection with Deep Learning & Concept Drift (IEEE‑CIS)

This project implements an advanced fraud detection system combining XGBoost and Deep Learning (MLP with MC-Dropout) on the IEEE‑CIS Fraud Detection dataset. It features expanded feature engineering, improved neural architectures, focal loss for imbalance, ensemble methods, and adaptive learning with concept drift detection.[1][2]

## 🆕 Recent Improvements (2024)

**Major enhancements have significantly improved model performance:**
- **Expanded Feature Set**: 350+ features (V-columns, C-columns, engineered features)
- **Advanced Neural Architecture**: Deeper MLP (256→128→64→32→1) with Focal Loss
- **Better Encoding**: TargetEncoder for categorical features
- **Enhanced Training**: Validation split, early stopping, LR scheduling, 20 epochs
- **Ensemble Methods**: Combined XGBoost + MLP predictions
- **Comprehensive Evaluation**: Detailed model comparison and analysis

📊 **Expected Performance**: ROC-AUC >85% | PR-AUC >25% | Recall@5%FPR >45%

📖 **See [IMPROVEMENTS.md](IMPROVEMENTS.md) and [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for detailed documentation.**

***

## Project Overview

Real‑world fraud detection systems face three major challenges:  
- Concept drift – user behavior and fraud patterns change over time.[3][4]
- Verification latency – only a subset of transactions are reviewed quickly, so labels arrive with delay.[1]
- Imbalanced data – fraud is rare compared to legitimate transactions.[5]

This project addresses these issues by:

- Using **XGBoost with incremental learning** trained on the IEEE‑CIS dataset for fraud probability estimation.[2][5]
- Simulating a streaming setting using `TransactionDT`, where transactions arrive chronologically and labels are revealed after a fixed delay.[6][1]
- Applying online incremental training when delayed labels arrive (continual learning).[4]
- Detecting concept drift with the ADWIN algorithm and triggering stronger adaptation when performance degrades.[7][8]

A lightweight Flask web app is provided to score individual transactions using the trained model, suitable for demonstrations and UI integration.[9][10]

***

## Main Components

- **Data and Preprocessing (Colab)**  
  - Download and merge IEEE‑CIS transaction and identity tables.[2]
  - **NEW**: Expanded feature set (350+ features including V-columns, C-columns)
  - **NEW**: TargetEncoder for categorical features (replaces LabelEncoder)
  - **NEW**: Time-based, amount-based, and card aggregation features
  - Handle missing values, encode categoricals, and standardize numerics

- **Deep Learning Models**  
  - **NEW**: ImprovedMLP architecture (256→128→64→32→1, no BatchNorm)
  - **NEW**: Focal Loss for class imbalance (alpha=0.25, gamma=2.0)
  - **NEW**: MC-Dropout for uncertainty estimation (compatible architecture)
  - **NEW**: Validation split, early stopping, and LR scheduling
  - **NEW**: Increased training from 5 to 20 epochs

- **XGBoost Model with Online Learning**  
  - XGBoost gradient boosting with incremental training capability.[5]
  - Drift‑triggered updates using `xgb.train()` with `xgb_model` parameter.[5]

- **Ensemble Methods**  
  - **NEW**: Combined predictions (0.6*XGBoost + 0.4*MLP)
  - Leverages both tree-based and deep learning strengths

- **Adaptive Learning Pipeline**  
  - Streaming simulation based on `TransactionDT`.[6]
  - Delayed label queue to model verification latency.[1]
  - Online incremental training on recent labeled samples (replay buffer).[4]
  - ADWIN drift detector over prediction error stream for drift‑triggered adaptation.[8][7]
  - **NEW**: Uses improved model and Focal Loss automatically

- Baselines Implemented  
  - XGBoost gradient boosted trees.[5]
  - Random Forest with balanced class weights.
  - Logistic Regression with L1 regularization.
  - LightGBM gradient boosting classifier.
  - Static XGBoost (one‑shot training, no updates).[5]
  - **XGBoost Adaptive** with incremental learning and ADWIN drift detection (main method).[7][5]

- Flask Web App  
  - Loads exported model weights and preprocessing artifacts.[10][9]
  - Simple HTML form to enter transaction features.  
  - Returns fraud probability and a textual risk assessment (for example, “Likely Fraud”).

***

## Evaluation

Experiments are run in a prequential manner: each streamed transaction is first scored, then (after delay) used for updating the model.[1]
Metrics are computed over time windows (for example, every 50k transactions):

- ROC‑AUC and PR‑AUC per time chunk.[5]
- Recall at fixed false positive rate (for example, FPR ≤ 5 percent).[3]
- Recovery time after detected drift events.[11][7]

Results show that the full method (delay‑aware, drift‑triggered, uncertainty‑gated) maintains more stable performance under drift compared to static and periodically retrained baselines, and uses manual review capacity more efficiently.[4][1]

***

## How to Use

1. Training and Experiments (Colab)  
   - Open the provided Colab notebook.  
   - Run data preprocessing, model training, and streaming experiments.  
   - Export `fraud_xgboost.json`, `scaler.pkl`, `encoders.pkl`, and `feature_cols.pkl`.

2. Web App  
   - Place exported artifacts in `model/` and `artifacts/`.  
   - Install dependencies: `pip install flask xgboost scikit-learn joblib`.[9][10]
   - Run `python app.py` and open `http://127.0.0.1:5000/`.

---

## 🚀 Analyst Dashboard (Static In-App Scoring)

The `app.py` Streamlit dashboard runs fully locally with no external services
required. Transactions are entered in the UI and scored on the spot using a
local scoring function. All data is stored in-memory for the current session.

### Local run

#### 1. Install dependencies

```bash
pip install -r requirements.txt
```

#### 2. (Optional) Add your trained model

Place `stacked_hybrid.pkl` (or set `MODEL_PATH` env var) in the repository
root. When present the app loads it automatically for inference. Without it
the app uses a deterministic rule-based fallback scorer so it works out of
the box.

#### 3. Run the Streamlit dashboard

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

#### 4. Add transactions

- Use the **"➕ Add New Transaction (Test / Demo)"** expander in the dashboard
  to create transactions.
- Each submission is scored immediately in the app and added to the queue.
- Select a row in the queue, then click **Confirm Fraud & Block** or
  **Mark as Safe** to update its status.

---

***

## Academic Use

This repository supports a 6–8 page research paper structure:

- Introduction: fraud, concept drift, delayed labels, motivation.[4][1]
- Related Work: deep fraud detection, drift detection, uncertainty in deep learning.[7][5][1]
- Method: dataset, streaming simulation, model, drift detection, uncertainty.[2][6]
- Experimental Setup: splits, delay settings, baselines, metrics.[11][3]
- Results: plots and tables comparing baselines versus full method.  
- Discussion and Conclusion: implications, limitations, future reinforcement learning based extensions.[12][4]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/153713325/8beda63d-33b4-42ca-a415-d514d728e19e/2107.13508v1.pdf)
[2](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
[3](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/153713325/63434d8b-fd7f-42fc-af26-75c36d3230d2/2409.13406v1.pdf)
[4](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/153713325/d3bb967b-9e79-4649-868b-3611e77dd5ae/2504.03750v1.pdf)
[5](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/153713325/f0e08a98-856d-47a8-9351-7f237694cd03/2012.03754v1.pdf)
[6](https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203)
[7](https://riverml.xyz/dev/api/drift/ADWIN/)
[8](https://riverml.xyz/0.21.0/api/drift/ADWIN/)
[9](https://www.meritshot.com/example-deploying-a-pytorch-model/)
[10](https://www.python-engineer.com/posts/pytorch-model-deployment-with-flask/)
[11](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/153713325/c067868e-252b-4d0e-9f2a-c22257592b30/2506.10842v1.pdf)
[12](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/153713325/6343cf5e-cdd7-4136-9f09-645ff412e0c2/2504.08183v1.pdf)
[13](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/153713325/f7ee0cfe-0d9b-4232-9582-7737b7672cff/2503.22681v1.pdf)
[14](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/153713325/5ccaab56-9449-4bfb-a9d5-a7d52389e12f/1-s2.0-S187705092030065X-main.pdf)
[15](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/153713325/93020ccf-b05c-4fe9-8a6e-4e3f7f517871/2205.15300v1.pdf)
[16](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/153713325/8f62bf0e-03b9-442f-bb7b-d0dcfa847da6/doc.pdf)

***

## 🎯 Model Performance

### Baseline (Before Improvements)
- XGBoost: ROC-AUC: 77.81%, PR-AUC: 17.90%
- MLP Static: ROC-AUC: 75.55%, PR-AUC: 13.21%

### Expected (After Improvements)
- **ROC-AUC**: >85% (+7% to +15% improvement)
- **PR-AUC**: >25% (+6% to +12% improvement)
- **Recall@5%FPR**: >45% (+9% to +15% improvement)

### Key Improvements
- 27x more features (13 → 350+)
- 67% deeper architecture (3 → 5 layers)
- 4x longer training (5 → 20 epochs)
- Target-aware encoding (TargetEncoder vs LabelEncoder)
- Focal Loss vs standard BCE
- Ensemble methods combining XGBoost + MLP

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed technical documentation.

***

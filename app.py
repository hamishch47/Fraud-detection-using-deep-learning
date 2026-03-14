# -*- coding: utf-8 -*-
"""Fraud Detection Dashboard — Multi-page Streamlit App

Comparing Traditional ML vs. Deep Learning for Credit Card Fraud Detection
(IEEE-CIS Dataset)
"""

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

try:
    import plotly.express as px
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# =========================
# Artifact paths — local relative path takes priority over Colab path
# =========================
REPO_DIR = Path(__file__).parent
LOCAL_ART_DIR = REPO_DIR / "results"
COLAB_ART_DIR = Path("/content/drive/MyDrive/Colab_Notebooks/fraud_app/results")


def _resolve_art_dir() -> Path:
    if LOCAL_ART_DIR.exists():
        return LOCAL_ART_DIR
    if COLAB_ART_DIR.exists():
        return COLAB_ART_DIR
    return LOCAL_ART_DIR


ART_DIR = _resolve_art_dir()

# =========================
# Model definition (MUST match your notebook)
# =========================
class ImprovedMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.dropout(self.relu(self.fc4(x)))
        return self.fc5(x)


def mc_predict(model: nn.Module, X: torch.Tensor, T: int = 20):
    """MC Dropout prediction: returns mean probability and variance."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    probs = []
    with torch.no_grad():
        for _ in range(T):
            logits = model(X)
            probs.append(torch.sigmoid(logits).cpu().numpy())

    probs = np.stack(probs, axis=0)
    model.eval()
    return probs.mean(axis=0).reshape(-1), probs.var(axis=0).reshape(-1)


# =========================
# Load MLP artifacts
# =========================
@st.cache_resource
def load_artifacts() -> tuple[Optional[dict], Optional[str]]:
    """Load MLP model artifacts. Returns (artifacts_dict, error_msg)."""
    try:
        mlp_path = ART_DIR / "mlp_state_dict.pt"
        if not mlp_path.exists():
            return None, f"MLP model not found at {mlp_path}. Please export artifacts from the Colab notebook."

        feature_input_cols = joblib.load(ART_DIR / "feature_input_cols.pkl")
        te = joblib.load(ART_DIR / "target_encoder.pkl")
        scaler = joblib.load(ART_DIR / "scaler.pkl")
        pca = joblib.load(ART_DIR / "pca.pkl")

        with open(ART_DIR / "target_mean.json", "r") as f:
            target_mean = json.load(f)["target_mean"]

        medians: dict = {}
        medians_path = ART_DIR / "medians.json"
        if medians_path.exists():
            with open(medians_path, "r") as f:
                medians = json.load(f)

        card_stats: dict = {}
        for card_col in ["card1", "card2", "card3"]:
            p = ART_DIR / f"{card_col}_stats.parquet"
            if p.exists():
                card_stats[card_col] = pd.read_parquet(p)

        config: dict = {}
        config_path = ART_DIR / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)

        model = ImprovedMLP(input_dim=len(feature_input_cols))
        state = torch.load(mlp_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        return {
            "config": config,
            "feature_input_cols": feature_input_cols,
            "te": te,
            "scaler": scaler,
            "pca": pca,
            "medians": medians,
            "target_mean": target_mean,
            "card_stats": card_stats,
            "model": model,
        }, None

    except Exception as e:
        return None, str(e)


@st.cache_resource
def load_xgb_model() -> tuple[Optional[object], Optional[str]]:
    """Load XGBoost model if available. Returns (booster, error_msg)."""
    try:
        import xgboost as xgb

        model_path = ART_DIR / "xgb_model.json"
        if not model_path.exists():
            return None, f"XGBoost model file not found at {model_path}"
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        return booster, None
    except ImportError:
        return None, "xgboost package not installed (pip install xgboost)"
    except Exception as e:
        return None, str(e)


# =========================
# Preprocessing (mirrors notebook logic)
# =========================
def preprocess_raw_to_features(
    df_raw: pd.DataFrame,
    te,
    pca,
    medians: dict,
    target_mean: float,
    card_stats: dict,
) -> pd.DataFrame:
    df = df_raw.copy()

    required = ["TransactionDT", "TransactionAmt"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Time features
    df["TransactionHour"] = (df["TransactionDT"] // 3600) % 24
    df["TransactionDay"] = df["TransactionDT"] // (3600 * 24)
    df["TransactionDayOfWeek"] = df["TransactionDay"] % 7

    # Amount features
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
    df["TransactionAmt_decimal"] = df["TransactionAmt"] - np.floor(df["TransactionAmt"])
    df["TransactionAmt_rounded"] = (df["TransactionAmt"] == np.round(df["TransactionAmt"])).astype(int)

    # Categorical fill
    cat_cols = ["ProductCD", "card4", "card6", "DeviceType", "P_emaildomain", "R_emaildomain", "DeviceInfo"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("object").fillna("missing")

    # Numeric fill using saved medians when available
    for col, med in medians.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(med)

    # Target encode
    present_cat = [c for c in cat_cols if c in df.columns]
    if present_cat:
        df[present_cat] = te.transform(df[present_cat])

    # Card aggregations
    for card_col in ["card1", "card2", "card3"]:
        if card_col in df.columns and card_col in card_stats:
            stats = card_stats[card_col]
            df = df.merge(stats, on=card_col, how="left")
            df[f"{card_col}_fraud_rate"] = df[f"{card_col}_fraud_rate"].fillna(target_mean)
            df[f"{card_col}_count"] = df[f"{card_col}_count"].fillna(0)

    # PCA for V columns
    v_cols = [c for c in df.columns if c.startswith("V")]
    if not v_cols:
        raise ValueError("No V* columns found. Upload must include V1..V339.")

    def v_key(c):
        try:
            return int(c[1:])
        except Exception:
            return 10**9

    v_cols = sorted(v_cols, key=v_key)
    Xv_df = df[v_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    Xv = Xv_df.values.astype(np.float32)
    Xv_pca = pca.transform(Xv)
    for i in range(Xv_pca.shape[1]):
        df[f"V_pca_{i}"] = Xv_pca[:, i]

    df = df.drop(columns=v_cols)
    if "TransactionDT" in df.columns:
        df = df.drop(columns=["TransactionDT"])

    return df


def align_and_scale(df_feat: pd.DataFrame, feature_input_cols: list[str], scaler) -> np.ndarray:
    for c in feature_input_cols:
        if c not in df_feat.columns:
            df_feat[c] = 0.0
    X = df_feat[feature_input_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
    return scaler.transform(X)


def flag_top_percent(probs: np.ndarray, top_percent: float) -> tuple[np.ndarray, float]:
    """Flag the top `top_percent` fraction (e.g., 5.0 -> top 5%)."""
    if len(probs) == 0:
        return np.array([], dtype=int), float("nan")
    top_percent = max(0.0, min(100.0, top_percent))
    if top_percent <= 0.0:
        return np.zeros_like(probs, dtype=int), float("inf")
    if top_percent >= 100.0:
        return np.ones_like(probs, dtype=int), float("-inf")
    cutoff = float(np.quantile(probs, 1.0 - (top_percent / 100.0)))
    return (probs >= cutoff).astype(int), cutoff


def build_compact_output(
    df_raw: pd.DataFrame,
    probs: np.ndarray,
    flags: np.ndarray,
    var: np.ndarray | None = None,
) -> pd.DataFrame:
    out = pd.DataFrame()
    if "TransactionID" in df_raw.columns:
        out["TransactionID"] = df_raw["TransactionID"]
    for c in ["TransactionDT", "TransactionAmt", "ProductCD", "card1", "card4", "DeviceType"]:
        if c in df_raw.columns:
            out[c] = df_raw[c]
    out["fraud_probability"] = probs
    if var is not None:
        out["uncertainty_variance"] = var
    out["fraud_flag"] = flags
    return out


def run_mlp_inference(
    artifacts: dict,
    df_raw: pd.DataFrame,
    use_mc: bool = False,
    T: int = 20,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run MLP inference on a DataFrame. Returns (probs, variance_or_None)."""
    df_feat = preprocess_raw_to_features(
        df_raw,
        artifacts["te"],
        artifacts["pca"],
        artifacts["medians"],
        artifacts["target_mean"],
        artifacts["card_stats"],
    )
    Xs = align_and_scale(df_feat, artifacts["feature_input_cols"], artifacts["scaler"])
    Xt = torch.tensor(Xs, dtype=torch.float32)
    if use_mc:
        return mc_predict(artifacts["model"], Xt, T=T)
    with torch.no_grad():
        probs = torch.sigmoid(artifacts["model"](Xt)).numpy().reshape(-1)
    return probs, None


# =========================
# Helper: load a results JSON file
# =========================
def _load_results_json(filename: str) -> tuple[Optional[dict], bool]:
    """Load a JSON file from the results directory. Returns (data, is_sample)."""
    path = ART_DIR / filename
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return data, bool(data.get("_sample", False))
    return None, True


# =========================
# Page 1: Project Overview
# =========================
def page_overview() -> None:
    st.title("🏠 Project Overview")
    st.markdown(
        """
        ## Adaptive Credit Card Fraud Detection
        ### IEEE-CIS Fraud Detection Dataset

        This dashboard showcases a comprehensive comparison of **Traditional Machine Learning** vs.
        **Deep Learning** approaches for credit card fraud detection using the
        [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection).
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🤖 Models Compared")
        st.markdown(
            """
            **Traditional ML:**
            - 🌳 XGBoost — incremental learning + ADWIN drift detection
            - 🌲 Random Forest — balanced class weights
            - 📉 Logistic Regression — L1 regularization
            - ⚡ LightGBM

            **Deep Learning:**
            - 🧠 ImprovedMLP: 256→128→64→32→1
              - MC-Dropout for uncertainty estimation
              - Focal Loss for class imbalance (α=0.25, γ=2.0)

            **Ensemble:**
            - 🎯 0.6 × XGBoost + 0.4 × MLP
            """
        )

    with col2:
        st.subheader("📊 Dataset & Challenges")
        st.markdown(
            """
            **Dataset:** IEEE-CIS Fraud Detection
            - 590,540 transactions
            - ~3.5% fraud rate (severe class imbalance)
            - 434 raw features (transaction + identity tables)

            **Key Challenges:**
            1. 🔄 **Concept Drift** — fraud patterns evolve over time
            2. ⏱️ **Verification Latency** — labels arrive with delay
            3. ⚖️ **Class Imbalance** — fraud is extremely rare
            """
        )

    st.subheader("🏗️ Architecture & Approach")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown(
            """
            **Feature Engineering:**
            - 350+ features including V-columns, C-columns
            - Time-based & amount-based features
            - Card aggregation features
            - Target encoding for categoricals
            - PCA dimensionality reduction on V-columns
            """
        )
    with col4:
        st.markdown(
            """
            **Adaptive Learning Pipeline:**
            - Streaming simulation via `TransactionDT`
            - Delayed label queue (verification latency)
            - Online incremental training (replay buffer)
            - ADWIN drift detector for adaptation triggers
            - Early stopping & LR scheduling for MLP
            """
        )

    st.subheader("📈 Baseline Performance (from README)")
    perf_data = {
        "Model": ["XGBoost", "MLP Static", "Ensemble (expected)"],
        "ROC-AUC": [0.7781, 0.7555, ">0.85"],
        "PR-AUC": [0.1790, 0.1321, ">0.25"],
    }
    st.table(pd.DataFrame(perf_data))

    st.info(
        "📌 Use the sidebar to explore **Model Performance**, run **Live Predictions**, "
        "score a **Single Transaction**, or view **Drift Analysis**."
    )


# =========================
# Page 2: Model Performance Comparison
# =========================
_CHART_COLORS = ["#3498db", "#e67e22", "#9b59b6", "#1abc9c", "#e74c3c", "#2ecc71"]
_CATEGORY_COLOR_MAP = {
    "Traditional ML": "#3498db",
    "Deep Learning": "#e74c3c",
    "Ensemble": "#2ecc71",
}
_TRADITIONAL_MODELS = {"XGBoost", "Random Forest", "Logistic Regression", "LightGBM"}
_DL_MODELS = {"MLP Static"}


def _model_category(name: str) -> str:
    if name in _TRADITIONAL_MODELS:
        return "Traditional ML"
    if name in _DL_MODELS:
        return "Deep Learning"
    return "Ensemble"


def page_model_performance() -> None:
    st.title("📊 Model Performance Comparison")

    metrics_data, is_sample = _load_results_json("metrics_comparison.json")

    if metrics_data is None:
        st.warning(
            "⚠️ `results/metrics_comparison.json` not found. "
            "Run `export_metrics.py` in your Colab notebook to generate real metrics."
        )
        return

    if is_sample:
        st.info(
            "ℹ️ Showing **sample / placeholder data**. "
            "Export real metrics from your Colab notebook with `export_metrics.py` to see actual results."
        )

    models: dict = metrics_data.get("models", {})
    if not models:
        st.error("No model data found in metrics_comparison.json")
        return

    # Build metrics DataFrame
    rows = []
    for model_name, m in models.items():
        rows.append(
            {
                "Model": model_name,
                "Category": _model_category(model_name),
                "ROC-AUC": float(m.get("roc_auc", 0)),
                "PR-AUC": float(m.get("pr_auc", 0)),
                "Recall@5%FPR": float(m.get("recall_at_5pct_fpr", 0)),
                "F1-Score": float(m.get("f1_score", 0)),
                "Precision": float(m.get("precision", 0)),
                "Recall": float(m.get("recall", 0)),
            }
        )
    df_metrics = pd.DataFrame(rows)

    # --- Bar charts ---
    st.subheader("📈 Metric Comparison")
    metric_cols = ["ROC-AUC", "PR-AUC", "Recall@5%FPR", "F1-Score"]

    if HAS_PLOTLY:
        chart_cols = st.columns(2)
        for i, metric in enumerate(metric_cols):
            fig = px.bar(
                df_metrics,
                x="Model",
                y=metric,
                color="Category",
                color_discrete_map=_CATEGORY_COLOR_MAP,
                title=f"{metric} by Model",
                text=metric,
            )
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig.update_layout(height=350, showlegend=(i == 0), yaxis_range=[0, 1.05])
            chart_cols[i % 2].plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(df_metrics.set_index("Model")[metric_cols])

    # --- ROC Curves ---
    st.subheader("📉 ROC Curves")
    roc_data, _ = _load_results_json("roc_curves.json")
    if roc_data and HAS_PLOTLY:
        fig_roc = go.Figure()
        for idx, (model_name, curve) in enumerate(roc_data.get("models", {}).items()):
            auc_val = models.get(model_name, {}).get("roc_auc")
            label = f"{model_name} (AUC={auc_val:.3f})" if auc_val is not None else model_name
            fig_roc.add_trace(
                go.Scatter(
                    x=curve["fpr"],
                    y=curve["tpr"],
                    mode="lines",
                    name=label,
                    line=dict(color=_CHART_COLORS[idx % len(_CHART_COLORS)], width=2),
                )
            )
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Classifier",
                line=dict(dash="dash", color="gray"),
            )
        )
        fig_roc.update_layout(
            title="ROC Curves — All Models",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=450,
            legend=dict(x=0.55, y=0.05),
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    elif roc_data:
        st.info("Install plotly for interactive ROC curves: `pip install plotly`")
    else:
        st.warning("`results/roc_curves.json` not found.")

    # --- PR Curves ---
    st.subheader("📉 Precision-Recall Curves")
    pr_data, _ = _load_results_json("pr_curves.json")
    if pr_data and HAS_PLOTLY:
        fig_pr = go.Figure()
        for idx, (model_name, curve) in enumerate(pr_data.get("models", {}).items()):
            pr_auc_val = models.get(model_name, {}).get("pr_auc")
            label = f"{model_name} (PR-AUC={pr_auc_val:.3f})" if pr_auc_val is not None else model_name
            fig_pr.add_trace(
                go.Scatter(
                    x=curve["recall"],
                    y=curve["precision"],
                    mode="lines",
                    name=label,
                    line=dict(color=_CHART_COLORS[idx % len(_CHART_COLORS)], width=2),
                )
            )
        fig_pr.update_layout(
            title="Precision-Recall Curves — All Models",
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=450,
            legend=dict(x=0.55, y=0.85),
        )
        st.plotly_chart(fig_pr, use_container_width=True)
    elif pr_data:
        st.info("Install plotly for interactive PR curves.")
    else:
        st.warning("`results/pr_curves.json` not found.")

    # --- Confusion Matrices ---
    st.subheader("🔲 Confusion Matrices")
    cm_data, _ = _load_results_json("confusion_matrices.json")
    if cm_data and HAS_PLOTLY:
        cm_models = list(cm_data.get("models", {}).items())
        n_grid_cols = 3
        cm_col_groups = [cm_models[i : i + n_grid_cols] for i in range(0, len(cm_models), n_grid_cols)]
        for group in cm_col_groups:
            cols = st.columns(n_grid_cols)
            for col_idx, (model_name, cm) in enumerate(group):
                matrix = cm["matrix"]
                labels = cm.get("labels", ["Legitimate", "Fraud"])
                fig_cm = px.imshow(
                    matrix,
                    text_auto=True,
                    labels=dict(x="Predicted", y="Actual"),
                    x=labels,
                    y=labels,
                    color_continuous_scale="Blues",
                    title=model_name,
                )
                fig_cm.update_layout(height=270, margin=dict(t=40, b=10, l=10, r=10))
                cols[col_idx].plotly_chart(fig_cm, use_container_width=True)
    elif cm_data:
        st.info("Install plotly for confusion matrix visualisations.")
    else:
        st.warning("`results/confusion_matrices.json` not found.")

    # --- Summary table ---
    st.subheader("📋 Summary Metrics Table")
    display_cols = ["Model", "Category", "ROC-AUC", "PR-AUC", "Recall@5%FPR", "F1-Score"]
    st.dataframe(
        df_metrics[display_cols]
        .sort_values("ROC-AUC", ascending=False)
        .reset_index(drop=True)
        .style.highlight_max(subset=["ROC-AUC", "PR-AUC", "Recall@5%FPR", "F1-Score"], color="lightgreen"),
        use_container_width=True,
    )

    # --- Key takeaways ---
    st.subheader("💡 Key Takeaways")
    best_idx = df_metrics["ROC-AUC"].idxmax()
    best_model = df_metrics.loc[best_idx, "Model"]
    best_roc = df_metrics.loc[best_idx, "ROC-AUC"]
    st.success(
        f"**Best model: {best_model}** (ROC-AUC: {best_roc:.4f})\n\n"
        "- 🏆 The **Ensemble** (0.6×XGBoost + 0.4×MLP) achieves the highest ROC-AUC and PR-AUC "
        "by combining gradient boosting and deep learning strengths.\n"
        "- 📊 **XGBoost** is the best single Traditional ML model, especially with adaptive drift detection.\n"
        "- 🧠 **MLP with MC-Dropout** provides uncertainty estimates valuable for risk management.\n"
        "- ⚡ **LightGBM** offers a fast alternative with competitive performance."
    )


# =========================
# Page 3: Live Prediction Comparison (CSV Upload)
# =========================
def page_live_prediction() -> None:
    st.title("🔍 Live Prediction Comparison")
    st.markdown(
        "Upload a CSV to run predictions through **MLP** (and XGBoost if available) simultaneously."
    )

    artifacts, art_err = load_artifacts()
    xgb_model, xgb_err = load_xgb_model()

    if artifacts is None:
        st.error(f"❌ MLP model not loaded: {art_err}")
        st.info("Place model artifacts in `results/` (export from the Colab notebook).")
        return

    if xgb_err:
        st.warning(f"⚠️ XGBoost not available ({xgb_err}). Running in **MLP-only** mode.")

    config = artifacts["config"]

    with st.sidebar:
        st.header("Prediction Options")
        use_mc = st.checkbox("MC Dropout uncertainty", value=False)
        T = st.slider("MC samples (T)", 5, 50, int(config.get("mc_dropout_T", 20)))
        top_percent = st.slider(
            "Flag top (%)",
            0.0,
            50.0,
            float(config.get("flag_top_percent", 5.0)),
            0.5,
        )
        st.caption("Flags the highest-risk X% of rows in the uploaded CSV.")

    uploaded = st.file_uploader(
        "Upload CSV (must include TransactionDT, TransactionAmt, and V1..V339)",
        type=["csv"],
        key="live_pred_upload",
    )

    if uploaded is None:
        st.info("📤 Upload a CSV file to run predictions.")
        return

    df_raw = pd.read_csv(uploaded)

    st.subheader("Input Preview")
    preview_cols = [
        c
        for c in ["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD", "card1", "card4", "DeviceType"]
        if c in df_raw.columns
    ]
    st.dataframe(df_raw[preview_cols].head(20) if preview_cols else df_raw.head(10), use_container_width=True)

    try:
        with st.spinner("Running MLP predictions…"):
            mlp_probs, mlp_var = run_mlp_inference(artifacts, df_raw, use_mc=use_mc, T=T)

        mlp_flags, mlp_cutoff = flag_top_percent(mlp_probs, top_percent)

        # Optionally run XGBoost on the same preprocessed features
        xgb_probs: np.ndarray | None = None
        if xgb_model is not None:
            try:
                import xgboost as xgb

                with st.spinner("Running XGBoost predictions…"):
                    df_feat = preprocess_raw_to_features(
                        df_raw,
                        artifacts["te"],
                        artifacts["pca"],
                        artifacts["medians"],
                        artifacts["target_mean"],
                        artifacts["card_stats"],
                    )
                    Xs = align_and_scale(df_feat, artifacts["feature_input_cols"], artifacts["scaler"])
                    xgb_probs = xgb_model.predict(xgb.DMatrix(Xs))
            except Exception as xe:
                st.warning(f"XGBoost inference failed: {xe}")

        # Build results table
        out = pd.DataFrame()
        if "TransactionID" in df_raw.columns:
            out["TransactionID"] = df_raw["TransactionID"]
        if "TransactionAmt" in df_raw.columns:
            out["TransactionAmt"] = df_raw["TransactionAmt"]

        out["MLP_Probability"] = mlp_probs
        if mlp_var is not None:
            out["MLP_Uncertainty"] = mlp_var
        out["MLP_Flag"] = mlp_flags

        if xgb_probs is not None:
            xgb_flags, _ = flag_top_percent(xgb_probs, top_percent)
            out["XGB_Probability"] = xgb_probs
            out["XGB_Flag"] = xgb_flags

            ensemble_probs = 0.6 * xgb_probs + 0.4 * mlp_probs
            ensemble_flags, _ = flag_top_percent(ensemble_probs, top_percent)
            out["Ensemble_Probability"] = ensemble_probs
            out["Ensemble_Flag"] = ensemble_flags

            out["Models_Agree"] = out["MLP_Flag"] == out["XGB_Flag"]

        # Summary statistics
        st.subheader("📊 Summary Statistics")
        if xgb_probs is not None:
            agree_count = int(out["Models_Agree"].sum())
            disagree_count = len(out) - agree_count
            agree_rate = 100.0 * agree_count / max(len(out), 1)

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Rows", len(out))
            c2.metric("MLP Flagged", int(out["MLP_Flag"].sum()))
            c3.metric("XGBoost Flagged", int(out["XGB_Flag"].sum()))
            c4.metric("Ensemble Flagged", int(out["Ensemble_Flag"].sum()))
            c5.metric("Agreement Rate", f"{agree_rate:.1f}%")

            if disagree_count > 0:
                st.subheader(f"⚡ Model Disagreements ({disagree_count} transactions)")
                st.caption("Transactions where MLP and XGBoost give different flag decisions.")
                disagree_df = out[~out["Models_Agree"]].copy()
                st.dataframe(disagree_df.head(30), use_container_width=True)
        else:
            c1, c2 = st.columns(2)
            c1.metric("Total Rows", len(out))
            c2.metric(
                "MLP Flagged",
                int(out["MLP_Flag"].sum()),
                help=f"Cutoff: {mlp_cutoff:.4f}" if np.isfinite(mlp_cutoff) else None,
            )

        st.subheader("📋 All Predictions (first 50 rows)")
        st.dataframe(out.head(50), use_container_width=True)

        st.subheader("🏆 Top 20 Highest-Risk Transactions")
        st.dataframe(out.sort_values("MLP_Probability", ascending=False).head(20), use_container_width=True)

        st.download_button(
            "⬇️ Download All Predictions (CSV)",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="predictions_comparison.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Inference failed: {e}")


# =========================
# Page 4: Single Transaction Scorer
# =========================
def _risk_level(prob: float) -> tuple[str, str]:
    """Return (display_label, streamlit_alert_type) for a fraud probability."""
    if prob < 0.20:
        return "🟢 Low Risk", "success"
    if prob < 0.50:
        return "🟡 Medium Risk", "warning"
    if prob < 0.75:
        return "🟠 High Risk", "error"
    return "🔴 Critical Risk", "error"


def page_single_transaction() -> None:
    st.title("🎯 Single Transaction Scorer")
    st.markdown("Enter transaction details and get fraud risk scores from all available models.")

    artifacts, art_err = load_artifacts()
    if artifacts is None:
        st.error(f"❌ MLP model not loaded: {art_err}")
        st.info("Place model artifacts in `results/` (export from the Colab notebook).")
        return

    with st.form("transaction_form"):
        st.subheader("Transaction Features")
        col1, col2, col3 = st.columns(3)

        with col1:
            amount = st.number_input(
                "Transaction Amount ($)", min_value=0.01, max_value=50_000.0, value=150.0, step=0.01
            )
            product_cd = st.selectbox("Product CD", ["W", "H", "C", "S", "R"])
            card4 = st.selectbox("Card Network (card4)", ["visa", "mastercard", "discover", "american express"])
            card6 = st.selectbox("Card Type (card6)", ["debit", "credit"])

        with col2:
            device_type = st.selectbox("Device Type", ["desktop", "mobile", "missing"])
            p_email = st.text_input("Purchaser Email Domain (P_emaildomain)", value="gmail.com")
            r_email = st.text_input("Recipient Email Domain (R_emaildomain)", value="gmail.com")
            device_info = st.text_input("Device Info", value="missing")

        with col3:
            card1 = st.number_input("card1", min_value=0, max_value=20_000, value=10_000)
            card2 = st.number_input("card2", min_value=0, max_value=600, value=320)
            transaction_dt = st.number_input(
                "TransactionDT (seconds)", min_value=0, max_value=15_811_131, value=86_400
            )
            mc_T = st.slider("MC Dropout samples (T)", 5, 50, 20)

        submitted = st.form_submit_button("🔍 Score Transaction", use_container_width=True)

    if not submitted:
        return

    # Build a synthetic single-row DataFrame with all required columns
    try:
        n_v_cols = 339  # Standard IEEE-CIS V-columns count
        row: dict = {
            "TransactionDT": [int(transaction_dt)],
            "TransactionAmt": [float(amount)],
            "ProductCD": [product_cd],
            "card1": [int(card1)],
            "card2": [int(card2)],
            "card3": [500],
            "card4": [card4],
            "card6": [card6],
            "DeviceType": [device_type],
            "P_emaildomain": [p_email],
            "R_emaildomain": [r_email],
            "DeviceInfo": [device_info],
        }
        for i in range(1, n_v_cols + 1):
            row[f"V{i}"] = [0.0]

        df_single = pd.DataFrame(row)

        with st.spinner("Running predictions…"):
            mlp_probs, mlp_var = run_mlp_inference(artifacts, df_single, use_mc=True, T=mc_T)

        mlp_prob = float(mlp_probs[0])
        mlp_uncertainty = float(mlp_var[0]) if mlp_var is not None else None

        st.subheader("🎯 Results")
        col_mlp, col_xgb = st.columns(2)

        with col_mlp:
            st.markdown("### 🧠 MLP (Deep Learning)")
            mlp_risk_label, mlp_risk_type = _risk_level(mlp_prob)
            st.metric("Fraud Probability", f"{mlp_prob:.1%}")
            if mlp_uncertainty is not None:
                st.metric("Uncertainty (Variance)", f"{mlp_uncertainty:.4f}")
            st.progress(min(mlp_prob, 1.0), text=mlp_risk_label)
            getattr(st, mlp_risk_type)(f"Assessment: {mlp_risk_label}")

        with col_xgb:
            xgb_model, xgb_err = load_xgb_model()
            if xgb_model is not None:
                try:
                    import xgboost as xgb

                    df_feat = preprocess_raw_to_features(
                        df_single,
                        artifacts["te"],
                        artifacts["pca"],
                        artifacts["medians"],
                        artifacts["target_mean"],
                        artifacts["card_stats"],
                    )
                    Xs = align_and_scale(df_feat, artifacts["feature_input_cols"], artifacts["scaler"])
X_df = pd.DataFrame(Xs, columns=artifacts["feature_input_cols"])
xgb_prob = float(
    xgb_model.predict(xgb.DMatrix(X_df, feature_names=artifacts["feature_input_cols"]))[0]
)

                    xgb_risk_label, xgb_risk_type = _risk_level(xgb_prob)
                    st.markdown("### 🌳 XGBoost (Traditional ML)")
                    st.metric("Fraud Probability", f"{xgb_prob:.1%}")
                    st.progress(min(xgb_prob, 1.0), text=xgb_risk_label)
                    getattr(st, xgb_risk_type)(f"Assessment: {xgb_risk_label}")

                    ensemble_prob = 0.6 * xgb_prob + 0.4 * mlp_prob
                    ens_risk_label, ens_risk_type = _risk_level(ensemble_prob)
                    st.markdown("### 🎯 Ensemble (0.6×XGB + 0.4×MLP)")
                    st.metric("Ensemble Fraud Probability", f"{ensemble_prob:.1%}")
                    st.progress(min(ensemble_prob, 1.0), text=ens_risk_label)
                    getattr(st, ens_risk_type)(f"Assessment: {ens_risk_label}")

                except Exception as xe:
                    st.warning(f"XGBoost scoring failed: {xe}")
            else:
                st.info(f"ℹ️ XGBoost not available: {xgb_err}")

        if mlp_uncertainty is not None:
            with st.expander("📊 Uncertainty Analysis (MC-Dropout)"):
                std_dev = float(np.sqrt(mlp_uncertainty))
                conf_level = "High" if mlp_uncertainty < 0.01 else ("Medium" if mlp_uncertainty < 0.05 else "Low")
                st.write(f"**Mean Fraud Probability:** {mlp_prob:.4f}")
                st.write(f"**Variance:** {mlp_uncertainty:.6f}")
                st.write(f"**Std Dev:** {std_dev:.4f}")
                st.write(f"**Model Confidence:** {conf_level}")
                st.write(
                    f"**95% CI (approx):** [{max(0.0, mlp_prob - 2 * std_dev):.4f}, "
                    f"{min(1.0, mlp_prob + 2 * std_dev):.4f}]"
                )

    except Exception as e:
        st.error(f"Scoring failed: {e}")


# =========================
# Page 5: Drift Analysis
# =========================
def page_drift_analysis() -> None:
    st.title("📈 Drift Analysis")
    st.markdown("How models perform across time windows and where ADWIN detects concept drift.")

    drift_data, is_sample = _load_results_json("drift_analysis.json")

    if drift_data is None:
        st.warning("⚠️ `results/drift_analysis.json` not found.")
        st.info(
            "To generate drift analysis data:\n"
            "1. Run the streaming simulation in your Colab notebook.\n"
            "2. Use `export_metrics.py` to export drift analysis results.\n"
            "3. Place `drift_analysis.json` in the `results/` directory."
        )
        return

    if is_sample:
        st.info(
            "ℹ️ Showing **sample / placeholder data**. "
            "Export real drift analysis from your Colab notebook with `export_metrics.py`."
        )

    time_windows: dict = drift_data.get("time_windows", {})
    drift_events: list = drift_data.get("drift_events", [])

    if not time_windows:
        st.error("No time-window data found in drift_analysis.json")
        return

    # Build long-format DataFrame
    records = []
    for model_name, windows in time_windows.items():
        for w in windows:
            records.append(
                {
                    "Model": model_name,
                    "Window": w["window"],
                    "ROC-AUC": float(w["roc_auc"]),
                    "PR-AUC": float(w["pr_auc"]),
                }
            )
    df_drift = pd.DataFrame(records)

    # ROC-AUC over time
    st.subheader("📊 ROC-AUC Over Time")
    if HAS_PLOTLY:
        fig_roc = go.Figure()
        for idx, model_name in enumerate(df_drift["Model"].unique()):
            mdf = df_drift[df_drift["Model"] == model_name].sort_values("Window")
            fig_roc.add_trace(
                go.Scatter(
                    x=mdf["Window"],
                    y=mdf["ROC-AUC"],
                    mode="lines+markers",
                    name=model_name,
                    line=dict(color=_CHART_COLORS[idx % len(_CHART_COLORS)], width=2),
                )
            )
        for event in drift_events:
            fig_roc.add_vline(
                x=event["window"],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Drift @ W{event['window']}",
                annotation_position="top right",
            )
        fig_roc.update_layout(
            title="ROC-AUC Performance Over Time Windows",
            xaxis_title="Time Window",
            yaxis_title="ROC-AUC",
            height=430,
            yaxis_range=[0.5, 1.0],
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    else:
        pivot = df_drift.pivot(index="Window", columns="Model", values="ROC-AUC")
        st.line_chart(pivot)

    # PR-AUC over time
    st.subheader("📊 PR-AUC Over Time")
    if HAS_PLOTLY:
        fig_pr = go.Figure()
        for idx, model_name in enumerate(df_drift["Model"].unique()):
            mdf = df_drift[df_drift["Model"] == model_name].sort_values("Window")
            fig_pr.add_trace(
                go.Scatter(
                    x=mdf["Window"],
                    y=mdf["PR-AUC"],
                    mode="lines+markers",
                    name=model_name,
                    line=dict(color=_CHART_COLORS[idx % len(_CHART_COLORS)], width=2),
                )
            )
        for event in drift_events:
            fig_pr.add_vline(x=event["window"], line_dash="dash", line_color="red")
        fig_pr.update_layout(
            title="PR-AUC Performance Over Time Windows",
            xaxis_title="Time Window",
            yaxis_title="PR-AUC",
            height=400,
        )
        st.plotly_chart(fig_pr, use_container_width=True)
    else:
        pivot_pr = df_drift.pivot(index="Window", columns="Model", values="PR-AUC")
        st.line_chart(pivot_pr)

    # Drift events table
    if drift_events:
        st.subheader("🚨 Detected Drift Events (ADWIN)")
        st.dataframe(pd.DataFrame(drift_events), use_container_width=True)

    st.subheader("💡 Analysis")
    st.markdown(
        """
        **Key observations:**
        - 🔄 **Concept Drift** (red dashed lines) causes performance degradation in static models over time.
        - ⚡ **Adaptive XGBoost** with ADWIN drift detection recovers faster after drift events.
        - 🧠 **MLP** maintains more stable performance than static XGBoost under drift.
        - 🎯 **ADWIN** flags significant distribution shifts, triggering targeted model updates.
        """
    )


# =========================
# Main — page navigation
# =========================
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.title("🛡️ Fraud Detection")
    st.caption("Traditional ML vs. Deep Learning")
    st.divider()

    page = st.radio(
        "Navigate to:",
        [
            "🏠 Project Overview",
            "📊 Model Performance",
            "🔍 Live Prediction",
            "🎯 Single Transaction",
            "📈 Drift Analysis",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(f"Artifacts: `{ART_DIR}`")

if page == "🏠 Project Overview":
    page_overview()
elif page == "📊 Model Performance":
    page_model_performance()
elif page == "🔍 Live Prediction":
    page_live_prediction()
elif page == "🎯 Single Transaction":
    page_single_transaction()
elif page == "📈 Drift Analysis":
    page_drift_analysis()

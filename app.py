# -*- coding: utf-8 -*-
"""Fraud Detection Dashboard — Multi-page Streamlit App

Comparing Traditional ML vs. Deep Learning for Credit Card Fraud Detection
(IEEE-CIS Dataset)
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
import sys
from typing import Optional


def _ensure_workspace_venv() -> None:
    """Relaunch with .venv interpreter when started from another environment."""
    app_path = Path(__file__).resolve()
    repo_dir = app_path.parent
    venv_python = repo_dir / ".venv" / "bin" / "python"

    if os.environ.get("FRAUD_APP_VENV_REEXEC") == "1":
        return
    if not venv_python.exists():
        return

    current_python = Path(sys.executable).resolve()
    if current_python == venv_python.resolve():
        return

    os.environ["FRAUD_APP_VENV_REEXEC"] = "1"
    os.execv(
        str(venv_python),
        [str(venv_python), "-m", "streamlit", "run", str(app_path), *sys.argv[1:]],
    )


_ensure_workspace_venv()

import numpy as np
import pandas as pd
from pandas.errors import ParserError
import streamlit as st
import torch
import torch.nn as nn

try:
    import joblib
except ImportError:
    class _JoblibCompat:
        """Minimal fallback interface when joblib is unavailable."""

        @staticmethod
        def load(path):
            with open(path, "rb") as f:
                return pickle.load(f)

    joblib = _JoblibCompat()

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
        state = torch.load(mlp_path, map_location="cpu", weights_only=True)
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
    st.title("🏦 Fraud Detection Command Center")
    st.markdown(
        """
        Welcome! This dashboard helps your team **detect and prevent credit card fraud** in real time.

        Use the sidebar on the left to navigate between tools. No technical knowledge is required.
        """
    )

    st.info(
        "💡 **Quick Guide:** Use **Live Transaction Monitor** to screen a batch of transactions, "
        "**Investigate Transaction** to check a single payment, **System Adaptation** to see how "
        "the AI adjusts to new fraud tactics, and **System Health** to review overall AI performance."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🤖 How the AI Works")
        st.markdown(
            """
            The system uses **multiple AI engines** cross-checking every transaction:

            - **Pattern Engine (XGBoost)** — learns from millions of past transactions to spot known fraud patterns
            - **Deep Learning Engine (Neural Network)** — finds complex hidden relationships in transaction data
            - **Combined Decision (Ensemble)** — merges both engines for the most reliable verdict

            When engines agree, you can act with confidence. When they disagree, a manual review is recommended.
            """
        )

    with col2:
        st.subheader("⚠️ Why Fraud Is Hard to Catch")
        st.markdown(
            """
            **Scale:** 590,540 transactions analysed — only ~3.5% are fraudulent.

            **Key Challenges the AI handles:**
            1. **Evolving Tactics** — fraudsters change behaviour; the AI continuously adapts
            2. **Confirmation Delays** — some fraud is only confirmed days later
            3. **Rare Events** — with so few fraud cases, the AI is tuned to avoid missing them
            """
        )

    st.subheader("📊 AI Engine Performance at a Glance")
    perf_data = {
        "AI Engine": ["Pattern Engine (XGBoost)", "Deep Learning Engine (Neural Network)", "Combined Decision (Ensemble, expected)"],
        "Overall Accuracy (ROC-AUC)": [0.7781, 0.7555, ">0.85"],
        "Fraud Catch Rate (PR-AUC)": [0.1790, 0.1321, ">0.25"],
    }
    st.table(pd.DataFrame(perf_data))

    with st.expander("🔬 View Technical Details"):
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


# =========================
# Page 2: System Health (Model Performance Comparison)
# =========================
_CHART_COLORS = ["#3498db", "#e67e22", "#9b59b6", "#1abc9c", "#e74c3c", "#2ecc71"]
_CATEGORY_COLOR_MAP = {
    "Traditional ML": "#3498db",
    "Deep Learning": "#e74c3c",
    "Ensemble": "#2ecc71",
}
_TRADITIONAL_MODELS = {"XGBoost", "Random Forest", "Logistic Regression", "LightGBM"}
_DL_MODELS = {"MLP Static"}

# Friendly display names for AI engines
_ENGINE_DISPLAY_NAMES = {
    "XGBoost": "Pattern Engine (XGBoost)",
    "Random Forest": "Balanced Forest Engine",
    "Logistic Regression": "Baseline Engine",
    "LightGBM": "Fast Pattern Engine (LightGBM)",
    "MLP Static": "Deep Learning Engine",
    "Ensemble": "Combined Decision Engine",
}


def _model_display_name(name: str) -> str:
    return _ENGINE_DISPLAY_NAMES.get(name, name)


def _model_category(name: str) -> str:
    if name in _TRADITIONAL_MODELS:
        return "Traditional ML"
    if name in _DL_MODELS:
        return "Deep Learning"
    return "Ensemble"


def page_model_performance() -> None:
    st.title("⚙️ System Health")
    st.markdown(
        "This page shows how well each AI engine is performing at detecting fraud. "
        "Higher scores mean the engine catches more fraud while raising fewer false alarms."
    )

    st.info(
        "📖 **Reading these scores:** An **Overall Accuracy** score close to 1.0 (or 100%) means the AI "
        "is very good at distinguishing fraud from legitimate transactions across the full range of risk "
        "thresholds. A **Fraud Catch Rate** close to 1.0 means it rarely misses actual fraud — "
        "particularly important because real fraud is rare (only ~3.5% of transactions)."
    )

    metrics_data, is_sample = _load_results_json("metrics_comparison.json")

    if metrics_data is None:
        st.warning(
            "Performance data not yet available. "
            "Ask your technical team to run `export_metrics.py` to generate the results."
        )
        return

    if is_sample:
        st.info(
            "ℹ️ Showing **illustrative sample data**. "
            "Ask your technical team to export real metrics to see actual system performance."
        )

    models: dict = metrics_data.get("models", {})
    if not models:
        st.error("No AI engine data found. Please contact your technical team.")
        return

    # Build metrics DataFrame with friendly column names
    rows = []
    for model_name, m in models.items():
        rows.append(
            {
                "AI Engine": _model_display_name(model_name),
                "Engine Type": _model_category(model_name),
                "Overall Accuracy": float(m.get("roc_auc", 0)),
                "Fraud Catch Rate": float(m.get("pr_auc", 0)),
                "Fraud Detection Rate": float(m.get("recall_at_5pct_fpr", 0)),
                "Balanced Score": float(m.get("f1_score", 0)),
                "Precision": float(m.get("precision", 0)),
                "Recall": float(m.get("recall", 0)),
            }
        )
    df_metrics = pd.DataFrame(rows)

    # --- Summary scorecards ---
    st.subheader("📊 Engine Performance Scorecards")
    metric_cols_display = ["Overall Accuracy", "Fraud Catch Rate", "Fraud Detection Rate", "Balanced Score"]

    if HAS_PLOTLY:
        chart_cols = st.columns(2)
        for i, metric in enumerate(metric_cols_display):
            fig = px.bar(
                df_metrics,
                x="AI Engine",
                y=metric,
                color="Engine Type",
                color_discrete_map=_CATEGORY_COLOR_MAP,
                title=f"{metric} by Engine",
                text=metric,
            )
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig.update_layout(height=350, showlegend=(i == 0), yaxis_range=[0, 1.05])
            chart_cols[i % 2].plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(df_metrics.set_index("AI Engine")[metric_cols_display])

    # --- Summary table ---
    st.subheader("📋 Full Performance Summary")
    display_cols = ["AI Engine", "Engine Type", "Overall Accuracy", "Fraud Catch Rate", "Fraud Detection Rate", "Balanced Score"]
    st.dataframe(
        df_metrics[display_cols]
        .sort_values("Overall Accuracy", ascending=False)
        .reset_index(drop=True)
        .style.highlight_max(
            subset=["Overall Accuracy", "Fraud Catch Rate", "Fraud Detection Rate", "Balanced Score"],
            color="lightgreen",
        ),
        use_container_width=True,
    )

    # --- Key takeaways ---
    st.subheader("✅ What This Means for Your Team")
    best_idx = df_metrics["Overall Accuracy"].idxmax()
    best_engine = df_metrics.loc[best_idx, "AI Engine"]
    best_score = df_metrics.loc[best_idx, "Overall Accuracy"]
    st.success(
        f"**Top Performing Engine: {best_engine}** (Overall Accuracy: {best_score:.1%})\n\n"
        "- The **Combined Decision Engine** delivers the best results by merging the strengths of "
        "both the Pattern Engine and the Deep Learning Engine.\n"
        "- The **Pattern Engine (XGBoost)** is the most reliable single-engine option, especially "
        "because it automatically adapts when new fraud tactics emerge.\n"
        "- The **Deep Learning Engine** provides a confidence level for each decision, which is "
        "useful when you need to know how certain the AI is.\n"
        "- The **Fast Pattern Engine (LightGBM)** offers a quick alternative with competitive accuracy."
    )

    with st.expander("🔬 View Technical Charts (ROC & Precision-Recall Curves)"):
        # --- ROC Curves ---
        st.subheader("Receiver Operating Characteristic (ROC) Curves")
        st.caption(
            "Each curve shows how well an engine separates fraud from legitimate transactions "
            "at different sensitivity settings. Curves closer to the top-left corner are better."
        )
        roc_data, _ = _load_results_json("roc_curves.json")
        if roc_data and HAS_PLOTLY:
            fig_roc = go.Figure()
            for idx, (model_name, curve) in enumerate(roc_data.get("models", {}).items()):
                auc_val = models.get(model_name, {}).get("roc_auc")
                label = f"{_model_display_name(model_name)} (Accuracy Score={auc_val:.3f})" if auc_val is not None else _model_display_name(model_name)
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
                    name="Random Guessing (baseline)",
                    line=dict(dash="dash", color="gray"),
                )
            )
            fig_roc.update_layout(
                title="Detection Performance Curves — All Engines",
                xaxis_title="False Alarm Rate",
                yaxis_title="Fraud Catch Rate",
                height=450,
                legend=dict(x=0.55, y=0.05),
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        elif roc_data:
            st.info("Install plotly for interactive charts: `pip install plotly`")
        else:
            st.warning("Detailed curve data not yet available.")

        # --- PR Curves ---
        st.subheader("Precision vs. Catch-Rate Curves")
        st.caption(
            "Shows the trade-off between catching more fraud (higher catch rate) and "
            "raising fewer false alarms (higher precision). Curves closer to the top-right are better."
        )
        pr_data, _ = _load_results_json("pr_curves.json")
        if pr_data and HAS_PLOTLY:
            fig_pr = go.Figure()
            for idx, (model_name, curve) in enumerate(pr_data.get("models", {}).items()):
                pr_auc_val = models.get(model_name, {}).get("pr_auc")
                label = f"{_model_display_name(model_name)} (Catch Rate Score={pr_auc_val:.3f})" if pr_auc_val is not None else _model_display_name(model_name)
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
                title="Precision vs. Catch-Rate — All Engines",
                xaxis_title="Fraud Catch Rate",
                yaxis_title="Precision (Low False Alarm Rate)",
                height=450,
                legend=dict(x=0.55, y=0.85),
            )
            st.plotly_chart(fig_pr, use_container_width=True)
        elif pr_data:
            st.info("Install plotly for interactive charts.")
        else:
            st.warning("Detailed curve data not yet available.")

        # --- Confusion Matrices ---
        st.subheader("Decision Outcome Grids")
        st.caption(
            "Shows how many transactions each engine correctly identified vs. misclassified. "
            "Ideally, the top-left (Legitimate → Legitimate) and bottom-right (Fraud → Fraud) cells should be large."
        )
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
                        title=_model_display_name(model_name),
                    )
                    fig_cm.update_layout(height=270, margin=dict(t=40, b=10, l=10, r=10))
                    cols[col_idx].plotly_chart(fig_cm, use_container_width=True)
        elif cm_data:
            st.info("Install plotly for decision outcome grids.")
        else:
            st.warning("Decision outcome grid data not yet available.")


# =========================
# Page 3: Live Transaction Monitor (CSV Upload)
# =========================
def page_live_prediction() -> None:
    st.title("📡 Live Transaction Monitor")
    st.markdown(
        "Upload a file of transactions and the AI will screen each one, flagging those that "
        "look suspicious so your team can take action."
    )

    st.info(
        "📂 **How to use:** Export a batch of recent transactions as a CSV file from your banking "
        "system and upload it here. The AI will analyse each transaction and highlight the ones "
        "that match known fraud patterns."
    )

    artifacts, art_err = load_artifacts()
    xgb_model, xgb_err = load_xgb_model()

    if artifacts is None:
        st.error(f"AI system not ready: {art_err}")
        st.info("Ask your technical team to place model artifacts in the `results/` folder.")
        return

    if xgb_err:
        st.warning(f"The Pattern Engine is currently unavailable ({xgb_err}). Running in Deep Learning Engine-only mode.")

    config = artifacts["config"]

    with st.sidebar:
        st.header("🎚️ Screening Options")
        use_mc = st.checkbox(
            "Enable AI Confidence Scoring",
            value=False,
            help="When enabled, the AI runs multiple checks on each transaction to report how confident it is in its decision. Slightly slower.",
        )
        T = st.slider(
            "Confidence check passes",
            5,
            50,
            int(config.get("mc_dropout_T", 20)),
            help="More passes = more accurate confidence score, but takes longer to run.",
        )
        top_percent = st.slider(
            "Flag top risky transactions (%)",
            0.0,
            50.0,
            float(config.get("flag_top_percent", 5.0)),
            0.5,
            help="Flags the highest-risk X% of transactions in the uploaded file for your team to review.",
        )
        st.caption(
            f"At {top_percent:.1f}%, only the most suspicious transactions will be flagged for review."
        )

    uploaded = st.file_uploader(
        "Upload transaction file (CSV format)",
        type=["csv"],
        key="live_pred_upload",
        help="The file must contain transaction data exported from your banking system, including columns such as TransactionDT, TransactionAmt, and the V-feature columns.",
    )

    if uploaded is None:
        st.info("👆 Upload a transaction file above to begin screening.")
        return

    try:
        df_raw = pd.read_csv(uploaded)
    except ParserError as pe:
        st.error("The file could not be read — it appears to be malformed.")
        st.caption(
            "Please ensure the file is a valid CSV where every row has the same number of columns "
            "as the header, and any fields containing commas are enclosed in quotes."
        )
        with st.expander("View technical error details"):
            st.code(str(pe))
        return
    except Exception as e:
        st.error(f"Could not open file: {e}")
        return

    st.subheader("📋 Transaction File Preview")
    st.caption("Showing the first 20 rows of your uploaded file.")
    preview_cols = [
        c
        for c in ["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD", "card1", "card4", "DeviceType"]
        if c in df_raw.columns
    ]
    st.dataframe(df_raw[preview_cols].head(20) if preview_cols else df_raw.head(10), use_container_width=True)

    try:
        with st.spinner("🔍 Screening transactions — please wait…"):
            mlp_probs, mlp_var = run_mlp_inference(artifacts, df_raw, use_mc=use_mc, T=T)

        mlp_flags, mlp_cutoff = flag_top_percent(mlp_probs, top_percent)

        # Optionally run Pattern Engine on the same preprocessed features
        xgb_probs: np.ndarray | None = None
        if xgb_model is not None:
            try:
                import xgboost as xgb

                with st.spinner("🔍 Running Pattern Engine check…"):
                    df_feat = preprocess_raw_to_features(
                        df_raw,
                        artifacts["te"],
                        artifacts["pca"],
                        artifacts["medians"],
                        artifacts["target_mean"],
                        artifacts["card_stats"],
                    )
                    Xs = align_and_scale(df_feat, artifacts["feature_input_cols"], artifacts["scaler"])
                    X_df = pd.DataFrame(Xs, columns=artifacts["feature_input_cols"])
                    xgb_probs = xgb_model.predict(xgb.DMatrix(X_df, feature_names=artifacts["feature_input_cols"]))
            except Exception as xe:
                st.warning(f"Pattern Engine check failed: {xe}")

        # Build results table with user-friendly column names
        out = pd.DataFrame()
        if "TransactionID" in df_raw.columns:
            out["Transaction ID"] = df_raw["TransactionID"]
        if "TransactionAmt" in df_raw.columns:
            out["Amount ($)"] = df_raw["TransactionAmt"]

        out["Risk Score (Deep Learning)"] = mlp_probs
        if mlp_var is not None:
            out["AI Certainty Score"] = mlp_var
        out["Flagged for Review (Deep Learning)"] = mlp_flags.astype(bool)

        if xgb_probs is not None:
            xgb_flags, _ = flag_top_percent(xgb_probs, top_percent)
            out["Risk Score (Pattern Engine)"] = xgb_probs
            out["Flagged for Review (Pattern Engine)"] = xgb_flags.astype(bool)

            ensemble_probs = 0.6 * xgb_probs + 0.4 * mlp_probs
            ensemble_flags, _ = flag_top_percent(ensemble_probs, top_percent)
            out["Risk Score (Combined)"] = ensemble_probs
            out["⚑ Final Flag"] = ensemble_flags.astype(bool)

            out["Engines Agree"] = mlp_flags == xgb_flags

        # --- Summary dashboard ---
        st.subheader("📊 Screening Summary")

        if xgb_probs is not None:
            agree_count = int((mlp_flags == xgb_flags).sum())
            disagree_count = len(out) - agree_count
            agree_rate = 100.0 * agree_count / max(len(out), 1)
            flagged_combined = int(ensemble_flags.sum())

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric(
                "Total Transactions",
                len(out),
                help="Total number of transactions in the uploaded file.",
            )
            c2.metric(
                "Flagged by Deep Learning",
                int(mlp_flags.sum()),
                help="Transactions flagged as high-risk by the Deep Learning Engine.",
            )
            c3.metric(
                "Flagged by Pattern Engine",
                int(xgb_flags.sum()),
                help="Transactions flagged as high-risk by the Pattern Engine.",
            )
            c4.metric(
                "⚑ Flagged by Combined Decision",
                flagged_combined,
                help="Transactions flagged by the Combined Decision Engine — the most reliable verdict.",
            )
            c5.metric(
                "Engines in Agreement",
                f"{agree_rate:.1f}%",
                help="Percentage of transactions where both AI engines reached the same decision. High agreement means more reliable results.",
            )

            if disagree_count > 0:
                st.warning(
                    f"⚠️ **{disagree_count} transaction(s) need manual review** — the two AI engines gave different verdicts. "
                    "These are shown below."
                )
                disagree_df = out[~out["Engines Agree"]].copy()
                st.dataframe(disagree_df.head(30), use_container_width=True)
        else:
            c1, c2 = st.columns(2)
            c1.metric("Total Transactions", len(out))
            c2.metric(
                "Flagged for Review",
                int(mlp_flags.sum()),
                help=f"Risk threshold used: {mlp_cutoff:.4f}" if np.isfinite(mlp_cutoff) else None,
            )

        # --- Top risks ---
        st.subheader("🚨 Highest-Risk Transactions (Top 20)")
        st.caption(
            "These transactions have the highest risk scores and should be prioritised for review. "
            "A score closer to 1.0 means higher suspicion of fraud."
        )
        risk_col = "Risk Score (Combined)" if "Risk Score (Combined)" in out.columns else "Risk Score (Deep Learning)"
        st.dataframe(out.sort_values(risk_col, ascending=False).head(20), use_container_width=True)

        with st.expander("📄 View All Screened Transactions"):
            st.caption("Showing first 50 rows.")
            st.dataframe(out.head(50), use_container_width=True)

        st.download_button(
            "⬇️ Download Full Screening Results (CSV)",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="fraud_screening_results.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Screening failed: {e}")


# =========================
# Page 4: Investigate Transaction
# =========================
def _decision_label(prob: float) -> tuple[str, str, str]:
    """Return (decision, emoji, streamlit_alert_type) for a fraud probability."""
    if prob < 0.20:
        return "✅ APPROVE", "🟢", "success"
    if prob < 0.50:
        return "🔍 MANUAL REVIEW", "🟡", "warning"
    if prob < 0.75:
        return "🚫 BLOCK — HIGH RISK", "🔴", "error"
    return "🚫 BLOCK — CRITICAL RISK", "🔴", "error"


def _risk_level(prob: float) -> tuple[str, str]:
    """Return (display_label, streamlit_alert_type) for a fraud probability."""
    if prob < 0.20:
        return "Low Risk", "success"
    if prob < 0.50:
        return "Medium Risk — Needs Review", "warning"
    if prob < 0.75:
        return "High Risk — Recommend Block", "error"
    return "Critical Risk — Block Immediately", "error"


def page_single_transaction() -> None:
    st.title("🔎 Investigate Transaction")
    st.markdown(
        "Enter the details of a specific transaction to get an instant AI verdict. "
        "The system will tell you whether to **Approve**, send for **Manual Review**, or **Block** the transaction."
    )

    st.info(
        "💡 **Tip:** Fill in as many fields as possible for the most accurate result. "
        "Fields you are unsure about can be left at their default values."
    )

    artifacts, art_err = load_artifacts()
    if artifacts is None:
        st.error(f"AI system not ready: {art_err}")
        st.info("Ask your technical team to place model artifacts in the `results/` folder.")
        return

    with st.form("transaction_form"):
        st.subheader("Transaction Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            amount = st.number_input(
                "Transaction Amount ($)",
                min_value=0.01,
                max_value=50_000.0,
                value=150.0,
                step=0.01,
                help="The dollar value of this transaction.",
            )
            product_cd = st.selectbox(
                "Product Category",
                ["W", "H", "C", "S", "R"],
                help="The product or service category code for this transaction.",
            )
            card4 = st.selectbox(
                "Card Network",
                ["visa", "mastercard", "discover", "american express"],
                help="The payment network of the card used.",
            )
            card6 = st.selectbox(
                "Card Type",
                ["debit", "credit"],
                help="Whether the card is a debit or credit card.",
            )

        with col2:
            device_type = st.selectbox(
                "Device Used",
                ["desktop", "mobile", "missing"],
                help="The type of device used to make the transaction.",
            )
            p_email = st.text_input(
                "Buyer Email Domain",
                value="gmail.com",
                help="The email domain of the person making the purchase (e.g. gmail.com, yahoo.com).",
            )
            r_email = st.text_input(
                "Recipient Email Domain",
                value="gmail.com",
                help="The email domain of the recipient, if applicable.",
            )
            device_info = st.text_input(
                "Device Identifier",
                value="missing",
                help="A device fingerprint or identifier string, if available.",
            )

        with col3:
            card1 = st.number_input(
                "Card Number Group (card1)",
                min_value=0,
                max_value=20_000,
                value=10_000,
                help="An anonymised grouping identifier for the card used.",
            )
            card2 = st.number_input(
                "Card Sub-Group (card2)",
                min_value=0,
                max_value=600,
                value=320,
                help="A secondary anonymised card grouping identifier.",
            )
            transaction_dt = st.number_input(
                "Transaction Timestamp (seconds from reference)",
                min_value=0,
                max_value=15_811_131,
                value=86_400,
                help="The transaction time as a number of seconds from the dataset reference date. For example, 86400 = 1 day in.",
            )
            mc_T = st.slider(
                "Confidence check passes",
                5,
                50,
                20,
                help="More passes = more reliable AI Certainty Score, but takes slightly longer.",
            )

        submitted = st.form_submit_button("🔍 Analyse Transaction", use_container_width=True)

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

        with st.spinner("🔍 Analysing transaction — please wait…"):
            mlp_probs, mlp_var = run_mlp_inference(artifacts, df_single, use_mc=True, T=mc_T)

        mlp_prob = float(mlp_probs[0])
        mlp_uncertainty = float(mlp_var[0]) if mlp_var is not None else None

        # ---- PRIMARY VERDICT ----
        st.markdown("---")
        decision, decision_emoji, alert_type = _decision_label(mlp_prob)
        risk_label, _ = _risk_level(mlp_prob)

        st.subheader("AI Verdict")
        getattr(st, alert_type)(f"## {decision}\n\nRisk Level: **{risk_label}**")

        # Traffic-light colour metric strip
        risk_pct = int(mlp_prob * 100)
        col_v1, col_v2, col_v3 = st.columns(3)
        col_v1.metric(
            "Risk Score",
            f"{risk_pct}%",
            help="A score from 0% (completely safe) to 100% (definitely fraud). Scores above 50% indicate high risk.",
        )
        if mlp_uncertainty is not None:
            # Variance from MC Dropout is typically in the range [0, 0.1]; scale to a 0–100 display range
            # by subtracting 1000× the variance from 100 (clamped to [0, 100]).
            _UNCERTAINTY_DISPLAY_SCALE = 1000
            certainty_pct = max(0.0, 100.0 - mlp_uncertainty * _UNCERTAINTY_DISPLAY_SCALE)
            conf_label = "High" if mlp_uncertainty < 0.01 else ("Medium" if mlp_uncertainty < 0.05 else "Low")
            col_v2.metric(
                "AI Certainty",
                f"{conf_label}",
                help="Indicates how confident the AI is in its verdict. High certainty means the decision is reliable. Low certainty means a human review is strongly recommended.",
            )
        col_v3.metric(
            "Amount",
            f"${amount:,.2f}",
            help="The transaction amount entered.",
        )

        st.progress(min(mlp_prob, 1.0), text=f"Risk level: {risk_label}")

        # ---- PATTERN ENGINE (XGBoost) ----
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

                xgb_decision, _, xgb_alert_type = _decision_label(xgb_prob)
                xgb_risk_label, _ = _risk_level(xgb_prob)

                ensemble_prob = 0.6 * xgb_prob + 0.4 * mlp_prob
                ens_decision, _, ens_alert_type = _decision_label(ensemble_prob)
                ens_risk_label, _ = _risk_level(ensemble_prob)

                st.markdown("---")
                st.subheader("Engine-by-Engine Breakdown")
                col_dl, col_xgb, col_ens = st.columns(3)

                with col_dl:
                    st.markdown("**Deep Learning Engine**")
                    getattr(st, alert_type)(f"{decision}\n\nScore: {mlp_prob:.1%}")

                with col_xgb:
                    st.markdown("**Pattern Engine**")
                    getattr(st, xgb_alert_type)(f"{xgb_decision}\n\nScore: {xgb_prob:.1%}")

                with col_ens:
                    st.markdown("**Combined Decision (Recommended)**")
                    getattr(st, ens_alert_type)(f"{ens_decision}\n\nScore: {ensemble_prob:.1%}")

                if _decision_label(mlp_prob)[0] != _decision_label(xgb_prob)[0]:
                    st.warning(
                        "⚠️ The two AI engines reached **different verdicts** for this transaction. "
                        "We recommend a **manual review** by a fraud analyst."
                    )

            except Exception as xe:
                st.warning(f"Pattern Engine check could not complete: {xe}")
        else:
            st.info(f"Pattern Engine not available: {xgb_err}")

        # ---- PROGRESSIVE DISCLOSURE ----
        with st.expander("🔬 Why did the AI make this decision?"):
            st.markdown(
                "The AI examined multiple signals in this transaction, including:"
            )
            st.markdown(
                f"""
                - **Transaction amount** (${amount:,.2f}) compared to typical fraud amounts in training data
                - **Time of transaction** relative to known fraud time patterns
                - **Device type** ({device_type}) and email domain patterns
                - **Card network** ({card4}) and card type ({card6}) historical fraud rates
                - **Hidden Pattern Indicators** — 339 anonymised behavioural signals compressed into key patterns
                """
            )
            st.info(
                "A high risk score means the combination of transaction amount, timing, device, "
                "and card details closely matches known fraud profiles seen in historical data."
            )

            if mlp_uncertainty is not None:
                st.markdown("---")
                st.markdown("**AI Certainty Details**")
                std_dev = float(np.sqrt(mlp_uncertainty))
                conf_label = "High" if mlp_uncertainty < 0.01 else ("Medium" if mlp_uncertainty < 0.05 else "Low")
                col_u1, col_u2, col_u3 = st.columns(3)
                col_u1.metric("Certainty Level", conf_label)
                col_u2.metric("Score Variability", f"±{std_dev:.3f}")
                col_u3.metric(
                    "Score Range (95%)",
                    f"{max(0.0, mlp_prob - 2*std_dev):.1%} – {min(1.0, mlp_prob + 2*std_dev):.1%}",
                )
                st.caption(
                    "The AI ran multiple independent checks on this transaction. "
                    "'Score Variability' shows how much the individual checks differed — "
                    "a low variability means all checks agreed, so the verdict is more reliable."
                )

        with st.expander("🛠️ View Raw Technical Data"):
            st.write(f"**Deep Learning Engine raw probability:** {mlp_prob:.6f}")
            if mlp_uncertainty is not None:
                st.write(f"**MC Dropout variance:** {mlp_uncertainty:.8f}")
                st.write(f"**Standard deviation:** {float(np.sqrt(mlp_uncertainty)):.6f}")
            st.caption("These are the underlying numeric outputs from the AI models before any thresholding is applied.")

    except Exception as e:
        st.error(f"Analysis failed: {e}")


# =========================
# Page 5: System Adaptation (Drift Analysis)
# =========================
def page_drift_analysis() -> None:
    st.title("🔄 System Adaptation")
    st.markdown(
        "Fraudsters constantly change their tactics. This page shows how the AI detects those changes "
        "and automatically updates itself to stay effective."
    )

    st.info(
        "📖 **What you're seeing:** The charts below track AI performance over time. "
        "When a **New Fraud Pattern** is detected (marked by a red dashed line), the system "
        "triggers an automatic update so it can continue catching fraud even as tactics evolve. "
        "A temporary dip in performance after a red line is normal — it means new fraud tactics "
        "briefly challenged the AI before it adapted."
    )

    drift_data, is_sample = _load_results_json("drift_analysis.json")

    if drift_data is None:
        st.warning("System adaptation data is not yet available.")
        st.info(
            "To generate this data, ask your technical team to:\n"
            "1. Run the streaming simulation in the analysis notebook.\n"
            "2. Export the results using `export_metrics.py`.\n"
            "3. Place `drift_analysis.json` in the `results/` directory."
        )
        return

    if is_sample:
        st.info(
            "ℹ️ Showing **illustrative sample data**. "
            "Ask your technical team to export real adaptation data to see actual system behaviour."
        )

    time_windows: dict = drift_data.get("time_windows", {})
    drift_events: list = drift_data.get("drift_events", [])

    if not time_windows:
        st.error("No time-period data found. Please contact your technical team.")
        return

    # Build long-format DataFrame with friendly names
    records = []
    for model_name, windows in time_windows.items():
        for w in windows:
            records.append(
                {
                    "AI Engine": _model_display_name(model_name),
                    "Time Period": w["window"],
                    "Overall Accuracy": float(w["roc_auc"]),
                    "Fraud Catch Rate": float(w["pr_auc"]),
                }
            )
    df_drift = pd.DataFrame(records)

    # Overall Accuracy over time
    st.subheader("📈 Overall Accuracy Over Time")
    st.caption(
        "Shows how well each AI engine performed during each time period. "
        "Red dashed lines mark moments when a **New Fraud Pattern** was detected and the system adapted."
    )
    if HAS_PLOTLY:
        fig_roc = go.Figure()
        for idx, engine_name in enumerate(df_drift["AI Engine"].unique()):
            mdf = df_drift[df_drift["AI Engine"] == engine_name].sort_values("Time Period")
            fig_roc.add_trace(
                go.Scatter(
                    x=mdf["Time Period"],
                    y=mdf["Overall Accuracy"],
                    mode="lines+markers",
                    name=engine_name,
                    line=dict(color=_CHART_COLORS[idx % len(_CHART_COLORS)], width=2),
                )
            )
        for event in drift_events:
            fig_roc.add_vline(
                x=event["window"],
                line_dash="dash",
                line_color="red",
                annotation_text=f"🚨 New Fraud Pattern @ Period {event['window']}",
                annotation_position="top right",
            )
        fig_roc.update_layout(
            title="AI Engine Accuracy Over Time — New Fraud Patterns Highlighted",
            xaxis_title="Time Period",
            yaxis_title="Overall Accuracy",
            height=430,
            yaxis_range=[0.5, 1.0],
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    else:
        pivot = df_drift.pivot(index="Time Period", columns="AI Engine", values="Overall Accuracy")
        st.line_chart(pivot)

    # Fraud Catch Rate over time
    st.subheader("📈 Fraud Catch Rate Over Time")
    st.caption(
        "Shows how many actual fraud cases each engine caught in each time period. "
        "Higher is better. Red lines show when new fraud patterns emerged."
    )
    if HAS_PLOTLY:
        fig_pr = go.Figure()
        for idx, engine_name in enumerate(df_drift["AI Engine"].unique()):
            mdf = df_drift[df_drift["AI Engine"] == engine_name].sort_values("Time Period")
            fig_pr.add_trace(
                go.Scatter(
                    x=mdf["Time Period"],
                    y=mdf["Fraud Catch Rate"],
                    mode="lines+markers",
                    name=engine_name,
                    line=dict(color=_CHART_COLORS[idx % len(_CHART_COLORS)], width=2),
                )
            )
        for event in drift_events:
            fig_pr.add_vline(x=event["window"], line_dash="dash", line_color="red")
        fig_pr.update_layout(
            title="Fraud Catch Rate Over Time — New Fraud Patterns Highlighted",
            xaxis_title="Time Period",
            yaxis_title="Fraud Catch Rate",
            height=400,
        )
        st.plotly_chart(fig_pr, use_container_width=True)
    else:
        pivot_pr = df_drift.pivot(index="Time Period", columns="AI Engine", values="Fraud Catch Rate")
        st.line_chart(pivot_pr)

    # Drift / New-Pattern events table
    if drift_events:
        st.subheader("🚨 New Fraud Pattern Events Detected")
        st.caption(
            "Each row below represents a point in time when the AI detected that fraudsters had "
            "changed their approach. The system automatically triggered an update at each of these moments."
        )
        df_events = pd.DataFrame(drift_events)
        if "window" in df_events.columns:
            df_events = df_events.rename(columns={"window": "Time Period"})
        st.dataframe(df_events, use_container_width=True)

    st.subheader("💡 Key Takeaways for Your Team")
    st.markdown(
        """
        - 🔴 **Red dashed lines = New Fraud Pattern Detected** — the AI noticed fraudsters changed their tactics and automatically started retraining.
        - 📉 **A dip after a red line** is expected — it takes a short time for the AI to fully adapt to the new pattern.
        - 🔄 **The Pattern Engine (XGBoost)** recovers fastest after new patterns emerge, thanks to its built-in auto-update capability.
        - 🧠 **The Deep Learning Engine** maintains more stable performance overall, but may adapt more slowly.
        - ✅ **Stable lines** mean no new fraud tactics were detected — the AI is performing as expected.
        """
    )

    with st.expander("🔬 View Technical Details"):
        st.markdown(
            """
            **Technical explanation:**
            - Performance is tracked across rolling time windows using `TransactionDT` as the time axis.
            - **New Fraud Pattern Detection** is performed using the **ADWIN (ADaptive WINdowing)** algorithm,
              which detects statistically significant shifts in the distribution of model error over time.
            - When ADWIN fires, the XGBoost model triggers an incremental retraining step using a replay buffer
              of recently confirmed fraud labels.
            - The MLP (Deep Learning Engine) uses early stopping and learning rate scheduling for stability.
            """
        )


# =========================
# Main — page navigation
# =========================
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.title("🏦 Fraud Detection")
    st.caption("AI-Powered Transaction Monitoring")
    st.divider()

    page = st.radio(
        "Navigate to:",
        [
            "Home",
            "System Health",
            "Live Transaction Monitor",
            "Investigate Transaction",
            "System Adaptation",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Need help? Contact your IT support team.")
    st.caption(f"Data location: `{ART_DIR}`")

if page == "Home":
    page_overview()
elif page == "System Health":
    page_model_performance()
elif page == "Live Transaction Monitor":
    page_live_prediction()
elif page == "Investigate Transaction":
    page_single_transaction()
elif page == "System Adaptation":
    page_drift_analysis()

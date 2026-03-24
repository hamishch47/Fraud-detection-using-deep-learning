# -*- coding: utf-8 -*-
"""Fraud Detection Dashboard — Simple, Human-in-the-Loop Streamlit App

Designed for non-technical users:
- Clear language
- Guided actions
- Confidence/uncertainty signaling
- Manual review queue
- Explainability panel
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
import sys
from typing import Optional, Any

import numpy as np
import pandas as pd
from pandas.errors import ParserError
import streamlit as st
import torch
import torch.nn as nn

# -------------------------
# Runtime environment helper
# -------------------------
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

# -------------------------
# Optional dependencies
# -------------------------
try:
    import joblib
except ImportError:
    class _JoblibCompat:
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

# -------------------------
# Paths
# -------------------------
REPO_DIR = Path(__file__).parent
LOCAL_ART_DIR = REPO_DIR / "results"
COLAB_ART_DIR = Path("/content/drive/MyDrive/Colab_Notebooks/fraud_app/results")
FEEDBACK_DIR = REPO_DIR / "feedback"
FEEDBACK_DIR.mkdir(exist_ok=True)


def _resolve_art_dir() -> Path:
    if LOCAL_ART_DIR.exists():
        return LOCAL_ART_DIR
    if COLAB_ART_DIR.exists():
        return COLAB_ART_DIR
    return LOCAL_ART_DIR


ART_DIR = _resolve_art_dir()


# -------------------------
# Model definition
# -------------------------
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


# -------------------------
# Artifact loading
# -------------------------
@st.cache_resource
def load_artifacts() -> tuple[Optional[dict], Optional[str]]:
    """Load MLP artifacts. Returns (artifact_dict, error)."""
    try:
        mlp_path = ART_DIR / "mlp_state_dict.pt"
        if not mlp_path.exists():
            return None, f"MLP model not found at: {mlp_path}"

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
    """Load XGBoost model if available. Returns (model, error)."""
    try:
        import xgboost as xgb

        model_path = ART_DIR / "xgb_model.json"
        if not model_path.exists():
            return None, f"XGBoost model file not found at: {model_path}"

        booster = xgb.Booster()
        booster.load_model(str(model_path))
        return booster, None
    except ImportError:
        return None, "xgboost is not installed"
    except Exception as e:
        return None, str(e)


# -------------------------
# Preprocessing
# -------------------------
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

    # Numeric fill
    for col, med in medians.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(med)

    # Target encode
    present_cat = [c for c in cat_cols if c in df.columns]
    if present_cat:
        df[present_cat] = te.transform(df[present_cat])

    # Card stats
    for card_col in ["card1", "card2", "card3"]:
        if card_col in df.columns and card_col in card_stats:
            stats = card_stats[card_col]
            df = df.merge(stats, on=card_col, how="left")
            df[f"{card_col}_fraud_rate"] = df[f"{card_col}_fraud_rate"].fillna(target_mean)
            df[f"{card_col}_count"] = df[f"{card_col}_count"].fillna(0)

    # PCA on V columns
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


def run_mlp_inference(
    artifacts: dict,
    df_raw: pd.DataFrame,
    use_mc: bool = True,
    T: int = 20,
) -> tuple[np.ndarray, np.ndarray | None]:
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


# -------------------------
# UI helpers
# -------------------------
def risk_band(prob: float) -> str:
    if prob < 0.20:
        return "Low"
    if prob < 0.50:
        return "Medium"
    if prob < 0.75:
        return "High"
    return "Critical"


def action_recommendation(prob: float, uncertain: bool) -> str:
    if uncertain:
        return "Send for manual review"
    if prob < 0.20:
        return "Approve"
    if prob < 0.50:
        return "Manual review"
    return "Block"


def confidence_label(var_value: float | None) -> str:
    if var_value is None:
        return "Not available"
    if var_value < 0.010:
        return "High"
    if var_value < 0.050:
        return "Medium"
    return "Low"


def explain_transaction(row: pd.Series) -> list[str]:
    reasons = []

    amt = float(row.get("TransactionAmt", 0))
    if amt > 1000:
        reasons.append("Transaction amount is relatively high.")

    hour = int((row.get("TransactionDT", 0) // 3600) % 24) if "TransactionDT" in row else 12
    if hour <= 5 or hour >= 23:
        reasons.append("Transaction time is outside typical daytime hours.")

    if str(row.get("DeviceType", "missing")).lower() in {"missing", "mobile"}:
        reasons.append("Device pattern is less consistent with known low-risk behavior.")

    if str(row.get("card4", "")).lower() in {"discover", "american express"}:
        reasons.append("Card network pattern is less common in this dataset profile.")

    if not reasons:
        reasons.append("No single factor dominates; decision comes from combined pattern signals.")
    return reasons[:4]


def ensure_session_state() -> None:
    if "review_queue" not in st.session_state:
        st.session_state.review_queue = pd.DataFrame()
    if "review_history" not in st.session_state:
        st.session_state.review_history = pd.DataFrame(columns=[
            "TransactionID",
            "risk_score",
            "risk_band",
            "recommended_action",
            "review_decision",
            "review_notes",
            "reviewer",
            "timestamp",
        ])


def add_to_review_queue(df: pd.DataFrame) -> None:
    ensure_session_state()
    if st.session_state.review_queue.empty:
        st.session_state.review_queue = df.copy()
    else:
        merged = pd.concat([st.session_state.review_queue, df], ignore_index=True)
        if "TransactionID" in merged.columns:
            merged = merged.drop_duplicates(subset=["TransactionID"], keep="last")
        st.session_state.review_queue = merged


def save_feedback_csv() -> Path:
    ensure_session_state()
    out_path = FEEDBACK_DIR / "manual_review_feedback.csv"
    st.session_state.review_history.to_csv(out_path, index=False)
    return out_path


# -------------------------
# Pages
# -------------------------
def page_home() -> None:
    st.title("Fraud Detection Dashboard")
    st.write("This system helps teams screen transactions, review uncertain cases, and record final decisions.")

    st.subheader("How to use this dashboard")
    st.markdown(
        """
        1. Go to **Transaction Screening** and upload a CSV file.
        2. Review high-risk or uncertain transactions in **Manual Review Queue**.
        3. Save investigator decisions to improve future model updates.
        """
    )

    st.subheader("Core workflow")
    col1, col2, col3 = st.columns(3)
    col1.info("Screen transactions with risk scores.")
    col2.info("Prioritize uncertain and high-risk cases.")
    col3.info("Capture final decisions as feedback.")

    st.subheader("Model summary")
    st.table(pd.DataFrame({
        "Engine": ["Pattern Engine (XGBoost)", "Deep Learning Engine (MLP)", "Combined Engine"],
        "Purpose": [
            "Detects structured fraud patterns",
            "Finds complex hidden behavior",
            "Combines both for final prioritization",
        ],
    }))


def page_transaction_screening() -> None:
    st.title("Transaction Screening")
    st.write("Upload transaction data. The system will score each transaction and suggest what to do next.")

    artifacts, art_err = load_artifacts()
    xgb_model, xgb_err = load_xgb_model()

    if artifacts is None:
        st.error(f"Model artifacts are not available: {art_err}")
        return

    with st.expander("Screening settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        top_percent = col1.slider("Flag top risk percent", 1.0, 50.0, 5.0, 0.5)
        uncertainty_threshold = col2.slider("Uncertainty threshold", 0.005, 0.100, 0.030, 0.001)
        mc_passes = col3.slider("Confidence passes", 5, 50, 20)

    uploaded = st.file_uploader("Upload CSV file", type=["csv"], key="screen_upload")
    if uploaded is None:
        st.info("Please upload a CSV file to begin.")
        return

    try:
        df_raw = pd.read_csv(uploaded)
    except ParserError as pe:
        st.error("The CSV file format appears invalid.")
        st.code(str(pe))
        return
    except Exception as e:
        st.error(f"File could not be read: {e}")
        return

    st.subheader("Data preview")
    st.dataframe(df_raw.head(20), use_container_width=True)

    try:
        with st.spinner("Running fraud screening..."):
            mlp_probs, mlp_var = run_mlp_inference(artifacts, df_raw, use_mc=True, T=mc_passes)

        xgb_probs = None
        if xgb_model is not None:
            try:
                import xgboost as xgb
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
                st.warning(f"Pattern engine unavailable for this run: {xe}")
        elif xgb_err:
            st.warning(f"Pattern engine not loaded: {xgb_err}")

        if xgb_probs is not None:
            combined = 0.6 * xgb_probs + 0.4 * mlp_probs
        else:
            combined = mlp_probs

        cutoff = float(np.quantile(combined, 1 - top_percent / 100.0))
        flagged = combined >= cutoff
        uncertain = (mlp_var is not None) & (mlp_var >= uncertainty_threshold)

        out = pd.DataFrame()
        out["TransactionID"] = df_raw["TransactionID"] if "TransactionID" in df_raw.columns else np.arange(len(df_raw))
        out["TransactionAmt"] = df_raw["TransactionAmt"] if "TransactionAmt" in df_raw.columns else np.nan
        out["risk_score"] = combined
        out["risk_band"] = [risk_band(v) for v in combined]
        out["mlp_score"] = mlp_probs
        out["xgb_score"] = xgb_probs if xgb_probs is not None else np.nan
        out["uncertainty_variance"] = mlp_var if mlp_var is not None else np.nan
        out["confidence"] = [confidence_label(v if mlp_var is not None else None) for v in (mlp_var if mlp_var is not None else [None]*len(out))]
        out["uncertain_case"] = uncertain if isinstance(uncertain, np.ndarray) else False
        out["flagged_top_risk"] = flagged
        out["recommended_action"] = [
            action_recommendation(prob=float(p), uncertain=bool(u))
            for p, u in zip(out["risk_score"].values, out["uncertain_case"].values)
        ]

        review_candidates = out[(out["flagged_top_risk"]) | (out["uncertain_case"])].copy()
        add_to_review_queue(review_candidates)

        st.subheader("Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total transactions", len(out))
        c2.metric("Flagged (top risk)", int(out["flagged_top_risk"].sum()))
        c3.metric("Uncertain cases", int(out["uncertain_case"].sum()))
        c4.metric("Manual review queue", len(st.session_state.review_queue))

        st.subheader("Prioritized transactions")
        st.dataframe(
            out.sort_values("risk_score", ascending=False).head(50),
            use_container_width=True,
        )

        st.download_button(
            "Download screening results (CSV)",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="screening_results.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Screening failed: {e}")


def page_case_explanations() -> None:
    st.title("Case Explanations")
    st.write("Select one transaction to see a simple explanation of why it was flagged.")

    ensure_session_state()
    queue = st.session_state.review_queue
    if queue.empty:
        st.info("No transactions in the review queue yet. Run Transaction Screening first.")
        return

    tx_ids = queue["TransactionID"].astype(str).tolist()
    selected = st.selectbox("Select transaction ID", tx_ids)
    row = queue[queue["TransactionID"].astype(str) == selected].iloc[0]

    st.subheader("Decision summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Risk score", f"{float(row['risk_score']):.1%}")
    col2.metric("Risk band", str(row["risk_band"]))
    col3.metric("Recommended action", str(row["recommended_action"]))

    st.subheader("Confidence and uncertainty")
    col4, col5 = st.columns(2)
    col4.metric("Confidence", str(row.get("confidence", "Not available")))
    uv = row.get("uncertainty_variance", np.nan)
    col5.metric("Uncertainty variance", f"{uv:.5f}" if pd.notna(uv) else "Not available")

    st.subheader("Why this transaction was flagged")
    reasons = explain_transaction(pd.Series({
        "TransactionAmt": row.get("TransactionAmt", np.nan),
        "DeviceType": "missing",
        "card4": "missing",
        "TransactionDT": 0,
    }))
    for r in reasons:
        st.write(f"- {r}")

    st.caption("These explanations are guidance signals and should support, not replace, human judgment.")


def page_manual_review_queue() -> None:
    st.title("Manual Review Queue")
    st.write("Review flagged or uncertain transactions and record final investigator decisions.")

    ensure_session_state()
    queue = st.session_state.review_queue
    if queue.empty:
        st.info("Queue is empty. Screen transactions first.")
        return

    st.subheader("Pending review items")
    st.dataframe(
        queue.sort_values("risk_score", ascending=False),
        use_container_width=True,
    )

    st.subheader("Record a review decision")
    tx_ids = queue["TransactionID"].astype(str).tolist()
    selected = st.selectbox("Transaction ID", tx_ids, key="review_tx")
    selected_row = queue[queue["TransactionID"].astype(str) == selected].iloc[0]

    col1, col2 = st.columns(2)
    reviewer = col1.text_input("Reviewer name", value="Investigator")
    decision = col2.selectbox("Decision", ["Confirm Fraud", "Mark Legitimate", "Escalate"])

    notes = st.text_area("Review notes", height=120, placeholder="Add investigation notes")

    if st.button("Save decision", use_container_width=True):
        record = {
            "TransactionID": selected_row["TransactionID"],
            "risk_score": selected_row["risk_score"],
            "risk_band": selected_row["risk_band"],
            "recommended_action": selected_row["recommended_action"],
            "review_decision": decision,
            "review_notes": notes,
            "reviewer": reviewer,
            "timestamp": pd.Timestamp.utcnow().isoformat(),
        }
        st.session_state.review_history = pd.concat(
            [st.session_state.review_history, pd.DataFrame([record])],
            ignore_index=True,
        )

        st.session_state.review_queue = queue[queue["TransactionID"].astype(str) != selected].copy()
        st.success("Decision saved and transaction removed from queue.")

    st.subheader("Review history")
    if st.session_state.review_history.empty:
        st.caption("No review decisions saved yet.")
    else:
        st.dataframe(st.session_state.review_history.sort_values("timestamp", ascending=False), use_container_width=True)

        csv_bytes = st.session_state.review_history.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download review feedback (CSV)",
            data=csv_bytes,
            file_name="manual_review_feedback.csv",
            mime="text/csv",
        )

        if st.button("Save feedback file to project folder"):
            path = save_feedback_csv()
            st.success(f"Feedback saved to: {path}")


def page_alerts_and_notifications() -> None:
    st.title("Alerts and Notifications")
    st.write("This section provides a simple operational view of alert volumes and recommended communication triggers.")

    ensure_session_state()

    queue_count = len(st.session_state.review_queue)
    history = st.session_state.review_history.copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Current queue size", queue_count)

    confirmed_fraud = int((history["review_decision"] == "Confirm Fraud").sum()) if not history.empty else 0
    escalated = int((history["review_decision"] == "Escalate").sum()) if not history.empty else 0
    c2.metric("Confirmed fraud cases", confirmed_fraud)
    c3.metric("Escalated cases", escalated)

    st.subheader("Notification guidance")
    st.markdown(
        """
        - Send administrator alert when a case is marked **Confirm Fraud**.
        - Send high-priority analyst alert when a case is **Escalate**.
        - Optionally notify customer support for customer communication workflows.
        """
    )

    if not history.empty:
        st.subheader("Recent confirmed/escalated outcomes")
        alert_df = history[history["review_decision"].isin(["Confirm Fraud", "Escalate"])].copy()
        st.dataframe(alert_df.sort_values("timestamp", ascending=False).head(50), use_container_width=True)


def page_system_health() -> None:
    st.title("System Health")
    st.write("This page shows model performance summaries for monitoring.")

    metrics_path = ART_DIR / "metrics_comparison.json"
    if not metrics_path.exists():
        st.info("No metrics file found in results folder.")
        return

    try:
        with open(metrics_path) as f:
            data = json.load(f)
        models = data.get("models", {})
        if not models:
            st.info("Metrics file exists but has no model entries.")
            return

        rows = []
        for model_name, m in models.items():
            rows.append({
                "Model": model_name,
                "ROC-AUC": float(m.get("roc_auc", 0)),
                "PR-AUC": float(m.get("pr_auc", 0)),
                "Precision": float(m.get("precision", 0)),
                "Recall": float(m.get("recall", 0)),
                "F1": float(m.get("f1_score", 0)),
            })
        df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)

        st.dataframe(df, use_container_width=True)

        if HAS_PLOTLY:
            fig = px.bar(df, x="Model", y=["ROC-AUC", "PR-AUC"], barmode="group", title="Model performance")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(df.set_index("Model")[["ROC-AUC", "PR-AUC"]])
    except Exception as e:
        st.error(f"Could not load metrics: {e}")


# -------------------------
# App shell
# -------------------------
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_session_state()

with st.sidebar:
    st.title("Fraud Detection")
    st.caption("Simple workflow for non-technical users")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Transaction Screening",
            "Case Explanations",
            "Manual Review Queue",
            "Alerts and Notifications",
            "System Health",
        ],
    )

    st.divider()
    st.caption(f"Artifacts folder: {ART_DIR}")

if page == "Home":
    page_home()
elif page == "Transaction Screening":
    page_transaction_screening()
elif page == "Case Explanations":
    page_case_explanations()
elif page == "Manual Review Queue":
    page_manual_review_queue()
elif page == "Alerts and Notifications":
    page_alerts_and_notifications()
elif page == "System Health":
    page_system_health()

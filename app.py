"""Credit Card Fraud Detection Analyst Dashboard."""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_HIGH_RISK_THRESHOLD = 75.0  # risk_score >= this → Pending Review

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

# Candidate paths are tried in order:
#   1. MODEL_PATH env var (if set)
#   2. ./stacked_hybrid.pkl
#   3. ./results/stacked_hybrid.pkl
_MODEL_CANDIDATES: list[Path] = (
    [Path(os.environ["MODEL_PATH"])] if os.environ.get("MODEL_PATH", "").strip() else []
) + [
    Path("stacked_hybrid.pkl"),
    Path("results/stacked_hybrid.pkl"),
]

_model: Any = None
_model_loaded_from: str | None = None


def _load_model() -> tuple[Any, str | None]:
    """Try each candidate path in order; return (model, path_str) or (None, None)."""
    for candidate in _MODEL_CANDIDATES:
        if candidate.exists():
            try:
                loaded = joblib.load(candidate)
                return loaded, str(candidate)
            except Exception:
                continue
    return None, None


_model, _model_loaded_from = _load_model()

# ---------------------------------------------------------------------------
# Local scoring
# ---------------------------------------------------------------------------


def _fallback_score(payload: dict) -> tuple[float, list[str]]:
    """Deterministic rule-based fallback scorer (no model required)."""
    amount = float(payload.get("amount", 0))
    velocity_1h = int(payload.get("velocity_1h", 0))
    geo_distance_km = float(payload.get("geo_distance_km", 0.0))
    device_new = int(payload.get("device_new", 0))

    prob = 0.05
    reasons: list[str] = []

    if amount > 50_000:
        prob += 0.40
        reasons.append("Very high transaction amount")
    elif amount > 10_000:
        prob += 0.20
        reasons.append("High transaction amount")
    elif amount > 5_000:
        prob += 0.10
        reasons.append("Elevated transaction amount")

    if velocity_1h >= 5:
        prob += 0.30
        reasons.append("High velocity (≥5 txns/h)")
    elif velocity_1h >= 3:
        prob += 0.15
        reasons.append("Elevated velocity (≥3 txns/h)")

    if geo_distance_km > 500:
        prob += 0.25
        reasons.append("Transaction far from home location")
    elif geo_distance_km > 100:
        prob += 0.10
        reasons.append("Transaction away from usual area")

    if device_new:
        prob += 0.15
        reasons.append("New/unrecognised device")

    prob = min(prob, 0.99)
    if not reasons:
        reasons = ["No significant risk flags"]

    return prob, reasons


def _score_with_model(payload: dict, model: Any) -> tuple[float, list[str]]:
    """Run inference using the loaded model artifact."""
    features = [
        float(payload.get("amount", 0)),
        int(payload.get("velocity_1h", 0)),
        float(payload.get("geo_distance_km", 0.0)),
        int(payload.get("device_new", 0)),
    ]
    X = np.array(features, dtype=float).reshape(1, -1)
    prob = float(model.predict_proba(X)[0, 1])
    reasons = [f"Model score: {prob * 100:.1f}"]
    return prob, reasons


def score_transaction(payload: dict) -> dict:
    """Score a transaction locally and return risk_score, decision, reason_codes."""
    if _model is not None:
        try:
            prob, reasons = _score_with_model(payload, _model)
        except Exception:
            prob, reasons = _fallback_score(payload)
    else:
        prob, reasons = _fallback_score(payload)

    risk_score = round(prob * 100, 2)
    decision = "Pending Review" if risk_score >= _HIGH_RISK_THRESHOLD else "Auto-Approved"
    return {
        "transaction_id": payload.get("id", ""),
        "risk_score": risk_score,
        "decision": decision,
        "reason_codes": reasons,
    }


# ---------------------------------------------------------------------------
# Session-state data helpers
# ---------------------------------------------------------------------------

_SEED_TRANSACTIONS = [
    {
        "ID": "TXN-SEED-001",
        "Amount": 48500.0,
        "Merchant": "ElectroMart",
        "Location": "Delhi, DL",
        "Risk Score": 88.0,
        "Status": "Pending Review",
        "Context": "Very high transaction amount; High velocity (≥5 txns/h)",
        "Created At": datetime(2024, 1, 15, 10, 23),
    },
    {
        "ID": "TXN-SEED-002",
        "Amount": 1200.0,
        "Merchant": "Swiggy",
        "Location": "Bengaluru, KA",
        "Risk Score": 12.0,
        "Status": "Auto-Approved",
        "Context": "No significant risk flags",
        "Created At": datetime(2024, 1, 15, 10, 25),
    },
    {
        "ID": "TXN-SEED-003",
        "Amount": 15000.0,
        "Merchant": "JewelHouse",
        "Location": "Mumbai, MH",
        "Risk Score": 79.0,
        "Status": "Pending Review",
        "Context": "High transaction amount; New/unrecognised device",
        "Created At": datetime(2024, 1, 15, 10, 31),
    },
]


def _init_session_state() -> None:
    """Initialise session-state keys on first run."""
    if "transactions_df" not in st.session_state:
        st.session_state.transactions_df = pd.DataFrame(_SEED_TRANSACTIONS)
    if "selected_index" not in st.session_state:
        st.session_state.selected_index = None


def update_transaction_status(txn_id: str, new_status: str) -> bool:
    """Update a transaction's status in session state. Returns True on success."""
    df: pd.DataFrame = st.session_state.transactions_df
    mask = df["ID"] == txn_id
    if not mask.any():
        return False
    st.session_state.transactions_df.loc[mask, "Status"] = new_status
    return True


def insert_transaction(txn: dict, score_resp: dict) -> bool:
    """Append a newly scored transaction to the session-state DataFrame."""
    new_row = {
        "ID": txn["id"],
        "Amount": float(txn["amount"]),
        "Merchant": txn["merchant"],
        "Location": txn["location"],
        "Risk Score": float(score_resp["risk_score"]),
        "Status": score_resp["decision"],
        "Context": ", ".join(score_resp.get("reason_codes", [])) or "No flags",
        "Created At": datetime.now(),
    }
    new_df = pd.DataFrame([new_row])
    st.session_state.transactions_df = pd.concat(
        [new_df, st.session_state.transactions_df], ignore_index=True
    )
    return True


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------


def render_kpis(df: pd.DataFrame) -> None:
    """Render top KPI metrics."""
    if df.empty or "Status" not in df.columns:
        pending_count = 0
        auto_approved_count = 0
        total_value_at_risk = 0.0
    else:
        pending_mask = df["Status"] == "Pending Review"
        pending_count = int(pending_mask.sum())
        auto_approved_count = int((df["Status"] == "Auto-Approved").sum())
        total_value_at_risk = (
            float(df.loc[pending_mask, "Amount"].sum()) if pending_count > 0 else 0.0
        )

    col1, col2, col3 = st.columns(3)
    col1.metric("Pending High Risk Alerts", pending_count)
    col2.metric("Auto-Approved", auto_approved_count)
    col3.metric("Total Value at Risk", f"₹{total_value_at_risk:,.2f}")


def render_alert_queue(df: pd.DataFrame) -> int | None:
    """Render transaction queue table and return selected row index."""
    st.subheader("The Analyst Alert Queue")

    if df.empty:
        st.info("No transactions found. Use the form below to add transactions.")
        return None

    display_cols = ["ID", "Amount", "Merchant", "Location", "Risk Score", "Status", "Context"]
    display_df = df[[c for c in display_cols if c in df.columns]]

    table_event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Amount": st.column_config.NumberColumn(
                "Amount",
                format="₹%.2f",
            ),
            "Risk Score": st.column_config.ProgressColumn(
                "Risk Score",
                min_value=0,
                max_value=100,
                format="%d",
            ),
            "Context": st.column_config.TextColumn("Context", width="large"),
        },
    )

    selected_rows = table_event.selection.get("rows", []) if table_event else []
    if selected_rows:
        return int(selected_rows[0])
    return None


def render_investigation_panel(selected_index: int, df: pd.DataFrame) -> None:
    """Render details and actions for selected transaction."""
    if selected_index >= len(df):
        return
    selected_txn = df.iloc[selected_index]
    txn_id = str(selected_txn["ID"])

    with st.container(border=True):
        st.subheader("Investigation Details")
        st.write(f"**ID:** {txn_id}")
        st.write(f"**Amount:** ₹{float(selected_txn['Amount']):,.2f}")
        st.write(f"**Location:** {selected_txn['Location']}")
        st.info(selected_txn["Context"])

        action_col_1, action_col_2 = st.columns(2)
        with action_col_1:
            fraud_clicked = st.button(
                "Confirm Fraud & Block",
                type="primary",
                use_container_width=True,
            )
        with action_col_2:
            safe_clicked = st.button(
                "Mark as Safe",
                use_container_width=True,
            )

        if fraud_clicked:
            if update_transaction_status(txn_id, "Fraud"):
                st.success(f"{txn_id} marked as Fraud.")
            else:
                st.error("Failed to update status.")
            st.session_state.selected_index = None
            st.rerun()

        if safe_clicked:
            if update_transaction_status(txn_id, "Approved"):
                st.success(f"{txn_id} marked as Approved.")
            else:
                st.error("Failed to update status.")
            st.session_state.selected_index = None
            st.rerun()


def render_add_transaction_form() -> None:
    """Render expander form to create and score a dummy transaction."""
    with st.expander("➕ Add New Transaction (Test / Demo)"):
        with st.form("new_txn_form", clear_on_submit=True):
            col_a, col_b = st.columns(2)
            with col_a:
                txn_id = st.text_input(
                    "Transaction ID",
                    value=f"TXN-{uuid.uuid4().hex[:6].upper()}",
                )
                amount = st.number_input("Amount (₹)", min_value=1.0, value=2500.0, step=100.0)
                merchant = st.text_input("Merchant", value="Amazon")
                location = st.text_input("Location", value="Mumbai, MH")
            with col_b:
                velocity_1h = st.number_input("Velocity (txns last 1 h)", min_value=0, value=1)
                geo_distance_km = st.number_input("Geo distance from home (km)", min_value=0.0, value=5.0, step=1.0)
                device_new = st.selectbox("New device?", options=[0, 1], format_func=lambda x: "Yes" if x else "No")

            submitted = st.form_submit_button("Score & Add Transaction")

        if submitted:
            payload = {
                "id": txn_id,
                "amount": amount,
                "merchant": merchant,
                "location": location,
                "velocity_1h": velocity_1h,
                "geo_distance_km": geo_distance_km,
                "device_new": int(device_new),
            }

            score_resp = score_transaction(payload)
            insert_transaction(payload, score_resp)
            st.success(
                f"✅ **{txn_id}** added — "
                f"Risk Score: **{score_resp['risk_score']}**, "
                f"Decision: **{score_resp['decision']}**"
            )
            st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Main Streamlit app entry point."""
    st.set_page_config(
        page_title="Credit Card Fraud Analyst Dashboard",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("🛡️ Credit Card Fraud Detection — Live Analyst Dashboard")
    if _model_loaded_from:
        st.caption(f"Model loaded from: `{_model_loaded_from}`")
    else:
        st.caption("No model artifact found — using rule-based fallback scorer.")

    _init_session_state()

    df: pd.DataFrame = st.session_state.transactions_df

    render_kpis(df)
    selected_index = render_alert_queue(df)

    if selected_index is not None:
        st.session_state.selected_index = selected_index

    if st.session_state.selected_index is not None:
        current_index = st.session_state.selected_index
        if 0 <= current_index < len(df):
            render_investigation_panel(current_index, df)

    render_add_transaction_form()


if __name__ == "__main__":
    main()

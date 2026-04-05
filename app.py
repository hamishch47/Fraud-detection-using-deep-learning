"""Credit Card Fraud Detection Analyst Dashboard."""

from __future__ import annotations

import os
import uuid
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/fraud",
)
_SCORER_URL = os.getenv("SCORER_URL", "http://localhost:8000/score")
_AUTOREFRESH_INTERVAL_MS = 5_000  # 5 seconds

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

_engine: object | None = None


def _get_engine():
    """Return a cached SQLAlchemy engine, or None if unavailable."""
    global _engine  # noqa: PLW0603
    if _engine is None:
        try:
            _engine = create_engine(_DB_URL, pool_pre_ping=True)
        except Exception:
            _engine = None
    return _engine


def fetch_recent_transactions(limit: int = 200) -> pd.DataFrame:
    """Fetch the most recent transactions from the DB.

    Falls back to an empty DataFrame (preserving expected columns) when the
    database is unavailable.
    """
    engine = _get_engine()
    if engine is None:
        return _empty_transactions_df()
    try:
        query = text(
            """
            SELECT id        AS "ID",
                   amount    AS "Amount",
                   merchant  AS "Merchant",
                   location  AS "Location",
                   risk_score AS "Risk Score",
                   status    AS "Status",
                   context   AS "Context",
                   created_at AS "Created At"
            FROM transactions
            ORDER BY created_at DESC
            LIMIT :limit
            """
        )
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"limit": limit})
        return df
    except SQLAlchemyError:
        return _empty_transactions_df()


def _empty_transactions_df() -> pd.DataFrame:
    """Return an empty DataFrame with the expected schema columns."""
    return pd.DataFrame(
        columns=["ID", "Amount", "Merchant", "Location", "Risk Score", "Status", "Context", "Created At"]
    )


def update_transaction_status(
    txn_id: str,
    new_status: str,
    analyst: str = "analyst-ui",
) -> bool:
    """Update transaction status and record the analyst action.

    Returns True on success, False on failure.
    """
    engine = _get_engine()
    if engine is None:
        return False
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE transactions SET status = :s, reviewed_at = NOW() WHERE id = :id"
                ),
                {"s": new_status, "id": txn_id},
            )
            conn.execute(
                text(
                    """
                    INSERT INTO analyst_actions (txn_id, action, actor, action_time)
                    VALUES (:id, :a, :actor, NOW())
                    """
                ),
                {"id": txn_id, "a": new_status, "actor": analyst},
            )
        return True
    except SQLAlchemyError:
        return False


def insert_transaction(txn: dict, score_resp: dict) -> bool:
    """Insert a newly scored transaction into the DB.

    Returns True on success, False on failure.
    """
    engine = _get_engine()
    if engine is None:
        return False
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO transactions
                        (id, amount, merchant, location, risk_score, status, context, created_at)
                    VALUES
                        (:id, :amount, :merchant, :location, :risk_score, :status, :context, NOW())
                    """
                ),
                {
                    "id": txn["id"],
                    "amount": float(txn["amount"]),
                    "merchant": txn["merchant"],
                    "location": txn["location"],
                    "risk_score": float(score_resp["risk_score"]),
                    "status": score_resp["decision"],
                    "context": ", ".join(score_resp.get("reason_codes", [])) or "No flags",
                },
            )
        return True
    except SQLAlchemyError:
        return False


# ---------------------------------------------------------------------------
# Scoring service helper
# ---------------------------------------------------------------------------


def call_scorer(payload: dict) -> dict | None:
    """POST payload to scoring service and return response dict, or None on error."""
    try:
        resp = requests.post(_SCORER_URL, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


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
        st.info("No transactions found. Use the form below to add dummy transactions, or check DB connectivity.")
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
                st.error("Failed to update status in DB. Please check DB connectivity.")
            st.session_state.selected_index = None
            st.rerun()

        if safe_clicked:
            if update_transaction_status(txn_id, "Approved"):
                st.success(f"{txn_id} marked as Approved.")
            else:
                st.error("Failed to update status in DB. Please check DB connectivity.")
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

            score_resp = call_scorer(payload)
            if score_resp is None:
                st.error(
                    "Scoring service unavailable. Ensure it is running at "
                    f"`{_SCORER_URL}` (see README for instructions)."
                )
                return

            ok = insert_transaction(payload, score_resp)
            if ok:
                st.success(
                    f"✅ **{txn_id}** inserted — "
                    f"Risk Score: **{score_resp['risk_score']}**, "
                    f"Decision: **{score_resp['decision']}**"
                )
                st.rerun()
            else:
                st.error("Transaction scored but DB insert failed. Check DB connectivity.")


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

    # Auto-refresh every 5 seconds to pick up new scored transactions.
    st_autorefresh(interval=_AUTOREFRESH_INTERVAL_MS, key="live_refresh")

    # Initialise UI selection state only (data comes from DB).
    if "selected_index" not in st.session_state:
        st.session_state.selected_index = None

    # DB connectivity banner.
    if _get_engine() is None:
        st.warning(
            "⚠️ **Database not connected.** "
            "Set `DATABASE_URL` env var and ensure Postgres is running. "
            "The dashboard will show an empty queue until the DB is reachable."
        )

    df = fetch_recent_transactions()

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

"""Credit Card Fraud Detection Analyst Dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def initialize_data() -> None:
    """Initialize transaction data in session state on first load."""
    if "transactions_df" not in st.session_state:
        st.session_state.transactions_df = pd.DataFrame(
            [
                {
                    "ID": "TXN-8829",
                    "Amount": 350000.00,
                    "Merchant": "Croma Online",
                    "Location": "Moscow, RU",
                    "Risk Score": 98,
                    "Status": "Pending Review",
                    "Context": (
                        "IP mismatch, 8000 km from billing pin code. "
                        "3 failed PIN attempts prior. Device is known proxy."
                    ),
                },
                {
                    "ID": "TXN-8830",
                    "Amount": 1050.00,
                    "Merchant": "Ola Cabs",
                    "Location": "Mumbai, MH",
                    "Risk Score": 12,
                    "Status": "Auto-Approved",
                    "Context": (
                        "Known device, typical travel pattern. "
                        "Matches historical velocity."
                    ),
                },
                {
                    "ID": "TXN-8831",
                    "Amount": 70000.00,
                    "Merchant": "Reliance Digital",
                    "Location": "Bangalore, KA",
                    "Risk Score": 75,
                    "Status": "Pending Review",
                    "Context": (
                        "High-value electronics, new device footprint. "
                        "Card present but irregular pin code."
                    ),
                },
                {
                    "ID": "TXN-8832",
                    "Amount": 3800.00,
                    "Merchant": "Starbucks",
                    "Location": "New Delhi, DL",
                    "Risk Score": 5,
                    "Status": "Auto-Approved",
                    "Context": "Daily routine transaction. Geolocates near home address.",
                },
                {
                    "ID": "TXN-8833",
                    "Amount": 95000.00,
                    "Merchant": "CoinDCX",
                    "Location": "Unknown",
                    "Risk Score": 92,
                    "Status": "Pending Review",
                    "Context": (
                        "First time crypto purchase, masked IP. "
                        "High velocity of small test charges prior to this."
                    ),
                },
            ]
        )

    if "selected_index" not in st.session_state:
        st.session_state.selected_index = None


def render_kpis(df: pd.DataFrame) -> None:
    """Render top KPI metrics."""
    pending_mask = df["Status"] == "Pending Review"
    pending_count = int(pending_mask.sum())
    auto_approved_count = int((df["Status"] == "Auto-Approved").sum())
    total_value_at_risk = float(df.loc[pending_mask, "Amount"].sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Pending High Risk Alerts", pending_count)
    col2.metric("Auto-Approved", auto_approved_count)
    col3.metric("Total Value at Risk", f"₹{total_value_at_risk:,.2f}")


def render_alert_queue(df: pd.DataFrame) -> int | None:
    """Render transaction queue table and return selected row index."""
    st.subheader("The Analyst Alert Queue")

    table_event = st.dataframe(
        df,
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
    selected_txn = df.iloc[selected_index]

    with st.container(border=True):
        st.subheader("Investigation Details")
        st.write(f"**ID:** {selected_txn['ID']}")
        st.write(f"**Amount:** ₹{selected_txn['Amount']:,.2f}")
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
            st.session_state.transactions_df.at[selected_index, "Status"] = "Fraud"
            st.success(f"{selected_txn['ID']} marked as Fraud.")
            st.rerun()

        if safe_clicked:
            st.session_state.transactions_df.at[selected_index, "Status"] = "Approved"
            st.success(f"{selected_txn['ID']} marked as Approved.")
            st.rerun()


def main() -> None:
    """Main Streamlit app entry point."""
    st.set_page_config(
        page_title="Credit Card Fraud Analyst Dashboard",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("Credit Card Fraud Detection Analyst Dashboard")

    initialize_data()
    df = st.session_state.transactions_df

    render_kpis(df)
    selected_index = render_alert_queue(df)

    if selected_index is not None:
        st.session_state.selected_index = selected_index

    if st.session_state.selected_index is not None:
        current_index = st.session_state.selected_index
        if 0 <= current_index < len(st.session_state.transactions_df):
            render_investigation_panel(current_index, st.session_state.transactions_df)


if __name__ == "__main__":
    main()

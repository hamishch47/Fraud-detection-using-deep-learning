"""Generate a sample transactions CSV file for testing the Fraud Detection Dashboard.

Usage:
    python generate_sample_csv.py [--rows 200] [--output sample_transactions.csv] [--seed 42]

The generated file contains synthetic transaction data in the IEEE-CIS format,
suitable for uploading to the Live Transaction Monitor page of the dashboard.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def generate_sample_transactions(n_rows: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic transaction DataFrame in IEEE-CIS format."""
    rng = np.random.default_rng(seed)

    # --- Transaction IDs and timestamps ---
    transaction_ids = np.arange(10_000_001, 10_000_001 + n_rows, dtype=int)
    # TransactionDT: seconds since a reference date, spanning ~6 months
    transaction_dt = rng.integers(86_400, 15_811_131, size=n_rows)

    # --- Transaction amounts (log-normal, typical card purchase range) ---
    transaction_amt = np.round(
        rng.lognormal(mean=4.5, sigma=1.2, size=n_rows).clip(1.0, 20_000.0), 2
    )

    # --- Categorical features ---
    product_cd = rng.choice(["W", "H", "C", "S", "R"], size=n_rows)
    card4 = rng.choice(
        ["visa", "mastercard", "american express", "discover"],
        p=[0.45, 0.35, 0.12, 0.08],
        size=n_rows,
    )
    card6 = rng.choice(["debit", "credit"], p=[0.6, 0.4], size=n_rows)
    device_type = rng.choice(
        ["desktop", "mobile", "missing"], p=[0.30, 0.35, 0.35], size=n_rows
    )
    p_emaildomain = rng.choice(
        ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "missing"],
        p=[0.35, 0.20, 0.15, 0.10, 0.20],
        size=n_rows,
    )
    r_emaildomain = rng.choice(
        ["gmail.com", "yahoo.com", "hotmail.com", "missing"],
        p=[0.30, 0.20, 0.10, 0.40],
        size=n_rows,
    )
    device_info = rng.choice(
        ["Windows", "MacOS", "iOS Device", "Android", "missing"],
        p=[0.25, 0.15, 0.20, 0.20, 0.20],
        size=n_rows,
    )

    # --- Numeric card / address fields ---
    card1 = rng.integers(1_000, 20_000, size=n_rows)
    card2 = rng.integers(100, 600, size=n_rows)
    card3 = rng.choice([150, 180, 185, 500], size=n_rows)
    card5 = rng.integers(100, 300, size=n_rows)
    addr1 = rng.integers(100, 600, size=n_rows)
    addr2 = rng.integers(10, 90, size=n_rows)
    dist1 = np.round(rng.exponential(scale=50, size=n_rows), 1)
    dist2 = np.round(rng.exponential(scale=30, size=n_rows), 1)

    # --- C-columns (count-like features, non-negative integers) ---
    c_cols = {f"C{i}": rng.integers(0, 15, size=n_rows) for i in range(1, 15)}

    # --- V-columns (Vesta proprietary features, V1–V339) ---
    # In the IEEE-CIS dataset, V-columns are anonymised and can contain NaN.
    # We generate values from a standard normal distribution to mimic typical ranges.
    v_cols = {}
    for i in range(1, 340):
        # ~20% chance of being mostly NaN for each column (mirrors real dataset sparsity)
        col_vals = rng.standard_normal(size=n_rows)
        mask = rng.random(size=n_rows) < 0.15
        col_as_float = col_vals.astype(float)
        col_as_float[mask] = np.nan
        v_cols[f"V{i}"] = col_as_float

    # --- Assemble DataFrame ---
    data: dict = {
        "TransactionID": transaction_ids,
        "TransactionDT": transaction_dt,
        "TransactionAmt": transaction_amt,
        "ProductCD": product_cd,
        "card1": card1,
        "card2": card2,
        "card3": card3,
        "card4": card4,
        "card6": card6,
        "DeviceType": device_type,
        "P_emaildomain": p_emaildomain,
        "R_emaildomain": r_emaildomain,
        "DeviceInfo": device_info,
        "card5": card5,
        "addr1": addr1,
        "addr2": addr2,
        "dist1": dist1,
        "dist2": dist2,
        **c_cols,
        **v_cols,
    }

    return pd.DataFrame(data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a sample transactions CSV for the Fraud Detection Dashboard."
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=200,
        help="Number of synthetic transactions to generate (default: 200).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sample_transactions.csv",
        help="Output CSV file path (default: sample_transactions.csv).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    print(f"Generating {args.rows} synthetic transactions ...")
    df = generate_sample_transactions(n_rows=args.rows, seed=args.seed)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}  ({output_path.stat().st_size:,} bytes)")
    print(
        "You can now upload this file to the 'Live Transaction Monitor' page "
        "of the Fraud Detection Dashboard."
    )


if __name__ == "__main__":
    main()

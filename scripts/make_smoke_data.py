"""
SMOKE-TEST DATA ONLY.

Read this before you use it: the frame produced here exists to prove the
pipeline executes end to end. It is NOT an evaluation dataset and no metric
computed on it means anything. Synthetic fraud is separable in ways real fraud
is not, so any score you get here is an artefact of the generator, not of the
model.

Every number in RESULTS.md must come from IEEE-CIS. This file is scaffolding.

Usage:
    python scripts/make_smoke_data.py --rows 20000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

RNG = np.random.default_rng(7)


def make(rows: int = 20000, days: int = 60) -> pd.DataFrame:
    n_cards = max(rows // 12, 50)
    t = np.sort(RNG.uniform(0, days * 86400, rows))

    card1 = RNG.integers(1000, 1000 + n_cards, rows)
    amt = np.round(np.exp(RNG.normal(6.2, 1.1, rows)), 2)

    products = np.array(["W", "C", "R", "H", "S"])
    product = RNG.choice(products, rows, p=[0.55, 0.2, 0.1, 0.1, 0.05])

    domains = np.array(["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", None], dtype=object)
    p_email = RNG.choice(domains, rows, p=[0.5, 0.2, 0.12, 0.1, 0.08])
    r_email = RNG.choice(domains, rows, p=[0.45, 0.2, 0.12, 0.1, 0.13])

    devices = np.array(["Windows", "iOS Device", "MacOS", "Android", None], dtype=object)
    device = RNG.choice(devices, rows, p=[0.35, 0.2, 0.15, 0.2, 0.1])

    addr1 = RNG.integers(100, 500, rows).astype(float)
    addr1[RNG.random(rows) < 0.05] = np.nan

    # Latent risk: a few cards are compromised in the second half of the window,
    # which gives the drift monitors something real to detect.
    compromised = set(RNG.choice(np.arange(1000, 1000 + n_cards), n_cards // 25, replace=False))
    late = t > (days * 86400 * 0.6)
    logit = -3.6 + 0.28 * np.log1p(amt) / 2
    logit += np.array([0.9 if c in compromised else 0.0 for c in card1])
    logit += 0.7 * late * np.array([1.0 if c in compromised else 0.0 for c in card1])
    logit += 0.35 * (product == "C")
    logit += 0.3 * (pd.isna(p_email))
    p = 1 / (1 + np.exp(-logit))
    y = (RNG.random(rows) < p).astype(int)

    df = pd.DataFrame({
        "TransactionID": np.arange(1, rows + 1),
        "TransactionDT": t.astype(int),
        "TransactionAmt": amt,
        "isFraud": y,
        "ProductCD": product,
        "card1": card1,
        "card4": RNG.choice(["visa", "mastercard", "amex", "discover"], rows),
        "card6": RNG.choice(["debit", "credit"], rows),
        "addr1": addr1,
        "dist1": RNG.exponential(30, rows).round(1),
        "P_emaildomain": p_email,
        "R_emaildomain": r_email,
        "DeviceInfo": device,
    })
    for i in range(1, 15):
        df[f"C{i}"] = RNG.poisson(2.0, rows).astype(float)
    for i in range(1, 8):
        df[f"V{i}"] = RNG.normal(0, 1, rows).round(3)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--days", type=int, default=60)
    args = ap.parse_args()

    df = make(args.rows, args.days)
    out = DATA / "train_transaction.csv"
    df.to_csv(out, index=False)
    print(f"[SMOKE DATA — NOT FOR EVALUATION] wrote {len(df):,} rows to {out}")
    print(f"fraud rate: {df['isFraud'].mean():.4f}")

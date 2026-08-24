"""
Loading and splitting.

The only thing that matters here is that the split is temporal and that the
calibration slice sits between train and test in time. If you take one thing
from this module: a random train_test_split on fraud data is a leak, and it is
the single most common reason a hackathon fraud model reports 0.99 AUC.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG, DATA_DIR

log = logging.getLogger(__name__)


def load_ieee_cis(data_dir: Path | None = None, nrows: int | None = None) -> pd.DataFrame:
    """
    Load the IEEE-CIS Fraud Detection training data and join the identity table.

    Expected files in data_dir (download from Kaggle, do not synthesise):
        train_transaction.csv
        train_identity.csv

    The identity table is a LEFT join -- most transactions have no identity row,
    and that missingness is itself signal, so we do not drop those rows.
    """
    data_dir = data_dir or DATA_DIR
    tx_path = data_dir / "train_transaction.csv"
    id_path = data_dir / "train_identity.csv"

    if not tx_path.exists():
        raise FileNotFoundError(
            f"{tx_path} not found.\n"
            "Download the IEEE-CIS Fraud Detection dataset from Kaggle:\n"
            "  kaggle competitions download -c ieee-fraud-detection\n"
            f"and unzip train_transaction.csv / train_identity.csv into {data_dir}"
        )

    tx = pd.read_csv(tx_path, nrows=nrows)
    log.info("loaded %s transactions", f"{len(tx):,}")

    if id_path.exists():
        ident = pd.read_csv(id_path)
        # Kaggle ships id_01..id_38 in train and id-01..id-38 in test; normalise.
        ident.columns = [c.replace("-", "_") for c in ident.columns]
        tx = tx.merge(ident, on=CONFIG.id_col, how="left")
        log.info("joined identity table (%s rows matched)", f"{ident[CONFIG.id_col].isin(tx[CONFIG.id_col]).sum():,}")
    else:
        log.warning("train_identity.csv not found -- continuing without identity features")

    return tx


def temporal_split(
    df: pd.DataFrame,
    train_end: float | None = None,
    calib_end: float | None = None,
    time_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into (train, calibration, test) by time.

    Returns three frames, each sorted by time, with no overlap. The calibration
    slice exists so isotonic calibration is fitted out-of-time -- calibrating on
    the training slice would give optimistically sharp probabilities.
    """
    cfg = CONFIG.split
    train_end = train_end if train_end is not None else cfg.train_end
    calib_end = calib_end if calib_end is not None else cfg.calib_end
    time_col = time_col or cfg.time_col

    if time_col not in df.columns:
        raise KeyError(f"time column {time_col!r} missing -- cannot split temporally")

    df = df.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    t = df[time_col].to_numpy()
    t_train = np.quantile(t, train_end)
    t_calib = np.quantile(t, calib_end)

    train = df[df[time_col] <= t_train]
    calib = df[(df[time_col] > t_train) & (df[time_col] <= t_calib)]
    test = df[df[time_col] > t_calib]

    for name, part in (("train", train), ("calib", calib), ("test", test)):
        rate = part[CONFIG.target_col].mean() if CONFIG.target_col in part else float("nan")
        log.info(
            "%-6s n=%-9s window=[%s, %s] fraud_rate=%.4f",
            name, f"{len(part):,}", int(part[time_col].min()), int(part[time_col].max()), rate,
        )

    return (
        train.reset_index(drop=True),
        calib.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def split_summary(train: pd.DataFrame, calib: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Table for the README -- shows the split is temporal and shows base-rate drift."""
    time_col = CONFIG.split.time_col
    rows = []
    for name, part in (("train", train), ("calibration", calib), ("test", test)):
        rows.append({
            "split": name,
            "n": len(part),
            "t_min": int(part[time_col].min()),
            "t_max": int(part[time_col].max()),
            "days_span": round((part[time_col].max() - part[time_col].min()) / 86400, 1),
            "fraud_rate": round(float(part[CONFIG.target_col].mean()), 5),
            "n_fraud": int(part[CONFIG.target_col].sum()),
        })
    return pd.DataFrame(rows)

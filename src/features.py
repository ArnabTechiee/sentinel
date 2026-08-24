"""
Feature engineering.

Design rule, enforced everywhere in this file: a feature for row i may only use
rows strictly before row i in time. Velocity counts exclude the current
transaction. This is why the rolling windows are implemented by hand with
searchsorted instead of with pandas .rolling(), which will happily include the
current row and leak the present into the past.

Note on ordering: velocity features are computed on the FULL frame before the
temporal split. That is correct and realistic -- when scoring a live
transaction you genuinely do know that card's history, including history from
before your training cutoff. What would be a leak is using rows from *after*
the transaction, which the searchsorted bound makes impossible.

Categorical encodings, by contrast, are fitted on the TRAIN slice only, because
those are learned parameters.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import CONFIG

log = logging.getLogger(__name__)

UNSEEN_CODE = -1  # dedicated bucket for categories not present at fit time


# --------------------------------------------------------------------------
# velocity
# --------------------------------------------------------------------------
def _group_slices(codes: np.ndarray) -> list[np.ndarray]:
    """
    Return, for each group, the array of row positions belonging to it, in the
    original (time-sorted) order. Rows with code == -1 (missing entity value)
    get no shared history: each becomes its own singleton group.
    """
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    boundaries = np.flatnonzero(np.diff(sorted_codes)) + 1
    groups = np.split(order, boundaries)
    out = []
    for g in groups:
        if len(g) and codes[g[0]] == -1:
            out.extend([np.array([i]) for i in g])  # no shared history for NaN
        else:
            out.append(g)
    return out


def add_velocity_features(
    df: pd.DataFrame,
    entity_cols: tuple | None = None,
    windows: tuple | None = None,
    amount_col: str | None = None,
    time_col: str | None = None,
) -> pd.DataFrame:
    """
    For each entity (card, address, email domain, device) add, for each window W:
        {entity}_cnt_{W}    -- number of PRIOR transactions on that entity in W
        {entity}_amt_{W}    -- sum of PRIOR amounts on that entity in W
        {entity}_amtratio_W -- current amount / mean prior amount in W
    plus, per entity:
        {entity}_tsprev     -- seconds since that entity's previous transaction

    The frame must already be sorted by time.
    """
    fcfg = CONFIG.features
    entity_cols = entity_cols or fcfg.entity_cols
    windows = windows or fcfg.windows
    amount_col = amount_col or fcfg.amount_col
    time_col = time_col or CONFIG.split.time_col

    df = df.copy()
    t = df[time_col].to_numpy(dtype=np.float64)
    if not np.all(np.diff(t) >= 0):
        raise ValueError("frame must be sorted by time before velocity features")
    amt = df[amount_col].to_numpy(dtype=np.float64)
    n = len(df)

    for col in entity_cols:
        if col not in df.columns:
            log.warning("entity column %r absent -- skipping", col)
            continue

        codes = pd.factorize(df[col], use_na_sentinel=True)[0]
        cnt = {w: np.zeros(n, dtype=np.float32) for w in windows}
        asum = {w: np.zeros(n, dtype=np.float32) for w in windows}
        tsprev = np.full(n, np.nan, dtype=np.float32)

        for g in _group_slices(codes):
            gt = t[g]
            ga = amt[g]
            cs = np.concatenate(([0.0], np.cumsum(ga)))  # cs[i] = sum of ga[:i]
            idx = np.arange(len(g))

            # seconds since this entity's previous transaction
            if len(g) > 1:
                tsprev[g[1:]] = (gt[1:] - gt[:-1]).astype(np.float32)

            for w in windows:
                left = np.searchsorted(gt, gt - w, side="left")
                cnt[w][g] = (idx - left).astype(np.float32)          # excludes current
                asum[w][g] = (cs[idx] - cs[left]).astype(np.float32)  # excludes current

        df[f"{col}_tsprev"] = tsprev
        for w in windows:
            wl = _wlabel(w)
            df[f"{col}_cnt_{wl}"] = cnt[w]
            df[f"{col}_amt_{wl}"] = asum[w]
            with np.errstate(divide="ignore", invalid="ignore"):
                prior_mean = np.where(cnt[w] > 0, asum[w] / np.maximum(cnt[w], 1), np.nan)
                df[f"{col}_amtratio_{wl}"] = (amt / prior_mean).astype(np.float32)

        log.info("velocity features built for %s", col)

    return df


def _wlabel(seconds: int) -> str:
    if seconds % 86400 == 0:
        return f"{seconds // 86400}d"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    return f"{seconds}s"


# --------------------------------------------------------------------------
# simple derived features
# --------------------------------------------------------------------------
def add_basic_features(df: pd.DataFrame, time_col: str | None = None) -> pd.DataFrame:
    """Cheap, well-understood features. The amount-cents trick is a known strong
    signal on IEEE-CIS: currency-converted transactions carry long decimals."""
    time_col = time_col or CONFIG.split.time_col
    amount_col = CONFIG.features.amount_col
    df = df.copy()

    df["amt_log"] = np.log1p(df[amount_col].clip(lower=0))
    cents = (df[amount_col] - np.floor(df[amount_col])).round(4)
    df["amt_cents"] = cents.astype(np.float32)
    df["amt_is_round"] = (cents == 0).astype(np.int8)
    df["amt_decimal_len"] = (
        df[amount_col].astype(str).str.split(".").str[-1].str.len().fillna(0).astype(np.int8)
    )

    # TransactionDT is a second offset from a fixed (unknown) reference point.
    # Absolute date is meaningless; time-of-day and day-of-week are not.
    df["hour"] = ((df[time_col] / 3600) % 24).astype(np.int8)
    df["dayofweek"] = ((df[time_col] / 86400) % 7).astype(np.int8)
    df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(np.int8)

    if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
        df["email_mismatch"] = (
            (df["P_emaildomain"] != df["R_emaildomain"])
            & df["P_emaildomain"].notna()
            & df["R_emaildomain"].notna()
        ).astype(np.int8)

    return df


# --------------------------------------------------------------------------
# encoding
# --------------------------------------------------------------------------
class CategoricalEncoder:
    """
    Ordinal encoder fitted on TRAIN only. Anything unseen at transform time maps
    to UNSEEN_CODE rather than raising -- this is the production failure mode
    (a new device string, a new email domain) and it must not crash the scorer.
    """

    def __init__(self) -> None:
        self.maps: dict[str, dict] = {}
        self.columns: list[str] = []
        self.unseen_counts: dict[str, int] = {}

    def fit(self, df: pd.DataFrame, columns: list[str]) -> "CategoricalEncoder":
        self.columns = [c for c in columns if c in df.columns]
        for c in self.columns:
            vals = df[c].astype("object").where(df[c].notna(), None)
            uniq = [v for v in pd.unique(vals) if v is not None]
            self.maps[c] = {v: i for i, v in enumerate(uniq)}
        log.info("encoder fitted on %d categorical columns", len(self.columns))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in self.columns:
            m = self.maps[c]
            vals = df[c].astype("object").where(df[c].notna(), None)
            codes = vals.map(lambda v: m.get(v, UNSEEN_CODE) if v is not None else np.nan)
            self.unseen_counts[c] = int((codes == UNSEEN_CODE).sum())
            df[c] = codes.astype(np.float32)
        return df

    def unseen_report(self) -> pd.DataFrame:
        if not self.unseen_counts:
            return pd.DataFrame(columns=["column", "unseen_rows"])
        return (
            pd.DataFrame(
                [{"column": k, "unseen_rows": v} for k, v in self.unseen_counts.items()]
            )
            .sort_values("unseen_rows", ascending=False)
            .reset_index(drop=True)
        )


def detect_categoricals(df: pd.DataFrame) -> list[str]:
    """
    pandas 2.x infers text columns as `object`; pandas 3.x infers `str`.
    Check both, plus explicit `category`, so the pipeline is version-portable.
    """
    skip = {CONFIG.id_col, CONFIG.target_col, CONFIG.split.time_col}
    out = []
    for c in df.columns:
        if c in skip:
            continue
        dt = df[c].dtype
        if (
            dt == "object"
            or isinstance(dt, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(dt)
        ):
            out.append(c)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    skip = {CONFIG.id_col, CONFIG.target_col, CONFIG.split.time_col}
    return [c for c in df.columns if c not in skip]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature build on the whole (time-sorted) frame, pre-split."""
    df = df.sort_values(CONFIG.split.time_col, kind="mergesort").reset_index(drop=True)
    df = add_basic_features(df)
    df = add_velocity_features(df)
    log.info("feature build complete: %d columns", df.shape[1])
    return df

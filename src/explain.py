"""
Reason codes.

An analyst cannot act on 0.83. They can act on "this card has 6 transactions in
the last 40 minutes and this one is 4.2x its usual amount". This module turns
SHAP values into that sentence.

TreeExplainer on an XGBoost model is exact and fast enough to run per-case at
review time, so this is a product feature rather than a post-hoc notebook plot.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
import shap

log = logging.getLogger(__name__)


# Templates keyed by a regex over the feature name. First match wins.
_TEMPLATES: list[tuple[str, str]] = [
    (r"^(?P<ent>\w+?)_cnt_(?P<win>\w+)$", "{ent_h}: {v:.0f} prior transactions in the last {win_h}"),
    (r"^(?P<ent>\w+?)_amt_(?P<win>\w+)$", "{ent_h}: Rs {v:,.0f} transacted in the last {win_h}"),
    (r"^(?P<ent>\w+?)_amtratio_(?P<win>\w+)$", "amount is {v:.1f}x {ent_h}'s {win_h} average"),
    (r"^(?P<ent>\w+?)_tsprev$", "{ent_h}: {tsprev_h} since previous transaction"),
    (r"^amt_log$", "transaction amount"),
    (r"^TransactionAmt$", "transaction amount Rs {v:,.0f}"),
    (r"^amt_cents$", "unusual decimal component in amount ({v:.2f})"),
    (r"^amt_decimal_len$", "amount has {v:.0f} decimal digits"),
    (r"^amt_is_round$", "amount is a round number"),
    (r"^hour$", "transaction hour ({v:.0f}:00)"),
    (r"^is_night$", "transaction placed overnight"),
    (r"^dayofweek$", "day of week"),
    (r"^email_mismatch$", "purchaser and recipient email domains differ"),
    (r"^dist\d$", "billing/shipping distance ({v:.0f})"),
    (r"^C\d+$", "velocity counter {feat}"),
    (r"^D\d+$", "timedelta feature {feat}"),
    (r"^V\d+$", "Vesta risk feature {feat}"),
    (r"^id_\d+$", "device/identity signal {feat}"),
]

_ENTITY_HUMAN = {
    "card1": "this card",
    "addr1": "this billing address",
    "P_emaildomain": "this email domain",
    "DeviceInfo": "this device",
}

_WINDOW_HUMAN = {"1h": "hour", "24h": "24 hours", "1d": "24 hours", "7d": "7 days"}

# Anonymised dataset columns -- real signal, but nothing an analyst can act on.
# Anonymised or ordinal-encoded columns. Real signal, but "card1 = 5812" and
# "R_emaildomain = 0" tell an analyst nothing -- the second is an encoder code,
# not a domain. Kept out of the reserved human-readable slots.
_OPAQUE = re.compile(
    r"^(V\d+|C\d+|D\d+|id_\d+|M\d+|card\d+|addr\d+|dist\d+"
    r"|DeviceType|DeviceInfo|ProductCD|\w*emaildomain)$"
)


def _humanise_seconds(s: float) -> str:
    if not np.isfinite(s):
        return "no prior activity"
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s / 60:.0f} min"
    if s < 86400:
        return f"{s / 3600:.1f} h"
    return f"{s / 86400:.1f} days"


def humanise(feature: str, value: float) -> str:
    for pattern, template in _TEMPLATES:
        m = re.match(pattern, feature)
        if not m:
            continue
        g = m.groupdict()
        ent = g.get("ent", "")
        win = g.get("win", "")
        try:
            return template.format(
                feat=feature,
                v=value,
                ent_h=_ENTITY_HUMAN.get(ent, ent),
                win_h=_WINDOW_HUMAN.get(win, win),
                tsprev_h=_humanise_seconds(value),
            )
        except (ValueError, KeyError):
            break
    return f"{feature} = {value:.4g}" if np.isfinite(value) else f"{feature} = missing"


class ReasonCoder:
    """Wraps a fitted CalibratedFraudModel and emits top-k reason codes."""

    def __init__(self, model, background: pd.DataFrame | None = None) -> None:
        self.model = model
        self.explainer = shap.TreeExplainer(model.model)
        log.info("shap TreeExplainer ready")

    def explain_frame(self, X: pd.DataFrame, top_k: int = 5) -> list[list[dict]]:
        """Return, per row, the top_k features pushing the score UP."""
        feats = self.model.features
        sv = self.explainer.shap_values(X[feats])
        if isinstance(sv, list):  # older shap returns a list per class
            sv = sv[1]
        sv = np.asarray(sv)

        out = []
        values = X[feats].to_numpy(dtype=np.float64)
        for i in range(len(X)):
            row = sv[i]
            idx = np.argsort(-row)
            pos = [j for j in idx if row[j] > 0]

            # IEEE-CIS's V/C/D/id/M columns are anonymised by Vesta, so no
            # template can make them meaningful. Ranked purely by SHAP, a case
            # comes back as "V156; V197; V258" -- true, and useless to a human.
            # Reserve at least two slots for features that carry meaning.
            opaque = [j for j in pos if _OPAQUE.match(feats[j])]
            clear = [j for j in pos if not _OPAQUE.match(feats[j])]
            chosen = (opaque[: max(top_k - 2, 0)] + clear[:2]) if clear else opaque[:top_k]
            chosen = sorted(chosen, key=lambda j: -row[j])[:top_k]

            out.append([
                {
                    "feature": feats[j],
                    "value": float(values[i, j]),
                    "shap": float(row[j]),
                    "reason": humanise(feats[j], values[i, j]),
                }
                for j in chosen
            ])
        return out

    def review_queue(
        self,
        X: pd.DataFrame,
        p: np.ndarray,
        amounts: np.ndarray,
        ids: np.ndarray,
        top_k: int = 5,
        limit: int = 100,
        cost=None,
    ) -> pd.DataFrame:
        """
        The artefact an analyst actually opens: cases ranked by expected loss,
        each with its reason codes attached.
        """
        from .config import CONFIG
        cost = cost or CONFIG.cost

        expected_loss = p * (amounts + cost.chargeback_fee + cost.representment_labor)
        order = np.argsort(-expected_loss)[:limit]
        sub = X.iloc[order]
        reasons = self.explain_frame(sub, top_k=top_k)

        return pd.DataFrame({
            "transaction_id": ids[order],
            "p_fraud": np.round(p[order], 4),
            "amount": np.round(amounts[order], 2),
            "expected_loss": np.round(expected_loss[order], 2),
            "reasons": ["; ".join(r["reason"] for r in rs) for rs in reasons],
        }).reset_index(drop=True)

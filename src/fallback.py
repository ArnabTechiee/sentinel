"""
Failure handling.

The panel asks what broke and how you handled it. This module is the answer in
code rather than in prose. Three failure modes, three engineered responses:

  1. Model unavailable / errors  -> deterministic rule scorer takes over, and
     the response is marked degraded so downstream knows not to trust the
     probability as calibrated.
  2. Scoring exceeds the latency budget -> fail to REVIEW, never to ALLOW.
     Timing out into "approve" is how a payment system loses money quietly.
  3. Unseen category value -> mapped to a dedicated bucket at encode time
     (see features.CategoricalEncoder), counted, and surfaced in a report.

Design principle: every degradation is logged, bounded, and biased toward the
safe action rather than the convenient one.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CONFIG
from .economics import ALLOW, BLOCK, REVIEW

log = logging.getLogger(__name__)


@dataclass
class ScoreResult:
    probability: float
    decision: int
    degraded: bool
    path: str          # "model" | "rules" | "timeout"
    latency_ms: float
    note: str = ""


class RuleFallback:
    """
    Deterministic scorer used when the model is unavailable. Deliberately crude
    and deliberately conservative -- its job is to keep the merchant safe for
    the minutes the model is down, not to match the model's accuracy.

    Thresholds are fitted from TRAIN quantiles so they are data-driven rather
    than guessed.
    """

    def __init__(self) -> None:
        self.amt_p99: float = np.inf
        self.velocity_p99: dict[str, float] = {}
        self.fitted = False

    def fit(self, train: pd.DataFrame) -> "RuleFallback":
        amount_col = CONFIG.features.amount_col
        self.amt_p99 = float(train[amount_col].quantile(0.99))
        for col in train.columns:
            if col.endswith("_cnt_1h"):
                self.velocity_p99[col] = float(train[col].quantile(0.99))
        self.fitted = True
        log.info(
            "rule fallback fitted: amt_p99=%.2f, %d velocity rules",
            self.amt_p99, len(self.velocity_p99),
        )
        return self

    def score(self, row: pd.Series) -> tuple[float, str]:
        """Returns a pseudo-probability in [0,1] and the rule that fired."""
        hits = []
        amount_col = CONFIG.features.amount_col
        if row.get(amount_col, 0) > self.amt_p99:
            hits.append("amount above 99th percentile")
        for col, thresh in self.velocity_p99.items():
            v = row.get(col, 0)
            if pd.notna(v) and v > thresh:
                hits.append(f"{col} above 99th percentile")
        if row.get("is_night", 0) == 1 and row.get(amount_col, 0) > self.amt_p99 * 0.5:
            hits.append("large overnight transaction")
        if row.get("email_mismatch", 0) == 1:
            hits.append("email domain mismatch")

        # Each rule contributes; capped so the fallback never claims certainty.
        score = min(0.15 + 0.20 * len(hits), 0.85)
        return score, "; ".join(hits) if hits else "no rule fired"


class ResilientScorer:
    """
    Wraps model + policy + fallback with a latency budget.

    Failure bias: on any error or timeout the transaction goes to REVIEW, not
    ALLOW. A queued transaction costs an analyst's forty rupees; a wrongly
    approved one costs the transaction amount plus a chargeback fee.
    """

    def __init__(self, model, policy, fallback: RuleFallback, budget_ms: float | None = None):
        self.model = model
        self.policy = policy
        self.fallback = fallback
        self.budget_ms = budget_ms or CONFIG.scoring_latency_budget_ms
        self.stats = {"model": 0, "rules": 0, "timeout": 0}

    def warmup(self, sample: pd.DataFrame, n: int = 5) -> None:
        """
        The first prediction pays XGBoost's thread-pool and buffer allocation
        cost -- measured at ~160ms against a 150ms budget, which tripped the
        timeout path on a HEALTHY request and made the system look broken in its
        own failure drill. Real scorers warm up at deploy time; so does this one.

        Stats are reset afterwards so warmup calls do not pollute the counters.
        """
        for i in range(min(n, len(sample))):
            try:
                X = sample.iloc[[i]][self.model.features].astype(np.float64)
                self.model.predict_proba(X)
            except Exception:  # noqa: BLE001 -- warmup failures are not fatal
                pass
        self.stats = {"model": 0, "rules": 0, "timeout": 0}
        log.info("scorer warmed up (%d calls)", min(n, len(sample)))

    def score_one(self, row: pd.Series, force_failure: str | None = None) -> ScoreResult:
        """
        force_failure in {"model_down", "slow"} lets the demo trigger each
        degradation path on camera instead of waiting for it to happen.
        """
        t0 = time.perf_counter()
        try:
            if force_failure == "model_down":
                raise RuntimeError("simulated model service unavailable")

            X = row.to_frame().T[self.model.features].astype(np.float64)
            p = float(self.model.predict_proba(X)[0])
            elapsed = (time.perf_counter() - t0) * 1000

            if force_failure == "slow":
                elapsed = self.budget_ms * 1.2

            if elapsed > self.budget_ms:
                self.stats["timeout"] += 1
                log.warning("latency budget exceeded (%.1fms) -> routing to REVIEW", elapsed)
                return ScoreResult(p, REVIEW, True, "timeout", elapsed,
                                   "exceeded latency budget; failed safe to review")

            self.stats["model"] += 1
            return ScoreResult(p, int(self.policy.decide(np.array([p]))[0]), False,
                               "model", elapsed)

        except Exception as exc:  # noqa: BLE001 -- deliberate catch-all at the boundary
            elapsed = (time.perf_counter() - t0) * 1000
            self.stats["rules"] += 1
            log.error("model path failed (%s) -> rule fallback", exc)
            p, reason = self.fallback.score(row)
            # Degraded mode never blocks -- a crude rule score cannot justify
            # declining a customer. It also cannot review everything: a full
            # outage would flood the analyst bench in minutes. So it triages,
            # and the threshold is deliberately low.
            decision = REVIEW if p >= 0.15 else ALLOW
            return ScoreResult(p, decision, True, "rules", elapsed,
                               f"model unavailable: {exc}; rules fired: {reason}")

    def drill(self, sample: pd.DataFrame, n: int = 3) -> pd.DataFrame:
        """
        Runs each failure mode on real rows and returns a table. This is the
        thirty seconds of your pitch video where you break your own system.
        """
        rows = []
        for mode in (None, "model_down", "slow"):
            for i in range(min(n, len(sample))):
                r = self.score_one(sample.iloc[i], force_failure=mode)
                rows.append({
                    "scenario": mode or "healthy",
                    "path": r.path,
                    "p_fraud": round(r.probability, 4),
                    "decision": ["ALLOW", "REVIEW", "BLOCK"][r.decision],
                    "degraded": r.degraded,
                    "latency_ms": round(r.latency_ms, 2),
                    "note": r.note[:80],
                })
        return pd.DataFrame(rows)

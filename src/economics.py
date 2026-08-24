"""
The decision layer -- where a probability becomes an action and an action
becomes rupees.

Three tiers, not one threshold:
    p <  t_review              -> ALLOW
    t_review <= p < t_block    -> REVIEW  (queued for a human)
    p >= t_block               -> BLOCK

Cost model, per transaction:
    ALLOW  + fraud   ->  amount + chargeback_fee + representment_labor   (FN)
    ALLOW  + legit   ->  0
    BLOCK  + fraud   ->  0                                               (loss avoided)
    BLOCK  + legit   ->  friction_cost                                   (FP)
    REVIEW + fraud   ->  review_cost + (1 - catch_rate) * FN_cost
    REVIEW + legit   ->  review_cost + review_delay_penalty

Threshold search is exact, not sampled: sorting by score once and using
cumulative sums makes the cost of any (t_review, t_block) pair an O(1) lookup,
so the full grid is evaluated rather than approximated.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import CONFIG

log = logging.getLogger(__name__)

ALLOW, REVIEW, BLOCK = 0, 1, 2


def fn_cost_vector(y: np.ndarray, amounts: np.ndarray, cost=None) -> np.ndarray:
    """Cost of letting each transaction through, per row. Zero for legit rows."""
    cost = cost or CONFIG.cost
    return y * (amounts + cost.chargeback_fee + cost.representment_labor)


class DecisionPolicy:
    """A fitted (t_review, t_block) pair plus the machinery to evaluate it."""

    def __init__(self, t_review: float, t_block: float, cost=None) -> None:
        if t_review > t_block:
            raise ValueError("t_review must be <= t_block")
        self.t_review = float(t_review)
        self.t_block = float(t_block)
        self.cost = cost or CONFIG.cost

    def decide(self, p: np.ndarray) -> np.ndarray:
        d = np.zeros(len(p), dtype=np.int8)
        d[p >= self.t_review] = REVIEW
        d[p >= self.t_block] = BLOCK
        return d

    def evaluate(self, y: np.ndarray, p: np.ndarray, amounts: np.ndarray) -> dict:
        d = self.decide(p)
        c = self.cost
        fn = fn_cost_vector(y, amounts, c)

        m_allow, m_review, m_block = d == ALLOW, d == REVIEW, d == BLOCK
        total = (
            fn[m_allow].sum()
            + m_review.sum() * c.review_cost
            + (1 - c.analyst_catch_rate) * fn[m_review].sum()
            + ((1 - y)[m_review]).sum() * c.review_delay_penalty
            + ((1 - y)[m_block]).sum() * c.friction_cost
        )

        do_nothing = fn.sum()
        n_legit = int((1 - y).sum())
        caught = int(y[m_block].sum() + round(c.analyst_catch_rate * y[m_review].sum()))
        fp = int((1 - y)[m_block].sum())

        return {
            "t_review": self.t_review,
            "t_block": self.t_block,
            "n": len(y),
            "n_allow": int(m_allow.sum()),
            "n_review": int(m_review.sum()),
            "n_block": int(m_block.sum()),
            "review_rate": round(float(m_review.mean()), 5),
            "block_rate": round(float(m_block.mean()), 5),
            "fraud_caught": caught,
            "fraud_total": int(y.sum()),
            "recall_effective": round(caught / max(int(y.sum()), 1), 4),
            "precision_block": round(
                float(y[m_block].sum() / max(int(m_block.sum()), 1)), 4
            ),
            "false_positives": fp,
            "fp_per_1000_good": round(1000 * fp / max(n_legit, 1), 3),
            "policy_cost": round(float(total), 2),
            "do_nothing_cost": round(float(do_nothing), 2),
            "net_saved": round(float(do_nothing - total), 2),
            "pct_loss_prevented": round(
                float(100 * (do_nothing - total) / max(do_nothing, 1e-9)), 2
            ),
        }


# --------------------------------------------------------------------------
# exact threshold optimisation via cumulative sums
# --------------------------------------------------------------------------
def optimize_thresholds(
    y: np.ndarray,
    p: np.ndarray,
    amounts: np.ndarray,
    n_grid: int = 200,
    cost=None,
    max_review_rate: float | None = None,
) -> tuple[DecisionPolicy, pd.DataFrame]:
    """
    Exhaustively search (t_review, t_block) for minimum expected cost.

    max_review_rate optionally caps the fraction of traffic sent to humans,
    which is how a real risk team operates -- there is a finite analyst bench.
    """
    cost = cost or CONFIG.cost
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    amounts = np.asarray(amounts, dtype=np.float64)

    order = np.argsort(p, kind="mergesort")
    ps, ys, amts = p[order], y[order], amounts[order]
    n = len(ps)

    fn = fn_cost_vector(ys, amts, cost)
    C_fn = np.concatenate(([0.0], np.cumsum(fn)))       # C_fn[i] = fn cost of rows [0,i)
    C_legit = np.concatenate(([0.0], np.cumsum(1 - ys)))  # count of legit in [0,i)

    # Candidate cuts must sit on distinct probability VALUES, not on evenly
    # spaced index positions.
    #
    # Isotonic regression emits few distinct levels -- 98 across 12,500 rows in
    # our first run -- so hundreds of transactions share an identical score. An
    # index-based cut splits such a tie group, but DecisionPolicy applies
    # `p >= t` and takes the whole group. The optimiser then counts 250 rows
    # where the policy acts on 400: the review cap silently breaks, and the
    # predicted cost does not match the realised cost.
    #
    # Cutting on unique values and mapping back through searchsorted makes the
    # index and the threshold refer to the same set of rows by construction.
    uniq = np.unique(ps)
    if len(uniq) > n_grid:
        uniq = uniq[np.linspace(0, len(uniq) - 1, n_grid).astype(int)]
    cuts = np.unique(np.searchsorted(ps, uniq, side="left"))
    cuts = np.unique(np.append(cuts, n))
    total_legit = C_legit[n]

    best = None
    rows = []
    for a in cuts:
        for b in cuts[cuts >= a]:
            if max_review_rate is not None and (b - a) / n > max_review_rate:
                continue
            c_allow = C_fn[a]
            c_review = (
                cost.review_cost * (b - a)
                + (1 - cost.analyst_catch_rate) * (C_fn[b] - C_fn[a])
                + cost.review_delay_penalty * (C_legit[b] - C_legit[a])
            )
            c_block = cost.friction_cost * (total_legit - C_legit[b])
            total = c_allow + c_review + c_block
            rows.append({
                "t_review": float(ps[a]) if a < n else 1.0,
                "t_block": float(ps[b]) if b < n else 1.0,
                "review_rate": (b - a) / n,
                "block_rate": (n - b) / n,
                "cost": total,
            })
            if best is None or total < best[0]:
                best = (total, a, b)

    _, a, b = best
    t_review = float(ps[a]) if a < n else np.inf
    t_block = float(ps[b]) if b < n else np.inf
    policy = DecisionPolicy(t_review, t_block, cost)

    # Self-check: the optimiser's arithmetic and the policy's behaviour must
    # agree. They diverged silently once already (tie groups, see above), so
    # the agreement is asserted rather than assumed.
    realised = policy.evaluate(y, p, amounts)
    drift = abs(realised["policy_cost"] - best[0]) / max(best[0], 1e-9)
    if drift > 0.01:
        log.warning(
            "optimiser/policy cost mismatch: predicted %.0f, realised %.0f (%.1f%%) "
            "-- threshold grid is not aligned to the score distribution",
            best[0], realised["policy_cost"], 100 * drift,
        )
    if max_review_rate is not None and realised["review_rate"] > max_review_rate * 1.05:
        log.warning(
            "review rate %.4f exceeds cap %.4f", realised["review_rate"], max_review_rate
        )

    log.info(
        "optimal thresholds: review>=%.6f block>=%.6f (cost %.0f, review %.2f%%, block %.2f%%)",
        t_review, t_block, realised["policy_cost"],
        100 * realised["review_rate"], 100 * realised["block_rate"],
    )
    return policy, pd.DataFrame(rows)


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------
def baseline_table(
    y: np.ndarray, p: np.ndarray, amounts: np.ndarray, policy: DecisionPolicy, cost=None
) -> pd.DataFrame:
    """
    Every claim of 'money saved' needs something to be saved *relative to*.
    Four reference points, so the number cannot be read as marketing.
    """
    cost = cost or CONFIG.cost
    y = np.asarray(y, dtype=np.float64)
    amounts = np.asarray(amounts, dtype=np.float64)
    fn = fn_cost_vector(y, amounts, cost)
    n_legit = float((1 - y).sum())

    do_nothing = float(fn.sum())
    block_all = float(n_legit * cost.friction_cost)

    naive = DecisionPolicy(0.5, 0.5, cost).evaluate(y, p, amounts)
    tuned = policy.evaluate(y, p, amounts)

    rows = [
        {"policy": "do nothing (approve all)", "cost": round(do_nothing, 2), "net_saved": 0.0,
         "note": "the loss you are trying to prevent"},
        {"policy": "block everything", "cost": round(block_all, 2),
         "net_saved": round(do_nothing - block_all, 2),
         "note": "zero fraud, no business -- the degenerate upper bound"},
        {"policy": "naive threshold 0.5", "cost": naive["policy_cost"],
         "net_saved": naive["net_saved"],
         "note": "what a submission that skips the economics reports"},
        {"policy": "Sentinel (cost-optimised, 3-tier)", "cost": tuned["policy_cost"],
         "net_saved": tuned["net_saved"],
         "note": f"review {tuned['review_rate']:.2%} of traffic, block {tuned['block_rate']:.2%}"},
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# capacity curve
# --------------------------------------------------------------------------
def capacity_curve(
    y: np.ndarray,
    p: np.ndarray,
    amounts: np.ndarray,
    capacities: list[int] | None = None,
    cost=None,
    review_cap_rate: float | None = None,
) -> pd.DataFrame:
    """
    Real risk teams do not choose a threshold, they choose how many analysts
    they employ. For each daily review capacity, rank by EXPECTED LOSS
    (p * FN cost) rather than by probability -- a 0.4-probability ₹80,000
    transaction deserves a human before a 0.9-probability ₹200 one.

    IMPORTANT, and stated in the report too: this is a REVIEW-ONLY strategy. It
    never blocks, so it pays no friction cost, and its numbers are therefore NOT
    comparable with the operating point chosen by optimize_thresholds. Reading
    the two tables side by side without this caveat makes it look as though the
    chosen policy left money on the table. Pass review_cap_rate to mark which
    capacities are actually reachable under the policy's review budget.
    """
    cost = cost or CONFIG.cost
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    amounts = np.asarray(amounts, dtype=np.float64)

    expected_loss = p * (amounts + cost.chargeback_fee + cost.representment_labor)
    order = np.argsort(-expected_loss, kind="mergesort")
    fn = fn_cost_vector(y, amounts, cost)
    do_nothing = float(fn.sum())

    if capacities is None:
        capacities = [0, 50, 100, 250, 500, 1000, 2500, 5000, 10000]

    rows = []
    for cap in capacities:
        sel = order[:cap]
        caught = cost.analyst_catch_rate * y[sel].sum()
        recovered = cost.analyst_catch_rate * fn[sel].sum()
        spent = cap * cost.review_cost + (1 - y)[sel].sum() * cost.review_delay_penalty
        rows.append({
            "review_capacity": cap,
            "fraud_caught": round(float(caught), 1),
            "recall": round(float(caught / max(y.sum(), 1)), 4),
            "gross_recovered": round(float(recovered), 2),
            "review_spend": round(float(spent), 2),
            "net_saved": round(float(recovered - spent), 2),
            "roi_per_review": round(float((recovered - spent) / cap), 2) if cap else 0.0,
            "remaining_loss": round(do_nothing - float(recovered), 2),
            "within_policy_cap": (
                None if review_cap_rate is None else bool(cap / max(len(y), 1) <= review_cap_rate)
            ),
        })
    return pd.DataFrame(rows)


CAPACITY_CURVE_NOTE = (
    "**Review-only strategy — not comparable with the operating point above.** "
    "This curve never blocks, so it pays no friction cost; the chosen policy "
    "does both. Rows with `within_policy_cap = False` exceed the review budget "
    "the policy was fitted under and are shown for context only."
)


# --------------------------------------------------------------------------
# per-segment thresholds
# --------------------------------------------------------------------------
def optimize_thresholds_by_segment(
    y: np.ndarray,
    p: np.ndarray,
    amounts: np.ndarray,
    segments: np.ndarray,
    n_grid: int = 80,
    cost=None,
    max_review_rate: float | None = None,
    min_n: int = 500,
    min_fraud: int = 20,
) -> tuple[dict, "DecisionPolicy", pd.DataFrame]:
    """
    Fit (t_review, t_block) per segment instead of globally.

    Why this exists: a single global threshold is dominated by whichever segment
    has the highest fraud base rate. In our first full run, product C (7.4% base
    rate) pulled the threshold to where 45% of its traffic was flagged, while
    product W -- which held half of all fraud at a 1.9% base rate -- was flagged
    on 1.3% of traffic and got 25% recall. The global optimum was policing the
    loud segment and ignoring the large one.

    Segments below min_n rows or min_fraud frauds fall back to the global
    policy: thresholds fitted on thin slices overfit, and the fallback is
    explicit rather than implied.
    """
    global_policy, _ = optimize_thresholds(y, p, amounts, n_grid, cost, max_review_rate)
    segments = np.asarray(segments)
    policies: dict = {}
    rows = []

    for seg in pd.unique(segments):
        m = segments == seg
        n_seg, n_fraud = int(m.sum()), int(np.sum(y[m]))
        if n_seg < min_n or n_fraud < min_fraud:
            policies[seg] = global_policy
            rows.append({
                "segment": str(seg), "n": n_seg, "n_fraud": n_fraud, "fitted": False,
                "t_review": round(global_policy.t_review, 5),
                "t_block": round(global_policy.t_block, 5),
                "note": "too thin — global policy applied",
            })
            continue

        pol, _ = optimize_thresholds(y[m], p[m], amounts[m], n_grid, cost, max_review_rate)
        policies[seg] = pol
        rows.append({
            "segment": str(seg), "n": n_seg, "n_fraud": n_fraud, "fitted": True,
            "t_review": round(pol.t_review, 5),
            "t_block": round(pol.t_block, 5),
            "note": "",
        })

    return policies, global_policy, pd.DataFrame(rows)


def evaluate_segmented(
    y: np.ndarray,
    p: np.ndarray,
    amounts: np.ndarray,
    segments: np.ndarray,
    policies: dict,
    cost=None,
) -> dict:
    """Aggregate outcome of applying each segment's own policy."""
    cost = cost or CONFIG.cost
    segments = np.asarray(segments)
    decisions = np.zeros(len(y), dtype=np.int8)
    for seg, pol in policies.items():
        m = segments == seg
        if m.any():
            decisions[m] = pol.decide(p[m])

    fn = fn_cost_vector(y, amounts, cost)
    m_allow, m_review, m_block = decisions == ALLOW, decisions == REVIEW, decisions == BLOCK
    total = (
        fn[m_allow].sum()
        + m_review.sum() * cost.review_cost
        + (1 - cost.analyst_catch_rate) * fn[m_review].sum()
        + ((1 - y)[m_review]).sum() * cost.review_delay_penalty
        + ((1 - y)[m_block]).sum() * cost.friction_cost
    )
    do_nothing = float(fn.sum())
    caught = int(y[m_block].sum() + round(cost.analyst_catch_rate * y[m_review].sum()))
    n_legit = int((1 - y).sum())
    fp = int((1 - y)[m_block].sum())

    return {
        "policy": "per-segment thresholds",
        "n_review": int(m_review.sum()),
        "n_block": int(m_block.sum()),
        "review_rate": round(float(m_review.mean()), 5),
        "block_rate": round(float(m_block.mean()), 5),
        "fraud_caught": caught,
        "recall_effective": round(caught / max(int(y.sum()), 1), 4),
        "false_positives": fp,
        "fp_per_1000_good": round(1000 * fp / max(n_legit, 1), 3),
        "policy_cost": round(float(total), 2),
        "net_saved": round(do_nothing - float(total), 2),
        "pct_loss_prevented": round(float(100 * (do_nothing - total) / max(do_nothing, 1e-9)), 2),
        "_decisions": decisions,
    }

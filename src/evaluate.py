"""
Evaluation.

Everything reported here is on the temporal test slice the model never saw.
Accuracy is deliberately absent -- at a 3.5% base rate, approving everything
scores 96.5% and catches nothing.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

log = logging.getLogger(__name__)


def headline_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "n": int(len(y)),
        "n_fraud": int(np.sum(y)),
        "base_rate": round(float(np.mean(y)), 5),
        "pr_auc": round(float(average_precision_score(y, p)), 5),
        "roc_auc": round(float(roc_auc_score(y, p)), 5),
        "brier": round(float(brier_score_loss(y, p)), 6),
        "lift_over_base_rate_at_top_1pct": round(
            float(np.mean(y[np.argsort(-p)[: max(1, len(y) // 100)]]) / max(np.mean(y), 1e-9)), 2
        ),
    }


def threshold_table(y: np.ndarray, p: np.ndarray, points: list[float] | None = None) -> pd.DataFrame:
    """Precision / recall / FP-per-1000-good across candidate block thresholds."""
    points = points or [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]
    n_legit = float(np.sum(1 - y))
    rows = []
    for t in points:
        flag = p >= t
        tp = float(np.sum(y[flag]))
        fp = float(np.sum(1 - y[flag]))
        rows.append({
            "threshold": t,
            "flagged": int(flag.sum()),
            "flag_rate": round(float(flag.mean()), 5),
            "precision": round(tp / max(flag.sum(), 1), 4),
            "recall": round(tp / max(float(np.sum(y)), 1), 4),
            "false_positives": int(fp),
            "fp_per_1000_good": round(1000 * fp / max(n_legit, 1), 2),
        })
    return pd.DataFrame(rows)


def pr_curve_frame(y: np.ndarray, p: np.ndarray, n_points: int = 100) -> pd.DataFrame:
    prec, rec, thr = precision_recall_curve(y, p)
    idx = np.linspace(0, len(thr) - 1, min(n_points, len(thr))).astype(int)
    return pd.DataFrame({
        "threshold": thr[idx],
        "precision": prec[idx],
        "recall": rec[idx],
    })


def write_markdown_report(path, sections: dict[str, pd.DataFrame | dict | str]) -> None:
    """Dump every table into one markdown file -- this becomes RESULTS.md."""
    lines = ["# Sentinel — results\n"]
    for title, obj in sections.items():
        lines.append(f"\n## {title}\n")
        if isinstance(obj, pd.DataFrame):
            lines.append(obj.to_markdown(index=False))
        elif isinstance(obj, dict):
            lines.append(
                "| metric | value |\n|---|---|\n"
                + "\n".join(f"| {k} | {v} |" for k, v in obj.items())
            )
        else:
            lines.append(str(obj))
        lines.append("")
    path.write_text("\n".join(lines))
    log.info("wrote %s", path)

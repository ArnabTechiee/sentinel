"""
Model + calibration.

The calibration step is the load-bearing part of this project and the reason
the economics downstream mean anything. Raw gradient-boosted scores are ranking
scores, not probabilities: a score of 0.30 does not mean "30% of these are
fraud". The decision layer multiplies P(fraud) by a rupee amount, so if the
probability is wrong the money numbers are fiction.

Isotonic regression is fitted on a held-out, out-of-time calibration slice.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier

from .config import CONFIG

log = logging.getLogger(__name__)


class CalibratedFraudModel:
    """XGBoost ranker + isotonic probability calibration."""

    def __init__(self, cfg=None) -> None:
        self.cfg = cfg or CONFIG.model
        self.model: XGBClassifier | None = None
        self.calibrator: IsotonicRegression | None = None
        self.features: list[str] = []
        self.scale_pos_weight: float = 1.0

    # ---------------------------------------------------------------- fit
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_calib: pd.DataFrame,
        y_calib: np.ndarray,
    ) -> "CalibratedFraudModel":
        self.features = list(X_train.columns)
        pos = float(np.sum(y_train))
        neg = float(len(y_train) - pos)
        self.scale_pos_weight = neg / max(pos, 1.0)

        self.model = XGBClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            learning_rate=self.cfg.learning_rate,
            subsample=self.cfg.subsample,
            colsample_bytree=self.cfg.colsample_bytree,
            min_child_weight=self.cfg.min_child_weight,
            reg_lambda=self.cfg.reg_lambda,
            random_state=self.cfg.random_state,
            n_jobs=self.cfg.n_jobs,
            tree_method=self.cfg.tree_method,
            eval_metric="aucpr",
            early_stopping_rounds=self.cfg.early_stopping_rounds,
            # NOTE: scale_pos_weight distorts probabilities badly. We deliberately
            # do NOT use it -- isotonic calibration handles imbalance properly and
            # keeps the output on the real probability scale.
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_calib, y_calib)],
            verbose=False,
        )
        best = getattr(self.model, "best_iteration", None)
        log.info("xgboost fitted (best_iteration=%s)", best)

        raw_calib = self._raw_predict(X_calib)
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.calibrator.fit(raw_calib, y_calib)

        before = brier_score_loss(y_calib, raw_calib)
        after = brier_score_loss(y_calib, self.calibrator.predict(raw_calib))
        log.info("brier on calibration slice: raw=%.6f -> calibrated=%.6f", before, after)
        return self

    # ------------------------------------------------------------ predict
    def _raw_predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.features])[:, 1]

    PROB_CEILING = 0.999

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calibrated P(fraud), capped strictly below 1.0.

        Isotonic regression saturates: if the top bin of the calibration slice
        is pure fraud, every score in it maps to exactly 1.0. A model fitted on
        tens of thousands of rows cannot justify a posterior of 1.0, and a
        certainty claim is the first thing a reviewer will attack. The cap is a
        statement about what the evidence supports, not a cosmetic clip.
        """
        raw = self._raw_predict(X)
        if self.calibrator is None:
            return raw
        return np.clip(self.calibrator.predict(raw), 1e-6, self.PROB_CEILING)

    def calibration_resolution(self, p: np.ndarray) -> dict:
        """
        Isotonic output is piecewise-constant, so the calibrated scores take a
        limited number of distinct values. Few levels means coarse threshold
        control -- if the chosen thresholds look like odd fractions (0.0714 =
        1/14), that is why. Reported rather than left to be discovered.
        """
        u = np.unique(p)
        return {
            "distinct_calibrated_values": int(len(u)),
            "n_at_ceiling": int((p >= self.PROB_CEILING).sum()),
            "max_probability": round(float(p.max()), 6),
            "median_gap_between_levels": round(
                float(np.median(np.diff(u))) if len(u) > 1 else 0.0, 6
            ),
        }

    def velocity_gain_share(self) -> tuple[pd.DataFrame, float]:
        """
        How much of the model's total gain comes from the velocity features we
        engineered, as opposed to the dataset's own C-series counters. If the
        share is small, say so -- an honest negative result is worth more than
        an implied claim the gain table contradicts.
        """
        imp = self.feature_importance(top_n=10_000)
        mask = imp["feature"].str.contains(r"_cnt_|_amt_|_amtratio_|_tsprev", regex=True)
        share = float(imp.loc[mask, "gain"].sum() / max(imp["gain"].sum(), 1e-9))
        return imp[mask].head(15).reset_index(drop=True), round(share, 4)

    # ----------------------------------------------------------- reports
    def reliability_table(self, y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
        """
        Predicted vs observed fraud rate per probability bin. If calibration
        worked, predicted_mean tracks observed_rate down the table. Put this in
        the README -- it is the evidence that the money numbers are legitimate.
        """
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
        rows = []
        for b in range(len(edges) - 1):
            m = bins == b
            if not m.any():
                continue
            rows.append({
                "bin": b,
                "n": int(m.sum()),
                "predicted_mean": round(float(p[m].mean()), 6),
                "observed_rate": round(float(y[m].mean()), 6),
                "gap": round(float(p[m].mean() - y[m].mean()), 6),
            })
        return pd.DataFrame(rows)

    def feature_importance(self, top_n: int = 30) -> pd.DataFrame:
        imp = self.model.get_booster().get_score(importance_type="gain")
        return (
            pd.DataFrame([{"feature": k, "gain": v} for k, v in imp.items()])
            .sort_values("gain", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

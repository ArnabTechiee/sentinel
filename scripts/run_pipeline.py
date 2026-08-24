"""
End-to-end run. Produces reports/RESULTS.md and the CSV artefacts behind it.

    python scripts/run_pipeline.py                 # full IEEE-CIS
    python scripts/run_pipeline.py --nrows 50000   # fast iteration
    python scripts/run_pipeline.py --smoke         # pipeline check on fake data
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CONFIG, REPORT_DIR  # noqa: E402
from src.data import load_ieee_cis, temporal_split, split_summary  # noqa: E402
from src import economics as econ  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src.explain import ReasonCoder  # noqa: E402
from src.fallback import ResilientScorer, RuleFallback  # noqa: E402
from src.features import (  # noqa: E402
    CategoricalEncoder,
    build_features,
    detect_categoricals,
    feature_columns,
)
from src.model import CalibratedFraudModel  # noqa: E402
from src.sentinel import SpikeSentinel, psi_report  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)-18s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrows", type=int, default=None, help="limit rows for fast iteration")
    ap.add_argument("--smoke", action="store_true", help="results are meaningless; pipeline check only")
    ap.add_argument("--review-cap", type=float, default=0.02, help="max fraction of traffic sent to humans")
    args = ap.parse_args()

    t0 = time.time()
    sections: dict = {}

    if args.smoke:
        log.warning("=" * 70)
        log.warning("SMOKE MODE — synthetic data. No metric below is reportable.")
        log.warning("=" * 70)

    # ---------------------------------------------------------------- load
    df = load_ieee_cis(nrows=args.nrows)
    target, idcol = CONFIG.target_col, CONFIG.id_col

    # ------------------------------------------------------------ features
    df = build_features(df)
    cats = detect_categoricals(df)
    train_raw, calib_raw, test_raw = temporal_split(df)
    sections["Temporal split"] = split_summary(train_raw, calib_raw, test_raw)

    enc = CategoricalEncoder().fit(train_raw, cats)
    train = enc.transform(train_raw)
    calib = enc.transform(calib_raw)
    test = enc.transform(test_raw)
    sections["Unseen categories at test time (fallback bucket)"] = enc.unseen_report().head(15)

    feats = feature_columns(train)
    feats = [f for f in feats if pd.api.types.is_numeric_dtype(train[f])]
    log.info("using %d features", len(feats))

    X_tr, y_tr = train[feats], train[target].to_numpy()
    X_ca, y_ca = calib[feats], calib[target].to_numpy()
    X_te, y_te = test[feats], test[target].to_numpy()
    amt_te = test[CONFIG.features.amount_col].to_numpy(dtype=float)

    # --------------------------------------------------------------- model
    model = CalibratedFraudModel().fit(X_tr, y_tr, X_ca, y_ca)
    p_te = model.predict_proba(X_te)

    sections["Headline metrics (temporal holdout)"] = ev.headline_metrics(y_te, p_te)
    sections["Calibration — predicted vs observed"] = model.reliability_table(y_te, p_te)
    sections["Calibration resolution"] = model.calibration_resolution(p_te)
    sections["Precision / recall / FP cost by threshold"] = ev.threshold_table(y_te, p_te)
    sections["Top features by gain"] = model.feature_importance(20)

    vel_tbl, vel_share = model.velocity_gain_share()
    sections["Do our engineered velocity features earn their place?"] = (
        f"Velocity features account for **{vel_share:.2%}** of total model gain. "
        + (
            "They carry real weight alongside the dataset's own C-series counters."
            if vel_share >= 0.03
            else "This is a small share: IEEE-CIS ships its own C-series velocity "
            "counters, which largely subsume ours. Reported honestly rather than "
            "implied otherwise. They are retained because the rule fallback and "
            "the reason codes both depend on them, and neither can read a V-column."
        )
    )
    if not vel_tbl.empty:
        sections["Velocity features by gain"] = vel_tbl

    # ----------------------------------------------------------- economics
    # Thresholds are fitted on the CALIBRATION slice, never on test.
    #
    # This was wrong in the first full run: the optimiser saw the test labels,
    # chose the cutoffs that minimised cost on them, and the reported saving was
    # therefore in-sample. The model was held out; the decision layer was not.
    # Cutoffs are parameters like any other and need their own fitting window.
    p_ca = model.predict_proba(X_ca)
    amt_ca = calib[CONFIG.features.amount_col].to_numpy(dtype=float)

    policy, curve = econ.optimize_thresholds(
        y_ca, p_ca, amt_ca, n_grid=200, max_review_rate=args.review_cap
    )
    result = policy.evaluate(y_te, p_te, amt_te)          # out-of-sample

    # How much does fitting thresholds without seeing the future actually cost?
    #
    # The naive diagnostic -- compare performance on the calibration window with
    # performance on test -- conflates two different things. Our calibration
    # window has a 4.04% fraud rate against test's 3.45%, so a larger share of
    # its rupees are preventable and it scores higher no matter how good the
    # thresholds are. That gap is prevalence, not overfitting.
    #
    # The right comparison is against an ORACLE: thresholds fitted directly on
    # the test labels. That is the best any cutoff could have done with perfect
    # hindsight, so the difference is exactly the price of not knowing.
    oracle, _ = econ.optimize_thresholds(
        y_te, p_te, amt_te, n_grid=200, max_review_rate=args.review_cap
    )
    oracle_result = oracle.evaluate(y_te, p_te, amt_te)
    in_sample = policy.evaluate(y_ca, p_ca, amt_ca)

    sections["Chosen operating point (fitted on calibration, evaluated on test)"] = result
    sections["Threshold generalisation"] = {
        "fitted_on": "calibration slice (never sees test labels)",
        "test_pct_loss_prevented": result["pct_loss_prevented"],
        "oracle_pct_loss_prevented": oracle_result["pct_loss_prevented"],
        "price_of_hindsight_pp": round(
            oracle_result["pct_loss_prevented"] - result["pct_loss_prevented"], 3
        ),
        "price_of_hindsight_rupees": round(
            oracle_result["net_saved"] - result["net_saved"], 2
        ),
        "calib_pct_loss_prevented": in_sample["pct_loss_prevented"],
        "calib_fraud_rate": round(float(y_ca.mean()), 5),
        "test_fraud_rate": round(float(y_te.mean()), 5),
        "note": "Compare test against ORACLE, not against calibration. The "
                "calibration window has a higher fraud rate, so it scores higher "
                "regardless of threshold quality. A small oracle gap means the "
                "cutoffs transfer.",
    }
    money = econ.baseline_table(y_te, p_te, amt_te, policy)

    # ---- per-segment thresholds -------------------------------------------
    seg_col = next((c for c in ["ProductCD", "card4", "card6"] if c in test_raw.columns), None)
    seg_result = None
    if seg_col is not None:
        segs_ca = calib_raw[seg_col].astype(str).to_numpy()
        segs_te = test_raw[seg_col].astype(str).to_numpy()
        seg_policies, _, seg_tbl = econ.optimize_thresholds_by_segment(
            y_ca, p_ca, amt_ca, segs_ca, n_grid=200, max_review_rate=args.review_cap
        )
        seg_result = econ.evaluate_segmented(y_te, p_te, amt_te, segs_te, seg_policies)
        sections[f"Per-segment thresholds (fitted on calibration, by {seg_col})"] = seg_tbl
        money = pd.concat([money, pd.DataFrame([{
            "policy": f"Sentinel + per-segment thresholds ({seg_col})",
            "cost": seg_result["policy_cost"],
            "net_saved": seg_result["net_saved"],
            "note": f"recall {seg_result['recall_effective']:.3f}, "
                    f"{seg_result['fp_per_1000_good']} FP per 1000 good",
        }])], ignore_index=True)
        sections["Per-segment operating point"] = {
            k: v for k, v in seg_result.items() if not k.startswith("_")
        }

    sections["Money: Sentinel vs baselines"] = money
    sections["Review-capacity curve"] = econ.CAPACITY_CURVE_NOTE
    sections["Review-capacity curve (review-only, no blocking)"] = econ.capacity_curve(
        y_te, p_te, amt_te, review_cap_rate=args.review_cap
    )
    curve.sort_values("cost").head(10).to_csv(REPORT_DIR / "threshold_grid.csv", index=False)

    # ------------------------------------------------------------ sentinel
    monitor_feats = [f for f in ["amt_log", "hour", "dist1", "card1_cnt_1h", "card1_amtratio_24h"] if f in test.columns]
    sections["Feature drift, train vs test (PSI)"] = psi_report(train, test, monitor_feats)

    seg_cols = [c for c in ["ProductCD", "card4", "card6", "DeviceInfo"] if c in test_raw.columns]
    sent = SpikeSentinel(seg_cols)

    # A 3-day holdout with daily buckets gives CUSUM nothing to work with, so
    # the bucket width is derived from the actual span rather than hard-coded.
    CONFIG.sentinel.window_days = SpikeSentinel.choose_window_days(test_raw)
    log.info("sentinel bucket width: %.3f days", CONFIG.sentinel.window_days)

    scan, skipped = sent.scan(test_raw, p_te, y_te)
    sections["Spike sentinel — segment scan"] = (
        scan.head(12) if not scan.empty
        else f"No segments met the monitoring criteria "
             f"({len(skipped)} skipped — see below). Bucket width "
             f"{CONFIG.sentinel.window_days:.2f} days."
    )
    if not skipped.empty:
        sections["Spike sentinel — segments not monitored, and why"] = skipped.head(12)

    decisions = seg_result["_decisions"] if seg_result else policy.decide(p_te)
    if seg_cols:
        sections["Where it fails — recall by segment"] = sent.recall_by_segment(
            test_raw, y_te, decisions, seg_cols[0]
        )

    # ------------------------------------------------------------- explain
    try:
        coder = ReasonCoder(model)
        queue = coder.review_queue(
            X_te, p_te, amt_te, test[idcol].to_numpy(), limit=15
        )
        sections["Analyst review queue (top 15 by expected loss)"] = queue
        queue.to_csv(REPORT_DIR / "review_queue.csv", index=False)
    except Exception as exc:  # noqa: BLE001
        log.error("reason codes failed: %s", exc)

    # ------------------------------------------------------- failure drill
    fb = RuleFallback().fit(train)
    scorer = ResilientScorer(model, policy, fb)
    scorer.warmup(X_te.head(5))
    sections["Failure drill — degradation paths"] = scorer.drill(X_te.head(3), n=1)

    # -------------------------------------------------------------- output
    banner = (
        "> **SMOKE RUN — synthetic data. These numbers are not reportable.**\n"
        if args.smoke else
        "All metrics below are on a temporal holdout the model never saw.\n"
    )
    sections = {"Run notes": banner + f"\nElapsed: {time.time() - t0:.1f}s"} | sections
    ev.write_markdown_report(REPORT_DIR / "RESULTS.md", sections)

    print("\n" + "=" * 70)
    print(f"done in {time.time() - t0:.1f}s -> {REPORT_DIR / 'RESULTS.md'}")
    print(f"net saved vs do-nothing: Rs {result['net_saved']:,.0f} "
          f"({result['pct_loss_prevented']}% of exposure)")
    print(f"recall {result['recall_effective']:.3f} | "
          f"FP per 1000 good {result['fp_per_1000_good']}")
    print("=" * 70)


if __name__ == "__main__":
    main()

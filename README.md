# Sentinel

**A chargeback-loss minimiser, not a fraud classifier.**

Most fraud submissions answer "can you predict fraud?" — a question with a
well-known answer. Sentinel answers the question a merchant's risk team
actually asks:

> Given a fixed analyst review budget, which transactions do I stop, and how
> much money did that save me *net of the good customers I annoyed*?

Razorpay AI Buildathon 2026 · Track 02, AI Risk Manager.
Defence-only: no fraud-pattern generator, no evasion tooling, nothing runnable
offensively. Public dataset (IEEE-CIS), no synthetic fraud.

---

## Result

147,635 transactions in a 52-day temporal holdout the model never saw.
3.45% fraud base rate. ₹9.21M of preventable loss on the table.

| policy | net saved | % of exposure |
|---|---:|---:|
| do nothing (approve all) | ₹0 | 0% |
| block everything | −₹90.6M | catastrophic |
| naive 0.5 threshold | ₹2,653,990 | 28.8% |
| **Sentinel** | **₹3,636,525** | **39.5%** |

Recall 48.8%, precision on blocks 68.5%, **6.1 false positives per 1,000 good
customers**, 2.0% of traffic reviewed and 1.9% blocked.
PR-AUC 0.486 at a 3.45% base rate; 24.8× lift in the top percentile.

**₹982,535 more than the naive threshold, from the identical model.** Not a
better classifier — a better decision layer.

---

## Why you can believe the number

**Thresholds are fitted on the calibration slice, never on test.**
Cutoffs are parameters and need their own fitting window. An earlier version of
this pipeline optimised them on the test labels; the model was held out but the
decision layer was not, and the reported saving was in-sample.

The honest way to measure what that costs is against an **oracle** — thresholds
fitted directly on the test labels, the best any cutoff could do with perfect
hindsight:

| | % of loss prevented |
|---|---:|
| Sentinel (fitted on calibration) | 39.50% |
| Oracle (perfect hindsight) | 39.61% |
| **Price of not knowing the future** | **0.11pp — ₹10,272** |

Note what this metric is *not*. The obvious diagnostic — compare calibration
performance (52.9%) with test performance (39.5%) — looks alarming and means
nothing: the calibration window runs a 4.04% fraud rate against test's 3.45%,
so a larger share of its rupees are preventable regardless of threshold
quality. Prevalence, not overfitting. The oracle gap is the real answer.

**The split is temporal, and so is the calibration slice.** Train (100 days,
3.38%) → calibration (29 days, 4.04%) → test (53 days, 3.45%). The base rate
visibly drifts; a random split would have hidden that and inflated everything.

**The probabilities are calibrated, because the economics depend on them.**
Raw XGBoost output is a ranking score — 0.30 does not mean "30% of these are
fraud". The decision layer multiplies P(fraud) by a rupee amount, so an
uncalibrated score makes every money figure fiction. Isotonic regression fitted
out-of-time; Brier 0.0222 → 0.0217; reliability table in RESULTS.md.
`scale_pos_weight` deliberately unused — it buys a metric that doesn't matter
here at the cost of the probability scale that does.

**Three decision tiers from an explicit cost matrix**, not a 0.5 cutoff:

| outcome | cost (INR) |
|---|---|
| allow a fraud | amount + ₹1,250 chargeback fee + ₹400 representment labour |
| block a good customer | ₹700 friction — support contact + churn allowance |
| send to review | ₹40 analyst + ₹30 delay; analyst catch rate 0.90, not 1.0 |

Threshold search is exact, not sampled: sorting by score once makes any
`(t_review, t_block)` pair an O(1) cumulative-sum lookup, so the full grid is
evaluated. The optimiser asserts its predicted cost against the policy's
realised cost on every run.

---

## Beyond the model

**Review queue ranked by expected loss, not probability.** A 0.4-probability
₹80,000 transaction deserves a human before a 0.9-probability ₹200 one. Each
case carries SHAP reason codes in plain English ("this billing address: amount
is 19.6× its 7-day average") — an analyst cannot act on `0.83`.

**Spike sentinel.** Chargebacks surface 30–90 days after the transaction, so
label-based alarms arrive far too late to be operational. PSI on inputs and
CUSUM on per-segment mean score both run without labels. On the holdout,
ProductCD S alarmed on both score and fraud rate (CUSUM 7.96, fraud rate
6.85% → 8.23% across the window).

**Failure handling**, all demonstrable via `ResilientScorer.drill()`:

| failure | response |
|---|---|
| model service unavailable | rule fallback; triages to review, never blocks |
| latency budget exceeded | fails to **REVIEW**, never to ALLOW |
| unseen category value | dedicated bucket — `id_31` had 13,556 unseen values at test time |

Timing out into "approve" is how a payment system loses money quietly.

---

## Where it fails

| product | n | frauds | recall |
|---|---:|---:|---:|
| **W** | 117,584 | **2,336 (46% of all fraud)** | **25.4%** |
| S | 4,018 | 220 | 39.6% |
| H | 4,148 | 250 | 68.0% |
| C | 15,251 | 1,985 | 75.1% |
| R | 6,634 | 309 | 77.4% |

Product W holds nearly half of all fraud at the lowest recall in the book. Its
1.99% base rate makes each individual transaction cheap to ignore and the
aggregate expensive. This is the first thing we would fix with more time.

### Known limitations

- **Email-domain velocity measures popularity, not risk.** `P_emaildomain_cnt_7d`
  reads ~9,600 for gmail.com in any window. It needs normalising against each
  domain's own baseline; it currently surfaces in reason codes where it is
  close to meaningless.
- **Calibration is imperfect in the tails.** Systematic under-prediction across
  the lower nine bins and 2.8pp over-prediction in the top bin — likely the
  4.04% → 3.45% prevalence shift between calibration and test.
- **CUSUM is tuned loose.** At h=4.0, 10 of 17 monitored segments alarmed. Useful
  for ranking severity, too sensitive to page anyone at 3am.
- **Small segments go unmonitored.** 956 skipped for insufficient history,
  listed with reasons in RESULTS.md rather than silently dropped.

---

## Things we built that did not work

**Per-segment thresholds.** Fitting cutoffs per ProductCD instead of globally
won by 6% at 50k rows and *lost* by 0.9% at full scale (₹3.605M vs ₹3.637M) —
under both fitting regimes, so not noise. With 536 distinct calibrated
probability levels the global optimiser already has fine control, and ten
fitted parameters instead of two bought nothing. Kept in the codebase and
reported because we built it.

**Raising the tree ceiling.** `best_iteration=538` against a 600-tree cap looked
like the cap was binding, so we raised it to 1500. It stopped at 538 again —
early stopping had genuinely converged and the ceiling was never the constraint.

---

## Setup

```bash
pip install -r requirements.txt
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d data/     # need train_transaction.csv + train_identity.csv

python scripts/run_pipeline.py --nrows 50000   # ~25s, fast iteration
python scripts/run_pipeline.py                 # ~8 min -> reports/RESULTS.md
```

IEEE-CIS was chosen because it has `TransactionDT` for a real temporal split and
`TransactionAmt` for per-transaction cost. Most fraud datasets lack the second,
which makes the economics impossible.

`scripts/make_smoke_data.py` exists only to verify the pipeline executes. Its
output is not reportable and is labelled as such at runtime — self-generated
fraud is separable in ways real fraud is not. Sanity check: the pipeline scores
ROC-AUC ≈ 0.52 on it. Noise in, noise out; a pipeline that returns 0.99 on
random data is leaking.

> **Filename collision — read before running both.** The smoke script writes to
> `data/train_transaction.csv`, the same path the real Kaggle file uses. If you
> run the smoke script and then unzip the real dataset (or the reverse), one
> silently overwrites the other and your numbers will not match this README.
> Check `data/` before a full run: the real `train_transaction.csv` is ~650MB.

```
src/config.py      cost matrix, split boundaries, window sizes — one place
src/data.py        loading, temporal split
src/features.py    past-only velocity features, unseen-safe encoding
src/model.py       XGBoost + out-of-time isotonic calibration
src/economics.py   cost matrix, 3-tier policy, exact threshold search, capacity curve
src/explain.py     SHAP -> plain-English reason codes, analyst queue
src/sentinel.py    PSI + CUSUM drift and spike monitoring
src/fallback.py    rule fallback, latency budget, failure drill
src/evaluate.py    metrics and report generation
```

---

## What broke while building this

- **Categorical detection silently found zero columns** under pandas 3.0, which
  infers text as `str` rather than `object`. The encoder ran on nothing, the
  model trained without categorical signal, and nothing raised. Silent no-ops
  are worse than crashes.

- **The threshold optimiser and the decision policy disagreed.** Isotonic
  calibration emitted only 98 distinct probability levels across 12,500 rows, so
  hundreds of transactions shared an identical score. The optimiser cut its grid
  on index positions and counted 250 rows; the policy applied `p >= t` and acted
  on 400. The 2% review cap was silently violated (actual 3.19%) and predicted
  cost missed realised cost by 8%. Found by cross-checking two things that
  should have agreed — now asserted on every run.

- **Thresholds were fitted on the test set.** The model was held out; the
  decision layer was not. Moved to the calibration slice, and the cost of the
  fix turned out to be 0.11pp — see the oracle table above.

- **Cold start tripped our own safety timeout.** First prediction took ~160ms
  against a 150ms budget (thread-pool and buffer allocation), so the drill
  reported the *healthy* path as degraded. Fixed with explicit warmup;
  steady-state is ~35ms. Only surfaced because latency was instrumented rather
  than assumed.

- **Isotonic calibration saturated at exactly 1.0** — eleven transactions came
  back certain. A model fitted on 30k rows cannot support a posterior of 1.0.
  Capped at 0.999, with a resolution report so the coarseness is visible.

- **The report contradicted itself.** The chosen policy saved ₹136k while the
  capacity curve showed ₹222k — money apparently left on the table. Two causes:
  that capacity exceeded the review budget, and the capacity curve never blocks
  so it pays no friction. Both now stated on the table.

- **The spike sentinel silently monitored nothing.** A 3-day holdout with 1-day
  buckets gives 3 buckets; CUSUM needs 4. Every segment was skipped and the
  section rendered empty. Bucket width now derives from the actual span, and
  skips are listed with reasons.

- **The reason codes were useless.** Ranked purely by SHAP they came back as
  "V156; V197; V258" — anonymised columns dominate and no template can make them
  meaningful. Two slots now reserved for features that carry semantics. A
  related embarrassment: `R_emaildomain = 0` was being shown to analysts as if
  it were a domain. It was an encoder code.

- **Friction cost was set too low.** At ₹250 the optimiser blocked ~5% of traffic
  (40 FP per 1,000 good) against the 1–2% decline rate real processors run.
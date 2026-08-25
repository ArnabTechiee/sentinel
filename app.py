"""
Sentinel — analyst review console.

The one screen a risk analyst actually works in: cases ranked by expected loss,
each with its reason codes and a disposition.

Deliberately dumb at runtime. It reads reports/review_queue.csv and writes
reports/decisions.json. No model is loaded, no network call is made, nothing is
retrained. Startup is instant and there is nothing that can hang mid-demo.

    streamlit run app.py
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
QUEUE = ROOT / "reports" / "review_queue.csv"
DECISIONS = ROOT / "reports" / "decisions.json"

# Operating point from the last full run, fitted on the calibration slice.
# Exposed in the sidebar so the thresholds are visible rather than buried.
T_REVIEW_DEFAULT = 0.2038
T_BLOCK_DEFAULT = 0.4615
AMT_P99 = 1092.0  # train 99th percentile, full run — used by the rule fallback

RUPEE = "&#8377;"  # HTML entity: survives any file encoding on any machine

st.set_page_config(page_title="Sentinel — review queue", page_icon="▚", layout="wide")

# --------------------------------------------------------------------------
# styling — a ledger, not a dashboard: ink on paper, figures in monospace
# --------------------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

  html, body, [class*="css"] { font-family: 'IBM Plex Sans', system-ui, sans-serif; }

  :root {
    --ink:      #131a24;
    --ink-soft: #5a6673;
    --rule:     #d8dee6;
    --paper:    #fbfbf9;
    --block:    #b3261e;
    --review:   #a86a00;
    --approve:  #1c6b45;
  }

  /* Theme is forced here, not delegated. .streamlit/config.toml sets it too,
     but a dot-folder does not survive every copy on Windows, and without it
     Streamlit falls back to the client's dark mode — dark ink on a dark
     background, metric row invisible. !important throughout so the console
     renders identically whether or not config.toml made the trip. */
  .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
  [data-testid="stHeader"] { background: var(--paper) !important; }

  .stApp p, .stApp li, .stApp label, .stApp span, .stApp div,
  .stApp h1, .stApp h2, .stApp h3, .stApp h4,
  .stMarkdown, .stMarkdown p { color: var(--ink) !important; }

  [data-testid="stMetricValue"] {
    color: var(--ink) !important; font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.9rem !important;
  }
  [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
    color: var(--ink-soft) !important; font-size: 0.68rem !important;
    letter-spacing: 0.1em; text-transform: uppercase;
  }
  [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
  .stCaption, .stCaption p { color: var(--ink-soft) !important; }

  [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
    background: #f2f1ed !important; border-right: 1px solid var(--rule);
  }
  [data-testid="stSidebar"] * { color: var(--ink) !important; }
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: var(--ink-soft) !important;
  }
  [data-testid="stExpander"] { border: 1px solid var(--rule); border-radius: 2px; }
  [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p {
    color: var(--ink) !important;
  }

  .eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--ink-soft) !important;
  }
  .figure {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.9rem; font-weight: 600; line-height: 1.15;
  }
  .figure-sub { font-size: 0.78rem; color: var(--ink-soft) !important; font-weight: 400; }

  .case-id {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.75rem; font-weight: 600; letter-spacing: -0.01em;
  }
  .verdict {
    display: inline-block; padding: 0.2rem 0.65rem; border-radius: 2px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem;
    letter-spacing: 0.11em; text-transform: uppercase; font-weight: 600;
  }
  .v-block   { background: #fbe9e7; color: var(--block) !important;   border: 1px solid #f0c4bf; }
  .v-review  { background: #fdf3e0; color: var(--review) !important;  border: 1px solid #f0dcb4; }
  .v-approve { background: #e7f4ed; color: var(--approve) !important; border: 1px solid #bfdfcd; }

  .reason {
    padding: 0.45rem 0 0.45rem 0.85rem;
    border-left: 2px solid var(--rule);
    margin-bottom: 0.32rem; font-size: 0.93rem; line-height: 1.45;
  }
  .reason-key { border-left-color: var(--review); font-weight: 500; }
  .reason-opaque {
    color: var(--ink-soft) !important; font-size: 0.84rem;
    font-family: 'IBM Plex Mono', monospace;
  }

  .meter { height: 9px; background: #eceef1; overflow: hidden;
           display: flex; margin: 0.4rem 0 0.2rem 0; }
  .meter-cleared { background: var(--approve); }
  .meter-open    { background: #c9ced6; }

  /* Signature element: the threshold ruler.
     The thesis of this project is that the decision layer is where the money
     is, so the decision layer gets the one bold piece of the screen. The bands
     are the live operating point; the needle is this case. Drag a sidebar
     slider and the bands move under the needle. */
  .ruler-wrap { margin: 0.2rem 0 0.9rem 0; }
  .ruler { position: relative; height: 34px; display: flex;
           border: 1px solid var(--rule); }
  .band { height: 100%; display: flex; align-items: center; justify-content: center;
          font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem;
          letter-spacing: 0.11em; overflow: hidden; white-space: nowrap; }
  .band-allow  { background: #b9dcc8; color: #12472e !important; font-weight: 600; }
  .band-review { background: #f4d99a; color: #6b4600 !important; font-weight: 600; }
  .band-block  { background: #eeb3ac; color: #6d1712 !important; font-weight: 600; }
  .band-off    { background: repeating-linear-gradient(45deg,#f1f1ee,#f1f1ee 5px,#e7e7e3 5px,#e7e7e3 10px);
                 color: #9aa2ab !important; }
  .needle { position: absolute; top: -6px; bottom: -6px; width: 2px;
            background: var(--ink); }
  .needle-cap { position: absolute; top: -19px; transform: translateX(-50%);
                font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem;
                font-weight: 600; white-space: nowrap; }
  .ruler-scale { display: flex; justify-content: space-between;
                 font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem;
                 color: var(--ink-soft) !important; margin-top: 0.25rem; }

  table.ledger { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
  table.ledger th {
    text-align: left; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.64rem; letter-spacing: 0.11em; text-transform: uppercase;
    color: var(--ink-soft) !important; border-bottom: 1px solid var(--rule);
    padding: 0.55rem 0.6rem; font-weight: 500;
  }
  table.ledger td { padding: 0.5rem 0.6rem; border-bottom: 1px solid #eceef1; }
  table.ledger td.num { font-family: 'IBM Plex Mono', monospace; text-align: right; }
  table.ledger td.id  { font-family: 'IBM Plex Mono', monospace; }
  table.ledger tr:hover td { background: #f4f4f1; }
  .expo-bar { height: 6px; background: #c9ced6; display: inline-block;
              vertical-align: middle; margin-right: 0.5rem; }
  .tag { font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem;
         letter-spacing: 0.08em; padding: 0.12rem 0.4rem; border-radius: 2px; }
  .t-block   { background: #fbe9e7; color: var(--block) !important; }
  .t-approve { background: #e7f4ed; color: var(--approve) !important; }
  .t-esc     { background: #eceef1; color: var(--ink-soft) !important; }

  .degraded {
    background: #fdf3e0; border: 1px solid #e8cf9d; border-left: 3px solid var(--review);
    padding: 0.72rem 0.95rem; margin-bottom: 1rem; font-size: 0.9rem;
  }
  .degraded, .degraded * { color: #6b4600 !important; }
  hr { border-color: var(--rule) !important; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
@st.cache_data
def load_queue() -> pd.DataFrame:
    if not QUEUE.exists():
        return pd.DataFrame()
    df = pd.read_csv(QUEUE)
    df["transaction_id"] = df["transaction_id"].astype(str)
    return df.sort_values("expected_loss", ascending=False).reset_index(drop=True)


def load_decisions() -> dict:
    if DECISIONS.exists():
        try:
            return json.loads(DECISIONS.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_decision(txn_id: str, action: str, degraded: bool) -> None:
    d = load_decisions()
    d[txn_id] = {
        "action": action,
        "at": datetime.now().isoformat(timespec="seconds"),
        "scored_by": "rules (degraded)" if degraded else "model",
    }
    DECISIONS.parent.mkdir(parents=True, exist_ok=True)
    DECISIONS.write_text(json.dumps(d, indent=2))


# --------------------------------------------------------------------------
# reason classification
# --------------------------------------------------------------------------
_OPAQUE = re.compile(
    r"^(Vesta risk feature|velocity counter|timedelta feature|device/identity signal)"
    r"|^\w+ = "
)


def split_reasons(blob: str) -> tuple[list[str], list[str]]:
    """Analyst-actionable reasons first; anonymised model internals second."""
    parts = [p.strip() for p in str(blob).split(";") if p.strip()]
    key = [p for p in parts if not _OPAQUE.match(p)]
    opaque = [p for p in parts if _OPAQUE.match(p)]
    return key, opaque


def rule_score(row: pd.Series) -> tuple[float, str]:
    """
    Mirrors src/fallback.RuleFallback for the outage drill. Crude on purpose:
    its job is to keep the merchant safe for the minutes the model is down, not
    to match the model. Capped at 0.85 so it never claims certainty — and it is
    emphatically not a calibrated probability, which is why the label above it
    changes when the fallback is active.
    """
    hits = []
    if row["amount"] > AMT_P99:
        hits.append("amount above 99th percentile")
    key, _ = split_reasons(row["reasons"])
    if sum(1 for k in key if "average" in k or "since previous" in k) >= 2:
        hits.append("multiple velocity signals")
    if any(re.search(r"hour \(([0-5]):", k) for k in key):
        hits.append("overnight transaction")
    return min(0.15 + 0.20 * len(hits), 0.85), "; ".join(hits) or "no rule fired"


def verdict(p: float, t_review: float, t_block: float, degraded: bool) -> tuple[str, str]:
    if degraded:
        # Degraded mode never blocks: a crude rule score cannot justify
        # declining a customer. It triages instead.
        return ("REVIEW", "v-review") if p >= 0.15 else ("APPROVE", "v-approve")
    if p >= t_block:
        return "BLOCK", "v-block"
    if p >= t_review:
        return "REVIEW", "v-review"
    return "APPROVE", "v-approve"


def threshold_ruler(p: float, t_review: float, t_block: float, degraded: bool) -> str:
    """The operating point, drawn. Bands are the policy; the needle is this case."""
    a = max(0.0, min(t_review, 1.0)) * 100
    r = max(0.0, min(t_block, 1.0) - min(t_review, 1.0)) * 100
    b = max(0.0, 100 - a - r)
    pos = min(max(p, 0.0), 1.0) * 100
    block_class = "band band-off" if degraded else "band band-block"
    block_label = "BLOCK — OFFLINE" if degraded else "BLOCK"
    return f"""
    <div class="ruler-wrap">
      <div class="ruler">
        <div class="band band-allow"  style="width:{a:.2f}%">{'ALLOW' if a > 9 else ''}</div>
        <div class="band band-review" style="width:{r:.2f}%">{'REVIEW' if r > 11 else ''}</div>
        <div class="{block_class}" style="width:{b:.2f}%">{block_label if b > 16 else ''}</div>
        <div class="needle" style="left:{pos:.2f}%"></div>
        <div class="needle-cap" style="left:{pos:.2f}%">{p:.3f}</div>
      </div>
      <div class="ruler-scale"><span>0.00</span>
        <span>review {t_review:.3f}</span><span>block {t_block:.3f}</span>
        <span>1.00</span></div>
    </div>"""


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="eyebrow">Operating point</div>', unsafe_allow_html=True)
    t_review = st.slider("Review at", 0.0, 1.0, T_REVIEW_DEFAULT, 0.001, format="%.3f")
    t_block = st.slider("Block at", 0.0, 1.0, T_BLOCK_DEFAULT, 0.001, format="%.3f")
    if t_block < t_review:
        st.warning("Block sits below review — nothing can reach the review tier.")
    st.caption("Fitted on the calibration slice, never on test. Perfect hindsight "
               "would have bought 0.11pp more.")

    st.divider()
    st.markdown('<div class="eyebrow">Failure drill</div>', unsafe_allow_html=True)
    degraded = st.toggle("Simulate model outage")
    st.caption("Scoring falls back to deterministic rules. Blocking is disabled — "
               "a rule score cannot justify declining a customer.")

    st.divider()
    if st.button("Clear all dispositions", width="stretch"):
        DECISIONS.unlink(missing_ok=True)
        st.rerun()

queue = load_queue()
if queue.empty:
    st.markdown('<div class="eyebrow">No queue file</div>', unsafe_allow_html=True)
    st.markdown("### Nothing to review yet")
    st.write("Run the pipeline to generate the queue:")
    st.code("python scripts/run_pipeline.py --nrows 50000", language="bash")
    st.stop()

decisions = load_decisions()
if degraded:
    scored = queue.apply(rule_score, axis=1)
    queue["p_shown"] = [s[0] for s in scored]
    queue["rule_note"] = [s[1] for s in scored]
else:
    queue["p_shown"] = queue["p_fraud"]
    queue["rule_note"] = ""

open_cases = queue[~queue["transaction_id"].isin(decisions)]
done_cases = queue[queue["transaction_id"].isin(decisions)]

# --------------------------------------------------------------------------
# header
# --------------------------------------------------------------------------
st.markdown('<div class="eyebrow">Sentinel · analyst review queue</div>',
            unsafe_allow_html=True)
st.markdown("## Cases ranked by expected loss")
st.markdown(f'<p style="color:#5a6673;font-size:0.88rem;margin-top:-0.5rem">'
            f'A {RUPEE}80,000 transaction at 0.4 probability outranks a {RUPEE}200 '
            'one at 0.9. The queue is sorted by rupees at risk, not by score.</p>',
            unsafe_allow_html=True)

if degraded:
    st.markdown(
        '<div class="degraded"><strong>Model unavailable — degraded scoring.</strong> '
        'Scores below come from the deterministic rule fallback and are not '
        'calibrated probabilities. Blocking is disabled; cases triage to review.'
        '</div>', unsafe_allow_html=True)

cleared = float(done_cases["expected_loss"].sum())
total = float(queue["expected_loss"].sum())
pct = cleared / total if total else 0.0

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="eyebrow">Dispositioned</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="figure">{len(done_cases)} <span class="figure-sub">'
                f'of {len(queue)}</span></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="eyebrow">Exposure cleared</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="figure">{RUPEE}{cleared:,.0f}</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="eyebrow">Still open</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="figure">{RUPEE}{total - cleared:,.0f}</div>',
                unsafe_allow_html=True)

st.markdown(
    f'<div class="meter"><div class="meter-cleared" style="width:{pct*100:.1f}%"></div>'
    f'<div class="meter-open" style="width:{(1-pct)*100:.1f}%"></div></div>',
    unsafe_allow_html=True)
st.divider()

# --------------------------------------------------------------------------
# current case
# --------------------------------------------------------------------------
if open_cases.empty:
    st.markdown("### Queue cleared")
    st.markdown(f'All {len(queue)} cases dispositioned — {RUPEE}{cleared:,.0f} of '
                'expected loss worked through. Clear dispositions in the sidebar '
                'to start over.', unsafe_allow_html=True)
else:
    row = open_cases.iloc[0]
    p = float(row["p_shown"])
    v_text, v_class = verdict(p, t_review, t_block, degraded)
    position = len(done_cases) + 1

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown(f'<div class="eyebrow">Case {position} of {len(queue)} · '
                    'highest open exposure</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="case-id">{row["transaction_id"]}</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<span class="verdict {v_class}">recommend {v_text}</span>',
                    unsafe_allow_html=True)
        st.write("")

        m1, m2, m3 = st.columns(3)
        m1.metric("Amount", f"₹{row['amount']:,.0f}")
        m2.metric("Rule score" if degraded else "P(fraud)", f"{p:.3f}")
        m3.metric("Expected loss", f"₹{row['expected_loss']:,.0f}")

        st.markdown('<div class="eyebrow" style="margin-top:1.1rem">Where this case '
                    'falls against the operating point</div>', unsafe_allow_html=True)
        st.markdown(threshold_ruler(p, t_review, t_block, degraded),
                    unsafe_allow_html=True)
        # The axis is score, not volume. Worth stating: a viewer sees a narrow
        # ALLOW band and reads it as "we allow very little", when in fact ~96%
        # of the holdout scores under 0.05 and sits in its leftmost sliver.
        st.caption("Axis is model score, not traffic share — about 96% of live "
                   "traffic scores under 0.05 and sits inside the left edge of "
                   "the allow band.")

        if degraded and row["rule_note"]:
            st.caption(f"Rules fired: {row['rule_note']}")

    with right:
        st.markdown('<div class="eyebrow">Why this case surfaced</div>',
                    unsafe_allow_html=True)
        key, opaque = split_reasons(row["reasons"])
        for k in key:
            st.markdown(f'<div class="reason reason-key">{k}</div>',
                        unsafe_allow_html=True)
        if not key:
            st.markdown('<div class="reason">No analyst-readable driver — this case '
                        'is carried entirely by anonymised model features.</div>',
                        unsafe_allow_html=True)
        if opaque:
            with st.expander(f"Model internals ({len(opaque)})"):
                st.caption("Anonymised dataset columns. Real signal, but nothing "
                           "an analyst can act on directly.")
                for o in opaque:
                    st.markdown(f'<div class="reason reason-opaque">{o}</div>',
                                unsafe_allow_html=True)

    st.write("")
    b1, b2, b3, _ = st.columns([1, 1, 1, 3])
    if b1.button("Block", type="primary", disabled=degraded, width="stretch"):
        save_decision(row["transaction_id"], "BLOCK", degraded)
        st.rerun()
    if b2.button("Approve", width="stretch"):
        save_decision(row["transaction_id"], "APPROVE", degraded)
        st.rerun()
    if b3.button("Escalate", width="stretch"):
        save_decision(row["transaction_id"], "ESCALATE", degraded)
        st.rerun()
    if degraded:
        st.caption("Block is unavailable while the model is down.")

st.divider()

# --------------------------------------------------------------------------
# rest of queue + audit trail
# --------------------------------------------------------------------------
tab_open, tab_log = st.tabs([f"Open queue ({len(open_cases)})",
                             f"Audit trail ({len(done_cases)})"])

with tab_open:
    if len(open_cases) > 1:
        rest = open_cases.iloc[1:]
        top = float(rest["expected_loss"].max()) or 1.0
        rows = []
        for r in rest.itertuples():
            v, _ = verdict(float(r.p_shown), t_review, t_block, degraded)
            tag = {"BLOCK": "t-block", "REVIEW": "t-esc", "APPROVE": "t-approve"}[v]
            width = 45.0 * float(r.expected_loss) / top
            rows.append(
                f'<tr><td class="id">{r.transaction_id}</td>'
                f'<td><span class="tag {tag}">{v}</span></td>'
                f'<td class="num">{r.p_shown:.4f}</td>'
                f'<td class="num">{RUPEE}{r.amount:,.0f}</td>'
                f'<td class="num"><span class="expo-bar" style="width:{width:.1f}px">'
                f'</span>{RUPEE}{r.expected_loss:,.0f}</td></tr>')
        st.markdown(
            '<table class="ledger"><thead><tr><th>Transaction</th><th>Recommendation</th>'
            '<th style="text-align:right">Score</th>'
            '<th style="text-align:right">Amount</th>'
            '<th style="text-align:right">Expected loss</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>', unsafe_allow_html=True)
        st.caption("Bar length is expected loss relative to the largest open case.")
    else:
        st.caption("Nothing else waiting.")

with tab_log:
    if decisions:
        exposure = dict(zip(queue["transaction_id"], queue["expected_loss"]))
        rows = []
        for k, v in decisions.items():
            tag = {"BLOCK": "t-block", "APPROVE": "t-approve",
                   "ESCALATE": "t-esc"}.get(v["action"], "t-esc")
            rows.append(
                f'<tr><td class="id">{k}</td>'
                f'<td><span class="tag {tag}">{v["action"]}</span></td>'
                f'<td class="num">{RUPEE}{exposure.get(k, 0):,.0f}</td>'
                f'<td>{v["scored_by"]}</td>'
                f'<td class="num">{v["at"].replace("T", " ")}</td></tr>')
        st.markdown(
            '<table class="ledger"><thead><tr><th>Transaction</th><th>Action</th>'
            '<th style="text-align:right">Exposure</th><th>Scored by</th>'
            '<th style="text-align:right">At</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>', unsafe_allow_html=True)
        st.caption("Written to reports/decisions.json — every disposition records "
                   "whether the model or the fallback scored it.")
    else:
        st.caption("No dispositions yet.")
"""
Sentinel — analyst review console.

The one screen a risk analyst actually works in: cases ranked by expected loss,
each with its reason codes, and a disposition.

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

# Operating point from the last full run (fitted on the calibration slice).
# Adjustable in the sidebar so the thresholds are visible rather than buried.
T_REVIEW_DEFAULT = 0.2038
T_BLOCK_DEFAULT = 0.4615
CHARGEBACK_FEE = 1250.0
REPRESENTMENT = 400.0

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
    --block:    #b3261e;
    --review:   #a86a00;
    --approve:  #1c6b45;
  }

  /* Theme is forced here, not delegated.
     .streamlit/config.toml sets the theme, but a dot-folder does not survive
     every copy on Windows and Streamlit falls back to the client's dark mode --
     at which point dark ink lands on a dark background and the metric row
     disappears. Everything below is !important so the console renders the same
     regardless of whether config.toml made the trip. */
  .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
  [data-testid="stHeader"] { background: #fbfbf9 !important; }

  .stApp p, .stApp li, .stApp label, .stApp span, .stApp div,
  .stApp h1, .stApp h2, .stApp h3, .stApp h4,
  .stMarkdown, .stMarkdown p { color: var(--ink) !important; }

  [data-testid="stMetricValue"] {
    color: var(--ink) !important; font-family: 'IBM Plex Mono', monospace !important;
  }
  [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
    color: var(--ink-soft) !important; font-size: 0.72rem !important;
    letter-spacing: 0.08em; text-transform: uppercase;
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

  /* Ledger table. st.dataframe renders to a canvas that stylesheets cannot
     reach, so under a dark client theme it stayed black on a light page. This
     is plain HTML instead -- fully themeable, and 15 rows do not need a grid. */
  table.ledger { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
  table.ledger th {
    text-align: left; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--ink-soft); border-bottom: 1px solid var(--rule);
    padding: 0.5rem 0.6rem; font-weight: 500;
  }
  table.ledger td {
    padding: 0.5rem 0.6rem; border-bottom: 1px solid #eceef1; color: var(--ink);
  }
  table.ledger td.num { font-family: 'IBM Plex Mono', monospace; text-align: right; }
  table.ledger tr:hover td { background: #f4f4f1; }

  .eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.70rem; letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--ink-soft);
  }
  .figure {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.85rem; font-weight: 600; color: var(--ink); line-height: 1.1;
  }
  .figure-sub { font-size: 0.78rem; color: var(--ink-soft); }

  .case-id {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem; font-weight: 600; color: var(--ink);
  }
  .verdict {
    display: inline-block; padding: 0.18rem 0.6rem; border-radius: 2px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
    letter-spacing: 0.1em; text-transform: uppercase; font-weight: 600;
  }
  .v-block   { background: #fbe9e7; color: var(--block);   border: 1px solid #f0c4bf; }
  .v-review  { background: #fdf3e0; color: var(--review);  border: 1px solid #f0dcb4; }
  .v-approve { background: #e7f4ed; color: var(--approve); border: 1px solid #bfdfcd; }

  .reason {
    padding: 0.42rem 0 0.42rem 0.85rem;
    border-left: 2px solid var(--rule);
    margin-bottom: 0.3rem; font-size: 0.92rem; color: var(--ink);
  }
  .reason-key { border-left-color: var(--review); font-weight: 500; }
  .reason-opaque { color: var(--ink-soft); font-size: 0.85rem;
                   font-family: 'IBM Plex Mono', monospace; }

  .meter { height: 10px; background: #eceef1; border-radius: 1px; overflow: hidden;
           display: flex; margin: 0.35rem 0 0.15rem 0; }
  .meter-cleared { background: var(--approve); }
  .meter-open    { background: #c9ced6; }

  .degraded {
    background: #fdf3e0; border: 1px solid #e8cf9d; border-left: 3px solid var(--review);
    padding: 0.7rem 0.95rem; margin-bottom: 1rem; font-size: 0.9rem; color: #6b4600;
  }
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
    to match the model. Capped so it never claims certainty.
    """
    hits = []
    if row["amount"] > 1092:  # train 99th percentile, full run
        hits.append("amount above 99th percentile")
    key, _ = split_reasons(row["reasons"])
    vel = sum(1 for k in key if "average" in k or "since previous" in k)
    if vel >= 2:
        hits.append("multiple velocity signals")
    if any("hour (0" in k or "hour (1:" in k or "hour (2:" in k for k in key):
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


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="eyebrow">Operating point</div>', unsafe_allow_html=True)
    t_review = st.slider("Review at", 0.0, 1.0, T_REVIEW_DEFAULT, 0.001, format="%.3f")
    t_block = st.slider("Block at", 0.0, 1.0, T_BLOCK_DEFAULT, 0.001, format="%.3f")
    st.caption("Fitted on the calibration slice, never on test. "
               "Perfect hindsight would have bought 0.11pp more.")

    st.divider()
    st.markdown('<div class="eyebrow">Failure drill</div>', unsafe_allow_html=True)
    degraded = st.toggle("Simulate model outage")
    st.caption("Scoring falls back to deterministic rules. Blocking is disabled — "
               "a rule score cannot justify declining a customer.")

    st.divider()
    if st.button("Clear all dispositions"):
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
st.caption("A ₹80,000 transaction at 0.4 probability outranks a ₹200 one at 0.9. "
           "The queue is sorted by rupees at risk, not by score.")

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
    st.markdown(f'<div class="figure">₹{cleared:,.0f}</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="eyebrow">Still open</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="figure">₹{total - cleared:,.0f}</div>',
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
    st.write(f"All {len(queue)} cases dispositioned — ₹{cleared:,.0f} of expected "
             f"loss worked through.")
else:
    row = open_cases.iloc[0]
    p = float(row["p_shown"])
    v_text, v_class = verdict(p, t_review, t_block, degraded)

    left, right = st.columns([3, 2])

    with left:
        st.markdown(f'<div class="eyebrow">Case · highest exposure</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="case-id">{row["transaction_id"]}</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<span class="verdict {v_class}">recommend {v_text}</span>',
                    unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Amount", f"₹{row['amount']:,.0f}")
        m2.metric("P(fraud)" if not degraded else "Rule score", f"{p:.3f}")
        m3.metric("Expected loss", f"₹{row['expected_loss']:,.0f}")

        if degraded and row["rule_note"]:
            st.caption(f"Rules fired: {row['rule_note']}")

    with right:
        st.markdown('<div class="eyebrow">Why this case surfaced</div>',
                    unsafe_allow_html=True)
        key, opaque = split_reasons(row["reasons"])
        for k in key:
            st.markdown(f'<div class="reason reason-key">{k}</div>',
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
        rows = "".join(
            f'<tr><td>{r.transaction_id}</td>'
            f'<td class="num">{r.p_shown:.4f}</td>'
            f'<td class="num">Rs {r.amount:,.0f}</td>'
            f'<td class="num">Rs {r.expected_loss:,.0f}</td></tr>'
            for r in open_cases.iloc[1:].itertuples()
        )
        st.markdown(
            '<table class="ledger"><thead><tr><th>Transaction</th>'
            '<th style="text-align:right">Score</th>'
            '<th style="text-align:right">Amount</th>'
            '<th style="text-align:right">Expected loss</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>', unsafe_allow_html=True)
    else:
        st.caption("Nothing else waiting.")

with tab_log:
    if decisions:
        rows = "".join(
            f'<tr><td>{k}</td><td>{v["action"]}</td>'
            f'<td>{v["scored_by"]}</td><td>{v["at"]}</td></tr>'
            for k, v in decisions.items()
        )
        st.markdown(
            '<table class="ledger"><thead><tr><th>Transaction</th><th>Action</th>'
            '<th>Scored by</th><th>At</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>', unsafe_allow_html=True)
        st.caption("Written to reports/decisions.json — every disposition records "
                   "whether the model or the fallback scored it.")
    else:
        st.caption("No dispositions yet.")
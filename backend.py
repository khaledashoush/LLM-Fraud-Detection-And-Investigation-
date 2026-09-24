# -*- coding: utf-8 -*-
"""
backend.py — AML Intelligence API Server (v2)
"""

import os, json, time, math, re
import numpy as np
import pandas as pd
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# ==========================================
# CONFIG
# ==========================================
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
EXPL_PATH = "explanations/explanations.json"
EDGE_META_PATH = "processed_data/edge_metadata.parquet"
NODE_IDX_PATH = "processed_data/node_to_idx.parquet"
REPORT_CACHE_DIR = "reports/cache"
MAX_TOKENS = 800
TEMPERATURE = 0.1

os.makedirs(REPORT_CACHE_DIR, exist_ok=True)

# ==========================================
# LOAD DATA
# ==========================================
print("=" * 50)
print("  Loading data...")

# ── Explanations ──
with open(EXPL_PATH) as f:
    _raw = json.load(f)
EXPLANATIONS = _raw.get("explanations", [])
print(f"  ✓ Loaded {len(EXPLANATIONS):,} explanations")

# ── Build indexed DataFrame ──
_rows = []
for exp in EXPLANATIONS:
    pred = exp.get("prediction", {})
    details = exp.get("transaction_details", {})
    signals = exp.get("structural_signals", {})
    _rows.append({
        "transaction_id": exp["transaction_id"],
        "y_proba": pred.get("y_proba", 0),
        "y_true": pred.get("y_true", 0),
        "pattern": exp.get("pattern_ground_truth", "UNKNOWN"),
        "amount": signals.get("approx_amount_paid", 0),
        "timestamp": details.get("timestamp", ""),
        "src_account": details.get("src_account", ""),
        "dst_account": details.get("dst_account", ""),
    })
DF = pd.DataFrame(_rows)
DF_LOOKUP = {e["transaction_id"]: e for e in EXPLANATIONS}

# ── Stats ──
STATS = {
    "total_flagged": len(DF),
    "total_tp": int((DF["y_true"] == 1).sum()),
    "total_fp": int((DF["y_true"] == 0).sum()),
    "pattern_distribution": DF["pattern"].value_counts().to_dict(),
    "avg_risk": round(float(DF["y_proba"].mean()), 3),
}
print(f"  ✓ Stats: {STATS['total_flagged']:,} flagged, "
      f"{len(STATS['pattern_distribution'])} patterns")

# ── Edge metadata (for attempt-level graphs) ──
try:
    _edge_df = pd.read_parquet(EDGE_META_PATH)
    _node_df = pd.read_parquet(NODE_IDX_PATH)
    _idx_to_node = dict(zip(_node_df["idx"], _node_df["node"]))

    _attempt_lookup = {}
    _tx_attempt_map = {}

    for _, row in _edge_df[_edge_df["attempt_id"].notna()].iterrows():
        aid = int(row["attempt_id"])
        tx_id = int(row["transaction_id"])
        _tx_attempt_map[tx_id] = aid
        if aid not in _attempt_lookup:
            _attempt_lookup[aid] = []
        _attempt_lookup[aid].append({
            "tx_id": tx_id,
            "src": _idx_to_node.get(int(row["src_idx"]), f"node_{int(row['src_idx'])}"),
            "dst": _idx_to_node.get(int(row["dst_idx"]), f"node_{int(row['dst_idx'])}"),
            "ts": str(row.get("Timestamp", "")),
        })
    print(f"  ✓ Loaded {len(_attempt_lookup):,} laundering attempts")
except Exception as e:
    print(f"  ⚠ Edge metadata not loaded: {e}")
    _attempt_lookup = {}
    _tx_attempt_map = {}
    _edge_df = pd.DataFrame()
    _idx_to_node = {}

print("=" * 50)

# ==========================================
# LLM REPORT GENERATION SETUP
# ==========================================
PATTERN_DESCRIPTIONS = {
    "SCATTER-GATHER": "Funds from one source are dispersed to intermediaries, then reconverge at a final destination",
    "GATHER-SCATTER": "Funds from multiple sources converge at one account, then are dispersed to new destinations",
    "FAN-OUT": "One source account disperses funds to many destination accounts",
    "FAN-IN": "Many source accounts send funds to a single destination account",
    "CYCLE": "Funds flow through a circular path, eventually returning to the originating account",
    "BIPARTITE": "A set of input accounts transfers funds to a set of output accounts",
    "STACK": "Multi-layered structure where funds pass through multiple bipartite layers",
    "RANDOM": "Funds move through a random walk of controlled accounts",
    "LONE": "Isolated illicit transaction without a clear structural pattern",
    "NONE": "No laundering pattern identified (possible false positive)",
}

PATTERN_RISK_CONTEXT = {
    "SCATTER-GATHER": {"risk_level": "HIGH",
        "significance": "Funds dispersed through intermediaries before reconvergence, making the money trail extremely difficult to trace.",
        "consequence": "If undetected, this scheme can integrate large volumes of illicit funds while making traceback nearly impossible."},
    "GATHER-SCATTER": {"risk_level": "HIGH",
        "significance": "Funds from multiple sources converge and are redistributed — a classic layering technique.",
        "consequence": "The mixing effect makes it extremely difficult to distinguish illicit from legitimate funds."},
    "FAN-OUT": {"risk_level": "MODERATE to HIGH",
        "significance": "Single source dispersing to many destinations — characteristic of smurfing operations.",
        "consequence": "Can bypass monitoring thresholds and spread illicit funds across multiple accounts."},
    "FAN-IN": {"risk_level": "MODERATE to HIGH",
        "significance": "Multiple sources converging on a single destination — suggests a collection point.",
        "consequence": "If this is a collection hub, the volume could be substantial."},
    "CYCLE": {"risk_level": "HIGH",
        "significance": "Circular fund flows designed to create complex audit trails.",
        "consequence": "Cycles can repeatedly 'wash' the same funds."},
    "STACK": {"risk_level": "HIGH",
        "significance": "Multi-layered structures indicate sophisticated operations.",
        "consequence": "The layered structure suggests professional laundering services."},
    "BIPARTITE": {"risk_level": "MODERATE",
        "significance": "Coordinated transfers between two sets of accounts suggest organized networks.",
        "consequence": "The coordinated nature suggests a planned operation."},
    "RANDOM": {"risk_level": "MODERATE",
        "significance": "Random walks designed to defeat pattern-based detection.",
        "consequence": "Deliberate randomness suggests awareness of monitoring systems."},
}

SYSTEM_PROMPT = """You are an experienced AML (Anti-Money Laundering) compliance investigator. An AI fraud detection system (FraudGT) has flagged a transaction as suspicious and identified the laundering pattern. Your task is to write a professional investigation report that goes BEYOND description to provide analytical insights.

## Your Role
- The pattern is ALREADY IDENTIFIED by the detection system
- Your job is to EXPLAIN, INTERPRET, and ASSESS — not just describe

## Analytical Depth Requirements
1. **Interpret**: What does this evidence MEAN in the context of laundering?
2. **Assess**: How serious is this risk? What's the potential impact?
3. **Connect**: How does this transaction fit into the broader scheme?
4. **Implicate**: What could happen if this continues undetected?

## Report Format (SAR)
### Executive Summary
### Subject Information (WHO)
### Suspicious Activity (WHAT)
### Temporal Analysis (WHEN)
### Network Analysis (WHERE)
### Suspicion Basis (WHY)
### Recommended Actions

## Writing Rules
1. Start immediately
2. Every fact should be followed by its SIGNIFICANCE
3. Use cause-and-effect: "because", "which indicates"
4. Do NOT use numeric model scores
5. Assess risk explicitly: "this represents a HIGH risk because..."
6. Keep under 450 words

## IMPORTANT — Action Guidelines
Risk levels (HIGH/MODERATE) are analytical context only.
Do NOT translate them into aggressive actions.
NEVER recommend freezing, blocking, or shutting down accounts.
Always use "consider", "recommend", or "suggest" language.
You are advising, not ordering.
"""


def build_evidence(exp):
    """Build evidence for LLM from an explanation."""
    tx_id = exp["transaction_id"]
    pred = exp.get("prediction", {})
    details = exp.get("transaction_details", {})
    signals = exp.get("structural_signals", {})
    attn = exp.get("attention_analysis", {})
    head = exp.get("head_component_importance", {})

    pattern = exp.get("pattern_ground_truth", "UNKNOWN")
    pattern_desc = PATTERN_DESCRIPTIONS.get(pattern, "Unknown")
    risk_ctx = PATTERN_RISK_CONTEXT.get(pattern, {})
    risk_level = risk_ctx.get("risk_level", "UNDER ASSESSMENT")
    risk_sig = risk_ctx.get("significance", "")
    risk_cons = risk_ctx.get("consequence", "")

    amount = signals.get("approx_amount_paid", 0)
    timestamp = details.get("timestamp", "unknown")
    src = details.get("src_account", "unknown")
    dst = details.get("dst_account", "unknown")
    y_proba = pred.get("y_proba", 0)

    lap = signals.get("log_amount_paid", 0)
    if lap > 14:
        amount_desc = f"a very large transaction (~{amount:,.0f} units)"
    elif lap > 11:
        amount_desc = f"a significant transaction (~{amount:,.0f} units)"
    else:
        amount_desc = f"a moderate transaction (~{amount:,.0f} units)"

    if head:
        dominant = max(["src", "edge", "dst"],
                       key=lambda k: head.get(f"{k}_drop", 0))
        if dominant == "dst":
            reasoning = "The model's decision was driven by the DESTINATION account's activity."
        elif dominant == "edge":
            reasoning = "The model's decision was driven by the transaction's own characteristics."
        else:
            reasoning = "The model's decision was driven by the SOURCE account's behavior."
    else:
        reasoning = "Model reasoning not available."

    layers = attn.get("layers", [])
    top_in = []
    top_out = []
    n_comp = 0
    if layers:
        last = layers[-1]
        n_comp = last.get("n_competitors", 0)
        top_in = [f"#{n['tx_id']}" for n in last.get("top_incoming", [])[:3]]
        top_out = [f"#{n['tx_id']}" for n in last.get("top_outgoing", [])[:3]]

    evidence = f"""## Transaction Evidence — Transaction #{tx_id}

**Identified Pattern:** {pattern}
**Pattern Description:** {pattern_desc}
**Risk Level:** {risk_level}
**Risk Significance:** {risk_sig}
**Potential Consequence:** {risk_cons}
**Detection Confidence:** {y_proba:.1%}

**Transaction Details:**
- Amount: {amount_desc}
- Date/Time: {timestamp}
- Source Account: {src}
- Destination Account: {dst}

**Model's Reasoning:**
{reasoning}

**Destination Activity (Incoming):**
- {n_comp} competing transactions arrive at the same destination
- Key related: {', '.join(top_in) if top_in else 'none detected'}

**Source Activity (Outgoing):**
- Key related: {', '.join(top_out) if top_out else 'none detected'}

**Task:** Write an investigation report for this {pattern} case.
Explain why flagged, INTERPRET evidence, ASSESS risk, RECOMMEND actions.
Do NOT just describe — ANALYZE."""

    return evidence


# ==========================================
# FASTAPI APP
# ==========================================
app = FastAPI(title="AML Intelligence API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_report_cache = {}


@app.get("/api/stats")
def get_stats():
    return STATS


@app.get("/api/transactions")
def get_transactions(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=100),
    pattern: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    min_risk: float = Query(0.0, ge=0.0, le=1.0),
    max_risk: float = Query(1.0, ge=0.0, le=1.0),
):
    df = DF.copy()

    if pattern and pattern != "All":
        df = df[df["pattern"] == pattern]
    if search:
        mask = (
            df["transaction_id"].astype(str).str.contains(search, na=False) |
            df["src_account"].astype(str).str.contains(search, na=False) |
            df["dst_account"].astype(str).str.contains(search, na=False)
        )
        df = df[mask]
    df = df[(df["y_proba"] >= min_risk) & (df["y_proba"] <= max_risk)]
    df = df.sort_values("y_proba", ascending=False)

    total = len(df)
    start = (page - 1) * per_page
    end = start + per_page
    page_df = df.iloc[start:end]

    return {
        "transactions": page_df.to_dict(orient="records"),
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": math.ceil(total / per_page),
    }


@app.get("/api/transaction/{tx_id}")
def get_transaction(tx_id: int):
    exp = DF_LOOKUP.get(tx_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Transaction {tx_id} not found")

    pred = exp.get("prediction", {})
    details = exp.get("transaction_details", {})
    signals = exp.get("structural_signals", {})
    attn = exp.get("attention_analysis", {})
    head = exp.get("head_component_importance", {})

    top_in = []
    top_out = []
    n_comp = 0
    if attn.get("layers"):
        last = attn["layers"][-1]
        n_comp = last.get("n_competitors", 0)
        top_in = last.get("top_incoming", [])[:5]
        top_out = last.get("top_outgoing", [])[:5]

    dominant = "unknown"
    if head:
        dominant = max(["src", "edge", "dst"],
                       key=lambda k: head.get(f"{k}_drop", 0))

    # ── Attempt data (full network) ──
    attempt_txs = []
    attempt_id = _tx_attempt_map.get(tx_id)
    if attempt_id is not None:
        attempt_txs = _attempt_lookup.get(attempt_id, [])

    return {
        "transaction_id": tx_id,
        "prediction": {
            "y_proba": pred.get("y_proba", 0),
            "y_true": pred.get("y_true", 0),
        },
        "pattern": exp.get("pattern_ground_truth", "UNKNOWN"),
        "pattern_description": PATTERN_DESCRIPTIONS.get(
            exp.get("pattern_ground_truth", ""), ""),
        "risk_context": PATTERN_RISK_CONTEXT.get(
            exp.get("pattern_ground_truth", ""), {}),
        "details": details,
        "signals": {
            "amount": signals.get("approx_amount_paid", 0),
            "log_amount": signals.get("log_amount_paid", 0),
        },
        "reasoning": {
            "dominant_component": dominant,
            "n_competitors": n_comp,
            "top_incoming": top_in,
            "top_outgoing": top_out,
        },
        "attempt": attempt_txs,
    }


@app.post("/api/generate_report/{tx_id}")
def generate_report(tx_id: int):
    exp = DF_LOOKUP.get(tx_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Transaction {tx_id} not found")

    if tx_id in _report_cache:
        return {"report": _report_cache[tx_id], "source": "cache"}

    cache_path = os.path.join(REPORT_CACHE_DIR, f"report_{tx_id}.md")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            report = f.read()
        _report_cache[tx_id] = report
        return {"report": report, "source": "disk-cache"}

    evidence = build_evidence(exp)

    try:
        client = OpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")
        response = client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": evidence},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        report = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"LLM server not available: {e}. "
                   f"Make sure vLLM is running on port 8000."
        )

    _report_cache[tx_id] = report
    with open(cache_path, "w") as f:
        f.write(report)

    return {"report": report, "source": "fresh"}


@app.get("/api/health")
def health():
    return {"status": "ok", "explanations_loaded": len(EXPLANATIONS)}


if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Backend v2 starting on http://localhost:8001")
    print("   API docs: http://localhost:8001/docs")
    print("   Make sure vLLM is running on port 8000!\n")
    uvicorn.run(app, host="0.0.0.0", port=8001)
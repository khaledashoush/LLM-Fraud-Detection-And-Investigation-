# -*- coding: utf-8 -*-
"""
generate_reports.py — LLM Investigation Reports for AML (v4)
================================================================
v4 Improvements:
  [FIX-1] Analytical Depth in SYSTEM_PROMPT (INTERPRET, ASSESS, CONNECT)
  [FIX-2] PATTERN_RISK_CONTEXT — risk level + significance + consequence per pattern
  [FIX-3] Updated Task instruction — demands ANALYSIS not just description

Architecture: LLM as Narrator (GraphXAIN approach)
Paper alignment:
  [GraphXAIN] Cedro & Martens, 2025 — narrative rules
  [SAR/FinCEN] 5 W's structure
  [LLMOps] Naik et al., 2026 — Llama-3.1-8B + vLLM
  [IBM] Altman et al., NeurIPS'23 — 8 pattern definitions

Usage:
  python generate_reports.py --n 10              ← أول 10 تقارير
  python generate_reports.py --n 50              ← 50 تقرير
"""

import os, json, time, logging, argparse, re
import numpy as np
from datetime import datetime
from openai import OpenAI

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("generate_reports")

# ==========================================
# CONFIG
# ==========================================
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
EXPL_PATH = "explanations/explanations.json"
OUTPUT_DIR = "reports"
MAX_TOKENS = 800
TEMPERATURE = 0.1

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

# ★ [FIX-2] Risk context per pattern — enables ANALYTICAL DEPTH
PATTERN_RISK_CONTEXT = {
    "SCATTER-GATHER": {
        "risk_level": "HIGH",
        "significance": "Funds are dispersed through intermediaries before reconvergence, making the money trail extremely difficult to trace. The multiple layers create significant complexity for investigators.",
        "consequence": "If undetected, this scheme can integrate large volumes of illicit funds while making traceback nearly impossible."
    },
    "GATHER-SCATTER": {
        "risk_level": "HIGH",
        "significance": "The convergence of funds from multiple sources followed by redistribution is a classic layering technique. The intermediate account acts as a mixing point for illicit funds.",
        "consequence": "The mixing effect makes it extremely difficult to distinguish illicit from legitimate funds once dispersed."
    },
    "FAN-OUT": {
        "risk_level": "MODERATE to HIGH",
        "significance": "A single source dispersing funds to many destinations is characteristic of smurfing operations, where large amounts are broken into smaller transactions to avoid reporting thresholds.",
        "consequence": "This technique can bypass transaction monitoring thresholds and spread illicit funds across multiple accounts."
    },
    "FAN-IN": {
        "risk_level": "MODERATE to HIGH",
        "significance": "Multiple sources converging on a single destination suggests either a collection point for illicit proceeds or a high-volume business account requiring verification.",
        "consequence": "If this is a collection hub, the volume could be substantial and cross multiple jurisdictions."
    },
    "CYCLE": {
        "risk_level": "HIGH",
        "significance": "Circular fund flows are designed to create complex audit trails. Funds return to their origin after passing through multiple accounts, obscuring the original source.",
        "consequence": "Cycles can repeatedly 'wash' the same funds, amplifying apparent legitimate activity."
    },
    "STACK": {
        "risk_level": "HIGH",
        "significance": "Multi-layered structures indicate sophisticated laundering operations with organizational complexity. Each layer increases the difficulty of tracing original funds.",
        "consequence": "The layered structure suggests professional laundering services rather than amateur activity."
    },
    "BIPARTITE": {
        "risk_level": "MODERATE",
        "significance": "Coordinated transfers between two sets of accounts suggest organized distribution networks, potentially involving mule accounts.",
        "consequence": "The coordinated nature suggests a larger, planned operation rather than isolated activity."
    },
    "RANDOM": {
        "risk_level": "MODERATE",
        "significance": "Random walks through controlled accounts are designed to defeat pattern-based detection systems, indicating awareness of monitoring.",
        "consequence": "The deliberate randomness suggests attempts to evade automated detection."
    },
}


# ==========================================
# ★ [FIX-1] SYSTEM PROMPT (v4 — with Analytical Depth)
# ==========================================
SYSTEM_PROMPT = """You are an experienced AML (Anti-Money Laundering) compliance investigator. An AI fraud detection system (FraudGT) has flagged a transaction as suspicious and identified the laundering pattern. Your task is to write a professional investigation report that goes BEYOND description to provide analytical insights.

## Your Role
- The pattern is ALREADY IDENTIFIED by the detection system
- Your job is to EXPLAIN, INTERPRET, and ASSESS — not just describe
- You are writing for compliance investigators who need actionable understanding

## Analytical Depth Requirements
Beyond stating the facts, you must provide ANALYTICAL INSIGHT:
1. **Interpret**: What does this evidence MEAN in the context of laundering?
2. **Assess**: How serious is this risk? What's the potential impact?
3. **Connect**: How does this transaction fit into the broader scheme?
4. **Implicate**: What could happen if this activity continues undetected?
5. **Contextualize**: Why is this pattern type particularly concerning?

Example of DESCRIPTION (not enough):
"The destination account received 173 transactions."

Example of ANALYSIS (what we need):
"The destination account received 173 transactions from multiple sources, which indicates it serves as a collection hub — a critical node in the laundering network. If this account continues operating unchecked, it could facilitate the integration of significant illicit funds into the legitimate financial system."

## Understanding Pattern Co-existence
A single transaction can be part of MULTIPLE patterns simultaneously:
- The SOURCE may disperse to many destinations (FAN-OUT behavior)
- The DESTINATION may receive from many sources (FAN-IN behavior)
- The identified PATTERN describes the OVERALL scheme structure

## Report Format (SAR — Suspicious Activity Report)

### Executive Summary
2-3 sentences: what happened, the pattern, risk level, and why it matters.

### Subject Information (WHO)
The accounts involved.

### Suspicious Activity (WHAT)
Transaction details + identified pattern + what this pattern represents in the laundering context.

### Temporal Analysis (WHEN)
Timing context + is the timing significant? What does the timing suggest?

### Network Analysis (WHERE)
Structural position + what role do these accounts play in the network?

### Suspicion Basis (WHY)
The evidence with INTERPRETATION:
- State the evidence
- Explain what it MEANS in the laundering context
- Assess the risk level (HIGH/MODERATE/LOW) and why
- Describe potential consequences if undetected
Use cause-and-effect narrative. Show analytical thinking.

### Recommended Actions
Suggested next steps with reasoning:
- WHY each action is recommended
- What specific outcome it aims to achieve

IMPORTANT: Risk levels (HIGH/MODERATE) provided in the evidence are
analytical context for understanding the severity — do NOT translate
them into aggressive actions. Always use "consider", "recommend", or
"suggest" language. Never recommend freezing, blocking, or shutting
down accounts. You are advising the compliance team, not issuing orders.

## Writing Rules
1. Start immediately — no preamble
2. Every fact should be followed by its SIGNIFICANCE
3. Use cause-and-effect: "because", "which indicates", "this suggests"
4. Do NOT use numeric model scores or technical terms
5. Assess risk explicitly: "this represents a HIGH/MODERATE risk because..."
6. Acknowledge ambiguity if present
7. Keep under 450 words
"""


# ==========================================
# HELPER: Time decoder
# ==========================================
def decode_time(timestamp_str):
    """Extract readable time info from timestamp."""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace(" ", "T"))
        hour = dt.hour
        dow = dt.weekday()

        if 6 <= hour < 12:
            time_of_day = "morning hours"
        elif 12 <= hour < 18:
            time_of_day = "afternoon hours"
        elif 18 <= hour < 22:
            time_of_day = "evening hours"
        else:
            time_of_day = "late night hours (outside typical business hours)"

        day_type = "weekend" if dow >= 5 else "business day"

        anomaly = ""
        if hour < 6 or hour >= 22:
            anomaly = " The timing is notable as it falls outside standard banking hours."
        elif dow >= 5:
            anomaly = " The timing is notable as it occurred on a weekend."

        return {
            "time_of_day": time_of_day,
            "day_type": day_type,
            "anomaly": anomaly,
            "formatted": dt.strftime("%B %d, %Y at %H:%M"),
        }
    except:
        return {
            "time_of_day": "unspecified time",
            "day_type": "unspecified day",
            "anomaly": "",
            "formatted": timestamp_str,
        }


# ==========================================
# ★ [FIX-3] EVIDENCE BUILDER (v4 — with risk context)
# ==========================================
def build_evidence(exp):
    """
    Builds evidence with:
    - Pattern identification (from detection system)
    - Risk context (level + significance + consequence)
    - BOTH source and destination information
    - Readable temporal info
    - Analytical task instruction
    """
    tx_id = exp["transaction_id"]
    pred = exp["prediction"]
    details = exp.get("transaction_details", {})
    signals = exp.get("structural_signals", {})
    attn = exp.get("attention_analysis", {})
    head = exp.get("head_component_importance", {})

    # ── Pattern + risk context ──
    pattern = exp.get("pattern_ground_truth", "UNKNOWN")
    pattern_desc = PATTERN_DESCRIPTIONS.get(pattern, "Unknown pattern")
    risk_ctx = PATTERN_RISK_CONTEXT.get(pattern, {})
    risk_level = risk_ctx.get("risk_level", "UNDER ASSESSMENT")
    risk_significance = risk_ctx.get("significance", "")
    risk_consequence = risk_ctx.get("consequence", "")

    # ── Basics ──
    amount = signals.get("approx_amount_paid", 0)
    timestamp = details.get("timestamp", "unknown")
    src = details.get("src_account", "unknown")
    dst = details.get("dst_account", "unknown")
    y_proba = pred.get("y_proba", 0)

    # Amount description
    lap = signals.get("log_amount_paid", 0)
    if lap > 14:
        amount_desc = f"a very large transaction (~{amount:,.0f} units)"
    elif lap > 11:
        amount_desc = f"a significant transaction (~{amount:,.0f} units)"
    else:
        amount_desc = f"a moderate transaction (~{amount:,.0f} units)"

    # Readable time
    time_info = decode_time(timestamp)

    # ── Attention: BOTH directions ──
    layers = attn.get("layers", [])
    top_incoming = []
    top_outgoing = []
    n_comp = 0

    if layers:
        last = layers[-1]
        n_comp = last.get("n_competitors", 0)
        for n in last.get("top_incoming", [])[:3]:
            top_incoming.append(f"#{n['tx_id']}")
        for n in last.get("top_outgoing", [])[:3]:
            top_outgoing.append(f"#{n['tx_id']}")

    # ── Head component → reasoning ──
    if head:
        dominant = max(["src", "edge", "dst"],
                       key=lambda k: head.get(f"{k}_drop", 0))
        if dominant == "dst":
            reasoning = ("The model's decision was primarily influenced by the DESTINATION account's activity. "
                        "The pattern of funds arriving at this destination was the key factor.")
        elif dominant == "edge":
            reasoning = ("The model's decision was primarily influenced by the transaction's own characteristics "
                        "(amount, timing, and structural signals of this specific transaction).")
        else:
            reasoning = ("The model's decision was primarily influenced by the SOURCE account's behavior "
                        "and its pattern of outgoing transactions.")

    # ── Direction analysis ──
    if n_comp > 50 and top_outgoing:
        direction_note = ("NOTE: The destination shows high incoming activity AND the source shows outgoing "
                         "activity. This is common in laundering schemes — the source disperses funds while "
                         "the destination simultaneously receives from multiple sources. The identified "
                         f"{pattern} pattern describes the OVERALL scheme structure.")
    elif n_comp > 50:
        direction_note = (f"The destination is a high-activity account receiving from many sources. "
                         f"In the context of a {pattern} pattern, this indicates the destination is a "
                         "significant node in the laundering network.")
    else:
        direction_note = ""

    # ── Build evidence ──
    evidence = f"""## Transaction Evidence — Transaction #{tx_id}

**Identified Pattern:** {pattern}
**Pattern Description:** {pattern_desc}
**Risk Level:** {risk_level}
**Risk Significance:** {risk_significance}
**Potential Consequence:** {risk_consequence}
**Detection Confidence:** {y_proba:.1%}

**Transaction Details:**
- Amount: {amount_desc}
- Date/Time: {time_info['formatted']} ({time_info['time_of_day']}, {time_info['day_type']}){time_info['anomaly']}
- Source Account: {src}
- Destination Account: {dst}

**Model's Reasoning:**
{reasoning}

**Destination Activity (Incoming):**
- {n_comp} competing transactions arrive at the same destination account
- Key related incoming transactions: {', '.join(top_incoming) if top_incoming else 'none detected in this context'}

**Source Activity (Outgoing):**
- Key related outgoing transactions from the source: {', '.join(top_outgoing) if top_outgoing else 'none detected in this context'}

{direction_note}

**Task:** Write an investigation report for this {pattern} case.

Your report must:
1. Explain why the model flagged this transaction (the evidence)
2. INTERPRET what the evidence means in the laundering context
3. ASSESS the risk level and potential consequences
4. RECOMMEND specific actions with reasoning

Do NOT just describe the evidence — ANALYZE it. Every fact should be followed by its significance. Show the compliance team WHY this matters and WHAT could happen if it's ignored."""

    return evidence


# ==========================================
# LLM Client
# ==========================================
def generate_report(client, evidence):
    """Send prompt to vLLM and get the report."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": evidence},
    ]
    try:
        response = client.chat.completions.create(
            model=VLLM_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return None


# ==========================================
# REPORT EVALUATION
# ==========================================
def evaluate_report(report_text, exp):
    """Evaluate report quality (deterministic — no LLM)."""
    if not report_text:
        return None

    report_lower = report_text.lower()

    # Section coverage
    has_summary = bool(re.search(r'executive summary', report_lower))
    has_who = bool(re.search(r'subject information', report_lower))
    has_what = bool(re.search(r'suspicious activity', report_lower))
    has_when = bool(re.search(r'temporal analysis', report_lower))
    has_where = bool(re.search(r'network analysis', report_lower))
    has_why = bool(re.search(r'suspicion basis', report_lower))
    has_actions = bool(re.search(r'recommend', report_lower))

    # Evidence faithfulness
    details = exp.get("transaction_details", {})
    src = details.get("src_account", "")
    dst = details.get("dst_account", "")
    pattern = exp.get("pattern_ground_truth", "")

    mentions_src = src.split("_")[-1][:6] in report_text if src else False
    mentions_dst = dst.split("_")[-1][:6] in report_text if dst else False
    mentions_pattern = pattern.lower() in report_lower

    # Check for aggressive recommendations
    has_aggressive = bool(re.search(r'freeze|immediately block|shut down', report_lower))

    # Check for analytical depth indicators
    has_risk_assessment = bool(re.search(r'\b(high|moderate|low|significant)\s+risk\b', report_lower))
    has_consequence = bool(re.search(r'if.*undetected|could result in|may lead to|potential(ly)? (result|lead|cause)', report_lower))
    has_significance = bool(re.search(r'this (suggests|indicates|means|represents)|which (suggests|indicates|means)', report_lower))

    # Word count
    word_count = len(report_text.split())

    # Validity
    is_valid = has_summary and has_why and has_actions

    return {
        "is_valid": is_valid,
        "sections": {
            "summary": has_summary, "who": has_who, "what": has_what,
            "when": has_when, "where": has_where, "why": has_why,
            "actions": has_actions,
        },
        "faithfulness": {
            "mentions_source": mentions_src,
            "mentions_destination": mentions_dst,
            "mentions_pattern": mentions_pattern,
        },
        "analytical_depth": {
            "has_risk_assessment": has_risk_assessment,
            "has_consequence": has_consequence,
            "has_significance": has_significance,
        },
        "has_aggressive_recommendations": has_aggressive,
        "word_count": word_count,
    }


# ==========================================
# MAIN
# ==========================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_start = time.time()

    # ── Load data ──
    logger.info(f"[Load] {EXPL_PATH}")
    with open(EXPL_PATH) as f:
        data = json.load(f)
    explanations = data.get("explanations", [])

    selected = explanations[args.offset:args.offset + args.n]
    logger.info(f"[Select] {len(selected)} transactions")

    # ── Connect to vLLM ──
    client = OpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")
    try:
        client.models.list()
        logger.info(f"[vLLM] connected ✓")
    except Exception as e:
        logger.error(f"[vLLM] Cannot connect: {e}")
        return

    # ── Generate reports ──
    reports, results = [], []

    for i, exp in enumerate(selected):
        tx_id = exp["transaction_id"]
        pattern = exp.get("pattern_ground_truth", "UNKNOWN")

        logger.info(f"\n[{i+1}/{len(selected)}] Transaction #{tx_id} ({pattern})")

        evidence = build_evidence(exp)
        report_text = generate_report(client, evidence)

        if not report_text:
            logger.warning(f"  ⚠️ Failed")
            continue

        quality = evaluate_report(report_text, exp)

        result = {
            "transaction_id": tx_id,
            "pattern": pattern,
            "is_valid": quality["is_valid"],
            "word_count": quality["word_count"],
            "sections": quality["sections"],
            "faithfulness": quality["faithfulness"],
            "analytical_depth": quality["analytical_depth"],
            "has_aggressive_recommendations": quality["has_aggressive_recommendations"],
        }
        results.append(result)
        reports.append({"transaction_id": tx_id, "pattern": pattern, "report": report_text})

        # Save individual report
        with open(os.path.join(OUTPUT_DIR, f"report_{tx_id}.md"), "w") as f:
            f.write(report_text)

        # Log
        valid = "✓" if quality["is_valid"] else "✗"
        sections_ok = sum(quality["sections"].values())
        faith_ok = sum(quality["faithfulness"].values())
        depth_ok = sum(quality["analytical_depth"].values())
        aggressive = "⚠️" if quality["has_aggressive_recommendations"] else ""
        logger.info(f"  Valid: {valid} | Sections: {sections_ok}/7 | "
                     f"Faithful: {faith_ok}/3 | Depth: {depth_ok}/3 | "
                     f"Words: {quality['word_count']} {aggressive}")

    # ── Summary ──
    logger.info("\n" + "=" * 70)
    logger.info(" REPORT QUALITY SUMMARY (v4 — with Analytical Depth)")
    logger.info("=" * 70)

    if results:
        n_valid = sum(1 for r in results if r["is_valid"])
        n_aggressive = sum(1 for r in results if r["has_aggressive_recommendations"])
        avg_words = np.mean([r["word_count"] for r in results])

        logger.info(f"Generated: {len(results)} reports")
        logger.info(f"Valid: {n_valid}/{len(results)} ({n_valid/len(results)*100:.0f}%)")
        logger.info(f"Aggressive recommendations: {n_aggressive}/{len(results)} (target: 0)")
        logger.info(f"Avg words: {avg_words:.0f}")

        # Section coverage
        logger.info(f"\nSection Coverage:")
        for s in ["summary", "who", "what", "when", "where", "why", "actions"]:
            count = sum(1 for r in results if r["sections"].get(s, False))
            logger.info(f"  {s:<12} {count}/{len(results)} ({count/len(results)*100:.0f}%)")

        # Faithfulness
        logger.info(f"\nFaithfulness:")
        for m in ["mentions_source", "mentions_destination", "mentions_pattern"]:
            count = sum(1 for r in results if r["faithfulness"].get(m, False))
            logger.info(f"  {m:<22} {count}/{len(results)} ({count/len(results)*100:.0f}%)")

        # ★ Analytical Depth (NEW in v4)
        logger.info(f"\n★ Analytical Depth:")
        for m in ["has_risk_assessment", "has_consequence", "has_significance"]:
            count = sum(1 for r in results if r["analytical_depth"].get(m, False))
            logger.info(f"  {m:<22} {count}/{len(results)} ({count/len(results)*100:.0f}%)")

        # Per-pattern
        logger.info(f"\nPer-Pattern Quality:")
        by_pattern = {}
        for r in results:
            p = r["pattern"]
            if p not in by_pattern:
                by_pattern[p] = {"total": 0, "valid": 0, "faithful": 0, "depth": 0}
            by_pattern[p]["total"] += 1
            if r["is_valid"]:
                by_pattern[p]["valid"] += 1
            if sum(r["faithfulness"].values()) >= 2:
                by_pattern[p]["faithful"] += 1
            if sum(r["analytical_depth"].values()) >= 2:
                by_pattern[p]["depth"] += 1

        for p, stats in sorted(by_pattern.items(), key=lambda x: -x[1]["total"]):
            logger.info(f"  {p:<20} valid: {stats['valid']}/{stats['total']} "
                         f"| faithful: {stats['faithful']}/{stats['total']} "
                         f"| depth: {stats['depth']}/{stats['total']}")

    # ── Save ──
    summary = {
        "model": VLLM_MODEL,
        "version": "v4",
        "n_generated": len(reports),
        "n_valid": sum(1 for r in results if r["is_valid"]),
        "n_aggressive": n_aggressive,
        "avg_word_count": round(avg_words, 0),
        "elapsed_min": round((time.time() - t_start) / 60, 1),
        "architecture": "LLM as Narrator + Analytical Depth (v4)",
        "results": results,
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "all_reports.json"), "w") as f:
        json.dump(reports, f, indent=2, default=str)

    logger.info(f"\n{'='*70}")
    logger.info(f"✅ Saved {len(reports)} reports to {OUTPUT_DIR}/")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
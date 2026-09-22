# -*- coding: utf-8 -*-
"""
generate_reports.py — LLM Investigation Reports for AML (v3)
================================================================
v3 Improvements (from report review):
  [FIX-1] Currency clarity — add currency context
  [FIX-2] Temporal readability — decode hour_sin/cos to "afternoon/evening"
  [FIX-3] Direction clarity — explain how FAN-OUT and FAN-IN co-exist
  [FIX-4] Recommendation softening — "consider" instead of direct actions

Architecture: LLM as Narrator (GraphXAIN approach)
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


# ==========================================
# SYSTEM PROMPT (v3 — improved)
# ==========================================
SYSTEM_PROMPT = """You are an experienced AML (Anti-Money Laundering) compliance investigator. An AI fraud detection system (FraudGT) has flagged a transaction as suspicious and identified the laundering pattern. Your task is to write a professional investigation report.

## Your Role
- The pattern is ALREADY IDENTIFIED by the detection system
- Your job is to EXPLAIN the pattern and evidence in clear language
- You are writing for compliance investigators, not technical audiences

## Understanding Pattern Co-existence
Important: A single transaction can be part of MULTIPLE patterns simultaneously.
- The SOURCE account may be dispersing funds to many destinations (FAN-OUT behavior)
- At the same time, the DESTINATION account may be receiving funds from many sources (FAN-IN behavior)
- The identified PATTERN describes the OVERALL scheme, not just this one transaction
- For example: in a FAN-OUT scheme, the source sends to many destinations, AND each destination may also receive from other sources

## Report Format (SAR — Suspicious Activity Report)

### Executive Summary
2-3 sentences: what happened, the pattern, and the risk level.

### Subject Information (WHO)
The accounts involved.

### Suspicious Activity (WHAT)
The transaction details, amount, and the identified pattern.

### Temporal Analysis (WHEN)
Timing context — is the timing unusual or notable?

### Network Analysis (WHERE)
Structural position: what's happening around the source and destination.

### Suspicion Basis (WHY)
The evidence — explain in narrative (story-like) format:
- WHY this transaction fits the identified pattern
- WHAT specific evidence supports the suspicion
- HOW the model arrived at its decision
Use cause-and-effect language. Focus on the top 5-7 pieces of evidence.

### Recommended Actions
Suggested next steps. Use "consider" or "recommend" language —
you are advising, not ordering. Do NOT recommend freezing accounts
or extreme actions unless the evidence is overwhelming.

## Writing Rules
1. Start immediately — no preamble or disclaimers
2. Use cause-and-effect: "because", "which indicates", "this suggests"
3. Do NOT use numeric model scores or technical terms
4. Describe evidence in plain, accessible language
5. Acknowledge ambiguity if present
6. Keep under 400 words
7. Be specific — reference actual transaction details
"""


# ==========================================
# HELPER: Time decoder
# [FIX-2] Convert timestamp to readable time
# ==========================================
def decode_time(timestamp_str):
    """Extract readable time info from timestamp."""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace(" ", "T"))
        hour = dt.hour
        dow = dt.weekday()  # 0=Monday, 6=Sunday

        # Time of day
        if 6 <= hour < 12:
            time_of_day = "morning hours"
        elif 12 <= hour < 18:
            time_of_day = "afternoon hours"
        elif 18 <= hour < 22:
            time_of_day = "evening hours"
        else:
            time_of_day = "late night hours (outside typical business hours)"

        # Day type
        if dow >= 5:
            day_type = "weekend"
        else:
            day_type = "business day"

        # Anomaly note
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
# EVIDENCE BUILDER (v3)
# ==========================================
def build_evidence(exp):
    """
    Builds evidence with:
    [FIX-1] Currency context
    [FIX-2] Readable temporal info
    [FIX-3] Clear direction analysis (both source AND destination)
    [FIX-4] Softened action language in prompt (handled in SYSTEM_PROMPT)
    """
    tx_id = exp["transaction_id"]
    pred = exp["prediction"]
    details = exp.get("transaction_details", {})
    signals = exp.get("structural_signals", {})
    attn = exp.get("attention_analysis", {})
    head = exp.get("head_component_importance", {})

    # ── Pattern (already identified) ──
    pattern = exp.get("pattern_ground_truth", "UNKNOWN")
    pattern_desc = PATTERN_DESCRIPTIONS.get(pattern, "Unknown pattern")

    # ── Basics ──
    amount = signals.get("approx_amount_paid", 0)
    timestamp = details.get("timestamp", "unknown")
    src = details.get("src_account", "unknown")
    dst = details.get("dst_account", "unknown")
    y_proba = pred.get("y_proba", 0)

    # [FIX-1] Amount with context
    lap = signals.get("log_amount_paid", 0)
    if lap > 14:
        amount_desc = f"a very large transaction (~{amount:,.0f} units)"
    elif lap > 11:
        amount_desc = f"a significant transaction (~{amount:,.0f} units)"
    else:
        amount_desc = f"a moderate transaction (~{amount:,.0f} units)"

    # [FIX-2] Readable time
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

    # [FIX-3] Direction analysis — explain co-existence
    incoming_str = (f"{n_comp} transactions arrive at the destination account"
                    if n_comp > 0 else "limited incoming activity detected")
    outgoing_str = (f"outgoing transactions from the source account were detected"
                    if top_outgoing else "limited outgoing activity from the source in this context")

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
**Risk Score:** {y_proba:.1%} confidence

**Transaction Details:**
- Amount: {amount_desc}
- Date/Time: {time_info['formatted']} ({time_info['time_of_day']}, {time_info['day_type']}){time_info['anomaly']}
- Source Account: {src}
- Destination Account: {dst}

**Model's Reasoning:**
{reasoning}

**Destination Activity (Incoming):**
- {incoming_str}
- Key related incoming transactions: {', '.join(top_incoming) if top_incoming else 'none detected in this context'}

**Source Activity (Outgoing):**
- {outgoing_str}
- Key related outgoing transactions: {', '.join(top_outgoing) if top_outgoing else 'none detected in this context'}

{direction_note}

**Task:** Write an investigation report for this {pattern} case. Explain why the model flagged this transaction, what the {pattern} pattern means in this context, and recommend appropriate actions."""

    return evidence


# ==========================================
# LLM Client
# ==========================================
def generate_report(client, evidence):
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
    """Evaluate report quality."""
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

    # [FIX-4] Check for aggressive recommendations
    has_aggressive = bool(re.search(r'freeze|immediately block|shut down', report_lower))

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

    logger.info(f"[Load] {EXPL_PATH}")
    with open(EXPL_PATH) as f:
        data = json.load(f)
    explanations = data.get("explanations", [])

    selected = explanations[args.offset:args.offset + args.n]
    logger.info(f"[Select] {len(selected)} transactions")

    client = OpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")
    try:
        client.models.list()
        logger.info(f"[vLLM] connected ✓")
    except Exception as e:
        logger.error(f"[vLLM] Cannot connect: {e}")
        return

    reports, results = [], []

    for i, exp in enumerate(selected):
        tx_id = exp["transaction_id"]
        pattern = exp.get("pattern_ground_truth", "UNKNOWN")

        logger.info(f"\n[{i+1}/{len(selected)}] Transaction #{tx_id} ({pattern})")

        evidence = build_evidence(exp)
        report_text = generate_report(client, evidence)

        if not report_text:
            continue

        quality = evaluate_report(report_text, exp)

        result = {
            "transaction_id": tx_id,
            "pattern": pattern,
            "is_valid": quality["is_valid"],
            "word_count": quality["word_count"],
            "sections": quality["sections"],
            "faithfulness": quality["faithfulness"],
            "has_aggressive_recommendations": quality["has_aggressive_recommendations"],
        }
        results.append(result)
        reports.append({"transaction_id": tx_id, "pattern": pattern, "report": report_text})

        with open(os.path.join(OUTPUT_DIR, f"report_{tx_id}.md"), "w") as f:
            f.write(report_text)

        valid = "✓" if quality["is_valid"] else "✗"
        sections_ok = sum(quality["sections"].values())
        faith_ok = sum(quality["faithfulness"].values())
        aggressive = "⚠️" if quality["has_aggressive_recommendations"] else ""
        logger.info(f"  Valid: {valid} | Sections: {sections_ok}/7 | "
                     f"Faithful: {faith_ok}/3 | Words: {quality['word_count']} {aggressive}")

    # ═══ Summary ═══
    logger.info("\n" + "=" * 70)
    logger.info(" REPORT QUALITY SUMMARY (v3)")
    logger.info("=" * 70)

    if results:
        n_valid = sum(1 for r in results if r["is_valid"])
        n_aggressive = sum(1 for r in results if r["has_aggressive_recommendations"])
        avg_words = np.mean([r["word_count"] for r in results])

        logger.info(f"Generated: {len(results)} | Valid: {n_valid}/{len(results)}")
        logger.info(f"Aggressive recommendations: {n_aggressive}/{len(results)} (target: 0)")
        logger.info(f"Avg words: {avg_words:.0f}")

        logger.info(f"\nSection Coverage:")
        for s in ["summary", "who", "what", "when", "where", "why", "actions"]:
            count = sum(1 for r in results if r["sections"].get(s, False))
            logger.info(f"  {s:<12} {count}/{len(results)} ({count/len(results)*100:.0f}%)")

        logger.info(f"\nFaithfulness:")
        for m in ["mentions_source", "mentions_destination", "mentions_pattern"]:
            count = sum(1 for r in results if r["faithfulness"].get(m, False))
            logger.info(f"  {m:<22} {count}/{len(results)} ({count/len(results)*100:.0f}%)")

    # ═══ Save ═══
    summary = {
        "model": VLLM_MODEL, "version": "v3",
        "n_generated": len(reports),
        "n_valid": sum(1 for r in results if r["is_valid"]),
        "n_aggressive": n_aggressive,
        "avg_word_count": round(avg_words, 0),
        "elapsed_min": round((time.time() - t_start) / 60, 1),
        "results": results,
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "all_reports.json"), "w") as f:
        json.dump(reports, f, indent=2, default=str)

    logger.info(f"\n✅ Saved {len(reports)} reports to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
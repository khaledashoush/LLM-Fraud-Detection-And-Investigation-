# -*- coding: utf-8 -*-
"""
evaluate_reports.py — Multi-Layer Evaluation for SAR Reports
================================================================
Evaluates report quality across 5 layers:
  Layer 1: Deterministic Checks (automated — no LLM)
  Layer 2: Evidence Faithfulness (automated — no LLM)
  Layer 3: Compliance Accuracy (LLM-as-Judge)
  Layer 4: Narrative Quality (LLM-as-Judge)
  Layer 5: Overall Summary + Per-Pattern Breakdown

Paper references:
  [LLMOps] Naik et al., 2026 — Section 4: schema-valid goodput
           Section 4.1: LLM-as-Judge Quality Gate
  [GraphXAIN] Cedro & Martens, 2025 — Section 6: 8 evaluation dimensions
  [GNNExplainer] Ying et al., NeurIPS'19 — Section 4.1: fidelity concept
  [xFraud] Rao et al., PVLDB'22 — Section 3.4: agreement metrics
  [FinCEN] SAR narrative guidelines — 5 W's framework
  [IBM] Altman et al., NeurIPS'23 — Section 2.2: SAR requirements

Usage:
  python evaluate_reports.py                      # evaluate all reports
  python evaluate_reports.py --skip-llm           # Layers 1+2 only (fast)
"""

import os, json, re, time, logging, argparse
import numpy as np
from collections import defaultdict
from openai import OpenAI

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("evaluate_reports")

# ==========================================
# CONFIG
# ==========================================
REPORTS_PATH = "reports/all_reports.json"
EXPL_PATH = "explanations/explanations.json"
OUTPUT_PATH = "reports/evaluation_results.json"
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

SAR_SECTIONS = [
    "executive summary", "subject information", "suspicious activity",
    "temporal analysis", "network analysis", "suspicion basis",
    "recommended actions"
]

AGGRESSIVE_PATTERNS = [
    r'\bfreeze\b', r'\bimmediately block\b', r'\bshut down\b',
    r'\bterminate\b', r'\bclose all accounts\b', r'\bseize\b'
]


# ==========================================
# LAYER 1: DETERMINISTIC CHECKS
# [LLMOps, Section 4: "schema-valid goodput"]
# ==========================================
def layer1_deterministic(report_text, evidence_exp):
    """
    Automated checks — no LLM needed.
    "A malformed or unsupported output is operationally equivalent to failure."
    """
    report_lower = report_text.lower()
    details = evidence_exp.get("transaction_details", {})
    signals = evidence_exp.get("structural_signals", {})

    # ── 1a. Schema validity: all SAR sections present ──
    sections_found = {}
    for section in SAR_SECTIONS:
        # Flexible matching (partial section names OK)
        found = bool(re.search(section.replace(" ", r"[\s_]*"), report_lower))
        sections_found[section] = found
    n_sections = sum(sections_found.values())
    schema_valid = n_sections >= 5  # at least 5 of 7

    # ── 1b. Field completeness ──
    src = details.get("src_account", "")
    dst = details.get("dst_account", "")
    pattern = evidence_exp.get("pattern_ground_truth", "")
    amount = signals.get("approx_amount_paid", 0)

    has_src = bool(src and src.split("_")[-1][:6] in report_text)
    has_dst = bool(dst and dst.split("_")[-1][:6] in report_text)
    has_pattern = pattern.lower() in report_lower
    has_amount = bool(re.search(r'\d{2,}', report_text))  # any substantial number

    fields_complete = has_src and has_dst and has_pattern

    # ── 1c. Hallucination check: no invented accounts ──
    # Extract account-like patterns from report
    report_accounts = set(re.findall(r'\b\d+_[A-F0-9]+\b', report_text))
    known_accounts = set()
    if src: known_accounts.add(src)
    if dst: known_accounts.add(dst)

    # Check all attention-mentioned transactions' accounts exist in evidence
    attn = evidence_exp.get("attention_analysis", {})
    for layer in attn.get("layers", []):
        for item in layer.get("top_incoming", []) + layer.get("top_outgoing", []):
            # We don't have account names for these, just IDs — skip
            pass

    hallucinated_accounts = report_accounts - known_accounts
    no_hallucination = len(hallucinated_accounts) == 0

    # ── 1d. Tone check: no aggressive language ──
    aggressive_found = []
    for pattern_re in AGGRESSIVE_PATTERNS:
        if re.search(pattern_re, report_lower):
            aggressive_found.append(pattern_re)
    no_aggressive = len(aggressive_found) == 0

    # ── 1e. Length check ──
    word_count = len(report_text.split())
    length_ok = 150 <= word_count <= 600

    # ── Score ──
    checks = {
        "schema_valid": schema_valid,
        "fields_complete": fields_complete,
        "no_hallucination": no_hallucination,
        "no_aggressive_language": no_aggressive,
        "length_ok": length_ok,
    }
    n_pass = sum(checks.values())
    pass_rate = n_pass / len(checks)

    return {
        "checks": checks,
        "sections_found": sections_found,
        "n_sections": n_sections,
        "sections_detail": {s: v for s, v in sections_found.items()},
        "word_count": word_count,
        "aggressive_phrases": aggressive_found,
        "pass_rate": round(pass_rate, 3),
        "passed": pass_rate >= 0.8,  # at least 4/5
    }


# ==========================================
# LAYER 2: EVIDENCE FAITHFULNESS
# [GNNExplainer, Section 4.1: fidelity]
# [xFraud, Section 3.4: agreement metrics]
# ==========================================
def layer2_faithfulness(report_text, evidence_exp):
    """
    Does the report accurately reflect the evidence?
    Coverage + Accuracy + Attribution + No Fabrication
    """
    report_lower = report_text.lower()
    details = evidence_exp.get("transaction_details", {})
    signals = evidence_exp.get("structural_signals", {})
    attn = evidence_exp.get("attention_analysis", {})
    pred = evidence_exp.get("prediction", {})

    evidence_items = []  # track what evidence exists
    evidence_mentioned = []  # track what's in the report

    # ── Evidence item 1: Source account ──
    src = details.get("src_account", "")
    if src:
        evidence_items.append("source_account")
        if src.split("_")[-1][:6] in report_text:
            evidence_mentioned.append("source_account")

    # ── Evidence item 2: Destination account ──
    dst = details.get("dst_account", "")
    if dst:
        evidence_items.append("destination_account")
        if dst.split("_")[-1][:6] in report_text:
            evidence_mentioned.append("destination_account")

    # ── Evidence item 3: Pattern ──
    pattern = evidence_exp.get("pattern_ground_truth", "")
    if pattern:
        evidence_items.append("pattern")
        if pattern.lower() in report_lower:
            evidence_mentioned.append("pattern")

    # ── Evidence item 4: Amount ──
    amount = signals.get("approx_amount_paid", 0)
    if amount > 0:
        evidence_items.append("amount")
        # Check if any number close to the amount is mentioned
        amount_str = f"{int(amount):,}"
        if amount_str in report_text or str(int(amount)) in report_text:
            evidence_mentioned.append("amount")

    # ── Evidence item 5: Number of competing transactions ──
    n_comp = 0
    if attn.get("layers"):
        n_comp = attn["layers"][-1].get("n_competitors", 0)
    if n_comp > 0:
        evidence_items.append("network_activity")
        # Check if any large number is mentioned as transactions
        if re.search(r'\b\d{2,}\b.*\btransactions?\b', report_lower) or \
           re.search(r'\btransactions?\b.*\b\d{2,}\b', report_lower):
            evidence_mentioned.append("network_activity")

    # ── Evidence item 6: Related transaction IDs ──
    tx_ids_in_evidence = set()
    if attn.get("layers"):
        for layer in attn["layers"]:
            for item in layer.get("top_incoming", [])[:3]:
                tx_ids_in_evidence.add(item["tx_id"])
            for item in layer.get("top_outgoing", [])[:3]:
                tx_ids_in_evidence.add(item["tx_id"])
    if tx_ids_in_evidence:
        evidence_items.append("related_transactions")
        n_tx_mentioned = sum(1 for tid in tx_ids_in_evidence
                            if f"#{tid}" in report_text or str(tid) in report_text)
        if n_tx_mentioned >= 1:  # at least one related tx mentioned
            evidence_mentioned.append("related_transactions")

    # ── Evidence item 7: Model reasoning direction ──
    head = evidence_exp.get("head_component_importance", {})
    if head:
        evidence_items.append("model_reasoning")
        if any(kw in report_lower for kw in ["source account", "destination account",
                                              "transaction's own", "model's decision",
                                              "model's reasoning"]):
            evidence_mentioned.append("model_reasoning")

    # ── Fabrication check: facts NOT in evidence ──
    # Check for specific claims that might be invented
    fabricated_claims = []

    # Check if report mentions currency (we don't provide it — should say "units")
    if re.search(r'\$|USD|EUR|GBP|dollars|euros', report_lower):
        fabricated_claims.append("currency_specified")

    # Check for specific dates not in evidence
    evidence_date = details.get("timestamp", "")
    if evidence_date:
        # Extract year from evidence
        import datetime
        try:
            ev_year = datetime.datetime.fromisoformat(evidence_date.replace(" ", "T")).year
            # Check if report mentions a different year prominently
            years_in_report = set(re.findall(r'\b(20\d{2})\b', report_text))
            wrong_years = years_in_report - {str(ev_year)}
            if wrong_years:
                fabricated_claims.append("wrong_date")
        except:
            pass

    # ── Scores ──
    coverage = len(evidence_mentioned) / len(evidence_items) if evidence_items else 0
    fabrication_rate = len(fabricated_claims) / 3 if fabricated_claims else 0  # normalize

    # Accuracy: all mentioned facts correct?
    accuracy_items = []
    if "source_account" in evidence_mentioned:
        accuracy_items.append(True)  # we already verified it's correct
    if "destination_account" in evidence_mentioned:
        accuracy_items.append(True)
    if "pattern" in evidence_mentioned:
        accuracy_items.append(True)
    accuracy = np.mean(accuracy_items) if accuracy_items else 0

    # Overall faithfulness
    faithfulness = (coverage * 0.4) + (accuracy * 0.4) + ((1 - fabrication_rate) * 0.2)

    return {
        "coverage": round(coverage, 3),
        "accuracy": round(accuracy, 3),
        "fabrication_rate": round(fabrication_rate, 3),
        "fabricated_claims": fabricated_claims,
        "evidence_items": evidence_items,
        "evidence_mentioned": evidence_mentioned,
        "evidence_missed": [e for e in evidence_items if e not in evidence_mentioned],
        "overall_faithfulness": round(faithfulness, 3),
    }


# ==========================================
# LAYER 3+4: LLM-as-JUDGE
# [LLMOps, Section 4.1: "multi-judge rubric scoring"]
# [GraphXAIN, Section 6: "8 evaluation dimensions"]
# ==========================================
JUDGE_PROMPT = """You are an expert AML compliance evaluator. Your task is to evaluate the quality of an investigation report generated by an AI system.

You will be given:
1. The EVIDENCE (structured data about the transaction)
2. The REPORT (the investigation report to evaluate)

Score the report on the following dimensions. Be critical but fair.

## Compliance Scores (0-10)

COMPLIANCE_SCORE: How well does the report follow SAR format?
- Are all 5 W's addressed (Who, What, When, Where, Why)?
- Is the format professional and structured?
- Score 0-10

PROFESSIONAL_TONE: Is the language appropriate for compliance?
- Formal, objective, no alarmist language
- Score 1-5

ACTIONABILITY: Are the recommendations useful?
- Specific, practical, proportionate to the evidence
- Score 1-5

## Narrative Quality Scores (1-5, from GraphXAIN dimensions)

UNDERSTANDABILITY: Can a non-technical compliance investigator understand it?
- Clear language, no unexplained jargon
- Score 1-5

INSIGHTFULNESS: Does it provide analytical insights beyond listing facts?
- Connects evidence to meaning, explains significance
- Score 1-5

CONVINCINGNESS: Does the evidence support the conclusions?
- Logical flow, evidence-based reasoning
- Score 1-5

COMMUNICABILITY: Is it suitable for sharing with management/regulators?
- Concise, well-organized, professional
- Score 1-5

## Output Format (respond EXACTLY in this format):

COMPLIANCE_SCORE: [number]
PROFESSIONAL_TONE: [number]
ACTIONABILITY: [number]
UNDERSTANDABILITY: [number]
INSIGHTFULNESS: [number]
CONVINCINGNESS: [number]
COMMUNICABILITY: [number]
RATIONALE: [2-3 sentences explaining your scoring]
"""


def layer34_llm_judge(client, report_text, evidence_text):
    """Send report + evidence to LLM judge for scoring."""
    prompt = (
        f"## EVIDENCE:\n{evidence_text}\n\n"
        f"## REPORT TO EVALUATE:\n{report_text}\n\n"
        f"{JUDGE_PROMPT}"
    )

    try:
        response = client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a critical but fair evaluator."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.0,  # deterministic for evaluation
        )
        judge_output = response.choices[0].message.content

        # Parse structured scores
        scores = {}
        rationale = ""

        patterns = {
            "compliance_score": r'COMPLIANCE_SCORE:\s*(\d+)',
            "professional_tone": r'PROFESSIONAL_TONE:\s*(\d+)',
            "actionability": r'ACTIONABILITY:\s*(\d+)',
            "understandability": r'UNDERSTANDABILITY:\s*(\d+)',
            "insightfulness": r'INSIGHTFULNESS:\s*(\d+)',
            "convincingness": r'CONVINCINGNESS:\s*(\d+)',
            "communicability": r'COMMUNICABILITY:\s*(\d+)',
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, judge_output)
            if match:
                scores[key] = int(match.group(1))
            else:
                scores[key] = None

        rationale_match = re.search(r'RATIONALE:\s*(.+?)(?:\n|$)', judge_output)
        if rationale_match:
            rationale = rationale_match.group(1).strip()

        return {"scores": scores, "rationale": rationale, "raw": judge_output}

    except Exception as e:
        logger.error(f"Judge error: {e}")
        return {"scores": {}, "rationale": "", "raw": ""}


def build_evidence_summary(exp):
    """Compact evidence summary for the judge."""
    pred = exp["prediction"]
    details = exp.get("transaction_details", {})
    signals = exp.get("structural_signals", {})
    pattern = exp.get("pattern_ground_truth", "UNKNOWN")

    return (f"Pattern: {pattern} | "
            f"Risk: {pred.get('y_proba', 0):.1%} | "
            f"Amount: ~{signals.get('approx_amount_paid', 0):,.0f} | "
            f"Source: {details.get('src_account', '?')} | "
            f"Destination: {details.get('dst_account', '?')}")


# ==========================================
# LAYER 5: AGGREGATE SUMMARY
# ==========================================
def aggregate_results(all_results):
    """Compute aggregate metrics across all reports."""
    if not all_results:
        return {}

    # ── Layer 1 aggregates ──
    l1_pass_rates = [r["layer1"]["pass_rate"] for r in all_results]
    l1_passed = sum(1 for r in all_results if r["layer1"]["passed"])

    # ── Layer 2 aggregates ──
    l2_faithfulness = [r["layer2"]["overall_faithfulness"] for r in all_results]
    l2_coverage = [r["layer2"]["coverage"] for r in all_results]
    l2_accuracy = [r["layer2"]["accuracy"] for r in all_results]

    # ── Layer 3+4 aggregates ──
    judge_scores = defaultdict(list)
    for r in all_results:
        if "layer34" in r and r["layer34"].get("scores"):
            for k, v in r["layer34"]["scores"].items():
                if v is not None:
                    judge_scores[k].append(v)

    judge_avgs = {k: round(np.mean(v), 2) for k, v in judge_scores.items() if v}

    # ── Per-pattern breakdown ──
    by_pattern = defaultdict(lambda: {
        "n": 0, "faithfulness": [], "compliance": [], "narrative_quality": []
    })
    for r in all_results:
        p = r["pattern"]
        by_pattern[p]["n"] += 1
        by_pattern[p]["faithfulness"].append(r["layer2"]["overall_faithfulness"])
        if "layer34" in r and r["layer34"].get("scores"):
            scores = r["layer34"]["scores"]
            if scores.get("compliance_score"):
                by_pattern[p]["compliance"].append(scores["compliance_score"])
            narrative_dims = ["understandability", "insightfulness",
                            "convincingness", "communicability"]
            vals = [scores.get(d) for d in narrative_dims if scores.get(d)]
            if vals:
                by_pattern[p]["narrative_quality"].append(np.mean(vals))

    pattern_summary = {}
    for p, stats in by_pattern.items():
        pattern_summary[p] = {
            "n": stats["n"],
            "avg_faithfulness": round(np.mean(stats["faithfulness"]), 3),
            "avg_compliance": round(np.mean(stats["compliance"]), 2) if stats["compliance"] else None,
            "avg_narrative_quality": round(np.mean(stats["narrative_quality"]), 2) if stats["narrative_quality"] else None,
        }

    # ── Overall score ──
    overall_faithfulness = np.mean(l2_faithfulness)
    overall_compliance = judge_avgs.get("compliance_score", 0)
    narrative_dims = ["understandability", "insightfulness",
                     "convincingness", "communicability"]
    overall_narrative = np.mean([judge_avgs.get(d, 0) for d in narrative_dims])

    # Weighted overall (faithfulness 40%, compliance 30%, narrative 30%)
    overall_score = (overall_faithfulness * 0.4 +
                     (overall_compliance / 10) * 0.3 +
                     (overall_narrative / 5) * 0.3)

    return {
        "n_reports": len(all_results),
        "layer1_summary": {
            "avg_pass_rate": round(np.mean(l1_pass_rates), 3),
            "n_passed": l1_passed,
            "pass_rate": round(l1_passed / len(all_results), 3),
        },
        "layer2_summary": {
            "avg_faithfulness": round(overall_faithfulness, 3),
            "avg_coverage": round(np.mean(l2_coverage), 3),
            "avg_accuracy": round(np.mean(l2_accuracy), 3),
        },
        "layer34_summary": judge_avgs,
        "overall_scores": {
            "faithfulness": round(overall_faithfulness, 3),
            "compliance": round(overall_compliance, 2),
            "narrative_quality": round(overall_narrative, 2),
            "overall_score_10": round(overall_score * 10, 2),
        },
        "per_pattern": pattern_summary,
    }


# ==========================================
# MAIN
# ==========================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-llm", action="store_true",
                    help="Skip LLM-as-Judge (Layers 3+4) — faster")
    args = ap.parse_args()

    t_start = time.time()

    # ── Load reports and evidence ──
    logger.info(f"[Load] {REPORTS_PATH}")
    with open(REPORTS_PATH) as f:
        reports = json.load(f)

    logger.info(f"[Load] {EXPL_PATH}")
    with open(EXPL_PATH) as f:
        explanations = json.load(f).get("explanations", [])

    # Build lookup for explanations
    expl_lookup = {e["transaction_id"]: e for e in explanations}

    # ── Connect to LLM (if not skipped) ──
    client = None
    if not args.skip_llm:
        try:
            client = OpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")
            client.models.list()
            logger.info(f"[vLLM] connected ✓ (LLM-as-Judge enabled)")
        except:
            logger.warning(f"[vLLM] not available — skipping LLM layers")
            args.skip_llm = True

    # ── Evaluate each report ──
    all_results = []
    for i, rep in enumerate(reports):
        tx_id = rep["transaction_id"]
        report_text = rep["report"]
        pattern = rep.get("pattern", "UNKNOWN")

        exp = expl_lookup.get(tx_id)
        if not exp:
            logger.warning(f"[{i+1}] No evidence for #{tx_id} — skipping")
            continue

        logger.info(f"\n[{i+1}/{len(reports)}] Evaluating #{tx_id} ({pattern})")

        # Layer 1: Deterministic
        l1 = layer1_deterministic(report_text, exp)
        logger.info(f"  L1: {'✓' if l1['passed'] else '✗'} "
                     f"(pass_rate={l1['pass_rate']:.0%}, sections={l1['n_sections']}/7)")

        # Layer 2: Faithfulness
        l2 = layer2_faithfulness(report_text, exp)
        logger.info(f"  L2: faithfulness={l2['overall_faithfulness']:.2f} "
                     f"(coverage={l2['coverage']:.0%}, accuracy={l2['accuracy']:.0%})")
        if l2["fabricated_claims"]:
            logger.warning(f"      ⚠️ Fabrication: {l2['fabricated_claims']}")

        # Layers 3+4: LLM-as-Judge
        l34 = None
        if not args.skip_llm:
            evidence_summary = build_evidence_summary(exp)
            l34 = layer34_llm_judge(client, report_text, evidence_summary)
            scores = l34.get("scores", {})
            compliance = scores.get("compliance_score", "?")
            understand = scores.get("understandability", "?")
            logger.info(f"  L3+4: compliance={compliance}/10, "
                         f"understandability={understand}/5")
            if l34.get("rationale"):
                logger.info(f"      Judge: {l34['rationale'][:100]}...")

        result = {
            "transaction_id": tx_id,
            "pattern": pattern,
            "layer1": l1,
            "layer2": l2,
        }
        if l34:
            result["layer34"] = l34
        all_results.append(result)

    # ── Layer 5: Aggregate ──
    summary = aggregate_results(all_results)

    # ── Print summary ──
    logger.info("\n" + "=" * 70)
    logger.info(" EVALUATION SUMMARY")
    logger.info(" [LLMOps: schema-valid goodput | GraphXAIN: 8 dimensions]")
    logger.info("=" * 70)

    if summary:
        logger.info(f"Reports evaluated: {summary['n_reports']}")

        logger.info(f"\n📊 Layer 1 — Deterministic Checks:")
        logger.info(f"   Pass rate: {summary['layer1_summary']['pass_rate']:.0%}")
        logger.info(f"   Avg checks passed: {summary['layer1_summary']['avg_pass_rate']:.0%}")

        logger.info(f"\n📊 Layer 2 — Evidence Faithfulness:")
        logger.info(f"   Overall: {summary['layer2_summary']['avg_faithfulness']:.2f}/1.0")
        logger.info(f"   Coverage: {summary['layer2_summary']['avg_coverage']:.0%}")
        logger.info(f"   Accuracy: {summary['layer2_summary']['avg_accuracy']:.0%}")

        if "layer34_summary" in summary and summary["layer34_summary"]:
            logger.info(f"\n📊 Layer 3 — Compliance (LLM-as-Judge):")
            logger.info(f"   Compliance: {summary['layer34_summary'].get('compliance_score', '?')}/10")
            logger.info(f"   Professional tone: {summary['layer34_summary'].get('professional_tone', '?')}/5")
            logger.info(f"   Actionability: {summary['layer34_summary'].get('actionability', '?')}/5")

            logger.info(f"\n📊 Layer 4 — Narrative Quality (LLM-as-Judge):")
            for dim in ["understandability", "insightfulness",
                        "convincingness", "communicability"]:
                val = summary["layer34_summary"].get(dim, "?")
                logger.info(f"   {dim}: {val}/5")

        logger.info(f"\n🏆 OVERALL SCORES:")
        os_scores = summary["overall_scores"]
        logger.info(f"   Faithfulness: {os_scores['faithfulness']:.2f}/1.0")
        logger.info(f"   Compliance: {os_scores['compliance']}/10")
        logger.info(f"   Narrative: {os_scores['narrative_quality']:.2f}/5")
        logger.info(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"   OVERALL: {os_scores['overall_score_10']:.1f}/10")

        if summary.get("per_pattern"):
            logger.info(f"\n📊 Per-Pattern Breakdown:")
            logger.info(f"   {'Pattern':<20} {'N':>3} {'Faith':>6} {'Comp':>6} {'Narr':>6}")
            logger.info(f"   {'─'*45}")
            for p, stats in sorted(summary["per_pattern"].items(),
                                    key=lambda x: -x[1]["n"]):
                f = stats["avg_faithfulness"]
                c = stats["avg_compliance"] or "—"
                n = stats["avg_narrative_quality"] or "—"
                logger.info(f"   {p:<20} {stats['n']:>3} {f:>6.2f} {str(c):>6} {str(n):>6}")

    # ── Save ──
    output = {
        "metadata": {
            "n_reports": len(all_results),
            "evaluation_time_min": round((time.time() - t_start) / 60, 1),
            "llm_judge_used": not args.skip_llm,
            "framework_version": "v1",
            "paper_refs": {
                "layer1": "LLMOps 2026, Section 4 (schema-valid goodput)",
                "layer2": "GNNExplainer NeurIPS'19 Sec 4.1 + xFraud PVLDB'22 Sec 3.4",
                "layer3_4": "LLMOps 2026, Section 4.1 (LLM-as-Judge) + GraphXAIN 2025, Section 6",
                "sar_format": "FinCEN 5 W's + IBM NeurIPS'23, Section 2.2",
            },
        },
        "aggregate": summary,
        "per_report": all_results,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\n✅ Evaluation saved to {OUTPUT_PATH}")
    logger.info(f"   Total time: {(time.time()-t_start)/60:.1f} minutes")


if __name__ == "__main__":
    main()
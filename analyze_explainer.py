# -*- coding: utf-8 -*-
"""
analyze_explainer.py — Per-Pattern Analysis of FraudGT Explanations
====================================================================
Analyzes the 8,957 explanations from explain_fraudgt.py, grouped by
laundering pattern type. No model re-run needed — works on the
existing explanations.json.

Paper references:
  [xFraud] Rao et al., PVLDB'22, Section 5.2:
    "GNNExplainer and centrality measures work well for DIFFERENT
     communities — none of them dominates the other."
    → Different patterns may need different explanation strategies.

  [IBM AMLworld] Altman et al., NeurIPS'23, Section 3.2:
    The 8 patterns have distinct structural definitions.
    → Attention signatures should differ per pattern.

  [InteractiveGNNExplainer] Singh & Mukherjea, 2025, Section 5.2:
    "GAT's explanations tend to be MORE FOCUSED than GCN's...
     High alignment builds confidence."
    → We measure focus (concentrated vs diffuse) per pattern.

  [Jain & Wallace 2019]:
    Occlusion verification — causal evidence from available samples.

Usage:
  python analyze_explainer.py                    ← analyze + print summary
  python analyze_explainer.py --occlusion        ← include occlusion deep-dive
"""

import os, json, math, logging, argparse
import numpy as np
from collections import defaultdict

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("analyze_explainer")

# ==========================================
# CONFIG
# ==========================================
EXPL_PATH = "explanations/explanations.json"
OUTPUT_PATH = "explanations/explainer_analysis.json"

# Pattern display order (by expected structural complexity)
PATTERN_ORDER = [
    "SCATTER-GATHER", "GATHER-SCATTER", "STACK", "CYCLE",
    "FAN-OUT", "FAN-IN", "BIPARTITE", "RANDOM", "LONE",
    "NONE"  # false positives
]

PATTERN_DESCRIPTIONS = {
    "SCATTER-GATHER": "Fan-out then reconvergence (same intermediaries)",
    "GATHER-SCATTER": "Funds converge then disperse again",
    "STACK": "Multi-layer bipartite structure",
    "CYCLE": "Circular flow returning to origin",
    "FAN-OUT": "One source → many destinations",
    "FAN-IN": "Many sources → one destination",
    "BIPARTITE": "Set of inputs → set of outputs",
    "RANDOM": "Random walk through controlled accounts",
    "LONE": "Isolated illicit (no clear pattern)",
    "NONE": "False positive (legitimate transaction)",
}


# ==========================================
# METRICS
# ==========================================
def attention_concentration(top_incoming):
    """
    [InteractiveGNNExplainer, Sec 5.2] Focus metric:
    What fraction of total competitor attention is in the top-3?
    
    High (>0.5) = FOCUSED: model relies on a few key neighbors
    Low  (<0.2) = DIFFUSE: model sees a broad pattern
    """
    if not top_incoming:
        return None
    attns = [c["attn"] for c in top_incoming]
    total = sum(attns)
    if total <= 0:
        return None
    top3 = sum(sorted(attns, reverse=True)[:3])
    return top3 / total


def extract_metrics(exp):
    """Extract all metrics from a single explanation."""
    m = {}

    # ── Attention analysis (from layers) ──
    layers = exp.get("attention_analysis", {}).get("layers", [])
    for L in layers:
        layer_num = L["layer"]
        m[f"own_attn_L{layer_num}"] = L.get("own_attention", 0)
        m[f"rank_L{layer_num}"] = L.get("own_rank", 0)
        m[f"n_comp_L{layer_num}"] = L.get("n_competitors", 0)

        # Focus metric (layer 2 = final layer, most important for prediction)
        if layer_num == 2:
            top_in = L.get("top_incoming", [])
            m["concentration"] = attention_concentration(top_in)
            m["n_top_incoming"] = len(top_in)
            if top_in:
                m["max_comp_attn"] = max(c["attn"] for c in top_in)

    # ── Head component importance ──
    hci = exp.get("head_component_importance", {})
    m["src_drop"] = hci.get("src_drop", 0)
    m["edge_drop"] = hci.get("edge_drop", 0)
    m["dst_drop"] = hci.get("dst_drop", 0)

    # ── Prediction ──
    pred = exp.get("prediction", {})
    m["y_proba"] = pred.get("y_proba", 0)
    m["y_true"] = pred.get("y_true", 0)

    # ── Neighbor occlusion (if available) ──
    no = exp.get("neighbor_occlusion")
    if no:
        m["occl_drop_top"] = no.get("drop_top", 0)
        m["occl_drop_random"] = no.get("drop_random", 0)

    # ── Structural signals ──
    ss = exp.get("structural_signals", {})
    m["amount"] = ss.get("approx_amount_paid", 0)

    return m


def aggregate_stats(metrics_list):
    """Compute statistics for a group of metrics."""
    if not metrics_list:
        return None

    result = {"n": len(metrics_list)}
    keys = [k for k in metrics_list[0].keys()
            if k not in ("y_true",)]  # exclude categorical

    for key in keys:
        values = [m[key] for m in metrics_list if m.get(key) is not None]
        if not values:
            continue
        result[f"{key}_mean"] = round(float(np.mean(values)), 6)
        result[f"{key}_std"] = round(float(np.std(values)), 6)
        result[f"{key}_median"] = round(float(np.median(values)), 6)

    return result


# ==========================================
# MAIN
# ==========================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--occlusion", action="store_true",
                    help="Include occlusion deep-dive analysis")
    args = ap.parse_args()

    logger.info(f"[Load] {EXPL_PATH}")
    with open(EXPL_PATH) as f:
        data = json.load(f)

    explanations = data.get("explanations", [])
    metadata = data.get("metadata", {})
    occlusion_summary = data.get("occlusion_summary", {})

    logger.info(f"[Load] {len(explanations):,} explanations | "
                f"threshold={metadata.get('threshold', '?')}")

    # ═══════════════════════════════════════
    # 1) Group by pattern
    # ═══════════════════════════════════════
    pattern_groups = defaultdict(list)
    for exp in explanations:
        pattern = exp.get("pattern_ground_truth", "UNKNOWN")
        pattern_groups[pattern].append(extract_metrics(exp))

    logger.info(f"[Group] {len(pattern_groups)} patterns found:")
    for p in PATTERN_ORDER:
        if p in pattern_groups:
            logger.info(f"  {p:<18} {len(pattern_groups[p]):>6,}")

    # ═══════════════════════════════════════
    # 2) Per-pattern statistics
    # ═══════════════════════════════════════
    analysis = {"metadata": {
        "n_explanations": len(explanations),
        "threshold": metadata.get("threshold"),
        "paper_refs": {
            "per_pattern_rationale": "xFraud PVLDB'22 Sec 5.2 (community analysis)",
            "pattern_definitions": "IBM AMLworld NeurIPS'23 Sec 3.2",
            "focus_metric": "InteractiveGNNExplainer 2025 Sec 5.2",
            "occlusion": "Jain & Wallace 2019",
        }
    }, "patterns": {}}

    print("\n" + "=" * 100)
    print(" PER-PATTERN ANALYSIS — FraudGT Attention Signatures")
    print(" [xFraud Sec 5.2: per-community analysis | IBM Sec 3.2: pattern defs]")
    print("=" * 100)

    # Table header
    header = (f"{'Pattern':<16} {'N':>6} │ {'Attn_L2':>8} {'Rank_L2':>8} "
              f"{'N_Comp':>7} {'Conc':>6} │ {'Src':>7} {'Edge':>7} {'Dst':>7} "
              f"│ {'Avg$':>10}")
    print(header)
    print("─" * 100)

    for pattern in PATTERN_ORDER:
        if pattern not in pattern_groups:
            continue

        stats = aggregate_stats(pattern_groups[pattern])
        analysis["patterns"][pattern] = {
            "description": PATTERN_DESCRIPTIONS.get(pattern, ""),
            "stats": stats,
        }

        # Format table row
        n = stats["n"]
        attn = stats.get("own_attn_L2_mean", 0)
        rank = stats.get("rank_L2_mean", 0)
        ncomp = stats.get("n_comp_L2_mean", 0)
        conc = stats.get("concentration_mean", 0) or 0
        src_d = stats.get("src_drop_mean", 0)
        edge_d = stats.get("edge_drop_mean", 0)
        dst_d = stats.get("dst_drop_mean", 0)
        amount = stats.get("amount_mean", 0)

        print(f"{pattern:<16} {n:>6,} │ {attn:>8.4f} {rank:>8.1f} "
              f"{ncomp:>7.1f} {conc:>6.3f} │ {src_d:>+7.4f} {edge_d:>+7.4f} "
              f"{dst_d:>+7.4f} │ {amount:>10,.0f}")

    print("─" * 100)
    print(f"{'Legend':}")
    print(f"  Attn_L2 = own attention in final layer (higher = more focused on this edge)")
    print(f"  Rank_L2 = rank among competitors (1 = most attended)")
    print(f"  N_Comp  = avg number of competing edges at same destination")
    print(f"  Conc    = attention concentration (top-3 share, 0-1; high=focused)")
    print(f"  Src/Edge/Dst = head occlusion drops (which component drives decision)")

    # ═══════════════════════════════════════
    # 3) Attention signature comparison
    # ═══════════════════════════════════════
    print("\n" + "=" * 100)
    print(" ATTENTION SIGNATURES — How does FraudGT 'see' each pattern?")
    print(" [InteractiveGNNExplainer: focused vs diffuse explanations]")
    print("=" * 100)

    for pattern in PATTERN_ORDER:
        if pattern not in pattern_groups:
            continue
        stats = analysis["patterns"][pattern]["stats"]
        conc = stats.get("concentration_mean", 0) or 0
        ncomp = stats.get("n_comp_L2_mean", 0)
        attn = stats.get("own_attn_L2_mean", 0)

        # Classify focus
        if conc > 0.5:
            focus = "🎯 FOCUSED"
        elif conc > 0.3:
            focus = "⚖️  BALANCED"
        else:
            focus = "🌐 DIFFUSE"

        # Head component
        components = {
            "src": stats.get("src_drop_mean", 0),
            "edge": stats.get("edge_drop_mean", 0),
            "dst": stats.get("dst_drop_mean", 0),
        }
        dominant = max(components, key=components.get)

        print(f"\n  {pattern} ({PATTERN_DESCRIPTIONS.get(pattern, '')})")
        print(f"    Focus: {focus} (concentration={conc:.3f})")
        print(f"    Own attention: {attn:.4f} | Competitors: {ncomp:.1f}")
        print(f"    Head: {dominant} dominates "
              f"(src={components['src']:+.3f}, edge={components['edge']:+.3f}, "
              f"dst={components['dst']:+.3f})")

    # ═══════════════════════════════════════
    # 4) Key findings (automatic)
    # ═══════════════════════════════════════
    print("\n" + "=" * 100)
    print(" KEY FINDINGS (for thesis)")
    print("=" * 100)

    # Finding 1: LONE detection
    lone_n = len(pattern_groups.get("LONE", []))
    total_tp = sum(len(v) for k, v in pattern_groups.items() if k != "NONE")
    if total_tp > 0:
        lone_pct = lone_n / total_tp * 100
        print(f"\n  1️⃣  LONE Detection Gap:")
        print(f"      LONE: {lone_n}/{total_tp} TPs ({lone_pct:.1f}%)")
        print(f"      → FraudGT nearly misses isolated illicit transactions.")
        print(f"      → Paper basis: IBM AMLworld — LONE = 35.4% of all illicit")
        print(f"         but only {lone_pct:.1f}% of our detections.")

    # Finding 2: Head component across patterns
    edge_dominant = []
    dst_dominant = []
    for pattern in PATTERN_ORDER:
        if pattern not in pattern_groups:
            continue
        stats = analysis["patterns"][pattern]["stats"]
        e = stats.get("edge_drop_mean", 0)
        d = stats.get("dst_drop_mean", 0)
        s = stats.get("src_drop_mean", 0)
        if e > d and e > s:
            edge_dominant.append(pattern)
        elif d > e and d > s:
            dst_dominant.append(pattern)

    print(f"\n  2️⃣  Head Component Variation:")
    print(f"      Edge-dominated: {', '.join(edge_dominant) or 'none'}")
    print(f"      Dst-dominated:  {', '.join(dst_dominant) or 'none'}")
    print(f"      → Different patterns are driven by different components.")

    # Finding 3: Concentration comparison
    concs = {}
    for pattern in PATTERN_ORDER:
        if pattern not in pattern_groups:
            continue
        c = analysis["patterns"][pattern]["stats"].get("concentration_mean")
        if c is not None:
            concs[pattern] = c
    if concs:
        most_focused = max(concs, key=concs.get)
        most_diffuse = min(concs, key=concs.get)
        print(f"\n  3️⃣  Attention Focus:")
        print(f"      Most FOCUSED: {most_focused} (conc={concs[most_focused]:.3f})")
        print(f"      Most DIFFUSE: {most_diffuse} (conc={concs[most_diffuse]:.3f})")
        print(f"      → Structured patterns (cycles, scatter-gather) may show")
        print(f"         distinct attention signatures vs unstructured (random).")

    # ═══════════════════════════════════════
    # 5) Occlusion deep-dive (optional)
    # ═══════════════════════════════════════
    if args.occlusion and occlusion_summary:
        print("\n" + "=" * 100)
        print(" OCCLUSION ANALYSIS — Causal Evidence")
        print(" [Jain & Wallace 2019: attention needs empirical verification]")
        print("=" * 100)

        n_tested = occlusion_summary.get("n_tested", 0)
        top_mean = occlusion_summary.get("top_mean_drop", 0)
        rand_mean = occlusion_summary.get("random_mean_drop", 0)
        ratio = occlusion_summary.get("ratio_top_vs_random", 0)

        print(f"\n  Tested: {n_tested} transactions")
        print(f"  Top-attention occlusion: Δprob = {top_mean:.4f}")
        print(f"  Random occlusion:        Δprob = {rand_mean:.4f}")
        print(f"  Ratio: {ratio:.1f}× → {'CAUSAL ✓' if ratio > 2 else 'weak evidence'}")

        # Per-pattern occlusion (from individual explanations)
        occl_by_pattern = defaultdict(list)
        for exp in explanations:
            no = exp.get("neighbor_occlusion")
            if no:
                pattern = exp.get("pattern_ground_truth", "UNKNOWN")
                occl_by_pattern[pattern].append(no["drop_top"])

        if occl_by_pattern:
            print(f"\n  Per-pattern occlusion Δprob:")
            for pattern, drops in sorted(occl_by_pattern.items(),
                                          key=lambda x: -np.mean(x[1])):
                print(f"    {pattern:<18} n={len(drops):>3} "
                      f"Δ={np.mean(drops):+.4f} ± {np.std(drops):.4f}")

    # ═══════════════════════════════════════
    # 6) Amount analysis per pattern
    # ═══════════════════════════════════════
    print("\n" + "=" * 100)
    print(" TRANSACTION AMOUNTS BY PATTERN")
    print(" [IBM AMLworld: different patterns involve different transaction scales]")
    print("=" * 100)

    for pattern in PATTERN_ORDER:
        if pattern not in pattern_groups:
            continue
        amounts = [m.get("amount", 0) for m in pattern_groups[pattern]
                   if m.get("amount")]
        if amounts:
            print(f"  {pattern:<18} median={np.median(amounts):>12,.0f} "
                  f"mean={np.mean(amounts):>12,.0f} "
                  f"max={np.max(amounts):>14,.0f}")

    # ═══════════════════════════════════════
    # 7) Save
    # ═══════════════════════════════════════
    with open(OUTPUT_PATH, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    logger.info(f"\n✅ Analysis saved to {OUTPUT_PATH}")

    # Summary for LLM layer
    llm_insights = {}
    for pattern in PATTERN_ORDER:
        if pattern not in pattern_groups:
            continue
        stats = analysis["patterns"][pattern]["stats"]
        conc = stats.get("concentration_mean", 0) or 0
        focus_type = "focused" if conc > 0.5 else ("balanced" if conc > 0.3 else "diffuse")
        dominant_comp = max(
            ["src", "edge", "dst"],
            key=lambda c: stats.get(f"{c}_drop_mean", 0))
        llm_insights[pattern] = {
            "focus": focus_type,
            "concentration": round(conc, 3),
            "dominant_component": dominant_comp,
            "avg_competitors": round(stats.get("n_comp_L2_mean", 0), 1),
            "avg_own_attention": round(stats.get("own_attn_L2_mean", 0), 4),
        }

    analysis["llm_layer_insights"] = llm_insights
    with open(OUTPUT_PATH, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    logger.info(f"✅ LLM insights saved (for generate_reports.py)")
    logger.info(f"\nDone! Total: {len(explanations):,} explanations analyzed "
                f"across {len(pattern_groups)} patterns.")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
frontend.py — AML Intelligence Dashboard (Professional v7)
"""

import os, io, json, time, math
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from fpdf import FPDF
from datetime import datetime

BACKEND_URL = "http://localhost:8001"

# ═════════════════════════════════════════════════
# PAGE CONFIG
# ═════════════════════════════════════════════════
st.set_page_config(
    page_title="AML Intelligence | Compliance Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═════════════════════════════════════════════════
# CSS
# ═════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.stApp { background: #0b0f19; color: #e2e8f0; }
#MainMenu, footer, header { display: none !important; }
.stDeployButton { display: none !important; }
.main .block-container { padding: 0 2rem 2rem 2rem; max-width: 1400px; margin-top: 0; }
.topbar { background: linear-gradient(180deg, #111827 0%, #0b0f19 100%); border-bottom: 1px solid #1e293b; padding: 1rem 2rem; display: flex; justify-content: space-between; align-items: center; margin: 0 -2rem 1.5rem -2rem; }
.topbar-title { font-size: 1.3rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.5px; display: flex; align-items: center; gap: 0.6rem; }
.topbar-title span { color: #3b82f6; }
.topbar-status { display: flex; align-items: center; gap: 1.5rem; }
.status-dot { width: 8px; height: 8px; background: #22c55e; border-radius: 50%; display: inline-block; animation: pulse 2s infinite; }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(34,197,94,0.4); } 70% { box-shadow: 0 0 0 6px rgba(34,197,94,0); } 100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); } }
.status-text { color: #22c55e; font-size: 0.8rem; font-weight: 600; }
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
.kpi-card { background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%); border: 1px solid #334155; border-radius: 12px; padding: 1.2rem; position: relative; overflow: hidden; }
.kpi-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; border-radius: 3px 3px 0 0; }
.kpi-card.blue::before { background: #3b82f6; }
.kpi-card.red::before { background: #ef4444; }
.kpi-card.green::before { background: #22c55e; }
.kpi-card.purple::before { background: #a855f7; }
.kpi-value { font-size: 2rem; font-weight: 800; color: #f8fafc; line-height: 1; margin-bottom: 0.4rem; }
.kpi-label { font-size: 0.72rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.8px; }
.section-title { font-size: 1rem; font-weight: 700; color: #f8fafc; margin: 1.5rem 0 1rem 0; padding-left: 0.8rem; border-left: 3px solid #3b82f6; }
.tx-row { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 0.9rem 1.2rem; display: flex; align-items: center; gap: 1rem; transition: all 0.15s ease; }
.tx-row:hover { border-color: #3b82f6; background: #253349; transform: translateX(4px); }
.tx-num { color: #475569; font-size: 0.8rem; font-weight: 600; min-width: 2.5rem; text-align: right; font-family: 'JetBrains Mono', monospace; }
.tx-id { font-family: 'JetBrains Mono', monospace; color: #60a5fa; font-size: 0.9rem; font-weight: 600; min-width: 7rem; }
.tx-badge { padding: 0.25rem 0.7rem; border-radius: 6px; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; min-width: 6.5rem; text-align: center; }
.badge-fan-out { background: #1e3a5f; color: #93c5fd; }
.badge-fan-in { background: #3b1e3a; color: #f9a8d4; }
.badge-stack { background: #1e3a2f; color: #86efac; }
.badge-cycle { background: #3a2e1e; color: #fcd34d; }
.badge-bipartite { background: #2d1e3a; color: #c4b5fd; }
.badge-gather-scatter { background: #1e2d3a; color: #7dd3fc; }
.badge-scatter-gather { background: #3a1e2d; color: #fda4af; }
.badge-random { background: #334155; color: #94a3b8; }
.badge-none { background: #1a2332; color: #64748b; }
.badge-lone { background: #3a3a1e; color: #d9f99d; }
.tx-risk { font-weight: 800; font-size: 0.95rem; min-width: 3.5rem; text-align: center; font-family: 'JetBrains Mono', monospace; }
.risk-high { color: #f87171; }
.risk-medium { color: #fbbf24; }
.tx-amount { color: #94a3b8; font-size: 0.85rem; min-width: 6rem; font-family: 'JetBrains Mono', monospace; }
.tx-date { color: #64748b; font-size: 0.8rem; min-width: 7rem; }
.tx-gt { font-size: 0.75rem; min-width: 4rem; text-align: center; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 600; }
.gt-flag { background: #450a0a; color: #fca5a5; }
.gt-ok { background: #052e16; color: #86efac; }
.detail-header { background: linear-gradient(145deg, #1e293b, #0f172a); border: 1px solid #334155; border-radius: 14px; padding: 1.5rem 2rem; margin-bottom: 1.2rem; display: flex; justify-content: space-between; align-items: center; }
.detail-title { font-size: 1.5rem; font-weight: 800; color: #f8fafc; font-family: 'JetBrains Mono', monospace; }
.detail-sub { color: #64748b; font-size: 0.85rem; margin-top: 0.3rem; }
.info-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.8rem; margin-bottom: 1.2rem; }
.info-card { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 1rem; text-align: center; }
.info-label { color: #64748b; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem; }
.info-value { color: #f8fafc; font-size: 1.1rem; font-weight: 700; }
.graph-legend { background: #1a2332; border: 1px solid #334155; border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 1rem; color: #64748b; font-size: 0.78rem; }
.legend-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin: 0 4px 0 12px; }
.report-box { background: #ffffff; color: #1e293b; border-radius: 12px; padding: 2rem; line-height: 1.8; font-size: 0.95rem; box-shadow: 0 10px 40px rgba(0,0,0,0.3); }
.report-box h1, .report-box h2, .report-box h3 { color: #0f172a !important; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.5rem; }
.stButton > button[kind="primary"] { background: #3b82f6 !important; border: none !important; border-radius: 8px !important; font-weight: 700 !important; padding: 0.5rem 1.5rem !important; }
.stButton > button[kind="primary"]:hover { background: #2563eb !important; transform: translateY(-1px) !important; box-shadow: 0 4px 12px rgba(59,130,246,0.4) !important; }
.js-plotly-plot { border-radius: 10px; }
.pagination { display: flex; justify-content: center; align-items: center; gap: 1rem; margin-top: 1.2rem; padding-top: 1rem; border-top: 1px solid #1e293b; }
.page-info { color: #64748b; font-size: 0.85rem; }
section[data-testid="stSidebar"] { background: #0f172a; border-right: 1px solid #1e293b; }
section[data-testid="stSidebar"] .stMarkdown h3 { color: #f8fafc; font-size: 0.9rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════
# API HELPERS
# ═════════════════════════════════════════════════
def api_get(endpoint, params=None):
    try:
        resp = requests.get(f"{BACKEND_URL}{endpoint}", params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return None
    except:
        return None

def api_post(endpoint):
    try:
        resp = requests.post(f"{BACKEND_URL}{endpoint}", timeout=120)
        if resp.status_code == 200:
            return resp.json()
        return {"error": "Failed"}
    except:
        return {"error": "Connection failed"}

# ═════════════════════════════════════════════════
# PDF EXPORT
# ═════════════════════════════════════════════════
def report_to_pdf(report_text, tx_id, pattern):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(190, 10, "SUSPICIOUS ACTIVITY REPORT (SAR)", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(190, 6, f"Transaction #{tx_id} | Pattern: {pattern}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(190, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    for line in report_text.split("\n"):
        line = line.strip()
        if not line:
            pdf.ln(3)
            continue
        clean = line.encode("ascii", "replace").decode("ascii")
        if clean.startswith("#"):
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(190, 7, clean.lstrip("# ").strip())
            pdf.set_font("Helvetica", "", 9)
        else:
            try:
                pdf.multi_cell(190, 4.5, clean)
            except:
                pass
    return bytes(pdf.output())

# ═════════════════════════════════════════════════
# PATTERN-SPECIFIC LAYOUT FUNCTIONS
# ═════════════════════════════════════════════════

def _get_cycle_order(attempt):
    """Follow the chain: A→B→C→...→A"""
    adj = {}
    for tx in attempt:
        adj[tx["src"]] = tx["dst"]
    if not adj:
        return []
    start = list(adj.keys())[0]
    order = [start]
    current = adj.get(start)
    while current and current != start:
        order.append(current)
        current = adj.get(current)
    return order


def _analyze_attempt(attempt):
    """Analyze network structure."""
    src_count, dst_count = {}, {}
    all_accounts = set()
    for tx in attempt:
        src, dst = tx["src"], tx["dst"]
        all_accounts.add(src)
        all_accounts.add(dst)
        src_count[src] = src_count.get(src, 0) + 1
        dst_count[dst] = dst_count.get(dst, 0) + 1
    return src_count, dst_count, all_accounts


def _positions_cycle(attempt, current_src, current_dst):
    """CYCLE: Ring."""
    order = _get_cycle_order(attempt)
    positions = {}
    if not order:
        return positions
    n = len(order)
    R = max(250, n * 35)
    for i, acc in enumerate(order):
        angle = 2 * math.pi * i / n
        positions[acc] = (int(math.cos(angle) * R), int(math.sin(angle) * R))
    return positions


def _positions_fan_out(attempt, current_src, current_dst):
    """FAN-OUT: Source center."""
    src_count, _, all_acc = _analyze_attempt(attempt)
    main_src = max(src_count, key=src_count.get)
    positions = {main_src: (0, 0)}
    others = [a for a in all_acc if a != main_src]
    n = len(others)
    for i, acc in enumerate(others):
        angle = 2 * math.pi * i / max(n, 1)
        positions[acc] = (int(math.cos(angle) * 350), int(math.sin(angle) * 350))
    return positions


def _positions_fan_in(attempt, current_src, current_dst):
    """FAN-IN: Destination center."""
    _, dst_count, all_acc = _analyze_attempt(attempt)
    main_dst = max(dst_count, key=dst_count.get)
    positions = {main_dst: (0, 0)}
    others = [a for a in all_acc if a != main_dst]
    n = len(others)
    for i, acc in enumerate(others):
        angle = 2 * math.pi * i / max(n, 1)
        positions[acc] = (int(math.cos(angle) * 350), int(math.sin(angle) * 350))
    return positions


def _positions_scatter_gather(attempt, current_src, current_dst):
    """SCATTER-GATHER: Butterfly (LEFT | MIDDLE | RIGHT)."""
    src_count, dst_count, all_acc = _analyze_attempt(attempt)
    main_src = max(src_count, key=src_count.get)
    main_dst = max(dst_count, key=dst_count.get)
    intermediaries = [a for a in all_acc if a not in (main_src, main_dst)]
    positions = {main_src: (-450, 0), main_dst: (450, 0)}
    n = len(intermediaries)
    for i, acc in enumerate(intermediaries):
        y = (i - n / 2) * 90
        positions[acc] = (0, int(y))
    return positions


def _positions_gather_scatter(attempt, current_src, current_dst):
    """GATHER-SCATTER: Hourglass."""
    src_count, dst_count, all_acc = _analyze_attempt(attempt)
    hub_score = {a: min(src_count.get(a, 0), dst_count.get(a, 0)) for a in all_acc}
    hub = max(hub_score, key=hub_score.get)
    sources = [a for a in all_acc if a != hub and src_count.get(a, 0) > dst_count.get(a, 0)]
    dests = [a for a in all_acc if a != hub and a not in sources]
    positions = {hub: (0, 0)}
    for i, acc in enumerate(sources):
        positions[acc] = (-450, int((i - len(sources) / 2) * 90))
    for i, acc in enumerate(dests):
        positions[acc] = (450, int((i - len(dests) / 2) * 90))
    return positions


def _positions_bipartite(attempt, current_src, current_dst):
    """BIPARTITE: Two columns."""
    src_count, dst_count, all_acc = _analyze_attempt(attempt)
    sources = [a for a in all_acc if src_count.get(a, 0) > 0 and a not in dst_count]
    dests = [a for a in all_acc if dst_count.get(a, 0) > 0 and a not in src_count]
    both = [a for a in all_acc if a in src_count and a in dst_count]
    positions = {}
    for i, acc in enumerate(sources):
        positions[acc] = (-450, int((i - len(sources) / 2) * 90))
    for i, acc in enumerate(dests):
        positions[acc] = (450, int((i - len(dests) / 2) * 90))
    for i, acc in enumerate(both):
        positions[acc] = (0, int((i - len(both) / 2) * 90))
    return positions


def _positions_stack(attempt, current_src, current_dst):
    """STACK: Parallel chains."""
    chains = []
    used_txs = set()
    for tx in attempt:
        if tx["tx_id"] in used_txs:
            continue
        chain = [tx["src"], tx["dst"]]
        used_txs.add(tx["tx_id"])
        changed = True
        while changed:
            changed = False
            for t2 in attempt:
                if t2["tx_id"] in used_txs:
                    continue
                if t2["src"] == chain[-1]:
                    chain.append(t2["dst"])
                    used_txs.add(t2["tx_id"])
                    changed = True
        chains.append(chain)
    positions = {}
    for row, chain in enumerate(chains):
        y = (row - len(chains) / 2) * 200
        for i, acc in enumerate(chain):
            if acc not in positions:
                positions[acc] = (int((i - len(chain) / 2) * 150), int(y))
    return positions


def _positions_random(attempt, current_src, current_dst):
    """RANDOM: Wavy chain."""
    _, _, all_acc = _analyze_attempt(attempt)
    accounts = list(all_acc)
    positions = {}
    n = len(accounts)
    for i, acc in enumerate(accounts):
        positions[acc] = (int((i - n / 2) * 150), int(math.sin(i * 0.7) * 100))
    return positions


_POSITION_FUNCS = {
    "CYCLE": _positions_cycle,
    "FAN-OUT": _positions_fan_out,
    "FAN-IN": _positions_fan_in,
    "SCATTER-GATHER": _positions_scatter_gather,
    "GATHER-SCATTER": _positions_gather_scatter,
    "BIPARTITE": _positions_bipartite,
    "STACK": _positions_stack,
    "RANDOM": _positions_random,
}

# ═════════════════════════════════════════════════
# PLOTLY NETWORK GRAPH
# ═════════════════════════════════════════════════

def render_transaction_graph(detail):
    """Beautiful Plotly network graph with pattern-specific layout."""

    attempt = detail.get("attempt", [])
    current_tx = detail["transaction_id"]
    pattern = detail.get("pattern", "")
    details = detail.get("details", {})
    current_src = details.get("src_account", "")
    current_dst = details.get("dst_account", "")

    if not attempt:
        return render_single_tx_graph(detail)

    # ── Calculate positions ──
    pos_func = _POSITION_FUNCS.get(pattern, _positions_random)
    positions = pos_func(attempt, current_src, current_dst)

    # ── Build node list ──
    nodes_data = []
    used = set()

    for tx in attempt:
        src, dst = tx["src"], tx["dst"]

        if src not in used:
            used.add(src)
            is_cur = (src == current_src)
            x, y = positions.get(src, (0, 0))
            nodes_data.append({
                "id": src, "x": x, "y": y,
                "is_cur": is_cur,
                "role": "src" if is_cur else "other",
            })

        if dst not in used:
            used.add(dst)
            is_cur = (dst == current_dst)
            x, y = positions.get(dst, (0, 0))
            nodes_data.append({
                "id": dst, "x": x, "y": y,
                "is_cur": is_cur,
                "role": "dst" if is_cur else "other",
            })

    # ── Edge traces ──
    fig_data = []

    # Regular edges
    for tx in attempt:
        src = tx["src"]
        dst = tx["dst"]
        is_current = tx["tx_id"] == current_tx

        src_n = next((n for n in nodes_data if n["id"] == src), None)
        dst_n = next((n for n in nodes_data if n["id"] == dst), None)
        if not src_n or not dst_n:
            continue

        # Edge line
        fig_data.append(go.Scatter(
            x=[src_n["x"], dst_n["x"]],
            y=[src_n["y"], dst_n["y"]],
            mode="lines",
            line=dict(
                color="#fbbf24" if is_current else "#2d4a6e",
                width=4 if is_current else 2,
            ),
            opacity=1.0 if is_current else 0.7,
            hoverinfo="skip",
            showlegend=False,
        ))

        # Arrow marker at midpoint
        mid_x = (src_n["x"] + dst_n["x"]) / 2
        mid_y = (src_n["y"] + dst_n["y"]) / 2
        fig_data.append(go.Scatter(
            x=[mid_x], y=[mid_y],
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                size=8,
                color="#fbbf24" if is_current else "#4a6b94",
                angle=math.degrees(
                    math.atan2(dst_n["y"] - src_n["y"], dst_n["x"] - src_n["x"])
                ) - 90,
            ),
            hoverinfo="skip",
            showlegend=False,
        ))

    # ── Node trace ──
    node_x = [n["x"] for n in nodes_data]
    node_y = [n["y"] for n in nodes_data]
    node_text = []
    node_sizes = []
    node_colors = []
    node_symbols = []

    for n in nodes_data:
        label = n["id"][-12:]
        node_text.append(f"<b>{label}</b><br>{'⭐ Current' if n['is_cur'] else n['id']}")

        if n["is_cur"] and n["role"] == "src":
            node_colors.append("#ef4444")
            node_sizes.append(28)
            node_symbols.append("circle")
        elif n["is_cur"] and n["role"] == "dst":
            node_colors.append("#3b82f6")
            node_sizes.append(28)
            node_symbols.append("circle")
        else:
            node_colors.append("#475569")
            node_sizes.append(14)
            node_symbols.append("circle")

    fig_data.append(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(color="#f8fafc", width=1),
        ),
        text=[n["id"][-10:] for n in nodes_data],
        textposition="top center",
        textfont=dict(color="#94a3b8", size=9),
        hovertext=node_text,
        hoverinfo="text",
        showlegend=False,
    ))

    # ── Layout ──
    fig = go.Figure(data=fig_data)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=500,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(
            showgrid=False, zeroline=False,
            showticklabels=False, range=[-550, 550],
        ),
        yaxis=dict(
            showgrid=False, zeroline=False,
            showticklabels=False, range=[-450, 450],
        ),
        font=dict(family="Inter, sans-serif"),
    )

    return fig


def render_single_tx_graph(detail):
    """Fallback: single transaction with attention analysis."""
    reasoning = detail.get("reasoning", {})
    details = detail.get("details", {})
    src = details.get("src_account", "src")
    dst = details.get("dst_account", "dst")
    top_in = reasoning.get("top_incoming", [])
    top_out = reasoning.get("top_outgoing", [])

    # Positions: src at left, dst at right, incoming top, outgoing bottom
    positions = {src: (-300, 0), dst: (300, 0)}
    for i, item in enumerate(top_in[:5]):
        nid = f"in_{item['tx_id']}"
        positions[nid] = (200, (i - 2) * 80)
    for i, item in enumerate(top_out[:5]):
        nid = f"out_{item['tx_id']}"
        positions[nid] = (-200, (i - 2) * 80)

    fig_data = []

    # Edges
    for item in top_in[:5]:
        nid = f"in_{item['tx_id']}"
        attn = item.get("attn", 0)
        if nid in positions:
            fig_data.append(go.Scatter(
                x=[positions[nid][0], positions[dst][0]],
                y=[positions[nid][1], positions[dst][1]],
                mode="lines",
                line=dict(color="#22c55e", width=max(1, int(attn * 6))),
                opacity=0.7,
                showlegend=False,
            ))

    for item in top_out[:5]:
        nid = f"out_{item['tx_id']}"
        attn = item.get("attn", 0)
        if nid in positions:
            fig_data.append(go.Scatter(
                x=[positions[src][0], positions[nid][0]],
                y=[positions[src][1], positions[nid][1]],
                mode="lines",
                line=dict(color="#a855f7", width=max(1, int(attn * 6))),
                opacity=0.7,
                showlegend=False,
            ))

    # Main edge
    fig_data.append(go.Scatter(
        x=[positions[src][0], positions[dst][0]],
        y=[positions[src][1], positions[dst][1]],
        mode="lines",
        line=dict(color="#ef4444", width=5),
        showlegend=False,
    ))

    # Nodes
    all_nodes = [{"id": src, "x": -300, "y": 0, "color": "#ef4444", "size": 30, "label": src[-10:]},
                 {"id": dst, "x": 300, "y": 0, "color": "#3b82f6", "size": 30, "label": dst[-10:]}]

    for item in top_in[:5]:
        nid = f"in_{item['tx_id']}"
        if nid in positions:
            attn = item.get("attn", 0)
            all_nodes.append({
                "id": nid, "x": positions[nid][0], "y": positions[nid][1],
                "color": "#22c55e" if attn > 0.3 else "#475569",
                "size": max(12, int(attn * 80)),
                "label": f"#{item['tx_id']}",
            })

    for item in top_out[:5]:
        nid = f"out_{item['tx_id']}"
        if nid in positions:
            attn = item.get("attn", 0)
            all_nodes.append({
                "id": nid, "x": positions[nid][0], "y": positions[nid][1],
                "color": "#a855f7" if attn > 0.3 else "#475569",
                "size": max(12, int(attn * 80)),
                "label": f"#{item['tx_id']}",
            })

    fig_data.append(go.Scatter(
        x=[n["x"] for n in all_nodes],
        y=[n["y"] for n in all_nodes],
        mode="markers+text",
        marker=dict(
            size=[n["size"] for n in all_nodes],
            color=[n["color"] for n in all_nodes],
            line=dict(color="#f8fafc", width=1),
        ),
        text=[n["label"] for n in all_nodes],
        textposition="top center",
        textfont=dict(color="#94a3b8", size=9),
        hovertext=[n["id"] for n in all_nodes],
        hoverinfo="text",
        showlegend=False,
    ))

    fig = go.Figure(data=fig_data)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig

# ═════════════════════════════════════════════════
# PATTERN CHARTS
# ═════════════════════════════════════════════════

def render_pattern_donut(pattern_dist):
    labels = list(pattern_dist.keys())
    values = list(pattern_dist.values())
    total = sum(values)
    colors = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b",
              "#a855f7", "#ec4899", "#14b8a6", "#f97316", "#6366f1"]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62,
        marker=dict(colors=colors[:len(labels)], line=dict(color="#0b0f19", width=2)),
        textinfo="percent", textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>"
    ))
    fig.add_annotation(text=f"<b>{total:,}</b><br>Transactions",
                       x=0.5, y=0.5, font=dict(size=22, color="#f8fafc"), showarrow=False)
    fig.update_layout(height=350, showlegend=True,
                      legend=dict(orientation="v", yanchor="middle", y=0.5, x=1.05,
                                  font=dict(color="#94a3b8", size=10)),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=20, r=120, t=20, b=20))
    return fig


def render_pattern_bars(pattern_dist):
    labels = list(pattern_dist.keys())[::-1]
    values = list(pattern_dist.values())[::-1]
    total = sum(values)
    colors = ["#6366f1", "#f97316", "#14b8a6", "#ec4899", "#a855f7",
              "#f59e0b", "#22c55e", "#ef4444", "#3b82f6"]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors[:len(labels)]),
        text=[f"{v:,} ({v/total*100:.1f}%)" for v in values],
        textposition="outside", textfont=dict(color="#94a3b8", size=10),
        hovertemplate="<b>%{y}</b><br>Count: %{x:,}<extra></extra>"
    ))
    fig.update_layout(height=350,
                      xaxis=dict(title="Number of Transactions", color="#64748b",
                                 gridcolor="#1e293b", zeroline=False),
                      yaxis=dict(color="#e2e8f0", gridcolor="#1e293b"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=150, r=100, t=10, b=40), font=dict(color="#94a3b8"))
    return fig


def badge_class(pattern):
    return f"badge-{pattern.lower().replace(' ', '-')}"

# ═════════════════════════════════════════════════
# SESSION STATE
# ═════════════════════════════════════════════════
if "selected_tx" not in st.session_state:
    st.session_state.selected_tx = None
if "report" not in st.session_state:
    st.session_state.report = None
if "page" not in st.session_state:
    st.session_state.page = 1

# ═════════════════════════════════════════════════
# CHECK BACKEND
# ═════════════════════════════════════════════════
stats = api_get("/api/stats")
if stats is None:
    st.error("⚠️ Backend not running on port 8001")
    st.code("python backend.py")
    st.stop()

# ═════════════════════════════════════════════════
# DETAIL VIEW
# ═════════════════════════════════════════════════
if st.session_state.selected_tx is not None:
    tx_id = st.session_state.selected_tx
    detail = api_get(f"/api/transaction/{tx_id}")
    if detail is None:
        st.error("Transaction not found")
        st.stop()

    y_proba = detail["prediction"]["y_proba"]
    pattern = detail["pattern"]
    risk_ctx = detail.get("risk_context", {})
    risk_level = risk_ctx.get("risk_level", "—")
    details = detail.get("details", {})
    signals = detail.get("signals", {})

    # ── Header ──
    col_back, _, _ = st.columns([1, 0.3, 4])
    with col_back:
        if st.button("←  Back"):
            st.session_state.selected_tx = None
            st.session_state.report = None
            st.rerun()

    risk_color = "#f87171" if y_proba > 0.9 else "#fbbf24"
    st.markdown(f"""
    <div class="detail-header">
        <div>
            <div class="detail-title">Transaction #{tx_id}</div>
            <div class="detail-sub">
                <span class="tx-badge {badge_class(pattern)}">{pattern}</span>
                &nbsp;·&nbsp; {risk_level} Risk
            </div>
        </div>
        <div style="text-align:right;">
            <div style="font-size:2.2rem; font-weight:800; color:{risk_color};">{y_proba:.1%}</div>
            <div class="kpi-label">Confidence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Info Grid ──
    st.markdown(f"""
    <div class="info-grid">
        <div class="info-card">
            <div class="info-label">Source Account</div>
            <div class="info-value" style="font-family:'JetBrains Mono'; font-size:0.8rem;">{details.get('src_account', 'N/A')}</div>
        </div>
        <div class="info-card">
            <div class="info-label">Destination Account</div>
            <div class="info-value" style="font-family:'JetBrains Mono'; font-size:0.8rem;">{details.get('dst_account', 'N/A')}</div>
        </div>
        <div class="info-card">
            <div class="info-label">Amount</div>
            <div class="info-value">~{signals.get('amount', 0):,.0f}</div>
        </div>
        <div class="info-card">
            <div class="info-label">Date & Time</div>
            <div class="info-value" style="font-size:0.8rem;">{details.get('timestamp', 'N/A')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Network Graph ──
    st.markdown('<div class="section-title">🕸️  Laundering Network</div>', unsafe_allow_html=True)
    attempt_count = len(detail.get("attempt", []))
    st.markdown(f"""
    <div class="graph-legend">
        <span class="legend-dot" style="background:#ef4444;"></span> Current Source
        &nbsp;<span class="legend-dot" style="background:#3b82f6;"></span> Current Destination
        &nbsp;<span class="legend-dot" style="background:#475569;"></span> Other Accounts
        &nbsp;<span class="legend-dot" style="background:#fbbf24;"></span> Current Transaction Edge
        &nbsp;&nbsp;|&nbsp;&nbsp; <b>{pattern}</b> layout · {attempt_count} transactions
    </div>
    """, unsafe_allow_html=True)

    fig = render_transaction_graph(detail)
    st.plotly_chart(fig, use_container_width=True)

    # ── Model Reasoning ──
    st.markdown('<div class="section-title">🧠  Model Reasoning</div>', unsafe_allow_html=True)
    reasoning = detail.get("reasoning", {})
    dominant = reasoning.get("dominant_component", "unknown")
    reason_map = {
        "dst": "The model focused on the **destination account's activity** — the flow of funds INTO this account was the key detection factor.",
        "edge": "The model focused on **the transaction's own characteristics** — amount, timing, and structural signals.",
        "src": "The model focused on the **source account's behavior** — its pattern of outgoing transactions.",
    }
    st.info(reason_map.get(dominant, "Not available."))

    col_in, col_out = st.columns(2)
    with col_in:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-label">📥 Incoming Transactions</div>
            <div class="info-value">{reasoning.get('n_competitors', 0):,}</div>
            <div class="info-label" style="margin-top:0.5rem;">at same destination</div>
        </div>
        """, unsafe_allow_html=True)
    with col_out:
        top_out = reasoning.get("top_outgoing", [])
        st.markdown(f"""
        <div class="info-card">
            <div class="info-label">📤 Key Related Outgoing</div>
            <div class="info-value">{len(top_out)}</div>
            <div class="info-label" style="margin-top:0.5rem;">from same source</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Report ──
    st.markdown('<div class="section-title">📄  SAR Investigation Report</div>', unsafe_allow_html=True)

    if st.session_state.report is None:
        if st.button("📄  Generate Report", type="primary"):
            with st.spinner("⏳ Generating SAR Report... (~15 seconds)"):
                result = api_post(f"/api/generate_report/{tx_id}")
            if "report" in result:
                st.session_state.report = result["report"]
                st.rerun()
            else:
                st.error(f"❌ {result.get('error', 'Failed')}")
    else:
        report = st.session_state.report
        st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.download_button("📥  Download PDF",
                data=report_to_pdf(report, tx_id, pattern),
                file_name=f"SAR_{tx_id}.pdf", mime="application/pdf")
        with col2:
            st.download_button("📄  Download Text",
                data=report, file_name=f"SAR_{tx_id}.md", mime="text/markdown")
        with col3:
            if st.button("🔄  Redo"):
                st.session_state.report = None
                st.rerun()

# ═════════════════════════════════════════════════
# MAIN PAGE
# ═════════════════════════════════════════════════
else:
    # ── Top Bar ──
    st.markdown("""
    <div class="topbar">
        <div class="topbar-title">🛡️ <span>AML</span> Intelligence</div>
        <div class="topbar-status">
            <span><span class="status-dot"></span> <span class="status-text">System Online</span></span>
            <span style="color:#475569; font-size:0.8rem;">FraudGT v2 · Llama-3.1-8B</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Cards ──
    n_patterns = len(stats["pattern_distribution"])
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card blue"><div class="kpi-value">{stats['total_flagged']:,}</div><div class="kpi-label">🚩 Flagged Transactions</div></div>
        <div class="kpi-card red"><div class="kpi-value">{stats['total_tp']:,}</div><div class="kpi-label">⚠️ Confirmed Laundering</div></div>
        <div class="kpi-card green"><div class="kpi-value">{stats['total_fp']:,}</div><div class="kpi-label">🔍 Under Review</div></div>
        <div class="kpi-card purple"><div class="kpi-value">{n_patterns}</div><div class="kpi-label">📊 Pattern Types</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### ⚙️  Filters")
        search = st.text_input("", placeholder="🔍  Search ID or Account...")
        patterns = ["All"] + sorted(stats["pattern_distribution"].keys())
        selected_pattern = st.selectbox("Pattern", patterns)
        min_risk, max_risk = st.slider("Risk Range", min_value=0.0, max_value=1.0,
                                        value=(0.80, 1.0), step=0.01)
        st.markdown("---")
        st.markdown("### 📊  Statistics")
        st.metric("Avg Confidence", f"{stats['avg_risk']:.1%}")

    # ── Pattern Charts ──
    st.markdown('<div class="section-title">📊  Pattern Analysis</div>', unsafe_allow_html=True)
    col_donut, col_bars = st.columns(2)
    with col_donut:
        st.plotly_chart(render_pattern_donut(stats["pattern_distribution"]), use_container_width=True)
    with col_bars:
        st.plotly_chart(render_pattern_bars(stats["pattern_distribution"]), use_container_width=True)

    # ── Transaction List ──
    st.markdown('<div class="section-title">📋  All Flagged Transactions</div>', unsafe_allow_html=True)

    params = {"page": st.session_state.page, "per_page": 50,
              "min_risk": min_risk, "max_risk": max_risk}
    if selected_pattern != "All":
        params["pattern"] = selected_pattern
    if search:
        params["search"] = search

    result = api_get("/api/transactions", params=params)
    if result is None:
        st.error("Failed to fetch")
        st.stop()

    transactions = result["transactions"]
    total = result["total"]
    total_pages = result["total_pages"]

    if not transactions:
        st.warning("No transactions match your filters")
    else:
        for idx, tx in enumerate(transactions, 1):
            global_idx = (st.session_state.page - 1) * 50 + idx
            tx_id = tx["transaction_id"]
            risk = tx["y_proba"]
            risk_class = "risk-high" if risk > 0.9 else "risk-medium"
            gt_class = "gt-flag" if tx["y_true"] == 1 else "gt-ok"
            gt_text = "⚠️ Fraud" if tx["y_true"] == 1 else "✓ Clean"
            date_str = tx.get("timestamp", "")[:16]

            st.markdown(f"""
            <div class="tx-row">
                <div class="tx-num">{global_idx}</div>
                <div class="tx-id">#{tx_id}</div>
                <div class="tx-badge {badge_class(tx['pattern'])}">{tx['pattern']}</div>
                <div class="tx-risk {risk_class}">{risk:.1%}</div>
                <div class="tx-amount">~{tx['amount']:,.0f}</div>
                <div class="tx-date">{date_str}</div>
                <div class="tx-gt {gt_class}">{gt_text}</div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Investigate →", key=f"btn_{tx_id}"):
                st.session_state.selected_tx = tx_id
                st.rerun()

        st.markdown(f"""
        <div class="pagination">
            <span class="page-info">Page {st.session_state.page} of {total_pages} · Showing {len(transactions)} of {total:,}</span>
        </div>
        """, unsafe_allow_html=True)

        col_prev, _, col_next = st.columns([1, 3, 1])
        with col_prev:
            if st.session_state.page > 1:
                if st.button("←  Previous"):
                    st.session_state.page -= 1
                    st.rerun()
        with col_next:
            if st.session_state.page < total_pages:
                if st.button("Next  →"):
                    st.session_state.page += 1
                    st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#334155; font-size:0.75rem; padding:1rem;'>"
        "AML Intelligence System · Master's Thesis · FraudGT + Llama-3.1-8B + vLLM</div>",
        unsafe_allow_html=True)
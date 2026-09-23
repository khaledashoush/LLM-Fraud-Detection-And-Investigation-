# -*- coding: utf-8 -*-
"""
frontend.py — AML Intelligence Dashboard (Professional v3)
"""

import os, io, json, time
import requests
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from fpdf import FPDF
from datetime import datetime

BACKEND_URL = "http://localhost:8001"

# ═════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════
st.set_page_config(
    page_title="AML Intelligence | Compliance Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═════════════════════════════════════════════════
# PROFESSIONAL CSS
# ═════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

* { font-family: 'Inter', sans-serif !important; }

.stApp {
    background: #0b0f19;
    color: #e2e8f0;
}

/* ── Hide Streamlit defaults ── */
#MainMenu, footer, header { display: none !important; }
.stDeployButton { display: none !important; }

.main .block-container {
    padding: 0 2rem 2rem 2rem;
    max-width: 1400px;
    margin-top: 0;
}

/* ── Top Bar ── */
.topbar {
    background: linear-gradient(180deg, #111827 0%, #0b0f19 100%);
    border-bottom: 1px solid #1e293b;
    padding: 1rem 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 0 -2rem 1.5rem -2rem;
}
.topbar-title {
    font-size: 1.3rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.5px;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.topbar-title span { color: #3b82f6; }
.topbar-status {
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.status-dot {
    width: 8px; height: 8px;
    background: #22c55e;
    border-radius: 50%;
    display: inline-block;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(34,197,94,0.4); }
    70% { box-shadow: 0 0 0 6px rgba(34,197,94,0); }
    100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); }
}
.status-text {
    color: #22c55e;
    font-size: 0.8rem;
    font-weight: 600;
}

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.kpi-card {
    background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 3px 3px 0 0;
}
.kpi-card.blue::before { background: #3b82f6; }
.kpi-card.red::before { background: #ef4444; }
.kpi-card.green::before { background: #22c55e; }
.kpi-card.purple::before { background: #a855f7; }

.kpi-value {
    font-size: 2rem;
    font-weight: 800;
    color: #f8fafc;
    line-height: 1;
    margin-bottom: 0.4rem;
}
.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* ── Section Title ── */
.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 1.5rem 0 1rem 0;
    padding-left: 0.8rem;
    border-left: 3px solid #3b82f6;
}

/* ── Transaction Cards ── */
.tx-list {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
}
.tx-row {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: all 0.15s ease;
}
.tx-row:hover {
    border-color: #3b82f6;
    background: #253349;
    transform: translateX(4px);
}
.tx-num {
    color: #475569;
    font-size: 0.8rem;
    font-weight: 600;
    min-width: 2.5rem;
    text-align: right;
    font-family: 'JetBrains Mono', monospace;
}
.tx-id {
    font-family: 'JetBrains Mono', monospace;
    color: #60a5fa;
    font-size: 0.9rem;
    font-weight: 600;
    min-width: 7rem;
}
.tx-badge {
    padding: 0.25rem 0.7rem;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    min-width: 6.5rem;
    text-align: center;
}
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

.tx-risk {
    font-weight: 800;
    font-size: 0.95rem;
    min-width: 3.5rem;
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
}
.risk-high { color: #f87171; }
.risk-medium { color: #fbbf24; }

.tx-amount {
    color: #94a3b8;
    font-size: 0.85rem;
    min-width: 6rem;
    font-family: 'JetBrains Mono', monospace;
}
.tx-date {
    color: #64748b;
    font-size: 0.8rem;
    min-width: 7rem;
}
.tx-gt {
    font-size: 0.75rem;
    min-width: 4rem;
    text-align: center;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-weight: 600;
}
.gt-flag { background: #450a0a; color: #fca5a5; }
.gt-ok { background: #052e16; color: #86efac; }

/* ── Detail View ── */
.detail-header {
    background: linear-gradient(145deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.detail-title {
    font-size: 1.5rem;
    font-weight: 800;
    color: #f8fafc;
    font-family: 'JetBrains Mono', monospace;
}
.detail-sub {
    color: #64748b;
    font-size: 0.85rem;
    margin-top: 0.3rem;
}

.info-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.8rem;
    margin-bottom: 1.2rem;
}
.info-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.info-label {
    color: #64748b;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.5rem;
}
.info-value {
    color: #f8fafc;
    font-size: 1.1rem;
    font-weight: 700;
}

/* ── Report ── */
.report-box {
    background: #ffffff;
    color: #1e293b;
    border-radius: 12px;
    padding: 2rem;
    line-height: 1.8;
    font-size: 0.95rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
}
.report-box h1, .report-box h2, .report-box h3 {
    color: #0f172a !important;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 0.5rem;
}

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: #3b82f6 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    background: #2563eb !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(59,130,246,0.4) !important;
}

/* ── Charts ── */
.js-plotly-plot { border-radius: 10px; }

/* ── Pagination ── */
.pagination {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 1rem;
    margin-top: 1.2rem;
    padding-top: 1rem;
    border-top: 1px solid #1e293b;
}
.page-info {
    color: #64748b;
    font-size: 0.85rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #f8fafc;
    font-size: 0.9rem;
    font-weight: 700;
}

/* ── Data Table ── */
[data-testid="stDataFrame"] {
    border: 1px solid #334155;
    border-radius: 10px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════
# HELPERS
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

def badge_class(pattern):
    p = pattern.lower().replace(" ", "-")
    return f"badge-{p}"

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
    st.error("Backend not running on port 8001")
    st.code("python backend.py")
    st.stop()

# ═════════════════════════════════════════════════
# DETAIL VIEW
# ═════════════════════════════════════════════════
if st.session_state.selected_tx is not None:
    tx_id = st.session_state.selected_tx
    detail = api_get(f"/api/transaction/{tx_id}")
    if detail is None:
        st.error("Not found")
        st.stop()

    y_proba = detail["prediction"]["y_proba"]
    pattern = detail["pattern"]
    risk_ctx = detail.get("risk_context", {})
    risk_level = risk_ctx.get("risk_level", "—")
    details = detail.get("details", {})
    signals = detail.get("signals", {})

    # ── Header ──
    col_back, _, col_header = st.columns([1, 0.3, 4])
    with col_back:
        if st.button("←  Back"):
            st.session_state.selected_tx = None
            st.session_state.report = None
            st.rerun()

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
            <div style="font-size:2.2rem; font-weight:800; color:{'#f87171' if y_proba > 0.9 else '#fbbf24'};">
                {y_proba:.1%}
            </div>
            <div class="kpi-label">Confidence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Info Grid ──
    st.markdown(f"""
    <div class="info-grid">
        <div class="info-card">
            <div class="info-label">Source Account</div>
            <div class="info-value" style="font-family:'JetBrains Mono'; font-size:0.85rem;">
                {details.get('src_account', 'N/A')}
            </div>
        </div>
        <div class="info-card">
            <div class="info-label">Destination Account</div>
            <div class="info-value" style="font-family:'JetBrains Mono'; font-size:0.85rem;">
                {details.get('dst_account', 'N/A')}
            </div>
        </div>
        <div class="info-card">
            <div class="info-label">Amount</div>
            <div class="info-value">~{signals.get('amount', 0):,.0f}</div>
        </div>
        <div class="info-card">
            <div class="info-label">Date & Time</div>
            <div class="info-value" style="font-size:0.85rem;">{details.get('timestamp', 'N/A')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Model Reasoning ──
    st.markdown('<div class="section-title">🧠 Model Reasoning</div>', unsafe_allow_html=True)

    reasoning = detail.get("reasoning", {})
    dominant = reasoning.get("dominant_component", "unknown")
    reason_map = {
        "dst": "The model focused on the **destination account's activity** — the flow of funds INTO this account was the key detection factor.",
        "edge": "The model focused on **the transaction's own characteristics** — amount, timing, and structural signals.",
        "src": "The model focused on the **source account's behavior** — its pattern of outgoing transactions.",
    }
    st.info(reason_map.get(dominant, "Not available"))

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
    st.markdown('<div class="section-title">📄 SAR Investigation Report</div>', unsafe_allow_html=True)

    if st.session_state.report is None:
        if st.button("📄  Generate Report", type="primary", use_container_width=True):
            with st.spinner("⏳ Generating SAR Report..."):
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
            st.download_button(
                "📥  Download PDF",
                data=report_to_pdf(report, tx_id, pattern),
                file_name=f"SAR_{tx_id}.pdf",
                mime="application/pdf",
            )
        with col2:
            st.download_button(
                "📄  Download Text",
                data=report,
                file_name=f"SAR_{tx_id}.md",
                mime="text/markdown",
            )
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
        <div class="topbar-title">
            🛡️ <span>AML</span> Intelligence
        </div>
        <div class="topbar-status">
            <span><span class="status-dot"></span> <span class="status-text">System Online</span></span>
            <span style="color:#475569; font-size:0.8rem;">FraudGT v2 · Llama-3.1-8B</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Cards ──
    total = stats["total_flagged"]
    high_risk = sum(1 for v in [1] if v) # placeholder
    n_patterns = len(stats["pattern_distribution"])

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card blue">
            <div class="kpi-value">{stats['total_flagged']:,}</div>
            <div class="kpi-label">🚩 Flagged Transactions</div>
        </div>
        <div class="kpi-card red">
            <div class="kpi-value">{stats['total_tp']:,}</div>
            <div class="kpi-label">⚠️ Confirmed Laundering</div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-value">{stats['total_fp']:,}</div>
            <div class="kpi-label">🔍 Under Review</div>
        </div>
        <div class="kpi-card purple">
            <div class="kpi-value">{n_patterns}</div>
            <div class="kpi-label">📊 Pattern Types</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### ⚙️  Filters")

        search = st.text_input("", placeholder="🔍  Search ID or Account...")

        patterns = ["All"] + sorted(stats["pattern_distribution"].keys())
        selected_pattern = st.selectbox("Pattern", patterns)

        min_risk, max_risk = st.slider(
            "Risk Range",
            min_value=0.0, max_value=1.0,
            value=(0.80, 1.0), step=0.01
        )

        st.markdown("---")
        st.markdown("### 📊  Statistics")
        st.metric("Avg Confidence", f"{stats['avg_risk']:.1%}")

    # ── Transaction List ──
    st.markdown('<div class="section-title">📋  All Flagged Transactions</div>', unsafe_allow_html=True)

    params = {
        "page": st.session_state.page,
        "per_page": 50,
        "min_risk": min_risk,
        "max_risk": max_risk,
    }
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
        # ── Transaction Cards ──
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

            # Click to select
            if st.button(f"Investigate →", key=f"btn_{tx_id}"):
                st.session_state.selected_tx = tx_id
                st.rerun()

        # ── Pagination ──
        st.markdown(f"""
        <div class="pagination">
            <span class="page-info">
                Page {st.session_state.page} of {total_pages} · 
                Showing {len(transactions)} of {total:,}
            </span>
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

    # ── Footer ──
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#334155; font-size:0.75rem; padding:1rem;'>"
        "AML Intelligence System · Master's Thesis · FraudGT + Llama-3.1-8B + vLLM"
        "</div>",
        unsafe_allow_html=True
    )
# 🔍 AML Intelligence System

**End-to-End Anti-Money Laundering: Detection → Explainability → Investigation Reports → Dashboard**

*Master's Thesis Project | Khaled Ashoush*

![Python](https://img.shields.io/badge/python-3.11-blue)
![Status](https://img.shields.io/badge/status-active%20development-brightgreen)
![License](https://img.shields.io/badge/license-CDLA--Sharing--1.0-lightgrey)
![F1](https://img.shields.io/badge/F1%40thr-0.7291-success)
![Reports](https://img.shields.io/badge/LLM%20reports-ollama)

---

![AML Dashboard](UI_Images/Screenshot%20From%202026-09-24%2013-44-03.png)
![AML Dashboard](UI_Images/Screenshot%20From%202026-09-24%2013-44-20.png)
---

## 📖 Table of Contents

- [Overview](#-overview)
- [System Architecture](#️-system-architecture)
- [Results Summary](#-results-summary)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#️-installation--setup)
- [Usage Guide](#-usage-guide)
- [Methodology](#-methodology)
- [Key Research Findings](#-key-research-findings)
- [Evaluation Framework](#-evaluation-framework)
- [AML Pattern Reference](#-aml-pattern-reference)
- [Roadmap](#️-roadmap)
- [References](#-references)
- [License](#-license)
- [Author](#-author)

---

## 🎯 Overview

This project builds an end-to-end **AML (Anti-Money Laundering) system** that goes beyond simple fraud detection. It combines a state-of-the-art **Graph Neural Network (FraudGT)** with a novel **attention-based explainer** and a **locally-hosted LLM** to generate SAR-compliant investigation reports — all served through an **interactive dashboard** for compliance investigators.

**The core philosophy:**

> The LLM is a narrative layer, **NOT** a detector — it converts model explanations into investigator-ready reports.

**What makes this project unique:**

| Component | Innovation | Paper Basis |
|---|---|---|
| **Explainer** | First attention-based explainer for FraudGT with causal verification (occlusion) | Jain & Wallace 2019 + InteractiveGNNExplainer 2025 |
| **Analysis** | Discovery of a three-tier causal hierarchy across laundering patterns | xFraud PVLDB'22 + IBM AMLworld NeurIPS'23 |
| **LLM Reports** | SAR-compliant narratives via self-hosted Llama-3.1-8B (LLM as Narrator) | GraphXAIN 2025 + LLMOps 2026 |
| **Evaluation** | 5-layer evaluation framework (deterministic + LLM-as-Judge) | LLMOps Sec 4 + GraphXAIN Sec 6 |
| **Dashboard** | Pattern-specific network graphs showing full laundering attempts | InteractiveGNNExplainer 2025 |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          AML INTELLIGENCE PIPELINE                       │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  PHASE 1: DATA                          PHASE 2: DETECTION               │
│  ┌───────────────────────┐              ┌───────────────────────┐       │
│  │  IBM AMLworld            │   287       │  FraudGT                │       │
│  │  HI-Medium                │─features──▶│  (Graph Transformer)   │       │
│  │  • 31.9M transactions     │             │                        │       │
│  │  • 2.08M accounts         │             │  F1@thr = 0.7291       │       │
│  │  • 0.11% illicit          │             └───────────┬────────────┘       │
│  └───────────────────────┘                           │                    │
│                                     flagged: 8,957 transactions           │
│                                                        │                    │
│  PHASE 3: EXPLAINABILITY                              ▼                    │
│  ┌───────────────────────┐              ┌───────────────────────┐       │
│  │  Per-Pattern Analysis    │◀─200-sample─│  Attention-based        │       │
│  │  (causal hierarchy)      │  occlusion  │  Explainer               │       │
│  │                          │             │  + Occlusion             │       │
│  │                          │             │  + Head Component Test  │       │
│  └───────────────────────┘              └───────────┬────────────┘       │
│                                                        │                    │
│  PHASE 4: LLM REPORTS                                 ▼                    │
│  ┌───────────────────────┐              ┌───────────────────────┐       │
│  │  SAR-Compliant           │◀─structured─│  Llama-3.1-8B            │       │
│  │  Investigation Reports   │  evidence   │  via vLLM (self-hosted)  │       │
│  └───────────────────────┘              └───────────┬────────────┘       │
│                                                        │                    │
│  PHASE 5: EVALUATION                                  ▼                    │
│  ┌───────────────────────┐              ┌───────────────────────┐       │
│  │  Overall: 8.8/10          │◀────────────│  5-Layer Framework       │       │
│  │  Faithfulness: 95–97%     │             │  Deterministic +         │       │
│  │                          │             │  LLM-as-Judge             │       │
│  └───────────────────────┘              └───────────────────────┘       │
│                                                                            │
│  PHASE 6: DASHBOARD                                                       │
│  ┌───────────────────────────────────────────────────────────────┐      │
│  │  FastAPI Backend (port 8001)                                    │      │
│  │  Streamlit Dashboard (port 8501)                                │      │
│  │  ├── All 8,957 transactions                                     │      │
│  │  ├── Pattern-specific network graphs                            │      │
│  │  ├── On-demand SAR report generation                            │      │
│  │  └── PDF / Markdown download                                    │      │
│  └───────────────────────────────────────────────────────────────┘      │
│                                                                            │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Results Summary

### Detection Performance (IBM AMLworld HI-Medium)

| Model | Features | Test F1@thr | PR-AUC | Status |
|---|---|---|---|---|
| **FraudGT** (main) | 15 | **0.7291** | 0.7258 | Active development |
| Multi-PNA | 15 | 0.6713 | 0.6666 | Above published (66.48) ✓ |
| PNA+EU | 10 | 0.6498 | 0.6535 | Above published (59.71) ✓ |
| XGBoost+GFP (baseline) | 274 | 0.6448 | 0.6436 | Matches published ✓ |

### Explainer Results (8,957 flagged transactions)

| Metric | Result | Method |
|---|---|---|
| Causal verification | Top-attn Δ=0.020, Random Δ=0.000 | 200-sample occlusion |
| Attention causality | Confirmed (ratio > 1000×) | Perturbation test |
| Faithful predictions | 100% match | Logit verification |
| Pattern coverage | All 8 patterns + LONE + NONE | Per-pattern analysis |

### LLM Report Quality (100 reports evaluated, v4.1)

| Evaluation Layer | Metric | Score |
|---|---|---|
| L1: Deterministic | Schema validity | **100%** |
| L1: Deterministic | No hallucinations | **100%** |
| L2: Faithfulness | Overall | **95%** |
| L2: Faithfulness | Fact accuracy | **100%** |
| L2: Faithfulness | Evidence coverage | **89%** |
| L3: Compliance | SAR format | **8.0/10** |
| L3: Compliance | Professional tone | **4.0/5** |
| L4: Narrative | Understandability | **4.4/5** |
| L4: Narrative | Insightfulness | **3.6/5** |
| L4: Narrative | Convincingness | **4.4/5** |
| L4: Narrative | Communicability | **4.7/5** |
| **Overall** | **Combined score** | **8.8/10** |

### Pattern Detection Breakdown

| Pattern | Flagged | Avg Risk | Dominant Head Component | Narrative Quality |
|---|---|---|---|---|
| GATHER-SCATTER | 1,985 | 0.96 | edge (0.145) | 4.3/5 |
| SCATTER-GATHER | 1,262 | 0.94 | **dst** (0.223) | 4.3/5 |
| STACK | 1,144 | 0.95 | edge (0.295) | 4.2/5 |
| FAN-IN | 850 | 0.94 | **dst** (0.237) | 3.9/5 |
| CYCLE | 652 | 0.94 | edge (0.218) | 4.1/5 |
| FAN-OUT | 535 | 0.94 | edge (0.224) | 4.3/5 |
| RANDOM | 460 | 0.97 | edge (0.280) | 4.4/5 |
| BIPARTITE | 367 | 0.92 | edge (0.276) | 4.2/5 |
| LONE | 26 | 1.00 | edge (0.297) | — |
| NONE (false positive) | 1,676 | — | **dst** (0.220) | 4.0/5 |

---

## 📁 Repository Structure

```
AML-Fraud-Detection-And-Investigation/
│
├── Core Pipeline
│   ├── Preprocessing.py        # 287 features + splits + ports/tds + patterns
│   ├── train_fraudgt.py        # FraudGT + attention capture (additive)
│   ├── train_multi_pna.py      # Multi-PNA (IBM official)
│   ├── train_pna_eu.py         # PNA + Edge Updates
│   ├── train_xgb.py            # XGBoost baseline (Optuna)
│   └── join_patterns.py        # Pattern ground-truth linking
│
├── Explainability Layer
│   ├── explain_fraudgt.py      # Attention + occlusion + head analysis
│   ├── analyze_explainer.py    # Per-pattern analysis + causal hierarchy
│   └── test_modification.py    # 7 verification tests (attention capture)
│
├── LLM Report Layer
│   ├── generate_reports.py     # SAR reports via Llama-3.1-8B (v4.1)
│   └── evaluate_reports.py     # 5-layer evaluation framework
│
├── Backend & Frontend
│   ├── backend.py               # FastAPI server (transactions, reports, graphs)
│   └── frontend.py              # Streamlit dashboard (professional dark theme)
│
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| GPU | 8 GB VRAM | 24 GB (RTX A5000) |
| RAM | 32 GB | 1 TB |
| Storage | 100 GB | 500+ GB |
| Python | 3.11 | 3.11 |
| CUDA | 12.1 | 12.1+ |

### Step 1: Environment Setup

```bash
# Clone
git clone https://github.com/khaledashoush/LLM-Fraud-Detection-And-Investigation-.git
cd LLM-Fraud-Detection-And-Investigation-

# Environment for training/GNN + backend/frontend
conda create -n aml-research python=3.11 -y
conda activate aml-research
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install torch-geometric snapml xgboost optuna scikit-learn pandas numpy pyarrow
pip install fastapi uvicorn streamlit plotly fpdf2

# Environment for LLM serving
conda create -n llm-serve python=3.11 -y
conda activate llm-serve
pip install vllm
```

### Step 2: Download Data

```bash
# Download IBM AMLworld from Kaggle:
# https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
# Select: HI-Medium (31.9M transactions)

mkdir -p data
# Copy: HI-Medium_Trans.csv, HI-Medium_accounts.csv, HI-Medium_Patterns.txt
```

### Step 3: Run the Full Pipeline

```bash
# Terminal 1 — vLLM server
conda activate llm-serve
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --gpu-memory-utilization 0.9 --max-model-len 8192 \
    --enable-prefix-caching --port 8000

# Terminal 2 — Backend
conda activate aml-research
python Preprocessing.py                  # ~3 hours (first time)
python train_fraudgt.py                  # ~10 hours (or load checkpoint)
python explain_fraudgt.py --occlusion-sample 200
python backend.py                        # Backend on port 8001

# Terminal 3 — Frontend
conda activate aml-research
streamlit run frontend.py                # Dashboard on port 8501
```

### Step 4: Access the Dashboard

Open your browser at **http://localhost:8501**

---

## 📖 Usage Guide

### For Investigators (Dashboard)

| Action | How |
|---|---|
| Browse all flagged transactions | Main page with pagination |
| Search | Search box (by ID or account) |
| Filter | Pattern dropdown + risk slider |
| View transaction details | Click **"Investigate →"** |
| View network graph | Pattern-specific layout on the detail page |
| Generate SAR report | Click **"Generate Report"** (~15 seconds) |
| Download report | **"Download PDF"** or **"Download Text"** |

### For Developers (API)

```bash
# Statistics
curl http://localhost:8001/api/stats

# List transactions (page 1, 50 per page, FAN-OUT only, risk > 0.9)
curl http://localhost:8001/api/transactions?page=1&per_page=50&pattern=FAN-OUT&min_risk=0.9

# Get transaction details + network
curl http://localhost:8001/api/transaction/30816087

# Generate report (POST)
curl -X POST http://localhost:8001/api/generate_report/30816087
```

---

## 🔬 Methodology

### Detection: FraudGT (Faithful Port)

Architecture faithfully implements **[FraudGT, ICAIF'24]**:

- `SparseNodeTransformer` with per-type Q/K/V/O projections
- Edge attention bias (`e_lin`) and message gate (`g_lin`) — Eqs. 5–8
- Eq. 9 prediction head: `y = σ(MLP(h_i ∥ e'_ij ∥ h_j))`
- Official training regime: 256 iters/epoch, warmup cosine, AdamW param groups
- Enhancements: RMP + Port Numbering + Ego IDs [Egressy et al., AAAI'24]

### Explainer: Multi-Level Attention-Based

| Level | Method | Paper |
|---|---|---|
| Signal | FraudGT's native attention | [FraudGT ICAIF'24] |
| Verification | Neighbor occlusion (causal check) | [Jain & Wallace 2019] |
| Head analysis | Component occlusion (src / edge / dst) | [FraudGT Eq. 9] |
| Structural | GFP features + patterns | [xFraud PVLDB'22] |

### LLM Reports: Narrator Architecture

> The LLM does **NOT** predict patterns — it explains them.

| Component | Source |
|---|---|
| Narrative rules | [GraphXAIN, Appendix A] |
| Report format | [FinCEN SAR guidelines] |
| Pattern definitions | [IBM AMLworld, Sec 3.2] |
| Serving stack | [LLMOps] vLLM + Llama-3.1-8B |

### Network Visualization

Each pattern is rendered with a layout chosen to make its fund-flow structure immediately legible to an investigator (full mapping in [AML Pattern Reference](#-aml-pattern-reference)). The **golden edge** highlights the current transaction within the full network graph.

---

## 🏆 Key Research Findings

### Finding 1: Three-Tier Causal Hierarchy (NOVEL)

Occlusion analysis across 200 flagged transactions reveals:

```
TIER 1 — HIGHLY CAUSAL (Δ > 0.02)
  STACK            Δ = +0.074   (each layer critical)
  CYCLE            Δ = +0.022   (breaking loop breaks pattern)
  RANDOM           Δ = +0.021   (specific next-hop matters)

TIER 2 — MODERATELY CAUSAL (0.003 < Δ < 0.015)
  SCATTER-GATHER   Δ = +0.015   (intermediaries matter but survivable)
  GATHER-SCATTER   Δ = +0.003   (nearly independent)

TIER 3 — AGGREGATE-BASED (Δ ≈ 0)
  FAN-IN           Δ = +0.000   (count matters, not specific neighbor)
  FAN-OUT          Δ = -0.001   (removing one destination doesn't matter)
  BIPARTITE        Δ = -0.002   (set structure, not individual edges)
```

This hierarchy aligns precisely with structural definitions **[IBM Sec 3.2]** and has not been previously reported.

### Finding 2: FraudGT Adapts Reasoning Per Pattern (NOVEL)

Head-component occlusion reveals distinct strategies:

| Pattern Type | Dominant Component | Interpretation |
|---|---|---|
| SCATTER-GATHER | dst (+0.223) | Model focuses on destination (gathering point) |
| FAN-IN | dst (+0.238) | Model focuses on destination (convergence) |
| FAN-OUT | edge (+0.224) | Model focuses on transaction characteristics |
| STACK | edge (+0.296) | Model focuses on structural signals |

### Finding 3: LONE Detection Gap

**LONE** (isolated illicit) transactions represent **35.4%** of all illicit activity in the dataset, yet FraudGT detects only **0.4%** of them (26 flagged in the current run). This highlights a fundamental limitation of message-passing architectures for isolated suspicious transactions.

### Finding 4: False Positives are Context-Dependent

False positives show the highest occlusion sensitivity (Δ=+0.035), indicating they were flagged due to specific related transactions. This enables targeted investigation of model errors.

---

## 📐 Evaluation Framework

| Layer | What it Measures | Method | Paper |
|---|---|---|---|
| L1: Deterministic | Schema validity, fields, tone | Automated checks | [LLMOps Sec 4] |
| L2: Faithfulness | Coverage, accuracy, fabrication | Evidence matching | [GNNExplainer + xFraud] |
| L3: Compliance | SAR format, tone, actionability | LLM-as-Judge | [LLMOps Sec 4.1] |
| L4: Narrative | 5 GraphXAIN dimensions | LLM-as-Judge | [GraphXAIN Sec 6] |
| L5: Overall | Weighted composite + per-pattern | Aggregation | [LLMOps Pareto] |

---

## 🗂️ AML Pattern Reference

| Pattern | Graph Layout | Meaning | Flagged | Avg Risk |
|---|---|---|---|---|
| CYCLE | Ring | Funds loop back to origin | 652 | 0.94 |
| FAN-OUT | Star (outward) | Source disperses to many | 535 | 0.94 |
| FAN-IN | Star (inward) | Many converge to destination | 850 | 0.94 |
| SCATTER-GATHER | Butterfly | Disperse then reconverge | 1,262 | 0.94 |
| GATHER-SCATTER | Hourglass | Converge then disperse | 1,985 | 0.96 |
| BIPARTITE | Two columns | Two account sets | 367 | 0.92 |
| STACK | Parallel chains | Multiple layers | 1,144 | 0.95 |
| RANDOM | Wavy chain | Random walk | 460 | 0.97 |
| LONE | — | Isolated illicit transaction | 26 | 1.00 |
| NONE | — | Flagged, no ground-truth pattern (false positive) | 1,676 | — |

*Dominant-attention component and narrative quality per pattern are in the [Pattern Detection Breakdown](#-results-summary) table above.*

---

## 🗺️ Roadmap

**Completed ✅**

- [x] Data pipeline (287 features, 8 patterns, 100% coverage)
- [x] FraudGT detection (F1 = 0.7291)
- [x] Attention-based explainer (8,957 explanations, 200 occlusion tests)
- [x] Per-pattern analysis (causal hierarchy, attention signatures)
- [x] LLM investigation reports (SAR format, v4.1, 100 reports evaluated)
- [x] 5-layer evaluation framework (overall 8.8/10)
- [x] Backend: FastAPI (`/api/stats`, `/api/transactions`, `/api/transaction/{id}`, `/api/generate_report/{id}`)
- [x] Frontend: Streamlit investigator dashboard (search, filter, network graphs, PDF/Markdown export)


**Planned 📋**

- [ ] Human evaluation study (GraphXAIN methodology)

---

## 📚 References

1. Altman et al., *"Realistic Synthetic Financial Transactions for AML Models,"* NeurIPS'23 D&B
2. Lin et al., *"FraudGT: A Simple, Effective, and Efficient Graph Transformer for Financial Fraud Detection,"* ICAIF'24
3. Egressy et al., *"Provably Powerful Graph Neural Networks for Directed Multigraphs,"* AAAI'24
4. Blanusa et al., *"Graph Feature Preprocessor,"* ICAIF'24
5. Ying et al., *"GNNExplainer: Generating Explanations for GNNs,"* NeurIPS'19
6. Yuan et al., *"SubgraphX: On Explainability via Subgraph Explorations,"* ICML'21
7. Rao et al., *"xFraud: Explainable Fraud Transaction Detection,"* PVLDB'22
8. Cedro & Martens, *"GraphXAIN: Narratives to Explain GNNs,"* 2025
9. Pirmorad, *"Exploring In-Context Learning for Money Laundering Detection,"* 2025
10. Naik et al., *"Rethinking LLMOps for Fraud and AML,"* 2026
11. Singh & Mukherjea, *"InteractiveGNNExplainer,"* 2025
12. Han et al., *"GraphRAG Survey,"* 2025
13. Jain & Wallace, *"Attention is not Explanation,"* 2019
14. Lu & Wang, *"Graph Contrastive Pre-training for AML,"* IJCIS 2024

---

## 📄 License

This project is for academic purposes (Master's thesis). The IBM AMLworld dataset is under the **CDLA-Sharing-1.0** license.

---

## 👤 Author

**Khaled Ashoush**
Master's Thesis — AML Fraud Detection & Investigation

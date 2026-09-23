# 🔍 AML Intelligence System

**End-to-End Anti-Money Laundering: Detection → Explainability → Investigation Reports**

*Master's Thesis Project | Khaled Ashoush*

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
- [Roadmap](#️-roadmap)
- [References](#-references)
- [License](#-license)
- [Author](#-author)

---

## 🎯 Overview

This project builds an end-to-end **AML (Anti-Money Laundering)** system that goes beyond simple fraud detection. It combines a state-of-the-art **Graph Neural Network (FraudGT)** with a novel **attention-based explainer** and a **locally-hosted LLM** to generate SAR-compliant investigation reports for compliance investigators.

**The core philosophy:**

> The LLM is a narrative layer, **NOT** a detector — it converts model explanations into investigator-ready reports.

**What makes this project unique:**

| Component | Innovation | Paper Basis |
|---|---|---|
| **Explainer** | First attention-based explainer for FraudGT with causal verification (occlusion) | Jain & Wallace 2019 + InteractiveGNNExplainer 2025 |
| **Analysis** | Discovery of a three-tier causal hierarchy across laundering patterns | xFraud PVLDB'22 + IBM AMLworld NeurIPS'23 |
| **LLM Reports** | SAR-compliant narratives via self-hosted Llama-3.1-8B (LLM as Narrator) | GraphXAIN 2025 + LLMOps 2026 |
| **Evaluation** | 5-layer evaluation framework (deterministic + LLM-as-Judge) | LLMOps Sec 4 + GraphXAIN Sec 6 |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          AML INTELLIGENCE PIPELINE                       │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  PHASE 1: DATA                          PHASE 2: DETECTION               │
│  ┌───────────────────────┐              ┌───────────────────────┐       │
│  │  IBM AMLworld          │   287        │  FraudGT               │       │
│  │  HI-Medium              │──features──▶│  (Graph Transformer)   │       │
│  │  • 31.9M transactions   │              │                        │       │
│  │  • 2.08M accounts       │              │  F1@thr = 0.7291       │       │
│  │  • 0.11% illicit        │              └───────────┬────────────┘       │
│  └───────────────────────┘                          │                    │
│                                     flagged: 8,957 transactions           │
│                                                       │                    │
│  PHASE 3: EXPLAINABILITY                             ▼                    │
│  ┌───────────────────────┐              ┌───────────────────────┐       │
│  │  Per-Pattern Analysis   │◀──200-sample│  Attention-based       │       │
│  │  (causal hierarchy)     │  occlusion  │  Explainer              │       │
│  │                         │              │  + Occlusion            │       │
│  │                         │              │  + Head Component Test  │       │
│  └───────────────────────┘              └───────────┬────────────┘       │
│                                                       │                    │
│  PHASE 4: LLM REPORTS                                ▼                    │
│  ┌───────────────────────┐              ┌───────────────────────┐       │
│  │  SAR-Compliant          │◀─structured─│  Llama-3.1-8B           │       │
│  │  Investigation Reports  │  evidence   │  via vLLM (self-hosted) │       │
│  └───────────────────────┘              └───────────┬────────────┘       │
│                                                       │                    │
│  PHASE 5: EVALUATION                                 ▼                    │
│  ┌───────────────────────┐              ┌───────────────────────┐       │
│  │  Overall: 8.8/10        │◀────────────│  5-Layer Framework      │       │
│  │  Faithfulness: 97%      │              │  Deterministic +        │       │
│  │                         │              │  LLM-as-Judge            │       │
│  └───────────────────────┘              └───────────────────────┘       │
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
| XGBoost+GFP (baseline) | 274 | 0.6448 | 0.6436 | Matches published (65.70) ✓ |

### Explainer Results (8,957 flagged transactions)

| Metric | Result | Method |
|---|---|---|
| Causal verification | Top-attn Δ=0.020, Random Δ=0.000 | 200-sample occlusion |
| Attention causality | Confirmed (ratio > 1000×) | Perturbation test |
| Faithful predictions | 100% match | Logit verification |
| Pattern coverage | All 8 patterns + LONE + NONE | Per-pattern analysis |

### LLM Report Quality (10 reports evaluated)

| Evaluation Layer | Metric | Score |
|---|---|---|
| L1: Deterministic | Schema validity | **100%** |
| L1: Deterministic | No hallucinations | **100%** |
| L2: Faithfulness | Overall | **97%** |
| L2: Faithfulness | Fact accuracy | **100%** |
| L2: Faithfulness | Evidence coverage | **93%** |
| L3: Compliance | SAR format | **8.0/10** |
| L3: Compliance | Professional tone | **4.0/5** |
| L4: Narrative | Understandability | **4.4/5** |
| L4: Narrative | Convincingness | **4.3/5** |
| L4: Narrative | Communicability | **4.4/5** |
| L4: Narrative | Insightfulness | **3.4/5** |
| **Overall** | **Combined score** | **8.8/10** |

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
│   └── test_modification.py    # Verification tests (attention capture)
│
├── LLM Report Layer
│   ├── generate_reports.py     # SAR reports via Llama-3.1-8B (vLLM)
│   └── evaluate_reports.py     # 5-layer evaluation framework
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
| Storage | 100 GB free | 500+ GB |
| Python | 3.11 | 3.11 |
| CUDA | 12.1 | 12.1+ |

### Step 1: Environment Setup

```bash
# Clone
git clone https://github.com/khaledashoush/LLM-Fraud-Detection-And-Investigation-.git
cd LLM-Fraud-Detection-And-Investigation-

# Environment for training/GNN
conda create -n aml-research python=3.11 -y
conda activate aml-research
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install torch-geometric snapml xgboost optuna scikit-learn pandas numpy pyarrow

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

# Place files in data directory:
mkdir -p data
# Copy: HI-Medium_Trans.csv, HI-Medium_accounts.csv, HI-Medium_Patterns.txt
```

### Step 3: Run Pipeline

```bash
conda activate aml-research

# 1) Preprocessing (~3 hours)
python Preprocessing.py

# 2) Train FraudGT (~10 hours) or load checkpoint
python train_fraudgt.py --research --epochs 300

# 3) Generate explanations (~5 minutes)
python explain_fraudgt.py --occlusion-sample 200

# 4) Per-pattern analysis
python analyze_explainer.py --occlusion

# 5) Start LLM server (new terminal)
conda activate llm-serve
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --enable-prefix-caching \
    --port 8000

# 6) Generate reports (back to aml-research)
conda activate aml-research
python generate_reports.py --n 10

# 7) Evaluate
python evaluate_reports.py
```

---

## 📖 Usage Guide

- 🔍 **Explainer Options** — configure occlusion sample size, head-component analysis, and per-pattern breakdowns via `explain_fraudgt.py` / `analyze_explainer.py`.
- 📝 **Report Generation Options** — control the number of generated SAR reports and the served LLM endpoint via `generate_reports.py`.
- 📊 **Evaluation Options** — run the 5-layer scoring framework and adjust which layers (deterministic / LLM-as-Judge) are included via `evaluate_reports.py`.

---

## 🔬 Methodology

### Detection: FraudGT (Faithful Port)

Architecture faithfully implements **[FraudGT, ICAIF'24]**:

- `SparseNodeTransformer` with per-type Q/K/V/O projections
- Edge attention bias (`e_lin`) and message gate (`g_lin`) — Eqs. 5–8
- Eq. 9 prediction head: `y = σ(MLP(h_i ∥ e'_ij ∥ h_j))`
- Official training regime: 256 iters/epoch, warmup cosine, AdamW param groups
- Enhancements: RMP + Port Numbering + Ego IDs (from Egressy et al., AAAI'24)

### Explainer: Multi-Level Attention-Based

| Level | Method | Paper Reference |
|---|---|---|
| Signal | FraudGT's native attention (Eqs. 5–8) | [FraudGT ICAIF'24] |
| Verification | Neighbor occlusion (causal check) | [Jain & Wallace 2019] |
| Head analysis | Component occlusion (src / edge / dst) | [FraudGT Eq. 9] |
| Structural | GFP features + degrees + time deltas | [xFraud PVLDB'22] |

### LLM Reports: LLM as Narrator (GraphXAIN Approach)

| Component | Source | Implementation |
|---|---|---|
| Narrative rules | [GraphXAIN, Appendix A] | 11 rules + Miller's Law (7±2) |
| Report format | [FinCEN SAR guidelines] | 5 W's (Who/What/When/Where/Why) |
| Pattern definitions | [IBM AMLworld, Sec 3.2] | 8 typologies in system prompt |
| Serving stack | [LLMOps, Sec 5] | vLLM + APC + Llama-3.1-8B |

### Evaluation: 5-Layer Framework

| Layer | What it Measures | Method | Paper |
|---|---|---|---|
| L1: Deterministic | Schema validity, fields, tone | Automated checks | [LLMOps Sec 4] |
| L2: Faithfulness | Coverage, accuracy, fabrication | Evidence matching | [GNNExplainer + xFraud] |
| L3: Compliance | SAR format, tone, actionability | LLM-as-Judge | [LLMOps Sec 4.1] |
| L4: Narrative | 5 GraphXAIN dimensions | LLM-as-Judge | [GraphXAIN Sec 6] |
| L5: Overall | Weighted composite + per-pattern | Aggregation | [LLMOps Pareto] |

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

**LONE** (isolated illicit) transactions represent **35.4%** of all illicit activity in the dataset, yet FraudGT detects only **0.4%** of them. This highlights a fundamental limitation of message-passing architectures for isolated suspicious transactions.

### Finding 4: False Positives are Context-Dependent

False positives show the highest occlusion sensitivity (Δ=+0.035), indicating they were flagged due to specific related transactions. This enables targeted investigation of model errors.

---

## 🗺️ Roadmap

**Completed ✅**

- [x] Data pipeline (287 features, 8 patterns, 100% coverage)
- [x] FraudGT detection (F1 = 0.7291)
- [x] Attention-based explainer (8,957 explanations, 200 occlusion tests)
- [x] Per-pattern analysis (causal hierarchy, attention signatures)
- [x] LLM investigation reports (SAR format, 97% faithfulness)
- [x] 5-layer evaluation framework (overall 8.8/10)

**In Progress 🔄**

- [ ] Insightfulness improvement (3.4 → 4.0+)
- [ ] FraudGT v3: 500 epochs targeting F1 > 0.75

**Planned 📋**

- [ ] Scale up: 100–500 reports for thesis statistics
- [ ] Human evaluation study (GraphXAIN methodology)
- [ ] Backend: FastAPI (`/detect`, `/explain`, `/report`)
- [ ] Frontend: investigator dashboard

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

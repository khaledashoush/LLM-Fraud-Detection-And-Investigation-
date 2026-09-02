# AML End-to-End Pipeline — Fraud Detection & Investigation System

**Master's Thesis Project** — Khaled Ashoush

An end-to-end Anti-Money-Laundering system: **Detection → Explainability → LLM Report → RAG → Backend**.
This repository documents the **Detection phase** (models under active development),
with the following phases in progress.

---

## Project Vision

IBM AMLworld Data -> Preprocessing -> Graph Construction -> Feature Engineering
-> Multiple Fraud Detectors -> Fraud/Not Fraud
-> Explainability -> Structured Explanation -> LLM (local)
-> AML Investigation Report (+ RAG, Backend, Frontend)

The LLM is a **narrative layer, NOT a detector** — it converts model explanations into
investigator-ready reports.

---

## Current Status (Detection Phase — In Progress)

All results on **IBM AMLworld HI-Medium** (31.9M transactions, 2.08M accounts, 0.11% illicit)
under our **unified, documented evaluation protocol** (details below).

### GNN Track (the project's main detection track)

| GNN Model | Features | Test F1@thr | PR-AUC | Status |
|---|---|---|---|---|
| **FraudGT** (SparseNodeTransformer) | 15 | **0.7279** | 0.7273 | active improvement (v3 running: 500 epochs + wider fanout — targeting 0.75+) |
| Multi-PNA (ports/tds/ego/RMP) | 15 | 0.6703 | 0.6659 | validated (above published Multi-PNA+EU 66.48) |
| PNA+EU | 10 | 0.6493 | 0.6535 | validated (+5.2 over published 59.71) |

**Reference points** (published, same dataset):
- FraudGT paper: 75.93 (Multi-FraudGT) — our current gap: 3.1, closing
- Multi-PNA+EU (Egressy AAAI'24): 66.48 — **we're above it**
- PNA+EU: 59.71 — **we're above it**

### Baseline Track (comparison experiments)

| Baseline | Purpose | Result |
|---|---|---|
| XGBoost + GFP (feature-rich) | feature-consumer reference | 0.7501 (high, but not the project's track — see finding #1) |
| XGBoost (paper-faithful anchor) | **pipeline validation** | 0.6448 = published 65.70 (anchor validated) |

### Research findings so far

1. **Feature-richness is architecture-dependent** (core finding): gradient-boosted trees
   exploit rich features (286), while GNNs degrade with them (message-passing
   bottleneck) — motivating our design: **GNN as the structural learner** (main track)
   with GBT kept as a comparison/reference experiment.
2. **Threshold optimization matters**: up to +23 F1 points; always tuned on validation
   only (never on test).
3. **Official-repo protocol fixes** yield reproducible gains (see Fix History): the
   FraudGT track went 0.7079 -> 0.7279 from three training-regime corrections alone,
   with v3 (in progress) targeting the published 75.93.

---

## Repository Structure

- Preprocessing.py — Full pipeline: 287 features + splits + ports/tds + patterns
- train_fraudgt.py — FraudGT (main GNN track) — official regime
- train_multi_pna.py — Multi-PNA with official adaptations
- train_pna_eu.py — PNA + Edge Updates
- train_xgb.py — XGBoost baseline (comparison experiment)
- join_patterns.py — Links laundering-pattern ground truth to transactions
- analyze_patterns.py — Per-pattern recall + attempt-level detection
- pipeline.log — Full preprocessing logs (all runs documented)
- runs/ — metrics.json + config.json per experiment (all results)

**Data is NOT included** — download IBM AMLworld from Kaggle:
https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
then run Preprocessing.py (~3 hours on 64 threads).

---

## Preprocessing (Preprocessing.py)

### Input: 3 files per dataset (HI-Medium)
- HI-Medium_Trans.csv — 31,898,238 transactions (11 columns)
- HI-Medium_accounts.csv — account-to-entity mapping (166K entities, 518K accounts)
- HI-Medium_Patterns.txt — ground-truth laundering attempts (2,756 attempts, 22,743 tx)

### Pipeline steps
1. **Node IDs**: bank+account -> unique node index (2,077,023 nodes)
2. **Temporal split** (60/20/20, day-level greedy — official algorithm):
   - train 61.1% | val 17.3% | test 21.6% (contiguous -> graphs share memory)
   - documented label drift: illicit rate 0.081% -> 0.154% -> 0.160%
3. **Feature groups (287 total)**:
   - basic_ (10): log-amounts, cyclical time, currency/format codes
   - gfp_ (264): snapml GraphFeaturePreprocessor — fan-in/out, scatter-gather,
     simple/temporal cycles (official time windows), vertex statistics
   - structural_ (6): ports + time-deltas + is-first flags
   - causal_ (6): **our contribution** — strictly-past cumulative degree/fan/amount
   - entity_ (1): same-entity transfers (KYC experiment only)
4. **Ports/TDS** (official data_util.py semantics, vectorized): per-graph,
   first-interaction-order ports; backward time-deltas
5. **Patterns join**: 100% coverage (22,743/22,743) — evaluation-only metadata
   (pattern_type, attempt_id) — **never used as features**

### Output (processed_data/)
features_all.parquet (superset) / edge_index.pt / y.pt / masks.pt /
node_to_idx.parquet / edge_metadata.parquet (+patterns) / feature_metadata.json

---

## Unified Evaluation Protocol

Applied identically to **all models**:

| Element | Our protocol | Official repos |
|---|---|---|
| Split | temporal 60/20/20 (greedy day-level) | same algorithm |
| Graphs | 3 separate (train / train+val / full) — zero leakage | same |
| Standardization | train statistics only | official normalizes val/test with own stats (leak) |
| Test evaluation | **once, at the end** | official evaluates test every epoch (peeking) |
| Threshold | tuned on val (best-F1-over-thresholds), reported on test | argmax (=0.5) only |
| Coverage | **100%** (sources+destinations seeding) | partial (implicit) |
| Early stopping | patience on val best-F1 | none (fixed epochs) |

GNN training uses [Egressy et al. AAAI'24] adaptations: reverse message passing,
ports, time-deltas, ego-IDs — with our deg-histogram fix (below).

---

## The Fix History (documented honestly)

Every issue found and fixed during development:

| # | Issue | Impact | Fix |
|---|---|---|---|
| 1 | structural_td_* misaligned | wrong values on ~50% rows | label-aligned assignment |
| 2 | GFP vertex-stats on Timestamp only | amount-stats missing (224->264 cols) | [Timestamp, Amount Rcvd, Amount Paid] |
| 3 | PNA deg as per-node vector | scalers muted -> zero predictions | official histogram: bincount(degree) |
| 4 | Loader seeds = sources only | 47.5% of edges never evaluated | sources + destinations -> 100% coverage |
| 5 | FraudGT dot-decoding saturation | AUC 0.003 | **Paper Eq. 9 head**: MLP(h_i, e_ij, h_j) concat |
| 6 | scatter offset across relations | CUDA OOB | nodes share local indexing |
| 7 | AdamW decay on LayerNorm/biases | slow degradation | official param groups |
| 8 | F1@0.5 selection | suboptimal checkpoints | best-F1-over-thresholds |
| 9 | compressed cosine schedule | early peak then decay | official 256-iters/epoch regime |

---

## Models — usage

    python train_fraudgt.py     # main GNN track (train / inference auto-mode)
    python train_multi_pna.py   # inference: loads winner checkpoint (~15 min)
    python train_pna_eu.py      # inference: loads winner checkpoint (~10 min)
    python train_xgb.py         # baseline experiment (deployment/research modes)

    # Research/experiments:
    python train_fraudgt.py --research --epochs 500 --fanout 100,100

### FraudGT (main track) — faithful port
- SparseNodeTransformer from official gt_layer.py: per-type Q/K/V/O,
  edge attention-bias (e_lin), message-gate (g_lin), scatter-softmax (linear cost),
  clamp(-5,5), Type-FFN (nodes AND edges)
- Official training regime: AdamW param-groups, warmup-from-zero cosine,
  256 iters/epoch, batch-accum 8, loss [1,6]
- Progress: v1 0.7079 -> v2 **0.7279** -> v3 in progress (targeting 0.75+)

### Multi-PNA / PNA+EU (IBM Multi-GNN faithful)
- Official hyperparameters + adaptations; each run saves checkpoint, metrics,
  predictions, and standardization for the downstream explainability phase.

### XGBoost (baseline experiment)
- 50-trial Optuna; paper-faithful anchor (0.6448 = published) validates the
  whole pipeline; rich-feature experiment documents the architecture-dependency finding.

---

## Pattern-Level Analysis (from HI-Medium_Patterns.txt)

- Illicit distribution: **LONE 35.4%** (hardest) / GATHER-SCATTER 12.2% /
  SCATTER-GATHER 11.3% / STACK 11.3% / FAN-IN 6.6% / CYCLE 6.3% /
  BIPARTITE 6.1% / FAN-OUT 6.0% / RANDOM 4.7%
- Test split structurally "easier" (LONE 24.3% vs train 44.3%) — documented
- Attempt-level detection (bank-alert perspective): 1,250 test attempts —
  analysis via analyze_patterns.py

---

## Roadmap

- [ ] FraudGT v3: 500 epochs + fanout 100,100 (target 0.75+) — running
- [ ] Further GNN improvements (seeds sweep, pretraining Small->Medium)
- [ ] Explainability: SHAP/pattern context -> Structured Explanation JSON
- [ ] LLM layer (local): Llama/Qwen -> AML Investigation Report
- [ ] RAG: AML typologies knowledge base
- [ ] Backend: FastAPI (/detect, /explain, /report)
- [ ] Frontend: investigator dashboard

---

## References

1. Altman et al., AMLworld, NeurIPS'23 D&B
2. Blanusa et al., Graph Feature Preprocessor, ICAIF'24
3. Egressy et al., Provably Powerful GNNs for Directed Multigraphs, AAAI'24
4. Lin et al., FraudGT, ICAIF'24
5. Rampasek et al., GPS recipe, NeurIPS'22

Official repos consulted file-by-file (IBM Multi-GNN, junhongmit/FraudGT) —
all ports documented with source-file references.

---

## Author
**Khaled Ashoush** — Master's thesis, AML Fraud Detection & Investigation

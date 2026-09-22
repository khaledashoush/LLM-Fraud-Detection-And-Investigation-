# -*- coding: utf-8 -*-
"""
explain_fraudgt.py — Attention-based Explanations for FraudGT
================================================================
Multi-level explainer for FraudGT predictions on IBM AMLworld HI-Medium.

Explainer type: attention-based (white-box, post-hoc, instance-level)
              + perturbation verification (occlusion)
              + structural signals (task-agnostic evidence)
              + LLM-ready output (structured JSON)

Paper references:
  [FraudGT]  Lin et al., ICAIF'24 — attention mechanism (Eqs. 5-8),
             edge-based gate, Eq. 9 prediction head.
  [xFraud]   Rao et al., PVLDB'22 — hybrid evidence: task-aware (attention)
             + task-agnostic (structural/GFP signals). Section 3.4.
  [Vanguard] Pirmorad, 2025 — k-hop context for LLM reasoning. Section 2.1.
  [Jain & Wallace 2019] "Attention is not Explanation" — motivated the
             occlusion verification (empirical causal check).
  [IBM]      Altman et al., NeurIPS'23 — AMLworld 8 patterns (ground truth).
  [InteractiveGNNExplainer] Singh & Mukherjea, 2025 — perturb-observe-explain
             paradigm validation (Case Study 1).

Usage:
  python explain_fraudgt.py --pilot 20 --max-n 10    ← PILOT (fast test)
  python explain_fraudgt.py --max-n 100              ← first 100 flagged
  python explain_fraudgt.py                          ← ALL flagged (~20-30 min)
"""

import os, gc, json, time, math, logging, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pyarrow.parquet as pq
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import HeteroData

# ── Model architecture from training script (uses modified FraudGTLayer) ──
from train_fraudgt import (FraudGTModel, FraudGTLayer,
                            get_feature_cols, FEATURE_SETS, GROUP_PREFIXES,
                            GT, set_seed, F32_MAX)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("explain_fraudgt")

# ==========================================
# CONFIG
# ==========================================
DATA_DIR   = "processed_data"
CKPT_DIR   = "runs/gnn_fraudgt_gnn_L0_seed42"
OUTPUT_DIR = "explanations"


# ==========================================
# HELPERS (copied from train_fraudgt.py — same semantics)
# ==========================================
class AddEgoIds:
    """Marks seed nodes — Ego IDs from Egressy et al. adaptations."""
    def __call__(self, batch):
        node_store = batch['node']
        if hasattr(node_store, 'batch_size') and node_store.batch_size is not None:
            ego = torch.zeros(node_store.num_nodes, 1, dtype=torch.float)
            ego[:node_store.batch_size] = 1.0
            node_store.x = torch.cat([node_store.x, ego.to(node_store.x.device)], dim=1)
        return batch


def compute_official_ports_tds(src, dst, ts):
    """Official data_util.py semantics — vectorized. Same as training."""
    n = len(src)
    eid = np.arange(n)
    df = pd.DataFrame({'src': src, 'dst': dst, 'ts': ts, 'eid': eid})
    d = df.sort_values(['dst', 'ts'], kind='mergesort')
    d['is_first'] = ~d.duplicated(['dst', 'src'], keep='first')
    d['cum'] = d.groupby('dst', sort=False)['is_first'].cumsum() - 1
    d['port'] = d.groupby(['dst', 'src'], sort=False)['cum'].transform('first')
    in_ports = np.zeros(n, dtype=np.float32)
    in_ports[d['eid'].to_numpy()] = d['port'].to_numpy(dtype=np.float32)
    del d
    d = df.sort_values(['src', 'ts'], kind='mergesort')
    d['is_first'] = ~d.duplicated(['src', 'dst'], keep='first')
    d['cum'] = d.groupby('src', sort=False)['is_first'].cumsum() - 1
    d['port'] = d.groupby(['src', 'dst'], sort=False)['cum'].transform('first')
    out_ports = np.zeros(n, dtype=np.float32)
    out_ports[d['eid'].to_numpy()] = d['port'].to_numpy(dtype=np.float32)
    del d
    d = df.sort_values(['dst', 'ts'], kind='mergesort')
    d['td'] = d.groupby('dst', sort=False)['ts'].diff().fillna(0.0)
    in_tds = np.zeros(n, dtype=np.float32)
    in_tds[d['eid'].to_numpy()] = d['td'].to_numpy(dtype=np.float32)
    del d
    d = df.sort_values(['src', 'ts'], kind='mergesort')
    d['td'] = d.groupby('src', sort=False)['ts'].diff().fillna(0.0)
    out_tds = np.zeros(n, dtype=np.float32)
    out_tds[d['eid'].to_numpy()] = d['td'].to_numpy(dtype=np.float32)
    del d, df
    gc.collect()
    return in_ports, out_ports, in_tds, out_tds


def build_hetero(e_idx_t, X_slice, y_t, tx_t, mask_name, target_mask, num_nodes):
    """Same as training + ★ edge_id on rev_to (needed for attention mapping)."""
    n_e = X_slice.shape[0]
    fwd = torch.from_numpy(np.concatenate(
        [X_slice, np.zeros((n_e, 1), dtype=np.float32)], axis=1))
    rev = torch.from_numpy(np.concatenate(
        [X_slice, np.ones((n_e, 1), dtype=np.float32)], axis=1))
    data = HeteroData()
    data['node'].x = torch.ones((num_nodes, 1))
    data['node', 'to', 'node'].edge_index = e_idx_t
    data['node', 'to', 'node'].edge_attr = fwd
    data['node', 'to', 'node'].y = y_t
    data['node', 'to', 'node'].edge_id = tx_t
    setattr(data['node', 'to', 'node'], mask_name, target_mask)
    data['node', 'rev_to', 'node'].edge_index = e_idx_t.flip(0)
    data['node', 'rev_to', 'node'].edge_attr = rev
    data['node', 'rev_to', 'node'].edge_id = tx_t  # ★ same IDs for reverse
    return data


# ==========================================
# SECTION 1: Forward with Attention Capture
# Replicates FraudGTModel.forward() exactly, but also returns
# attention captures from each layer + final embeddings for head occlusion.
# ==========================================
@torch.no_grad()
def forward_full(model, x_dict, edge_index_dict, edge_attr_dict,
                 edge_label_index, edge_label_local_ids):
    """
    Full forward pass with attention capture enabled.
    Uses the model's OWN modules (same weights — no re-definition).
    Returns: (logits, attention_captures, final_h, final_ea)
    """
    # Enable attention capture on all layers
    for layer in model.layers:
        layer.capture_attention = True

    # ── Encoder (same as FraudGTModel.forward) ──
    h = {nt: model.node_encoder[nt](x_dict[nt]) for nt in x_dict}
    h = {nt: model.input_dropout(v) for nt, v in h.items()}
    ea = {et: model.edge_encoder["__".join(et)](edge_attr_dict[et])
          for et in edge_attr_dict}

    # ── GT layers (captures stored as side effect) ──
    for layer in model.layers:
        h, ea = layer(h, edge_index_dict, ea)

    attention_captures = [layer._attention_capture for layer in model.layers]

    # ── Eq. 9 head (same as FraudGTModel.forward) ──
    rel = ('node', 'to', 'node')
    src, dst = edge_label_index
    edge_emb = ea[rel][edge_label_local_ids]
    feats = torch.cat([h['node'][src], edge_emb, h['node'][dst]], dim=-1)
    logits = model.head(feats).squeeze(-1)

    # Disable capture (cleanup)
    for layer in model.layers:
        layer.capture_attention = False

    return logits, attention_captures, h, ea


# ==========================================
# SECTION 2: Attention Analysis
# [FraudGT Eqs. 5-8] + [xFraud Section 3.4 hybrid evidence]
# ==========================================
def analyze_attention(target_global_pos, captures, all_edge_ids, top_k=10):
    """
    For a target edge at `target_global_pos` in the concatenated edge list:
    - Own attention (per layer, avg over heads)
    - Top-k competitors (same destination, highest attention)
    - Top-k source edges (same source, highest attention)

    The 'to' + 'rev_to' concatenation means competitors include BOTH
    incoming transactions AND reversed outgoing transactions at the
    destination node — this is the RMP effect [Egressy AAAI'24].
    """
    results = {"layers": []}

    for layer_idx, cap in enumerate(captures):
        if cap is None:
            continue

        attn = cap["attn"].mean(dim=0)        # [E_total] — avg over H heads
        dst_flat = cap["dst_flat"]
        src_flat = cap["src_flat"]

        # ── Own attention ──
        own_attn = float(attn[target_global_pos])

        # ── Competitors: same destination (excluding self) ──
        dst_node = int(dst_flat[target_global_pos])
        same_dst = (dst_flat == dst_node)
        same_dst[target_global_pos] = False
        n_comp = int(same_dst.sum())

        top_incoming = []
        if n_comp > 0:
            comp_attn = attn[same_dst]
            comp_ids = all_edge_ids[same_dst]
            order = torch.argsort(comp_attn, descending=True)
            for rank in range(min(top_k, n_comp)):
                idx = int(order[rank])
                top_incoming.append({
                    "tx_id": int(comp_ids[idx]),
                    "attn": round(float(comp_attn[idx]), 6),
                    "rank": rank + 1,
                })

        # ── Source edges: same source (excluding self) ──
        src_node = int(src_flat[target_global_pos])
        same_src = (src_flat == src_node)
        same_src[target_global_pos] = False

        top_outgoing = []
        if same_src.sum() > 0:
            src_attn = attn[same_src]
            src_ids = all_edge_ids[same_src]
            order = torch.argsort(src_attn, descending=True)
            for rank in range(min(top_k, int(same_src.sum()))):
                idx = int(order[rank])
                top_outgoing.append({
                    "tx_id": int(src_ids[idx]),
                    "attn": round(float(src_attn[idx]), 6),
                    "rank": rank + 1,
                })

        # Rank of own attention among competitors
        own_rank = 1
        if n_comp > 0:
            own_rank = int((attn[same_dst] > own_attn).sum()) + 1

        results["layers"].append({
            "layer": layer_idx + 1,
            "own_attention": round(own_attn, 6),
            "own_rank": own_rank,
            "n_competitors": n_comp,
            "top_incoming": top_incoming,
            "top_outgoing": top_outgoing,
        })

    return results


# ==========================================
# SECTION 3: Head-Component Occlusion
# [FraudGT Eq. 9]: logits = MLP(h_src ∥ e_ij ∥ h_dst)
# Attention explains message-passing — NOT how the head weighs components.
# ==========================================
@torch.no_grad()
def head_occlusion(model, final_h, final_ea,
                   edge_label_index, edge_label_local_ids):
    """
    Zero out each Eq.9 component → measure probability drop.
    Cheap: only re-runs the MLP head (no full forward).
    """
    rel = ('node', 'to', 'node')
    src, dst = edge_label_index
    edge_emb = final_ea[rel][edge_label_local_ids]
    h_src = final_h['node'][src]
    h_dst = final_h['node'][dst]

    # Baseline
    feats_orig = torch.cat([h_src, edge_emb, h_dst], dim=-1)
    prob_orig = torch.sigmoid(model.head(feats_orig).squeeze(-1))

    # Occlude each component
    drops = {}
    names = ["src", "edge", "dst"]
    for i, name in enumerate(names):
        parts = [h_src, edge_emb, h_dst]
        parts[i] = torch.zeros_like(parts[i])
        feats_occ = torch.cat(parts, dim=-1)
        prob_occ = torch.sigmoid(model.head(feats_occ).squeeze(-1))
        drops[name] = (prob_orig - prob_occ).cpu().numpy()

    return {"baseline": prob_orig.cpu().numpy(), "drops": drops}


# ==========================================
# SECTION 4: Neighbor Occlusion (causal verification)
# [Jain & Wallace 2019]: attention ≠ causation → empirical check.
# [InteractiveGNNExplainer]: perturb-observe-explain paradigm.
# ==========================================
@torch.no_grad()
def neighbor_occlusion(model, batch, competitor_global_pos,
                       edge_label_index, edge_label_local_ids, n_to):
    """
    Zero out a competitor edge's features → re-run FULL forward → Δprob.
    Expensive (full forward) — use on sample only (~50 tx).
    """
    # Map global position → (relation, local position)
    if competitor_global_pos < n_to:
        rel = ('node', 'to', 'node')
        local = competitor_global_pos
    else:
        rel = ('node', 'rev_to', 'node')
        local = competitor_global_pos - n_to

    # Save + zero out
    original = batch.edge_attr_dict[rel][local].clone()
    batch.edge_attr_dict[rel][local] = 0.0

    # Re-run forward (same path as prediction)
    logits_occ = model(batch.x_dict, batch.edge_index_dict,
                       batch.edge_attr_dict,
                       edge_label_index, edge_label_local_ids)
    prob_occ = torch.sigmoid(logits_occ).cpu().numpy()

    # Restore
    batch.edge_attr_dict[rel][local] = original

    return prob_occ


# ==========================================
# SECTION 5: Enrichment
# ==========================================
def enrich_explanations(explanations, edge_meta, node_df, features_df):
    """Add transaction details, account names, and structural signals."""
    idx_to_node = dict(zip(node_df["idx"], node_df["node"]))
    meta_idx = edge_meta.set_index("transaction_id")

    for tx_id, exp in explanations.items():
        # Transaction details
        if tx_id in meta_idx.index:
            row = meta_idx.loc[tx_id]
            exp["transaction_details"] = {
                "timestamp": str(row.get("Timestamp", "")),
                "src_account": idx_to_node.get(int(row["src_idx"]), "unknown"),
                "dst_account": idx_to_node.get(int(row["dst_idx"]), "unknown"),
                "src_idx": int(row["src_idx"]),
                "dst_idx": int(row["dst_idx"]),
            }
            # Ground truth — EVALUATION ONLY (never in LLM prompts!)
            exp["pattern_ground_truth"] = str(row.get("pattern_type", "UNKNOWN"))

        # Structural signals from features
        if tx_id < len(features_df):
            fr = features_df.iloc[tx_id]
            lap = float(fr.get("basic_log_amount_paid", 0))
            exp["structural_signals"] = {
                "approx_amount_paid": round(math.expm1(abs(lap)), 2),
                "log_amount_paid": round(lap, 4),
                "log_amount_received": round(float(fr.get("basic_log_amount_received", 0)), 4),
                "hour_sin": round(float(fr.get("basic_hour_sin", 0)), 4),
                "hour_cos": round(float(fr.get("basic_hour_cos", 0)), 4),
            }
            # GFP signals (first 5 — key pattern indicators)
            for i in range(5):
                col = f"gfp_feature_{i}"
                if col in fr.index:
                    exp["structural_signals"][col] = round(float(fr[col]), 4)

    return explanations


# ==========================================
# MAIN
# ==========================================
def main():
    ap = argparse.ArgumentParser(
        description="Attention-based explanations for FraudGT")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Default: from metrics.json")
    ap.add_argument("--max-n", type=int, default=None,
                    help="Max transactions to explain")
    ap.add_argument("--top-k-neighbors", type=int, default=10)
    ap.add_argument("--occlusion-sample", type=int, default=50,
                    help="Sample size for occlusion tests")
    ap.add_argument("--pilot", type=int, default=None,
                    help="Pilot mode: process only first N batches")
    ap.add_argument("--fanout", default="100,100")
    ap.add_argument("--batch-size", type=int, default=2048)
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t_start = time.time()

    # ═══════════════════════════════════════
    # 1) Load metrics + threshold
    # ═══════════════════════════════════════
    with open(os.path.join(CKPT_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    threshold = args.threshold or metrics.get("threshold", 0.5)
    fs = metrics.get("feature_set", "gnn_L0")
    logger.info(f"[Config] threshold={threshold:.4f} | feature_set={fs} | "
                f"device={device}")

    # ═══════════════════════════════════════
    # 2) Load predictions → flagged set
    # ═══════════════════════════════════════
    preds = pd.read_parquet(os.path.join(CKPT_DIR, "predictions.parquet"))
    flagged = preds[preds["y_proba"] >= threshold].copy()
    flagged_ids = set(flagged["transaction_id"].tolist())
    flagged_proba = dict(zip(flagged["transaction_id"], flagged["y_proba"]))
    logger.info(f"[Flagged] {len(flagged):,} transactions ≥ {threshold:.4f}")

    if args.max_n and len(flagged) > args.max_n:
        flagged = flagged.head(args.max_n)
        flagged_ids = set(flagged["transaction_id"].tolist())
        logger.info(f"[Flagged] limited to {len(flagged)}")

    # ═══════════════════════════════════════
    # 3) Load features (same path as training)
    # ═══════════════════════════════════════
    parquet_path = f"{DATA_DIR}/features_all.parquet"
    pf = pq.ParquetFile(parquet_path)
    n_rows = pf.metadata.num_rows
    schema_names = pf.schema_arrow.names
    feature_cols = get_feature_cols(schema_names, fs)
    logger.info(f"[Load] {fs}: {len(feature_cols)} features | {n_rows:,} rows")

    small = pf.read(columns=["split", "transaction_id"]).to_pandas()
    split_np = small["split"].astype(str).to_numpy()
    txid = small["transaction_id"].to_numpy(dtype=np.int64)
    tr_pd = split_np == "train"
    del small; gc.collect()

    X = np.empty((n_rows, len(feature_cols)), dtype=np.float32)
    for j, c in enumerate(feature_cols):
        col = pf.read(columns=[c]).column(0).to_numpy(zero_copy_only=False)
        bad = ~np.isfinite(col) | (np.abs(col) > F32_MAX)
        if bad.any():
            col = col.copy(); col[bad] = np.nan
        mu = float(np.nanmean(col[tr_pd]))
        sig = float(np.nanstd(col[tr_pd]))
        if not np.isfinite(mu): mu = 0.0
        if not np.isfinite(sig) or sig <= 0: sig = 1.0
        c32 = col.astype(np.float32)
        c32 -= np.float32(mu); c32 /= np.float32(sig)
        np.clip(c32, -10.0, 10.0, out=c32)
        np.nan_to_num(c32, nan=0.0, copy=False)
        X[:, j] = c32
        del col, c32
    logger.info(f"[Mem] X = {X.nbytes / 1e9:.1f} GB ✓")

    # ═══════════════════════════════════════
    # 4) Graph components + ports/tds
    # ═══════════════════════════════════════
    edge_index = torch.load(f"{DATA_DIR}/edge_index.pt",
                            weights_only=False)
    y_all = torch.load(f"{DATA_DIR}/y.pt", weights_only=False)
    masks = torch.load(f"{DATA_DIR}/masks.pt", weights_only=False)
    num_nodes = int(edge_index.max().item()) + 1
    y_lookup = y_all.numpy()

    tr, va, te = masks['train'], masks['val'], masks['test']
    tr_np, te_np = tr.numpy(), te.numpy()
    n_train = int(tr_np.sum())
    logger.info(f"[Graph] {n_rows:,} edges | {num_nodes:,} nodes | "
                f"train {n_train:,} | test {int(te_np.sum()):,}")

    # Timestamps for ports/tds
    ts_df = pq.ParquetFile(f"{DATA_DIR}/edge_metadata.parquet").read(
        columns=["Timestamp"]).to_pandas()
    ts_all_np = (ts_df["Timestamp"].astype("int64") // 10**9).to_numpy().astype(np.int64)
    del ts_df; gc.collect()

    # ── Build TRAIN graph (for ports/tds normalization stats) ──
    logger.info("[Ports+TDS] building train graph for normalization stats...")
    train_graph = build_hetero(
        edge_index[:, tr].contiguous(), X[:n_train],
        y_all[tr], torch.from_numpy(txid[:n_train]),
        "train_mask", torch.ones(n_train, dtype=torch.bool), num_nodes)

    store_tr = train_graph['node', 'to', 'node']
    eids_tr = store_tr.edge_id.numpy()
    src_tr = store_tr.edge_index[0].numpy()
    dst_tr = store_tr.edge_index[1].numpy()
    ts_tr = ts_all_np[eids_tr]
    in_p, out_p, in_td, out_td = compute_official_ports_tds(src_tr, dst_tr, ts_tr)
    extra_tr = torch.from_numpy(np.stack([in_p, out_p, in_td, out_td], axis=1))
    store_tr.edge_attr = torch.cat([store_tr.edge_attr, extra_tr.float()], dim=1)

    # ★ Normalization stats from TRAIN graph (same as training)
    mu_ports = store_tr.edge_attr[:, -4:].mean(0)
    sd_ports = store_tr.edge_attr[:, -4:].std(0).clamp(min=1)
    logger.info(f"[Ports+TDS] train stats: mu={mu_ports.tolist()}, sd={sd_ports.tolist()}")
    del train_graph, store_tr, extra_tr
    gc.collect()

    # ── Build TEST graph ──
    logger.info("[Ports+TDS] building test graph...")
    test_graph = build_hetero(
        edge_index.contiguous(), X, y_all,
        torch.from_numpy(txid), "test_mask", te, num_nodes)

    store_te = test_graph['node', 'to', 'node']
    eids_te = store_te.edge_id.numpy()
    src_te = store_te.edge_index[0].numpy()
    dst_te = store_te.edge_index[1].numpy()
    ts_te = ts_all_np[eids_te]
    in_p, out_p, in_td, out_td = compute_official_ports_tds(src_te, dst_te, ts_te)
    extra_fwd = torch.from_numpy(np.stack([in_p, out_p, in_td, out_td], axis=1))
    extra_rev = extra_fwd[:, [0, 1, 3, 2]].contiguous()

    store_te.edge_attr = torch.cat([store_te.edge_attr, extra_fwd.float()], dim=1)
    rev_store = test_graph['node', 'rev_to', 'node']
    rev_store.edge_attr = torch.cat([rev_store.edge_attr, extra_rev.float()], dim=1)

    # ★ Normalize with TRAIN stats (identical to training)
    store_te.edge_attr[:, -4:] = (store_te.edge_attr[:, -4:] - mu_ports) / sd_ports
    rev_store.edge_attr[:, -4:] = (rev_store.edge_attr[:, -4:] - mu_ports) / sd_ports

    del extra_fwd, extra_rev
    gc.collect()
    logger.info("[Ports+TDS] test graph ✓ (normalized with train stats)")

    e_dim = store_te.edge_attr.shape[1]
    logger.info(f"[Graph] edge_dim = {e_dim}")

    # ═══════════════════════════════════════
    # 5) Eval loader (same as training)
    # ═══════════════════════════════════════
    fanout = [int(x) for x in args.fanout.split(",")]
    nn_dict = {('node', 'to', 'node'): fanout,
               ('node', 'rev_to', 'node'): fanout}

    mask = store_te.test_mask
    seeds = torch.cat([store_te.edge_index[0, mask].unique(),
                       store_te.edge_index[1, mask].unique()]).unique()
    logger.info(f"[Loader] test: {len(seeds):,} seeds")

    test_loader = NeighborLoader(
        test_graph, num_neighbors=nn_dict,
        batch_size=args.batch_size,
        input_nodes=('node', seeds),
        shuffle=False, num_workers=4,
        transform=AddEgoIds())

    # ═══════════════════════════════════════
    # 6) Load model
    # ═══════════════════════════════════════
    metadata = [('node',), (('node', 'to', 'node'), ('node', 'rev_to', 'node'))]
    model = FraudGTModel(
        node_dim_in=2, edge_dim_in=e_dim,
        num_heads=GT["attn_heads"], dim_hidden=GT["dim_hidden"],
        num_layers=GT["layers"], metadata=metadata).to(device)

    ckpt_path = os.path.join(CKPT_DIR, "model.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                      weights_only=False))
    model.eval()
    logger.info(f"[Model] loaded from {ckpt_path} ✓")

    # ═══════════════════════════════════════
    # 7) Main loop: process batches
    # ═══════════════════════════════════════
    explanations = {}
    occlusion_sample_ids = sorted(flagged_ids)[:args.occlusion_sample]
    n_batches = 0
    n_verified = 0
    max_diff = 0.0
    occlusion_stats = {"top_drops": [], "random_drops": []}

    pilot_label = f"PILOT ({args.pilot} batches)" if args.pilot else "FULL"
    logger.info(f"[Loop] processing test batches | {pilot_label} | "
                f"target: {len(flagged_ids)} transactions")

    for batch in test_loader:
        n_batches += 1
        if args.pilot and n_batches > args.pilot:
            logger.info(f"[Pilot] reached {args.pilot} batches — stopping")
            break

        batch = batch.to(device)
        store = batch['node', 'to', 'node']
        m = store.test_mask
        if m.sum() == 0:
            continue

        # ── Skip batches without flagged transactions (saves GPU time) ──
        batch_edge_ids = store.edge_id.cpu().numpy()
        batch_flagged = set(batch_edge_ids.tolist()) & flagged_ids
        if not batch_flagged:
            continue

        # ── Forward with attention capture ──
        local_pos = torch.nonzero(m, as_tuple=False).squeeze(-1).to(device)
        logits, captures, final_h, final_ea = forward_full(
            model, batch.x_dict, batch.edge_index_dict,
            batch.edge_attr_dict, store.edge_index[:, m], local_pos)

        probas = torch.sigmoid(logits).cpu().numpy()

        # ── Sanity: logits match saved predictions ──
        for i, pos in enumerate(local_pos.cpu().numpy()):
            tx_id = int(batch_edge_ids[pos])
            if tx_id in flagged_proba:
                diff = abs(float(probas[i]) - flagged_proba[tx_id])
                max_diff = max(max_diff, diff)
                n_verified += 1

        # ── Build edge_id mapping for ALL edges (to + rev_to) ──
        to_ids = store.edge_id.cpu()
        rev_store = batch['node', 'rev_to', 'node']
        rev_ids = rev_store.edge_id.cpu()
        all_edge_ids = torch.cat([to_ids, rev_ids])
        n_to = len(to_ids)

        # ── Per-flagged-transaction analysis ──
        for i, pos in enumerate(local_pos.cpu().numpy()):
            tx_id = int(batch_edge_ids[pos])
            if tx_id not in flagged_ids or tx_id in explanations:
                continue

            # (a) Attention analysis
            # 'to' edges are at offset 0 in the concatenated list
            attn_exp = analyze_attention(
                pos, captures, all_edge_ids, top_k=args.top_k_neighbors)

            # (b) Head-component occlusion [Eq. 9] — always (cheap)
            head_occ = head_occlusion(
                model, final_h, final_ea,
                store.edge_index[:, m], local_pos)
            head_drops = {k: round(float(v[i]), 6)
                          for k, v in head_occ["drops"].items()}

            # (c) Neighbor occlusion (sample only — expensive)
            neigh_occ = None
            if tx_id in occlusion_sample_ids and captures:
                last_cap = captures[-1]  # final layer
                dst_node = int(last_cap["dst_flat"][pos])
                same_dst = (last_cap["dst_flat"] == dst_node)
                same_dst[pos] = False
                if same_dst.sum() > 0:
                    attn_avg = last_cap["attn"].mean(dim=0)
                    comp_attn = attn_avg[same_dst]
                    best_local = int(torch.argmax(comp_attn))
                    comp_positions = torch.nonzero(same_dst)
                    comp_global = int(comp_positions[best_local])

                    # Top-attention occlusion
                    prob_top = neighbor_occlusion(
                        model, batch, comp_global,
                        store.edge_index[:, m], local_pos, n_to)

                    # Random baseline
                    rand_pos = int(torch.randint(0, len(all_edge_ids), (1,)))
                    if rand_pos == pos:
                        rand_pos = (rand_pos + 1) % len(all_edge_ids)
                    prob_rand = neighbor_occlusion(
                        model, batch, rand_pos,
                        store.edge_index[:, m], local_pos, n_to)

                    drop_top = float(probas[i] - prob_top[i])
                    drop_rand = float(probas[i] - prob_rand[i])
                    occlusion_stats["top_drops"].append(drop_top)
                    occlusion_stats["random_drops"].append(drop_rand)

                    neigh_occ = {
                        "top_competitor_tx": int(all_edge_ids[comp_global]),
                        "prob_original": round(float(probas[i]), 6),
                        "prob_top_occluded": round(float(prob_top[i]), 6),
                        "prob_random_occluded": round(float(prob_rand[i]), 6),
                        "drop_top": round(drop_top, 6),
                        "drop_random": round(drop_rand, 6),
                    }

            # (d) Assemble explanation
            explanations[tx_id] = {
                "transaction_id": tx_id,
                "prediction": {
                    "y_proba": round(float(probas[i]), 6),
                    "y_proba_saved": round(flagged_proba.get(tx_id, 0), 6),
                    "y_true": int(y_lookup[tx_id]),
                    "model": "fraudgt_v2",
                    "threshold": threshold,
                },
                "attention_analysis": attn_exp,
                "head_component_importance": {
                    "src_drop": head_drops.get("src", 0),
                    "edge_drop": head_drops.get("edge", 0),
                    "dst_drop": head_drops.get("dst", 0),
                    "note": "Probability drop when zeroing each Eq.9 component",
                },
                "neighbor_occlusion": neigh_occ,
            }

        # Progress
        done = len(explanations)
        if done > 0 and done % 20 == 0:
            elapsed = time.time() - t_start
            rate = done / max(n_batches, 1)
            logger.info(f"[Progress] batch {n_batches} | {done}/{len(flagged_ids)} "
                        f"explained | {elapsed/60:.1f}m elapsed")

        if len(explanations) >= len(flagged_ids):
            logger.info(f"[Done] all {len(flagged_ids)} explained!")
            break

    # ═══════════════════════════════════════
    # 8) Verification report
    # ═══════════════════════════════════════
    logger.info("=" * 60)
    logger.info(f"[Verify] predictions checked: {n_verified}")
    logger.info(f"[Verify] max prob difference: {max_diff:.6f}")
    if max_diff > 0.01:
        logger.warning(f"[Verify] ⚠️ max_diff={max_diff:.6f} > 0.01 — "
                       f"attention context may differ from saved predictions")
    elif max_diff > 0.001:
        logger.warning(f"[Verify] moderate diff={max_diff:.6f} — "
                       f"likely CUDA non-determinism (acceptable)")
    else:
        logger.info(f"[Verify] ✓ predictions match saved (within tolerance)")

    # ═══════════════════════════════════════
    # 9) Occlusion summary
    # ═══════════════════════════════════════
    occlusion_summary = {}
    if occlusion_stats["top_drops"]:
        td = occlusion_stats["top_drops"]
        rd = occlusion_stats["random_drops"]
        ratio = np.mean(td) / max(np.mean(rd), 1e-8)
        occlusion_summary = {
            "n_tested": len(td),
            "top_mean_drop": round(float(np.mean(td)), 6),
            "top_std_drop": round(float(np.std(td)), 6),
            "random_mean_drop": round(float(np.mean(rd)), 6),
            "ratio_top_vs_random": round(float(ratio), 2),
            "verdict": "attention IS causal (ratio > 2)" if ratio > 2
                       else "weak evidence (ratio ≤ 2)",
        }
        logger.info("=" * 60)
        logger.info(f"[Occlusion] {len(td)} transactions tested:")
        logger.info(f"  Top-attention occlusion Δprob: {np.mean(td):.4f} ± {np.std(td):.4f}")
        logger.info(f"  Random occlusion Δprob:       {np.mean(rd):.4f} ± {np.std(rd):.4f}")
        logger.info(f"  Ratio: {ratio:.2f}× → {occlusion_summary['verdict']}")

    # ═══════════════════════════════════════
    # 10) Enrichment
    # ═══════════════════════════════════════
    logger.info("[Enrich] loading metadata...")
    edge_meta = pd.read_parquet(f"{DATA_DIR}/edge_metadata.parquet")
    node_df = pd.read_parquet(f"{DATA_DIR}/node_to_idx.parquet")

    # Load key features (subset for speed)
    key_cols = ["basic_log_amount_paid", "basic_log_amount_received",
                "basic_hour_sin", "basic_hour_cos"]
    gfp_cols = [f"gfp_feature_{i}" for i in range(5)]
    available = [c for c in key_cols + gfp_cols if c in schema_names]
    features_df = pq.ParquetFile(parquet_path).read(
        columns=available).to_pandas()

    explanations = enrich_explanations(explanations, edge_meta, node_df,
                                        features_df)

    # ═══════════════════════════════════════
    # 11) Save output
    # ═══════════════════════════════════════
    output = {
        "metadata": {
            "model": "fraudgt_v2",
            "checkpoint": ckpt_path,
            "threshold": threshold,
            "feature_set": fs,
            "n_flagged_total": len(preds[preds["y_proba"] >= threshold]),
            "n_explained": len(explanations),
            "n_batches_processed": n_batches,
            "pilot_mode": bool(args.pilot),
            "max_prob_diff": round(max_diff, 8),
            "elapsed_min": round((time.time() - t_start) / 60, 1),
            "explainer_type": "attention-based (white-box) + occlusion "
                              "verification + structural signals",
            "paper_refs": {
                "attention": "FraudGT ICAIF'24 Eqs. 5-8",
                "hybrid_evidence": "xFraud PVLDB'22 Sec 3.4",
                "occlusion_rationale": "Jain & Wallace 2019",
                "perturb_observe": "InteractiveGNNExplainer 2025",
                "structural_signals": "IBM AMLworld NeurIPS'23 (8 patterns)",
            },
        },
        "occlusion_summary": occlusion_summary,
        "explanations": list(explanations.values()),
    }

    out_path = os.path.join(OUTPUT_DIR, "explanations.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    size_mb = os.path.getsize(out_path) / 1e6
    logger.info(f"✅ Saved {len(explanations)} explanations to {out_path} "
                f"({size_mb:.1f} MB)")

    # ═══════════════════════════════════════
    # 12) Summary stats
    # ═══════════════════════════════════════
    if explanations:
        tp = sum(1 for e in explanations.values()
                 if e["prediction"]["y_true"] == 1)
        fp = len(explanations) - tp
        logger.info("=" * 60)
        logger.info(f"[Summary] Explained: {len(explanations)}")
        logger.info(f"  True positives:  {tp} ({tp/len(explanations)*100:.1f}%)")
        logger.info(f"  False positives: {fp} ({fp/len(explanations)*100:.1f}%)")

        patterns = {}
        for e in explanations.values():
            pt = e.get("pattern_ground_truth", "UNKNOWN")
            patterns[pt] = patterns.get(pt, 0) + 1
        logger.info("  Pattern distribution (ground truth):")
        for pt, cnt in sorted(patterns.items(), key=lambda x: -x[1]):
            logger.info(f"    {pt:<18} {cnt:>6} ({cnt/len(explanations)*100:.1f}%)")

        # Head component summary
        src_drops = [e["head_component_importance"]["src_drop"]
                     for e in explanations.values()]
        edge_drops = [e["head_component_importance"]["edge_drop"]
                      for e in explanations.values()]
        dst_drops = [e["head_component_importance"]["dst_drop"]
                     for e in explanations.values()]
        logger.info("  Head component importance (mean drops):")
        logger.info(f"    src:  {np.mean(src_drops):.4f}")
        logger.info(f"    edge: {np.mean(edge_drops):.4f}")
        logger.info(f"    dst:  {np.mean(dst_drops):.4f}")

        logger.info(f"  Total time: {(time.time() - t_start) / 60:.1f} minutes")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
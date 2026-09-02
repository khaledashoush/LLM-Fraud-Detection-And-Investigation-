# -*- coding: utf-8 -*-
"""
train_fraudgt.py — FraudGT (SparseNodeTransformer) — official training regime
================================================================================
Usage:
  python train_fraudgt.py --research --epochs 300        ← TRAIN with official regime
  python train_fraudgt.py                               ← INFERENCE after first successful run
  python train_fraudgt.py --research --feature-set gnn_L2 --epochs 300   ← L2 experiment

Architecture — faithful to the PAPER (FraudGT, ICAIF'24) + official repo:
  FeatureEncoder (Linear per type) → 2× GTLayer(SparseNodeTransformer) → Edge Head (Eq.9)

Faithful port sources:
  - gt_layer.py         (SparseNodeTransformer: Q/K/V + e_lin bias + g_lin gate +
                         scatter softmax + clamp(-5,5) + Type-FFN, nodes AND edges)
  - gt_model.py         (structure)
  - head.py + Eq.9      (edge prediction: y = σ(MLP(h_i ∥ e'_ij ∥ h_j)))
  - AML-Medium-HI-SparseNodeGT+ports+Ego.yaml   (all hyperparameters)
  - extra_optimizers.py (optimizer + scheduler semantics)

Official-faithful fixes:
  [P-OPTIM]   AdamW parameter groups: NO weight decay on LayerNorm/biases
  [P-SCHED]   warmup from ~0: max(1e-6, step/warmup) — official semantics
  [P-REGIME]  official train.iter_per_epoch=256 (256 iterations per epoch)
  [P-EQ9]     edge prediction head per paper Eq. 9
  [P-LOSS]    weighted CE [1,6] ≡ BCEWithLogitsLoss(pos_weight=6)

Selection criterion:
  [FIX-BESTF1] ★ checkpoint/early-stopping select the epoch with the BEST
              achievable F1 (threshold-optimized on val) — NOT F1@0.5.
              The paper-reported F1 is the optimized one; the operating
              threshold is chosen on val and applied to test as-is.

Engineering fixes:
  [FIX-GELU]  nn.GELU() module in Sequential
  [FIX-SCAT]  no offset in scatter (nodes share local indexing)
  [FIX-POS]   edge positions computed LOCALLY per batch via nonzero(mask)
"""

import os, gc, json, time, logging, argparse, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyarrow.parquet as pq
from torch_scatter import scatter_max
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import HeteroData
from torch_geometric.utils import degree
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             average_precision_score, precision_recall_curve)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("train_fraudgt")

# ==========================================
# CONFIG — official SparseNodeGT+ports+Ego yaml
# ==========================================
DATA_DIR   = "processed_data"
RUNS_DIR   = "runs"
EPOCHS     = 300          # official regime (500 max — 300 fits our time budget)
PATIENCE   = 25           # longer patience for the long-cosine regime
F32_MAX    = float(np.finfo(np.float32).max)
TEST_PASSES = 2

GT = {
    "dim_hidden": 64,
    "layers": 2,
    "attn_heads": 8,
    "act": "gelu",
    "dropout": 0.2,
    "attn_dropout": 0.3,
    "layer_norm": True,
    "batch_norm": False,
    "residual": "fixed",
    "ffn": "type",
}
OPTIM = {
    "optimizer": "adamW",
    "base_lr": 0.001,
    "weight_decay": 1e-5,
    "clip_grad_norm": True,
    "warmup_epochs": 5,
    "batch_accumulation": 8,
    "iter_per_epoch": 256,
}
TRAIN_ITERS_PER_EPOCH = 256
LOSS_WEIGHT = [1.0, 6.0]

WINNER_DIR = "runs/gnn_fraudgt_gnn_L0_seed42"

GROUP_PREFIXES = {
    "basic": ("basic_",), "gfp": ("gfp_",), "structural": ("structural_",),
    "causal": ("causal_",), "entity": ("entity_",),
}
FEATURE_SETS = {
    "gnn_L0": ["basic"],
    "gnn_L1": ["basic", "gfp"],
    "gnn_L2": ["basic", "gfp", "structural", "causal"],
    "gnn_L3": ["basic", "gfp", "structural", "causal", "entity"],
}


def get_feature_cols(all_cols, fs):
    prefixes = tuple(p for g in FEATURE_SETS[fs] for p in GROUP_PREFIXES[g])
    cols = [c for c in all_cols if c.startswith(prefixes)]
    assert not ({"transaction_id", "Is Laundering", "split"} & set(cols)), "LEAK!"
    return cols


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


ACT = {"gelu": F.gelu, "relu": F.relu}


# ==========================================
# [FIX-BESTF1] selection criterion helper
# ==========================================
def best_f1_over_thresholds(truths, probas):
    """Best achievable F1 over ALL thresholds (papers report this — the
    operating point is a business decision, not a modeling one).
    Returns (best_f1, best_threshold)."""
    prec, rec, thr = precision_recall_curve(truths, probas)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    idx = int(np.argmax(f1s[:-1]))          # off-by-one fix
    return float(f1s[idx]), float(thr[idx])


# ==========================================
# FraudGTLayer — faithful port of GTLayer / SparseNodeTransformer
# ==========================================
class FraudGTLayer(nn.Module):
    def __init__(self, dim_h, num_heads, metadata, layer_idx):
        super().__init__()
        self.dim_h = dim_h
        self.num_heads = num_heads
        self.metadata = metadata
        self.layer_idx = layer_idx
        self.act = ACT[GT["act"]]
        H, D = num_heads, dim_h // num_heads
        self.H, self.D = H, D

        # ---- per node-type Q/K/V/O (gt_layer.py) ----
        self.k_lin = nn.ModuleDict()
        self.q_lin = nn.ModuleDict()
        self.v_lin = nn.ModuleDict()
        self.o_lin = nn.ModuleDict()
        for node_type in metadata[0]:
            self.k_lin[node_type] = nn.Linear(dim_h, dim_h)
            self.q_lin[node_type] = nn.Linear(dim_h, dim_h)
            self.v_lin[node_type] = nn.Linear(dim_h, dim_h)
            self.o_lin[node_type] = nn.Linear(dim_h, dim_h)

        # ---- per edge-type: e_lin (bias) + g_lin (gate) + oe_lin (edge out) ----
        self.e_lin = nn.ModuleDict()
        self.g_lin = nn.ModuleDict()
        self.oe_lin = nn.ModuleDict()
        for edge_type in metadata[1]:
            key = "__".join(edge_type)
            self.e_lin[key] = nn.Linear(dim_h, dim_h)
            self.g_lin[key] = nn.Linear(dim_h, dim_h)
            self.oe_lin[key] = nn.Linear(dim_h, dim_h)

        # ---- norms (LayerNorm per official config, pre-norm) ----
        self.norm1_global = nn.ModuleDict()
        self.norm1_edge_global = nn.ModuleDict()
        self.norm2_ffn = nn.ModuleDict()
        self.norm2_edge_ffn = nn.ModuleDict()
        for node_type in metadata[0]:
            self.norm1_global[node_type] = nn.LayerNorm(dim_h)
            self.norm2_ffn[node_type] = nn.LayerNorm(dim_h)
        for edge_type in metadata[1]:
            key = "__".join(edge_type)
            self.norm1_edge_global[key] = nn.LayerNorm(dim_h)
            self.norm2_edge_ffn[key] = nn.LayerNorm(dim_h)

        # ---- Type-FFN: dim → 2dim → dim, nodes AND edges ----
        self.ff1_node = nn.ModuleDict()
        self.ff2_node = nn.ModuleDict()
        for node_type in metadata[0]:
            self.ff1_node[node_type] = nn.Linear(dim_h, dim_h * 2)
            self.ff2_node[node_type] = nn.Linear(dim_h * 2, dim_h)
        self.ff1_edge = nn.ModuleDict()
        self.ff2_edge = nn.ModuleDict()
        for edge_type in metadata[1]:
            key = "__".join(edge_type)
            self.ff1_edge[key] = nn.Linear(dim_h, dim_h * 2)
            self.ff2_edge[key] = nn.Linear(dim_h * 2, dim_h)

        self.dropout_global = nn.Dropout(GT["dropout"])
        self.dropout_attn = nn.Dropout(GT["attn_dropout"])
        self.ff_dropout1 = nn.Dropout(GT["dropout"])
        self.ff_dropout2 = nn.Dropout(GT["dropout"])

    def forward(self, h_node, edge_index_dict, edge_attr_dict):
        """
        h_node: {node_type: [N, dim_h]}   edge_index_dict: {edge_type: [2, E]}
        edge_attr_dict: {edge_type: [E, dim_h]}
        Returns updated (h_node, edge_attr_dict) — nodes AND edges evolve.
        """
        h_in = {nt: h for nt, h in h_node.items()}
        edge_in = {et: ea for et, ea in edge_attr_dict.items()}

        # ---- Pre-normalization ----
        h = {nt: self.norm1_global[nt](h_node[nt]) for nt in h_node}
        ea = {et: self.norm1_edge_global["__".join(et)](edge_attr_dict[et])
              for et in edge_attr_dict}

        # ---- projections per type ----
        q = {nt: self.q_lin[nt](h[nt]) for nt in h}
        k = {nt: self.k_lin[nt](h[nt]) for nt in h}
        v = {nt: self.v_lin[nt](h[nt]) for nt in h}
        e_bias = {et: self.e_lin["__".join(et)](ea[et]) for et in ea}
        e_gate = {et: self.g_lin["__".join(et)](ea[et]) for et in ea}

        # ---- Sparse edge attention (official gt_layer.py) ----
        # [FIX-SCAT] nodes share local indexing across relations — no offset
        H, D = self.H, self.D
        L = next(iter(h.values())).shape[0]

        qs, ks, vs, biases, gates, dsts, Ls = [], [], [], [], [], [], []
        for et, edge_index in edge_index_dict.items():
            src, dst = edge_index
            n_e = edge_index.shape[1]
            qs.append(q[et[2]][dst].view(-1, H, D).transpose(0, 1))
            ks.append(k[et[0]][src].view(-1, H, D).transpose(0, 1))
            vs.append(v[et[0]][src].view(-1, H, D).transpose(0, 1))
            biases.append(e_bias[et].view(-1, H, D).transpose(0, 1))
            gates.append(e_gate[et].view(-1, H, D).transpose(0, 1))
            dsts.append(dst)
            Ls.append(n_e)

        edge_q = torch.cat(qs, dim=1)     # [H, E_total, D]
        edge_k = torch.cat(ks, dim=1)
        edge_v = torch.cat(vs, dim=1)
        edge_b = torch.cat(biases, dim=1)
        edge_g = torch.cat(gates, dim=1)
        dst_flat = torch.cat(dsts)
        E_total = edge_q.shape[1]

        # official: scores = (q ⊙ k + bias), scaled, clamped [-5, 5]
        scores = edge_q * edge_k + edge_b
        scores = scores.sum(dim=-1) / math.sqrt(D)
        scores = torch.clamp(scores, min=-5, max=5)

        # official message gate: v ← v ⊙ sigmoid(gate)
        vals = edge_v * torch.sigmoid(edge_g)

        # ---- scatter softmax (official) ----
        expanded_dst = dst_flat.repeat(H, 1)
        max_scores, _ = scatter_max(scores, expanded_dst, dim=1, dim_size=L)
        max_scores = max_scores.gather(1, expanded_dst)
        exp_scores = torch.exp(scores - max_scores)
        sum_exp = torch.zeros((H, L), device=scores.device)
        sum_exp.scatter_add_(1, expanded_dst, exp_scores)
        attn = exp_scores / sum_exp.gather(1, expanded_dst)
        attn = attn.unsqueeze(-1)
        attn = self.dropout_attn(attn)

        out = torch.zeros((H, L, D), device=edge_q.device)
        out.scatter_add_(1, dst_flat.unsqueeze(-1).expand(H, E_total, D),
                         attn * vals)
        out = out.transpose(0, 1).contiguous().view(-1, H * D)

        # node outputs (per-type O projection)
        h_out = {}
        for nt in h_node:
            h_out[nt] = self.dropout_global(self.o_lin[nt](out))

        # edge outputs: (q ⊙ k + bias) pass through oe_lin (official)
        edge_out_flat = (edge_q * edge_k + edge_b)
        edge_out_flat = edge_out_flat.transpose(0, 1).contiguous().view(-1, H * D)
        ea_out = {}
        e_offset = 0
        for et, n_e in zip(edge_index_dict.keys(), Ls):
            key = "__".join(et)
            sl = slice(e_offset, e_offset + n_e)
            ea_out[et] = self.oe_lin[key](edge_out_flat[sl])
            e_offset += n_e

        # ---- Fixed residual (nodes AND edges) ----
        h_res = {nt: h_out[nt] + h_in[nt] for nt in h_node}
        ea_res = {et: ea_out[et] + edge_in[et] for et in edge_attr_dict}

        # ---- Type-FFN block (pre-norm + residual) ----
        h_ff = {nt: self.norm2_ffn[nt](h_res[nt]) for nt in h_node}
        h_final = {}
        for nt in h_node:
            ff = self.ff_dropout2(self.ff2_node[nt](
                self.act(self.ff_dropout1(self.ff1_node[nt](h_ff[nt])))))
            h_final[nt] = h_res[nt] + ff

        ea_ff = {et: self.norm2_edge_ffn["__".join(et)](ea_res[et])
                 for et in edge_attr_dict}
        ea_final = {}
        for et in edge_attr_dict:
            key = "__".join(et)
            ff = self.ff_dropout2(self.ff2_edge[key](
                self.act(self.ff_dropout1(self.ff1_edge[key](ea_ff[et])))))
            ea_final[et] = ea_res[et] + ff

        return h_final, ea_final


# ==========================================
# FraudGTModel — encoder → GT layers → Paper-Eq9 edge head
# ==========================================
class FraudGTModel(nn.Module):
    def __init__(self, node_dim_in, edge_dim_in, num_heads=GT["attn_heads"],
                 dim_hidden=GT["dim_hidden"], num_layers=GT["layers"],
                 metadata=None):
        super().__init__()
        self.dim_h = dim_hidden
        self.metadata = metadata
        self.input_dropout = nn.Dropout(0.0)

        # ---- Hetero_Raw encoders: linear projections per type ----
        self.node_encoder = nn.ModuleDict()
        for node_type in metadata[0]:
            self.node_encoder[node_type] = nn.Linear(node_dim_in, dim_hidden)
        self.edge_encoder = nn.ModuleDict()
        for edge_type in metadata[1]:
            self.edge_encoder["__".join(edge_type)] = nn.Linear(edge_dim_in, dim_hidden)

        # ---- GT layers ----
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(FraudGTLayer(dim_hidden, num_heads, metadata, i))

        # [P-EQ9] Edge prediction head — Paper Eq. 9:
        #   y_ij = σ(MLP(h_i ∥ e'_ij ∥ h_j))
        self.head = nn.Sequential(
            nn.Linear(dim_hidden * 3, dim_hidden), nn.GELU(),
            nn.Dropout(GT["dropout"]),
            nn.Linear(dim_hidden, 1))

    def forward(self, x_dict, edge_index_dict, edge_attr_dict,
                edge_label_index, edge_label_local_ids):
        """
        edge_label_index: [2, E_eval] — target edges (LOCAL node indices).
        edge_label_local_ids: [E_eval] — LOCAL positions within the batch's
        'to' edge list (== nonzero(mask) — aligned with loader re-indexing).
        Returns logits [E_eval].
        """
        h = {nt: self.node_encoder[nt](x_dict[nt]) for nt in x_dict}
        h = {nt: self.input_dropout(v) for nt, v in h.items()}
        ea = {et: self.edge_encoder["__".join(et)](edge_attr_dict[et])
              for et in edge_attr_dict}

        for layer in self.layers:
            h, ea = layer(h, edge_index_dict, ea)

        # [P-EQ9] head: concat(h_src, e'_ij, h_dst) → MLP → logit
        rel = ('node', 'to', 'node')
        src, dst = edge_label_index
        edge_emb = ea[rel][edge_label_local_ids]
        feats = torch.cat([h['node'][src], edge_emb, h['node'][dst]], dim=-1)
        logits = self.head(feats).squeeze(-1)
        return logits


# ==========================================
# Main
# ==========================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", choices=list(FEATURE_SETS), default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--fanout", default="50,50")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=GT["dim_hidden"])
    ap.add_argument("--heads", type=int, default=GT["attn_heads"])
    ap.add_argument("--lr", type=float, default=OPTIM["base_lr"])
    ap.add_argument("--accum", type=int, default=OPTIM["batch_accumulation"])
    ap.add_argument("--iters-per-epoch", type=int, default=TRAIN_ITERS_PER_EPOCH)
    ap.add_argument("--ports-tds", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--research", action="store_true")
    ap.add_argument("--load-checkpoint", default=None)
    args = ap.parse_args()

    fs = args.feature_set or "gnn_L0"
    seed = args.seed
    fanout = [int(x) for x in args.fanout.split(",")]

    run_dir = WINNER_DIR
    ckpt_path = os.path.join(run_dir, "model.pt")
    if args.load_checkpoint:
        mode = "inference"
        ckpt_path = args.load_checkpoint
        assert os.path.exists(ckpt_path), f"checkpoint not found: {ckpt_path}"
    elif os.path.exists(ckpt_path) and not args.research:
        mode = "inference"
    else:
        mode = "train"

    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"FraudGT | features={fs} | seed={seed} | fanout={fanout} | "
                f"batch={args.batch_size}x{args.accum} | hidden={args.hidden} | "
                f"heads={args.heads} | lr={args.lr} | iters/epoch={args.iters_per_epoch} | "
                f"ports_tds={args.ports_tds} | mode={mode} | {device}")
    logger.info(f"[Config] epochs≤{args.epochs} | patience={PATIENCE} | "
                f"loss_weight={LOSS_WEIGHT} | warmup={OPTIM['warmup_epochs']} | "
                f"param_groups ✓ | selection: BEST F1 over thresholds")

    # ==========================================
    # Column-by-column loading
    # ==========================================
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
    del small
    gc.collect()

    X = np.empty((n_rows, len(feature_cols)), dtype=np.float32)
    mu_arr = np.zeros(len(feature_cols), dtype=np.float32)
    sig_arr = np.ones(len(feature_cols), dtype=np.float32)
    n_bad = 0
    for j, c in enumerate(feature_cols):
        col = pf.read(columns=[c]).column(0).to_numpy(zero_copy_only=False)
        bad = ~np.isfinite(col) | (np.abs(col) > F32_MAX)
        if bad.any():
            n_bad += int(bad.sum())
            col = col.copy()
            col[bad] = np.nan
        mu_j = float(np.nanmean(col[tr_pd]))
        sig_j = float(np.nanstd(col[tr_pd]))
        if not np.isfinite(mu_j):
            mu_j = 0.0
        if not np.isfinite(sig_j) or sig_j <= 0:
            sig_j = 1.0
        mu_arr[j], sig_arr[j] = mu_j, sig_j
        col32 = col.astype(np.float32)
        col32 -= np.float32(mu_j)
        col32 /= np.float32(sig_j)
        np.clip(col32, -10.0, 10.0, out=col32)
        col32 = np.nan_to_num(col32, nan=0.0, copy=False)
        X[:, j] = col32
        del col, col32
    logger.info(f"[Clean] inf/too-large → NaN: {n_bad:,}")
    logger.info(f"[Mem] X = {X.nbytes / 1e9:.1f} GB ✓")

    edge_index = torch.load(f"{DATA_DIR}/edge_index.pt")
    y_all = torch.load(f"{DATA_DIR}/y.pt")
    masks = torch.load(f"{DATA_DIR}/masks.pt")
    num_nodes = int(edge_index.max().item()) + 1
    y_lookup = y_all.numpy()
    logger.info(f"[Load] {n_rows:,} edges | {num_nodes:,} nodes")

    tr, va, te = masks['train'], masks['val'], masks['test']
    tr_np, va_np, te_np = tr.numpy(), va.numpy(), te.numpy()
    n_train, n_val, n_test = int(tr_np.sum()), int(va_np.sum()), int(te_np.sum())

    # ==========================================
    # Ports + TDS (official, vectorized)
    # ==========================================
    if args.ports_tds:
        logger.info("[Ports+TDS] reading raw timestamps ...")
        ts_df = pq.ParquetFile(f"{DATA_DIR}/edge_metadata.parquet").read(
            columns=["Timestamp"]).to_pandas()
        ts_all_np = (ts_df["Timestamp"].astype("int64") // 10**9).to_numpy().astype(np.int64)
        assert len(ts_all_np) == n_rows
        del ts_df
        gc.collect()

    def compute_official_ports_tds(src, dst, ts):
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

    # ==========================================
    # Build hetero graphs
    # ==========================================
    def build_hetero(e_idx_t, X_slice, y_t, tx_t, mask_name, target_mask):
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
        return data

    train_graph = build_hetero(edge_index[:, tr].contiguous(), X[:n_train],
                               y_all[tr], torch.from_numpy(txid[:n_train]),
                               "train_mask", torch.ones(n_train, dtype=torch.bool))
    va_comb = tr | va
    val_graph = build_hetero(edge_index[:, va_comb].contiguous(), X[:n_train + n_val],
                             y_all[va_comb], torch.from_numpy(txid[:n_train + n_val]),
                             "val_mask", va[va_comb])
    test_graph = build_hetero(edge_index.contiguous(), X, y_all,
                              torch.from_numpy(txid), "test_mask", te)
    logger.info("[Graphs] train / val / test ✓ (hetero + direction flag)")
    gc.collect()

    if args.ports_tds:
        def add_ports_tds(graph, name):
            t0 = time.time()
            store = graph['node', 'to', 'node']
            eids = store.edge_id.numpy()
            src = store.edge_index[0].numpy()
            dst = store.edge_index[1].numpy()
            ts = ts_all_np[eids]
            in_p, out_p, in_td, out_td = compute_official_ports_tds(src, dst, ts)
            extra_fwd = torch.from_numpy(np.stack([in_p, out_p, in_td, out_td], axis=1))
            extra_rev = extra_fwd[:, [0, 1, 3, 2]].contiguous()
            store.edge_attr = torch.cat([store.edge_attr, extra_fwd], dim=1)
            graph['node', 'rev_to', 'node'].edge_attr = torch.cat(
                [graph['node', 'rev_to', 'node'].edge_attr, extra_rev], dim=1)
            logger.info(f"[Ports+TDS] {name}: {time.time()-t0:.0f}s "
                        f"(edge_attr → {store.edge_attr.shape[1]})")
        add_ports_tds(train_graph, "train")
        add_ports_tds(val_graph, "val")
        add_ports_tds(test_graph, "test")
        tr_attr = train_graph['node', 'to', 'node'].edge_attr
        mu_new = tr_attr[:, -4:].mean(0)
        sd_new = tr_attr[:, -4:].std(0).clamp(min=1)
        for g in (train_graph, val_graph, test_graph):
            for rel in ('to', 'rev_to'):
                a = g['node', rel, 'node'].edge_attr
                a[:, -4:] = (a[:, -4:] - mu_new) / sd_new
        logger.info("[Ports+TDS] normalized with train stats ✓")
        gc.collect()

    e_dim = train_graph['node', 'to', 'node'].edge_attr.shape[1]
    metadata = [('node',), (('node', 'to', 'node'), ('node', 'rev_to', 'node'))]
    logger.info(f"[Model] edge_dim = {e_dim} "
                f"({len(feature_cols)} base + 1 direction"
                + (" + 4 ports/tds" if args.ports_tds else "") + ")")

    # ==========================================
    # Loaders — full-coverage seeding
    # ==========================================
    nn_dict = {('node', 'to', 'node'): fanout, ('node', 'rev_to', 'node'): fanout}

    class AddEgoIds:
        def __call__(self, batch):
            node_store = batch['node']
            if hasattr(node_store, 'batch_size') and node_store.batch_size is not None:
                ego = torch.zeros(node_store.num_nodes, 1, dtype=torch.float)
                ego[:node_store.batch_size] = 1.0
                node_store.x = torch.cat([node_store.x, ego.to(node_store.x.device)], dim=1)
            return batch

    def make_eval_loader(graph, mask_name):
        store = graph['node', 'to', 'node']
        mask = getattr(store, mask_name)
        src_nodes = store.edge_index[0, mask].unique()
        dst_nodes = store.edge_index[1, mask].unique()
        all_seeds = torch.cat([src_nodes, dst_nodes]).unique()
        logger.info(f"[Loader] {mask_name}: {len(all_seeds):,} seeds")
        return NeighborLoader(graph, num_neighbors=nn_dict,
                              batch_size=args.batch_size,
                              input_nodes=('node', all_seeds),
                              shuffle=False, num_workers=4, transform=AddEgoIds())

    train_loader = NeighborLoader(train_graph, num_neighbors=nn_dict,
                                  batch_size=args.batch_size,
                                  input_nodes=('node',
                                               train_graph['node', 'to', 'node'].edge_index[0].unique()),
                                  shuffle=True, num_workers=4, transform=AddEgoIds())
    val_loader = make_eval_loader(val_graph, "val_mask")
    test_loader = make_eval_loader(test_graph, "test_mask")

    # ==========================================
    # Model + [P-OPTIM] official AdamW with parameter groups
    # ==========================================
    model = FraudGTModel(node_dim_in=2, edge_dim_in=e_dim,
                         num_heads=args.heads, dim_hidden=args.hidden,
                         num_layers=GT["layers"],
                         metadata=metadata).to(device)

    # [P-OPTIM] official semantics: NO weight decay on LayerNorm/biases
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    grouped_params = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": OPTIM["weight_decay"]},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped_params, lr=args.lr)

    # [P-LOSS] weighted CE [1,6] ≡ BCEWithLogitsLoss(pos_weight=6)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(LOSS_WEIGHT[1]).to(device))

    def warmup_lr(epoch):
        # [P-SCHED] official huggingface semantics: warmup from ~0
        w = OPTIM["warmup_epochs"]
        if epoch < w:
            return max(1e-6, epoch / max(1, w))
        T = max(args.epochs - w, 1)
        progress = (epoch - w) / T
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    # ==========================================
    # Train / evaluate — [P-REGIME] 256 iters/epoch
    # ==========================================
    def train():
        model.train()
        optimizer.zero_grad()
        total_loss, n_seen, accum, iters = 0.0, 0, 0, 0
        for batch in train_loader:
            # [P-REGIME] official train.iter_per_epoch=256
            if iters >= args.iters_per_epoch:
                break
            batch = batch.to(device)
            store = batch['node', 'to', 'node']
            m = store.train_mask
            if m.sum() == 0:
                continue
            local_pos = torch.nonzero(m, as_tuple=False).squeeze(-1).to(device)
            scores = model(batch.x_dict, batch.edge_index_dict,
                           batch.edge_attr_dict,
                           edge_label_index=store.edge_index[:, m],
                           edge_label_local_ids=local_pos)
            y = store.y[m].float()
            loss = loss_fn(scores, y)
            (loss / args.accum).backward()
            total_loss += float(loss) * int(m.sum())
            n_seen += int(m.sum())
            accum += 1
            iters += 1
            if accum % args.accum == 0:
                if OPTIM["clip_grad_norm"]:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        return total_loss / max(n_seen, 1)

    @torch.no_grad()
    def evaluate(loader, mask_name, n_passes=1):
        model.eval()
        prob_by_id = {}
        for _ in range(n_passes):
            for batch in loader:
                batch = batch.to(device)
                store = batch['node', 'to', 'node']
                m = getattr(store, mask_name)
                if m.sum() == 0:
                    continue
                local_pos = torch.nonzero(m, as_tuple=False).squeeze(-1).to(device)
                scores = model(batch.x_dict, batch.edge_index_dict,
                               batch.edge_attr_dict,
                               edge_label_index=store.edge_index[:, m],
                               edge_label_local_ids=local_pos)
                proba = torch.sigmoid(scores).cpu().numpy()
                for i, pr in zip(store.edge_id[m].cpu().numpy(), proba):
                    prob_by_id[int(i)] = float(pr)
        ids = np.array(sorted(prob_by_id))
        truths = y_lookup[ids]
        probas = np.array([prob_by_id[i] for i in ids])
        preds = (probas >= 0.5).astype(int)
        total = int(masks[mask_name.replace("_mask", "")].sum())
        return {"f1": f1_score(truths, preds, zero_division=0),
                "precision": precision_score(truths, preds, zero_division=0),
                "recall": recall_score(truths, preds, zero_division=0),
                "pr_auc": average_precision_score(truths, probas) if len(ids) else 0.0,
                "coverage": len(ids) / max(total, 1),
                "n_scored": len(ids), "n_total": total,
                "ids": ids, "truths": truths, "probas": probas}

    # ==========================================
    # Training OR inference — [FIX-BESTF1] selection
    # ==========================================
    os.makedirs(run_dir, exist_ok=True)
    best_val_f1, best_epoch, best_thr = 0.0, 0, 0.5

    if mode == "inference":
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        meta_path = os.path.join(os.path.dirname(ckpt_path), "metrics.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            best_val_f1 = meta.get("best_val_f1", -1)
            best_epoch = meta.get("best_epoch", -1)
            best_thr = meta.get("threshold", 0.5)
        logger.info(f"[Load] model weights loaded ✓ (best epoch {best_epoch}, "
                    f"val best-F1 {best_val_f1:.4f}, thr {best_thr:.4f})")
    else:
        no_improve = 0
        logger.info(f"[Train] starting — up to {args.epochs} epochs | "
                    f"patience={PATIENCE} | warmup={OPTIM['warmup_epochs']} | "
                    f"{args.iters_per_epoch} iters/epoch (official regime) | "
                    f"selection: BEST F1 over thresholds")
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            lr_now = args.lr * warmup_lr(epoch)
            for g in optimizer.param_groups:
                g["lr"] = lr_now
            loss = train()
            val = evaluate(val_loader, "val_mask", n_passes=1)

            # [FIX-BESTF1] ★ selection criterion = best achievable F1
            # (threshold-optimized) — NOT F1@0.5
            val_best_f1, val_thr = best_f1_over_thresholds(val["truths"], val["probas"])

            logger.info(f"Epoch {epoch:03d} | Loss {loss:.4f} | "
                        f"Val F1(best) {val_best_f1:.4f} @thr={val_thr:.3f} | "
                        f"(F1@0.5: {val['f1']:.4f}) | AUC {val['pr_auc']:.4f} | "
                        f"cov {val['coverage']*100:.1f}% | lr {lr_now:.5f} | "
                        f"{time.time()-t0:.0f}s")

            if val_best_f1 > best_val_f1:
                best_val_f1, best_thr, best_epoch = val_best_f1, val_thr, epoch
                no_improve = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    logger.info(f"[Early Stop] {PATIENCE} epochs without improvement — "
                                f"best epoch: {best_epoch} (val best-F1 {best_val_f1:.4f})")
                    break
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # ==========================================
    # [FIX-BESTF1] threshold = saved from the BEST epoch — no recomputation
    # ==========================================
    logger.info(f"[Threshold] using saved best-threshold: {best_thr:.4f} "
                f"(from best epoch {best_epoch}, val best-F1 {best_val_f1:.4f})")

    logger.info(f"[Test] final evaluation with {TEST_PASSES} passes ...")
    test = evaluate(test_loader, "test_mask", n_passes=TEST_PASSES)

    # النتيجة الرسمية: F1 بالعتبة المختارة من val (best epoch)
    test_preds = (test["probas"] >= best_thr).astype(int)
    test_f1_thr = f1_score(test["truths"], test_preds, zero_division=0)
    test_p_thr = precision_score(test["truths"], test_preds, zero_division=0)
    test_r_thr = recall_score(test["truths"], test_preds, zero_division=0)

    logger.info("=" * 60)
    logger.info(f"🎯 TEST [FraudGT | {fs} | seed {seed} | hidden {args.hidden} | "
                f"heads {args.heads} | ports_tds={args.ports_tds} | mode={mode} | "
                f"best epoch {best_epoch}]")
    logger.info(f"F1@0.5  : {test['f1']:.4f} | P {test['precision']:.4f} | "
                f"R {test['recall']:.4f}")
    logger.info(f"F1@thr  : {test_f1_thr:.4f} | P {test_p_thr:.4f} | "
                f"R {test_r_thr:.4f}   ← threshold={best_thr:.4f} (official)")
    logger.info(f"PR-AUC  : {test['pr_auc']:.4f}")
    logger.info(f"Coverage: {test['n_scored']:,}/{test['n_total']:,} "
                f"({test['coverage']*100:.1f}%)")
    logger.info("=" * 60)

    # ---- Saving ----
    pd.DataFrame({"transaction_id": test["ids"], "split": "test",
                  "y_true": test["truths"], "y_proba": test["probas"]}).to_parquet(
        os.path.join(run_dir, "predictions.parquet"), engine="pyarrow", index=False)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump({"model": "fraudgt", "feature_set": fs, "seed": seed,
                   "mode": mode, "fanout": fanout,
                   "batch_size": args.batch_size, "accum": args.accum,
                   "iters_per_epoch": args.iters_per_epoch,
                   "hidden": args.hidden, "heads": args.heads,
                   "ports_tds": args.ports_tds,
                   "threshold": best_thr,
                   "head": "paper_eq9_concat",
                   "optim": "adamW_param_groups_no_decay_norms",
                   "selection": "best_f1_over_thresholds",
                   "best_val_f1": best_val_f1, "best_epoch": best_epoch,
                   "test_f1_at_050": test["f1"],
                   "test_precision_at_050": test["precision"],
                   "test_recall_at_050": test["recall"],
                   "test_f1_at_thr": test_f1_thr,
                   "test_precision_at_thr": test_p_thr,
                   "test_recall_at_thr": test_r_thr,
                   "test_pr_auc": test["pr_auc"],
                   "test_coverage": test["coverage"]}, f, indent=2)
    np.savez(os.path.join(run_dir, "standardization.npz"), mu=mu_arr, sigma=sig_arr)
    logger.info(f"✅ Saved to {run_dir}/")


if __name__ == "__main__":
    main()
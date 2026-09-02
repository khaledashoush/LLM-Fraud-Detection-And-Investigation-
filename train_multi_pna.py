# -*- coding: utf-8 -*-
"""
train_multi_pna.py — Multi-PNA with official adaptations (Inference / Deployment / Research)
================================================================================
Usage:
  python train_multi_pna.py                       ← DEFAULT: INFERENCE — load winner checkpoint, evaluate only (~15 min)
  python train_multi_pna.py --train               ← DEPLOYMENT: winner config, train from scratch (~2 h)
  python train_multi_pna.py --research --feature-set gnn_L2 --hidden 64   ← RESEARCH

Modes:
  INFERENCE (default, no args):
    Loads the trained winner checkpoint and evaluates — NO training.
    Use case: everyday usage, backend, explainability, demo.
    Winner documented: test F1@thr = 0.6713 (above the published 66.48).

  DEPLOYMENT (--train): winner configuration (gnn_L0 + ports/tds, hidden=20),
    trained from scratch — for reproducibility verification.

  RESEARCH (--research): any feature-set/hidden/ports — full custom training.
"""

import os, gc, json, time, logging, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyarrow.parquet as pq
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import PNAConv, BatchNorm, Linear, to_hetero
from torch_geometric.data import HeteroData
from torch_geometric.utils import degree
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             average_precision_score, precision_recall_curve)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("train_multi_pna")

# ==========================================
# CONFIG (identical to the proven runs)
# ==========================================
DATA_DIR   = "processed_data"
RUNS_DIR   = "runs"
EPOCHS     = 100
PATIENCE   = 15
LR         = 0.0006
CE_WEIGHT  = [1.0, 7.07]
N_HIDDEN   = 20
FINAL_DROPOUT = 0.288
F32_MAX    = float(np.finfo(np.float32).max)
TEST_PASSES = 2

# ★ Deployment winner — validated 2026-08-30 (documented test F1@thr = 0.6713)
WINNER = {
    "feature_set": "gnn_L0",
    "hidden": 20,
    "fanout": "100,100",
    "batch_size": 2048,
    "ports_tds": True,
    "test_f1": 0.6713,
    "checkpoint": "runs/gnn_multi_pna_gnn_L0_seed42_ports/model.pt",
}

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


# ==========================================
# Model — official IBM architecture (per-relation via to_hetero)
# ==========================================
class PNA(nn.Module):
    def __init__(self, num_features, n_classes=2, n_hidden=20, num_gnn_layers=2,
                 edge_updates=True, edge_dim=None, final_dropout=0.288, deg=None):
        super().__init__()
        n_hidden = int((n_hidden // 5) * 5)
        self.n_hidden = n_hidden
        self.edge_updates = edge_updates

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.convs, self.emlps, self.batch_norms = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.convs.append(PNAConv(n_hidden, n_hidden, aggregators=aggregators,
                                      scalers=scalers, deg=deg, edge_dim=n_hidden,
                                      towers=5, pre_layers=1, post_layers=1,
                                      divide_input=False))
            if edge_updates:
                self.emlps.append(nn.Sequential(
                    nn.Linear(3 * n_hidden, n_hidden), nn.ReLU(),
                    nn.Linear(n_hidden, n_hidden)))
            self.batch_norms.append(BatchNorm(n_hidden))

        self.mlp = nn.Sequential(
            Linear(n_hidden * 3, 50), nn.ReLU(), nn.Dropout(final_dropout),
            Linear(50, 25), nn.ReLU(), nn.Dropout(final_dropout),
            Linear(25, n_classes))

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        for i in range(len(self.convs)):
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
            if self.edge_updates:
                edge_attr = edge_attr + self.emlps[i](
                    torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2
        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        x = torch.cat((x, edge_attr), dim=1)
        return self.mlp(x)


class AddEgoIds:
    """Hetero version — marks seed nodes of type 'node'."""
    def __call__(self, batch):
        node_store = batch['node']
        if hasattr(node_store, 'batch_size') and node_store.batch_size is not None:
            ego = torch.zeros(node_store.num_nodes, 1, dtype=torch.float)
            ego[:node_store.batch_size] = 1.0
            node_store.x = torch.cat([node_store.x, ego.to(node_store.x.device)], dim=1)
        return batch


# ==========================================
# [Ports + TDS] Official adaptations — vectorized (pandas)
# ==========================================
def compute_official_ports_tds(src, dst, ts):
    """Implements official data_util.py semantics, vectorized.

    PORTS: for edge (u,v):
      in_port  = rank of u among v's unique in-neighbors, ordered by
                 first-interaction time with v
      out_port = rank of v among u's unique out-neighbors, ordered by
                 first-interaction time with u

    TIME-DELTAS (official, causal):
      per node, edges sorted by time: td[first] = 0, td[i] = t_i - t_{i-1}
    """
    n = len(src)
    eid = np.arange(n)
    df = pd.DataFrame({'src': src, 'dst': dst, 'ts': ts, 'eid': eid})

    # ---- in-ports (group by dst) ----
    d = df.sort_values(['dst', 'ts'], kind='mergesort')
    d['is_first'] = ~d.duplicated(['dst', 'src'], keep='first')
    d['cum'] = d.groupby('dst', sort=False)['is_first'].cumsum() - 1
    d['port'] = d.groupby(['dst', 'src'], sort=False)['cum'].transform('first')
    in_ports = np.zeros(n, dtype=np.float32)
    in_ports[d['eid'].to_numpy()] = d['port'].to_numpy(dtype=np.float32)
    del d

    # ---- out-ports (group by src) ----
    d = df.sort_values(['src', 'ts'], kind='mergesort')
    d['is_first'] = ~d.duplicated(['src', 'dst'], keep='first')
    d['cum'] = d.groupby('src', sort=False)['is_first'].cumsum() - 1
    d['port'] = d.groupby(['src', 'dst'], sort=False)['cum'].transform('first')
    out_ports = np.zeros(n, dtype=np.float32)
    out_ports[d['eid'].to_numpy()] = d['port'].to_numpy(dtype=np.float32)
    del d

    # ---- in-tds (per dst, previous-gap, first = 0) ----
    d = df.sort_values(['dst', 'ts'], kind='mergesort')
    d['td'] = d.groupby('dst', sort=False)['ts'].diff().fillna(0.0)
    in_tds = np.zeros(n, dtype=np.float32)
    in_tds[d['eid'].to_numpy()] = d['td'].to_numpy(dtype=np.float32)
    del d

    # ---- out-tds (per src) ----
    d = df.sort_values(['src', 'ts'], kind='mergesort')
    d['td'] = d.groupby('src', sort=False)['ts'].diff().fillna(0.0)
    out_tds = np.zeros(n, dtype=np.float32)
    out_tds[d['eid'].to_numpy()] = d['td'].to_numpy(dtype=np.float32)
    del d
    del df
    gc.collect()

    return in_ports, out_ports, in_tds, out_tds


# ==========================================
# Main
# ==========================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", choices=list(FEATURE_SETS), default=None,
                    help="default = winner (gnn_L0). Required with --research.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--fanout", default=WINNER["fanout"])
    ap.add_argument("--batch-size", type=int, default=WINNER["batch_size"])
    ap.add_argument("--hidden", type=int, default=WINNER["hidden"])
    ap.add_argument("--ports-tds", action=argparse.BooleanOptionalAction,
                    default=WINNER["ports_tds"],
                    help="Official ports + time-deltas adaptations (default ON = winner)")
    ap.add_argument("--research", action="store_true",
                    help="Custom configuration — full training from scratch.")
    ap.add_argument("--train", action="store_true",
                    help="Train the winner config from scratch (deployment "
                         "reproduction). Default WITHOUT this flag = load "
                         "winner checkpoint and evaluate only.")
    ap.add_argument("--load-checkpoint", default=None,
                    help="Override: use a specific checkpoint path instead of "
                         "the winner default (still inference mode).")
    args = ap.parse_args()

    # ===== Mode resolution =====
    if args.research:
        assert args.feature_set, "--feature-set is REQUIRED with --research"
        fs = args.feature_set
        mode = "research"
        ckpt_path = None
        logger.info(f"[Mode] RESEARCH — custom configuration on {fs}")
    elif args.train:
        fs = args.feature_set or WINNER["feature_set"]
        mode = "deployment"
        ckpt_path = None
        logger.info(f"[Mode] DEPLOYMENT — winner config, training from scratch "
                    f"(documented test F1@thr = {WINNER['test_f1']})")
    else:
        # DEFAULT = INFERENCE on the winner checkpoint
        ckpt_path = args.load_checkpoint or WINNER["checkpoint"]
        assert os.path.exists(ckpt_path), (
            f"Checkpoint not found: {ckpt_path}\n"
            f"Hint: train first with  --train  (or check the path)")
        fs = args.feature_set or WINNER["feature_set"]
        mode = "inference"
        logger.info(f"[Mode] INFERENCE — loading winner checkpoint: {ckpt_path} "
                    f"(no training, evaluation only — documented 0.6713)")

    seed = args.seed
    fanout = [int(x) for x in args.fanout.split(",")]
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Multi-PNA | features={fs} | seed={seed} | fanout={fanout} | "
                f"batch={args.batch_size} | hidden={args.hidden} | "
                f"ports_tds={args.ports_tds} | mode={mode} | {device}")
    logger.info(f"[Config] epochs≤{args.epochs} | patience={PATIENCE} | test passes={TEST_PASSES}")

    # ==========================================
    # Column-by-column loading (proven memory path)
    # ==========================================
    parquet_path = f"{DATA_DIR}/features_all.parquet"
    pf = pq.ParquetFile(parquet_path)
    n_rows = pf.metadata.num_rows
    schema_names = pf.schema_arrow.names
    feature_cols = get_feature_cols(schema_names, fs)
    logger.info(f"[Load] {fs}: {len(feature_cols)} features | {n_rows:,} rows (column-streaming)")

    small = pf.read(columns=["split", "transaction_id"]).to_pandas()
    split_np = small["split"].astype(str).to_numpy()
    txid = small["transaction_id"].to_numpy(dtype=np.int64)
    tr_pd = split_np == "train"
    del small
    gc.collect()

    X = np.empty((n_rows, len(feature_cols)), dtype=np.float32)
    mu_arr = np.zeros(len(feature_cols), dtype=np.float32)
    sig_arr = np.ones(len(feature_cols), dtype=np.float32)
    n_bad_total = 0

    for j, c in enumerate(feature_cols):
        col = pf.read(columns=[c]).column(0).to_numpy(zero_copy_only=False)
        bad = ~np.isfinite(col) | (np.abs(col) > F32_MAX)
        if bad.any():
            n_bad_total += int(bad.sum())
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
    logger.info(f"[Clean] inf/too-large values → NaN: {n_bad_total:,}")
    logger.info(f"[Mem] X = {X.nbytes / 1e9:.1f} GB — loading peak done ✓")

    # ==========================================
    # Graph components
    # ==========================================
    edge_index = torch.load(f"{DATA_DIR}/edge_index.pt")
    y_all = torch.load(f"{DATA_DIR}/y.pt")
    masks = torch.load(f"{DATA_DIR}/masks.pt")
    num_nodes = int(edge_index.max().item()) + 1
    assert np.array_equal(txid, np.arange(n_rows)), "transaction_id != row index!"
    y_lookup = y_all.numpy()
    logger.info(f"[Load] {n_rows:,} edges | {num_nodes:,} nodes")

    tr, va, te = masks['train'], masks['val'], masks['test']
    tr_np, va_np, te_np = tr.numpy(), va.numpy(), te.numpy()
    n_train, n_val, n_test = int(tr_np.sum()), int(va_np.sum()), int(te_np.sum())

    contiguous = (tr_np[:n_train].all() and not tr_np[n_train:].any()
                  and va_np[n_train:n_train + n_val].all()
                  and not va_np[:n_train].any() and not va_np[n_train + n_val:].any()
                  and te_np[n_train + n_val:].all() and not te_np[:n_train + n_val:].any())
    assert contiguous, "Splits are not contiguous!"
    logger.info(f"[Graphs] splits contiguous ✓ (train {n_train:,} | val {n_val:,} | "
                f"test {n_test:,})")

    # ---- deg histogram (fixed) from FULL train graph ----
    d = degree(edge_index[1, tr], num_nodes=num_nodes, dtype=torch.long)
    deg = torch.bincount(d, minlength=1).float()
    logger.info(f"[Deg] histogram from FULL train graph | length={len(deg)} "
                f"| max_degree={int(d.max())}")

    # ==========================================
    # RAW timestamps (for official ports/tds)
    # ==========================================
    if args.ports_tds:
        logger.info("[Ports+TDS] reading raw timestamps from edge_metadata ...")
        ts_df = pq.ParquetFile(f"{DATA_DIR}/edge_metadata.parquet").read(
            columns=["Timestamp"]).to_pandas()
        ts_all_np = (ts_df["Timestamp"].astype("int64") // 10**9).to_numpy().astype(np.int64)
        assert len(ts_all_np) == n_rows, "edge_metadata rows != features rows!"
        del ts_df
        gc.collect()

    # ==========================================
    # Build the three HeteroData graphs (+ direction flag)
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

    # ==========================================
    # [Ports + TDS] — per graph, official semantics, train-stats normalization
    # ==========================================
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
            # Official create_hetero_obj swaps the LAST TWO columns for reverse
            # edges (the two tds columns) — mirrored here:
            extra_rev = extra_fwd[:, [0, 1, 3, 2]].contiguous()
            store.edge_attr = torch.cat([store.edge_attr, extra_fwd], dim=1)
            graph['node', 'rev_to', 'node'].edge_attr = torch.cat(
                [graph['node', 'rev_to', 'node'].edge_attr, extra_rev], dim=1)
            logger.info(f"[Ports+TDS] {name}: done in {time.time()-t0:.0f}s "
                        f"(edge_attr → {store.edge_attr.shape[1]} cols)")

        add_ports_tds(train_graph, "train")
        add_ports_tds(val_graph, "val")
        add_ports_tds(test_graph, "test")

        # Normalize the 4 new columns with TRAIN statistics only
        tr_attr = train_graph['node', 'to', 'node'].edge_attr
        mu_new = tr_attr[:, -4:].mean(0)
        sd_new = tr_attr[:, -4:].std(0).clamp(min=1)
        for g in (train_graph, val_graph, test_graph):
            for rel in ('to', 'rev_to'):
                a = g['node', rel, 'node'].edge_attr
                a[:, -4:] = (a[:, -4:] - mu_new) / sd_new
        logger.info(f"[Ports+TDS] normalized with train stats "
                    f"(mu={mu_new.tolist()}, sd={sd_new.tolist()})")
        gc.collect()

    e_dim = train_graph['node', 'to', 'node'].edge_attr.shape[1]
    logger.info(f"[Model] edge_dim = {e_dim} "
                f"({len(feature_cols)} base + 1 direction"
                + (" + 4 ports/tds" if args.ports_tds else "") + ")")

    # ==========================================
    # Loaders — full-coverage seeding
    # ==========================================
    nn_dict = {('node', 'to', 'node'): fanout, ('node', 'rev_to', 'node'): fanout}

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
    # Model — to_hetero on both relations
    # ==========================================
    base = PNA(num_features=2, edge_dim=e_dim, n_hidden=args.hidden,
               final_dropout=FINAL_DROPOUT, deg=deg)
    model = to_hetero(base, test_graph.metadata(), aggr='mean').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                           factor=0.5, patience=5)
    loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor(CE_WEIGHT).to(device))

    def train():
        model.train(); total_loss = 0; n_seen = 0
        for batch in train_loader:
            batch = batch.to(device)
            store = batch['node', 'to', 'node']
            m = store.train_mask
            if m.sum() == 0: continue
            optimizer.zero_grad()
            out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            loss = loss_fn(out['node', 'to', 'node'][m], store.y[m].long())
            loss.backward(); optimizer.step()
            total_loss += float(loss) * int(m.sum()); n_seen += int(m.sum())
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
                if m.sum() == 0: continue
                out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
                proba = torch.softmax(out['node', 'to', 'node'][m], dim=1)[:, 1].cpu().numpy()
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
    # Run directory
    # ==========================================
    run_dir = os.path.join(RUNS_DIR,
                           f"gnn_multi_pna_{fs}_seed{seed}"
                           + (f"_h{args.hidden}" if args.hidden != N_HIDDEN else "")
                           + ("_ports" if args.ports_tds else "")
                           + ({"deployment": "_deploy", "inference": "_inference"}
                              .get(mode, "")))
    os.makedirs(run_dir, exist_ok=True)

    # ==========================================
    # Training OR checkpoint loading
    # ==========================================
    best_val_f1, best_epoch = 0.0, 0

    if mode == "inference":
        # INFERENCE: the checkpoint IS the model — no training
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        meta_path = os.path.join(os.path.dirname(ckpt_path), "metrics.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            best_val_f1 = meta.get("best_val_f1", -1)
            best_epoch = meta.get("best_epoch", -1)
        logger.info(f"[Load] model weights loaded ✓ "
                    f"(documented: best epoch {best_epoch}, val F1 {best_val_f1})")
        ckpt = ckpt_path
    else:
        # DEPLOYMENT / RESEARCH: full training with early stopping
        ckpt = os.path.join(run_dir, "model.pt")
        no_improve = 0
        logger.info(f"[Train] starting — up to {args.epochs} epochs | patience={PATIENCE}")
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            loss = train()
            val = evaluate(val_loader, "val_mask", n_passes=1)
            scheduler.step(val["f1"])
            logger.info(f"Epoch {epoch:03d} | Loss {loss:.4f} | Val F1 {val['f1']:.4f} "
                        f"| P {val['precision']:.4f} | R {val['recall']:.4f} "
                        f"| AUC {val['pr_auc']:.4f} | cov {val['coverage']*100:.1f}% "
                        f"| {time.time()-t0:.0f}s")
            if val["f1"] > best_val_f1:
                best_val_f1, best_epoch = val["f1"], epoch
                no_improve = 0
                torch.save(model.state_dict(), ckpt)
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    logger.info(f"[Early Stop] {PATIENCE} epochs without improvement — "
                                f"best epoch: {best_epoch}")
                    break
        model.load_state_dict(torch.load(ckpt, map_location=device))

    # ==========================================
    # Threshold tuning on val
    # ==========================================
    logger.info("[Threshold] tuning on val ...")
    val_final = evaluate(val_loader, "val_mask", n_passes=1)
    prec, rec, thr = precision_recall_curve(val_final["truths"], val_final["probas"])
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    best_thr = float(thr[np.argmax(f1s[:-1])])
    logger.info(f"[Threshold] {best_thr:.4f} | Val F1@thr = {np.max(f1s[:-1]):.4f} "
                f"(was @0.5: {val_final['f1']:.4f})")

    # ---- Final test evaluation (once) ----
    logger.info(f"[Test] final evaluation with {TEST_PASSES} passes — please wait...")
    test = evaluate(test_loader, "test_mask", n_passes=TEST_PASSES)

    test_preds_thr = (test["probas"] >= best_thr).astype(int)
    test_f1_thr   = f1_score(test["truths"], test_preds_thr, zero_division=0)
    test_p_thr    = precision_score(test["truths"], test_preds_thr, zero_division=0)
    test_r_thr    = recall_score(test["truths"], test_preds_thr, zero_division=0)

    logger.info("=" * 60)
    logger.info(f"🎯 TEST [Multi-PNA | {fs} | seed {seed} | hidden {args.hidden} | "
                f"ports_tds={args.ports_tds} | mode={mode} | best epoch {best_epoch}]")
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
        json.dump({"model": "multi_pna", "feature_set": fs, "seed": seed,
                   "mode": mode,
                   "fanout": fanout, "batch_size": args.batch_size,
                   "hidden": args.hidden, "ports_tds": args.ports_tds,
                   "test_passes": TEST_PASSES, "threshold": best_thr,
                   "best_val_f1": best_val_f1, "best_epoch": best_epoch,
                   "test_f1_at_050": test["f1"],
                   "test_precision_at_050": test["precision"],
                   "test_recall_at_050": test["recall"],
                   "test_f1_at_thr": test_f1_thr,
                   "test_precision_at_thr": test_p_thr,
                   "test_recall_at_thr": test_r_thr,
                   "test_pr_auc": test["pr_auc"],
                   "test_coverage": test["coverage"]}, f, indent=2)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"feature_set": fs, "seed": seed, "fanout": fanout,
                   "batch_size": args.batch_size, "hidden": args.hidden,
                   "ports_tds": args.ports_tds, "mode": mode,
                   "feature_cols": feature_cols}, f, indent=2)
    np.savez(os.path.join(run_dir, "standardization.npz"), mu=mu_arr, sigma=sig_arr)
    logger.info(f"✅ Saved to {run_dir}/")


if __name__ == "__main__":
    main()
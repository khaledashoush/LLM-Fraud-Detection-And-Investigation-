# -*- coding: utf-8 -*-
"""
train_pna_eu.py — PNA + Edge Updates (Inference / Deployment / Research)
=========================================================================
Usage:
  python train_pna_eu.py                          ← DEFAULT: INFERENCE — load winner checkpoint, evaluate only (~10 min)
  python train_pna_eu.py --train                  ← DEPLOYMENT: winner config, train from scratch (~50 min)
  python train_pna_eu.py --research --feature-set gnn_L2 --hidden 64   ← RESEARCH: any configuration

Modes:
  INFERENCE (default, no args):
    Loads the trained winner checkpoint and evaluates — NO training.
    Use case: everyday usage, backend, explainability, demo.
    Guarantees the EXACT documented model (same weights = same predictions).

  DEPLOYMENT (--train): winner configuration (gnn_L0, hidden=20, official
    params), trained from scratch — for reproducibility verification.

  RESEARCH (--research): any feature-set/hidden/deg — full custom training.
"""

import os, gc, json, time, logging, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyarrow.parquet as pq
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import PNAConv, BatchNorm, Linear
from torch_geometric.data import Data
from torch_geometric.utils import degree
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             average_precision_score, precision_recall_curve)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("train_pna_eu")

# ==========================================
# CONFIG (official defaults + our patience)
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

# ★ Deployment winner — validated 2026-08-30 (documented test F1@0.5 = 0.6498)
WINNER = {
    "feature_set": "gnn_L0",
    "hidden": 20,
    "fanout": "100,100",
    "batch_size": 2048,
    "test_f1": 0.6498,
    "checkpoint": "runs/gnn_pna_eu_gnn_L0_seed42/model.pt",
}

GROUP_PREFIXES = {
    "basic": ("basic_",), "gfp": ("gfp_",), "structural": ("structural_",),
    "causal": ("causal_",), "entity": ("entity_",),
}
FEATURE_SETS = {
    "gnn_L0": ["basic"],                                           # ★ winner
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
# Model — official IBM architecture
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
    """Marks seed nodes — Ego IDs from Egressy et al. adaptations."""
    def __call__(self, batch):
        if hasattr(batch, 'batch_size') and batch.batch_size is not None:
            ego = torch.zeros(batch.num_nodes, 1, dtype=torch.float)
            ego[:batch.batch_size] = 1.0
            batch.x = torch.cat([batch.x, ego.to(batch.x.device)], dim=1)
        return batch


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
    ap.add_argument("--deg-mode", choices=["full", "batch"], default="full")
    ap.add_argument("--research", action="store_true",
                    help="Custom configuration — full training from scratch.")
    ap.add_argument("--train", action="store_true",
                    help="Train the winner config from scratch (deployment "
                         "reproduction). Default WITHOUT this flag = load "
                         "winner checkpoint and evaluate only (~10 min).")
    ap.add_argument("--load-checkpoint", default=None,
                    help="Override: use a specific checkpoint path instead of "
                         "the winner default (still inference mode).")
    args = ap.parse_args()

    # ===== Mode resolution =====
    # Priority: --research > --train > default(INFERENCE on winner checkpoint)
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
                    f"(documented test F1 = {WINNER['test_f1']})")
    else:
        # DEFAULT = INFERENCE on the winner checkpoint
        ckpt_path = args.load_checkpoint or WINNER["checkpoint"]
        assert os.path.exists(ckpt_path), (
            f"Checkpoint not found: {ckpt_path}\n"
            f"Hint: train first with  --train  (or check the path)")
        fs = args.feature_set or WINNER["feature_set"]
        mode = "inference"
        logger.info(f"[Mode] INFERENCE — loading winner checkpoint: {ckpt_path} "
                    f"(no training, evaluation only)")

    seed = args.seed
    fanout = [int(x) for x in args.fanout.split(",")]
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"PNA+EU | features={fs} | seed={seed} | fanout={fanout} | "
                f"batch={args.batch_size} | hidden={args.hidden} | "
                f"deg_mode={args.deg_mode} | mode={mode} | {device}")
    logger.info(f"[Config] epochs≤{args.epochs} | patience={PATIENCE} | test passes={TEST_PASSES}")

    # ==========================================
    # [FIX-9] Column-by-column loading
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

        bad = ~np.isfinite(col) | (np.abs(col) > F32_MAX)      # [FIX-8]
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

    # [FIX-9b] Verify split contiguity → graphs as views
    contiguous = (tr_np[:n_train].all() and not tr_np[n_train:].any()
                  and va_np[n_train:n_train + n_val].all()
                  and not va_np[:n_train].any() and not va_np[n_train + n_val:].any()
                  and te_np[n_train + n_val:].all() and not te_np[:n_train + n_val:].any())
    assert contiguous, "Splits are not contiguous! Check preprocessing"
    logger.info(f"[Graphs] splits contiguous ✓ (train {n_train:,} | val {n_val:,} | "
                f"test {n_test:,}) — graphs share memory")

    # ==========================================
    # [FIX-1] + [FIX-2] Separate graphs
    # ==========================================
    def build(e_idx, e_attr, y, tx, mask_name, target_mask):
        d = Data(x=torch.ones((num_nodes, 1)), edge_index=e_idx, edge_attr=e_attr)
        d.y = y
        d.edge_id = tx
        setattr(d, mask_name, target_mask)
        return d

    va_comb = tr | va
    train_graph = build(edge_index[:, tr], torch.from_numpy(X[:n_train]),
                        y_all[tr], torch.from_numpy(txid[:n_train]),
                        "train_mask", torch.ones(n_train, dtype=torch.bool))
    val_graph = build(edge_index[:, va_comb], torch.from_numpy(X[:n_train + n_val]),
                      y_all[va_comb], torch.from_numpy(txid[:n_train + n_val]),
                      "val_mask", va[va_comb])
    test_graph = build(edge_index, torch.from_numpy(X), y_all,
                       torch.from_numpy(txid), "test_mask", te)
    logger.info("[Graphs] train / val / test ✓ (separate — zero leakage, shared memory)")
    gc.collect()

    # ==========================================
    # [UPD-4] deg histogram (official shape)
    # ==========================================
    if args.deg_mode == "full":
        d = degree(edge_index[1, tr], num_nodes=num_nodes, dtype=torch.long)
        deg = torch.bincount(d, minlength=1).float()
        logger.info(f"[Deg] histogram from FULL train graph | length={len(deg)} "
                    f"| max_degree={int(d.max())}")
    else:
        rng = np.random.default_rng(seed)
        sample_size = min(args.batch_size * 8, int(tr.sum()))
        sample_edges = rng.choice(np.flatnonzero(tr_np), size=sample_size, replace=False)
        d = degree(edge_index[1, torch.from_numpy(sample_edges)], dtype=torch.long)
        deg = torch.bincount(d, minlength=1).float()
        logger.info(f"[Deg] histogram from ONE batch of {sample_size:,} edges "
                    f"(official repo behavior) | length={len(deg)}")

    # ==========================================
    # [FIX-12] Loaders — full-coverage seeding
    # ==========================================
    def make_eval_loader(graph, mask_name):
        mask = getattr(graph, mask_name)
        src_nodes = graph.edge_index[0, mask].unique()
        dst_nodes = graph.edge_index[1, mask].unique()
        all_seeds = torch.cat([src_nodes, dst_nodes]).unique()
        logger.info(f"[Loader] {mask_name}: {len(all_seeds):,} seeds (sources + destinations)")
        return NeighborLoader(graph, num_neighbors=fanout,
                              batch_size=args.batch_size,
                              input_nodes=all_seeds,
                              shuffle=False, num_workers=4, transform=AddEgoIds())

    train_loader = NeighborLoader(train_graph, num_neighbors=fanout,
                                  batch_size=args.batch_size,
                                  input_nodes=train_graph.edge_index[0].unique(),
                                  shuffle=True, num_workers=4, transform=AddEgoIds())
    val_loader = make_eval_loader(val_graph, "val_mask")
    test_loader = make_eval_loader(test_graph, "test_mask")

    # ---- Model ----
    model = PNA(num_features=2, edge_dim=len(feature_cols), n_hidden=args.hidden,
                final_dropout=FINAL_DROPOUT, deg=deg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                           factor=0.5, patience=5)
    loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor(CE_WEIGHT).to(device))

    def train():
        model.train(); total_loss = 0; n_seen = 0
        for batch in train_loader:
            batch = batch.to(device)
            m = batch.train_mask
            if m.sum() == 0: continue
            optimizer.zero_grad()
            loss = loss_fn(model(batch.x, batch.edge_index, batch.edge_attr)[m],
                           batch.y[m].long())
            loss.backward(); optimizer.step()
            total_loss += float(loss) * int(m.sum()); n_seen += int(m.sum())
        return total_loss / max(n_seen, 1)

    # ==========================================
    # [FIX-10] Multi-pass evaluation
    # ==========================================
    @torch.no_grad()
    def evaluate(loader, mask_name, n_passes=1):
        model.eval()
        prob_by_id = {}
        for _ in range(n_passes):
            for batch in loader:
                batch = batch.to(device)
                m = getattr(batch, mask_name)
                if m.sum() == 0: continue
                out = model(batch.x, batch.edge_index, batch.edge_attr)[m]
                proba = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                for i, pr in zip(batch.edge_id[m].cpu().numpy(), proba):
                    prob_by_id[int(i)] = float(pr)        # dedupe — last wins
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
                           f"gnn_pna_eu_{fs}_seed{seed}"
                           + (f"_h{args.hidden}" if args.hidden != N_HIDDEN else "")
                           + (f"_{args.deg_mode}deg" if args.deg_mode != "full" else "")
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
    # [FIX-13] Threshold tuning on val
    # ==========================================
    logger.info("[Threshold] tuning on val ...")
    val_final = evaluate(val_loader, "val_mask", n_passes=1)
    prec, rec, thr = precision_recall_curve(val_final["truths"], val_final["probas"])
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    best_thr = float(thr[np.argmax(f1s[:-1])])
    logger.info(f"[Threshold] {best_thr:.4f} | Val F1@thr = {np.max(f1s[:-1]):.4f} "
                f"(was @0.5: {val_final['f1']:.4f})")

    # ---- Final test evaluation ----
    logger.info(f"[Test] final evaluation with {TEST_PASSES} passes — please wait...")
    test = evaluate(test_loader, "test_mask", n_passes=TEST_PASSES)

    test_preds_thr = (test["probas"] >= best_thr).astype(int)
    test_f1_thr   = f1_score(test["truths"], test_preds_thr, zero_division=0)
    test_p_thr    = precision_score(test["truths"], test_preds_thr, zero_division=0)
    test_r_thr    = recall_score(test["truths"], test_preds_thr, zero_division=0)

    logger.info("=" * 60)
    logger.info(f"🎯 TEST [PNA+EU | {fs} | seed {seed} | hidden {args.hidden} | "
                f"deg {args.deg_mode} | mode {mode} | best epoch {best_epoch}]")
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
        json.dump({"model": "pna_eu", "feature_set": fs, "seed": seed,
                   "mode": mode,
                   "fanout": fanout, "batch_size": args.batch_size,
                   "hidden": args.hidden, "deg_mode": args.deg_mode,
                   "test_passes": TEST_PASSES,
                   "threshold": best_thr,
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
                   "deg_mode": args.deg_mode, "mode": mode,
                   "feature_cols": feature_cols}, f, indent=2)
    np.savez(os.path.join(run_dir, "standardization.npz"), mu=mu_arr, sigma=sig_arr)
    logger.info(f"✅ Saved to {run_dir}/")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
train_xgb.py — XGBoost + GFP (Research + Deployment modes)
============================================================
Usage:
  python train_xgb.py                                    ← DEPLOYMENT: winner (xgb_L2, saved params, ~25 min)
  python train_xgb.py --research --feature-set xgb_L1    ← RESEARCH: full Optuna (50 trials)
  python train_xgb.py --research --feature-set xgb_L0    ← paper-faithful anchor

Modes:
  DEPLOYMENT (default, no args):
    Loads the validated best params from runs/xgb_xgb_L2_seed42/metrics.json
    (the 50-trial Optuna study already completed) and trains the final model
    directly — NO re-tuning. Documented result: test F1@thr = 0.7501.

  RESEARCH (--research):
    Full Optuna search on the chosen feature-set (as before).
    Note: --feature-set is required with --research.
"""

import os, sys, json, time, logging, argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             average_precision_score, precision_recall_curve)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler("train_xgb.log")])
logger = logging.getLogger("train_xgb")
optuna.logging.set_verbosity(optuna.logging.INFO)

# ==========================================
# CONFIG
# ==========================================
DATA_PATH = "processed_data/features_all.parquet"
RUNS_DIR = "runs"
N_TRIALS = 50
F32_MAX = float(np.finfo(np.float32).max)
N_JOBS = 128

# ★ Deployment winner — validated 2026-08-29 (50-trial Optuna, test F1@thr = 0.7501)
WINNER_PARAMS_PATH = "runs/xgb_xgb_L2_seed42/metrics.json"
WINNER_FEATURE_SET = "xgb_L2"
WINNER_TEST_F1 = 0.7501

# ==========================================
# Feature sets — same definitions as the Preprocessing
# ==========================================
GROUP_PREFIXES = {
    "basic": ("basic_",), "gfp": ("gfp_",), "structural": ("structural_",),
    "causal": ("causal_",), "entity": ("entity_",),
}
FEATURE_SETS = {
    "xgb_L0": ["basic", "gfp"],                                    # paper-faithful (GFP paper)
    "xgb_L1": ["basic", "gfp", "structural"],
    "xgb_L2": ["basic", "gfp", "structural", "causal"],            # ★ winner
    "xgb_L3": ["basic", "gfp", "structural", "causal", "entity"],  # perfect-KYC
}


def get_feature_cols(all_cols, feature_set):
    prefixes = tuple(p for g in FEATURE_SETS[feature_set] for p in GROUP_PREFIXES[g])
    cols = [c for c in all_cols if c.startswith(prefixes)]
    assert "transaction_id" not in cols, "LEAK: transaction_id!"
    assert "Is Laundering" not in cols and "split" not in cols, "LEAK!"
    return cols


# ==========================================
# Search spaces
# ==========================================
def suggest_params(trial, feature_set):
    if feature_set == "xgb_L0":
        # GFP paper ranges (Table 3) — paper-faithful for the 65.70 comparison
        return {
            "max_depth": trial.suggest_int("max_depth", 1, 15),
            "learning_rate": trial.suggest_float("learning_rate", 10**-2.5, 10**-1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 10**-2, 10**2, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
    else:
        # Our tuned ranges (from previous experiments)
        return {
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 50.0, 900.0, log=True),
        }


# ==========================================
# [FIX-inf v2] clean values invalid for float32
# ==========================================
def clean_inf(df):
    """Any inf or |x| > float32max → NaN.
    XGBoost works in float32 — values above 3.4e38 (var/skew/kurtosis on huge
    amounts) are fine in float64 but explode after conversion. NaN → missing."""
    all_feat = [c for c in df.columns
                if c.startswith(("basic_", "gfp_", "structural_", "causal_", "entity_"))]
    bad_cols = []
    for c in all_feat:
        x = df[c].to_numpy()
        m = np.isinf(x) | (np.abs(x) > F32_MAX)
        if m.any():
            bad_cols.append((c, int(m.sum())))
            df.loc[m, c] = np.nan
    if bad_cols:
        total = sum(n for _, n in bad_cols)
        logger.warning(f"[Clean] {total:,} values (inf or >float32max={F32_MAX:.2e}) in "
                       f"{len(bad_cols)} columns → converted to NaN")
        logger.warning(f"[Clean] affected columns: {bad_cols[:10]}")
    else:
        logger.info("[Clean] no inf or too-large values ✓")
    return df


# ==========================================
# Helpers
# ==========================================
def predict_proba_in_batches(model, X, batch_size=1_000_000):
    """X = numpy float32 array. XGBoost handles NaN automatically."""
    probas = []
    for i in range(0, len(X), batch_size):
        probas.append(model.predict_proba(X[i:i + batch_size])[:, 1])
    return np.concatenate(probas)


def eval_at_threshold(y_true, proba, thr):
    pred = (proba >= thr).astype(int)
    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


# ==========================================
# Main
# ==========================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", choices=list(FEATURE_SETS), default=None,
                    help="default = deployment winner (xgb_L2). Required with --research")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--research", action="store_true",
                    help="Full Optuna tuning (deployment skips tuning entirely)")
    args = ap.parse_args()

    # ===== Mode resolution =====
    research = args.research
    if research:
        assert args.feature_set, "--feature-set is REQUIRED with --research"
        fs = args.feature_set
        logger.info(f"[Mode] RESEARCH — full Optuna {args.trials} trials on {fs}")
    else:
        fs = args.feature_set or WINNER_FEATURE_SET
        if fs == WINNER_FEATURE_SET and os.path.exists(WINNER_PARAMS_PATH):
            logger.info(f"[Mode] DEPLOYMENT — validated winner {WINNER_FEATURE_SET} "
                        f"(documented test F1@thr = {WINNER_TEST_F1}) — NO re-tuning")
        else:
            research = True   # cannot deploy without saved params → tune
            logger.info(f"[Mode] RESEARCH (auto) — no saved params for '{fs}', "
                        f"falling back to full Optuna")
    seed = args.seed

    np.random.seed(seed)
    run_dir = os.path.join(RUNS_DIR, f"xgb_{fs}_seed{seed}"
                           + ("" if research else "_deploy"))
    os.makedirs(run_dir, exist_ok=True)

    # ---- 1) Load + clean + convert to float32 once ----
    t0 = time.time()
    logger.info(f"[Load] {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    df = clean_inf(df)

    feature_cols = get_feature_cols(df.columns, fs)
    logger.info(f"[Load] {fs}: {len(feature_cols)} features | {len(df):,} rows")

    X_all = df[feature_cols].to_numpy(dtype=np.float32)
    assert not np.isinf(X_all).any(), "still inf after cleaning!"
    y_all = df["Is Laundering"].to_numpy()
    split_all = df["split"].astype(str).to_numpy()
    txid_all = df["transaction_id"].to_numpy()

    tr = split_all == "train"
    va = split_all == "val"
    te = split_all == "test"
    X_train, y_train = X_all[tr], y_all[tr]
    X_val, y_val = X_all[va], y_all[va]
    logger.info(f"[Load] train {tr.sum():,} | val {va.sum():,} | test {te.sum():,}")
    logger.info(f"[Load] illicit: train {y_train.sum():,} | val {y_val.sum():,} | "
                f"test {y_all[te].sum():,}")
    logger.info(f"[Mem] X_all = {X_all.nbytes / 1e9:.1f} GB (RAM) — CPU mode, n_jobs={N_JOBS}")
    del df
    import gc; gc.collect()

    # ---- 2) Params: saved (deployment) or Optuna (research) ----
    if not research:
        with open(WINNER_PARAMS_PATH) as f:
            saved = json.load(f)
        best_params = saved["best_params"]
        best_val_f1 = saved.get("val_f1_at_thr", 0.0)
        logger.info(f"[Params] loaded from {WINNER_PARAMS_PATH} "
                    f"(tuning val F1 = {best_val_f1:.4f}) — skipping Optuna")
    else:
        logger.info(f"[Tune] Optuna {args.trials} trials (seed={seed}) | CPU x{N_JOBS}")

        def objective(trial):
            params = suggest_params(trial, fs)
            model = xgb.XGBClassifier(
                verbosity=0, objective="binary:logistic", eval_metric="aucpr",
                tree_method="hist", device="cpu", n_jobs=N_JOBS,
                random_state=seed,
                n_estimators=1000, early_stopping_rounds=50, **params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            proba = model.predict_proba(X_val)[:, 1]
            prec, rec, thr = precision_recall_curve(y_val, proba)
            f1s = 2 * prec * rec / (prec + rec + 1e-8)
            return float(np.max(f1s[:-1]))          # off-by-one fix

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=args.trials)
        best_params = study.best_trial.params
        best_val_f1 = study.best_trial.value
        logger.info(f"[Tune] best val F1: {best_val_f1:.4f}")
        logger.info(f"[Tune] best params: {best_params}")

    # ---- 3) Final model ----
    final_params = {
        **best_params,
        "objective": "binary:logistic", "eval_metric": "aucpr",
        "tree_method": "hist", "device": "cpu", "n_jobs": N_JOBS,
        "random_state": seed,
        "n_estimators": 3000, "early_stopping_rounds": 50, "verbosity": 0,
    }
    logger.info("[Train] final model...")
    model = xgb.XGBClassifier(**final_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    logger.info(f"[Train] best_iteration: {model.best_iteration}")

    # ---- 4) Threshold from val ----
    y_val_proba = model.predict_proba(X_val)[:, 1]
    prec, rec, thr = precision_recall_curve(y_val, y_val_proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    best_thr = float(thr[np.argmax(f1s[:-1])])
    logger.info(f"[Threshold] {best_thr:.4f} (from val)")

    # ---- 5) Evaluate all splits ----
    logger.info("[Eval] predicting (in batches)...")
    proba_all = predict_proba_in_batches(model, X_all)
    df_out = pd.DataFrame({
        "transaction_id": txid_all,
        "split": split_all,
        "y_true": y_all,
        "y_proba": proba_all,
    })

    metrics = {"feature_set": fs, "seed": seed,
               "mode": "deployment" if not research else "research",
               "n_features": len(feature_cols),
               "threshold": best_thr, "best_iteration": int(model.best_iteration),
               "best_params": {k: (float(v) if isinstance(v, (int, float)) else v)
                               for k, v in best_params.items()},
               "tuning_time_min": round((time.time() - t0) / 60, 1)}
    logger.info("\n" + "=" * 70)
    logger.info(f"{'':16s} {'F1@0.5':>8s} {'F1@thr':>8s} {'Prec':>8s} {'Rec':>8s} {'PR-AUC':>8s}")
    for s in ["train", "val", "test"]:
        sub = df_out[df_out["split"] == s]
        m0 = eval_at_threshold(sub["y_true"], sub["y_proba"], 0.5)
        mt = eval_at_threshold(sub["y_true"], sub["y_proba"], best_thr)
        auc = float(average_precision_score(sub["y_true"], sub["y_proba"]))
        metrics[f"{s}_f1_at_050"] = m0["f1"]
        metrics[f"{s}_f1_at_thr"] = mt["f1"]
        metrics[f"{s}_precision_at_thr"] = mt["precision"]
        metrics[f"{s}_recall_at_thr"] = mt["recall"]
        metrics[f"{s}_pr_auc"] = auc
        logger.info(f"{s:16s} {m0['f1']:8.4f} {mt['f1']:8.4f} "
                    f"{mt['precision']:8.4f} {mt['recall']:8.4f} {auc:8.4f}")
    logger.info("=" * 70)

    gap = metrics["val_f1_at_thr"] - metrics["test_f1_at_thr"]
    logger.info(f"[Diag] val−test F1 gap: {gap:+.4f} "
                f"({'OK' if abs(gap) < 0.05 else 'WARNING: large gap'})")

    # ---- 6) Save ----
    model.get_booster().save_model(os.path.join(run_dir, "model.ubj"))
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"feature_set": fs, "seed": seed,
                   "mode": "research" if research else "deployment",
                   "trials": args.trials if research else None,
                   "device": "cpu", "n_jobs": N_JOBS,
                   "feature_cols": feature_cols}, f, indent=2)
    df_out.to_parquet(os.path.join(run_dir, "predictions.parquet"), engine="pyarrow", index=False)

    # ---- 7) Feature importance (Top 20) ----
    imp = model.get_booster().get_score(importance_type="gain")
    imp_df = pd.DataFrame.from_dict(imp, orient="index", columns=["gain"]).sort_values(
        "gain", ascending=False).head(20)
    imp_df.to_csv(os.path.join(run_dir, "feature_importance_top20.csv"))
    logger.info("[Save] ✓ run directory: " + run_dir)
    logger.info(f"✅ Completed in {(time.time() - t0) / 60:.1f} min")
    logger.info("\n[Importance] Top 20:\n" + imp_df.to_string())


if __name__ == "__main__":
    main()
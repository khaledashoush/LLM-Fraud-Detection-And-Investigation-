# -*- coding: utf-8 -*-
# finalize_xgb_L2.py — تدريب نهائي بالـ best params من اللوج (من غير إعادة الـ 50 trials)
import os, json, time, logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, precision_recall_curve

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("finalize")

DATA_PATH = "processed_data/features_all.parquet"
RUNS_DIR = "runs"
F32_MAX = float(np.finfo(np.float32).max)

# === Best params من Trial 22 (best val F1 = 0.6891) ===
BEST_PARAMS = {
    "max_depth": 8,
    "learning_rate": 0.013006273094522035,
    "gamma": 0.3127343644435275,
    "min_child_weight": 5,
    "subsample": 0.7635231412791496,
    "colsample_bytree": 0.9270239075471184,
    "reg_lambda": 0.008880128363757263,
    "scale_pos_weight": 50.03441381949553,
}

GROUP_PREFIXES = {
    "basic": ("basic_",), "gfp": ("gfp_",), "structural": ("structural_",),
    "causal": ("causal_",), "entity": ("entity_",),
}
FEATURE_SETS = {
    "xgb_L0": ["basic", "gfp"],
    "xgb_L1": ["basic", "gfp", "structural"],
    "xgb_L2": ["basic", "gfp", "structural", "causal"],
    "xgb_L3": ["basic", "gfp", "structural", "causal", "entity"],
}


def get_feature_cols(all_cols, feature_set):
    prefixes = tuple(p for g in FEATURE_SETS[feature_set] for p in GROUP_PREFIXES[g])
    cols = [c for c in all_cols if c.startswith(prefixes)]
    assert "transaction_id" not in cols, "LEAK"
    return cols


def clean_inf(df):
    all_feat = [c for c in df.columns
                if c.startswith(("basic_", "gfp_", "structural_", "causal_", "entity_"))]
    for c in all_feat:
        x = df[c].to_numpy()
        m = np.isinf(x) | (np.abs(x) > F32_MAX)
        if m.any():
            df.loc[m, c] = np.nan
    return df


def predict_proba_in_batches(model, X, batch_size=1_000_000):
    probas = []
    for i in range(0, len(X), batch_size):
        probas.append(model.predict_proba(X[i:i + batch_size])[:, 1])
    return np.concatenate(probas)


def eval_at_threshold(y_true, proba, thr):
    pred = (proba >= thr).astype(int)
    return {"precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0))}


def main():
    fs, seed = "xgb_L2", 42
    t0 = time.time()

    # 1) Load + clean
    logger.info("[Load] ...")
    df = pd.read_parquet(DATA_PATH)
    df = clean_inf(df)
    feature_cols = get_feature_cols(df.columns, fs)

    X_all = df[feature_cols].to_numpy(dtype=np.float32)
    y_all = df["Is Laundering"].to_numpy()
    split_all = df["split"].astype(str).to_numpy()
    txid_all = df["transaction_id"].to_numpy()
    del df
    import gc; gc.collect()

    tr = split_all == "train"
    va = split_all == "val"
    X_train, y_train = X_all[tr], y_all[tr]
    X_val, y_val = X_all[va], y_all[va]

    # 2) Final model — بالـ best params مباشرة (بدون Optuna)
    logger.info("[Train] final model (best params from 50-trial Optuna) ...")
    final_params = {
        **BEST_PARAMS,
        "objective": "binary:logistic", "eval_metric": "aucpr",
        "tree_method": "hist", "device": "cpu", "n_jobs": 128,
        "random_state": seed,
        "n_estimators": 3000, "early_stopping_rounds": 50, "verbosity": 0,
    }
    model = xgb.XGBClassifier(**final_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    logger.info(f"[Train] best_iteration: {model.best_iteration}")

    # 3) Threshold من val
    y_val_proba = model.predict_proba(X_val)[:, 1]
    prec, rec, thr = precision_recall_curve(y_val, y_val_proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    best_thr = float(thr[np.argmax(f1s[:-1])])
    logger.info(f"[Threshold] {best_thr:.4f}")

    # 4) Eval
    logger.info("[Eval] predicting ...")
    proba_all = predict_proba_in_batches(model, X_all)
    df_out = pd.DataFrame({"transaction_id": txid_all, "split": split_all,
                           "y_true": y_all, "y_proba": proba_all})

    metrics = {"feature_set": fs, "seed": seed, "n_features": len(feature_cols),
               "threshold": best_thr, "best_iteration": int(model.best_iteration),
               "best_params": BEST_PARAMS, "note": "finalized from 50-trial Optuna study"}

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
    logger.info(f"[Diag] val−test F1 gap: {gap:+.4f}")

    # 5) Save
    run_dir = os.path.join(RUNS_DIR, f"xgb_{fs}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    model.get_booster().save_model(os.path.join(run_dir, "model.ubj"))
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"feature_set": fs, "seed": seed, "device": "cpu",
                   "n_jobs": 128, "feature_cols": feature_cols}, f, indent=2)
    df_out.to_parquet(os.path.join(run_dir, "predictions.parquet"), engine="pyarrow", index=False)

    imp = model.get_booster().get_score(importance_type="gain")
    imp_df = pd.DataFrame.from_dict(imp, orient="index", columns=["gain"]).sort_values(
        "gain", ascending=False).head(20)
    imp_df.to_csv(os.path.join(run_dir, "feature_importance_top20.csv"))
    logger.info(f"✅ Completed in {(time.time() - t0) / 60:.1f} min")
    logger.info("\n[Importance] Top 20:\n" + imp_df.to_string())


if __name__ == "__main__":
    main()
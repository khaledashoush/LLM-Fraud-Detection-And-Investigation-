# -*- coding: utf-8 -*-
"""
Ultimate SOTA AML Pipeline (Robust Saving Logic)
Fixed: PyTorch tensors saved individually to prevent Zip I/O errors. 
"""

import os
import sys
import gc
import logging
import contextlib
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta

# ==========================================
# Setup Logging
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# Utility: OS-Level Output Suppression
# ==========================================
@contextlib.contextmanager
def suppress_c_output():
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(devnull_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)

# ==========================================
# Phase 0: Setup & Data Loading
# ==========================================
DATA_DIR = "/home/jovyan/Ahmed Magdi work/data"
TRANS_FILE = "HI-Medium_Trans.csv"
ACCOUNTS_FILE = "HI-Medium_accounts.csv"

TRANS_PATH = os.path.join(DATA_DIR, TRANS_FILE)
ACCOUNTS_PATH = os.path.join(DATA_DIR, ACCOUNTS_FILE)

if not os.path.exists(TRANS_PATH):
    logger.warning(f"Real data not found. Falling back to synthetic data.")
    RUNNING_ON_REAL_DATA = False
    rng = np.random.default_rng(42)
    BASE_TIME = datetime(2022, 9, 1, 0, 0)
    rows, acc_rows = [], []
    def add_row(m, fb, a, tb, a1, amt, fmt="ACH", il=0):
        ts = BASE_TIME + timedelta(minutes=int(m))
        rows.append({"Timestamp": ts.strftime("%Y/%m/%d %H:%M"), "From Bank": fb, "Account": a, "To Bank": tb, "Account.1": a1, "Amount Received": amt, "Receiving Currency": "US Dollar", "Amount Paid": amt, "Payment Currency": "US Dollar", "Payment Format": fmt, "Is Laundering": il})
    for _ in range(400):
        src, dst = rng.choice([f"ACC{a:05d}" for a in range(1, 61)], size=2, replace=False)
        add_row(rng.integers(0, 20000), "010", src, "010", dst, float(rng.lognormal(6, 1.2)))
    add_row(18000, "010", "CYC_A", "010", "CYC_B", 9000, il=1)
    add_row(18020, "010", "CYC_B", "010", "CYC_C", 8800, il=1)
    add_row(18040, "010", "CYC_C", "010", "CYC_A", 8600, il=1)
    for i, inter in enumerate(["SG_I1", "SG_I2", "SG_I3"]):
        add_row(12000 + i*10, "020", "SG_SRC", "020", inter, 5000, il=1)
        add_row(12000 + i*10 + 90, "020", inter, "020", "SG_SINK", 4950, il=1)
    for acc in [f"ACC{a:05d}" for a in range(1, 61)] + ["CYC_A", "CYC_B", "CYC_C", "SG_SRC", "SG_I1", "SG_I2", "SG_I3", "SG_SINK"]:
        acc_rows.append({"Bank ID": "10", "Account Number": acc, "Entity ID": f"ENT{rng.integers(1, 20)}"})
    pd.DataFrame(rows).sample(frac=1, random_state=1).to_csv("synthetic_trans.csv", index=False)
    pd.DataFrame(acc_rows).to_csv("synthetic_acc.csv", index=False)
    TRANS_PATH = "synthetic_trans.csv"
    ACCOUNTS_PATH = "synthetic_acc.csv"
else:
    RUNNING_ON_REAL_DATA = True

RAW_DTYPES = {
    "From Bank": "category", "Account": "string", "To Bank": "category",
    "Account.1": "string", "Amount Received": "float64", "Receiving Currency": "category",
    "Amount Paid": "float64", "Payment Currency": "category", "Payment Format": "category",
}

def load_raw(path):
    df = pd.read_csv(path, dtype={k: v for k, v in RAW_DTYPES.items()})
    df["Is Laundering"] = df["Is Laundering"].astype("int8")
    logger.info(f"[Load] {len(df):,} transactions loaded.")
    return df

# ==========================================
# Phase 1: Cleaning & Node ID Construction
# ==========================================
def clean_and_build_node_ids(df):
    df = df.dropna(subset=["Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Amount Paid"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
    df["src_bank_norm"] = df["From Bank"].astype(str).str.lstrip('0').replace('', '0')
    df["dst_bank_norm"] = df["To Bank"].astype(str).str.lstrip('0').replace('', '0')
    df["src_node"] = df["src_bank_norm"] + "_" + df["Account"].astype(str)
    df["dst_node"] = df["dst_bank_norm"] + "_" + df["Account.1"].astype(str)
    df["is_self_loop"] = (df["src_node"] == df["dst_node"])
    df["src_node"] = df["src_node"].astype("category")
    df["dst_node"] = df["dst_node"].astype("category")
    df = df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    df["transaction_id"] = df.index.astype("int32")
    return df

def build_global_node_index(df):
    all_nodes = pd.unique(pd.concat([df["src_node"], df["dst_node"]], ignore_index=True))
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    n_nodes = len(all_nodes)
    df["src_idx"] = df["src_node"].map(node_to_idx).astype("int32")
    df["dst_idx"] = df["dst_node"].map(node_to_idx).astype("int32")
    return df, node_to_idx, n_nodes

# ==========================================
# Phase 1.5: Entity Enhancement
# ==========================================
def enhance_with_entities(df):
    if not os.path.exists(ACCOUNTS_PATH):
        df["src_entity"], df["dst_entity"], df["entity_is_same"] = "-1", "-1", False
        return df
        
    df_acc = pd.read_csv(ACCOUNTS_PATH)
    bank_col = 'Bank ID' if 'Bank ID' in df_acc.columns else next((c for c in df_acc.columns if 'bank' in c.lower()), None)
    acc_col = 'Account Number' if 'Account Number' in df_acc.columns else next((c for c in df_acc.columns if 'account' in c.lower()), None)
    ent_col = 'Entity ID' if 'Entity ID' in df_acc.columns else next((c for c in df_acc.columns if 'entity' in c.lower()), None)
    
    if bank_col and acc_col and ent_col:
        df_acc["Bank_Account"] = df_acc[bank_col].astype(str).str.lstrip('0').replace('', '0') + "_" + df_acc[acc_col].astype(str)
        ent_map = dict(zip(df_acc["Bank_Account"], df_acc[ent_col].astype(str)))
        df["src_entity"] = df["src_node"].astype(str).map(ent_map).fillna("-1").astype("string")
        df["dst_entity"] = df["dst_node"].astype(str).map(ent_map).fillna("-1").astype("string")
        df["entity_is_same"] = (df["src_entity"] == df["dst_entity"]) & (df["src_entity"] != "-1")
        
        match_rate = (df["src_entity"] != "-1").mean() * 100
        logger.info(f"[Entity] Join Match Rate: {match_rate:.2f}% mapped.")
        logger.info(f"[Entity] Found {df['entity_is_same'].sum():,} same-entity transfers.")
    else:
        logger.warning("[Entity] Could not identify columns. Skipping.")
        df["src_entity"], df["dst_entity"], df["entity_is_same"] = "-1", "-1", False
    return df

# ==========================================
# Phase 2: Temporal Split
# ==========================================
def temporal_split(df, split_frac=(0.6, 0.2, 0.2)):
    day_idx = (df["Timestamp"] - df["Timestamp"].min()).dt.days
    n_days = int(day_idx.max()) + 1
    daily_counts = np.bincount(day_idx, minlength=n_days)
    cum = np.concatenate([[0], np.cumsum(daily_counts)])
    
    best_err, best_i, best_j = None, 0, n_days
    for i in range(n_days + 1):
        for j in range(i, n_days + 1):
            props = [(cum[i]-cum[0])/len(df), (cum[j]-cum[i])/len(df), (cum[n_days]-cum[j])/len(df)]
            err = max(abs(p - t) / t if t > 0 else 0.0 for p, t in zip(props, split_frac))
            if best_err is None or err < best_err: best_err, best_i, best_j = err, i, j
            
    split_labels = np.where(day_idx < best_i, "train", np.where(day_idx < best_j, "val", "test"))
    df["split"] = pd.Categorical(split_labels, categories=["train", "val", "test"])
    logger.info(f"[Split] Train: {best_i} days, Val: {best_j-best_i} days, Test: {n_days-best_j} days.")
    return df

# ==========================================
# Phase 3: Time Conversion
# ==========================================
def to_unix_seconds(ts_series):
    return ts_series.astype("datetime64[s]").astype("int64").to_numpy().astype("int32")

# ==========================================
# Phase 4: Basic Edge Features
# ==========================================
FORBIDDEN = {"src_idx", "dst_idx", "src_node", "dst_node", "transaction_id", "Account", "Account.1", "From Bank", "To Bank", "src_bank_norm", "dst_bank_norm", "src_entity", "dst_entity"}
AMOUNT_EPS = 1.0

def check_forbidden(cols):
    leak = FORBIDDEN.intersection(set(cols))
    if leak: raise AssertionError(f"FORBIDDEN LEAK: {leak}")

def add_basic_features(df, train_df):
    df["basic_log_amount_paid"] = np.log1p(df["Amount Paid"])
    df["basic_log_amount_received"] = np.log1p(df["Amount Received"])
    df["basic_amount_ratio"] = df["Amount Received"] / (df["Amount Paid"] + AMOUNT_EPS)
    hour = df["Timestamp"].dt.hour + df["Timestamp"].dt.minute / 60.0
    df["basic_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["basic_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    dow = df["Timestamp"].dt.dayofweek
    df["basic_dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["basic_dow_cos"] = np.cos(2 * np.pi * dow / 7)
    for c in ["Receiving Currency", "Payment Currency", "Payment Format"]:
        cats = sorted(train_df[c].astype(str).unique())
        mapping = {v: i for i, v in enumerate(cats)}
        df[f"basic_{c.lower().replace(' ', '_')}_code"] = df[c].astype(str).map(mapping).fillna(-1).astype("int32")
    return df

def add_port_numbers_and_deltas(df):
    fo = df.drop_duplicates(["src_idx", "dst_idx"], keep="first")[["src_idx", "dst_idx"]].copy()
    fo["structural_port_out"] = fo.groupby("src_idx").cumcount()
    df = df.merge(fo, on=["src_idx", "dst_idx"], how="left")
    fi = df.drop_duplicates(["dst_idx", "src_idx"], keep="first")[["dst_idx", "src_idx"]].copy()
    fi["structural_port_in"] = fi.groupby("dst_idx").cumcount()
    df = df.merge(fi, on=["dst_idx", "src_idx"], how="left")
    td_out = df.sort_values(["src_idx", "Timestamp"]).groupby("src_idx")["Timestamp"].diff().dt.total_seconds()
    df["structural_td_out"] = td_out.fillna(-1.0).values
    df["basic_is_first_out_tx"] = df["structural_td_out"].isna()
    df["structural_td_out"] = df["structural_td_out"].fillna(-1.0)
    td_in = df.sort_values(["dst_idx", "Timestamp"]).groupby("dst_idx")["Timestamp"].diff().dt.total_seconds()
    df["structural_td_in"] = td_in.fillna(-1.0).values
    df["basic_is_first_in_tx"] = df["structural_td_in"].isna()
    df["structural_td_in"] = df["structural_td_in"].fillna(-1.0)
    return df

# ==========================================
# Phase 5: Node Features (Strict-Causal Priors)
# ==========================================
def add_node_priors(df):
    out_s = df.sort_values(["src_idx", "Timestamp"]).copy()
    is_new = ~out_s.duplicated(["src_idx", "dst_idx"], keep="first")
    grp = out_s.groupby("src_idx")
    out_s["basic_src_out_degree"] = grp.cumcount()
    out_s["basic_src_out_fan"] = is_new.groupby(out_s["src_idx"]).cumsum() - is_new.astype(int)
    out_s["basic_src_out_amt_sum"] = grp["Amount Paid"].cumsum() - out_s["Amount Paid"]
    df = df.join(out_s[["basic_src_out_degree", "basic_src_out_fan", "basic_src_out_amt_sum"]])
    
    in_s = df.sort_values(["dst_idx", "Timestamp"]).copy()
    is_new = ~in_s.duplicated(["dst_idx", "src_idx"], keep="first")
    grp = in_s.groupby("dst_idx")
    in_s["basic_dst_in_degree"] = grp.cumcount()
    in_s["basic_dst_in_fan"] = is_new.groupby(in_s["dst_idx"]).cumsum() - is_new.astype(int)
    in_s["basic_dst_in_amt_sum"] = grp["Amount Received"].cumsum() - in_s["Amount Received"]
    df = df.join(in_s[["basic_dst_in_degree", "basic_dst_in_fan", "basic_dst_in_amt_sum"]])
    return df

# ==========================================
# Phase 6: Pattern Mining (Full Features)
# ==========================================
def run_pattern_mining(df, unix_ts):
    from snapml import GraphFeaturePreprocessor
    
    logger.info("[SnapML] Initializing Graph Feature Preprocessor (Max Performance)...")
    gfp = GraphFeaturePreprocessor()
    
    correct_params = {
        "fan": True,
        "vertex_stats": True,
        "degree": True,
        "scatter-gather": True,
        "scatter-gather_tw": 6 * 3600,
        "temp-cycle": True,
        "temp-cycle_tw": 24 * 3600,
        "lc-cycle": True,
        "lc-cycle_tw": 24 * 3600,
        "lc-cycle_len": 10,
        "fan_tw": 24 * 3600
    }
    gfp.set_params(correct_params)
    
    gfp_input = pd.DataFrame({
        "Edge ID": df["transaction_id"].astype("int64"),
        "Source": df["src_idx"].astype("int64"),
        "Target": df["dst_idx"].astype("int64"),
        "Timestamp": unix_ts,
        "Amount Received": df["Amount Received"].astype("float32"),
        "Receiving Currency": df["basic_receiving_currency_code"].astype("int32"),
        "Amount Paid": df["Amount Paid"].astype("float32"),
        "Payment Currency": df["basic_payment_currency_code"].astype("int32"),
        "Payment Format": df["basic_payment_format_code"].astype("int32")
    })

    train_mask = df["split"] == "train"
    val_mask = df["split"] == "val"
    test_mask = df["split"] == "test"
    
    BATCH_SIZE = 1024
    
    def batch_transform(input_df, split_name):
        n_edges = len(input_df)
        first_batch = input_df.iloc[0:BATCH_SIZE]
        with suppress_c_output():
            first_features = gfp.transform(first_batch)
        n_features = first_features.shape[1]
        
        logger.info(f"[SnapML] Detected {n_features} features per edge. Allocating memory...")
        res = np.zeros((n_edges, n_features), dtype=np.float32)
        res[0:len(first_features)] = first_features
        del first_features
        gc.collect()
        
        for i in range(BATCH_SIZE, n_edges, BATCH_SIZE):
            batch = input_df.iloc[i:i+BATCH_SIZE]
            with suppress_c_output():
                batch_features = gfp.transform(batch)
            actual_batch_size = batch_features.shape[0]
            res[i:i+actual_batch_size] = batch_features
            del batch_features
            
            batch_num = i // BATCH_SIZE
            if batch_num % 2000 == 0 and i > 0:
                logger.info(f"  [{split_name}] Processed {i:,}/{n_edges:,} edges...")
                gc.collect()
                
        logger.info(f"  [{split_name}] Processed {n_edges:,}/{n_edges:,} edges. Done.")
        return res
    
    logger.info("[SnapML] Extracting features for Train (Batch Size=1024)...")
    train_features = batch_transform(gfp_input[train_mask], "Train")
    gc.collect()
    
    logger.info("[SnapML] Extracting features for Val (Batch Size=1024)...")
    val_features = batch_transform(gfp_input[val_mask], "Val")
    gc.collect()
    
    logger.info("[SnapML] Extracting features for Test (Batch Size=1024)...")
    test_features = batch_transform(gfp_input[test_mask], "Test")
    gc.collect()
    
    feature_names = [f"gfp_feature_{i}" for i in range(train_features.shape[1])]
    
    features_df = pd.DataFrame(
        np.concatenate([train_features, val_features, test_features]),
        columns=feature_names, 
        index=df.index
    )
    
    del train_features, val_features, test_features, gfp_input
    gc.collect()
    
    df = pd.concat([df, features_df], axis=1)
    logger.info("[SnapML] Pattern mining completed successfully!")
    return df

# ==========================================
# Phase 7: Assembly & Output (Robust Component Saving)
# ==========================================
FEATURE_PREFIXES = ("basic_", "gfp_", "structural_", "entity_")

def downcast_dtypes(df):
    for c in df.columns:
        if df[c].dtype == np.float64: df[c] = df[c].astype(np.float32)
        elif df[c].dtype == np.int64:
            if df[c].min() >= np.iinfo(np.int32).min and df[c].max() <= np.iinfo(np.int32).max:
                df[c] = df[c].astype(np.int32)
    return df

def assemble_outputs(df, n_nodes):
    feature_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]
    check_forbidden(feature_cols)
    
    logger.info(f"[Assembly] Final Feature Matrix Shape: ({len(df):,}, {len(feature_cols)})")
    logger.info(f"Features extracted: {len(feature_cols)}")
    
    out_dir = "processed_data"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Save XGBoost Table directly without unnecessary copies
    logger.info("Saving XGBoost data...")
    xgb_cols = feature_cols + ["Is Laundering", "split"]
    try:
        df[xgb_cols].to_parquet(os.path.join(out_dir, "xgboost_features.parquet"), engine='pyarrow', index=False)
    except Exception as e:
        logger.warning(f"Parquet failed ({e}), falling back to CSV...")
        df[xgb_cols].to_csv(os.path.join(out_dir, "xgboost_features.csv"), index=False)
    logger.info(f"Saved XGBoost data to {out_dir}")
    
    # 2. Save Graph Components Individually (Prevents Zip I/O Errors)
    try:
        import torch
        
        logger.info("Extracting arrays for PyTorch (Zero-Copy)...")
        # Convert directly to numpy arrays
        edge_attr_np = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
        y_np = df["Is Laundering"].to_numpy(dtype=np.int64)
        src_idx_np = df["src_idx"].to_numpy(dtype=np.int64)
        dst_idx_np = df["dst_idx"].to_numpy(dtype=np.int64)
        split_np = df["split"].to_numpy()
        
        # Delete the massive dataframe immediately to free RAM
        del df
        gc.collect()
        
        logger.info("Creating PyTorch Tensors...")
        edge_attr = torch.from_numpy(edge_attr_np)
        y_tensor = torch.from_numpy(y_np)
        src_idx = torch.from_numpy(src_idx_np)
        dst_idx = torch.from_numpy(dst_idx_np)
        
        train_mask = torch.from_numpy((split_np == "train").astype(bool))
        val_mask = torch.from_numpy((split_np == "val").astype(bool))
        test_mask = torch.from_numpy((split_np == "test").astype(bool))
        
        edge_index = torch.vstack([src_idx, dst_idx])
        
        # FIX: Save components individually to avoid Zip container errors on huge files
        logger.info("Saving edge_index.pt...")
        torch.save(edge_index, os.path.join(out_dir, "edge_index.pt"))
        
        logger.info("Saving edge_attr.pt...")
        torch.save(edge_attr, os.path.join(out_dir, "edge_attr.pt"))
        
        logger.info("Saving y.pt...")
        torch.save(y_tensor, os.path.join(out_dir, "y.pt"))
        
        logger.info("Saving masks.pt...")
        torch.save({"train": train_mask, "val": val_mask, "test": test_mask}, os.path.join(out_dir, "masks.pt"))
        
        logger.info(f"Saved PyG components successfully to {out_dir}")
        logger.info("NOTE: For HeteroData (Multi-PNA), edge_attr will be cloned and swapped dynamically during training.")
        
    except ImportError:
        logger.warning("Torch not installed. Skipped GNN data saving.")

# ==========================================
# Execution Pipeline
# ==========================================
if __name__ == "__main__":
    logger.info("Starting Ultimate SOTA AML Pipeline (16 Cores / 128GB RAM)...")
    df = load_raw(TRANS_PATH)
    df = clean_and_build_node_ids(df)
    df = enhance_with_entities(df)
    df, node_to_idx, n_nodes = build_global_node_index(df)
    df = temporal_split(df)
    
    train_mask = df["split"] == "train"
    unix_ts = to_unix_seconds(df["Timestamp"])
    
    df = add_basic_features(df, df.loc[train_mask])
    df = add_port_numbers_and_deltas(df)
    df = add_node_priors(df)
    
    df = run_pattern_mining(df, unix_ts)
    
    assemble_outputs(df, n_nodes)
    logger.info("✅ Ultimate SOTA Pipeline completed successfully!")
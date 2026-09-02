# -*- coding: utf-8 -*-
"""
AML Preprocessing — Single-File FINAL v3 (HI-Medium)
=====================================================
الإصلاحات عن النسخة القديمة:
  [FIX-1] structural_td_out/in: إسناد label-aligned (كانت .values مبعثرة بعد sort)
  [FIX-2] structural_is_first_out/in: تُحسب قبل fillna (كانت ميتة دائمًا = 0%)
  [FIX-3] GFP: مدخل الأعمدة الـ 9 + params كلها ints
  [FIX-3b] vertex_stats_cols = [3, 4, 6] (Timestamp + Amounts) —
           الافتراضي كان [3] = Timestamp فقط → stats المبالغ كانت غايبة تمامًا
           (تأكد بالبصمة: 5 features ≈1.66e9 + اختبار الطيران: 264 بدل 224)
  [FIX-4] causal_* منفصلة عن basic_* (نظام الـ ablation ladder)
  [FIX-5] entity تُحسب وتُحفظ — L3 فقط يستخدمها (perfect-KYC experiment)
  [FIX-6] حفظ: features_all.parquet (superset) + node_to_idx + edge_metadata
          + feature_metadata.json — والـ standardization بيتم وقت التدريب
  [FIX-7] tie-breaking حتمي (transaction_id) في كل الـ sorts
  [FIX-8] ربط الأنماط (patterns) في نفس التشغيل — للتقييم فقط

Protocol: temporal split 60-20-20 (يومي) + GFP streaming بترتيب
          train→val→test بنفس الـ object (مطابق لبروتوكول ورقة GFP)

التشغيل:  python Preprocessing.py
الوقت المتوقع: ~3 ساعات على HI-Medium
"""

import os
import sys
import gc
import json
import time
import shutil
import logging
import contextlib
import re
from io import StringIO

import numpy as np
import pandas as pd

# ==========================================
# CONFIG — عدّل هنا بس
# ==========================================
DATA_DIR      = "/home/jovyan/Ahmed Magdi work/data"
TRANS_FILE    = "HI-Medium_Trans.csv"
ACCOUNTS_FILE = "HI-Medium_accounts.csv"     # يُحسب ويُحفظ — L3 فقط يستخدمه
PATTERNS_FILE = "HI-Medium_Patterns.txt"     # للتقييم فقط — ممنوع كـ feature
OUTPUT_DIR    = "processed_data"
COMPUTE_ENTITY = True

GFP_CONFIG = {
    "batch_size": 128,                # رقم الورقة — أعلى دقة على Medium
    "num_threads": 64,
    "buffer_batches": 4000,           # = 512K صف لكل parquet chunk
    "fan_tw": 24 * 3600,
    "degree_tw": 24 * 3600,
    "vertex_stats_tw": 24 * 3600,
    # [FIX-3b] مواضع الأعمدة في مدخل GFP بالترتيب:
    #   0=Edge ID, 1=Source, 2=Target, 3=Timestamp,
    #   4=Amount Received, 5=Receiving Currency, 6=Amount Paid, ...
    "vertex_stats_cols": [3, 4, 6],   # Timestamp + Amount Received + Amount Paid (زي الورقة)
    "scatter_gather_tw": 6 * 3600,
    "temp_cycle_tw": 24 * 3600,
    "lc_cycle_len": 10,
    "lc_cycle_tw": 24 * 3600,
}

TRANS_PATH    = os.path.join(DATA_DIR, TRANS_FILE)
ACCOUNTS_PATH = os.path.join(DATA_DIR, ACCOUNTS_FILE)
PATTERNS_PATH = os.path.join(DATA_DIR, PATTERNS_FILE)

# ==========================================
# Logging
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.FileHandler("pipeline.log"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("preprocessing")


@contextlib.contextmanager
def suppress_c_output():
    """كتم مخرجات C بتاع snapml (بدونها اللوج بيتغرق بملايين السطور)."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout, saved_stderr = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull_fd, 1); os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_stdout, 1); os.dup2(saved_stderr, 2)
        os.close(devnull_fd); os.close(saved_stdout); os.close(saved_stderr)


def to_unix_seconds(ts: pd.Series) -> np.ndarray:
    """int64 — متوافق مع كل إصدارات pandas."""
    return ((ts - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")).to_numpy().astype("int64")


# ==========================================
# FEATURE SETS — نظام الـ ablation ladder
# ==========================================
GROUP_PREFIXES = {
    "basic":      ("basic_",),
    "gfp":        ("gfp_",),
    "structural": ("structural_",),
    "causal":     ("causal_",),
    "entity":     ("entity_",),
}

FEATURE_SETS = {
    # XGBoost ladder
    "xgb_L0": ["basic", "gfp"],                                    # paper-faithful (GFP paper)
    "xgb_L1": ["basic", "gfp", "structural"],
    "xgb_L2": ["basic", "gfp", "structural", "causal"],
    "xgb_L3": ["basic", "gfp", "structural", "causal", "entity"],  # perfect-KYC
    # GNN ladder
    "gnn_L0": ["basic"],                                           # paper-faithful (Egressy)
    "gnn_L1": ["basic", "gfp"],                                    # ← fusion (مساهمتنا)
    "gnn_L2": ["basic", "gfp", "structural", "causal"],
    "gnn_L3": ["basic", "gfp", "structural", "causal", "entity"],
}

FORBIDDEN = {"transaction_id", "src_idx", "dst_idx", "Is Laundering", "split",
             "src_node", "dst_node", "src_entity", "dst_entity"}


def get_feature_cols(all_cols, feature_set):
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set '{feature_set}'. Available: {list(FEATURE_SETS)}")
    prefixes = tuple(p for g in FEATURE_SETS[feature_set] for p in GROUP_PREFIXES[g])
    cols = [c for c in all_cols if c.startswith(prefixes)]
    leaked = set(cols) & FORBIDDEN
    assert not leaked, f"LEAK DETECTED in '{feature_set}': {leaked}"
    return cols


# ==========================================
# SELF-TESTS — بتشتغل تلقائيًا قبل الـ pipeline
# ==========================================
def run_self_tests():
    logger.info("=" * 60)
    logger.info("[Tests] تشغيل الاختبارات الذاتية...")

    # ---- اختبار 1: محاذاة td_out + is_first_out (الـ bug القديم) ----
    df = pd.DataFrame({
        "src_idx": [1, 2, 2, 1], "dst_idx": [9, 8, 8, 9],
        "Timestamp": pd.to_datetime(["2022-01-01 01:00", "2022-01-01 02:00",
                                     "2022-01-01 03:00", "2022-01-01 04:00"]),
        "transaction_id": [0, 1, 2, 3],
    })
    out = add_ports_and_time_deltas(df.copy())
    assert out.loc[0, "structural_td_out"] == -1.0, "td_out: أول معاملة لازم تكون -1"
    assert out.loc[3, "structural_td_out"] == 10800.0, "td_out: MISALIGNMENT — الـ bug القديم!"
    assert out.loc[1, "structural_is_first_out"] == 1
    assert out.loc[3, "structural_is_first_out"] == 0
    assert out.loc[2, "structural_td_out"] == 3600.0
    logger.info("  ✓ PASS: td_out alignment + is_first_out")

    # ---- اختبار 2: محاذاة td_in ----
    df = pd.DataFrame({
        "src_idx": [5, 6, 5], "dst_idx": [9, 9, 9],
        "Timestamp": pd.to_datetime(["2022-01-01 01:00", "2022-01-01 02:00",
                                     "2022-01-01 05:00"]),
        "transaction_id": [0, 1, 2],
    })
    out = add_ports_and_time_deltas(df.copy())
    assert out.loc[0, "structural_td_in"] == -1.0
    assert out.loc[1, "structural_td_in"] == 3600.0
    assert out.loc[2, "structural_td_in"] == 10800.0
    logger.info("  ✓ PASS: td_in alignment")

    # ---- اختبار 3: causal priors ----
    df = pd.DataFrame({
        "src_idx": [1, 1], "dst_idx": [9, 8],
        "Timestamp": pd.to_datetime(["2022-01-01 01:00", "2022-01-01 02:00"]),
        "transaction_id": [0, 1],
        "Amount Paid": [100.0, 50.0], "Amount Received": [100.0, 50.0],
    })
    out = add_causal_priors(df.copy())
    assert out.loc[0, "causal_src_out_degree"] == 0
    assert out.loc[1, "causal_src_out_degree"] == 1
    assert out.loc[1, "causal_src_out_amt_sum"] == 100.0
    assert out.loc[0, "causal_src_out_fan"] == 0
    assert out.loc[1, "causal_src_out_fan"] == 1
    logger.info("  ✓ PASS: causal priors")

    # ---- اختبار 4: registry ----
    cols = ["basic_log_amount_paid", "gfp_feature_0", "structural_port_out",
            "causal_src_out_degree", "entity_is_same", "transaction_id"]
    assert set(get_feature_cols(cols, "xgb_L0")) == {"basic_log_amount_paid", "gfp_feature_0"}
    assert set(get_feature_cols(cols, "gnn_L0")) == {"basic_log_amount_paid"}
    assert "entity_is_same" in get_feature_cols(cols, "xgb_L3")
    assert "entity_is_same" not in get_feature_cols(cols, "xgb_L2")
    logger.info("  ✓ PASS: feature sets registry")

    logger.info("[Tests] ✅ ALL TESTS PASSED")
    logger.info("=" * 60)


# ==========================================
# Phase 0: Load & Clean
# ==========================================
RAW_DTYPES = {
    "From Bank": "category", "Account": "string", "To Bank": "category",
    "Account.1": "string", "Amount Received": "float64", "Receiving Currency": "category",
    "Amount Paid": "float64", "Payment Currency": "category", "Payment Format": "category",
}


def load_raw(path):
    df = pd.read_csv(path, dtype={k: v for k, v in RAW_DTYPES.items()})
    df["Is Laundering"] = df["Is Laundering"].astype("int8")
    logger.info(f"[Load] {len(df):,} tx | illicit {df['Is Laundering'].sum():,} "
                f"({df['Is Laundering'].mean() * 100:.3f}%)")
    return df


def clean_and_build_ids(df):
    df = df.dropna(subset=["Timestamp", "From Bank", "Account",
                           "To Bank", "Account.1", "Amount Paid"]).copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
    df["src_bank_norm"] = df["From Bank"].astype(str).str.lstrip('0').replace('', '0')
    df["dst_bank_norm"] = df["To Bank"].astype(str).str.lstrip('0').replace('', '0')
    df["src_node"] = (df["src_bank_norm"] + "_" + df["Account"].astype(str)).astype("category")
    df["dst_node"] = (df["dst_bank_norm"] + "_" + df["Account.1"].astype(str)).astype("category")
    # sort مستقر — ترتيب الصفوف حتمي (أساس تطابق transaction_id مع ملف الأنماط)
    df = df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    df["transaction_id"] = df.index.astype("int32")
    logger.info(f"[Clean] {len(df):,} rows | {df['Timestamp'].min()} → {df['Timestamp'].max()}")
    return df


def build_node_index(df):
    all_nodes = pd.unique(pd.concat([df["src_node"], df["dst_node"]], ignore_index=True))
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    df["src_idx"] = df["src_node"].map(node_to_idx).astype("int32")
    df["dst_idx"] = df["dst_node"].map(node_to_idx).astype("int32")
    logger.info(f"[Nodes] {len(all_nodes):,} unique accounts")
    return df, node_to_idx, len(all_nodes)


# ==========================================
# Phase 2: Temporal Split
# ==========================================
def temporal_split(df, fractions=(0.6, 0.2, 0.2)):
    day_idx = (df["Timestamp"] - df["Timestamp"].min()).dt.days
    n_days = int(day_idx.max()) + 1
    daily = np.bincount(day_idx, minlength=n_days)
    cum = np.concatenate([[0], np.cumsum(daily)])

    best_err, best_i, best_j = None, 0, n_days
    for i in range(n_days + 1):
        for j in range(i, n_days + 1):
            props = [cum[i] / len(df), (cum[j] - cum[i]) / len(df),
                     (cum[n_days] - cum[j]) / len(df)]
            err = max(abs(p - t) / t for p, t in zip(props, fractions))
            if best_err is None or err < best_err:
                best_err, best_i, best_j = err, i, j

    labels = np.where(day_idx < best_i, "train",
                      np.where(day_idx < best_j, "val", "test"))
    df["split"] = pd.Categorical(labels, categories=["train", "val", "test"])
    logger.info(f"[Split] day boundaries: train<{best_i} ≤ val<{best_j} ≤ test")
    for s in ["train", "val", "test"]:
        sub = df[df["split"] == s]
        pos = int(sub["Is Laundering"].sum())
        logger.info(f"[Split] {s:5s}: {len(sub):,} ({len(sub) / len(df) * 100:.1f}%) "
                    f"| illicit {pos:,} ({pos / max(len(sub), 1) * 100:.3f}%)")
    return df


# ==========================================
# Phase 4: Basic Features
# ==========================================
AMOUNT_EPS = 1.0


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

    # encodings من train فقط — no leakage
    for c in ["Receiving Currency", "Payment Currency", "Payment Format"]:
        cats = sorted(train_df[c].astype(str).unique())
        mapping = {v: i for i, v in enumerate(cats)}
        df[f"basic_{c.lower().replace(' ', '_')}_code"] = (
            df[c].astype(str).map(mapping).fillna(-1).astype("int32"))
    return df


# ==========================================
# Phase 4.5: Ports & Time Deltas — [FIX-1] + [FIX-2]
# ==========================================
def add_ports_and_time_deltas(df):
    # Port numbering (precompute global — مطابق لبروتوكول الريبو الرسمي)
    fo = df.drop_duplicates(["src_idx", "dst_idx"], keep="first")[["src_idx", "dst_idx"]].copy()
    fo["structural_port_out"] = fo.groupby("src_idx").cumcount()
    df = df.merge(fo, on=["src_idx", "dst_idx"], how="left")

    fi = df.drop_duplicates(["dst_idx", "src_idx"], keep="first")[["dst_idx", "src_idx"]].copy()
    fi["structural_port_in"] = fi.groupby("dst_idx").cumcount()
    df = df.merge(fi, on=["dst_idx", "src_idx"], how="left")
    df = df.reset_index(drop=True)

    # ===== الإصلاح الجوهري: إسناد بالـ index labels (مش .values) =====
    # + is_first قبل fillna + tie-break حتمي بـ transaction_id
    td_out = (df.sort_values(["src_idx", "Timestamp", "transaction_id"], kind="mergesort")
                .groupby("src_idx")["Timestamp"].diff().dt.total_seconds())
    df["structural_td_out"] = td_out                                   # label-aligned ✓
    df["structural_is_first_out"] = df["structural_td_out"].isna().astype("int8")
    df["structural_td_out"] = df["structural_td_out"].fillna(-1.0)

    td_in = (df.sort_values(["dst_idx", "Timestamp", "transaction_id"], kind="mergesort")
               .groupby("dst_idx")["Timestamp"].diff().dt.total_seconds())
    df["structural_td_in"] = td_in
    df["structural_is_first_in"] = df["structural_td_in"].isna().astype("int8")
    df["structural_td_in"] = df["structural_td_in"].fillna(-1.0)
    return df


# ==========================================
# Phase 5: Causal Node Priors — [FIX-4] مساهمتك رقم 1
# ==========================================
def add_causal_priors(df):
    # ---- Outgoing (المصدر) ----
    out_s = df.sort_values(["src_idx", "Timestamp", "transaction_id"], kind="mergesort")
    is_new = ~out_s.duplicated(["src_idx", "dst_idx"], keep="first")
    grp = out_s.groupby("src_idx")
    out_s["causal_src_out_degree"] = grp.cumcount()
    out_s["causal_src_out_fan"] = is_new.groupby(out_s["src_idx"]).cumsum() - is_new.astype(int)
    out_s["causal_src_out_amt_sum"] = grp["Amount Paid"].cumsum() - out_s["Amount Paid"]
    df = df.join(out_s[["causal_src_out_degree", "causal_src_out_fan",
                        "causal_src_out_amt_sum"]])

    # ---- Incoming (المستقبِل) ----
    in_s = df.sort_values(["dst_idx", "Timestamp", "transaction_id"], kind="mergesort")
    is_new = ~in_s.duplicated(["dst_idx", "src_idx"], keep="first")
    grp = in_s.groupby("dst_idx")
    in_s["causal_dst_in_degree"] = grp.cumcount()
    in_s["causal_dst_in_fan"] = is_new.groupby(in_s["dst_idx"]).cumsum() - is_new.astype(int)
    in_s["causal_dst_in_amt_sum"] = grp["Amount Received"].cumsum() - in_s["Amount Received"]
    df = df.join(in_s[["causal_dst_in_degree", "causal_dst_in_fan",
                       "causal_dst_in_amt_sum"]])
    return df


# ==========================================
# Phase 5.5: Entity Features — [FIX-5] تُحفظ لكن L3 فقط يستخدمها
# ==========================================
def add_entity_features(df, accounts_path):
    df_acc = pd.read_csv(accounts_path)
    bank_col = 'Bank ID' if 'Bank ID' in df_acc.columns else next(
        (c for c in df_acc.columns if 'bank' in c.lower()), None)
    acc_col = 'Account Number' if 'Account Number' in df_acc.columns else next(
        (c for c in df_acc.columns if 'account' in c.lower()), None)
    ent_col = 'Entity ID' if 'Entity ID' in df_acc.columns else next(
        (c for c in df_acc.columns if 'entity' in c.lower()), None)

    if not (bank_col and acc_col and ent_col):
        logger.warning("[Entity] أعمدة غير معروفة — تخطي")
        return df, None

    df_acc["Bank_Account"] = (df_acc[bank_col].astype(str).str.lstrip('0').replace('', '0')
                              + "_" + df_acc[acc_col].astype(str))
    ent_map = dict(zip(df_acc["Bank_Account"], df_acc[ent_col].astype(str)))

    src_ent = df["src_node"].astype(str).map(ent_map).fillna("-1")
    dst_ent = df["dst_node"].astype(str).map(ent_map).fillna("-1")
    df["entity_is_same"] = ((src_ent == dst_ent) & (src_ent != "-1")).astype("int8")

    # إحصائية الـ lift — توثيق قيمة تجربة L3
    n_same = int(df["entity_is_same"].sum())
    base_rate = float(df["Is Laundering"].mean())
    same_rate = float(df.loc[df["entity_is_same"] == 1, "Is Laundering"].mean()) if n_same else 0.0
    lift = (same_rate / base_rate) if base_rate > 0 and n_same else float('nan')

    logger.info(f"[Entity] match {(src_ent != '-1').mean() * 100:.2f}% | "
                f"same-entity {n_same:,} | illicit-lift {lift:.2f}x  ← يحدد حجم تجربة L3")
    return df, lift


# ==========================================
# Phase 6: GFP Pattern Mining — [FIX-3] + [FIX-3b]
# ==========================================
def run_pattern_mining(df, unix_ts, out_dir="gfp_temp_chunks"):
    from snapml import GraphFeaturePreprocessor
    import pyarrow as pa
    import pyarrow.parquet as pq

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    batch_size = GFP_CONFIG["batch_size"]
    buffer_batches = GFP_CONFIG["buffer_batches"]

    gfp = GraphFeaturePreprocessor()
    # ⚠️ القواعد المكتشفة تجريبيًا:
    #   1) طبقة C++ في snapml تقبل فقط int أو list of ints (الـ strings بتكسرها)
    #   2) الافتراضي vertex_stats_cols=[3] يعني Timestamp فقط — stats المبالغ
    #      كانت غايبة (اتأكدنا بالبصمة ≈1.66e9 في 5 features)
    #   ⟵ [FIX-3b]: [3, 4, 6] = Timestamp + Amount Received + Amount Paid (زي الورقة)
    requested = {
        "num_threads": GFP_CONFIG["num_threads"],
        "fan": True, "fan_tw": GFP_CONFIG["fan_tw"],
        "degree": True, "degree_tw": GFP_CONFIG["degree_tw"],
        "vertex_stats": True, "vertex_stats_tw": GFP_CONFIG["vertex_stats_tw"],
        "vertex_stats_cols": GFP_CONFIG["vertex_stats_cols"],
        "scatter-gather": True, "scatter-gather_tw": GFP_CONFIG["scatter_gather_tw"],
        "temp-cycle": True, "temp-cycle_tw": GFP_CONFIG["temp_cycle_tw"],
        "lc-cycle": True, "lc-cycle_len": GFP_CONFIG["lc_cycle_len"],
        "lc-cycle_tw": GFP_CONFIG["lc_cycle_tw"],
    }
    available = set(gfp.get_params().keys())
    unknown = sorted(set(requested) - available)
    if unknown:
        logger.warning(f"[GFP] باراميترات غير معروفة في نسختك (تم تجاهلها بأمان): {unknown}")
    params = {k: v for k, v in requested.items() if k in available}
    gfp.set_params(params)
    logger.info(f"[GFP] active params: {params}")

    # ==== مدخل بنفس schema المُجرَّب (9 أعمدة) ====
    # الترتيب مهم — vertex_stats_cols بيشاور على مواضع الأعمدة هنا:
    #   0=Edge ID, 1=Source, 2=Target, 3=Timestamp, 4=Amount Received,
    #   5=Receiving Currency, 6=Amount Paid, 7=Payment Currency, 8=Payment Format
    gfp_input = pd.DataFrame({
        "Edge ID":   df["transaction_id"].astype("int64"),
        "Source":    df["src_idx"].astype("int64"),
        "Target":    df["dst_idx"].astype("int64"),
        "Timestamp": pd.Series(unix_ts, index=df.index).astype("int32"),
        "Amount Received": df["Amount Received"].astype("float32"),
        "Receiving Currency": df["basic_receiving_currency_code"].astype("int32"),
        "Amount Paid":   df["Amount Paid"].astype("float32"),
        "Payment Currency": df["basic_payment_currency_code"].astype("int32"),
        "Payment Format":   df["basic_payment_format_code"].astype("int32"),
    })

    # Streaming: train → val → test بنفس الـ object (val يرى تاريخ train،
    # test يرى train+val — بروتوكول الورقة، صفر leakage من المستقبل)
    for split_name in ["train", "val", "test"]:
        mask = (df["split"] == split_name).to_numpy()
        split_input = gfp_input[mask]
        n_edges = len(split_input)
        if n_edges == 0:
            continue
        n_batches = (n_edges + batch_size - 1) // batch_size
        logger.info(f"[GFP] {split_name}: {n_edges:,} edges | {n_batches:,} batches "
                    f"(batch={batch_size})")
        buffer, chunk_idx = [], 0
        t0 = time.time()
        for i in range(0, n_edges, batch_size):
            batch = split_input.iloc[i:i + batch_size]
            with suppress_c_output():
                feats = gfp.transform(batch)
            buffer.append((feats, batch["Edge ID"].to_numpy()))

            if len(buffer) >= buffer_batches or i + batch_size >= n_edges:
                arr = np.concatenate([b[0] for b in buffer])
                ids = np.concatenate([b[1] for b in buffer])
                chunk = pd.DataFrame(arr, columns=[f"gfp_feature_{j}" for j in range(arr.shape[1])])
                chunk["transaction_id"] = ids
                chunk.to_parquet(os.path.join(out_dir, f"{split_name}_chunk_{chunk_idx}.parquet"),
                                 engine="pyarrow", index=False)
                buffer.clear(); chunk_idx += 1; gc.collect()
                done = i + batch_size
                elapsed = time.time() - t0
                eta = elapsed / done * (n_edges - done)
                logger.info(f"  [{split_name}] {done:,}/{n_edges:,} "
                            f"({done / n_edges * 100:.1f}%) | {elapsed / 60:.1f}m | "
                            f"ETA {eta / 60:.1f}m")
        logger.info(f"[GFP] {split_name} ✓ done in {(time.time() - t0) / 60:.1f}m")

    # دمج الـ chunks
    chunk_files = [os.path.join(out_dir, f) for f in sorted(os.listdir(out_dir))
                   if f.endswith(".parquet")]
    features_df = pa.concat_tables([pq.read_table(f) for f in chunk_files]).to_pandas()
    for f in chunk_files:
        os.remove(f)
    os.rmdir(out_dir)

    n_gfp_cols = features_df.shape[1] - 1
    logger.info(f"[GFP] ✓ {n_gfp_cols} gfp columns  ← المفروض 264 (كان 224 قبل [FIX-3b])")

    df = df.merge(features_df, on="transaction_id", how="left")
    del features_df
    gc.collect()
    return df, n_gfp_cols


# ==========================================
# Phase 6.5: Patterns Join — للتقييم فقط [FIX-8]
# (نفس الكود المُجرَّب في join_patterns.py — بيشتغل من الذاكرة)
# ==========================================
TRANS_COLS = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1",
              "Amount Received", "Receiving Currency", "Amount Paid",
              "Payment Currency", "Payment Format", "Is Laundering"]


def parse_patterns_file(path):
    row_texts, row_meta, attempts = [], [], []
    attempt_id, current = 0, None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            upper = line.upper()
            if upper.startswith("BEGIN LAUNDERING ATTEMPT"):
                m = re.search(r"ATTEMPT\s*-\s*([A-Z\-]+)", upper)
                current = m.group(1) if m else "UNKNOWN"
                attempt_id += 1
                attempts.append({"attempt_id": attempt_id,
                                 "pattern_type": current, "raw_header": line})
            elif upper.startswith("END LAUNDERING ATTEMPT"):
                current = None
            elif line.startswith("Timestamp"):
                continue
            elif current is not None:
                row_texts.append(line)
                row_meta.append((attempt_id, current))

    logger.info(f"[Patterns] {len(attempts):,} attempts | {len(row_texts):,} tx rows")
    csv_text = ",".join(TRANS_COLS) + "\n" + "\n".join(row_texts)
    pat_df = pd.read_csv(StringIO(csv_text), dtype=str)
    pat_df["attempt_id"], pat_df["pattern_type"] = zip(*row_meta)
    vc = pd.DataFrame(attempts)["pattern_type"].value_counts()
    logger.info(f"[Patterns] attempt types: {vc.to_dict()}")
    return pat_df


def _row_key(timestamps, from_bank, account, to_bank, account1,
             amt_recv, recv_cur, amt_paid, pay_cur, fmt, is_lau):
    """مفتاح نصي حتمي للصف — يحل مشاكل تطابق الـ floats والـ formats."""
    return (timestamps.astype(str) + "|" + from_bank.astype(str).str.strip() + "|"
            + account.astype(str).str.strip() + "|" + to_bank.astype(str).str.strip() + "|"
            + account1.astype(str).str.strip() + "|"
            + amt_recv.round(6).astype(str) + "|" + recv_cur.astype(str).str.strip() + "|"
            + amt_paid.round(6).astype(str) + "|" + pay_cur.astype(str).str.strip() + "|"
            + fmt.astype(str).str.strip() + "|" + is_lau.astype(str))


def join_patterns_in_memory(df, patterns_path, out_dir):
    """ربط الأنماط بـ transaction_id باستخدام الـ df الموجود في الذاكرة."""
    pat_df = parse_patterns_file(patterns_path)

    key_t = _row_key(df["Timestamp"].dt.strftime("%Y/%m/%d %H:%M"),
                     df["From Bank"], df["Account"], df["To Bank"], df["Account.1"],
                     df["Amount Received"], df["Receiving Currency"],
                     df["Amount Paid"], df["Payment Currency"],
                     df["Payment Format"], df["Is Laundering"])

    pat_ts = pd.to_datetime(pat_df["Timestamp"], format="%Y/%m/%d %H:%M")
    key_p = _row_key(pat_ts.dt.strftime("%Y/%m/%d %H:%M"),
                     pat_df["From Bank"], pat_df["Account"], pat_df["To Bank"], pat_df["Account.1"],
                     pd.to_numeric(pat_df["Amount Received"]), pat_df["Receiving Currency"],
                     pd.to_numeric(pat_df["Amount Paid"]), pat_df["Payment Currency"],
                     pat_df["Payment Format"], pat_df["Is Laundering"].astype(int))

    # فك تشابه الصفوف المتطابقة تمامًا (بترتيب الظهور)
    dup_t = key_t.groupby(key_t).cumcount()
    dup_p = key_p.groupby(key_p).cumcount()

    t_small = pd.DataFrame({"key": key_t, "_dup": dup_t,
                            "transaction_id": df["transaction_id"].to_numpy()})
    p_small = pd.DataFrame({"key": key_p.values, "_dup": dup_p.values,
                            "attempt_id": pat_df["attempt_id"].to_numpy(),
                            "pattern_type": pat_df["pattern_type"].to_numpy()})
    del key_t, key_p, dup_t, dup_p
    gc.collect()

    merged = t_small.merge(p_small, on=["key", "_dup"], how="left")
    matched = merged.dropna(subset=["attempt_id"])[["transaction_id", "attempt_id", "pattern_type"]]
    coverage = len(matched) / len(p_small) * 100
    logger.info(f"[Match] pattern rows covered: {len(matched):,}/{len(p_small):,} "
                f"({coverage:.2f}%)")
    if coverage < 99.0:
        logger.error(f"[Match] ⚠️ تغطية منخفضة ({coverage:.2f}%) — "
                     f"ابعتلي اللوج ده. أكمل بالمتاح للتحليل.")

    # تحديث edge_metadata
    meta = pd.read_parquet(os.path.join(out_dir, "edge_metadata.parquet"))
    meta = meta.merge(matched, on="transaction_id", how="left")

    # [FIX] assignment مشروط بدل fillna(ndarray) — pandas مايقبلش ndarray
    meta["pattern_type"] = meta["pattern_type"].astype("object")
    meta.loc[meta["pattern_type"].isna() & (meta["Is Laundering"] == 1), "pattern_type"] = "LONE"
    meta.loc[meta["pattern_type"].isna() & (meta["Is Laundering"] == 0), "pattern_type"] = "NONE"
    meta["pattern_type"] = meta["pattern_type"].astype("category")

    meta["attempt_id"] = meta["attempt_id"].astype("Int64")
    meta.to_parquet(os.path.join(out_dir, "edge_metadata.parquet"),
                    engine="pyarrow", index=False)

    # توزيع المعاملات الإجرامية حسب النمط
    il = meta.loc[meta["Is Laundering"] == 1, "pattern_type"].value_counts()
    total_il = int((meta["Is Laundering"] == 1).sum())
    logger.info("=" * 55)
    logger.info("[Patterns] توزيع المعاملات الإجرامية:")
    for k, v in il.items():
        logger.info(f"   {k:<18} {v:>10,} ({v / total_il * 100:5.1f}%)")
    logger.info("=" * 55)

    # توزيع لكل split
    for s in ["train", "val", "test"]:
        sub = meta[(meta["split"] == s) & (meta["Is Laundering"] == 1)]
        n_att = sub["attempt_id"].dropna().nunique()
        n_lone = int((sub["pattern_type"] == "LONE").sum())
        logger.info(f"[Split] {s:5s}: illicit {len(sub):,} | attempts {n_att:,} | LONE {n_lone:,}")

    # ملخص المحاولات
    att = (meta.dropna(subset=["attempt_id"])
               .groupby(["attempt_id", "pattern_type"], observed=True)
               .size().reset_index(name="n_tx"))
    att.to_csv(os.path.join(out_dir, "patterns_summary.csv"), index=False)
    logger.info(f"[Patterns] ✓ patterns_summary.csv — {att['attempt_id'].nunique():,} attempts "
                f"| avg {att['n_tx'].mean():.1f} tx/attempt")
    return meta


# ==========================================
# Phase 7: Sanity Gates + Assembly
# ==========================================
def sanity_gates(df, feature_cols):
    logger.info("=" * 60)
    logger.info("[Sanity] بوابات الجودة:")

    r1 = df["structural_is_first_out"].mean()
    r2 = df["structural_is_first_in"].mean()
    assert r1 > 0 and r2 > 0, f"is_first ميتة! out={r1}, in={r2} — الـ FIX-2 مااشتغلش"
    logger.info(f"  ✓ is_first_out {r1 * 100:.2f}% | is_first_in {r2 * 100:.2f}%")

    assert not (df["structural_td_out"] < -1).any(), "td_out فيه قيم < -1"
    assert not (df["structural_td_in"] < -1).any(), "td_in فيه قيم < -1"
    logger.info("  ✓ td values منطقية")

    nulls = df[feature_cols].isna().sum()
    bad = nulls[nulls > 0]
    assert len(bad) == 0, f"NULLs في features: {list(bad.index)}"
    logger.info("  ✓ صفر NULLs في الـ features")
    logger.info("=" * 60)


def assemble_outputs(df, n_nodes, node_to_idx, n_gfp_cols, entity_lift):
    all_feature_cols = [c for c in df.columns
                        if c.startswith(("basic_", "gfp_", "structural_", "causal_", "entity_"))]
    sanity_gates(df, all_feature_cols)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"[Assembly] Feature matrix: ({len(df):,} × {len(all_feature_cols)})")

    # 1) جدول الـ superset (RAW — الـ standardization بيتم وقت التدريب على الـ subset)
    save_cols = all_feature_cols + ["Is Laundering", "split", "transaction_id"]
    df[save_cols].to_parquet(os.path.join(OUTPUT_DIR, "features_all.parquet"),
                             engine="pyarrow", index=False)
    logger.info("[Assembly] ✓ features_all.parquet")

    # 2) node index (للـ inference والـ explanations)
    pd.DataFrame({"node": list(node_to_idx.keys()),
                  "idx": list(node_to_idx.values())}).to_parquet(
        os.path.join(OUTPUT_DIR, "node_to_idx.parquet"), engine="pyarrow", index=False)
    logger.info("[Assembly] ✓ node_to_idx.parquet")

    # 3) edge metadata (هيتحدث بالأنماط في الخطوة الجاية)
    df[["transaction_id", "Timestamp", "split", "Is Laundering",
        "src_idx", "dst_idx"]].to_parquet(
        os.path.join(OUTPUT_DIR, "edge_metadata.parquet"), engine="pyarrow", index=False)
    logger.info("[Assembly] ✓ edge_metadata.parquet")

    # 4) metadata (التوثيق + الـ standardization وقت التدريب)
    group_counts = {
        "basic": sum(c.startswith("basic_") for c in all_feature_cols),
        "gfp": int(n_gfp_cols),
        "structural": sum(c.startswith("structural_") for c in all_feature_cols),
        "causal": sum(c.startswith("causal_") for c in all_feature_cols),
        "entity": sum(c.startswith("entity_") for c in all_feature_cols),
    }
    with open(os.path.join(OUTPUT_DIR, "feature_metadata.json"), "w") as f:
        json.dump({
            "trans_file": TRANS_FILE,
            "n_nodes": int(n_nodes), "n_edges": int(len(df)),
            "n_features": len(all_feature_cols),
            "feature_groups": group_counts,
            "entity_illicit_lift": entity_lift,
            "gfp_vertex_stats_cols": GFP_CONFIG["vertex_stats_cols"],
            "feature_cols": all_feature_cols,
        }, f, indent=2)
    logger.info(f"[Assembly] ✓ feature_metadata.json | groups: {group_counts}")

    # 5) مكونات الجراف (edge_attr بيتعمل وقت التدريب من features_all + الـ subset)
    import torch
    edge_index = torch.vstack([
        torch.from_numpy(df["src_idx"].to_numpy(dtype=np.int64)),
        torch.from_numpy(df["dst_idx"].to_numpy(dtype=np.int64))])
    y = torch.from_numpy(df["Is Laundering"].to_numpy(dtype=np.int64))
    masks = {s: torch.from_numpy((df["split"] == s).to_numpy().copy())
             for s in ["train", "val", "test"]}
    torch.save(edge_index, os.path.join(OUTPUT_DIR, "edge_index.pt"))
    torch.save(y, os.path.join(OUTPUT_DIR, "y.pt"))
    torch.save(masks, os.path.join(OUTPUT_DIR, "masks.pt"))
    logger.info("[Assembly] ✓ edge_index / y / masks (.pt)")

    # 6) تقرير الـ feature sets (الـ ladder كله من التشغيلة دي)
    for fs_name in FEATURE_SETS:
        cols = get_feature_cols(all_feature_cols, fs_name)
        logger.info(f"[Assembly] {fs_name:8s} → {len(cols):3d} features")


# ==========================================
# MAIN
# ==========================================
def main():
    t_total = time.time()
    logger.info("=" * 60)
    logger.info(f"AML Preprocessing (single-file v3) | {TRANS_FILE}")
    logger.info(f"entity={'ON (saved for L3)' if COMPUTE_ENTITY else 'OFF'}")
    logger.info(f"vertex_stats_cols = {GFP_CONFIG['vertex_stats_cols']} (Timestamp + Amounts)")
    try:
        free_gb = os.statvfs(".").f_bavail * os.statvfs(".").f_frsize / 1e9
        logger.info(f"Free disk: {free_gb:.0f} GB")
    except Exception:
        pass
    logger.info("=" * 60)

    assert os.path.exists(TRANS_PATH), f"الملف غير موجود: {TRANS_PATH}"
    assert os.path.exists(PATTERNS_PATH), f"ملف الأنماط غير موجود: {PATTERNS_PATH}"

    # ---- الاختبارات الذاتية أولًا ----
    run_self_tests()

    # ---- Pipeline ----
    df = load_raw(TRANS_PATH)
    df = clean_and_build_ids(df)
    df, node_to_idx, n_nodes = build_node_index(df)
    df = temporal_split(df)

    train_df = df[df["split"] == "train"]
    unix_ts = to_unix_seconds(df["Timestamp"])

    df = add_basic_features(df, train_df)
    df = add_ports_and_time_deltas(df)          # [FIX-1] + [FIX-2]
    df = add_causal_priors(df)

    entity_lift = None
    if COMPUTE_ENTITY and os.path.exists(ACCOUNTS_PATH):
        df, entity_lift = add_entity_features(df, ACCOUNTS_PATH)
    else:
        logger.info("[Entity] OFF — مش هتتحسب")

    df, n_gfp_cols = run_pattern_mining(df, unix_ts)   # [FIX-3] + [FIX-3b]

    assemble_outputs(df, n_nodes, node_to_idx, n_gfp_cols, entity_lift)

    # ---- ربط الأنماط (آخر خطوة — للتقييم فقط) ----
    if os.path.exists(PATTERNS_PATH):
        try:
            join_patterns_in_memory(df, PATTERNS_PATH, OUTPUT_DIR)
        except Exception as e:
            logger.error(f"[Patterns] فشل الربط: {e} — الـ features محفوظة، "
                         f"شغّل join_patterns.py يدوي وابعتلي الـ traceback")
    else:
        logger.warning(f"[Patterns] الملف غير موجود: {PATTERNS_PATH} — تخطي")

    logger.info(f"✅ Completed in {(time.time() - t_total) / 3600:.2f} hours")


if __name__ == "__main__":
    main()
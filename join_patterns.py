# -*- coding: utf-8 -*-
"""
join_patterns.py — ربط الأنماط بـ transaction_id
==================================================
للتقييم فقط — ممنوع منعًا باتًا دخول مخرجاته كـ features في أي موديل
(الملف بيغطي المعاملات الإجرامية بس = العضوية فيه هي الـ label نفسه).

التشغيل: python join_patterns.py --patterns-file "HI-Medium_Patterns.txt"
"""

import os
import re
import gc
import logging
import argparse
from io import StringIO

import numpy as np
import pandas as pd

# ==========================================
# CONFIG
# ==========================================
DATA_DIR   = "/home/jovyan/Ahmed Magdi work/data"
TRANS_FILE = "HI-Medium_Trans.csv"
OUTPUT_DIR = "processed_data"

TRANS_COLS = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1",
              "Amount Received", "Receiving Currency", "Amount Paid",
              "Payment Currency", "Payment Format", "Is Laundering"]

RAW_DTYPES = {
    "From Bank": "category", "Account": "string", "To Bank": "category",
    "Account.1": "string", "Amount Received": "float64", "Receiving Currency": "category",
    "Amount Paid": "float64", "Payment Currency": "category", "Payment Format": "category",
}

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("join_patterns")


# ==========================================
# 1) Parser
# ==========================================
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
                attempts.append({"attempt_id": attempt_id, "pattern_type": current})
            elif upper.startswith("END LAUNDERING ATTEMPT"):
                current = None
            elif line.startswith("Timestamp"):
                continue          # سطر header متكرر داخل الملف
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


# ==========================================
# 2) إعادة بناء transaction_id (نفس منطق الـ preprocessing بالظبط)
# ==========================================
def rebuild_transaction_ids():
    df = pd.read_csv(os.path.join(DATA_DIR, TRANS_FILE),
                     dtype={k: v for k, v in RAW_DTYPES.items()})
    df = df.dropna(subset=["Timestamp", "From Bank", "Account",
                           "To Bank", "Account.1", "Amount Paid"]).copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
    df = df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    df["transaction_id"] = df.index.astype("int32")
    return df


# ==========================================
# 3) مفاتيح مطابقة حتمية
# ==========================================
def _row_key(timestamps, from_bank, account, to_bank, account1,
             amt_recv, recv_cur, amt_paid, pay_cur, fmt, is_lau):
    """مفتاح نصي حتمي للصف — يحل مشاكل تطابق الـ floats والـ formats."""
    return (timestamps.astype(str) + "|" + from_bank.astype(str).str.strip() + "|"
            + account.astype(str).str.strip() + "|" + to_bank.astype(str).str.strip() + "|"
            + account1.astype(str).str.strip() + "|"
            + amt_recv.round(6).astype(str) + "|" + recv_cur.astype(str).str.strip() + "|"
            + amt_paid.round(6).astype(str) + "|" + pay_cur.astype(str).str.strip() + "|"
            + fmt.astype(str).str.strip() + "|" + is_lau.astype(str))


# ==========================================
# 4) Main
# ==========================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns-file", default="HI-Medium_Patterns.txt")
    args = ap.parse_args()
    patterns_path = os.path.join(DATA_DIR, args.patterns_file)
    assert os.path.exists(patterns_path), f"غير موجود: {patterns_path}"

    pat_df = parse_patterns_file(patterns_path)
    df = rebuild_transaction_ids()

    # مفاتيح الطرفين
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

    # المطابقة
    merged = t_small.merge(p_small, on=["key", "_dup"], how="left")
    matched = merged.dropna(subset=["attempt_id"])[["transaction_id", "attempt_id", "pattern_type"]]
    coverage = len(matched) / len(p_small) * 100
    logger.info(f"[Match] covered: {len(matched):,}/{len(p_small):,} ({coverage:.2f}%)")
    if coverage < 99.0:
        logger.error("⚠️ تغطية منخفضة — ابعتلي اللوج ده")

    # ===== تحديث edge_metadata =====
    meta = pd.read_parquet(os.path.join(OUTPUT_DIR, "edge_metadata.parquet"))
    meta = meta.merge(matched, on="transaction_id", how="left")

    # [FIX] استبدال fillna(ndarray) بـ assignment مشروط — pandas مايقبلش ndarray في fillna
    meta["pattern_type"] = meta["pattern_type"].astype("object")
    meta.loc[meta["pattern_type"].isna() & (meta["Is Laundering"] == 1), "pattern_type"] = "LONE"
    meta.loc[meta["pattern_type"].isna() & (meta["Is Laundering"] == 0), "pattern_type"] = "NONE"
    meta["pattern_type"] = meta["pattern_type"].astype("category")

    # [FIX] تحويل آمن للـ nullable integer
    meta["attempt_id"] = meta["attempt_id"].astype("Int64")
    meta.to_parquet(os.path.join(OUTPUT_DIR, "edge_metadata.parquet"),
                    engine="pyarrow", index=False)

    # ===== توزيع المعاملات الإجرامية حسب النمط =====
    il = meta.loc[meta["Is Laundering"] == 1, "pattern_type"].value_counts()
    total_il = int((meta["Is Laundering"] == 1).sum())
    logger.info("=" * 55)
    logger.info("[Patterns] توزيع المعاملات الإجرامية:")
    for k, v in il.items():
        logger.info(f"   {k:<18} {v:>10,} ({v / total_il * 100:5.1f}%)")
    logger.info("=" * 55)

    # توزيع لكل split (مهم للتحليل الجاي)
    for s in ["train", "val", "test"]:
        sub = meta[(meta["split"] == s) & (meta["Is Laundering"] == 1)]
        n_att = sub["attempt_id"].dropna().nunique()
        n_lone = int((sub["pattern_type"] == "LONE").sum())
        logger.info(f"[Split] {s:5s}: illicit {len(sub):,} | attempts {n_att:,} | LONE {n_lone:,}")

    # ===== ملخص المحاولات =====
    att = (meta.dropna(subset=["attempt_id"])
               .groupby(["attempt_id", "pattern_type"], observed=True)
               .size().reset_index(name="n_tx"))
    att.to_csv(os.path.join(OUTPUT_DIR, "patterns_summary.csv"), index=False)
    logger.info(f"✅ {att['attempt_id'].nunique():,} attempts | avg {att['n_tx'].mean():.1f} tx/attempt")


if __name__ == "__main__":
    main()
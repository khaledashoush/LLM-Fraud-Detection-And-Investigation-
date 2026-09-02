# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_parquet("processed_data/features_all.parquet")
gfp = [c for c in df.columns if c.startswith("gfp_")]
sample = df[gfp].sample(min(2_000_000, len(df)), random_state=42)
nun = sample.nunique()
dead = nun[nun <= 1]

print(f"GFP columns: {len(gfp)} | حية: {len(gfp) - len(dead)} | ميتة (constant): {len(dead)}")
if len(dead):
    print("أمثلة ميتة:", list(dead.index[:15]))

n_same = int(df["entity_is_same"].sum())
n_il = int(df.loc[df["entity_is_same"] == 1, "Is Laundering"].sum())
print(f"[Entity] same-entity: {n_same:,} | illicit بينهم: {n_il}")
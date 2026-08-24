import pandas as pd
import torch
import os

DATA_DIR = "processed_data"

print("=" * 60)
print("1. Inspecting XGBoost Data (xgboost_features.parquet)")
print("=" * 60)

xgb_path = os.path.join(DATA_DIR, "xgboost_features.parquet")
if os.path.exists(xgb_path):
    # Read the parquet file
    df_xgb = pd.read_parquet(xgb_path)
    
    print(f"Data Shape: {df_xgb.shape} (rows, features)")
    print(f"Total Features: {len(df_xgb.columns) - 2} features (excluding Target and Split columns)")
    
    print("\nFirst 5 rows of the data:")
    print(df_xgb.head())
    
    print("\nSample of extracted feature names:")
    sample_cols = [c for c in df_xgb.columns if c.startswith('basic_')][:3] + \
                  [c for c in df_xgb.columns if c.startswith('gfp_')][:3] + \
                  [c for c in df_xgb.columns if c.startswith('structural_')][:3] + \
                  [c for c in df_xgb.columns if c.startswith('entity_')][:1]
    print(sample_cols)
    
    print("\nData Distribution (Train/Val/Test):")
    print(df_xgb['split'].value_counts())
    
    print("\nLaundering Transactions Count (Is Laundering == 1):")
    print(df_xgb['Is Laundering'].value_counts())
else:
    print("XGBoost file not found!")

print("\n" + "=" * 60)
print("2. Inspecting Graph Data (PyTorch Tensors)")
print("=" * 60)

def inspect_tensor(name):
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path):
        tensor = torch.load(path)
        print(f"\n--- {name} ---")
        print(f"Type: {type(tensor)}")
        if isinstance(tensor, torch.Tensor):
            print(f"Shape: {tensor.shape}")
            print(f"Dtype: {tensor.dtype}")
            print(f"Sample values (first 5 elements): {tensor.flatten()[:5]}")
        elif isinstance(tensor, dict):
            for k, v in tensor.items():
                print(f"  Key '{k}': Shape={v.shape}, Sum (True values)={v.sum()}")
        return tensor
    return None

# Inspect edge_index (Graph connectivity)
edge_index = inspect_tensor("edge_index.pt")

# Inspect edge_attr (Edge features)
edge_attr = inspect_tensor("edge_attr.pt")

# Inspect y (Labels)
y = inspect_tensor("y.pt")

# Inspect masks (Train/Val/Test splits)
masks = inspect_tensor("masks.pt")

print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)
if edge_index is not None and edge_attr is not None:
    print(f"✅ The graph contains {edge_index.shape[1]:,} edges (transactions).")
    print(f"✅ Each edge contains {edge_attr.shape[1]} features ready for training.")
    print(f"✅ edge_index shape is [2, {edge_index.shape[1]}] representing a (Source, Target) matrix.")
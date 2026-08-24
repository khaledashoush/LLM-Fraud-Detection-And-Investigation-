# -*- coding: utf-8 -*-
"""
Comprehensive XGBoost Evaluation Script (SOTA Standard)
Reports: Precision, Recall, F1, and PR-AUC.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, classification_report, precision_recall_curve, average_precision_score, precision_score, recall_score
import time
import logging

# ==========================================
# Setup Logging
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. Load Processed Data
# ==========================================
logger.info("Loading processed data...")
DATA_PATH = "processed_data/xgboost_features.parquet"

df = pd.read_parquet(DATA_PATH)
logger.info(f"Data loaded successfully. Shape: {df.shape}")

train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'val']
test_df = df[df['split'] == 'test']

FEATURE_COLS = [c for c in df.columns if c not in ['Is Laundering', 'split']]

X_train = train_df[FEATURE_COLS]
y_train = train_df['Is Laundering']

X_val = val_df[FEATURE_COLS]
y_val = val_df['Is Laundering']

X_test = test_df[FEATURE_COLS]
y_test = test_df['Is Laundering']

logger.info(f"Train size: {len(X_train):,} | Val size: {len(X_val):,} | Test size: {len(X_test):,}")

# ==========================================
# 2. Balanced XGBoost Hyperparameters
# ==========================================
scale_pos_weight = 100.0 
logger.info(f"Scale Pos Weight (Balanced): {scale_pos_weight}")

logger.info("Initializing XGBoost Classifier...")
model = xgb.XGBClassifier(
    n_estimators=3000,           
    learning_rate=0.01,          
    max_depth=6,                 
    min_child_weight=1,          
    gamma=0.1,                  
    subsample=0.8,              
    colsample_bytree=0.8,       
    reg_lambda=1.0,              
    scale_pos_weight=scale_pos_weight,
    eval_metric='aucpr',
    early_stopping_rounds=50,    
    tree_method='hist',          
    device='cuda',               
    random_state=42
)

logger.info("Starting balanced training...")
t0 = time.time()

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

train_time = time.time() - t0
logger.info(f"Training completed in {train_time:.2f} seconds. Best iteration: {model.best_iteration}")

# ==========================================
# 3. Comprehensive Evaluation (F1 + PR-AUC)
# ==========================================
logger.info("Evaluating model on Val & Test Sets...")

# Get predicted probabilities
y_val_proba = model.predict_proba(X_val)[:, 1]
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate PR-AUC (Average Precision) - Evaluates ranking quality across all thresholds
val_pr_auc = average_precision_score(y_val, y_val_proba)
test_pr_auc = average_precision_score(y_test, y_pred_proba)

# Find Best Threshold (Max F1) on Val set
precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_idx]

# Apply optimal threshold to Val and Test sets
y_pred_val = (y_val_proba >= best_threshold).astype(int)
y_pred_test = (y_pred_proba >= best_threshold).astype(int)

# Calculate Final Metrics
val_precision = precision_score(y_val, y_pred_val)
val_recall = recall_score(y_val, y_pred_val)
val_f1 = f1_score(y_val, y_pred_val)

test_precision = precision_score(y_test, y_pred_test)
test_recall = recall_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test)

# ==========================================
# Print Results Table
# ==========================================
logger.info("\n" + "="*50)
logger.info("🎯 FINAL RESULTS SUMMARY")
logger.info("="*50)
logger.info(f"Optimal Threshold Found on Val Set: {best_threshold:.4f}")
logger.info("-" * 50)
logger.info(f"{'Metric':<12} | {'Validation':<12} | {'Test':<12}")
logger.info("-" * 50)
logger.info(f"{'PR-AUC':<12} | {val_pr_auc:<12.4f} | {test_pr_auc:<12.4f}")
logger.info(f"{'Precision':<12} | {val_precision:<12.4f} | {test_precision:<12.4f}")
logger.info(f"{'Recall':<12} | {val_recall:<12.4f} | {test_recall:<12.4f}")
logger.info(f"{'F1 Score':<12} | {val_f1:<12.4f} | {test_f1:<12.4f}")
logger.info("="*50)

logger.info("\nDetailed Classification Report (Test Set):")
print(classification_report(y_test, y_pred_test, target_names=["Normal", "Laundering"]))

# ==========================================
# 4. Feature Importance (Top 25)
# ==========================================
logger.info("Top 25 most important features:")
importance = model.get_booster().get_score(importance_type='gain')
importance_df = pd.DataFrame.from_dict(importance, orient='index', columns=['Gain'])
importance_df = importance_df.sort_values(by='Gain', ascending=False).head(25)
print(importance_df)

logger.info("✅ Comprehensive XGBoost Training Completed!")
# -*- coding: utf-8 -*-
"""
Comprehensive XGBoost Evaluation Script with Optuna 
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import json
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, precision_score, recall_score, classification_report
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
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==========================================
# 1. Load Processed Data
# ==========================================
logger.info("Loading processed data...")
DATA_PATH = "processed_data/xgboost_features.parquet"

df = pd.read_parquet(DATA_PATH)
train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'val']
test_df = df[df['split'] == 'test']

FEATURE_COLS = [c for c in df.columns if c not in ['Is Laundering', 'split']]

X_train, y_train = train_df[FEATURE_COLS], train_df['Is Laundering']
X_val, y_val = val_df[FEATURE_COLS], val_df['Is Laundering']
X_test, y_test = test_df[FEATURE_COLS], test_df['Is Laundering']

logger.info(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

# ==========================================
# 2. Optuna Objective Function (Maximizing F1 Score)
# ==========================================
def objective(trial):
    param = {
        'verbosity': 0,
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr', 
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state': 42,
        'n_estimators': 1000,
        'early_stopping_rounds': 50,
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 50.0, 900.0, log=True)
    }
    
    model = xgb.XGBClassifier(**param)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Validation
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_f1 = np.max(f1_scores)
    
    return best_f1 

# ==========================================
# 3. Run Optuna Study
# ==========================================
logger.info("Starting Optuna Auto-Tuning to Maximize F1 Score (50 trials)...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

logger.info(f"Best Trial F1-Score: {study.best_trial.value:.4f}")
logger.info("Best Hyperparameters Found:")
for key, value in study.best_trial.params.items():
    logger.info(f"    {key}: {value}")

# ==========================================
# 4. Train Final Model with Best Params
# ==========================================
logger.info("Training final model with best parameters...")
best_params = study.best_trial.params
best_params.update({
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': 42,
    'n_estimators': 3000,
    'early_stopping_rounds': 50
})

final_model = xgb.XGBClassifier(**best_params)
t0 = time.time()

final_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=True)
logger.info(f"Final training took: {time.time() - t0:.2f}s. Best iteration: {final_model.best_iteration}")

# ==========================================
# 5. Final Evaluation (Precision, Recall, F1, PR-AUC)
# ==========================================
logger.info("Evaluating model on Train, Val & Test Sets...")

# دالة مساعدة للتنبؤ على دفعات لتجنب نفاد ذاكرة كرت الشاشة (OOM)
def predict_proba_in_batches(model, X, batch_size=500000):
    probas = []
    for i in range(0, len(X), batch_size):
        batch = X.iloc[i:i+batch_size]
        probas.append(model.predict_proba(batch)[:, 1])
    return np.concatenate(probas)

logger.info("Predicting in batches to avoid GPU OOM...")
y_train_proba = predict_proba_in_batches(final_model, X_train)
y_val_proba = predict_proba_in_batches(final_model, X_val)
y_test_proba = predict_proba_in_batches(final_model, X_test)

# إيجاد الـ Threshold الأمثل بناءً على أعلى F1 على الـ Validation
precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_idx]

# تطبيق الـ Threshold للتنبؤ النهائي
y_pred_train = (y_train_proba >= best_threshold).astype(int)
y_pred_val = (y_val_proba >= best_threshold).astype(int)
y_pred_test = (y_test_proba >= best_threshold).astype(int)

# حساب المقاييس (Precision, Recall, F1) لكل من Train و Val و Test
train_prec, train_rec, train_f1 = precision_score(y_train, y_pred_train), recall_score(y_train, y_pred_train), f1_score(y_train, y_pred_train)
val_prec, val_rec, val_f1 = precision_score(y_val, y_pred_val), recall_score(y_val, y_pred_val), f1_score(y_val, y_pred_val)
test_prec, test_rec, test_f1 = precision_score(y_test, y_pred_test), recall_score(y_test, y_pred_test), f1_score(y_test, y_pred_test)

# حساب PR-AUC للاطلاع فقط
train_pr_auc = average_precision_score(y_train, y_train_proba)
val_pr_auc = average_precision_score(y_val, y_val_proba)
test_pr_auc = average_precision_score(y_test, y_test_proba)

# ==========================================
# Print Results Table
# ==========================================
logger.info("\n" + "="*65)
logger.info(" FINAL AUTO-TUNED XGBOOST RESULTS (Primary Metric: F1 Score)")
logger.info("="*65)
logger.info(f"Optimal Threshold Found on Val Set: {best_threshold:.4f}")
logger.info("-" * 65)
logger.info(f"{'Metric':<12} | {'Train':<10} | {'Validation':<12} | {'Test':<10}")
logger.info("-" * 65)
logger.info(f"{'Precision':<12} | {train_prec:<10.4f} | {val_prec:<12.4f} | {test_prec:<10.4f}")
logger.info(f"{'Recall':<12} | {train_rec:<10.4f} | {val_rec:<12.4f} | {test_rec:<10.4f}")
logger.info(f"{'F1 Score':<12} | {train_f1:<10.4f} | {val_f1:<12.4f} | {test_f1:<10.4f}")
logger.info(f"{'PR-AUC':<12} | {train_pr_auc:<10.4f} | {val_pr_auc:<12.4f} | {test_pr_auc:<10.4f}")
logger.info("="*65)

logger.info("\nDetailed Classification Report (Test Set):")
print(classification_report(y_test, y_pred_test, target_names=["Normal", "Laundering"]))

# ==========================================
# 6. Overfitting / Underfitting Analysis (Based on F1)
# ==========================================
logger.info("\n" + "="*50)
logger.info(" OVERFITTING / UNDERFITTING ANALYSIS (F1 Gap)")
logger.info("="*50)

gap = train_f1 - test_f1
if gap > 0.15:
    logger.warning(f" OVERFITTING DETECTED: F1 Gap is {gap:.2f}. Model memorized training data.")
elif train_f1 < 0.50:
    logger.warning(f" UNDERFITTING DETECTED: Train F1 is low ({train_f1:.2f}). Model is too simple.")
else:
    logger.info(f" HEALTHY MODEL: F1 Gap is {gap:.2f}. Model generalizes well.")

# ==========================================
# 7. Plot Learning Curve
# ==========================================
logger.info("Generating Learning Curve plot...")
results = final_model.evals_result()

plt.figure(figsize=(10, 6))
plt.plot(results['validation_0']['aucpr'], label='Train PR-AUC', color='blue')
plt.plot(results['validation_1']['aucpr'], label='Val PR-AUC', color='orange')
plt.axvline(final_model.best_iteration, color='gray', linestyle='--', label=f'Best Iteration ({final_model.best_iteration})')
plt.title("XGBoost Learning Curve (Overfitting Check)")
plt.xlabel("Number of Trees (Estimators)")
plt.ylabel("PR-AUC Score")
plt.legend()
plt.grid(True)
plt.savefig("xgboost_learning_curve.png")
logger.info(" Learning curve saved as 'xgboost_learning_curve.png'")

# ==========================================
# 8. Feature Importance (Top 25)
# ==========================================
logger.info("Top 25 most important features:")
importance = final_model.get_booster().get_score(importance_type='gain')
importance_df = pd.DataFrame.from_dict(importance, orient='index', columns=['Gain'])
importance_df = importance_df.sort_values(by='Gain', ascending=False).head(25)
print(importance_df)

# ==========================================
# 9. Save Best Hyperparameters for Future Use
# ==========================================
PARAMS_PATH = "best_xgboost_params.json"

logger.info(f"Saving best hyperparameters to {PARAMS_PATH}...")
with open(PARAMS_PATH, "w") as f:
    best_params_to_save = study.best_trial.params
    best_params_to_save.update({
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state': 42
    })
    json.dump(best_params_to_save, f, indent=4)
logger.info(" Parameters saved successfully! You can reuse them later without re-tuning.")

logger.info(" Comprehensive XGBoost Auto-Tuning Completed!")
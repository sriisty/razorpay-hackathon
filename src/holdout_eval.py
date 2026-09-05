"""
Razorpay Buildathon - Track 2: AI Risk Manager (Return-Risk Scorer)
Day 3, Task 1: Held-out evaluation.

This is the FIRST time the holdout split is touched anywhere in this
project. Threshold (0.32) and cost assumptions (FP=30, FN=300) are
carried over unchanged from the validation-set sweep -- nothing is
re-tuned here. If holdout metrics diverge sharply from validation,
that's a real finding to report honestly, not something to fix by
re-tuning against the holdout (that would defeat the point of holding
it out).
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_score, recall_score, f1_score, classification_report
)

RAW_CSV = "data/ecommerce_returns_synthetic.csv"
CAT_COLS = ["product_category", "payment_method", "shipping_method", "device_type"]
TARGET = "returned"
THRESHOLD = 0.32       # locked from Day 2 val-set cost sweep -- not re-tuned here
FP_COST, FN_COST = 30, 300

# ---------------------------------------------------------------------
# 1. Recreate the exact same split as training (same random_state),
#    so X_test/y_test here is identical to the holdout set that was
#    set aside from the very start.
# ---------------------------------------------------------------------
df = pd.read_csv(RAW_CSV)
X = df.drop(columns=[TARGET, "order_id"])
y = df[TARGET]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

for c in CAT_COLS:
    X_test[c] = X_test[c].astype("category")

print(f"Holdout set size: {len(X_test):,}  (never evaluated before this run)")

# ---------------------------------------------------------------------
# 2. Load the already-trained model (no retraining, no re-tuning)
# ---------------------------------------------------------------------
model = lgb.Booster(model_file="models/model_final.txt")
test_prob = model.predict(X_test)
test_pred = (test_prob >= THRESHOLD).astype(int)

# ---------------------------------------------------------------------
# 3. Report metrics -- exactly as they land, no adjustment
# ---------------------------------------------------------------------
tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
total_cost = fp * FP_COST + fn * FN_COST

holdout_metrics = {
    "n_orders": int(len(X_test)),
    "threshold_used": THRESHOLD,
    "roc_auc": round(roc_auc_score(y_test, test_prob), 4),
    "pr_auc": round(average_precision_score(y_test, test_prob), 4),
    "precision": round(precision_score(y_test, test_pred), 4),
    "recall": round(recall_score(y_test, test_pred), 4),
    "f1": round(f1_score(y_test, test_pred), 4),
    "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    "total_cost_inr": int(total_cost),
}

print("\n=== HELD-OUT TEST SET METRICS (final, reportable numbers) ===")
for k, v in holdout_metrics.items():
    print(f"{k:18s}: {v}")

print("\n=== Classification report ===")
print(classification_report(y_test, test_pred))

# ---------------------------------------------------------------------
# 4. Compare against validation metrics to check for drift/overfitting
# ---------------------------------------------------------------------
with open("results/final_metrics.json") as f:
    val_metrics = json.load(f)

print("\n=== Validation vs. Holdout comparison ===")
print(f"{'metric':12s}{'validation':>12s}{'holdout':>12s}{'delta':>10s}")
for k in ["roc_auc", "pr_auc", "precision", "recall", "f1"]:
    v, h = val_metrics.get(k), holdout_metrics.get(k)
    print(f"{k:12s}{v:>12.4f}{h:>12.4f}{h-v:>+10.4f}")

with open("results/holdout_metrics.json", "w") as f:
    json.dump(holdout_metrics, f, indent=2)
print("\nSaved: results/holdout_metrics.json")

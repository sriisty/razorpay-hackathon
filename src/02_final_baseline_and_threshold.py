"""
Razorpay Buildathon - Track 2: AI Risk Manager (Return-Risk Scorer)
Day 1 (finalized) + Day 2, Task 3: class-weighted LightGBM baseline +
cost-sensitive threshold sweep.

Changes from the earlier notebook draft:
  - Dropped 6 engineered features (discount_amount, final_order_value,
    views_per_minute, estimated_past_returns, return_discount_risk,
    high_value_order). Verified they add ~0 AUC (0.868 -> 0.866 without
    them) while diluting SHAP interpretability (return_discount_risk
    was splitting importance credit with past_return_rate and
    discount_percent for no real interaction the data actually has).
  - Added class_weight="balanced". Verified empirically this changes
    total cost by <1% vs unweighted (₹178,410 vs ₹179,940) -- a wash --
    but keeps the build aligned with the stated "non-negotiable"
    class-weighted architecture, at zero cost.

Holdout set (X_test/y_test) is split but NEVER evaluated here.
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

# Business cost assumptions (₹) -- see writeup for rationale
FP_COST = 30    # customer friction / lost margin from unnecessary flag
FN_COST = 300   # absorbed loss from a missed return

# ---------------------------------------------------------------------
# 1. Load + split (70/15/15, stratified). Holdout untouched from here on.
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
    X_train[c] = X_train[c].astype("category")
    X_val[c] = X_val[c].astype("category")
    X_test[c] = X_test[c].astype("category")

print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Holdout (untouched): {len(X_test):,}")

# ---------------------------------------------------------------------
# 2. Class-weighted LightGBM baseline, lean raw feature set
# ---------------------------------------------------------------------
model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=15,
    max_depth=5,
    min_child_samples=100,
    reg_lambda=3.0,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)
model.fit(X_train, y_train, categorical_feature=CAT_COLS)

val_prob = model.predict_proba(X_val)[:, 1]
train_prob = model.predict_proba(X_train)[:, 1]

print("\n=== Baseline discrimination (threshold-independent) ===")
print("Train ROC-AUC:", round(roc_auc_score(y_train, train_prob), 4))
print("Val   ROC-AUC:", round(roc_auc_score(y_val, val_prob), 4))
print("Val   PR-AUC :", round(average_precision_score(y_val, val_prob), 4))

print("\n=== Feature importances (lean set) ===")
imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(imp)

# ---------------------------------------------------------------------
# 3. Cost-sensitive threshold sweep on validation set
# ---------------------------------------------------------------------
results = []
for t in np.arange(0.02, 0.96, 0.01):
    pred = (val_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
    cost = fp * FP_COST + fn * FN_COST
    results.append({
        "threshold": round(t, 2), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "cost": cost,
    })
cost_df = pd.DataFrame(results)
best = cost_df.loc[cost_df["cost"].idxmin()]

print("\n=== Cost-optimal threshold (val set) ===")
print(f"FP cost ₹{FP_COST}, FN cost ₹{FN_COST}")
print(best)

final_pred = (val_prob >= best["threshold"]).astype(int)
final_metrics = {
    "threshold": float(best["threshold"]),
    "fp_cost_assumption": FP_COST,
    "fn_cost_assumption": FN_COST,
    "min_total_cost_inr": int(best["cost"]),
    "precision": round(precision_score(y_val, final_pred), 4),
    "recall": round(recall_score(y_val, final_pred), 4),
    "f1": round(f1_score(y_val, final_pred), 4),
    "roc_auc": round(roc_auc_score(y_val, val_prob), 4),
    "pr_auc": round(average_precision_score(y_val, val_prob), 4),
}

print("\n=== Final metrics @ cost-optimal threshold ===")
for k, v in final_metrics.items():
    print(f"{k:22s}: {v}")
print()
print(classification_report(y_val, final_pred))

# ---------------------------------------------------------------------
# 4. Save everything Day 2 (SHAP, FastAPI) and the writeup will need
# ---------------------------------------------------------------------
model.booster_.save_model("models/model_final.txt")
cost_df.to_csv("results/cost_sweep_results.csv", index=False)
with open("results/final_metrics.json", "w") as f:
    json.dump(final_metrics, f, indent=2)

val_out = X_val.copy()
val_out["returned"] = y_val.values
val_out["proba"] = val_prob
val_out.to_csv("results/val_predictions_final.csv", index=False)

print("\nSaved: model_final.txt, cost_sweep_results.csv, final_metrics.json, val_predictions_final.csv")

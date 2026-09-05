"""
Razorpay Buildathon - Track 2: AI Risk Manager (Return-Risk Scorer)
Day 2, Task 4: SHAP explainability.

Built as reusable functions (not a one-off script) because the same
"score an order + return top contributing factors" logic is exactly
what the FastAPI endpoint (Task 5) and the RAG explainer (differentiator
layer) both need. Import `explain_order` / `explain_batch` directly from
there rather than duplicating this logic.

Loads model_final.txt (the trained Booster) -- no retraining here.
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap

MODEL_PATH = "models/model_final.txt"
THRESHOLD = 0.32  # locked in from the Day 2 cost-sensitive sweep
CAT_COLS = ["product_category", "payment_method", "shipping_method", "device_type"]


def load_model(path: str = MODEL_PATH) -> lgb.Booster:
    return lgb.Booster(model_file=path)


def build_explainer(model: lgb.Booster) -> shap.TreeExplainer:
    return shap.TreeExplainer(model)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


def _to_native(val):
    """Convert numpy/pandas scalar types to plain Python types so
    FastAPI/Pydantic can JSON-serialize them. Without this, integer-typed
    columns (customer_age, delivery_delay_days, past_purchase_count,
    num_product_views) come back as numpy.int64, which pydantic_core
    cannot serialize -> intermittent 500s depending on which features
    SHAP ranks into the top_k for a given order.
    """
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    return str(val) if isinstance(val, pd.Categorical) else val


def explain_batch(df: pd.DataFrame, model: lgb.Booster, explainer: shap.TreeExplainer,
                   top_k: int = 3) -> pd.DataFrame:
    """
    df: feature rows (no target column). Returns a DataFrame with the
    original index, risk_score, flag, and top_k contributing features
    as (name, shap_value, direction) tuples, ready to serialize to JSON
    for the FastAPI response.
    """
    X = _prep(df)
    scores = model.predict(X)
    shap_values = explainer.shap_values(X)

    records = []
    for i in range(len(X)):
        row_shap = shap_values[i]
        order = np.argsort(-np.abs(row_shap))[:top_k]
        top_features = [
            {
                "feature": X.columns[j],
                "value": _to_native(X.iloc[i][X.columns[j]]),
                "shap_value": round(float(row_shap[j]), 4),
                "direction": "increases_risk" if row_shap[j] > 0 else "decreases_risk",
            }
            for j in order
        ]
        records.append({
            "risk_score": round(float(scores[i]), 4),
            "flag": bool(scores[i] >= THRESHOLD),
            "top_features": top_features,
        })
    return pd.DataFrame(records)


def explain_order(order_features: dict, model: lgb.Booster, explainer: shap.TreeExplainer,
                   top_k: int = 3) -> dict:
    """Single-order convenience wrapper -- what the FastAPI endpoint calls directly."""
    df = pd.DataFrame([order_features])
    result = explain_batch(df, model, explainer, top_k=top_k)
    return result.iloc[0].to_dict()


if __name__ == "__main__":
    model = load_model()
    explainer = build_explainer(model)

    val = pd.read_csv("results/val_predictions_final.csv")
    feature_cols = [c for c in val.columns if c not in ["returned", "proba"]]

    # Demo on the 5 highest-scored flagged orders and 3 correctly-cleared ones
    sample = val.sort_values("proba", ascending=False).head(5)
    explained = explain_batch(sample[feature_cols], model, explainer, top_k=3)

    print("=== Sample explained orders (highest risk) ===")
    for i, row in explained.iterrows():
        print(f"\nRisk score: {row['risk_score']}  Flagged: {row['flag']}")
        for feat in row["top_features"]:
            print(f"  - {feat['feature']} = {feat['value']}  "
                  f"(SHAP {feat['shap_value']:+.4f}, {feat['direction']})")

    # Global importance (mean |SHAP|) for the writeup/pitch
    X_all = _prep(val[feature_cols])
    shap_all = explainer.shap_values(X_all)
    global_imp = pd.Series(np.abs(shap_all).mean(axis=0), index=feature_cols).sort_values(ascending=False)
    print("\n=== Global mean |SHAP| (validation set) ===")
    print(global_imp)

    global_imp.to_csv("results/shap_global_importance.csv", header=["mean_abs_shap"])
    print("\nSaved: results/shap_global_importance.csv")
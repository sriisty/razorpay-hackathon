# Return-Risk Manager

**Razorpay Buildathon — Track 2: AI Risk Manager**

Predicts which e-commerce orders are likely to be returned, using a cost-sensitive
decision threshold (not just accuracy) so the model optimizes for actual ₹ business
cost rather than a generic classification metric.

---

## The problem

Returns cost money on both sides of a wrong call:
- **Flag a good order** → customer friction, lost margin (false positive)
- **Miss a real return** → the full absorbed loss (false negative)

A model tuned for F1 or accuracy ignores this asymmetry. This system scores every
order, then picks its flagging threshold by directly minimizing expected ₹ cost.

---

## Dataset

Originally sourced from a Kaggle e-commerce returns dataset, but validated
(logistic regression achieved nearly identical AUC to LightGBM, ~0.59) that it
carried very little real signal and contained physically impossible values
(negative prices, negative view counts). Replaced with a **custom-generated
synthetic dataset** (`generate_dataset.py`) built from an explicit, documented
causal structure:

- 80,000 orders, 17.2% return rate
- Label generated via a latent logistic risk score + Bernoulli sampling (not a
  hard rule), so the achievable AUC ceiling is realistic rather than trivial
- Dominant driver: `past_return_rate`, followed by `discount_percent`,
  `product_rating`, `product_category`, `payment_method` (COD higher risk),
  `delivery_delay_days`
- Several genuinely uninformative "noise" columns included on purpose
  (`device_type`, `customer_age`, `session_length_minutes`) to stress-test that
  SHAP correctly learns to ignore them

All coefficients are explicit in `generate_dataset.py` — fully auditable, not a
black box.

---

## Model & results

**Class-weighted LightGBM**, trained on a 70/15/15 stratified train/val/holdout
split. Holdout was never touched until final evaluation.

| Metric | Validation | Held-out (final) |
|---|---|---|
| ROC-AUC | 0.868 | 0.867 |
| PR-AUC | 0.640 | 0.631 |
| Precision @ threshold | 0.345 | 0.340 |
| Recall @ threshold | 0.880 | 0.880 |
| F1 | 0.495 | 0.491 |
| Total cost | ₹178,410 | ₹179,700 |

Validation → holdout delta is under 0.01 on every metric — the model generalizes
cleanly, not overfit to validation quirks.

### Cost-sensitive threshold

- False positive cost: ₹30 (customer friction / lost margin)
- False negative cost: ₹300 (absorbed return loss)
- Threshold swept 0.02–0.95; **0.32** minimizes total expected cost
- The cost curve is a shallow bowl near the optimum (not a sharp spike) — the
  business isn't highly sensitive to hitting exactly 0.32 vs. 0.30

### Explainability

SHAP (`TreeExplainer`) generates per-order feature attributions. Global
importance ranking on validation confirms the model recovered the intended
causal structure (`past_return_rate` dominates by >3x over the next feature).

---

## Architecture

```
data/        Dataset (generate_dataset.py output)
models/      Trained LightGBM booster (model_final.txt)
results/     Metrics, cost sweep, SHAP importance, predictions
src/
  02_final_baseline_and_threshold.py   Train model + cost-sensitive sweep
  explain.py                           SHAP explainability (reusable module)
  app.py                               FastAPI serving layer
  04_holdout_eval.py                   Final held-out evaluation
streamlit_app.py   Demo UI (calls the FastAPI endpoint)
.streamlit/config.toml   Dark theme
```

`explain.py` is deliberately built as importable functions
(`load_model`, `build_explainer`, `explain_order`) rather than a script, since
both the FastAPI endpoint and any future explainer agent need the same
score-and-explain logic.

---

## Running it

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Start the API** (from project root)
```bash
uvicorn app:app --app-dir src --port 8000
```
Interactive docs at `http://127.0.0.1:8000/docs`.

**3. (Optional) Start the demo UI**, in a second terminal
```bash
streamlit run streamlit_app.py
```

**Example request:**
```bash
curl -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "customer_age": 25, "order_value": 4999.0, "product_category": "apparel",
    "discount_percent": 45.0, "used_coupon": 1, "product_rating": 2.5,
    "past_purchase_count": 8, "past_return_rate": 0.70, "delivery_delay_days": 4,
    "payment_method": "cod", "shipping_method": "standard", "device_type": "mobile",
    "session_length_minutes": 8.0, "num_product_views": 12
  }'
```
```json
{
  "risk_score": 0.9955,
  "flag": true,
  "top_features": [
    {"feature": "past_return_rate", "value": 0.7, "shap_value": 3.6381, "direction": "increases_risk"},
    {"feature": "discount_percent", "value": 45.0, "shap_value": 0.9145, "direction": "increases_risk"},
    {"feature": "payment_method", "value": "cod", "shap_value": 0.4776, "direction": "increases_risk"}
  ]
}
```

---

## What's next

The scoped extension (not built due to time) is a **RAG-grounded explainer
agent**: retrieve relevant merchant return-policy text and combine it with the
SHAP output to answer natural-language questions like "why was this flagged"
or "what if we changed the policy for this segment," reusing a LangChain
retrieval chain (and optionally LangGraph for multi-turn follow-ups).

## Scope notes

- Defense-only: this system flags risk for internal review, it does not make
  or block purchase decisions, and contains no fraud-enabling logic.
- Dataset is synthetic, built to mirror realistic e-commerce return patterns
  rather than drawn from live production data.

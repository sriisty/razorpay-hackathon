"""
Razorpay Buildathon - Track 2: AI Risk Manager (Return-Risk Scorer)
Day 2, Task 5: FastAPI endpoint.

Pure serving layer -- no modeling logic lives here. Model loading, SHAP
explanation, and thresholding all come from explain.py (Task 4). This
file just wraps that in a POST route.

Run from the project root:
    uvicorn app:app --reload --app-dir src
(or `cd src && uvicorn app:app --reload` -- either works as long as the
relative paths models/... and results/... resolve from the project root,
so prefer running uvicorn FROM the project root with --app-dir src)
"""

from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field

from shap_explain import load_model, build_explainer, explain_order

app = FastAPI(
    title="Return-Risk Scorer",
    description="Predicts return risk for an order, with SHAP-based explanation.",
    version="1.0.0",
)

# Loaded once at startup, not per-request.
_model = None
_explainer = None


@app.on_event("startup")
def _load_artifacts():
    global _model, _explainer
    _model = load_model()
    _explainer = build_explainer(_model)


class OrderFeatures(BaseModel):
    customer_age: int = Field(..., ge=10, le=100)
    order_value: float = Field(..., gt=0)
    product_category: Literal["apparel", "electronics", "home", "beauty", "sports", "toys"]
    discount_percent: float = Field(..., ge=0, le=100)
    used_coupon: int = Field(..., ge=0, le=1)
    product_rating: float = Field(..., ge=1, le=5)
    past_purchase_count: int = Field(..., ge=0)
    past_return_rate: float = Field(..., ge=0, le=1)
    delivery_delay_days: int = Field(..., ge=0)
    payment_method: Literal["upi", "card", "netbanking", "cod", "wallet"]
    shipping_method: Literal["standard", "express"]
    device_type: Literal["mobile", "desktop", "tablet"]
    session_length_minutes: float = Field(..., ge=0)
    num_product_views: int = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "customer_age": 29,
                "order_value": 2499.0,
                "product_category": "apparel",
                "discount_percent": 45.0,
                "used_coupon": 1,
                "product_rating": 2.8,
                "past_purchase_count": 6,
                "past_return_rate": 0.55,
                "delivery_delay_days": 2,
                "payment_method": "cod",
                "shipping_method": "standard",
                "device_type": "mobile",
                "session_length_minutes": 12.5,
                "num_product_views": 8,
            }
        }


class TopFeature(BaseModel):
    feature: str
    value: object
    shap_value: float
    direction: Literal["increases_risk", "decreases_risk"]


class ScoreResponse(BaseModel):
    risk_score: float
    flag: bool
    top_features: list[TopFeature]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/score", response_model=ScoreResponse)
def score_order(order: OrderFeatures):
    result = explain_order(order.model_dump(), _model, _explainer, top_k=3)
    return result

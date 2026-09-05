import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/score"

st.set_page_config(page_title="Return-Risk Scorer", layout="centered")
st.title("Return-Risk Scorer")

PRESETS = {
    "High risk (COD + heavy discount + bad history)": {
        "customer_age": 25, "order_value": 4999.0, "product_category": "apparel",
        "discount_percent": 45.0, "used_coupon": 1, "product_rating": 2.5,
        "past_purchase_count": 8, "past_return_rate": 0.70, "delivery_delay_days": 4,
        "payment_method": "cod", "shipping_method": "standard", "device_type": "mobile",
        "session_length_minutes": 8.0, "num_product_views": 12,
    },
    "Low risk (clean history, electronics)": {
        "customer_age": 40, "order_value": 8000.0, "product_category": "electronics",
        "discount_percent": 5.0, "used_coupon": 0, "product_rating": 4.6,
        "past_purchase_count": 20, "past_return_rate": 0.03, "delivery_delay_days": 0,
        "payment_method": "upi", "shipping_method": "express", "device_type": "desktop",
        "session_length_minutes": 6.0, "num_product_views": 3,
    },
}

choice = st.selectbox("Load a sample order", list(PRESETS.keys()))
order = PRESETS[choice]

with st.form("order_form"):
    c1, c2 = st.columns(2)
    with c1:
        order["customer_age"] = st.number_input("Customer age", 10, 100, order["customer_age"])
        order["order_value"] = st.number_input("Order value (₹)", 0.0, value=order["order_value"])
        order["product_category"] = st.selectbox(
            "Category", ["apparel", "electronics", "home", "beauty", "sports", "toys"],
            index=["apparel", "electronics", "home", "beauty", "sports", "toys"].index(order["product_category"]))
        order["discount_percent"] = st.number_input("Discount %", 0.0, 100.0, order["discount_percent"])
        order["used_coupon"] = st.selectbox("Used coupon", [0, 1], index=order["used_coupon"])
        order["product_rating"] = st.number_input("Product rating", 1.0, 5.0, order["product_rating"])
        order["past_purchase_count"] = st.number_input("Past purchase count", 0, value=order["past_purchase_count"])
    with c2:
        order["past_return_rate"] = st.number_input("Past return rate", 0.0, 1.0, order["past_return_rate"])
        order["delivery_delay_days"] = st.number_input("Delivery delay (days)", 0, value=order["delivery_delay_days"])
        order["payment_method"] = st.selectbox(
            "Payment method", ["upi", "card", "netbanking", "cod", "wallet"],
            index=["upi", "card", "netbanking", "cod", "wallet"].index(order["payment_method"]))
        order["shipping_method"] = st.selectbox(
            "Shipping method", ["standard", "express"],
            index=["standard", "express"].index(order["shipping_method"]))
        order["device_type"] = st.selectbox(
            "Device type", ["mobile", "desktop", "tablet"],
            index=["mobile", "desktop", "tablet"].index(order["device_type"]))
        order["session_length_minutes"] = st.number_input("Session length (min)", 0.0, value=order["session_length_minutes"])
        order["num_product_views"] = st.number_input("Num product views", 0, value=order["num_product_views"])

    submitted = st.form_submit_button("Score this order")

if submitted:
    try:
        resp = requests.post(API_URL, json=order, timeout=5)
        resp.raise_for_status()
        result = resp.json()

        score = result["risk_score"]
        flagged = result["flag"]

        if flagged:
            st.error(f"FLAGGED — Risk score: {score:.3f}")
        else:
            st.success(f"CLEARED — Risk score: {score:.3f}")

        st.subheader("Top contributing factors")
        for feat in result["top_features"]:
            arrow = "🔺" if feat["direction"] == "increases_risk" else "🔻"
            st.write(f"{arrow} **{feat['feature']}** = {feat['value']}  (SHAP {feat['shap_value']:+.4f})")

    except Exception as e:
        st.error(f"Request failed — is the FastAPI server running on :8000? ({e})")

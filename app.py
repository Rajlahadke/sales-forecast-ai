import bz2

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Sales Forecast AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_FILE = "best_sales_forecast_model.pkl.bz2"

@st.cache_resource
def load_model():
    with bz2.open(MODEL_FILE, "rb") as model_file:
        return joblib.load(model_file)

try:
    model = load_model()
except Exception as e:
    st.error("Model could not be loaded.")
    st.code(str(e))
    st.stop()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(99,102,241,0.18), transparent 28%),
        radial-gradient(circle at 90% 15%, rgba(14,165,233,0.14), transparent 25%),
        #080b14;
    color: #f8fafc;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 35px;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    background: rgba(255,255,255,0.035);
    margin-bottom: 25px;
}

.badge {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 999px;
    background: rgba(99,102,241,0.14);
    border: 1px solid rgba(129,140,248,0.25);
    color: #a5b4fc;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}

.hero h1 {
    font-size: 46px;
    font-weight: 800;
    margin: 15px 0 8px 0;
    color: #ffffff;
}

.hero p {
    color: #94a3b8;
    font-size: 16px;
    margin: 0;
}

.card {
    padding: 22px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.07);
    background: rgba(255,255,255,0.035);
}

.card-title {
    color: #94a3b8;
    font-size: 13px;
    margin-bottom: 7px;
}

.card-value {
    color: #ffffff;
    font-size: 23px;
    font-weight: 700;
}

.section-title {
    font-size: 24px;
    font-weight: 750;
    margin: 25px 0 15px 0;
}

.result-box {
    padding: 30px;
    border-radius: 22px;
    background: linear-gradient(
        135deg,
        rgba(99,102,241,0.18),
        rgba(14,165,233,0.10)
    );
    border: 1px solid rgba(129,140,248,0.25);
    text-align: center;
    margin-top: 25px;
}

.result-label {
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 8px;
}

.result-number {
    font-size: 52px;
    font-weight: 800;
    color: #ffffff;
}

.result-unit {
    color: #a5b4fc;
    font-size: 15px;
}

div[data-testid="stButton"] button {
    width: 100%;
    border-radius: 12px;
    height: 48px;
    font-weight: 700;
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.07);
    padding: 18px;
    border-radius: 16px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <span class="badge">MACHINE LEARNING • SALES ANALYTICS</span>
    <h1>📈 Sales Forecast AI</h1>
    <p>Predict expected product demand using a trained regression model.</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="card">
        <div class="card-title">MODEL</div>
        <div class="card-value">ML Powered</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card">
        <div class="card-title">TARGET</div>
        <div class="card-value">Quantity Sold</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="card">
        <div class="card-title">FORECAST TYPE</div>
        <div class="card-value">Sales Demand</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">📊 Enter Sales Details</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    store_id = st.number_input(
        "Store ID",
        min_value=1,
        max_value=20,
        value=1,
        step=1
    )

    sales_amount = st.number_input(
        "Sales Amount",
        min_value=0.0,
        value=500.0,
        step=10.0
    )

    unit_cost = st.number_input(
        "Unit Cost",
        min_value=0.0,
        value=100.0,
        step=1.0
    )

    unit_price = st.number_input(
        "Unit Price",
        min_value=0.01,
        value=150.0,
        step=1.0
    )

    discount = st.number_input(
        "Discount (%)",
        min_value=0.0,
        max_value=30.0,
        value=5.0,
        step=1.0
    )

    sales_rep = st.selectbox(
        "Sales Representative",
        ["Bob", "Charlie", "David", "Eve"]
    )

with col2:

    month = st.selectbox(
        "Sale Month",
        list(range(1, 13)),
        index=0
    )

    year = st.number_input(
        "Sale Year",
        min_value=2024,
        max_value=2030,
        value=2025,
        step=1
    )

    region = st.selectbox(
        "Region",
        ["North", "South", "West"]
    )

    product_category = st.selectbox(
        "Product Category",
        ["Electronics", "Food", "Furniture"]
    )

    customer_type = st.selectbox(
        "Customer Type",
        ["New", "Returning"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        ["Cash", "Credit Card"]
    )

    sales_channel = st.selectbox(
        "Sales Channel",
        ["Online", "Retail"]
    )

st.markdown("")

if st.button("🔮 Predict Sales Demand"):

    if sales_amount < 0:
        st.error("Sales Amount cannot be negative.")
        st.stop()

    if unit_cost < 0:
        st.error("Unit Cost cannot be negative.")
        st.stop()

    if unit_price <= 0:
        st.error("Unit Price must be greater than 0.")
        st.stop()

    input_data = pd.DataFrame([{
        "Store_ID": store_id,
        "Sales_Amount": sales_amount,
        "Unit_Cost": unit_cost,
        "Unit_Price": unit_price,
        "Discount": discount / 100,
        "Sale_Month": month,
        "Sale_Year": year,
        "Sales_Rep_Bob": int(sales_rep == "Bob"),
        "Sales_Rep_Charlie": int(sales_rep == "Charlie"),
        "Sales_Rep_David": int(sales_rep == "David"),
        "Sales_Rep_Eve": int(sales_rep == "Eve"),
        "Region_North": int(region == "North"),
        "Region_South": int(region == "South"),
        "Region_West": int(region == "West"),
        "Product_Category_Electronics": int(
            product_category == "Electronics"
        ),
        "Product_Category_Food": int(
            product_category == "Food"
        ),
        "Product_Category_Furniture": int(
            product_category == "Furniture"
        ),
        "Customer_Type_Returning": int(
            customer_type == "Returning"
        ),
        "Payment_Method_Cash": int(
            payment_method == "Cash"
        ),
        "Payment_Method_Credit Card": int(
            payment_method == "Credit Card"
        ),
        "Sales_Channel_Retail": int(
            sales_channel == "Retail"
        )
    }])

    try:

        if hasattr(model, "feature_names_in_"):
            input_data = input_data.reindex(
                columns=model.feature_names_in_,
                fill_value=0
            )

        prediction = model.predict(input_data)

        prediction = float(
            np.asarray(prediction).reshape(-1)[0]
        )

        prediction = max(0, prediction)

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">
                PREDICTED SALES QUANTITY
            </div>
            <div class="result-number">
                {prediction:.2f}
            </div>
            <div class="result-unit">
                units expected to be sold
            </div>
        </div>
        """, unsafe_allow_html=True)

        if prediction < 20:
            st.info("📦 Low expected demand")
        elif prediction < 50:
            st.success("📊 Moderate expected demand")
        else:
            st.success("🔥 High expected demand")

    except Exception as e:

        st.error("Prediction failed.")
        st.code(str(e))

st.markdown("---")

st.markdown(
    '<div class="section-title">🤖 Model Information</div>',
    unsafe_allow_html=True
)

m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        "Model",
        type(model).__name__
    )

with m2:
    st.metric(
        "Features",
        getattr(model, "n_features_in_", "N/A")
    )

with m3:
    st.metric(
        "Target",
        "Quantity Sold"
    )

st.caption(
    "Predictions are generated using the trained "
    "best_sales_forecast_model.pkl."
)

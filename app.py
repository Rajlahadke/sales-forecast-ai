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

MODEL_FILE = "best_sales_forecast_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

try:
    model = load_model()
except Exception as e:
    st.error("Model could not be loaded.")
    st.code(str(e))
    st.stop()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Libre+Franklin:wght@600;700&display=swap');

* {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f3f0e8;
    color: #252824;
}

.block-container {
    max-width: 1120px;
    padding-top: 2.2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 34px 36px;
    border: 1px solid #d8d2c6;
    border-radius: 14px;
    background: #faf8f3;
    margin-bottom: 22px;
    box-shadow: 0 6px 20px rgba(47, 45, 39, 0.05);
}

.badge {
    display: inline-block;
    padding: 5px 0;
    color: #8b5e3c;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.6px;
}

.hero h1 {
    font-family: 'Libre Franklin', sans-serif;
    font-size: 42px;
    font-weight: 700;
    margin: 12px 0 8px 0;
    color: #1f2923;
    letter-spacing: -1.2px;
}

.hero p {
    color: #6d7069;
    font-size: 16px;
    margin: 0;
}

.card {
    padding: 20px 21px;
    border-radius: 12px;
    border: 1px solid #ddd7cc;
    background: #fffdf8;
    min-height: 94px;
}

.card-title {
    color: #878178;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 8px;
}

.card-value {
    color: #2c312d;
    font-size: 21px;
    font-weight: 600;
}

.section-title {
    font-family: 'Libre Franklin', sans-serif;
    color: #28332d;
    font-size: 22px;
    font-weight: 700;
    margin: 31px 0 15px 0;
}

.result-box {
    padding: 30px;
    border-radius: 14px;
    background: #e6ece3;
    border: 1px solid #cbd6c7;
    text-align: center;
    margin-top: 24px;
}

.result-label {
    color: #667166;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.1px;
    margin-bottom: 7px;
}

.result-number {
    font-family: 'Libre Franklin', sans-serif;
    font-size: 50px;
    font-weight: 700;
    color: #2e4a3a;
}

.result-unit {
    color: #68766d;
    font-size: 14px;
}

div[data-testid="stButton"] button {
    width: 100%;
    border-radius: 9px;
    height: 48px;
    background: #314c3f;
    color: #fffdf8;
    border: 1px solid #314c3f;
    font-weight: 700;
}

div[data-testid="stButton"] button:hover {
    background: #263d33;
    color: #ffffff;
    border-color: #263d33;
}

div[data-testid="stMetric"] {
    background: #fffdf8;
    border: 1px solid #ddd7cc;
    padding: 18px;
    border-radius: 12px;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #2f342f;
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background: #fffdf8;
    border-color: #d7d1c6;
}

hr {
    border-color: #d8d2c6 !important;
}

[data-testid="stCaptionContainer"] {
    color: #7a776f;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <span class="badge">SALES FORECASTING</span>
    <h1>Sales Forecast</h1>
    <p>Estimate product demand from sales, pricing and customer inputs.</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="card">
        <div class="card-title">MODEL</div>
        <div class="card-value">Regression Model</div>
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
        <div class="card-value">Demand Forecast</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">Enter sales details</div>',
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

if st.button("Predict demand"):

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
                FORECASTED QUANTITY
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
            st.info("Low expected demand")
        elif prediction < 50:
            st.success("Moderate expected demand")
        else:
            st.success("High expected demand")

    except Exception as e:

        st.error("Prediction failed.")
        st.code(str(e))

st.markdown("---")

st.markdown(
    '<div class="section-title">Model details</div>',
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

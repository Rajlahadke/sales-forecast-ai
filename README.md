# Sales Forecast AI

A Streamlit machine-learning app that predicts expected product sales quantity from store, pricing, date, region, product category, customer type, payment method, sales representative, and sales channel inputs.

## Model

The app loads `best_sales_forecast_model.pkl`, a trained `RandomForestRegressor`, and aligns user input to the feature names stored in the model before prediction.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

- Repository: `Rajlahadke/sales-forecast-ai`
- Branch: `main`
- Main file path: `app.py`
- Python version: `3.12`

The model file must remain in the repository root beside `app.py`.

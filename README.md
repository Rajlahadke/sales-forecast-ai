# Sales Forecast AI

A Streamlit machine-learning app that predicts expected product sales quantity from store, pricing, date, region, product category, customer type, payment method, sales representative, and sales channel inputs.

## Model

The app loads the trained RandomForestRegressor from `best_sales_forecast_model.pkl.bz2`. The file is a lossless bzip2-compressed copy of the original `best_sales_forecast_model.pkl`, used only to make binary upload through the GitHub connector reliable.

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

Keep the compressed model file in the repository root beside `app.py`.

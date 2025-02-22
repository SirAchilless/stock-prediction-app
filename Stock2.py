import pandas as pd
import datetime
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Fetch stock data
def fetch_yahoo_finance_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="3mo")
        if data.empty:
            return None
        data = data.reset_index()
        data["ds"] = pd.to_datetime(data["Date"]).dt.tz_localize(None)
        data = data.rename(columns={"Close": "y"})
        data["momentum_5d"] = data["y"].diff(5).fillna(0)
        data["volatility"] = data["y"].rolling(5).std().fillna(0)
        data["cap"] = data["y"].max() * 1.05
        data["floor"] = data["y"].min() * 0.95
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

# Generate next trading days
def get_next_trading_days(start_date, days=21):
    trading_days = []
    while len(trading_days) < days:
        start_date += datetime.timedelta(days=1)
        if start_date.weekday() < 5:
            trading_days.append(start_date)
    return trading_days

# Train & predict using Prophet
def predict_stock_prices(data, days=21):
    try:
        model = Prophet(growth="logistic", weekly_seasonality=True, changepoint_prior_scale=0.3)
        model.add_seasonality(name="quarterly", period=90, fourier_order=10)
        model.add_regressor("momentum_5d")
        model.add_regressor("volatility")
        model.fit(data)
        future_dates = get_next_trading_days(data["ds"].max(), days)
        future = pd.DataFrame({"ds": future_dates})
        future["momentum_5d"] = data["momentum_5d"].iloc[-1]
        future["volatility"] = data["volatility"].iloc[-1]
        future["cap"] = data["cap"].max()
        future["floor"] = data["floor"].min()
        forecast = model.predict(future)
        return forecast[["ds", "yhat"]]
    except Exception as e:
        st.error(f"Error in Prophet prediction: {e}")
        return None

# Streamlit UI
st.title("Stock Price Prediction App")

stock_symbol = st.text_input("Enter Stock Symbol", "AAPL")
if st.button("Predict"):
    stock_data = fetch_yahoo_finance_data(stock_symbol)
    if stock_data is None:
        st.error("Stock data fetch failed.")
    else:
        predictions = predict_stock_prices(stock_data, days=21)
        if predictions is not None:
            st.write("### Predicted Stock Prices:")
            st.dataframe(predictions)
            fig, ax = plt.subplots()
            ax.plot(predictions["ds"], predictions["yhat"], marker="o", linestyle="-", color="b", label="Predicted Price")
            ax.set_xlabel("Date")
            ax.set_ylabel("Stock Price")
            ax.set_title("Stock Price Predictions")
            plt.xticks(rotation=45)
            plt.legend()
            st.pyplot(fig)

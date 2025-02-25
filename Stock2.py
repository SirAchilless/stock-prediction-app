import pandas as pd
import datetime
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import os
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from pandas_ta import momentum, trend, volatility

MODEL_PATH = "prophet_model.pkl"
SCALER_PATH = "scaler.pkl"

# Fetch stock data
def fetch_yahoo_finance_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="5y")  # Increased to 5 years for better training
        if data.empty:
            return None, None
        data = data.reset_index()
        data["ds"] = pd.to_datetime(data["Date"]).dt.tz_localize(None)
        data = data.rename(columns={"Close": "y"})
        
        # Add technical indicators
        data["rsi"] = momentum.rsi(data["y"], length=14).fillna(0)
        data["macd"] = trend.macd(data["y"])["MACD_12_26_9"].fillna(0)
        data["bollinger"] = volatility.bbands(data["y"])["BBP_5_2.0"].fillna(0)
        
        # Scale stock prices
        scaler = MinMaxScaler()
        data[["y", "rsi", "macd", "bollinger"]] = scaler.fit_transform(
            data[["y", "rsi", "macd", "bollinger"]]
        )
        
        # Save scaler for future inverse transformations
        joblib.dump(scaler, SCALER_PATH)
        return data, scaler
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None, None

# Load or train Prophet model
def get_trained_model(data):
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    model.add_regressor("rsi")
    model.add_regressor("macd")
    model.add_regressor("bollinger")
    model.fit(data)
    
    joblib.dump(model, MODEL_PATH)  # Save model for future use
    return model

# Generate future dates
def get_future_dates(last_date, days=15):
    trading_days = []
    while len(trading_days) < days:
        last_date += datetime.timedelta(days=1)
        if last_date.weekday() < 5:
            trading_days.append(last_date)
    return trading_days

# Make predictions
def predict_stock_prices(data, model, scaler, days=15):
    try:
        future_dates = get_future_dates(data["ds"].max(), days)
        future = pd.DataFrame({"ds": future_dates})
        future["rsi"] = data["rsi"].iloc[-1]
        future["macd"] = data["macd"].iloc[-1]
        future["bollinger"] = data["bollinger"].iloc[-1]
        
        forecast = model.predict(future)
        forecast_values = np.hstack((forecast[["yhat"]], np.zeros((len(forecast), 3))))
        forecast["yhat"] = scaler.inverse_transform(forecast_values)[:, 0]
        
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None

# Streamlit UI
st.title("Stock Price Prediction App with Continuous Learning")

stock_symbol = st.text_input("Enter Stock Symbol", "AAPL")
if st.button("Predict"):
    stock_data, scaler = fetch_yahoo_finance_data(stock_symbol)
    if stock_data is None or scaler is None:
        st.error("Stock data fetch failed.")
    else:
        model = get_trained_model(stock_data)
        predictions = predict_stock_prices(stock_data, model, scaler, days=15)
        
        if predictions is not None:
            st.write("### Predicted Stock Prices (Next 15 Days):")
            st.dataframe(predictions)
            
            fig, ax = plt.subplots()
            ax.plot(predictions["ds"], predictions["yhat"], marker="o", linestyle="-", color="b", label="Predicted Price")
            ax.set_xlabel("Date")
            ax.set_ylabel("Stock Price")
            ax.set_title("Stock Price Predictions")
            plt.xticks(rotation=45)
            plt.legend()
            st.pyplot(fig)

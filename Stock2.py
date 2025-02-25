import pandas as pd
import datetime
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from prophet import Prophet
from prophet.diagnostics import cross_validation
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error
import talib

# Fetch stock data
def fetch_yahoo_finance_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="5y")
    if data.empty:
        return None, None
    
    data = data.reset_index()
    data["ds"] = pd.to_datetime(data["Date"]).dt.tz_localize(None)
    data = data.rename(columns={"Close": "y"})
    
    # Technical indicators
    data["RSI"] = talib.RSI(data["y"], timeperiod=14).fillna(0)
    data["MACD"], _, _ = talib.MACD(data["y"], fastperiod=12, slowperiod=26, signalperiod=9)
    data["Upper_Band"], data["Middle_Band"], data["Lower_Band"] = talib.BBANDS(data["y"], timeperiod=20)
    data.fillna(0, inplace=True)
    
    scaler = MinMaxScaler()
    features = ["y", "RSI", "MACD", "Upper_Band", "Middle_Band", "Lower_Band"]
    data[features] = scaler.fit_transform(data[features])
    
    return data, scaler

# Generate next trading days
def get_next_trading_days(start_date, days=15):
    trading_days = []
    while len(trading_days) < days:
        start_date += datetime.timedelta(days=1)
        if start_date.weekday() < 5:
            trading_days.append(start_date)
    return trading_days

# Train & predict using Prophet
def predict_with_prophet(data, scaler, days=15):
    model = Prophet(daily_seasonality=True, changepoint_prior_scale=0.1)
    model.add_regressor("RSI")
    model.add_regressor("MACD")
    model.fit(data)
    
    future_dates = get_next_trading_days(data["ds"].max(), days)
    future = pd.DataFrame({"ds": future_dates, "RSI": data["RSI"].iloc[-1], "MACD": data["MACD"].iloc[-1]})
    forecast = model.predict(future)
    forecast_values = np.hstack((forecast[["yhat"]], np.zeros((len(forecast), len(data.columns) - 2))))
    forecast["yhat"] = scaler.inverse_transform(forecast_values)[:, 0]
    
    return forecast[["ds", "yhat"]]

# LSTM Model
def train_lstm(data):
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    
    X_train, y_train = np.array(train.drop(["ds", "y"], axis=1)), np.array(train["y"])
    X_test, y_test = np.array(test.drop(["ds", "y"], axis=1)), np.array(test["y"])
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test), verbose=1)
    
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    st.write(f"LSTM Model MAE: {mae}")
    
    return model

# Streamlit UI
st.title("Enhanced Stock Price Prediction App")

stock_symbol = st.text_input("Enter Stock Symbol", "AAPL")
if st.button("Predict"):
    stock_data, scaler = fetch_yahoo_finance_data(stock_symbol)
    if stock_data is None or scaler is None:
        st.error("Stock data fetch failed.")
    else:
        predictions_prophet = predict_with_prophet(stock_data, scaler, days=15)
        lstm_model = train_lstm(stock_data)
        
        st.write("### Prophet Predictions (15 Days Ahead):")
        st.dataframe(predictions_prophet)
        
        fig, ax = plt.subplots()
        ax.plot(stock_data["ds"].tail(30), stock_data["y"].tail(30), marker="o", linestyle="-", color="g", label="Actual Price")
        ax.plot(predictions_prophet["ds"], predictions_prophet["yhat"], marker="o", linestyle="-", color="b", label="Prophet Prediction")
        
        ax.set_xlabel("Date")
        ax.set_ylabel("Stock Price")
        ax.set_title("Stock Price Actuals & Predictions")
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(fig)

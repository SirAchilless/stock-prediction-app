import pandas as pd
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from prophet import Prophet

# Fetch stock data
def fetch_yahoo_finance_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="3mo")
        if data.empty:
            return None
        data = data.reset_index()
        data["ds"] = pd.to_datetime(data["Date"]).dt.tz_localize(None)
        data = data.rename(columns={"Close": "Price"})
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

# Generate next trading days
def get_next_trading_days(start_date, days=15):
    trading_days = []
    while len(trading_days) < days:
        start_date += datetime.timedelta(days=1)
        if start_date.weekday() < 5:
            trading_days.append(start_date)
    return trading_days

# Train & predict using Prophet
def predict_stock_prices(data, days=15):
    try:
        model = Prophet(daily_seasonality=True)
        model.fit(data.rename(columns={"Price": "y"}))
        future_dates = get_next_trading_days(data["ds"].max(), days)
        future = pd.DataFrame({"ds": future_dates})
        forecast = model.predict(future)
        forecast = forecast[["ds", "yhat"]].rename(columns={"yhat": "Price"})
        return forecast
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
        last_7_days = stock_data.tail(7)  # Get last 7 days of actual prices
        predictions = predict_stock_prices(stock_data, days=15)  # Predict next 15 days
        if predictions is not None:
            combined_data = pd.concat([last_7_days, predictions])
            combined_data = combined_data.rename(columns={"ds": "Date"})
            
            st.write("### Stock Prices (Actual & Predicted):")
            st.dataframe(combined_data)
            
            fig, ax = plt.subplots()
            ax.plot(combined_data["Date"], combined_data["Price"], marker="o", linestyle="-", color="b", label="Stock Price")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.set_title("Stock Price Trends")
            plt.xticks(rotation=45)
            plt.legend()
            st.pyplot(fig)

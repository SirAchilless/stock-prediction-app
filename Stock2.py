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
        data = data.rename(columns={"Close": "y"})
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
        model.fit(data)
        future_dates = get_next_trading_days(data["ds"].max(), days)
        future = pd.DataFrame({"ds": future_dates})
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
        last_7_days = stock_data.tail(7)  # Get last 7 days of actual prices
        predictions = predict_stock_prices(stock_data, days=15)  # Predict next 15 days
        if predictions is not None:
            st.write("### Last 7 Days Actual Prices:")
            st.dataframe(last_7_days[["ds", "y"]])
            
            st.write("### Predicted Stock Prices for Next 15 Days:")
            st.dataframe(predictions)
            
            fig, ax = plt.subplots()
            ax.plot(last_7_days["ds"], last_7_days["y"], marker="o", linestyle="-", color="g", label="Actual Price (Last 7 Days)")
            ax.plot(predictions["ds"], predictions["yhat"], marker="o", linestyle="-", color="b", label="Predicted Price (Next 15 Days)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Stock Price")
            ax.set_title("Stock Price Trends")
            plt.xticks(rotation=45)
            plt.legend()
            st.pyplot(fig)

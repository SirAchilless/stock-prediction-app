import pandas as pd
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.preprocessing import StandardScaler

# Fetch stock data
def fetch_yahoo_finance_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="6mo")
        if data.empty:
            return None, None
        data = data.reset_index()
        data["ds"] = pd.to_datetime(data["Date"]).dt.tz_localize(None)
        data = data.rename(columns={"Close": "y"})
        
        # Add market sentiment indicators
        data["momentum_5d"] = data["y"].diff(5).fillna(0)
        data["volatility"] = data["y"].rolling(5).std().fillna(0)
        
        # Scale stock prices using StandardScaler
        scaler = StandardScaler()
        data[["y", "momentum_5d", "volatility"]] = scaler.fit_transform(data[["y", "momentum_5d", "volatility"]])
        
        return data, scaler
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None, None

# Generate next trading days
def get_next_trading_days(start_date, days=15):
    trading_days = []
    while len(trading_days) < days:
        start_date += datetime.timedelta(days=1)
        if start_date.weekday() < 5:
            trading_days.append(start_date)
    return trading_days

# Train & predict using Prophet
def predict_stock_prices(data, scaler, days=15):
    try:
        model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True, changepoint_prior_scale=0.2)
        model.add_regressor("momentum_5d")
        model.add_regressor("volatility")
        model.fit(data)
        
        future_dates = get_next_trading_days(data["ds"].max(), days)
        future = pd.DataFrame({"ds": future_dates})
        future["momentum_5d"] = data["momentum_5d"].iloc[-1]
        future["volatility"] = data["volatility"].iloc[-1]
        
        forecast = model.predict(future)
        
        # Denormalize predictions for stock prices only
        forecast["yhat"] = scaler.inverse_transform(forecast[["yhat"]].values.reshape(-1, 1)).flatten()
        
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    except Exception as e:
        st.error(f"Error in Prophet prediction: {e}")
        return None

# Streamlit UI
st.title("Stock Price Prediction App")

stock_symbol = st.text_input("Enter Stock Symbol", "AAPL")
if st.button("Predict"):
    stock_data, scaler = fetch_yahoo_finance_data(stock_symbol)
    if stock_data is None or scaler is None:
        st.error("Stock data fetch failed.")
    else:
        last_7_days = stock_data.tail(7).copy()
        last_7_days[["y", "momentum_5d", "volatility"]] = scaler.inverse_transform(last_7_days[["y", "momentum_5d", "volatility"]])
        predictions = predict_stock_prices(stock_data, scaler, days=15)
        if predictions is not None:
            st.write("### Last 7 Days Actual Prices:")
            st.dataframe(last_7_days[["ds", "y"]])
            
            st.write("### Predicted Stock Prices (Next 15 Days):")
            st.dataframe(predictions)
            
            fig, ax = plt.subplots()
            ax.plot(last_7_days["ds"], last_7_days["y"], marker="o", linestyle="-", color="g", label="Actual Price")
            ax.plot(predictions["ds"], predictions["yhat"], marker="o", linestyle="-", color="b", label="Predicted Price")
            
            ax.set_xlabel("Date")
            ax.set_ylabel("Stock Price")
            ax.set_title("Stock Price Actuals & Predictions")
            plt.xticks(rotation=45)
            plt.legend()
            st.pyplot(fig)

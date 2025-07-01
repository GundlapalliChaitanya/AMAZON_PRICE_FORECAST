import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Amazon Price Forecast", layout="centered")

st.title("📦 Amazon Product Price Forecast")
st.write("This app predicts future prices of an Amazon product using machine learning.")

df = pd.read_csv("data/amazon_prices.csv")
forecast = pd.read_csv("data/price_forecast.csv")

df['Date'] = pd.to_datetime(df['Date'])
forecast['Date'] = pd.to_datetime(forecast['Date'])

st.subheader("📊 Historical Prices")
st.line_chart(df.set_index("Date")["Price"])

st.subheader("🔮 Forecasted Prices")
st.line_chart(forecast.set_index("Date")["Forecasted_Price"])

if st.button("Show Combined Plot"):
    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["Price"], label="Actual Price")
    ax.plot(forecast["Date"], forecast["Forecasted_Price"], linestyle="--", label="Forecasted")
    ax.set_title("Actual vs Forecasted Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (INR)")
    ax.legend()
    st.pyplot(fig)

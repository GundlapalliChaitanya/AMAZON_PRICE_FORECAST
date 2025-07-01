# amazon_price_forecast.py

import os
import pandas as pd
import numpy as np
import datetime
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Ensure necessary folders exist ---
os.makedirs("model", exist_ok=True)
os.makedirs("data", exist_ok=True)

# --- Load and preprocess data ---
def preprocess_data():
    csv_path = os.path.join('data', 'amazon_prices.csv')
    if not os.path.isfile(csv_path):
        print("Error: CSV file not found.")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', 'Price'])
    df['Day'] = df['Date'].dt.dayofyear
    return df

# --- Train model ---
def train_model(df):
    if df.empty:
        print("Dataset is empty! Please check your CSV.")
        return

    X = df[['Day']]
    y = df['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if len(X_train) == 0:
        print("Not enough data to train the model.")
        return

    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)

    joblib.dump(model, 'model/price_forecast_model.pkl')

    y_pred = model.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# --- Forecast future prices ---
def forecast_next_days(days=7):
    model_path = 'model/price_forecast_model.pkl'
    if not os.path.isfile(model_path):
        print("Trained model not found. Please run training first.")
        return

    model = joblib.load(model_path)
    today = datetime.datetime.now()
    future_days = np.array([[today.timetuple().tm_yday + i] for i in range(1, days + 1)])
    predictions = model.predict(future_days)
    future_dates = [(today + datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, days + 1)]

    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted_Price': predictions})
    forecast_df.to_csv('data/price_forecast.csv', index=False)
    print(forecast_df)

# --- Visualization ---
def visualize_forecast():
    df = preprocess_data()
    forecast_path = 'data/price_forecast.csv'
    if not os.path.isfile(forecast_path):
        print("Forecast file not found. Run forecast step first.")
        return

    forecast_df = pd.read_csv(forecast_path)
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Date', y='Price', label='Actual Price')
    sns.lineplot(data=forecast_df, x='Date', y='Forecasted_Price', label='Forecasted Price', linestyle='--')
    plt.title('Amazon Product Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price (INR)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.savefig('data/price_forecast_plot.png')
    plt.show()

# --- Main execution ---
if __name__ == '__main__':
    df = preprocess_data()
    train_model(df)
    forecast_next_days(days=7)
    visualize_forecast()

import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime

HEADERS = {"User-Agent": "Mozilla/5.0"}
PRODUCT_URL = 'https://www.amazon.in/dp/B0CSY8H9VT'
DATA_DIR = 'data'
CSV_PATH = os.path.join(DATA_DIR, 'amazon_prices.csv')

# Create the data folder if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

def get_price_amazon():
    try:
        page = requests.get(PRODUCT_URL, headers=HEADERS)
        soup = BeautifulSoup(page.content, 'html.parser')
        price_tag = soup.find('span', {'class': 'a-price-whole'})
        if price_tag:
            price = float(price_tag.get_text().replace(',', '').replace('₹', '').strip())
        else:
            price = None
        return datetime.datetime.now().strftime('%Y-%m-%d'), price
    except Exception as e:
        print("Error:", e)
        return datetime.datetime.now().strftime('%Y-%m-%d'), None

def collect_data():
    date, price = get_price_amazon()
    df = pd.DataFrame([[date, price]], columns=['Date', 'Price'])

    # Save or append to CSV
    file_exists = os.path.isfile(CSV_PATH)
    df.to_csv(CSV_PATH, mode='a', header=not file_exists, index=False)

if __name__ == '__main__':
    collect_data()

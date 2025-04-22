import requests
import pandas as pd
import datetime
from sqlalchemy import create_engine

def fetch_coingecko_data(coin_id, days='365'):
    """Fetch historical price data from CoinGecko for a given coin_id."""
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily'
    }
    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch data for {coin_id}: {response.json()}")

    data = response.json()
    prices = data['prices']
    
    # Convert to DataFrame
    df = pd.DataFrame(prices, columns=['timestamp', coin_id])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    return df

def fetch_multiple_coins(coin_ids, days='365'):
    """Fetch and merge data for multiple coins from CoinGecko."""
    merged_df = None
    for coin_id in coin_ids:
        df = fetch_coingecko_data(coin_id, days)
        if merged_df is None:
            merged_df = df
        else:
            merged_df = merged_df.join(df, how='outer')
    return merged_df.dropna()

def store_to_sqlite(df, db_name='crypto_data.db', table_name='prices'):
    """Store DataFrame to SQLite database."""
    engine = create_engine(f'sqlite:///{db_name}', echo=False)
    df.to_sql(table_name, con=engine, if_exists='replace')
    print(f"✅ Stored data in {db_name}, table '{table_name}'")

if __name__ == "__main__":
    # Example usage
    coin_ids = ['bitcoin', 'ethereum', 'solana']
    df = fetch_multiple_coins(coin_ids, days='365')
    print(df.head())
    df.to_csv('crypto_prices.csv')
    # Save to SQLite
    store_to_sqlite(df)
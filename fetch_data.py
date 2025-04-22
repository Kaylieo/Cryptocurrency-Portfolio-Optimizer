import requests
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import streamlit as st
import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class CryptoDataFetcher:
    def __init__(self, coin_ids, days='365', db_name='crypto_data.db', table_name='prices'):
        self.coin_ids = coin_ids
        self.days = days
        self.db_name = db_name
        self.table_name = table_name
        self.engine = create_engine(f'sqlite:///{db_name}', echo=False)

    def fetch_from_api(self, coin_id):
        url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
        params = {
            'vs_currency': 'usd',
            'days': self.days,
            'interval': 'daily'
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch data for {coin_id}: {response.json()}")
        data = response.json()
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', coin_id])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def fetch_all(self):
        merged_df = None
        for coin_id in self.coin_ids:
            df = self.fetch_from_api(coin_id)
            if merged_df is None:
                merged_df = df
            else:
                merged_df = merged_df.join(df, how='outer')
        return merged_df.dropna()

    def store_to_sqlite(self, df):
        df = df.copy()
        df['pulled_at'] = pd.Timestamp.now()
        df.to_sql(self.table_name, con=self.engine, if_exists='replace')
        print(f"✅ Stored data in {self.db_name}, table '{self.table_name}'")

    def load_cached_data(self):
        try:
            df = pd.read_sql(f"SELECT * FROM {self.table_name}", con=self.engine, index_col='timestamp', parse_dates=['timestamp'])
            last_pull = pd.to_datetime(df['pulled_at'].max())
            # Drop the timestamp column and validate cache contents
            df_cached = df.drop(columns='pulled_at')
            if set(df_cached.columns) == set(self.coin_ids) and (pd.Timestamp.now() - last_pull).total_seconds() < 86400:
                print("🕒 Using cached data from SQLite.")
                return df_cached
            else:
                print("🕒 Cached data is stale or does not match requested coins; refetching.")
        except Exception as e:
            print(f"⚠️ No valid cache found: {e}")
        return None

    def get_data(self):
        df = self.load_cached_data()
        if df is None:
            df = self.fetch_all()
            self.store_to_sqlite(df)
        return df


if __name__ == "__main__":
    fetcher = CryptoDataFetcher(['bitcoin', 'ethereum', 'solana'])
    df = fetcher.get_data()
    print(df.head())
    df.to_csv('crypto_prices.csv')

def load_price_data(coin_ids, days='365'):
    fetcher = CryptoDataFetcher(coin_ids, days)
    df = fetcher.get_data()
    return df

def calculate_returns(price_df):
    return np.log(price_df / price_df.shift(1)).dropna()

def portfolio_performance(weights, mean_returns, cov_matrix):
    expected_return = np.dot(weights, mean_returns)  # already annualized
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # already annualized
    return expected_return, volatility

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # Already annualized

def optimize_min_volatility(cov_matrix):
    num_assets = len(cov_matrix)
    init_guess = [1. / num_assets] * num_assets
    bounds = tuple((0.05, 1) for _ in range(num_assets))  # Min 5% per asset
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    result = minimize(portfolio_volatility,
                      init_guess,
                      args=(cov_matrix,),
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)
    return result.x

def optimize_max_sharpe(mean_returns, cov_matrix, risk_free_rate=0.01):
    num_assets = len(mean_returns)
    init_guess = [1. / num_assets] * num_assets
    bounds = tuple((0.05, 1) for _ in range(num_assets))
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
        ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
        return -(ret - risk_free_rate) / vol

    result = minimize(
        neg_sharpe_ratio,
        init_guess,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    print(f"🔎 Max Sharpe Weights: {dict(zip(mean_returns.index, result.x))}")
    return result.x


def plot_pie(weights, assets):
    weights = np.array(weights)
    assets = np.array(assets)
    nonzero_indices = weights > 0.001  # Filter out very small weights
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(weights[nonzero_indices], labels=assets[nonzero_indices], autopct='%1.1f%%')
    ax.set_title("Optimized Portfolio Allocation")
    return fig

def plot_efficient_frontier(mean_returns, cov_matrix, n_portfolios=5000):
    results = np.zeros((3, n_portfolios))

    for i in range(n_portfolios):
        weights = np.random.dirichlet(np.ones(len(mean_returns)))
        ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
        sharpe = (ret - 0.01) / vol

        results[0, i] = ret
        results[1, i] = vol
        results[2, i] = sharpe

    return results
st.set_page_config(page_title="Crypto Portfolio Optimizer", layout="centered")
st.title("💸 Cryptocurrency Portfolio Optimizer")

# Sidebar Inputs
st.sidebar.header("Portfolio Settings")
coin_options = ["bitcoin", "ethereum", "solana", "cardano", "binancecoin"]
selected_coins = st.sidebar.multiselect("Select Cryptocurrencies:", coin_options, default=coin_options[:3])
days = st.sidebar.selectbox("Timeframe (days):", [90, 180, 365, 730], index=2)

# Optimization Mode
opt_mode = st.sidebar.radio("Optimization Mode", ["Minimum Volatility", "Maximum Sharpe Ratio"])

if selected_coins:
    st.info(f"Fetching data for: {', '.join(selected_coins)}")
    fetcher = CryptoDataFetcher(selected_coins, days=str(days))
    df_prices = fetcher.get_data()
    df_returns = calculate_returns(df_prices)

    mean_returns = df_returns.mean() * 252
    cov_matrix = df_returns.cov() * 252

    st.success("Data fetched and cleaned successfully.")

    if opt_mode == "Minimum Volatility":
        weights = optimize_min_volatility(cov_matrix)
        st.subheader("🚀 Optimizing Portfolio (Minimum Volatility)")
    else:
        weights = optimize_max_sharpe(mean_returns, cov_matrix, risk_free_rate=0.01)
        st.subheader("🚀 Optimizing Portfolio (Maximum Sharpe Ratio)")

    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    sharpe = (ret - 0.01) / vol

    st.markdown(f"### {opt_mode} Portfolio")
    for coin, weight in zip(selected_coins, weights):
        st.write(f"**{coin.capitalize()}**: {weight:.2%}")

    st.write(f"**Expected Return**: `{ret:.2%}`")
    st.write(f"**Volatility**: `{vol:.2%}`")
    st.write(f"**Sharpe Ratio**: `{sharpe:.2f}`")

    # Pie Chart
    fig = plot_pie(weights, [c.capitalize() for c in selected_coins])
    st.pyplot(fig)

    # Efficient Frontier
    st.subheader("📈 Efficient Frontier")
    results = plot_efficient_frontier(mean_returns, cov_matrix)
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(results[1], results[0], c=results[2], cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Sharpe Ratio')
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Expected Return')
    ax.set_title('Efficient Frontier')
    st.pyplot(fig)

else:
    st.warning("Please select at least one cryptocurrency to begin.")

import pandas as pd
import numpy as np
import requests
import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# -----------------------------------
# 1. Fetch Historical Crypto Prices
# -----------------------------------
def fetch_coingecko_data(coin_id, days='365'):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily'
    }

    for attempt in range(3):
        response = requests.get(url, params=params)
        if response.status_code == 200:
            prices = response.json()['prices']
            df = pd.DataFrame(prices, columns=['timestamp', coin_id])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        elif response.status_code == 429:
            print(f"Rate limit hit for {coin_id}, retrying ({attempt + 1}/3)...")
            time.sleep(3 + attempt * 2)
        else:
            raise Exception(f"Failed to fetch data for {coin_id}: {response.text}")
    raise Exception(f"Failed to fetch data for {coin_id} after 3 attempts.")

def fetch_multiple_coins(coin_ids, days='365'):
    merged_df = None
    for coin_id in coin_ids:
        df = fetch_coingecko_data(coin_id, days)
        if merged_df is None:
            merged_df = df
        else:
            merged_df = merged_df.join(df, how='outer')
        time.sleep(1.5)  # Prevent rate limiting
    return merged_df.dropna()

# -----------------------------------
# 2. Optimization: Minimize Volatility and Maximize Sharpe Ratio
# -----------------------------------
def calculate_returns(price_df):
    return np.log(price_df / price_df.shift(1)).dropna()

def portfolio_performance(weights, mean_returns, cov_matrix):
    expected_return = np.dot(weights, mean_returns) * 252  # Annualized return
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))  # Annualized volatility
    return expected_return, volatility

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))  # Annualized volatility

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

    result = minimize(neg_sharpe_ratio, init_guess,
                      args=(mean_returns, cov_matrix, risk_free_rate),
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)
    return result.x

# -----------------------------------
# 3. Visualization (Optional in Streamlit)
# -----------------------------------
def plot_pie(weights, assets):
    plt.figure(figsize=(6, 6))
    plt.pie(weights, labels=assets, autopct='%1.1f%%')
    plt.title("Optimized Portfolio Allocation")
    plt.show()

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

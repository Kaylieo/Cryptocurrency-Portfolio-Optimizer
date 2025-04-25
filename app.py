"""Cryptocurrency Portfolio Optimizer App

This Streamlit app fetches historical price data for selected cryptocurrencies,
computes daily returns, and optimizes portfolios for minimum volatility or
maximum Sharpe ratio. Displays allocation, performance metrics, and
efficient frontier plots.
"""

import logging
import time
from sqlalchemy import create_engine
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf
import streamlit as st
from typing import List, Tuple

# === Constants ===
TRADING_DAYS = 252
DEFAULT_RISK_FREE_RATE = 0.01

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CryptoDataFetcher:
    #  Map CoinGecko IDs to Yahoo Finance symbols for fallback
    YF_TICKER_MAP = {
        'bitcoin':      'BTC-USD',
        'ethereum':     'ETH-USD',
        'solana':       'SOL-USD',
        'cardano':      'ADA-USD',
        'binancecoin':  'BNB-USD'
    }

    def __init__(self, coin_ids: List[str], days: str = '365', db_name: str = 'crypto_data.db', table_name: str = 'prices'):
        self.coin_ids = coin_ids
        self.days = days
        self.db_name = db_name
        self.table_name = table_name
        self.engine = create_engine(f'sqlite:///{db_name}', echo=False)


    @staticmethod
    def fetch_from_coingecko(coin_id: str, days: str = '365') -> pd.DataFrame:
        """Fetch prices for a single coin from CoinGecko with retry/backoff on rate limit.
        Returns a DataFrame with 'timestamp' as index and coin_id as column."""
        url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily'
        }
        max_retries = 5
        backoff = 1
        for attempt in range(1, max_retries + 1):
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                prices = data['prices']
                df = pd.DataFrame(prices, columns=['timestamp', coin_id])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
            elif response.status_code == 429:
                print(f"⚠️ Rate limit hit for {coin_id}, retry {attempt}/{max_retries} after {backoff}s...")  #  Rate limit warning
                time.sleep(backoff)
                backoff *= 2
            else:
                print(f"❌ Failed to fetch data for {coin_id} from CoinGecko: {response.text}")  #  API failure
                break
        print(f"❌ Exceeded max retries for {coin_id} from CoinGecko.")  #  Max retries exceeded
        return None


    def fetch_all(self) -> pd.DataFrame:
        #  1. Try Yahoo Finance for all coins that have a mapping
        symbols = [self.YF_TICKER_MAP[c] for c in self.coin_ids if c in self.YF_TICKER_MAP]
        data = None
        if symbols:
            try:
                yf_data = yf.download(
                    symbols,
                    period=f"{self.days}d",
                    interval="1d",
                    progress=False
                )
                if not yf_data.empty and 'Close' in yf_data:
                    close = yf_data['Close']
                    close.columns = [coin for coin in self.coin_ids if coin in self.YF_TICKER_MAP]
                    close.index.name = 'timestamp'
                    data = close
                    logger.info("✅ Yahoo Finance data fetched.")
            except Exception as e:
                logger.warning(f"⚠️ Yahoo Finance fetch error: {e}")
        #  2. For any missing coins or missing data, try CoinGecko
        missing_coins = [coin for coin in self.coin_ids if data is None or coin not in data.columns]
        if missing_coins:
            logger.info(f"🔄 Fetching missing coins from CoinGecko: {missing_coins}")
            cg_frames = []
            for coin in missing_coins:
                cg_df = self.fetch_from_coingecko(coin, self.days)
                if cg_df is not None:
                    cg_frames.append(cg_df)
            if cg_frames:
                cg_data = pd.concat(cg_frames, axis=1)
                if data is not None:
                    data = pd.concat([data, cg_data], axis=1)
                else:
                    data = cg_data
        if data is None or data.empty:
            raise Exception("No data could be fetched from Yahoo Finance or CoinGecko.")
        #  3. Align time index, drop rows with missing values
        data = data.sort_index()
        data = data.loc[~data.index.duplicated(keep='first')]
        #  Interpolate small gaps (max 2 days), then drop remaining NaNs
        data = data.interpolate(method='time', limit=2)
        n_missing = data.isna().sum().sum()
        if n_missing > 0:
            logger.warning(f"⚠️ {n_missing} missing values after interpolation, dropping rows with NaNs.")
        data = data.dropna()
        #  4. Remove outliers (returns > 4 std from mean)
        returns = np.log(data / data.shift(1))
        outlier_mask = (np.abs(returns - returns.mean()) > 4 * returns.std())
        if outlier_mask.any().any():
            logger.warning("⚠️ Outliers detected and set to NaN.")
            data[outlier_mask] = np.nan
            data = data.interpolate(method='time', limit=1).dropna()
        logger.info(f"✅ Final data shape: {data.shape}")
        return data


    def store_to_sqlite(self, df: pd.DataFrame) -> None:
        df = df.copy()
        df['pulled_at'] = pd.Timestamp.now()
        try:
            df.to_sql(self.table_name, con=self.engine, if_exists='replace', index=True, method='multi')
            logger.info(f"✅ Stored data in {self.db_name}, table '{self.table_name}'")
        except Exception as e:
            logger.error(f"❌ Failed to store data in SQLite: {e}")


    def load_cached_data(self) -> pd.DataFrame:
        try:
            df = pd.read_sql(f"SELECT * FROM {self.table_name}", con=self.engine, index_col='timestamp', parse_dates=['timestamp'])
            last_pull = pd.to_datetime(df['pulled_at'].max())
            #  Drop the pulled_at column and validate cache contents
            df_cached = df.drop(columns='pulled_at')
            if set(df_cached.columns) == set(self.coin_ids) and (pd.Timestamp.now() - last_pull).total_seconds() < 86400:
                logger.info("🕒 Using cached data from SQLite.")
                return df_cached
            else:
                logger.info("🕒 Cached data is stale or does not match requested coins; refetching.")
        except Exception as e:
            logger.warning(f"⚠️ No valid cache found: {e}")
        return None


    def get_data(self) -> pd.DataFrame:
        df = self.load_cached_data()
        if df is None:
            df = self.fetch_all()
            self.store_to_sqlite(df)
        #  Save fetched DataFrame for downstream calculations
        self.data = df.copy()
        logger.info(f"✅ Data loaded for coins: {list(df.columns)} | {len(df)} rows.")
        return df


    def calculate_returns(self) -> pd.DataFrame:
        """Calculate and store daily log returns of the fetched price data.
        Uses the actual number of trading days for annualization.
        Returns:
            pd.DataFrame: Daily log returns."""
        if not hasattr(self, 'data'):
            raise AttributeError("Data not loaded. Call get_data() first.")
        returns = np.log(self.data / self.data.shift(1))
        returns = returns.dropna()
        self.returns = returns
        logger.info(f"✅ Calculated returns: {returns.shape[0]} days")
        return returns


    def get_annualized_mean_returns(self) -> pd.Series:
        """Calculate and return annualized mean returns based on daily log returns.
        Returns:
            pd.Series: Annualized mean returns."""
        if not hasattr(self, 'returns'):
            self.calculate_returns()
        return self.returns.mean() * TRADING_DAYS


    def get_annualized_covariance(self) -> pd.DataFrame:
        """Calculate and return annualized covariance matrix based on daily log returns.
        Returns:
            pd.DataFrame: Annualized covariance matrix."""
        if not hasattr(self, 'returns'):
            self.calculate_returns()
        return self.returns.cov() * TRADING_DAYS


@st.cache_data
def load_price_data(coin_ids: List[str], days: str = '365') -> pd.DataFrame:
    """Load price data for given coins and timeframe."""
    fetcher = CryptoDataFetcher(coin_ids, days)
    df = fetcher.get_data()
    return df


def calculate_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    return np.log(price_df / price_df.shift(1)).dropna()


def portfolio_performance(weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> Tuple[float, float]:
    expected_return = np.dot(weights, mean_returns)    #  Already annualized
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))    #  Already annualized
    return expected_return, volatility


def portfolio_volatility(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))    #  Already annualized


def optimize_min_volatility(cov_matrix: pd.DataFrame, max_weight: float = 1.0) -> np.ndarray:
    """Optimize for minimum portfolio volatility given a covariance matrix and weight cap."""
    num_assets = len(cov_matrix)
    init_guess = np.array([1.0 / num_assets] * num_assets)
    #  Allow zero weight so assets can be excluded entirely
    bounds = tuple((0, max_weight) for _ in range(num_assets))
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    result = minimize(portfolio_volatility,
                      init_guess,
                      args=(cov_matrix,),
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)
    return result.x


def optimize_max_sharpe(mean_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float = DEFAULT_RISK_FREE_RATE, max_weight: float = 1.0) -> np.ndarray:
    """Optimize for maximum Sharpe ratio given expected returns, covariance, and risk-free rate."""
    num_assets = len(mean_returns)
    #  Start from equal weights
    init_guess = np.array([1.0 / num_assets] * num_assets)
    #  Allow zero weight so assets can be excluded for max Sharpe
    bounds = tuple((0, max_weight) for _ in range(num_assets))
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
        ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
        return -(ret - risk_free_rate) / vol  # Use provided rate or fallback to constant

    result = minimize(
        neg_sharpe_ratio,
        init_guess,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    logger.info(f"🔎 Max Sharpe Weights: {dict(zip(mean_returns.index, result.x))}")
    return result.x


def plot_pie(weights: np.ndarray, assets: List[str]) -> plt.Figure:
    weights = np.array(weights)
    assets = np.array(assets)
    nonzero_indices = weights > 0.001    #  Filter out very small weights
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(weights[nonzero_indices])))
    ax.pie(weights[nonzero_indices], labels=assets[nonzero_indices], autopct='%1.1f%%', colors=colors)
    return fig


def plot_efficient_frontier(mean_returns: pd.Series, cov_matrix: pd.DataFrame, n_portfolios: int = 5000, optimal_points: List[Tuple[float, float]] = []) -> np.ndarray:
    results = np.zeros((3, n_portfolios))
    for i in range(n_portfolios):
        weights = np.random.dirichlet(np.ones(len(mean_returns)))
        ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
        sharpe = (ret - DEFAULT_RISK_FREE_RATE) / vol
        results[0, i] = ret
        results[1, i] = vol
        results[2, i] = sharpe
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(results[1], results[0], c=results[2], cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Sharpe Ratio')
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Expected Return')
    for i, point in enumerate(optimal_points):
        if i == 0:
            ax.scatter(point[1], point[0], marker='^', color='#43AA8B', s=150, label='Min Volatility Portfolio')
        else:
            ax.scatter(point[1], point[0], marker='o', color='#F94144', s=150, label='Max Sharpe Portfolio')
    ax.legend()
    return results


def apply_theme() -> None:
    """Inject custom CSS theme into Streamlit."""
    css = """
    <style>
        /* Background and font */
        body, .main { background-color: #0B132B; color: #E0E1DD; font-family: 'Segoe UI', sans-serif; }
        /* Header colors */
        h1, h2, h3, h4, h5, h6 { text-align: center; color: #E0E1DD; }
        /* Sidebar tweaks */
        section[data-testid="stSidebar"] { background-color: #1C2541; color: #E0E1DD; }
        section[data-testid="stSidebar"] *:not(input):not(select):not(option) { color: #E0E1DD !important; }
        /* Info and success messages */
        div[role="alert"][class*="stAlert-success"] { background-color: #1D3557 !important; border-left: 5px solid #43AA8B; color: #E0E1DD; }
        div[role="alert"][class*="stAlert-info"] { background-color: #274C77 !important; border-left: 5px solid #A9DEF9; color: #E0E1DD; }
        /* Buttons */
        .stButton>button { background-color: #3E92CC; color: white; }
        /* Dropdown and multiselect dark theme fixes */
        div[data-baseweb="select"] input, div[data-baseweb="select"] div, div[data-baseweb="select"] * { color: #E0E1DD !important; }
        div[data-baseweb="select"] div[role="option"] { color: #E0E1DD !important; }
        div[data-baseweb="tag"] span { color: #0B132B !important; background-color: #E0E1DD !important; border-radius: 5px; padding: 2px 6px; }
        div[data-baseweb="tag"] div { color: #0B132B !important; background-color: #E0E1DD !important; font-weight: bold; border-radius: 5px; padding: 2px 6px; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_sidebar() -> Tuple[List[str], int, float, float, str]:
    """Render Streamlit sidebar controls and return user selections."""
    st.sidebar.header("Portfolio Settings")
    coin_options = list(CryptoDataFetcher.YF_TICKER_MAP.keys())
    selected_coins = st.sidebar.multiselect("Select Cryptocurrencies:", coin_options, default=coin_options[:3])
    days = st.sidebar.selectbox("Timeframe (days):", [90, 180, 365, 730], index=2)
    max_weight = st.sidebar.slider("Max Weight Per Asset", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
    risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.1) / 100
    opt_mode = st.sidebar.radio("Optimization Mode", ["Minimum Volatility", "Maximum Sharpe Ratio"])
    return selected_coins, days, max_weight, risk_free_rate, opt_mode


def render_results(selected_coins: List[str], days: int, max_weight: float, risk_free_rate: float, opt_mode: str) -> None:
    """Render portfolio results: fetch data, optimize and display charts."""
    price_df = load_price_data(selected_coins, days)
    returns = calculate_returns(price_df)
    mean_returns = returns.mean() * TRADING_DAYS
    cov_matrix = returns.cov() * TRADING_DAYS
    if opt_mode == "Minimum Volatility":
        weights = optimize_min_volatility(cov_matrix, max_weight=max_weight)
    else:
        weights = optimize_max_sharpe(mean_returns, cov_matrix, risk_free_rate, max_weight)
    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    # Calculate Sharpe ratio
    sharpe = (ret - risk_free_rate) / vol
    weights = np.round(weights, 4)
    # Display allocations, metrics, and charts (unchanged)
    st.markdown(
        f"<div style='text-align: center; color: #31708f; background-color: #d9edf7; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>Fetching data for: {', '.join(selected_coins)}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div style='text-align: center; color: #3c763d; background-color: #dff0d8; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>Data fetched and cleaned successfully.</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"""
        <div style='display:flex;justify-content:center;'>
            <div style='background-color:#f0f2f6;padding:15px;border-radius:10px;margin-top:20px;margin-bottom:10px;width:fit-content;'>
                <b>Optimization Mode:</b> {opt_mode}<br>
                <b>Risk-Free Rate:</b> {risk_free_rate * 100:.1f}%<br>
                <b>Max Weight Per Asset:</b> {max_weight:.2f}<br>
                <b>Timeframe:</b> {days} days
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    #  Coin logo map (png links, no redirects)
    COIN_LOGO_MAP = {
        "bitcoin": "https://assets.coingecko.com/coins/images/1/thumb/bitcoin.png",
        "ethereum": "https://assets.coingecko.com/coins/images/279/thumb/ethereum.png",
        "solana": "https://assets.coingecko.com/coins/images/4128/thumb/solana.png",
        "cardano": "https://assets.coingecko.com/coins/images/975/thumb/cardano.png",
        "binancecoin": "https://assets.coingecko.com/coins/images/825/thumb/binance-coin-logo.png"
    }
    st.markdown("<div style='display:flex;flex-direction:column;align-items:center;'>", unsafe_allow_html=True)
    #  Show coin allocations with logos
    st.markdown("<div style='display:flex;flex-direction:column;align-items:center;width:100%;'>", unsafe_allow_html=True)
    for coin, weight in zip(selected_coins, weights):
        icon_url = COIN_LOGO_MAP.get(coin, f"https://cryptoicons.org/api/icon/{coin[:3]}/32")
        st.markdown(
            f"<div style='display:flex;align-items:center;justify-content:center;margin:2px;'><img src='{icon_url}' style='height:20px;margin-right:10px;'> <b>{coin.capitalize()}</b>: {weight:.2%}</div>",
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)
    #  Show metrics in a horizontal row
    st.markdown("""
    <div style='display: flex; justify-content: center; gap: 40px; margin-top: 20px;'>
        <div style='text-align: center;'>
            <div style='font-weight:bold;'>Expected Return</div>
            <div style='font-size: 24px;'>{:.2f}%</div>
        </div>
        <div style='text-align: center;'>
            <div style='font-weight:bold;'>Volatility</div>
            <div style='font-size: 24px;'>{:.2f}%</div>
        </div>
        <div style='text-align: center;'>
            <div style='font-weight:bold;'>Sharpe Ratio</div>
            <div style='font-size: 24px;'>{:.2f}</div>
        </div>
    </div>
    """.format(ret * 100, vol * 100, sharpe), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    #  Pie Chart
    st.markdown("<h3 style='text-align:center;'>Optimized Portfolio Allocation</h3>", unsafe_allow_html=True)
    fig = plot_pie(weights, [c.capitalize() for c in selected_coins])
    st.pyplot(fig)
    #  Efficient Frontier
    st.markdown("<h3 style='text-align: center;'>Efficient Frontier</h3>", unsafe_allow_html=True)
    minvol_weights = optimize_min_volatility(cov_matrix, max_weight=max_weight)
    minvol_ret, minvol_vol = portfolio_performance(minvol_weights, mean_returns, cov_matrix)
    maxsharpe_weights = optimize_max_sharpe(mean_returns, cov_matrix, risk_free_rate=risk_free_rate, max_weight=max_weight)
    maxsharpe_ret, maxsharpe_vol = portfolio_performance(maxsharpe_weights, mean_returns, cov_matrix)
    results = plot_efficient_frontier(
        mean_returns, cov_matrix,
        optimal_points=[(minvol_ret, minvol_vol), (maxsharpe_ret, maxsharpe_vol)]
    )
    st.pyplot(plt.gcf())


def main() -> None:
    """Main entrypoint: configure page, apply theme and render UI."""
    st.set_page_config(page_title="Crypto Portfolio Optimizer", layout="centered", page_icon="💹", initial_sidebar_state="expanded")
    apply_theme()
    st.title("Cryptocurrency Portfolio Optimizer")
    selected_coins, days, max_weight, risk_free_rate, opt_mode = render_sidebar()
    if selected_coins:
        render_results(selected_coins, days, max_weight, risk_free_rate, opt_mode)
    else:
        st.warning("Please select at least one cryptocurrency to begin.")


if __name__ == "__main__":
    main()

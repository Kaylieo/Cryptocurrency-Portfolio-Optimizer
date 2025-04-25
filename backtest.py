
# === Imports ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from app import calculate_returns, optimize_max_sharpe, optimize_min_volatility, TRADING_DAYS, DEFAULT_RISK_FREE_RATE


# === Portfolio Backtesting ===
def backtest_portfolio(
    price_df: pd.DataFrame,
    optimization_date: str,
    method: str = 'sharpe',
    holding_days: int = 60,
    max_weight: float = 1.0,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE
) -> dict:
    """
    Backtests portfolio performance after an optimization date.

    Args:
        price_df (pd.DataFrame): Historical price data with dates as index.
        optimization_date (str): The date to optimize the portfolio (format 'YYYY-MM-DD').
        method (str): Optimization method: 'sharpe' or 'volatility'.
        holding_days (int): Number of days to hold the portfolio after optimization.
        max_weight (float): Maximum weight per asset.
        risk_free_rate (float): Risk-free rate for Sharpe calculation.

    Returns:
        dict: Performance metrics and final portfolio value.
    """
    returns = calculate_returns(price_df)
    returns_before = returns.loc[:optimization_date]
    returns_after = returns.loc[optimization_date:]
    returns_after = returns_after.head(holding_days)
    print("Price data range:", price_df.index.min(), "to", price_df.index.max())
    print("Returns before optimization date:", returns_before.shape)
    print(returns_before.head())

    mean_returns = returns_before.mean() * TRADING_DAYS
    cov_matrix = returns_before.cov() * TRADING_DAYS
    if method == 'sharpe':
        weights = optimize_max_sharpe(mean_returns, cov_matrix, risk_free_rate, max_weight)
    else:
        weights = optimize_min_volatility(cov_matrix, max_weight)
    portfolio_returns = (returns_after * weights).sum(axis=1)
    cumulative_return = (portfolio_returns + 1).prod() - 1
    volatility = portfolio_returns.std() * np.sqrt(TRADING_DAYS)
    sharpe_ratio = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(TRADING_DAYS)
    initial_value = 10000
    portfolio_value = initial_value * (portfolio_returns + 1).cumprod()
    print("Mean returns:", mean_returns)
    print("Covariance matrix:\n", cov_matrix)
    plt.figure(figsize=(10, 5))
    plt.plot(portfolio_value, label='Portfolio Value')
    plt.title(f'Portfolio Backtest ({method.capitalize()}) from {optimization_date}')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return {
        'Optimization Method': method,
        'Optimization Date': optimization_date,
        'Holding Days': holding_days,
        'Cumulative Return (%)': round(cumulative_return * 100, 2),
        'Annualized Volatility (%)': round(volatility * 100, 2),
        'Sharpe Ratio': round(sharpe_ratio, 3),
        'Final Portfolio Value ($)': round(portfolio_value.iloc[-1], 2)
    }
if __name__ == "__main__":
    # Example usage for quick testing

    # Load some sample price data
    df = pd.read_csv("crypto_prices.csv", index_col=0, parse_dates=True)

    # Backtest portfolio
    result = backtest_portfolio(
        price_df=df,
        optimization_date='2024-05-20',
        method='sharpe',
        holding_days=60
    )

    # Print backtest results
    print(result)

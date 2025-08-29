# === Imports ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Optional, Tuple
from sklearn.covariance import LedoitWolf

from app import calculate_returns, optimize_max_sharpe, optimize_min_volatility, TRADING_DAYS, DEFAULT_RISK_FREE_RATE


# === Helpers ===

def _equal_weight_cum_return(returns_after: pd.DataFrame) -> float:
    """Cumulative arithmetic return of an equal‑weight portfolio over the OOS window.
    Assumes `returns_after` contains (log) daily returns per asset. We convert to
    arithmetic approximation via exp(sum) - 1 on the equal‑weighted log series.
    """
    if returns_after.empty:
        return float('nan')
    # equal-weight portfolio daily log return (approx as mean of log returns)
    ew_log = returns_after.mean(axis=1)
    cum_factor = float(np.exp(np.sum(ew_log)))
    return cum_factor - 1.0


# === Portfolio Backtesting ===
def backtest_portfolio(
    price_df: pd.DataFrame,
    optimization_date: str,
    method: str = 'sharpe',
    holding_days: int = 60,
    max_weight: float = 1.0,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE
) -> Dict[str, float | int | str]:
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
    # Normalize date input for type checkers and pandas
    opt_dt = pd.to_datetime(optimization_date)

    returns = calculate_returns(price_df)
    # Strictly PRE/POST split to avoid look-ahead bias
    returns_before: pd.DataFrame = returns.loc[: opt_dt - pd.Timedelta(days=1)]
    returns_after: pd.DataFrame = returns.loc[opt_dt + pd.Timedelta(days=1):]
    returns_after = returns_after.head(holding_days)

    if returns_before.empty:
      raise ValueError("No in-sample data before the optimization date. Choose a later date or increase the data window.")
    if returns_after.empty:
      raise ValueError("No out-of-sample data after the optimization date. Choose an earlier date or provide more recent data.")

    # Annualized mean (log) and covariance with shrinkage for stability
    mean_returns: pd.Series = returns_before.mean(numeric_only=True) * TRADING_DAYS
    lw = LedoitWolf().fit(returns_before.values)
    cov_matrix: pd.DataFrame = pd.DataFrame(
        lw.covariance_, index=returns_before.columns, columns=returns_before.columns
    ) * TRADING_DAYS
    if method == 'sharpe':
        weights = optimize_max_sharpe(mean_returns, cov_matrix, risk_free_rate, max_weight)
    else:
        weights = optimize_min_volatility(cov_matrix, max_weight)
    # Ensure numpy float dtype for downstream math and static type checkers
    weights = np.asarray(weights, dtype=float)

    # Expected (model) metrics from in-sample estimates, reported in arithmetic terms
    port_mu_log: float = float(np.dot(weights, mean_returns))
    port_var_log: float = float(np.dot(weights, np.dot(cov_matrix, weights)))
    expected_return_annual: float = float(np.exp(port_mu_log + 0.5 * port_var_log) - 1.0)
    expected_vol_annual: float = float(np.sqrt(port_var_log))
    expected_sharpe: float = float((expected_return_annual - risk_free_rate) / expected_vol_annual) if expected_vol_annual > 0 else float("nan")

    # Elementwise multiply (aligns by columns) then sum across assets to get daily portfolio return
    portfolio_returns: pd.Series = (returns_after.astype(float).multiply(weights, axis=1)).sum(axis=1)

    # Use concrete numpy float array for numeric ops
    pr_arr: np.ndarray = portfolio_returns.to_numpy(dtype=float)

    # Cumulative growth factor and return (treat pr_arr as **log** daily returns)
    cum_factor = float(np.exp(np.sum(pr_arr)))
    cumulative_return: float = cum_factor - 1.0

    # Daily hit rate vs equal‑weight benchmark (compare log returns day‑by‑day)
    ew_log_daily = returns_after.mean(axis=1).to_numpy(dtype=float)
    daily_outperf = (pr_arr > ew_log_daily)
    daily_hit_rate = float(np.round(100.0 * np.mean(daily_outperf.astype(float)), 2))

    # Annualized volatility
    volatility: float = float(np.std(pr_arr) * np.sqrt(TRADING_DAYS))

    # Annualized Sharpe (guard against zero std)
    pr_std = float(np.std(pr_arr))
    pr_mean = float(np.mean(pr_arr))
    sharpe_ratio: float = float((pr_mean / pr_std) * np.sqrt(TRADING_DAYS)) if pr_std > 0 else float("nan")

    initial_value = 10000
    pv_values: np.ndarray = initial_value * np.cumprod(pr_arr + 1.0)
    portfolio_value: pd.Series = pd.Series(pv_values, index=returns_after.index)
    print("Mean returns:", mean_returns)
    print("Covariance matrix:\n", cov_matrix)
    plt.figure(figsize=(10, 5))
    x = portfolio_value.index.to_numpy(dtype='datetime64[ns]')
    y = portfolio_value.to_numpy(dtype=float)
    plt.plot(x, y, label='Portfolio Value')
    plt.title(f'Portfolio Backtest (OOS) — {method.capitalize()} from {optimization_date}')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Calibration: daily coverage within 1σ under normal assumption ---
    # Convert in-sample annual log moments to daily
    mu_daily = port_mu_log / TRADING_DAYS
    sigma_daily = float(np.sqrt(port_var_log / TRADING_DAYS))
    # Use the same (approximate) daily log portfolio returns
    within_1sigma = np.abs(pr_arr - mu_daily) <= sigma_daily
    coverage_1sigma = float(np.round(100.0 * np.mean(within_1sigma.astype(float)), 2))

    # Equal-weight benchmark cumulative OOS return for a simple % outperformance view
    ew_cum_ret = _equal_weight_cum_return(returns_after)
    outperf_vs_ew = float(np.round(100.0 * (cumulative_return - ew_cum_ret), 2))

    return {
        'Optimization Method': str(method),
        'Optimization Date': str(opt_dt.date()),
        'Holding Days (OOS)': int(holding_days),
        # Expected (model, in-sample, arithmetic)
        'Expected Return (Annual, %)': float(np.round(expected_return_annual * 100.0, 2)),
        'Expected Volatility (Annual, %)': float(np.round(expected_vol_annual * 100.0, 2)),
        'Expected Sharpe': float(np.round(expected_sharpe, 3)) if np.isfinite(expected_sharpe) else float('nan'),
        # Realized (out-of-sample)
        'Realized Cum Return (OOS, %)': float(np.round(cumulative_return * 100.0, 2)),
        'Realized Volatility (OOS, Annual, %)': float(np.round(volatility * 100.0, 2)),
        'Coverage within 1σ (Daily, %)': coverage_1sigma,
        'Outperformance vs EW (OOS, %-pts)': outperf_vs_ew,
        'Daily Outperformance vs EW (Hit Rate, %)': daily_hit_rate,
        'Final Portfolio Value ($)': float(np.round(float(portfolio_value.iloc[-1]), 2))
    }

# === Walk‑forward evaluation ===

def walk_forward_outperformance_rate(
    price_df: pd.DataFrame,
    method: str = 'sharpe',
    holding_days: int = 60,
    max_weight: float = 1.0,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    step: int = 20,
) -> Dict[str, float]:
    """Rolling walk‑forward backtest and % windows the strategy beats equal‑weight.

    Returns
    -------
    dict with keys:
      - windows: number of evaluated windows
      - outperformance_rate (%): percentage of windows with higher OOS return
      - avg_oos_cum_return (%): average OOS cumulative return of the strategy
      - avg_ew_cum_return (%): average OOS cumulative return of equal‑weight
    """
    returns = calculate_returns(price_df)
    dates = returns.index
    # need at least 252 days for in‑sample, plus holding window
    min_start = 252
    wins = 0
    total = 0
    strat_rets = []
    bench_rets = []
    for i in range(min_start, len(dates) - holding_days - 1, step):
        opt_date = str(dates[i].date())
        res = backtest_portfolio(
            price_df=price_df,
            optimization_date=opt_date,
            method=method,
            holding_days=holding_days,
            max_weight=max_weight,
            risk_free_rate=risk_free_rate,
        )
        # rebuild the same OOS window to compute EW benchmark
        opt_dt = pd.to_datetime(opt_date)
        oos = returns.loc[opt_dt + pd.Timedelta(days=1):].head(holding_days)
        ew = _equal_weight_cum_return(oos)
        strat = float(res['Realized Cum Return (OOS, %)']) / 100.0
        if not np.isnan(strat) and not np.isnan(ew):
            total += 1
            wins += int(strat > ew)
            strat_rets.append(strat)
            bench_rets.append(ew)
    rate = 0.0 if total == 0 else 100.0 * wins / total
    avg_strat = 0.0 if not strat_rets else 100.0 * float(np.mean(strat_rets))
    avg_ew = 0.0 if not bench_rets else 100.0 * float(np.mean(bench_rets))
    return {
        'windows': float(total),
        'outperformance_rate (%)': float(np.round(rate, 2)),
        'avg_oos_cum_return (%)': float(np.round(avg_strat, 2)),
        'avg_ew_cum_return (%)': float(np.round(avg_ew, 2)),
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

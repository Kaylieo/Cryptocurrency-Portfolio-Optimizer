import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from optimize_portfolio import (
    fetch_multiple_coins,
    calculate_returns,
    portfolio_performance,
    optimize_min_volatility,
    optimize_max_sharpe,
    plot_pie,
    plot_efficient_frontier
)

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
    df_prices = fetch_multiple_coins(selected_coins, days=str(days))
    df_returns = calculate_returns(df_prices)

    mean_returns = df_returns.mean()
    cov_matrix = df_returns.cov()

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

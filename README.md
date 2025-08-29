# 💹 Cryptocurrency Portfolio Optimizer

An advanced Streamlit-based application that helps users build and optimize cryptocurrency portfolios using **Modern Portfolio Theory** (MPT). Beyond a simple UI demo, this tool incorporates **robust data cleaning, outlier handling, Ledoit-Wolf covariance shrinkage, and backtesting** to provide realistic and reliable portfolio optimization. The tool integrates data from CoinGecko and Yahoo Finance, and offers real-time visualizations and caching for fast analysis.

## 🔬 Technical Overview

This project implements a streamlined Python architecture for financial modeling:

- **Modern Portfolio Theory**: Optimize portfolios based on Sharpe Ratio or Volatility
- **Efficient Frontier Simulation**: Generates thousands of portfolios using Dirichlet distributions
- **SQLite Caching for Reproducibility**: Efficient historical price storage to reduce API calls and ensure consistent results
- **Outlier Filtering**: Robust data cleaning to handle anomalies and improve model stability
- **Ledoit-Wolf Covariance Shrinkage**: Improved covariance matrix estimation for better risk assessment
- **Walk-Forward Evaluation (In Progress)**: Ongoing development of out-of-sample testing and backtesting features
- **Themeable UI**: Integrated dark mode with CSS overrides
- **Streamlit Dashboard**: Real-time interactive UI with customizable inputs

## ✨ Key Features

### Portfolio Optimization
- 📈 **Max Sharpe Ratio**: Risk-adjusted performance optimizer
- 📉 **Min Volatility**: Conservative portfolio construction
- 🔒 **Weight Cap**: Limit allocation per coin to avoid unrealistic concentration
- 🧠 **Risk-Free Rate Input**: Customize Sharpe ratio assumptions
- ⚠️ **Weight Concentration Awareness**: Optimization can sometimes concentrate weights due to crypto volatility, but constraints and Ledoit-Wolf shrinkage improve realism

### Data Pipeline
- 🌐 **CoinGecko + Yahoo Finance**: Hybrid API fallback logic
- 💾 **SQLite Caching**: Smart local storage of clean price data for reproducibility
- 📊 **Data Cleaning**: Interpolation, outlier removal, and log returns
- 🚫 **Outlier Filtering**: Robust handling of anomalous price moves

### UI & Visualization
- 🌓 **Dark Mode**: Fully themed interface with custom CSS
- 🧩 **Pie Charts & Logos**: Visual portfolio breakdown with coin icons
- 📉 **Efficient Frontier**: Color-coded scatter plot with optimal markers
- 📇 **Portfolio Card**: Summary of metrics and selected settings

### Backtesting & Evaluation
- 📈 **Out-of-Sample Tests**: Evaluate portfolio performance on unseen data
- 📊 **Cumulative Return Curves**: Visualize portfolio growth over time
- 🧮 **Covariance Matrix Estimation**: Ledoit-Wolf shrinkage for improved risk modeling
- 🔄 **Walk-Forward Evaluation (In Progress)**: Ongoing development of rolling window backtesting to assess stability

## 🔧 Project Structure

```
Cryptocurrency-Portfolio-Optimization/
├── app.py        # Main app file (data fetching + Streamlit UI)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## 🛠️ Installation

### Prerequisites
- Python 3.12+
- pip

### Setup Instructions

1. **Clone the Repository**
```bash
git clone https://github.com/YOUR_USERNAME/Cryptocurrency-Portfolio-Optimization.git
cd Cryptocurrency-Portfolio-Optimization
```

2. **Create a Virtual Environment (Recommended)**

It is recommended to use a virtual environment to manage dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate     # On Windows
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the App**
```bash
streamlit run app.py
```

## 🎯 Usage

1. Select cryptocurrencies from the sidebar
2. Choose optimization mode (Sharpe or Volatility)
3. Adjust weights, risk-free rate, and timeframe
4. Analyze the allocation, metrics, and frontier
5. Save screenshots or export CSV manually

## 📸 Preview

| Optimized Pie | Efficient Frontier | Summary Card |
|---------------|--------------------|---------------|
| ![](screenshots/optimized_pie.png) | ![](screenshots/efficient_frontier.png) | ![](screenshots/summary_card.png) |

## 📚 Dependencies

See `requirements.txt` for the full list of packages used, including:

- `streamlit`
- `pandas`
- `numpy`
- `matplotlib`
- `yfinance`
- `requests`
- `sqlalchemy`
- `scipy`
- `scikit-learn`  <!-- For Ledoit-Wolf covariance shrinkage -->

## 🚀 Future Improvements

- ✅ Export portfolio weights to CSV
- 📡 Real-time data streaming via WebSocket
- 📱 Mobile UI layout adjustments
- 💡 Add user-uploaded portfolios for comparison
- 🔁 Portfolio rebalancing calendar view
- 📊 Benchmarks such as equal-weight and BTC-only portfolios
- 🔄 Walk-forward validation and backtesting enhancements

## 📬 Contact

- 📧 Email: Kaylieoneal@yahoo.com
- 🐙 GitHub: [Kaylieo](https://github.com/Kaylieo)

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

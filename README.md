
# 💹 Cryptocurrency Portfolio Optimizer

An elegant Streamlit-based application that helps users build and optimize cryptocurrency portfolios using **Modern Portfolio Theory** (MPT). The tool integrates data from CoinGecko and Yahoo Finance, and offers real-time visualizations and caching for fast analysis.

## 🔬 Technical Overview

This project implements a streamlined Python architecture for financial modeling:

- **Modern Portfolio Theory**: Optimize portfolios based on Sharpe Ratio or Volatility
- **Efficient Frontier Simulation**: Generates thousands of portfolios using Dirichlet distributions
- **SQLite Caching**: Efficient historical price storage to reduce API calls
- **Themeable UI**: Integrated dark mode with CSS overrides
- **Streamlit Dashboard**: Real-time interactive UI with customizable inputs

## ✨ Key Features

### Portfolio Optimization
- 📈 **Max Sharpe Ratio**: Risk-adjusted performance optimizer
- 📉 **Min Volatility**: Conservative portfolio construction
- 🔒 **Weight Cap**: Limit allocation per coin
- 🧠 **Risk-Free Rate Input**: Customize Sharpe ratio assumptions

### Data Pipeline
- 🌐 **CoinGecko + Yahoo Finance**: Hybrid API fallback logic
- 💾 **SQLite Caching**: Smart local storage of clean price data
- 📊 **Data Cleaning**: Interpolation, outlier removal, and log returns

### UI & Visualization
- 🌓 **Dark Mode**: Fully themed interface with custom CSS
- 🧩 **Pie Charts & Logos**: Visual portfolio breakdown with coin icons
- 📉 **Efficient Frontier**: Color-coded scatter plot with optimal markers
- 📇 **Portfolio Card**: Summary of metrics and selected settings

## 🔧 Project Structure

```
Cryptocurrency-Portfolio-Optimization/
├── app.py        # Main app file (data fetching + Streamlit UI)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## 🛠️ Installation

### Prerequisites
- Python 3.9+
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
python -m venv .venv
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
| ![](screenshots/Screenshot%202025-05-01%20at%209.26.04%20AM.png) | ![](screenshots/Screenshot%202025-05-01%20at%209.26.26%20AM.png) | ![](screenshots/Screenshot%202025-05-01%20at%209.26.51%20AM.png) |

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

## 🚀 Future Improvements

- ✅ Export portfolio weights to CSV
- 📡 Real-time data streaming via WebSocket
- 📱 Mobile UI layout adjustments
- 💡 Add user-uploaded portfolios for comparison
- 🔁 Portfolio rebalancing calendar view

## 📬 Contact

- 📧 Email: Kaylieoneal@yahoo.com
- 🐙 GitHub: [Kaylieo](https://github.com/Kaylieo)

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

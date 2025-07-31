# Crypto Trading AI Agent

**An end-to-end backtesting and simulation framework for cryptocurrency trading powered by machine learning and deep learning techniques.**

---

## 📖 Overview

This repository contains a modular Python pipeline that:

1. **Fetches** historical OHLCV data for one or more cryptocurrencies using `yfinance`.
2. **Engineers** a rich set of technical and statistical features (EMA, MACD, RSI, Bollinger Bands, ATR, OBV, GARCH volatility, stochastic indicators, ADX, CCI, etc.).
3. **Trains** supervised models (XGBoost, Random Forest, GRU recurrent networks) with:

   * **Hyperparameter tuning** via `RandomizedSearchCV` and `TimeSeriesSplit`.
   * **Probability calibration** using Platt scaling (`CalibratedClassifierCV`).
   * **Imbalanced data handling** via SMOTE.
4. **Optimizes** decision thresholds by sweeping (e.g., 50–90%) to maximize backtested profit.
5. **Simulates** bar-by-bar P\&L with realistic:

   * **Position sizing** (fixed risk % of capital).
   * **Profit targets & stop-losses**.
   * **Slippage** and **commission** costs.
   * **Multi-symbol portfolio** support.
6. **Reports** comprehensive performance metrics:

   * Total return %, Sharpe ratio, max drawdown, win rate, trade count.
   * Equity curve plots and detailed trade logs.
7. **(Future)** supports walk-forward validation, live/paper trading integration, and dashboard automation.

---

## ⚙️ Features

* **Data Fetching:** `fetch_data(symbol, start, end, interval)`
* **Feature Engineering:**

  * EMA, MACD, RSI, Bollinger Bands, ATR, OBV
  * GARCH(1,1) volatility, Stochastic %K/%D, ADX, CCI
  * Interaction terms (e.g., RSI × MACD)
* **Modeling:**

  * XGBoost and RandomForest with `TimeSeriesSplit` CV
  * GRU-based RNN for sequence modeling
  * Calibration with `CalibratedClassifierCV`
  * SMOTE resampling for class imbalance
* **Threshold Optimization:** find optimal confidence cutoff via backtest sweep
* **Simulation Engine:** realistic trade simulation with slippage & fees
* **Portfolio Backtest:** loop over multiple symbols, aggregate results

---

## 🛠️ Installation

1. **Clone** this repository:

   ```bash
   git clone https://github.com/your-username/crypto-trading-ai-agent.git
   cd crypto-trading-ai-agent
   ```

2. **Install** dependencies (recommended in a virtual environment):

   ```bash
   pip install yfinance scikit-learn matplotlib pandas arch xgboost imblearn joblib tensorflow
   ```

---

## 🚀 Usage

1. **Configure** parameters at the top of the main script (e.g., `config.py` or directly in `main.py`):

   ```python
   SYMBOLS = ['BTC-USD', 'ETH-USD', 'SOL-USD']
   START_DATE = '2024-01-01'
   END_DATE   = '2025-07-31'
   INTERVAL   = '1h'
   PROFIT_TARGET_PCT = 2.0
   STOP_LOSS_PCT     = 1.5
   INITIAL_CAPITAL   = 10000.0
   RISK_PCT          = 0.02
   SLIPPAGE_PCT      = 0.0005
   COMMISSION_RATE   = 0.001
   ```

2. **Run** the simulation pipeline:

   ```bash
   python run_portfolio.py
   ```

3. **View** outputs:

   * Printed metrics for each symbol (return, Sharpe, drawdown).
   * Equity curve plots displayed via Matplotlib.
   * Trade log CSV files in `logs/` (if enabled).

---

## 🗂️ Project Structure

```
crypto-trading-ai-agent/
├── run_portfolio.py        # Main entry point: portfolio backtest
├── simulate_symbol.py      # Single-symbol simulation functions
├── data_utils.py           # fetch_data & feature engineering
├── model_utils.py          # tuning & calibration helpers
├── backtest_utils.py       # threshold sweep & P&L simulation
├── config.py               # User-configurable parameters
├── requirements.txt        # Pin dependencies
└── README.md               # Project overview and instructions
```

---

## 🛣️ Roadmap

1. **Walk-Forward Validation:** automate rolling retrain & test windows for robust performance evaluation.
2. **Live/Paper Trading:** integrate CCXT or exchange APIs for real-time signal execution.
3. **Dashboard & Alerts:** build a Streamlit/React dashboard and email/Slack notifications.
4. **Portfolio Optimization:** risk-parity or mean-variance allocation across signals.
5. **Expanded Feature Sets:** incorporate on-chain metrics, sentiment analysis, macro indicators.

---

## 🤝 Contributing

Feel free to open issues or submit pull requests. Please:

* Fork the repo
* Create a feature branch (`git checkout -b feature/xyz`)
* Commit your changes (`git commit -am 'Add xyz'`)
* Push to the branch (`git push origin feature/xyz`)
* Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.

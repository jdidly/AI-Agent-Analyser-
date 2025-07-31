
# Crypto AI Agent Trading System

A sophisticated cryptocurrency trading AI agent that uses machine learning to predict profitable trades based on technical indicators.

## Features

- **Technical Analysis**: RSI, MACD, Bollinger Bands, EMA crossovers, volume spikes
- **Machine Learning**: Random Forest classifier for trade prediction
- **Risk Management**: Position sizing, stop-loss, profit targets
- **Realistic Trading**: Includes spreads, commissions, and slippage
- **Backtesting**: Historical performance analysis with metrics
- **Visualization**: Equity curves and performance charts

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the improved trading system:
```bash
python crypto_agent.py
```

3. Or use the original Jupyter notebook:
```bash
jupyter notebook "V12 Crypto Agent Sim.ipynb"
```

## Project Structure

```
├── crypto_agent.py          # Main trading system
├── data_handler.py          # Data fetching and preprocessing
├── ml_models.py             # Machine learning models
├── trading_strategy.py      # Trading logic and signals
├── risk_management.py       # Position sizing and risk controls
├── backtester.py           # Backtesting engine
├── config.py               # Configuration management
├── utils.py                # Utility functions
├── tests/                  # Unit tests
├── requirements.txt        # Dependencies
├── README.md              # This file
└── V12 Crypto Agent Sim.ipynb  # Original notebook (fixed)
```

## Configuration

Edit `config.py` to customize:
- Trading symbols
- Risk parameters
- Model parameters
- Backtesting settings

## Improvements Made

1. **Fixed Pandas Series Bug**: Resolved the ambiguous Series truth value error
2. **Modular Architecture**: Split code into logical modules
3. **Configuration Management**: Centralized parameter management
4. **Enhanced ML Models**: Better feature engineering and validation
5. **Improved Risk Management**: More sophisticated position sizing
6. **Logging System**: Comprehensive logging for debugging
7. **Unit Tests**: Test coverage for critical components
8. **Data Validation**: Robust error handling for data issues

## Performance Metrics

The system tracks:
- Total return
- Sharpe ratio
- Maximum drawdown
- Win rate
- Number of trades
- Risk-adjusted returns

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.
=======

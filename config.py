"""
Configuration settings for the Crypto AI Trading Agent
"""

# Trading Parameters
SYMBOLS = ['BTC-USD']
PERIOD = '1y'
INTERVAL = '1h'

# Risk Management
PROFIT_TARGET = 0.02        # 2%
STOP_LOSS = 0.015          # 1.5%
RISK_PCT = 0.01            # 1% of capital per trade
INITIAL_CAPITAL = 10000    # Starting capital

# Trading Costs
SPREAD = 0.0005            # 0.05% round-trip spread
COMMISSION = 0.001         # 0.1% round-trip commission

# ML Model Parameters
CONFIDENCE_THRESHOLD = 65  # Minimum prediction confidence
N_ESTIMATORS = 100
MAX_DEPTH = 10
RANDOM_STATE = 42

# Feature Engineering
RSI_PERIOD = 14
EMA_SHORT = 50
EMA_LONG = 200
BOLLINGER_PERIOD = 20
VOLUME_PERIOD = 20
MACD_FAST = 12
MACD_SLOW = 26

# Backtesting
TEST_SIZE = 0.2

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
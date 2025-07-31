"""
Test with simulated data to verify the system works
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from ml_models import MLModels
from trading_strategy import TradingStrategy

def create_mock_data(n_samples=500):
    """Create mock crypto trading data for testing"""
    np.random.seed(42)
    
    # Generate price data
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='H')
    
    # Random walk for price
    returns = np.random.normal(0, 0.02, n_samples)
    prices = 50000 * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, n_samples)
    }, index=dates)
    
    # Add technical indicators
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_fast = df['Close'].ewm(span=12).mean()
    ema_slow = df['Close'].ewm(span=26).mean()
    df['MACD_diff'] = ema_fast - ema_slow
    
    # EMA crossover
    df['EMA50'] = df['Close'].ewm(span=50).mean()
    df['EMA200'] = df['Close'].ewm(span=200).mean()
    df['EMA50_above_EMA200'] = (df['EMA50'] > df['EMA200']).astype(int)
    
    # Volume spike
    volume_ma = df['Volume'].rolling(20).mean()
    df['Volume_Spike'] = (df['Volume'] > volume_ma).astype(int)
    
    # Bollinger distance
    bb_mean = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['Bollinger_Dist'] = (df['Close'] - bb_mean) / (bb_std + 1e-9)
    
    # Price momentum
    df['Price_Momentum'] = df['Close'].pct_change(5)
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(20).std() / df['Close'].rolling(20).mean()
    
    # Target (future return > 2%)
    df['Future_Return'] = df['Close'].shift(-1) / df['Close'] - 1
    df['Target'] = (df['Future_Return'] >= 0.02).astype(int)
    
    # Clean data
    df = df.dropna()
    
    print(f"Created mock dataset with {len(df)} samples")
    print(f"Target distribution: {df['Target'].value_counts().to_dict()}")
    
    return df

def test_with_mock_data():
    """Test the system with mock data"""
    print("Testing with mock data...")
    
    df = create_mock_data()
    
    features = ['RSI', 'MACD_diff', 'EMA50_above_EMA200', 'Volume_Spike', 
                'Bollinger_Dist', 'Price_Momentum', 'Volatility']
    
    # Test ML model
    ml_model = MLModels()
    X, y = ml_model.prepare_data(df, features)
    
    if X is None:
        print("Failed to prepare data")
        return False
    
    print(f"Features prepared: {len(X)} samples, {len(features)} features")
    
    # Train model
    if not ml_model.train_model(X, y):
        print("Failed to train model")
        return False
    
    print("Model trained successfully")
    
    # Get feature importance
    importance = ml_model.get_feature_importance()
    if not importance.empty:
        print("Top features:")
        print(importance.head().to_string(index=False))
    
    # Test predictions
    predictions = ml_model.predict(X)
    probabilities = ml_model.predict_proba(X)
    
    print(f"Predictions generated: {len(predictions)} samples")
    print(f"Positive predictions: {predictions.sum()}")
    
    # Test strategy
    strategy = TradingStrategy()
    df_with_signals = strategy.generate_signals(df.loc[X.index], predictions, probabilities)
    
    analysis = strategy.analyze_signals()
    print(f"\nSignal Analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    
    # Simple backtest
    backtest = strategy.backtest_signals(df_with_signals)
    if backtest:
        print(f"\nBacktest Results:")
        for key, value in backtest.items():
            print(f"  {key}: {value}")
    
    print("\nSystem test completed successfully!")
    return True

if __name__ == "__main__":
    test_with_mock_data()
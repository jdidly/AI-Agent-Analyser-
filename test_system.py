"""
Quick test of the improved crypto agent with limited data
"""

import warnings
warnings.filterwarnings('ignore')

from data_handler import DataHandler
from ml_models import MLModels
from trading_strategy import TradingStrategy
import config

def test_system():
    print("Testing Crypto AI Trading System...")
    
    # Test with smaller dataset
    data_handler = DataHandler()
    df = data_handler.fetch_data('BTC-USD', period='3mo', interval='1d', profit_target=2.0)
    
    if df is None:
        print("Failed to fetch data")
        return False
    
    print(f"Data fetched: {len(df)} rows")
    
    # Test ML model
    features = data_handler.get_features()
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
    
    # Test predictions
    predictions = ml_model.predict(X)
    probabilities = ml_model.predict_proba(X)
    
    print(f"Predictions generated: {len(predictions)} samples")
    
    # Test strategy
    strategy = TradingStrategy()
    df_with_signals = strategy.generate_signals(df.loc[X.index], predictions, probabilities)
    
    analysis = strategy.analyze_signals()
    print(f"Signals generated: {analysis.get('buy_signals', 0)} buy signals")
    
    print("System test completed successfully!")
    return True

if __name__ == "__main__":
    test_system()
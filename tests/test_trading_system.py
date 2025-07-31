"""
Unit tests for the crypto trading system
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_handler import DataHandler
from ml_models import MLModels
from trading_strategy import TradingStrategy
import config


class TestDataHandler(unittest.TestCase):
    def setUp(self):
        self.data_handler = DataHandler()
        
    def test_calculate_rsi(self):
        """Test RSI calculation"""
        prices = pd.Series([44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 46.08, 45.89])
        rsi = self.data_handler._calculate_rsi(prices, period=5)
        
        # RSI should be between 0 and 100
        self.assertTrue(all(rsi.dropna() >= 0))
        self.assertTrue(all(rsi.dropna() <= 100))
    
    def test_create_target_labels(self):
        """Test target label creation"""
        df = pd.DataFrame({
            'Close': [100, 102, 105, 103, 107]  # 2%, 2.94%, -1.9%, 3.88% returns
        })
        
        df_with_targets = self.data_handler._create_target_labels(df, profit_target=2.0)
        
        # Check that targets are created correctly
        self.assertIn('Target', df_with_targets.columns)
        self.assertIn('Future_Return', df_with_targets.columns)
        
        # First target should be 1 (2% return >= 2% target)
        self.assertEqual(df_with_targets['Target'].iloc[0], 1)
        # Second target should be 1 (2.94% return >= 2% target)
        self.assertEqual(df_with_targets['Target'].iloc[1], 1)
        # Third target should be 0 (-1.9% return < 2% target)
        self.assertEqual(df_with_targets['Target'].iloc[2], 0)
    
    def test_validate_data(self):
        """Test data validation"""
        # Valid data
        valid_df = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [103, 104, 105],
            'Volume': [1000, 1100, 1200]
        })
        
        self.assertTrue(self.data_handler.validate_data(valid_df))
        
        # Invalid data (missing column)
        invalid_df = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Close': [103, 104, 105]
        })
        
        self.assertFalse(self.data_handler.validate_data(invalid_df))


class TestMLModels(unittest.TestCase):
    def setUp(self):
        self.ml_model = MLModels()
        
        # Create sample data
        np.random.seed(42)
        self.df = pd.DataFrame({
            'RSI': np.random.uniform(20, 80, 100),
            'MACD_diff': np.random.normal(0, 1, 100),
            'EMA50_above_EMA200': np.random.choice([0, 1], 100),
            'Volume_Spike': np.random.choice([0, 1], 100),
            'Bollinger_Dist': np.random.normal(0, 1, 100),
            'Price_Momentum': np.random.normal(0, 0.02, 100),
            'Volatility': np.random.uniform(0.01, 0.05, 100),
            'Target': np.random.choice([0, 1], 100, p=[0.8, 0.2])
        })
        
        self.features = ['RSI', 'MACD_diff', 'EMA50_above_EMA200', 'Volume_Spike', 
                        'Bollinger_Dist', 'Price_Momentum', 'Volatility']
        
    def test_prepare_data(self):
        """Test data preparation"""
        X, y = self.ml_model.prepare_data(self.df, self.features)
        
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(X.columns), len(self.features))
        
    def test_train_model(self):
        """Test model training"""
        X, y = self.ml_model.prepare_data(self.df, self.features)
        
        success = self.ml_model.train_model(X, y)
        self.assertTrue(success)
        self.assertTrue(self.ml_model.is_trained)
        
    def test_predictions(self):
        """Test model predictions"""
        X, y = self.ml_model.prepare_data(self.df, self.features)
        self.ml_model.train_model(X, y)
        
        predictions = self.ml_model.predict(X)
        probabilities = self.ml_model.predict_proba(X)
        
        self.assertEqual(len(predictions), len(X))
        self.assertEqual(len(probabilities), len(X))
        self.assertEqual(probabilities.shape[1], 2)  # Binary classification
        
        # Predictions should be 0 or 1
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
        
        # Probabilities should sum to 1
        prob_sums = probabilities.sum(axis=1)
        self.assertTrue(all(abs(s - 1.0) < 1e-6 for s in prob_sums))


class TestTradingStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = TradingStrategy()
        
        # Create sample data with signals
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        self.df = pd.DataFrame({
            'Close': 50000 + np.cumsum(np.random.normal(0, 100, 100)),
            'RSI': np.random.uniform(20, 80, 100),
            'MACD_diff': np.random.normal(0, 10, 100),
            'EMA50': np.random.uniform(49000, 51000, 100),
            'EMA200': np.random.uniform(48000, 52000, 100),
            'Volume': np.random.randint(1000, 5000, 100),
            'Volatility': np.random.uniform(0.01, 0.05, 100),
            'Future_Return': np.random.normal(0, 0.02, 100)
        }, index=dates)
        
        self.df['EMA50_above_EMA200'] = (self.df['EMA50'] > self.df['EMA200']).astype(int)
        self.df['Volume_Spike'] = np.random.choice([0, 1], 100)
        self.df['Bollinger_Dist'] = np.random.normal(0, 1, 100)
        
        # Mock ML predictions
        self.predictions = np.random.choice([0, 1], 100, p=[0.8, 0.2])
        self.probabilities = np.random.rand(100, 2)
        self.probabilities[:, 1] = np.random.uniform(0.5, 1.0, 100)  # Confidence scores
        self.probabilities[:, 0] = 1 - self.probabilities[:, 1]
        
    def test_generate_signals(self):
        """Test signal generation"""
        df_with_signals = self.strategy.generate_signals(
            self.df, self.predictions, self.probabilities
        )
        
        # Check that signal columns are added
        self.assertIn('ML_Prediction', df_with_signals.columns)
        self.assertIn('ML_Confidence', df_with_signals.columns)
        self.assertIn('Buy_Signal', df_with_signals.columns)
        self.assertIn('Final_Signal', df_with_signals.columns)
        self.assertIn('Signal_Strength', df_with_signals.columns)
        
        # Signals should be binary
        self.assertTrue(all(s in [0, 1] for s in df_with_signals['Final_Signal']))
        
        # Signal strength should be between 0 and 1
        self.assertTrue(all(0 <= s <= 1 for s in df_with_signals['Signal_Strength']))
        
    def test_analyze_signals(self):
        """Test signal analysis"""
        df_with_signals = self.strategy.generate_signals(
            self.df, self.predictions, self.probabilities
        )
        
        analysis = self.strategy.analyze_signals()
        
        # Check that analysis contains expected keys
        expected_keys = ['total_periods', 'buy_signals', 'signal_rate', 'avg_confidence']
        for key in expected_keys:
            self.assertIn(key, analysis)
            
        # Values should be reasonable
        self.assertEqual(analysis['total_periods'], len(self.df))
        self.assertTrue(0 <= analysis['signal_rate'] <= 100)
        self.assertTrue(0 <= analysis['avg_confidence'] <= 100)


class TestConfigValidation(unittest.TestCase):
    def test_valid_config(self):
        """Test valid configuration"""
        from utils import validate_config
        
        valid_config = {
            'SYMBOLS': ['BTC-USD'],
            'PROFIT_TARGET': 0.02,
            'STOP_LOSS': 0.015,
            'INITIAL_CAPITAL': 10000,
            'RISK_PCT': 0.01,
            'CONFIDENCE_THRESHOLD': 65
        }
        
        self.assertTrue(validate_config(valid_config))
        
    def test_invalid_config(self):
        """Test invalid configuration"""
        from utils import validate_config
        
        # Missing required parameter
        invalid_config = {
            'SYMBOLS': ['BTC-USD'],
            'PROFIT_TARGET': 0.02,
            # Missing STOP_LOSS
        }
        
        self.assertFalse(validate_config(invalid_config))
        
        # Invalid range
        invalid_config2 = {
            'SYMBOLS': ['BTC-USD'],
            'PROFIT_TARGET': -0.02,  # Negative
            'STOP_LOSS': 0.015,
            'INITIAL_CAPITAL': 10000,
            'RISK_PCT': 0.01,
            'CONFIDENCE_THRESHOLD': 65
        }
        
        self.assertFalse(validate_config(invalid_config2))


if __name__ == '__main__':
    unittest.main()
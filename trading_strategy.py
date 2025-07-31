"""
Trading strategy and signal generation
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional
import config

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class TradingStrategy:
    """Generates trading signals based on ML predictions and technical analysis"""
    
    def __init__(self):
        self.signals = None
        self.confidence_scores = None
    
    def generate_signals(self, df: pd.DataFrame, predictions: np.ndarray, 
                        probabilities: np.ndarray) -> pd.DataFrame:
        """
        Generate trading signals based on ML predictions
        
        Args:
            df: DataFrame with market data
            predictions: ML model predictions
            probabilities: ML model prediction probabilities
            
        Returns:
            DataFrame with trading signals
        """
        try:
            df_signals = df.copy()
            
            # Add ML predictions
            df_signals['ML_Prediction'] = predictions
            df_signals['ML_Confidence'] = probabilities[:, 1] * 100  # Probability of positive class
            
            # Generate buy signals based on confidence threshold
            df_signals['Buy_Signal'] = (
                df_signals['ML_Confidence'] >= config.CONFIDENCE_THRESHOLD
            ).astype(int)
            
            # Add technical analysis filters
            df_signals = self._add_technical_filters(df_signals)
            
            # Combine ML and technical signals
            df_signals['Final_Signal'] = (
                df_signals['Buy_Signal'] & 
                df_signals['Technical_Filter']
            ).astype(int)
            
            # Add signal strength
            df_signals['Signal_Strength'] = self._calculate_signal_strength(df_signals)
            
            self.signals = df_signals
            
            signal_count = df_signals['Final_Signal'].sum()
            logger.info(f"Generated {signal_count} buy signals out of {len(df_signals)} periods")
            
            return df_signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return df
    
    def _add_technical_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis filters to improve signal quality"""
        
        # Trend filter: Only buy in uptrend (EMA50 > EMA200)
        trend_filter = df['EMA50_above_EMA200'] == 1
        
        # RSI filter: Avoid overbought conditions
        rsi_filter = df['RSI'] < 70
        
        # Volume filter: Require volume confirmation
        volume_filter = df['Volume_Spike'] == 1
        
        # Volatility filter: Avoid extremely volatile periods
        if 'Volatility' in df.columns:
            volatility_filter = df['Volatility'] < df['Volatility'].quantile(0.9)
        else:
            volatility_filter = pd.Series(True, index=df.index)
        
        # Combine all technical filters
        df['Technical_Filter'] = (
            trend_filter & 
            rsi_filter & 
            volume_filter & 
            volatility_filter
        ).astype(int)
        
        return df
    
    def _calculate_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate signal strength based on multiple factors"""
        
        strength = pd.Series(0.0, index=df.index)
        
        # ML confidence contributes to strength
        strength += df['ML_Confidence'] / 100 * 0.4
        
        # RSI momentum
        rsi_momentum = (50 - abs(df['RSI'] - 50)) / 50  # Higher when RSI near 50
        strength += rsi_momentum * 0.2
        
        # MACD strength
        if 'MACD_diff' in df.columns:
            macd_strength = np.tanh(df['MACD_diff'] / df['MACD_diff'].std()) * 0.2
            strength += macd_strength.fillna(0)
        
        # Volume strength  
        strength += df['Volume_Spike'] * 0.1
        
        # Bollinger position
        if 'Bollinger_Dist' in df.columns:
            # Stronger signal when price is oversold (negative Bollinger distance)
            bollinger_strength = np.clip(-df['Bollinger_Dist'], 0, 1) * 0.1
            strength += bollinger_strength.fillna(0)
        
        return np.clip(strength, 0, 1)  # Normalize to [0, 1]
    
    def get_entry_signals(self) -> pd.DataFrame:
        """Get DataFrame with only entry signals"""
        if self.signals is None:
            logger.error("No signals generated. Call generate_signals() first.")
            return pd.DataFrame()
        
        entry_signals = self.signals[self.signals['Final_Signal'] == 1].copy()
        return entry_signals
    
    def analyze_signals(self) -> dict:
        """Analyze signal quality and characteristics"""
        if self.signals is None:
            logger.error("No signals generated. Call generate_signals() first.")
            return {}
        
        total_signals = len(self.signals)
        buy_signals = self.signals['Final_Signal'].sum()
        ml_signals = self.signals['Buy_Signal'].sum()
        technical_filtered = self.signals['Technical_Filter'].sum()
        
        avg_confidence = self.signals[self.signals['Final_Signal'] == 1]['ML_Confidence'].mean()
        avg_strength = self.signals[self.signals['Final_Signal'] == 1]['Signal_Strength'].mean()
        
        analysis = {
            'total_periods': total_signals,
            'buy_signals': buy_signals,
            'ml_signals': ml_signals,
            'technical_filtered': technical_filtered,
            'signal_rate': buy_signals / total_signals * 100,
            'filter_efficiency': buy_signals / ml_signals * 100 if ml_signals > 0 else 0,
            'avg_confidence': avg_confidence if not pd.isna(avg_confidence) else 0,
            'avg_strength': avg_strength if not pd.isna(avg_strength) else 0
        }
        
        logger.info("Signal Analysis:")
        logger.info(f"  Total periods: {analysis['total_periods']}")
        logger.info(f"  Buy signals: {analysis['buy_signals']} ({analysis['signal_rate']:.1f}%)")
        logger.info(f"  ML signals (before filter): {analysis['ml_signals']}")
        logger.info(f"  Filter efficiency: {analysis['filter_efficiency']:.1f}%")
        logger.info(f"  Average confidence: {analysis['avg_confidence']:.1f}%")
        logger.info(f"  Average strength: {analysis['avg_strength']:.3f}")
        
        return analysis
    
    def backtest_signals(self, df: pd.DataFrame) -> dict:
        """
        Simple backtest of signals against future returns
        
        Args:
            df: DataFrame with signals and future returns
            
        Returns:
            Dictionary with backtest results
        """
        if 'Future_Return' not in df.columns:
            logger.error("Future_Return column not found for backtesting")
            return {}
        
        signals_df = df[df['Final_Signal'] == 1].copy()
        
        if len(signals_df) == 0:
            logger.warning("No signals to backtest")
            return {}
        
        # Calculate returns for signaled periods
        returns = signals_df['Future_Return']
        
        # Basic statistics
        total_returns = len(returns)
        positive_returns = (returns > 0).sum()
        negative_returns = (returns <= 0).sum()
        
        win_rate = positive_returns / total_returns * 100 if total_returns > 0 else 0
        avg_return = returns.mean() * 100
        std_return = returns.std() * 100
        
        # Risk metrics
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        max_return = returns.max() * 100
        min_return = returns.min() * 100
        
        results = {
            'total_signals': total_returns,
            'winning_signals': positive_returns,
            'losing_signals': negative_returns,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'return_std': std_return,
            'sharpe_ratio': sharpe_ratio,
            'max_return': max_return,
            'min_return': min_return
        }
        
        logger.info("Signal Backtest Results:")
        logger.info(f"  Total signals: {results['total_signals']}")
        logger.info(f"  Win rate: {results['win_rate']:.1f}%")
        logger.info(f"  Average return: {results['avg_return']:.2f}%")
        logger.info(f"  Sharpe ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"  Max return: {results['max_return']:.2f}%")
        logger.info(f"  Min return: {results['min_return']:.2f}%")
        
        return results
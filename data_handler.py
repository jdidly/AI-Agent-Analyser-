"""
Data handling and preprocessing for crypto trading data
"""

import yfinance as yf
import pandas as pd
import numpy as np
import logging
from typing import Optional
import config

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class DataHandler:
    """Handles data fetching and preprocessing for crypto trading"""
    
    def __init__(self):
        self.data = None
    
    def fetch_data(self, symbol: str = 'BTC-USD', period: str = '1y', 
                   interval: str = '1h', profit_target: float = 2.0) -> Optional[pd.DataFrame]:
        """
        Fetch and preprocess cryptocurrency data
        
        Args:
            symbol: Trading symbol (e.g., 'BTC-USD')
            period: Data period (e.g., '1y', '6mo', '3mo')
            interval: Data interval (e.g., '1h', '1d')
            profit_target: Target profit percentage for labeling
            
        Returns:
            DataFrame with processed features or None if failed
        """
        try:
            logger.info(f"Fetching data for {symbol} with period {period} and interval {interval}")
            
            # Download data
            df = yf.download(symbol, period=period, interval=interval,
                           progress=False, auto_adjust=True)
            
            if df.empty:
                logger.error(f"No data retrieved for {symbol}")
                return None
            
            logger.info(f"Retrieved {len(df)} data points for {symbol}")
            
            # Add technical indicators
            df = self._add_technical_indicators(df)
            
            # Create target labels
            df = self._create_target_labels(df, profit_target)
            
            # Clean data
            df = self._clean_data(df)
            
            self.data = df
            logger.info(f"Data preprocessing complete. Final dataset has {len(df)} rows")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe"""
        
        # Bollinger Bands distance (normalized)
        bb_mean = df['Close'].rolling(config.BOLLINGER_PERIOD).mean()
        bb_std = df['Close'].rolling(config.BOLLINGER_PERIOD).std()
        df['Bollinger_Dist'] = (df['Close'] - bb_mean) / (bb_std + 1e-9)
        
        # EMA crossover
        df['EMA50'] = df['Close'].ewm(span=config.EMA_SHORT).mean()
        df['EMA200'] = df['Close'].ewm(span=config.EMA_LONG).mean()
        df['EMA50_above_EMA200'] = (df['EMA50'] > df['EMA200']).astype(int)
        
        # MACD difference
        ema_fast = df['Close'].ewm(span=config.MACD_FAST).mean()
        ema_slow = df['Close'].ewm(span=config.MACD_SLOW).mean()
        df['MACD_diff'] = ema_fast - ema_slow
        
        # RSI
        df['RSI'] = self._calculate_rsi(df['Close'], config.RSI_PERIOD)
        
        # Volume spike indicator
        volume_ma = df['Volume'].rolling(config.VOLUME_PERIOD).mean()
        df['Volume_Spike'] = (df['Volume'] > volume_ma).astype(int)
        
        # Price momentum
        df['Price_Momentum'] = df['Close'].pct_change(5)  # 5-period momentum
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(20).std() / df['Close'].rolling(20).mean()
        
        logger.info("Technical indicators added successfully")
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(com=period-1, adjust=False).mean()
        avg_loss = loss.ewm(com=period-1, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _create_target_labels(self, df: pd.DataFrame, profit_target: float) -> pd.DataFrame:
        """Create target labels for machine learning"""
        df['Future_Return'] = df['Close'].shift(-1) / df['Close'] - 1
        df['Target'] = (df['Future_Return'] >= profit_target / 100).astype(int)
        
        logger.info(f"Target labels created with {df['Target'].sum()} positive samples out of {len(df)}")
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data by removing NaN values and outliers"""
        initial_rows = len(df)
        
        # Remove NaN values
        df = df.dropna()
        
        # Remove extreme outliers (optional)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['Target', 'EMA50_above_EMA200', 'Volume_Spike']:
                q1 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                df = df[(df[col] >= q1) & (df[col] <= q99)]
        
        final_rows = len(df)
        removed_rows = initial_rows - final_rows
        
        logger.info(f"Data cleaning complete. Removed {removed_rows} rows ({removed_rows/initial_rows*100:.1f}%)")
        
        return df
    
    def get_features(self) -> list:
        """Get list of feature column names"""
        return ['RSI', 'MACD_diff', 'EMA50_above_EMA200', 'Volume_Spike', 
                'Bollinger_Dist', 'Price_Momentum', 'Volatility']
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data quality"""
        if df is None or df.empty:
            logger.error("DataFrame is None or empty")
            return False
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for sufficient data
        if len(df) < 200:
            logger.warning(f"Limited data available: {len(df)} rows")
        
        logger.info("Data validation passed")
        return True
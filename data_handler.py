"""
Data handling and preprocessing for crypto trading data
"""

import requests
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
    
    def fetch_data(self, symbol: str = 'BTC-USD', period: str = None, 
                   interval: str = '1h', profit_target: float = 2.0, limit: int = 1000) -> Optional[pd.DataFrame]:
        """
        Fetch and preprocess cryptocurrency data from Binance API
        Args:
            symbol: Trading symbol (e.g., 'BTC-USD')
            period: (ignored, for compatibility)
            interval: Data interval (e.g., '1h', '1d')
            profit_target: Target profit percentage for labeling
            limit: Number of data points to fetch (max 1000 for Binance public API)
        Returns:
            DataFrame with processed features or None if failed
        """
        try:
            logger.info(f"Fetching data for {symbol} from Binance with interval {interval}")
            binance_symbol = symbol.replace('-USD', 'USDT')
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': binance_symbol,
                'interval': interval,
                'limit': limit
            }
            r = requests.get(url, params=params)
            data = r.json()
            if isinstance(data, dict) and data.get('code'):
                logger.error(f"Failed to fetch data: {data.get('msg', 'Unknown error')}")
                return None
            df = pd.DataFrame(data, columns=[
                'open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'close_time', 'Quote_asset_volume', 'Number_of_trades',
                'Taker_buy_base', 'Taker_buy_quote', 'Ignore'
            ])
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('open_time', inplace=True)
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = df[col].astype(float)
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

        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(14).mean()

        # Stochastic Oscillator %K
        low_min = df['Low'].rolling(window=14).min()
        high_max = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-9)

        # Williams %R
        df['WilliamsR'] = -100 * (high_max - df['Close']) / (high_max - low_min + 1e-9)

        # CCI (Commodity Channel Index)
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        cci_ma = tp.rolling(20).mean()
        cci_std = tp.rolling(20).std()
        df['CCI'] = (tp - cci_ma) / (0.015 * cci_std + 1e-9)

        # ADX (Average Directional Index)
        plus_dm = df['High'].diff()
        minus_dm = df['Low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        tr = true_range
        plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / (tr.ewm(alpha=1/14).mean() + 1e-9))
        minus_di = 100 * (minus_dm.abs().ewm(alpha=1/14).mean() / (tr.ewm(alpha=1/14).mean() + 1e-9))
        dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
        df['ADX'] = dx.ewm(alpha=1/14).mean()

        # OBV (On-Balance Volume)
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['OBV'] = obv

        # ROC (Rate of Change)
        df['ROC'] = df['Close'].pct_change(10)

        # SMA (Simple Moving Average)
        df['SMA20'] = df['Close'].rolling(20).mean()

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
        return [
            'RSI', 'MACD_diff', 'EMA50_above_EMA200', 'Volume_Spike',
            'Bollinger_Dist', 'Price_Momentum', 'Volatility',
            'ATR', 'Stoch_K', 'WilliamsR', 'CCI', 'ADX', 'OBV', 'ROC', 'SMA20'
        ]
    
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
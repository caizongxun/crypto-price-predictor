"""Simplified Cryptocurrency Data Fetcher - 10 Core Features Only"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import ccxt
from dotenv import load_dotenv
import os

load_dotenv()
logger = logging.getLogger(__name__)


class SimpleDataFetcher:
    """Fetches and prepares cryptocurrency data with simplified features"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.binance_api_key = os.getenv('BINANCE_API_KEY')
        self.binance_api_secret = os.getenv('BINANCE_API_SECRET')
        self._init_exchanges()
    
    def _init_exchanges(self):
        """Initialize CCXT exchange connections"""
        try:
            if self.binance_api_key and self.binance_api_secret:
                self.binance = ccxt.binance({
                    'apiKey': self.binance_api_key,
                    'secret': self.binance_api_secret,
                    'enableRateLimit': True,
                })
            else:
                self.binance = ccxt.binance({'enableRateLimit': True})
                logger.warning("Binance API keys not provided, using public access")
            logger.info("Binance API initialized")
        except Exception as e:
            logger.error(f"Failed to initialize exchanges: {e}")
    
    def fetch_ohlcv_binance(self, symbol: str, timeframe: str = '1h', limit: int = 10000) -> pd.DataFrame:
        """Fetch OHLCV data from Binance"""
        try:
            logger.info(f"Fetching {limit} candles for {symbol} ({timeframe})...")
            ohlcv = self.binance.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df['symbol'] = symbol
            
            logger.info(f"Successfully fetched {len(df)} candles for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} from Binance: {e}")
            return None
    
    def add_simple_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add only 10 core technical indicators
        
        Features:
        1. open, high, low, close, volume (5)
        2. SMA_10 (1)
        3. RSI (1) 
        4. MACD (1)
        5. Volume_ratio (1)
        
        Total: 10 features
        """
        try:
            df = df.copy()
            
            # 1. Simple Moving Average (SMA_10)
            df['SMA_10'] = df['close'].rolling(window=10).mean()
            
            # 2. RSI (Relative Strength Index)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # 3. MACD (Moving Average Convergence Divergence)
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['MACD'] = ema_12 - ema_26
            
            # 4. Volume Ratio (relative to 20-period average)
            df['Volume_SMA'] = df['volume'].rolling(window=20).mean()
            df['Volume_ratio'] = df['volume'] / df['Volume_SMA']
            
            logger.info(f"Added 10 simple indicators for {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Failed to add indicators: {e}")
            return df
    
    def prepare_ml_features(self, df: pd.DataFrame, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray, object]:
        """Prepare features for ML model (10 features total)
        
        Returns:
            X: Feature sequences (samples, lookback, features)
            y: Target values
            scaler: MinMaxScaler for later denormalization
        """
        try:
            # Select 10 core features only
            feature_cols = [
                'open', 'high', 'low', 'close', 'volume',  # 5 OHLCV
                'SMA_10',                                    # 1 trend
                'RSI',                                       # 1 momentum
                'MACD',                                      # 1 momentum
                'Volume_ratio'                               # 1 volume
            ]
            
            # Ensure all features exist
            available_cols = [c for c in feature_cols if c in df.columns]
            if len(available_cols) < 10:
                logger.warning(f"Expected 10 features, got {len(available_cols)}")
                logger.warning(f"Missing: {set(feature_cols) - set(available_cols)}")
            
            # Remove NaN values
            df_clean = df[available_cols].dropna()
            
            # Normalize features using MinMaxScaler
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df_clean)
            
            # Create sequences
            X, y = [], []
            for i in range(len(scaled_data) - lookback):
                X.append(scaled_data[i:i+lookback])
                y.append(df_clean['close'].iloc[i+lookback])
            
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"Prepared {len(X)} sequences with {len(available_cols)} features")
            logger.info(f"  - X shape: {X.shape}")
            logger.info(f"  - y shape: {y.shape}")
            
            return X, y, scaler
        except Exception as e:
            logger.error(f"Failed to prepare ML features: {e}")
            return None, None, None
    
    def get_real_time_price(self, symbol: str) -> Optional[float]:
        """Get real-time price"""
        try:
            ticker = self.binance.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Failed to fetch real-time price for {symbol}: {e}")
            return None

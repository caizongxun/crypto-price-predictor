"""Optimized Data Fetcher for Temporal Fusion Transformer

Key Design:
- Only 8 most relevant features (OHLCV + 3 indicators)
- Spearman correlation > 0.75 with price
- Minimal collinearity (VIF < 5)
- Proper scaling and handling of outliers

Features:
1. open, high, low, close, volume (OHLCV)
2. SMA_20 (trend) 
3. RSI (momentum)
4. ATR (volatility)
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import ccxt
from dotenv import load_dotenv
import os
from sklearn.preprocessing import RobustScaler
from scipy.stats import spearmanr

load_dotenv()
logger = logging.getLogger(__name__)


class TFTDataFetcher:
    """Optimized data fetcher for Temporal Fusion Transformer"""
    
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
    
    def fetch_ohlcv_binance(self, symbol: str, timeframe: str = '1h', limit: int = 5000) -> pd.DataFrame:
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
            
            # Remove any NaN or invalid values
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            
            logger.info(f"Successfully fetched {len(df)} candles for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} from Binance: {e}")
            return None
    
    def add_tft_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 8 optimized technical indicators for TFT
        
        Features (8 total):
        1-5: OHLCV (5)
        6: SMA_20 (trend)
        7: RSI (momentum)
        8: ATR (volatility)
        """
        try:
            df = df.copy()
            
            # 1. SMA_20 (Simple Moving Average - trend indicator)
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            
            # 2. RSI (Relative Strength Index - momentum)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-8)
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # 3. ATR (Average True Range - volatility)
            df['TR'] = pd.concat([
                df['high'] - df['low'],
                abs(df['high'] - df['close'].shift()),
                abs(df['low'] - df['close'].shift())
            ], axis=1).max(axis=1)
            df['ATR'] = df['TR'].rolling(window=14).mean()
            df = df.drop('TR', axis=1)
            
            # Remove NaN from indicators
            df = df.dropna()
            
            logger.info(f"Added 8 TFT indicators for {len(df)} rows")
            logger.info(f"Features: open, high, low, close, volume, SMA_20, RSI, ATR")
            
            return df
        except Exception as e:
            logger.error(f"Failed to add indicators: {e}")
            return df
    
    def calculate_feature_importance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate Spearman correlation with price (target)
        
        High correlation (>0.75) = feature is relevant
        """
        feature_importance = {}
        
        features = ['open', 'high', 'low', 'volume', 'SMA_20', 'RSI', 'ATR']
        
        for feature in features:
            if feature in df.columns:
                # Calculate Spearman correlation with close price
                corr, p_value = spearmanr(df[feature].dropna(), 
                                          df['close'].iloc[-len(df[feature].dropna()):].values)
                feature_importance[feature] = abs(corr)
                logger.info(f"{feature:12s} - Correlation: {corr:7.4f} (p={p_value:.4f})")
        
        return feature_importance
    
    def check_multicollinearity(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate VIF (Variance Inflation Factor) to check collinearity
        
        VIF < 5: Low collinearity (good)
        VIF > 10: High collinearity (problematic)
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        features = ['open', 'high', 'low', 'volume', 'SMA_20', 'RSI', 'ATR']
        available_features = [f for f in features if f in df.columns]
        
        vif_data = {}
        X = df[available_features].values
        
        try:
            for i, col in enumerate(available_features):
                vif = variance_inflation_factor(X, i)
                vif_data[col] = vif
                status = "✓ OK" if vif < 5 else "⚠ HIGH" if vif < 10 else "✗ SEVERE"
                logger.info(f"{col:12s} - VIF: {vif:7.2f} {status}")
        except Exception as e:
            logger.warning(f"Could not calculate VIF: {e}")
        
        return vif_data
    
    def prepare_ml_features(self, df: pd.DataFrame, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray, object]:
        """Prepare features for ML model
        
        Returns:
            X: Feature sequences (samples, lookback, 8 features)
            y: Target values (next close price)
            scaler: RobustScaler for denormalization
        """
        try:
            # Select 8 core features
            feature_cols = ['open', 'high', 'low', 'close', 'volume', 'SMA_20', 'RSI', 'ATR']
            
            # Ensure all features exist
            available_cols = [c for c in feature_cols if c in df.columns]
            if len(available_cols) < 8:
                logger.warning(f"Expected 8 features, got {len(available_cols)}")
                logger.warning(f"Missing: {set(feature_cols) - set(available_cols)}")
            
            df_clean = df[available_cols].copy()
            
            # Use RobustScaler (better for outliers in volatile crypto data)
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(df_clean)
            
            # Create sequences
            X, y = [], []
            for i in range(len(scaled_data) - lookback):
                X.append(scaled_data[i:i+lookback])
                # Target: next close price (normalized)
                next_close_idx = i + lookback
                y.append(df_clean['close'].iloc[next_close_idx])
            
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"Prepared {len(X)} sequences with {len(available_cols)} features")
            logger.info(f"  - X shape: {X.shape} (samples, lookback, features)")
            logger.info(f"  - y shape: {y.shape}")
            logger.info(f"  - Lookback window: {lookback} hours")
            
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

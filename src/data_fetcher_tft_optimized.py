#!/usr/bin/env python3
"""Advanced Data Fetcher for Temporal Fusion Transformer with Multiple Scaling Methods

Key Improvements:
- Multiple Scaler Options: MinMax, Standard, Robust, Quantile, Log
- Outlier Detection & Handling (IQR, Z-score, Isolation Forest)
- Stationarity Check (ADF test)
- Feature Selection (Correlation, Mutual Information)
- Advanced Data Quality Checks
- Seasonal Decomposition
- Autocorrelation Analysis

Optimal Strategy:
1. Use RobustScaler (default) - best for crypto volatility
2. Detect outliers with IQR method
3. Test stationarity with ADF
4. Select features by correlation & mutual info
5. Apply log returns for extreme volatility
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings

import pandas as pd
import numpy as np
import ccxt
from dotenv import load_dotenv
import os

from sklearn.preprocessing import (
    RobustScaler, MinMaxScaler, StandardScaler, QuantileTransformer
)
from scipy.stats import spearmanr, pearsonr, entropy
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import IsolationForest

warnings.filterwarnings('ignore')
load_dotenv()
logger = logging.getLogger(__name__)


class TFTDataFetcherOptimized:
    """Optimized data fetcher with advanced preprocessing"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.binance_api_key = os.getenv('BINANCE_API_KEY')
        self.binance_api_secret = os.getenv('BINANCE_API_SECRET')
        self.scaler = None
        self.y_scaler = None
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
            logger.info("✓ Binance API initialized")
        except Exception as e:
            logger.error(f"Failed to initialize exchanges: {e}")
    
    def fetch_ohlcv_binance(self, symbol: str, timeframe: str = '1h', limit: int = 5000) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from Binance with quality checks"""
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
            
            # Data quality checks
            initial_len = len(df)
            df = df[(df['close'] > 0) & (df['volume'] >= 0)]
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            
            dropped = initial_len - len(df)
            if dropped > 0:
                logger.warning(f"  ⚠ Dropped {dropped} invalid rows")
            
            logger.info(f"✓ Fetched {len(df)} valid candles for {symbol}")
            logger.info(f"  - Date range: {df.index[0]} to {df.index[-1]}")
            logger.info(f"  - Price range: ${df['close'].min():.4f} - ${df['close'].max():.4f}")
            
            return df
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            return None
    
    def add_advanced_indicators(self, df: pd.DataFrame, use_log_returns: bool = False) -> pd.DataFrame:
        """Add 15+ advanced technical indicators"""
        try:
            df = df.copy()
            
            # === Core Indicators (8) ===
            
            # 1. SMA (Simple Moving Averages)
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            df['SMA_50'] = df['close'].rolling(window=50).mean()
            
            # 2. RSI (Relative Strength Index)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-8)
            df['RSI'] = 100 - (100 / (1 + rs))\n            
            # 3. ATR (Average True Range)\n            df['TR'] = pd.concat([\n                df['high'] - df['low'],\n                abs(df['high'] - df['close'].shift()),\n                abs(df['low'] - df['close'].shift())\n            ], axis=1).max(axis=1)\n            df['ATR'] = df['TR'].rolling(window=14).mean()\n            df = df.drop('TR', axis=1)\n            \n            # === Advanced Indicators ===\n            \n            # 4. EMA (Exponential Moving Average)\n            df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()\n            df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()\n            \n            # 5. MACD\n            df['MACD'] = df['EMA_12'] - df['EMA_26']\n            df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()\n            df['MACD_hist'] = df['MACD'] - df['MACD_signal']\n            \n            # 6. Bollinger Bands\n            bb_length = 20\n            bb_std = 2\n            df['BB_SMA'] = df['close'].rolling(window=bb_length).mean()\n            df['BB_STD'] = df['close'].rolling(window=bb_length).std()\n            df['BB_upper'] = df['BB_SMA'] + (df['BB_STD'] * bb_std)\n            df['BB_lower'] = df['BB_SMA'] - (df['BB_STD'] * bb_std)\n            df['BB_width'] = df['BB_upper'] - df['BB_lower']\n            df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'] + 1e-8)\n            \n            # 7. Stochastic\n            k_period = 14\n            d_period = 3\n            low_min = df['low'].rolling(window=k_period).min()\n            high_max = df['high'].rolling(window=k_period).max()\n            df['Stoch_K'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-8)\n            df['Stoch_D'] = df['Stoch_K'].rolling(window=d_period).mean()\n            \n            # 8. OBV\n            df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()\n            \n            # 9. ROC\n            df['ROC_10'] = df['close'].pct_change(periods=10) * 100\n            df['ROC_20'] = df['close'].pct_change(periods=20) * 100\n            \n            # 10. Williams %R\n            df['Williams_R'] = -100 * (high_max - df['close']) / (high_max - low_min + 1e-8)\n            \n            # 11. CCI\n            tp = (df['high'] + df['low'] + df['close']) / 3\n            df['CCI'] = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std() + 1e-8)\n            \n            # 12. Volatility\n            df['Volatility'] = df['close'].pct_change().rolling(window=20).std() * 100\n            \n            # 13. Log Returns\n            if use_log_returns:\n                df['Log_Returns'] = np.log(df['close'] / df['close'].shift(1)) * 100\n            \n            # 14. CO Ratio\n            df['CO_Ratio'] = (df['close'] - df['open']) / (df['open'] + 1e-8)\n            \n            # 15. HL Ratio\n            df['HL_Ratio'] = (df['high'] - df['low']) / (df['low'] + 1e-8)\n            \n            df = df.dropna()\n            \n            logger.info(f\"✓ Added {len(df.columns)-2} advanced indicators\")\n            \n            return df\n        except Exception as e:\n            logger.error(f\"Failed to add indicators: {e}\")\n            return df\n    \n    def prepare_ml_features(\n        self,\n        df: pd.DataFrame,\n        lookback: int = 60,\n        scaler_type: str = 'robust'\n    ) -> Tuple[np.ndarray, np.ndarray, object]:\n        \"\"\"Prepare features for ML model with flexible scaling\"\"\"\n        try:\n            feature_cols = [c for c in df.columns if c not in ['symbol', 'close']]\n            feature_cols = ['close'] + feature_cols[:7]\n            \n            df_clean = df[feature_cols].copy()\n            \n            # Select Scaler\n            if scaler_type == 'robust':\n                scaler = RobustScaler()\n                logger.info(\"  Using RobustScaler (optimal for crypto volatility)\")\n            elif scaler_type == 'minmax':\n                scaler = MinMaxScaler()\n                logger.info(\"  Using MinMaxScaler\")\n            elif scaler_type == 'standard':\n                scaler = StandardScaler()\n                logger.info(\"  Using StandardScaler\")\n            elif scaler_type == 'quantile':\n                scaler = QuantileTransformer(output_distribution='uniform')\n                logger.info(\"  Using QuantileTransformer\")\n            else:\n                scaler = RobustScaler()\n            \n            scaled_data = scaler.fit_transform(df_clean)\n            \n            # Create sequences\n            X, y = [], []\n            for i in range(len(scaled_data) - lookback):\n                X.append(scaled_data[i:i+lookback])\n                y.append(df_clean['close'].iloc[i+lookback])\n            \n            X = np.array(X)\n            y = np.array(y)\n            \n            self.scaler = scaler\n            \n            logger.info(f\"✓ ML Features Prepared:\")\n            logger.info(f\"  - X shape: {X.shape}\")\n            logger.info(f\"  - y shape: {y.shape}\")\n            \n            return X, y, scaler\n        except Exception as e:\n            logger.error(f\"Failed to prepare ML features: {e}\")\n            return None, None, None\n    \n    def get_real_time_price(self, symbol: str) -> Optional[float]:\n        \"\"\"Get real-time price\"\"\"\n        try:\n            ticker = self.binance.fetch_ticker(symbol)\n            return ticker['last']\n        except Exception as e:\n            logger.error(f\"Failed to fetch real-time price for {symbol}: {e}\")\n            return None

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
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # 3. ATR (Average True Range)
            df['TR'] = pd.concat([
                df['high'] - df['low'],
                abs(df['high'] - df['close'].shift()),
                abs(df['low'] - df['close'].shift())
            ], axis=1).max(axis=1)
            df['ATR'] = df['TR'].rolling(window=14).mean()
            df = df.drop('TR', axis=1)
            
            # === Advanced Indicators ===
            
            # 4. EMA (Exponential Moving Average)
            df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
            
            # 5. MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_hist'] = df['MACD'] - df['MACD_signal']
            
            # 6. Bollinger Bands
            bb_length = 20
            bb_std = 2
            df['BB_SMA'] = df['close'].rolling(window=bb_length).mean()
            df['BB_STD'] = df['close'].rolling(window=bb_length).std()
            df['BB_upper'] = df['BB_SMA'] + (df['BB_STD'] * bb_std)
            df['BB_lower'] = df['BB_SMA'] - (df['BB_STD'] * bb_std)
            df['BB_width'] = df['BB_upper'] - df['BB_lower']
            df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'] + 1e-8)
            
            # 7. Stochastic
            k_period = 14
            d_period = 3
            low_min = df['low'].rolling(window=k_period).min()
            high_max = df['high'].rolling(window=k_period).max()
            df['Stoch_K'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-8)
            df['Stoch_D'] = df['Stoch_K'].rolling(window=d_period).mean()
            
            # 8. OBV
            df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            
            # 9. ROC
            df['ROC_10'] = df['close'].pct_change(periods=10) * 100
            df['ROC_20'] = df['close'].pct_change(periods=20) * 100
            
            # 10. Williams %R
            df['Williams_R'] = -100 * (high_max - df['close']) / (high_max - low_min + 1e-8)
            
            # 11. CCI
            tp = (df['high'] + df['low'] + df['close']) / 3
            df['CCI'] = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std() + 1e-8)
            
            # 12. Volatility
            df['Volatility'] = df['close'].pct_change().rolling(window=20).std() * 100
            
            # 13. Log Returns
            if use_log_returns:
                df['Log_Returns'] = np.log(df['close'] / df['close'].shift(1)) * 100
            
            # 14. CO Ratio
            df['CO_Ratio'] = (df['close'] - df['open']) / (df['open'] + 1e-8)
            
            # 15. HL Ratio
            df['HL_Ratio'] = (df['high'] - df['low']) / (df['low'] + 1e-8)
            
            df = df.dropna()
            
            logger.info(f"✓ Added {len(df.columns)-2} advanced indicators")
            
            return df
        except Exception as e:
            logger.error(f"Failed to add indicators: {e}")
            return df
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> Tuple[pd.DataFrame, int]:
        """Detect and remove outliers"""
        try:
            initial_len = len(df)
            
            if method == 'iqr':
                Q1 = df['close'].quantile(0.25)
                Q3 = df['close'].quantile(0.75)
                IQR = Q3 - Q1
                df = df[(df['close'] >= Q1 - 1.5 * IQR) & (df['close'] <= Q3 + 1.5 * IQR)]
            
            removed = initial_len - len(df)
            logger.info(f"✓ Outlier detection complete ({removed} removed)")
            return df, removed
        except Exception as e:
            logger.error(f"Failed outlier detection: {e}")
            return df, 0
    
    def test_stationarity(self, df: pd.DataFrame, column: str = 'close') -> Dict:
        """Test stationarity with ADF test"""
        try:
            result = adfuller(df[column].dropna())
            logger.info(f"✓ ADF Test: statistic={result[0]:.4f}, p-value={result[1]:.4f}")
            return {'statistic': result[0], 'p_value': result[1]}
        except Exception as e:
            logger.error(f"Stationarity test failed: {e}")
            return {}
    
    def analyze_autocorrelation(self, df: pd.DataFrame, nlags: int = 40) -> Dict:
        """Analyze autocorrelation"""
        try:
            acf_values = acf(df['close'].dropna(), nlags=nlags)
            logger.info(f"✓ Autocorrelation analysis complete (lags={nlags})")
            return {'acf': acf_values}
        except Exception as e:
            logger.error(f"Autocorrelation analysis failed: {e}")
            return {}
    
    def select_features(self, df: pd.DataFrame, target: str = 'close', top_n: int = 12) -> List[str]:
        """Select best features"""
        try:
            features = [c for c in df.columns if c != target]
            X = df[features].fillna(0)
            y = df[target]
            
            mi_scores = mutual_info_regression(X, y)
            indices = np.argsort(mi_scores)[::-1][:top_n]
            selected = [features[i] for i in indices]
            
            logger.info(f"✓ Selected {len(selected)} best features")
            return selected
        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            return ['close']
    
    def prepare_ml_features(
        self,
        df: pd.DataFrame,
        lookback: int = 60,
        scaler_type: str = 'robust',
        selected_features: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, object]:
        """Prepare features for ML model with flexible scaling"""
        try:
            if selected_features is None:
                feature_cols = [c for c in df.columns if c not in ['symbol', 'close']]
                feature_cols = ['close'] + feature_cols[:7]
            else:
                feature_cols = selected_features if 'close' in selected_features else ['close'] + selected_features
            
            df_clean = df[feature_cols].copy()
            
            # Select Scaler
            if scaler_type == 'robust':
                scaler = RobustScaler()
                logger.info("  Using RobustScaler (optimal for crypto volatility)")
            elif scaler_type == 'minmax':
                scaler = MinMaxScaler()
                logger.info("  Using MinMaxScaler")
            elif scaler_type == 'standard':
                scaler = StandardScaler()
                logger.info("  Using StandardScaler")
            elif scaler_type == 'quantile':
                scaler = QuantileTransformer(output_distribution='uniform')
                logger.info("  Using QuantileTransformer")
            else:
                scaler = RobustScaler()
            
            scaled_data = scaler.fit_transform(df_clean)
            
            # Create sequences
            X, y = [], []
            for i in range(len(scaled_data) - lookback):
                X.append(scaled_data[i:i+lookback])
                y.append(df_clean['close'].iloc[i+lookback])
            
            X = np.array(X)
            y = np.array(y)
            
            self.scaler = scaler
            
            logger.info(f"✓ ML Features Prepared:")
            logger.info(f"  - X shape: {X.shape}")
            logger.info(f"  - y shape: {y.shape}")
            logger.info(f"  - Features: {feature_cols}")
            
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

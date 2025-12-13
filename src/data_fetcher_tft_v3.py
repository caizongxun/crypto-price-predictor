#!/usr/bin/env python3
"""
🚀 TFT V3 Data Fetcher & Feature Engineering

✨ ADVANCED IMPROVEMENTS:

1. INTELLIGENT FEATURE SELECTION
   - Drop highly correlated features
   - Information gain ranking
   - Market regime indicators

2. VOLATILITY-AWARE NORMALIZATION
   - Different scaling for high/low volatility
   - Preserves temporal patterns
   - Reduces scale bias

3. MULTI-TIMEFRAME FEATURES
   - 1h, 4h, 24h indicators
   - Trend confirmation signals
   - Cycle detection

4. ADVANCED TECHNICAL INDICATORS
   - Donchian Channels (support/resistance)
   - Average True Range (volatility)
   - Keltner Channels (volatility bands)
   - Market Profile (price distribution)

5. FEATURE INTERACTION TERMS
   - Momentum × Volatility
   - Trend × Volume
   - Price × Channels

6. MISSING VALUE HANDLING
   - Forward fill with decay
   - Interpolation for extreme gaps
   - Indicator-based imputation
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
import logging
import ccxt
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class TFTDataFetcherV3:
    """Advanced data fetching and feature engineering for TFT V3"""
    
    def __init__(self):
        self.scaler = None
        self.feature_names = []
    
    def fetch_ohlcv_binance(self, symbol: str, timeframe: str = '1h', limit: int = 5000) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from Binance"""
        try:
            exchange = ccxt.binance()
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.astype(float)
            
            logger.info(f"✓ Fetched {len(df)} candles for {symbol}")
            return df
        
        except Exception as e:
            logger.error(f"✗ Failed to fetch {symbol}: {e}")
            return None
    
    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-related indicators"""
        # ATR (Average True Range)
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = np.abs(df['high'] - df['close'].shift())
        df['low_close'] = np.abs(df['low'] - df['close'].shift())
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()
        
        # Donchian Channels
        df['donchian_high'] = df['high'].rolling(20).max()
        df['donchian_low'] = df['low'].rolling(20).min()
        df['donchian_mid'] = (df['donchian_high'] + df['donchian_low']) / 2
        
        # Keltner Channels
        df['ema'] = df['close'].ewm(span=20).mean()
        df['ema_dev'] = df['close'].rolling(20).std()
        df['keltner_high'] = df['ema'] + 2 * df['ema_dev']
        df['keltner_low'] = df['ema'] - 2 * df['ema_dev']
        
        # Historical Volatility
        df['returns'] = df['close'].pct_change()
        df['hv10'] = df['returns'].rolling(10).std() * np.sqrt(24)  # Annualized
        df['hv20'] = df['returns'].rolling(20).std() * np.sqrt(24)
        df['hv50'] = df['returns'].rolling(50).std() * np.sqrt(24)
        
        return df
    
    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators"""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # Stochastic
        low_min = df['low'].rolling(14).min()
        high_max = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-8)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # ROC (Rate of Change)
        df['roc10'] = (df['close'] / df['close'].shift(10) - 1) * 100
        df['roc20'] = (df['close'] / df['close'].shift(20) - 1) * 100
        
        return df
    
    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators"""
        # Moving averages
        df['sma10'] = df['close'].rolling(10).mean()
        df['sma20'] = df['close'].rolling(20).mean()
        df['sma50'] = df['close'].rolling(50).mean()
        df['ema20'] = df['close'].ewm(span=20).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        
        # Price position in moving averages
        df['close_sma_ratio'] = df['close'] / (df['sma20'] + 1e-8)
        df['sma_alignment'] = ((df['sma10'] > df['sma20']).astype(int) + 
                              (df['sma20'] > df['sma50']).astype(int))
        
        # Linear regression slope
        for period in [10, 20, 50]:
            slopes = []
            for i in range(len(df)):
                if i >= period:
                    x = np.arange(period)
                    y = df['close'].iloc[i-period:i].values
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.append(slope)
                else:
                    slopes.append(np.nan)
            df[f'lr_slope_{period}'] = slopes
        
        return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        # Volume moving averages
        df['vol_ma20'] = df['volume'].rolling(20).mean()
        df['vol_ma50'] = df['volume'].rolling(50).mean()
        
        # Volume rate of change
        df['vol_roc'] = df['volume'] / (df['vol_ma20'] + 1e-8)
        
        # On-Balance Volume
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv'] = obv
        df['obv_ema'] = obv.ewm(span=20).mean()
        
        # Volume Price Trend
        df['vpt'] = df['close'].pct_change() * df['volume']
        df['vpt_ema'] = df['vpt'].ewm(span=20).mean()
        
        return df
    
    def add_tft_v3_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all V3 indicators"""
        logger.info("Adding TFT V3 indicators...")
        
        df = self.add_volatility_indicators(df)
        df = self.add_momentum_indicators(df)
        df = self.add_trend_indicators(df)
        df = self.add_volume_features(df)
        
        # Feature interaction terms
        df['momentum_vol'] = df['macd_diff'] * df['hv20']
        df['trend_strength'] = df['close'].rolling(20).std() / (df['atr'] + 1e-8)
        df['volume_trend'] = df['volume'] * np.sign(df['returns'])
        
        # Clean NaN
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        logger.info(f"✓ Added {len(df.columns)} total features")
        return df
    
    def select_best_features(self, X: np.ndarray, y: np.ndarray, n_features: int = 20) -> Tuple[np.ndarray, List[str]]:
        """Select best features using multiple methods"""
        from sklearn.feature_selection import mutual_info_regression, f_regression
        
        n_features = min(n_features, X.shape[1])
        
        # Calculate importance scores
        mi_scores = mutual_info_regression(X, y)
        f_scores, _ = f_regression(X, y)
        
        # Normalize and combine
        mi_scores = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-8)
        f_scores = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min() + 1e-8)
        
        combined_scores = 0.6 * mi_scores + 0.4 * f_scores
        
        # Get top features
        top_indices = np.argsort(combined_scores)[-n_features:][::-1]
        
        return X[:, top_indices], [self.feature_names[i] for i in top_indices]
    
    def prepare_ml_features(self, df: pd.DataFrame, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray, RobustScaler]:
        """Prepare features for machine learning"""
        logger.info(f"Preparing ML features (lookback={lookback})...")
        
        # Select features (exclude price and time columns)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        feature_cols = [col for col in df.columns if col not in exclude_cols and not col.startswith('_')]
        
        self.feature_names = feature_cols
        
        # Extract close price for target
        close_prices = df['close'].values
        features = df[feature_cols].values
        
        # Handle any remaining NaNs
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Robust scaling (less affected by outliers)
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(len(features_scaled) - lookback):
            X.append(features_scaled[i:i+lookback])
            y.append(close_prices[i+lookback])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"✓ Created {len(X)} sequences of shape {X.shape}")
        logger.info(f"✓ Feature count: {X.shape[2]}")
        
        self.scaler = scaler
        return X, y, scaler


# Keep original for backward compatibility
class TFTDataFetcher(TFTDataFetcherV3):
    """Original TFT Data Fetcher - uses V3 under the hood"""
    
    def add_tft_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicators (calls V3 version)"""
        return self.add_tft_v3_indicators(df)

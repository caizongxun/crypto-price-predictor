"""Real-time cryptocurrency data fetcher from multiple sources."""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
from dotenv import load_dotenv
import os

load_dotenv()
logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches real-time and historical cryptocurrency data."""
    
    def __init__(self, config: Dict = None):
        """Initialize data fetcher with configuration.
        
        Args:
            config: Configuration dictionary with API settings
        """
        self.config = config or {}
        self.binance_api_key = os.getenv('BINANCE_API_KEY')
        self.binance_api_secret = os.getenv('BINANCE_API_SECRET')
        self.coingecko_api_key = os.getenv('COINGECKO_API_KEY')
        
        # Initialize exchanges
        self.binance = None
        self.kraken = None
        self._init_exchanges()
        
    def _init_exchanges(self):
        """Initialize CCXT exchange connections."""
        try:
            if self.binance_api_key and self.binance_api_secret:
                self.binance = ccxt.binance({
                    'apiKey': self.binance_api_key,
                    'secret': self.binance_api_secret,
                    'enableRateLimit': True,
                })
                logger.info("Binance API initialized")
            else:
                self.binance = ccxt.binance({'enableRateLimit': True})
                logger.warning("Binance API keys not provided, using public access")
                
            self.kraken = ccxt.kraken({'enableRateLimit': True})
            logger.info("Kraken API initialized")
        except Exception as e:
            logger.error(f"Failed to initialize exchanges: {e}")
    
    def fetch_ohlcv_binance(self, symbol: str, timeframe: str = '1d', 
                           limit: int = 500) -> pd.DataFrame:
        """Fetch OHLCV data from Binance.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candlestick timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
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
    
    def fetch_ohlcv_yfinance(self, symbol: str, period: str = '1y',
                            interval: str = '1d') -> pd.DataFrame:
        """Fetch OHLCV data using yfinance.
        
        Args:
            symbol: Ticker symbol (e.g., 'BTC-USD')
            period: Period to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y)
            interval: Interval (1m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            df.columns = df.columns.str.lower()
            df['symbol'] = symbol
            
            logger.info(f"Fetched {len(df)} candles for {symbol} from yfinance")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} from yfinance: {e}")
            return None
    
    def fetch_multiple_cryptocurrencies(self, symbols: List[Dict],
                                        timeframe: str = '1d',
                                        limit: int = 500) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple cryptocurrencies.
        
        Args:
            symbols: List of dicts with symbol and trading_pair info
            timeframe: Candlestick timeframe
            limit: Number of candles to fetch
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data = {}
        
        for crypto in symbols:
            try:
                symbol = crypto['symbol']
                trading_pair = crypto.get('trading_pair', f"{symbol}/USDT")
                
                df = self.fetch_ohlcv_binance(trading_pair, timeframe, limit)
                if df is not None and not df.empty:
                    data[symbol] = df
                else:
                    logger.warning(f"No data fetched for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
        
        logger.info(f"Successfully fetched data for {len(data)} cryptocurrencies")
        return data
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to OHLCV data (simplified for better generalization).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators (20 total features)
        """
        try:
            df = df.copy()
            
            # ========================================
            # CORE TECHNICAL INDICATORS (12 features)
            # ========================================
            
            # Moving Averages
            df['SMA_10'] = df['close'].rolling(window=10).mean()
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            df['EMA_12'] = df['close'].ewm(span=12).mean()
            
            # RSI (Relative Strength Index)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD (Moving Average Convergence Divergence)
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['MACD'] = ema_12 - ema_26
            df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
            
            # Bollinger Bands (Upper and Lower)
            bb_mid = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['BB_upper'] = bb_mid + (bb_std * 2)
            df['BB_lower'] = bb_mid - (bb_std * 2)
            
            # Volume indicators
            df['Volume_SMA'] = df['volume'].rolling(window=20).mean()
            df['Volume_ratio'] = df['volume'] / df['Volume_SMA']
            
            # Price momentum
            df['Daily_return'] = df['close'].pct_change()
            
            # ========================================
            # SIMPLIFIED LEADING INDICATORS (8 features)
            # Removed: Williams%R, MFI, CCI (too complex for current data size)
            # ========================================
            
            # 1. Rate of Change (ROC) - Pure momentum
            df['ROC_1'] = df['close'].pct_change(periods=1)  # 1-period momentum
            df['ROC_5'] = df['close'].pct_change(periods=5)  # 5-period momentum
            
            # 2. Price Acceleration - Rate of change of momentum
            df['Price_accel'] = df['ROC_1'].diff()
            
            # 3. Volume Acceleration - Leading volume signal
            df['Volume_accel'] = df['Volume_ratio'].diff()
            
            # 4. Stochastic Oscillator - Sensitive momentum indicator
            def stochastic(high, low, close, period=14):
                lowest_low = low.rolling(window=period).min()
                highest_high = high.rolling(window=period).max()
                k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-8))
                d_percent = k_percent.rolling(window=3).mean()
                return k_percent, d_percent
            
            df['Stoch_K'], df['Stoch_D'] = stochastic(df['high'], df['low'], df['close'])
            
            # 5. Volatility Indicator - Market regime detection
            df['Volatility'] = df['close'].rolling(window=10).std() / df['close'].rolling(window=10).mean()
            
            logger.info(f"Added 20 technical indicators (12 core + 8 leading) for {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Failed to add technical indicators: {e}")
            return df
    
    def prepare_ml_features(self, df: pd.DataFrame, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for machine learning models.
        
        Args:
            df: DataFrame with OHLCV and technical indicators
            lookback: Number of time steps to use for lookback
            
        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        try:
            # Select relevant features - SIMPLIFIED to 20 features
            feature_cols = [
                'open', 'high', 'low', 'close', 'volume',
                'SMA_10', 'SMA_20', 'EMA_12',
                'RSI', 'MACD', 'MACD_signal',
                'BB_upper', 'BB_lower',
                'Volume_ratio', 'Daily_return',
                'ROC_1', 'ROC_5',
                'Price_accel', 'Volume_accel',
                'Stoch_K', 'Stoch_D', 'Volatility'
            ]
            
            # Remove NaN values
            df_clean = df[feature_cols].dropna()
            
            # Normalize features
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
            
            logger.info(f"Prepared {len(X)} sequences with {len(feature_cols)} features (lookback={lookback})")
            logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
            return X, y, scaler
        except Exception as e:
            logger.error(f"Failed to prepare ML features: {e}")
            return None, None, None
    
    def get_real_time_price(self, symbol: str) -> Optional[float]:
        """Get real-time price for a cryptocurrency.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            
        Returns:
            Current price as float
        """
        try:
            ticker = self.binance.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Failed to fetch real-time price for {symbol}: {e}")
            return None
    
    def get_market_data(self, symbol: str) -> Optional[Dict]:
        """Get comprehensive market data for a symbol.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Dictionary with market data
        """
        try:
            ticker = self.binance.fetch_ticker(symbol)
            return {
                'last': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'high': ticker['high'],
                'low': ticker['low'],
                'volume': ticker['quoteVolume'],
                'timestamp': ticker['timestamp']
            }
        except Exception as e:
            logger.error(f"Failed to fetch market data for {symbol}: {e}")
            return None

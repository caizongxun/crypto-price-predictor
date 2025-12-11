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
            ohlcv = self.binance.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df['symbol'] = symbol
            
            logger.info(f"Fetched {len(df)} candles for {symbol}")
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
        """Add common technical indicators to OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        try:
            df = df.copy()
            
            # Simple Moving Averages
            df['SMA_10'] = df['close'].rolling(window=10).mean()
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            df['SMA_50'] = df['close'].rolling(window=50).mean()
            
            # Exponential Moving Averages
            df['EMA_12'] = df['close'].ewm(span=12).mean()
            df['EMA_26'] = df['close'].ewm(span=26).mean()
            
            # RSI (Relative Strength Index)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD (Moving Average Convergence Divergence)
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_diff'] = df['MACD'] - df['MACD_signal']
            
            # Bollinger Bands
            df['BB_mid'] = df['close'].rolling(window=20).mean()
            df['BB_std'] = df['close'].rolling(window=20).std()
            df['BB_upper'] = df['BB_mid'] + (df['BB_std'] * 2)
            df['BB_lower'] = df['BB_mid'] - (df['BB_std'] * 2)
            
            # ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR'] = tr.rolling(window=14).mean()
            
            # Volume indicators
            df['Volume_SMA'] = df['volume'].rolling(window=20).mean()
            df['Volume_ratio'] = df['volume'] / df['Volume_SMA']
            
            # Price change
            df['Daily_return'] = df['close'].pct_change()
            df['Price_momentum'] = df['close'].pct_change(periods=5)
            
            logger.info(f"Added technical indicators for {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Failed to add technical indicators: {e}")
            return df
    
    def prepare_ml_features(self, df: pd.DataFrame, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for machine learning models.
        
        Args:
            df: DataFrame with OHLCV and technical indicators
            lookback: Number of days to use for lookback
            
        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        try:
            # Select relevant features
            feature_cols = [
                'open', 'high', 'low', 'close', 'volume',
                'SMA_10', 'SMA_20', 'SMA_50',
                'RSI', 'MACD', 'MACD_diff',
                'BB_upper', 'BB_lower', 'ATR',
                'Volume_ratio', 'Daily_return', 'Price_momentum'
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
            
            logger.info(f"Prepared {len(X)} sequences for ML training")
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

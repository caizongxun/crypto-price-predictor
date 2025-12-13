#!/usr/bin/env python
"""Training script using public APIs (no API keys required).

Usage:
    python local_train_public_api.py

Features:
    - Uses CCXT for public market data (no authentication needed)
    - Falls back to yfinance if CCXT unavailable
    - Trains LSTM models on 60 historical data points
    - Predicts 5 future candles
    - Saves models to models/saved_models/
"""

import os
import sys
import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from src.model_trainer import ModelTrainer
from src.signal_generator import SignalGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PublicDataTrainer:
    """Trainer using public APIs (no authentication)."""
    
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️  Using device: {self.device}")
        
        self.signal_gen = SignalGenerator()
        self.output_dir = Path('models/saved_models')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_data_ccxt(self, symbol: str, limit: int = 500) -> pd.DataFrame:
        """Fetch data using CCXT (supports 100+ exchanges)."""
        try:
            import ccxt
            exchange = ccxt.binance({'enableRateLimit': True})
            
            ohlcv = exchange.fetch_ohlcv(f"{symbol}/USDT", timeframe='1h', limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.debug(f"CCXT fetch failed for {symbol}: {e}")
            return None

    def fetch_data_yfinance(self, symbol: str, limit: int = 500) -> pd.DataFrame:
        """Fallback: Fetch data using yfinance."""
        try:
            import yfinance as yf
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=limit)
            
            ticker = f"{symbol}USD=X" if symbol != "BTC" else "BTC-USD"
            df = yf.download(ticker, start=start_date, end=end_date, interval='1h', progress=False)
            
            if df is None or len(df) == 0:
                return None
                
            df['timestamp'] = df.index
            df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)
        except Exception as e:
            logger.debug(f"yfinance fetch failed for {symbol}: {e}")
            return None

    def fetch_data(self, symbol: str) -> pd.DataFrame:
        """Fetch data with automatic fallback."""
        logger.info(f"📥 Fetching data for {symbol}...")
        
        # Try CCXT first (fastest)
        df = self.fetch_data_ccxt(symbol)
        if df is not None and len(df) > 100:
            logger.info(f"✅ Fetched {len(df)} candles via CCXT")
            return df
        
        # Fallback to yfinance
        logger.info(f"⚠️  CCXT failed, trying yfinance...")
        df = self.fetch_data_yfinance(symbol)
        if df is not None and len(df) > 100:
            logger.info(f"✅ Fetched {len(df)} candles via yfinance")
            return df
        
        logger.error(f"❌ Could not fetch data for {symbol} from any source")
        return None

    def prepare_sequences(self, features: np.ndarray, lookback: int = 60, horizon: int = 5):
        """Prepare training sequences."""
        X, y = [], []
        close_idx = 3  # Close price column index in features
        
        for i in range(len(features) - lookback - horizon):
            X.append(features[i:i+lookback])
            y.append(features[i+lookback:i+lookback+horizon, close_idx])
        
        return np.array(X), np.array(y) if X else (None, None)

    def train_symbol(self, symbol: str) -> bool:
        """Train model for a single symbol."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {symbol}...")
        
        # Fetch data
        df = self.fetch_data(symbol)
        if df is None: 
            return False
        
        # Prepare features
        try:
            features = self.signal_gen.prepare_features(df['close'].values)
            if features is None or len(features) < 100:
                logger.error(f"❌ Insufficient features for {symbol}")
                return False
                
            X, y = self.prepare_sequences(features)
            if X is None or len(X) < 10:
                logger.error(f"❌ Insufficient sequences for {symbol}")
                return False
            
            # Convert to tensors
            X_train = torch.from_numpy(X).float().to(self.device)
            y_train = torch.from_numpy(y).float().to(self.device)
            
            logger.info(f"📊 Training set: {len(X)} sequences")
            
            # Train model
            trainer = ModelTrainer(model_type='lstm', config={
                'hidden_size': 256,
                'num_layers': 3,
                'dropout': 0.3,
                'learning_rate': 0.001,
                'epochs': 50,
                'batch_size': 32
            })
            
            # Initialize model
            trainer.create_model(input_size=17, output_size=5)
            trainer.model.to(self.device)
            
            # Train
            logger.info(f"🚀 Training model...")
            loss = trainer.train(X_train, y_train)
            logger.info(f"✅ Training complete. Final loss: {loss:.6f}")
            
            # Save model
            model_path = self.output_dir / f"{symbol}_lstm_model.pth"
            trainer.save_model(str(model_path))
            logger.info(f"💾 Model saved: {model_path}")
            logger.info(f"📦 File size: {model_path.stat().st_size / (1024**2):.2f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Training failed for {symbol}: {e}", exc_info=True)
            return False

    def train_all(self, symbols=None):
        """Train all symbols."""
        if symbols is None:
            symbols = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'MATIC', 'AVAX', 'LINK']
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🚀 STARTING TRAINING (PUBLIC API MODE)")
        logger.info(f"{'='*70}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Models: {len(symbols)} cryptocurrencies\n")
        
        results = {'success': 0, 'failed': 0, 'skipped': 0}
        
        for symbol in symbols:
            success = self.train_symbol(symbol)
            if success:
                results['success'] += 1
            else:
                results['failed'] += 1
        
        # Summary
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ TRAINING COMPLETE")
        logger.info(f"  Success: {results['success']}")
        logger.info(f"  Failed:  {results['failed']}")
        logger.info(f"{'='*70}\n")


def main():
    try:
        trainer = PublicDataTrainer()
        trainer.train_all()
        return True
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

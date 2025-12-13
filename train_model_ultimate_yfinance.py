#!/usr/bin/env python3
"""
Ultimate Cryptocurrency Price Prediction Model Trainer (YFinance Version)
Supports long-duration training without Binance API key

Usage:
  python train_model_ultimate_yfinance.py --symbol BTC --epochs 300
  python train_model_ultimate_yfinance.py --symbol ETH --epochs 300 --batch-size 16
  python train_model_ultimate_yfinance.py --symbol SOL --epochs 100 --device cuda
"""

import argparse
import logging
from pathlib import Path
import sys
import numpy as np
import torch
import yfinance as yf
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging, create_directories
from src.model_trainer_ultimate import UltimateModelTrainer
from sklearn.preprocessing import StandardScaler

load_dotenv()
setup_logging(log_level='INFO', log_file='logs/training_ultimate_yfinance.log')
logger = logging.getLogger(__name__)


def fetch_crypto_data_yfinance(symbol: str, period: str = '1y') -> dict:
    """
    Fetch cryptocurrency data from yfinance.
    
    Args:
        symbol: Cryptocurrency symbol (BTC, ETH, SOL, etc.)
        period: Period to fetch (1y, 2y, 5y, etc.)
        
    Returns:
        Dictionary with OHLCV data
    """
    try:
        # Convert symbol to yfinance format (e.g., BTC -> BTC-USD)
        yf_symbol = f"{symbol}-USD"
        logger.info(f"Fetching {yf_symbol} data from yfinance...")
        
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period, interval='1d')
        
        if df.empty:
            logger.error(f"No data fetched for {symbol}")
            return None
        
        # Rename columns to match expected format
        df.columns = df.columns.str.lower()
        df = df.rename(columns={'adj close': 'close'})
        
        logger.info(f"Successfully fetched {len(df)} days of data for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to fetch data for {symbol}: {e}")
        return None


def add_technical_indicators(df):
    """
    Add technical indicators to OHLCV data.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with technical indicators
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
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_diff'] = df['MACD'] - df['MACD_signal']
        
        # Bollinger Bands
        df['BB_mid'] = df['close'].rolling(window=20).mean()
        df['BB_std'] = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_mid'] + (df['BB_std'] * 2)
        df['BB_lower'] = df['BB_mid'] - (df['BB_std'] * 2)
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = np.concatenate([high_low.values.reshape(-1, 1), 
                            high_close.values.reshape(-1, 1),
                            low_close.values.reshape(-1, 1)], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean() if len(tr) > 0 else 0
        
        # Volume indicators
        df['Volume_SMA'] = df['volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['volume'] / (df['Volume_SMA'] + 1e-8)
        
        # Price changes
        df['Daily_return'] = df['close'].pct_change()
        df['Price_momentum'] = df['close'].pct_change(periods=5)
        
        logger.info(f"Added technical indicators for {len(df)} rows")
        return df
        
    except Exception as e:
        logger.error(f"Failed to add technical indicators: {e}")
        return df


def prepare_ml_features(df, lookback: int = 60):
    """
    Prepare features and labels for ML models.
    
    Args:
        df: DataFrame with OHLCV and technical indicators
        lookback: Number of days for lookback
        
    Returns:
        Tuple of (X, y) as numpy arrays
    """
    try:
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'SMA_10', 'SMA_20', 'SMA_50',
            'RSI', 'MACD', 'MACD_diff',
            'BB_upper', 'BB_lower', 'ATR',
            'Volume_ratio', 'Daily_return', 'Price_momentum'
        ]
        
        df_clean = df[feature_cols].dropna()
        
        if len(df_clean) < lookback + 1:
            logger.error(f"Not enough data: {len(df_clean)} < {lookback + 1}")
            return None, None
        
        # Normalize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_clean)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - lookback):
            X.append(scaled_data[i:i+lookback])
            y.append(df_clean['close'].iloc[i+lookback])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Prepared {len(X)} sequences for ML training")
        return X, y
        
    except Exception as e:
        logger.error(f"Failed to prepare ML features: {e}")
        return None, None


def train_model_ultimate_yfinance(
    symbol: str,
    lookback: int = 60,
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 0.00005,
    device: str = 'auto'
):
    """
    Train ultimate cryptocurrency prediction model using yfinance data.
    
    Args:
        symbol: Cryptocurrency symbol (BTC, ETH, SOL, etc.)
        lookback: Lookback period in days
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Training device (auto, cuda, cpu)
    """
    try:
        # Create directories
        create_directories()
        
        # Determine device
        if device == 'auto':
            selected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            selected_device = device
        
        logger.info("="*80)
        logger.info("ULTIMATE CRYPTOCURRENCY PRICE PREDICTION MODEL TRAINING (YFinance)")
        logger.info("="*80)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Data Source: YFinance (no API key required)")
        logger.info(f"Lookback Period: {lookback} days")
        logger.info(f"Training Configuration:")
        logger.info(f"  - Epochs: {epochs}")
        logger.info(f"  - Batch Size: {batch_size}")
        logger.info(f"  - Learning Rate: {learning_rate}")
        logger.info(f"  - Device: {selected_device.upper()}")
        logger.info("="*80)
        
        # Step 1: Fetch data from yfinance
        logger.info("\n[Step 1/5] Fetching historical data from yfinance...")
        df = fetch_crypto_data_yfinance(symbol, period='2y')
        
        if df is None or df.empty:
            logger.error(f"Failed to fetch data for {symbol}")
            return False
        
        logger.info(f"[OK] Fetched {len(df)} days of data")
        
        # Step 2: Add technical indicators
        logger.info("\n[Step 2/5] Adding technical indicators...")
        df = add_technical_indicators(df)
        logger.info(f"[OK] Added technical indicators")
        
        # Step 3: Prepare ML features
        logger.info("\n[Step 3/5] Preparing ML features...")
        X, y = prepare_ml_features(df, lookback=lookback)
        
        if X is None or y is None:
            logger.error("Failed to prepare features")
            return False
        
        logger.info(f"[OK] Prepared {X.shape[0]} sequences")
        logger.info(f"    - X shape: {X.shape}")
        logger.info(f"    - y shape: {y.shape}")
        
        # Step 4: Train model
        logger.info("\n[Step 4/5] Training ultimate ensemble model...")
        trainer = UltimateModelTrainer(device=selected_device)
        
        model, history = trainer.train_ultimate_ensemble(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            accumulation_steps=2
        )
        
        # Step 5: Save model
        logger.info("\n[Step 5/5] Saving trained model...")
        model_dir = Path('models/saved_models')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        symbol_model_path = model_dir / f"{symbol}_ultimate_model.pth"
        torch.save(model.state_dict(), symbol_model_path)
        logger.info(f"[OK] Model saved to {symbol_model_path}")
        
        # Training summary
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"Final Training Loss: {history['train_loss'][-1]:.6f}")
        logger.info(f"Final Validation Loss: {history['val_loss'][-1]:.6f}")
        logger.info(f"Best Validation Loss: {min(history['val_loss']):.6f}")
        logger.info(f"Final Overfitting Ratio: {history['overfitting_ratio'][-1]:.3f}")
        logger.info(f"Total Epochs Trained: {len(history['train_loss'])}")
        logger.info(f"Model Location: {symbol_model_path}")
        logger.info("="*80 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train ultimate cryptocurrency prediction model using yfinance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_model_ultimate_yfinance.py --symbol BTC --epochs 100
  python train_model_ultimate_yfinance.py --symbol ETH --epochs 300 --batch-size 16
  python train_model_ultimate_yfinance.py --symbol SOL --epochs 50 --device cuda
        """
    )
    
    parser.add_argument('--symbol', default='BTC', help='Cryptocurrency symbol (default: BTC)')
    parser.add_argument('--lookback', type=int, default=60, help='Lookback period in days (default: 60)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--learning-rate', type=float, default=0.00005, help='Learning rate (default: 0.00005)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='Training device (default: auto)')
    
    args = parser.parse_args()
    
    success = train_model_ultimate_yfinance(
        symbol=args.symbol,
        lookback=args.lookback,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    exit(0 if success else 1)

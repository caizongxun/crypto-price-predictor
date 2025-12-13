#!/usr/bin/env python3
"""
Train Simplified LSTM Model (Option A)

🎯 Goal: Stability and interpretability over complexity

Features:
- Single LSTM (hidden_size=128, num_layers=2)
- 10 core indicators only
- Dropout 0.4 (balanced regularization)
- MAE loss (simple, direct)
- 150+ epochs for convergence

Expected MAE: 0.15-0.20

Usage:
  python train_simple_lstm.py --symbol SOL --epochs 150 --device cuda
  python train_simple_lstm.py --symbol BTC --epochs 150 --device cuda
"""

import argparse
import logging
from pathlib import Path
import sys
import torch
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging, create_directories
from src.data_fetcher_simple import SimpleDataFetcher
from src.model_trainer_simple import SimpleLSTMTrainer

load_dotenv()
setup_logging(log_level='INFO', log_file='logs/training_simple.log')
logger = logging.getLogger(__name__)


def train_simple_lstm(
    symbol: str,
    trading_pair: str,
    lookback: int = 60,
    epochs: int = 150,
    batch_size: int = 32,
    learning_rate: float = 0.0001,
    device: str = 'auto'
):
    """
    Train simplified LSTM model with 10 core features.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., SOL)
        trading_pair: Trading pair (e.g., SOL/USDT)
        lookback: Lookback period in hours
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        device: Training device (auto, cuda, or cpu)
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
        logger.info("🚀 SIMPLIFIED LSTM MODEL TRAINING (Option A)")
        logger.info("="*80)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Trading Pair: {trading_pair}")
        logger.info(f"Lookback: {lookback} hours")
        logger.info(f"\n📋 Configuration:")
        logger.info(f"  - Architecture: Simple LSTM (hidden=128, layers=2)")
        logger.info(f"  - Features: 10 core indicators")
        logger.info(f"  - Dropout: 0.4 (balanced)")
        logger.info(f"  - Epochs: {epochs}")
        logger.info(f"  - Batch Size: {batch_size}")
        logger.info(f"  - Learning Rate: {learning_rate}")
        logger.info(f"  - Loss: MAE (L1Loss)")
        logger.info(f"  - Device: {selected_device.upper()}")
        logger.info("="*80)
        
        # Step 1: Fetch data
        logger.info("\n[Step 1/5] Fetching historical data (10,000 candles)...")
        fetcher = SimpleDataFetcher()
        df = fetcher.fetch_ohlcv_binance(trading_pair, timeframe='1h', limit=10000)
        
        if df is None or df.empty:
            logger.error(f"Failed to fetch data for {symbol}")
            return False
        
        logger.info(f"[✓] Fetched {len(df)} candles for {trading_pair}")
        logger.info(f"    - Date range: {df.index[0]} to {df.index[-1]}")
        
        # Step 2: Add simple indicators
        logger.info("\n[Step 2/5] Adding 10 core technical indicators...")
        df = fetcher.add_simple_indicators(df)
        logger.info(f"[✓] Added indicators for {len(df)} rows")
        logger.info(f"    - Features: open, high, low, close, volume (OHLCV)")
        logger.info(f"    -           SMA_10, RSI, MACD, Volume_ratio")
        
        # Step 3: Prepare ML features
        logger.info("\n[Step 3/5] Preparing ML features...")
        X, y, scaler = fetcher.prepare_ml_features(df, lookback=lookback)
        
        if X is None or y is None:
            logger.error("Failed to prepare features")
            return False
        
        logger.info(f"[✓] Prepared {X.shape[0]} sequences")
        logger.info(f"    - X shape: {X.shape} (samples, lookback, features)")
        logger.info(f"    - y shape: {y.shape}")
        
        # Step 4: Train model
        logger.info("\n[Step 4/5] Training simplified LSTM model...")
        trainer = SimpleLSTMTrainer(device=selected_device)
        
        model, history = trainer.train_simple_lstm(
            X, y,
            symbol=symbol,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_size=128,
            dropout=0.4
        )
        
        logger.info("\n[Step 5/5] Training completed!")
        logger.info(f"[✓] Model saved to: models/saved_models/{symbol}_simple_model.pth")
        
        logger.info("\n" + "="*80)
        logger.info("✅ TRAINING SUMMARY")
        logger.info("="*80)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Best Validation Loss: {min(history['val_loss']):.6f}")
        logger.info(f"Final Overfitting Ratio: {history['overfitting_ratio'][-1]:.3f}")
        logger.info(f"Total Epochs: {len(history['train_loss'])}")
        logger.info(f"\n🎯 Expected Performance:")
        logger.info(f"  - MAE: 0.15-0.20 (vs old 0.33)")
        logger.info(f"  - Model Size: ~150K parameters (vs Ensemble 8.5M)")
        logger.info(f"  - Inference Speed: 100x faster")
        logger.info(f"  - Training Time: ~5-10 minutes")
        logger.info(f"\n📊 Next Steps:")
        logger.info(f"  1. Visualize: python visualize_simple_lstm.py --symbol {symbol}")
        logger.info(f"  2. Compare: Check if MAE < 0.25 (30% improvement)")
        logger.info(f"  3. Deploy: If good, use this model in production")
        logger.info("="*80 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train simplified LSTM model with 10 core features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard training
  python train_simple_lstm.py --symbol SOL --epochs 150 --device cuda
  
  # Different symbol
  python train_simple_lstm.py --symbol BTC --epochs 150 --device cuda
  
  # CPU training (slower)
  python train_simple_lstm.py --symbol SOL --epochs 150 --device cpu
  
Key Differences (Option A vs Ultimate):
  ✓ Simpler: 1 model vs 3 models
  ✓ Fewer features: 10 vs 20-40
  ✓ Smaller: 150K params vs 8.5M
  ✓ Faster: Minutes vs tens of minutes
  ✓ Clearer: Easy to debug and understand
        """
    )
    
    parser.add_argument('--symbol', default='SOL', help='Cryptocurrency symbol (default: SOL)')
    parser.add_argument('--trading-pair', default=None, help='Trading pair (default: {SYMBOL}/USDT)')
    parser.add_argument('--lookback', type=int, default=60, help='Lookback period in hours (default: 60)')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs (default: 150)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate (default: 0.0001)')
    parser.add_argument('--device', default='cuda', choices=['auto', 'cuda', 'cpu'], help='Training device (default: cuda)')
    
    args = parser.parse_args()
    trading_pair = args.trading_pair or f"{args.symbol}/USDT"
    
    success = train_simple_lstm(
        symbol=args.symbol,
        trading_pair=trading_pair,
        lookback=args.lookback,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    exit(0 if success else 1)

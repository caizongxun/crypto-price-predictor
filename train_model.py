#!/usr/bin/env python3
"""
Optimized Cryptocurrency Price Prediction Model Trainer
Supports Ensemble models with attention mechanisms and mixed precision training
"""

import argparse
import logging
from pathlib import Path
import sys
import numpy as np
import torch
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging, create_directories
from src.data_fetcher import DataFetcher
from src.model_trainer_optimized import OptimizedModelTrainer

load_dotenv()
setup_logging(log_level='INFO', log_file='logs/training_optimized.log')
logger = logging.getLogger(__name__)


def train_model_optimized(
    symbol: str,
    trading_pair: str,
    model_type: str = 'ensemble',
    lookback: int = 60,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    device: str = 'auto'
):
    """
    Train optimized cryptocurrency prediction model with ensemble approach.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., BTC)
        trading_pair: Trading pair (e.g., BTC/USDT)
        model_type: Type of model (ensemble, lstm, or gru)
        lookback: Lookback period in days
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate for optimizer
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
        
        logger.info("="*70)
        logger.info("OPTIMIZED CRYPTOCURRENCY PRICE PREDICTION MODEL TRAINING")
        logger.info("="*70)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Trading Pair: {trading_pair}")
        logger.info(f"Model Type: {model_type}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batch Size: {batch_size}")
        logger.info(f"Lookback Period: {lookback} days")
        logger.info(f"Learning Rate: {learning_rate}")
        logger.info(f"Device: {selected_device.upper()}")
        logger.info("="*70)
        
        # Step 1: Fetch historical data
        logger.info("\n[Step 1/5] Fetching historical data...")
        data_fetcher = DataFetcher()
        df = data_fetcher.fetch_ohlcv_binance(trading_pair, timeframe='1d', limit=500)
        
        if df is None or df.empty:
            logger.error(f"Failed to fetch data for {symbol}")
            return False
        
        logger.info(f"✓ Fetched {len(df)} candles for {trading_pair}")
        
        # Step 2: Add technical indicators
        logger.info("\n[Step 2/5] Adding technical indicators...")
        df = data_fetcher.add_technical_indicators(df)
        logger.info(f"✓ Added technical indicators for {len(df)} rows")
        
        # Step 3: Prepare ML features
        logger.info("\n[Step 3/5] Preparing ML features...")
        features_dict = data_fetcher.prepare_ml_features(df, lookback=lookback)
        X = features_dict['X']
        y = features_dict['y']
        
        if X is None or y is None:
            logger.error("Failed to prepare features")
            return False
        
        logger.info(f"✓ Prepared {X.shape[0]} sequences")
        logger.info(f"  - X shape: {X.shape}")
        logger.info(f"  - y shape: {y.shape}")
        
        # Step 4: Initialize trainer and train model
        logger.info("\n[Step 4/5] Training ensemble model with attention mechanisms...")
        trainer = OptimizedModelTrainer(device=selected_device)
        
        model, history = trainer.train_ensemble(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        # Step 5: Save model
        logger.info("\n[Step 5/5] Saving trained model...")
        model_dir = Path('models/saved_models')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save symbol-specific model
        symbol_model_path = model_dir / f"{symbol}_lstm_model.pth"
        torch.save(model.state_dict(), symbol_model_path)
        logger.info(f"✓ Model saved to {symbol_model_path}")
        
        # Save as best model for backward compatibility
        best_model_path = model_dir / 'best_lstm_model.pth'
        torch.save(model.state_dict(), best_model_path)
        logger.info(f"✓ Model also saved as best_lstm_model.pth")
        
        # Training summary
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        logger.info(f"Final Training Loss: {history['train_loss'][-1]:.6f}")
        logger.info(f"Final Validation Loss: {history['val_loss'][-1]:.6f}")
        logger.info(f"Best Validation Loss: {min(history['val_loss']):.6f}")
        logger.info(f"Total Epochs Trained: {len(history['train_loss'])}")
        logger.info(f"Model Location: {symbol_model_path}")
        logger.info("="*70)
        logger.info("\nKey Features Implemented:")
        logger.info("  ✓ Bidirectional LSTM + GRU ensemble")
        logger.info("  ✓ Multi-head attention mechanism")
        logger.info("  ✓ Huber loss (robust to outliers)")
        logger.info("  ✓ Early stopping with patience")
        logger.info("  ✓ Mixed precision training (GPU)")
        logger.info("  ✓ Gradient clipping")
        logger.info("  ✓ Cosine annealing learning rate schedule")
        logger.info("="*70 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train optimized cryptocurrency prediction model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard training with GPU
  python train_model.py --symbol BTC --epochs 100 --batch-size 64
  
  # Fast training
  python train_model.py --symbol BTC --epochs 50 --batch-size 128
  
  # High precision training
  python train_model.py --symbol BTC --epochs 150 --batch-size 32
  
  # Force CPU training
  python train_model.py --symbol BTC --device cpu
        """
    )
    
    parser.add_argument('--symbol', default='BTC', help='Cryptocurrency symbol (default: BTC)')
    parser.add_argument('--trading-pair', default='BTC/USDT', help='Trading pair (default: BTC/USDT)')
    parser.add_argument('--model', default='ensemble', choices=['ensemble', 'lstm', 'gru'],
                        help='Model type (default: ensemble)')
    parser.add_argument('--lookback', type=int, default=60, help='Lookback period in days (default: 60)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='Training device (default: auto)')
    
    args = parser.parse_args()
    
    success = train_model_optimized(
        symbol=args.symbol,
        trading_pair=args.trading_pair,
        model_type=args.model,
        lookback=args.lookback,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    exit(0 if success else 1)

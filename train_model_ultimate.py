#!/usr/bin/env python3
"""
Ultimate Cryptocurrency Price Prediction Model Trainer
Supports long-duration training with advanced regularization for maximum accuracy

Usage:
  python train_model_ultimate.py --symbol BTC --epochs 300
  python train_model_ultimate.py --symbol ETH --epochs 300 --batch-size 16
  python train_model_ultimate.py --symbol SOL --epochs 500 --device cuda
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
from src.model_trainer_ultimate import UltimateModelTrainer

load_dotenv()
setup_logging(log_level='INFO', log_file='logs/training_ultimate.log')
logger = logging.getLogger(__name__)


def train_model_ultimate(
    symbol: str,
    trading_pair: str,
    lookback: int = 60,
    epochs: int = 300,
    batch_size: int = 16,
    learning_rate: float = 0.00005,
    device: str = 'auto'
):
    """
    Train ultimate cryptocurrency prediction model with maximum accuracy focus.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., BTC)
        trading_pair: Trading pair (e.g., BTC/USDT)
        lookback: Lookback period in days
        epochs: Number of training epochs (recommended: 200-500)
        batch_size: Batch size (smaller = more stable, default: 16)
        learning_rate: Learning rate (default: 0.00005)
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
        logger.info("🚀 ULTIMATE CRYPTOCURRENCY PRICE PREDICTION MODEL TRAINING")
        logger.info("="*80)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Trading Pair: {trading_pair}")
        logger.info(f"Lookback Period: {lookback} days")
        logger.info(f"Training Configuration:")
        logger.info(f"  - Epochs: {epochs}")
        logger.info(f"  - Batch Size: {batch_size}")
        logger.info(f"  - Learning Rate: {learning_rate}")
        logger.info(f"  - Device: {selected_device.upper()}")
        logger.info("="*80)
        
        # Step 1: Fetch historical data
        logger.info("\n[Step 1/5] Fetching historical data...")
        data_fetcher = DataFetcher()
        df = data_fetcher.fetch_ohlcv_binance(trading_pair, timeframe='1d', limit=1000)
        
        if df is None or df.empty:
            logger.error(f"Failed to fetch data for {symbol}")
            return False
        
        logger.info(f"[OK] Fetched {len(df)} candles for {trading_pair}")
        
        # Step 2: Add technical indicators
        logger.info("\n[Step 2/5] Adding technical indicators...")
        df = data_fetcher.add_technical_indicators(df)
        logger.info(f"[OK] Added technical indicators for {len(df)} rows")
        
        # Step 3: Prepare ML features
        logger.info("\n[Step 3/5] Preparing ML features...")
        features_result = data_fetcher.prepare_ml_features(df, lookback=lookback)
        
        # Handle both dict and tuple returns for backward compatibility
        if isinstance(features_result, dict):
            X = features_result['X']
            y = features_result['y']
        elif isinstance(features_result, tuple):
            X = features_result[0]
            y = features_result[1]
        else:
            logger.error("Unexpected features_result type")
            return False
        
        if X is None or y is None:
            logger.error("Failed to prepare features")
            return False
        
        logger.info(f"[OK] Prepared {X.shape[0]} sequences")
        logger.info(f"    - X shape: {X.shape}")
        logger.info(f"    - y shape: {y.shape}")
        
        # Step 4: Initialize trainer and train model
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
        
        # Save symbol-specific model
        symbol_model_path = model_dir / f"{symbol}_ultimate_model.pth"
        torch.save(model.state_dict(), symbol_model_path)
        logger.info(f"[OK] Ultimate model saved to {symbol_model_path}")
        
        # Also save as best model for backward compatibility
        best_model_path = model_dir / 'best_lstm_model.pth'
        torch.save(model.state_dict(), best_model_path)
        logger.info(f"[OK] Model also saved as best_lstm_model.pth")
        
        # Training summary
        logger.info("\n" + "="*80)
        logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"Final Training Loss: {history['train_loss'][-1]:.6f}")
        logger.info(f"Final Validation Loss: {history['val_loss'][-1]:.6f}")
        logger.info(f"Best Validation Loss: {min(history['val_loss']):.6f}")
        logger.info(f"Final Overfitting Ratio: {history['overfitting_ratio'][-1]:.3f}")
        logger.info(f"Total Epochs Trained: {len(history['train_loss'])}")
        logger.info(f"Model Location: {symbol_model_path}")
        logger.info("="*80)
        logger.info("\n🎯 Key Features Implemented:")
        logger.info("  [✓] Ultra-deep ensemble (LSTM-5 + GRU-5 + Transformer-4)")
        logger.info("  [✓] Hidden size: 512 (8.5M parameters)")
        logger.info("  [✓] Aggressive dropout: 0.6 + L2 regularization: 1e-3")
        logger.info("  [✓] Multi-loss function (MAE + Huber + L1)")
        logger.info("  [✓] Residual connections for better gradient flow")
        logger.info("  [✓] Warmup + Cosine annealing learning rate schedule")
        logger.info("  [✓] Gradient accumulation for stable training")
        logger.info("  [✓] Mixed precision training (GPU)")
        logger.info("  [✓] Early stopping with patient: 50 epochs")
        logger.info("="*80 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train ultimate cryptocurrency prediction model with maximum accuracy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard ultimate training (300 epochs)
  python train_model_ultimate.py --symbol BTC --epochs 300
  
  # Ultra-long training for maximum accuracy (500 epochs)
  python train_model_ultimate.py --symbol ETH --epochs 500 --batch-size 16
  
  # Custom configuration
  python train_model_ultimate.py --symbol SOL --epochs 300 --batch-size 8 --learning-rate 0.00003
  
  # Force CPU training
  python train_model_ultimate.py --symbol BTC --device cpu --epochs 200
        """
    )
    
    parser.add_argument('--symbol', default='BTC', help='Cryptocurrency symbol (default: BTC)')
    parser.add_argument('--trading-pair', default=None, help='Trading pair (default: {SYMBOL}/USDT)')
    parser.add_argument('--lookback', type=int, default=60, help='Lookback period in days (default: 60)')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs (default: 300)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size (default: 16, smaller = more stable)')
    parser.add_argument('--learning-rate', type=float, default=0.00005, help='Learning rate (default: 0.00005)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='Training device (default: auto)')
    
    args = parser.parse_args()
    
    # Set trading pair if not provided
    trading_pair = args.trading_pair or f"{args.symbol}/USDT"
    
    success = train_model_ultimate(
        symbol=args.symbol,
        trading_pair=trading_pair,
        lookback=args.lookback,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    exit(0 if success else 1)

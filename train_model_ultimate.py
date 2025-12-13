#!/usr/bin/env python3
"""
Ultimate Cryptocurrency Price Prediction Model Trainer
With Enhanced Leading Indicators to Reduce Prediction Lag

Usage:
  python train_model_ultimate.py --symbol SOL --epochs 100 --device cuda
  python train_model_ultimate.py --symbol BTC --epochs 150 --device cuda
  python train_model_ultimate.py --symbol ETH --epochs 100 --device cuda
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
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 0.00005,
    device: str = 'auto'
):
    """
    Train ultimate cryptocurrency prediction model with enhanced leading indicators.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., SOL)
        trading_pair: Trading pair (e.g., SOL/USDT)
        lookback: Lookback period in hours
        epochs: Number of training epochs
        batch_size: Batch size
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
        logger.info("ULTIMATE MODEL TRAINING WITH ENHANCED LEADING INDICATORS")
        logger.info("="*80)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Trading Pair: {trading_pair}")
        logger.info(f"Lookback Period: {lookback} hours")
        logger.info(f"Training Configuration:")
        logger.info(f"  - Epochs: {epochs}")
        logger.info(f"  - Batch Size: {batch_size}")
        logger.info(f"  - Learning Rate: {learning_rate}")
        logger.info(f"  - Device: {selected_device.upper()}")
        logger.info("="*80)
        
        # Step 1: Fetch historical data
        logger.info("\n[Step 1/5] Fetching historical data...")
        data_fetcher = DataFetcher()
        df = data_fetcher.fetch_ohlcv_binance(trading_pair, timeframe='1h', limit=1000)
        
        if df is None or df.empty:
            logger.error(f"Failed to fetch data for {symbol}")
            return False
        
        logger.info(f"[OK] Fetched {len(df)} candles for {trading_pair}")
        
        # Step 2: Add technical indicators (including leading indicators)
        logger.info("\n[Step 2/5] Adding technical indicators + leading indicators...")
        df = data_fetcher.add_technical_indicators(df)
        logger.info(f"[OK] Added indicators for {len(df)} rows")
        logger.info(f"    - Standard indicators: 17 features")
        logger.info(f"    - Leading indicators: 23 features (ROC, Stochastic, Williams %R, MFI, CCI, etc.)")
        logger.info(f"    - Total: 40 features")
        
        # Step 3: Prepare ML features
        logger.info("\n[Step 3/5] Preparing ML features with enhanced feature set...")
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
        logger.info(f"    - X shape: {X.shape} ({X.shape[2]} features x {X.shape[1]} time steps)")
        logger.info(f"    - y shape: {y.shape}")
        
        # Step 4: Initialize trainer and train model
        logger.info("\n[Step 4/5] Training ultimate ensemble model...")
        trainer = UltimateModelTrainer(device=selected_device)
        
        model, history = trainer.train_ultimate_ensemble(
            X, y,
            symbol=symbol,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            accumulation_steps=2
        )
        
        # Step 5: Save model
        logger.info("\n[Step 5/5] Saving trained model...")
        model_dir = Path('models/saved_models')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Model is automatically saved during training
        model_path = model_dir / f"{symbol}_model.pth"
        logger.info(f"[OK] Model saved to {model_path}")
        
        # Training summary
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"Final Training Loss: {history['train_loss'][-1]:.6f}")
        logger.info(f"Final Validation Loss: {history['val_loss'][-1]:.6f}")
        logger.info(f"Best Validation Loss: {min(history['val_loss']):.6f}")
        logger.info(f"Final Overfitting Ratio: {history['overfitting_ratio'][-1]:.3f}")
        logger.info(f"Total Epochs Trained: {len(history['train_loss'])}")
        logger.info(f"Model Location: {model_path}")
        logger.info("\nOptimizations Applied:")
        logger.info("  1. Enhanced Feature Engineering:")
        logger.info("     - Rate of Change (ROC): 1-period, 3-period, 5-period momentum")
        logger.info("     - Price Acceleration: Momentum delta detection")
        logger.info("     - Volume Acceleration: Leading volume signals")
        logger.info("     - Stochastic Oscillator: Lead RSI momentum shifts")
        logger.info("     - Williams %R: Overbought/oversold detection")
        logger.info("     - Money Flow Index: Volume-based momentum")
        logger.info("     - Commodity Channel Index: Momentum oscillator")
        logger.info("  2. Advanced Regularization: Dropout 0.6 + L2 1e-3")
        logger.info("  3. Multi-loss Function: MAE + Huber + L1")
        logger.info("  4. Ensemble Architecture: LSTM-5 + GRU-5 + Transformer-4")
        logger.info("  5. Learning Rate Scheduling: Warmup + Cosine Annealing")
        logger.info("  6. Model Backup: Automatic old model backup with timestamp")
        logger.info("="*80)
        logger.info("\nNext Steps:")
        logger.info(f"  1. Visualize predictions: python visualize_predictions_ultimate.py --symbol {symbol}")
        logger.info(f"  2. Compare with previous: Check MAE improvement vs baseline")
        logger.info(f"  3. Train other symbols: BTC, ETH, etc.")
        logger.info("="*80 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train ultimate cryptocurrency prediction model with leading indicators',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard training with enhanced features
  python train_model_ultimate.py --symbol SOL --epochs 100 --device cuda
  
  # Train BTC with longer epochs
  python train_model_ultimate.py --symbol BTC --epochs 150 --device cuda
  
  # Train ETH
  python train_model_ultimate.py --symbol ETH --epochs 100 --device cuda
  
  # CPU training (slower)
  python train_model_ultimate.py --symbol SOL --epochs 100 --device cpu
        """
    )
    
    parser.add_argument('--symbol', default='SOL', help='Cryptocurrency symbol (default: SOL)')
    parser.add_argument('--trading-pair', default=None, help='Trading pair (default: {SYMBOL}/USDT)')
    parser.add_argument('--lookback', type=int, default=60, help='Lookback period in hours (default: 60)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--learning-rate', type=float, default=0.00005, help='Learning rate (default: 0.00005)')
    parser.add_argument('--device', default='cuda', choices=['auto', 'cuda', 'cpu'], help='Training device (default: cuda)')
    
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

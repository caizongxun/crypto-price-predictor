#!/usr/bin/env python3
"""
TFT V3 Simple Training - Price Prediction Only

Simpler approach focusing on price prediction accuracy
Removing directional head for better convergence

Key Improvements:
1. Single MSE loss (simpler, more stable)
2. Focus on price prediction accuracy
3. Better gradient flow
4. Faster convergence
5. More reliable predictions

Expected Performance:
- MAE: < 5 USD
- MAPE: < 3%
- R2: > 0.5
- Training time: 2-5 minutes

Usage:
  python train_tft_v3_simple.py --symbol SOL --epochs 150
  python train_tft_v3_simple.py --symbol BTC --epochs 200 --lr 0.00005
"""

import argparse
import logging
from pathlib import Path
import sys
import torch
import torch.nn as nn
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging, create_directories
from src.data_fetcher_tft_v3 import TFTDataFetcher
from src.model_tft_v3_optimized import TemporalFusionTransformerV3Optimized
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

load_dotenv()
setup_logging(log_level='INFO', log_file='logs/training_tft_v3_simple.log')
logger = logging.getLogger(__name__)


class TFTSimpleTrainer:
    """Trainer for price-prediction-only TFT model"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self, model, train_loader, optimizer, loss_fn, device):
        """Train one epoch"""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(-1)
            
            # Forward pass - simple price prediction
            price_pred = model(X_batch)
            
            # MSE loss
            loss = loss_fn(price_pred, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def evaluate(self, model, val_loader, loss_fn, device):
        """Evaluate model on validation set"""
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).unsqueeze(-1)
                
                price_pred = model(X_batch)
                loss = loss_fn(price_pred, y_batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def train(
        self,
        model,
        X,
        y,
        symbol,
        epochs=150,
        batch_size=32,
        learning_rate=0.0001,
        device='cuda'
    ):
        """Complete training pipeline"""
        logger.info("\n" + "="*80)
        logger.info("TFT V3 SIMPLE TRAINING (Price Only)")
        logger.info("="*80)
        
        # Split data
        logger.info(f"\n[1/5] Splitting data (80/20 train/val)...")
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        logger.info(f"  Train: {len(X_train)}, Val: {len(X_val)}")
        
        # Create dataloaders
        logger.info(f"\n[2/5] Creating dataloaders...")
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Setup training
        logger.info(f"\n[3/5] Setting up training...")
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        loss_fn = nn.MSELoss()
        
        logger.info(f"  Optimizer: AdamW (lr={learning_rate})")
        logger.info(f"  Loss: MSE (simple, stable)")
        logger.info(f"  Model: {model.__class__.__name__}")
        
        # Training loop
        logger.info(f"\n[4/5] Training for {epochs} epochs...\n")
        
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        best_model_state = None
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss = self.evaluate(model, val_loader, loss_fn, device)
            scheduler.step()
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                self.patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1:3d}/{epochs} | "
                          f"Train Loss: {train_loss:.6f} | "
                          f"Val Loss: {val_loss:.6f}")
            
            if self.patience_counter >= 30:
                logger.warning(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info(f"Restored best model")
        
        # Save model
        logger.info(f"\n[5/5] Saving model...")
        model_dir = Path('models/saved_models')
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f'{symbol}_tft_model.pth'
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info(f"TRAINING SUMMARY - {symbol}")
        logger.info("="*80)
        logger.info(f"\nMetrics:")
        logger.info(f"  Best Val Loss: {min(history['val_loss']):.6f}")
        logger.info(f"  Final Train Loss: {history['train_loss'][-1]:.6f}")
        logger.info(f"  Final Val Loss: {history['val_loss'][-1]:.6f}")
        logger.info(f"\nAnalysis:")
        if min(history['val_loss']) < 0.1:
            logger.info(f"  EXCELLENT - Model converged well")
        elif min(history['val_loss']) < 0.5:
            logger.info(f"  GOOD - Model is learning")
        else:
            logger.info(f"  FAIR - Model needs more training or better features")
        logger.info("="*80 + "\n")
        
        return model, history


def main():
    parser = argparse.ArgumentParser(
        description='TFT V3 Simple Training - Price Prediction Only'
    )
    
    parser.add_argument('--symbol', default='SOL', help='Crypto symbol')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    try:
        create_directories()
        
        # Fetch data
        logger.info(f"Fetching data for {args.symbol}...")
        fetcher = TFTDataFetcher()
        trading_pair = f"{args.symbol}/USDT"
        df = fetcher.fetch_ohlcv_binance(trading_pair, timeframe='1h', limit=5000)
        
        if df is None:
            logger.error(f"Failed to fetch data")
            return False
        
        # Add indicators
        logger.info("Adding indicators...")
        df = fetcher.add_tft_indicators(df)
        
        # Prepare features
        logger.info("Preparing features...")
        X, y, scaler = fetcher.prepare_ml_features(df, lookback=60)
        
        if X is None:
            return False
        
        # Initialize model WITHOUT direction head
        model = TemporalFusionTransformerV3Optimized(
            input_size=X.shape[2],
            hidden_size=256,
            num_heads=8,
            num_layers=2,
            dropout=0.2,
            use_direction_head=False  # Simple price prediction
        )
        
        # Train
        trainer = TFTSimpleTrainer(device=device)
        model, history = trainer.train(
            model, X, y,
            symbol=args.symbol,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device
        )
        
        logger.info("Training completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)

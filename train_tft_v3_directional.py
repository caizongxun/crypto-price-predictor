#!/usr/bin/env python3
"""
🎯 TFT Training V3 Directional - Optimized for Direction Prediction

✨ Key Improvements for Directional Accuracy:

1. Dual-Head Architecture
   - Price regression head (MSE loss)
   - Direction classification head (Cross-entropy loss)
   - Shared transformer encoder learns better features

2. Direction-Aware Attention
   - Attention mechanism focuses on price direction patterns
   - Direction gating in feed-forward network
   - Helps model understand trend changes

3. Combined Loss Function
   - Primary: Price prediction (MSE)
   - Secondary: Direction classification (CE)
   - Weighted sum balances both objectives

4. Enhanced Features
   - Momentum (rate of change)
   - Acceleration (change in momentum)
   - Recent direction history
   - Volatility-adjusted returns

📊 Expected Results:
- Directional Accuracy: 70-80%
- Better trend following
- More stable predictions
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

# UTF-8 encoding support for Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging, create_directories
from src.data_fetcher_tft_v3 import TFTDataFetcher
from src.model_tft_v3_optimized import (
    TemporalFusionTransformerV3Optimized,
    DirectionalLossV2,
    DirectionalAccuracyMetric
)
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

load_dotenv()
setup_logging(log_level='INFO', log_file='logs/training_tft_v3_directional.log')
logger = logging.getLogger(__name__)


class TFTDirectionalTrainer:
    """Trainer for direction-optimized TFT model"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def compute_direction_targets(self, y: np.ndarray) -> np.ndarray:
        """
        Compute direction targets from price sequence
        Returns: {0: down, 1: neutral, 2: up}
        """
        directions = np.zeros(len(y), dtype=np.long)
        
        # Compute price changes
        price_changes = np.diff(y, prepend=y[0])
        
        # Classify
        # Using threshold to avoid noise
        threshold = np.std(price_changes) * 0.1
        
        directions[price_changes > threshold] = 2   # Up
        directions[price_changes < -threshold] = 0  # Down
        directions[np.abs(price_changes) <= threshold] = 1  # Neutral
        
        return directions
    
    def train_epoch(self, model, train_loader, optimizer, loss_fn, device):
        """Train one epoch with direction supervision"""
        model.train()
        total_loss = 0.0
        total_dir_acc = 0.0
        num_batches = 0
        
        for X_batch, y_batch, dir_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(-1)  # (batch, 1)
            dir_batch = dir_batch.to(device).long()     # (batch,)
            
            # Forward pass - get both price and direction predictions
            price_pred, direction_logits = model(X_batch, return_direction_logits=True)
            
            # Compute loss
            loss = loss_fn(
                price_pred,
                y_batch,
                direction_logits,
                dir_batch
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Compute directional accuracy
            with torch.no_grad():
                dir_pred = direction_logits.argmax(dim=1)
                dir_acc = (dir_pred == dir_batch).float().mean().item()
                total_dir_acc += dir_acc
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_dir_acc = total_dir_acc / max(num_batches, 1)
        
        return avg_loss, avg_dir_acc
    
    def evaluate(self, model, val_loader, loss_fn, device):
        """Evaluate model on validation set"""
        model.eval()
        total_loss = 0.0
        total_dir_acc = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch, dir_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).unsqueeze(-1)
                dir_batch = dir_batch.to(device).long()
                
                price_pred, direction_logits = model(X_batch, return_direction_logits=True)
                
                loss = loss_fn(
                    price_pred,
                    y_batch,
                    direction_logits,
                    dir_batch
                )
                
                dir_pred = direction_logits.argmax(dim=1)
                dir_acc = (dir_pred == dir_batch).float().mean().item()
                
                total_loss += loss.item()
                total_dir_acc += dir_acc
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_dir_acc = total_dir_acc / max(num_batches, 1)
        
        return avg_loss, avg_dir_acc
    
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
        """Complete training pipeline with direction optimization"""
        logger.info("\n" + "="*80)
        logger.info("TFT V3 DIRECTIONAL TRAINING")
        logger.info("="*80)
        
        # Compute direction targets
        logger.info("\n[1/6] Computing direction targets...")
        dir_targets = self.compute_direction_targets(y)
        logger.info(f"Direction distribution: Down={np.sum(dir_targets==0)}, "
                   f"Neutral={np.sum(dir_targets==1)}, Up={np.sum(dir_targets==2)}")
        
        # Split data
        logger.info(f"\n[2/6] Splitting data (80/20 train/val)...")
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        dir_train, dir_val = dir_targets[:split_idx], dir_targets[split_idx:]
        
        logger.info(f"  Train: {len(X_train)}, Val: {len(X_val)}")
        
        # Create dataloaders
        logger.info(f"\n[3/6] Creating dataloaders...")
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(dir_train, dtype=torch.long)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
            torch.tensor(dir_val, dtype=torch.long)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Setup training
        logger.info(f"\n[4/6] Setting up training...")
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        
        # Weighted loss: balance price and direction
        loss_fn = DirectionalLossV2(direction_weight=0.5)
        
        logger.info(f"  Optimizer: AdamW (lr={learning_rate})")
        logger.info(f"  Loss: Combined (MSE + Direction Classification)")
        logger.info(f"  Direction Weight: 0.5")
        
        # Training loop
        logger.info(f"\n[5/6] Training for {epochs} epochs...\n")
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_dir_acc': [],
            'val_dir_acc': []
        }
        
        best_model_state = None
        
        for epoch in range(epochs):
            train_loss, train_dir_acc = self.train_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss, val_dir_acc = self.evaluate(model, val_loader, loss_fn, device)
            scheduler.step()
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_dir_acc'].append(train_dir_acc)
            history['val_dir_acc'].append(val_dir_acc)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                self.patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1:3d}/{epochs} | "
                          f"Loss: {train_loss:.6f}/{val_loss:.6f} | "
                          f"Dir Acc: {train_dir_acc:.2%}/{val_dir_acc:.2%}")
            
            if self.patience_counter >= 25:
                logger.warning(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info(f"Restored best model")
        
        # Save model
        logger.info(f"\n[6/6] Saving model...")
        model_dir = Path('models/saved_models')
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f'{symbol}_tft_directional_model.pth'
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info(f"TRAINING SUMMARY - {symbol}")
        logger.info("="*80)
        logger.info(f"\nMetrics:")
        logger.info(f"  Best Val Loss: {self.best_val_loss:.6f}")
        logger.info(f"  Best Val Dir Acc: {max(history['val_dir_acc']):.2%}")
        logger.info(f"  Final Train Dir Acc: {history['train_dir_acc'][-1]:.2%}")
        logger.info(f"  Final Val Dir Acc: {history['val_dir_acc'][-1]:.2%}")
        logger.info("="*80 + "\n")
        
        return model, history


def main():
    parser = argparse.ArgumentParser(
        description='TFT V3 Directional Training - Optimized for Direction Prediction'
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
        
        # Initialize model with direction head
        model = TemporalFusionTransformerV3Optimized(
            input_size=X.shape[2],
            hidden_size=256,
            num_heads=8,
            num_layers=2,
            dropout=0.2,
            use_direction_head=True
        )
        
        # Train
        trainer = TFTDirectionalTrainer(device=device)
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

#!/usr/bin/env python3
"""
TFT Enhanced Training - Better Performance

Key Improvements:
1. Deeper model (4 layers vs 2)
2. Larger hidden size (512 vs 256)
3. Residual connections
4. Better loss function (ScaledMSE)
5. Stronger architecture

Expected Performance:
- MAE: < 3 USD
- MAPE: < 2%
- R2: > 0.7

Usage:
  python train_tft_enhanced.py --symbol SOL --epochs 200
  python train_tft_enhanced.py --symbol BTC --lr 0.00005 --batch-size 16
"""

import argparse
import logging
from pathlib import Path
import sys
import torch
import torch.nn as nn
import numpy as np
from dotenv import load_dotenv
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging, create_directories
from src.data_fetcher_tft_v3 import TFTDataFetcher
from src.model_tft_v3_enhanced import (
    TemporalFusionTransformerV3Enhanced,
    ScaledMSELoss,
    HuberLoss
)
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

load_dotenv()
setup_logging(log_level='INFO', log_file='logs/training_tft_enhanced.log')
logger = logging.getLogger(__name__)


class EnhancedTrainer:
    """Trainer for enhanced TFT model"""
    
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
            
            # Forward
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def evaluate(self, model, val_loader, loss_fn, device):
        """Evaluate model"""
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).unsqueeze(-1)
                
                pred = model(X_batch)
                loss = loss_fn(pred, y_batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def train(
        self,
        model,
        X,
        y,
        symbol,
        epochs=200,
        batch_size=32,
        learning_rate=0.0001,
        device='cuda'
    ):
        """Complete training pipeline"""
        logger.info("\n" + "="*80)
        logger.info("TFT ENHANCED TRAINING")
        logger.info("="*80)
        
        # Split data
        logger.info(f"\n[1/5] Preparing data...")
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        logger.info(f"  Train: {len(X_train)}, Val: {len(X_val)}")
        logger.info(f"  Price range: {y.min():.2f} - {y.max():.2f} USD")
        logger.info(f"  Std dev: {y.std():.2f} USD")
        
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
        
        # Setup
        logger.info(f"\n[3/5] Setting up training...")
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        # Use ScaledMSE loss for better gradient signal
        loss_fn = ScaledMSELoss()
        
        logger.info(f"  Device: {device}")
        logger.info(f"  Model: Enhanced TFT V3")
        logger.info(f"  Hidden size: 512")
        logger.info(f"  Layers: 4")
        logger.info(f"  Loss: ScaledMSE (relative error)")
        logger.info(f"  Optimizer: AdamW (lr={learning_rate})")
        
        # Training
        logger.info(f"\n[4/5] Training for {epochs} epochs...\n")
        
        history = {'train_loss': [], 'val_loss': []}
        best_model = None
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss = self.evaluate(model, val_loader, loss_fn, device)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                best_model = model.state_dict().copy()
            else:
                self.patience_counter += 1
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1:3d}/{epochs} | "
                          f"Train Loss: {train_loss:.6f} | "
                          f"Val Loss: {val_loss:.6f}")
            
            if self.patience_counter >= 30:
                logger.warning(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best
        if best_model is not None:
            model.load_state_dict(best_model)
            logger.info(f"Restored best model")
        
        # Save
        logger.info(f"\n[5/5] Saving model...")
        model_dir = Path('models/saved_models')
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f'{symbol}_tft_enhanced_model.pth'
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info(f"TRAINING COMPLETE - {symbol}")
        logger.info("="*80)
        logger.info(f"\nFinal Metrics:")
        logger.info(f"  Best Val Loss: {min(history['val_loss']):.6f}")
        logger.info(f"  Final Train Loss: {history['train_loss'][-1]:.6f}")
        logger.info(f"  Final Val Loss: {history['val_loss'][-1]:.6f}")
        if history['train_loss'][0] > 0:
            logger.info(f"  Improvement: {((history['train_loss'][0] - history['train_loss'][-1]) / history['train_loss'][0] * 100):.1f}%")
        logger.info("="*80 + "\n")
        
        return model, history


def main():
    parser = argparse.ArgumentParser(description='TFT Enhanced Training')
    parser.add_argument('--symbol', default='SOL')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    try:
        create_directories()
        
        logger.info(f"Fetching data for {args.symbol}...")
        fetcher = TFTDataFetcher()
        df = fetcher.fetch_ohlcv_binance(f"{args.symbol}/USDT", timeframe='1h', limit=5000)
        
        if df is None:
            logger.error("Failed to fetch data")
            return False
        
        logger.info("Adding indicators...")
        df = fetcher.add_tft_indicators(df)
        
        logger.info("Preparing features...")
        X, y, scaler = fetcher.prepare_ml_features(df, lookback=60)
        
        if X is None:
            return False
        
        # Use enhanced model
        model = TemporalFusionTransformerV3Enhanced(
            input_size=X.shape[2],
            hidden_size=512,
            num_heads=8,
            num_layers=4,
            dropout=0.2,
            output_size=1
        )
        
        trainer = EnhancedTrainer(device=device)
        model, history = trainer.train(
            model, X, y,
            symbol=args.symbol,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device
        )
        
        logger.info("Training completed!")
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)

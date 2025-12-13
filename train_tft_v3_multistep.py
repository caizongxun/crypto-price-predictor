#!/usr/bin/env python3
"""
🚀 TFT V3 Multi-Step Forecasting Trainer (v1.2+)

Trains TFT model to predict 3-5 candles ahead with confidence intervals.

Features:
- Multi-step loss optimization
- Direction classification auxiliary task
- Volatility-aware training
- Performance metrics: MAE, MAPE, Direction Accuracy
- Automatic model checkpointing
- GPU memory optimized

Usage:
  python train_tft_v3_multistep.py --symbol SOL --epochs 100
  python train_tft_v3_multistep.py --symbol BTC --lr 0.0005 --batch-size 16
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json

sys.path.insert(0, str(Path(__file__).parent))

from src.data_fetcher_tft_v3 import TFTDataFetcher
from src.model_tft_v3_enhanced_optimized import (
    TemporalFusionTransformerV3EnhancedOptimized,
    EnhancedOptimizedLoss
)
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class MultiStepTrainer:
    """Trainer for multi-step forecasting"""
    
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        device,
        symbol: str = 'SOL',
        lookback: int = 60,
        forecast_steps: int = 5
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.symbol = symbol
        self.lookback = lookback
        self.forecast_steps = forecast_steps
        
        self.scaler = None
        self.train_history = {'loss': [], 'val_loss': []}
        self.metrics_history = {}
    
    def prepare_data(self, df):
        """Prepare training data"""
        fetcher = TFTDataFetcher()
        df = fetcher.add_tft_indicators(df)
        
        X, y_original, self.scaler = fetcher.prepare_ml_features(
            df,
            lookback=self.lookback
        )
        
        if X is None:
            return None, None, None, None
        
        # Create multi-step targets
        y_multistep = self._create_multistep_targets(y_original)
        
        # Split train/val (80/20)
        split_idx = int(len(X) * 0.8)
        
        X_train = torch.tensor(X[:split_idx], dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_original[:split_idx], dtype=torch.float32).view(-1, 1).to(self.device)
        y_train_multistep = torch.tensor(
            y_multistep[:split_idx],
            dtype=torch.float32
        ).to(self.device)
        
        X_val = torch.tensor(X[split_idx:], dtype=torch.float32).to(self.device)
        y_val = torch.tensor(y_original[split_idx:], dtype=torch.float32).view(-1, 1).to(self.device)
        y_val_multistep = torch.tensor(
            y_multistep[split_idx:],
            dtype=torch.float32
        ).to(self.device)
        
        return (X_train, y_train, y_train_multistep), (X_val, y_val, y_val_multistep)
    
    def _create_multistep_targets(self, y: np.ndarray) -> np.ndarray:
        """Create multi-step ahead targets"""
        y_multistep = []
        
        for i in range(len(y) - self.forecast_steps):
            steps = y[i+1:i+1+self.forecast_steps]
            y_multistep.append(steps)
        
        # Pad last few samples
        for i in range(self.forecast_steps):
            y_multistep.append(np.repeat(y[-1], self.forecast_steps))
        
        return np.array(y_multistep)
    
    def _compute_direction_target(self, y_prev: torch.Tensor, y_curr: torch.Tensor) -> torch.Tensor:
        """Compute direction labels: 0=down, 1=neutral, 2=up"""
        price_diff = y_curr - y_prev
        directions = torch.zeros(y_curr.shape[0], dtype=torch.long, device=self.device)
        
        up_mask = price_diff > 0
        down_mask = price_diff < 0
        
        directions[up_mask.squeeze()] = 2
        directions[down_mask.squeeze()] = 0
        directions[(~up_mask & ~down_mask).squeeze()] = 1
        
        return directions
    
    def train_epoch(self, X_train, y_train, y_train_multistep, batch_size: int = 16):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        indices = torch.randperm(len(X_train))
        
        for i in tqdm(range(0, len(X_train), batch_size), desc="Training"):
            batch_indices = indices[i:i+batch_size]
            
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            y_multistep_batch = y_train_multistep[batch_indices]
            
            # Forward pass - returns dict when return_full_forecast=True
            predictions = self.model(X_batch, return_full_forecast=True)
            
            # Extract predictions with safety checks
            if isinstance(predictions, dict):
                price_pred = predictions.get('price')
                direction_logits = predictions.get('direction', None)
                multistep_pred = predictions.get('multistep', None)
            else:
                # Fallback if model returns tensor instead of dict
                price_pred = predictions
                direction_logits = None
                multistep_pred = None
            
            # Compute direction target
            y_prev = X_batch[:, -1:, 0].unsqueeze(-1)  # Last price in sequence
            direction_target = self._compute_direction_target(y_prev, y_batch)
            
            # Loss computation
            loss = self.loss_fn(
                price_pred=price_pred,
                price_target=y_batch,
                direction_logits=direction_logits,
                direction_target=direction_target,
                multistep_pred=multistep_pred,
                multistep_target=y_multistep_batch
            )
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(1, num_batches)
    
    def validate(self, X_val, y_val, y_val_multistep):
        """Validation step"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(X_val), 16):
                X_batch = X_val[i:i+16]
                y_batch = y_val[i:i+16]
                y_multistep_batch = y_val_multistep[i:i+16]
                
                predictions = self.model(X_batch, return_full_forecast=True)
                
                # Extract predictions with safety checks
                if isinstance(predictions, dict):
                    price_pred = predictions.get('price')
                    direction_logits = predictions.get('direction', None)
                    multistep_pred = predictions.get('multistep', None)
                else:
                    price_pred = predictions
                    direction_logits = None
                    multistep_pred = None
                
                y_prev = X_batch[:, -1:, 0].unsqueeze(-1)
                direction_target = self._compute_direction_target(y_prev, y_batch)
                
                loss = self.loss_fn(
                    price_pred=price_pred,
                    price_target=y_batch,
                    direction_logits=direction_logits,
                    direction_target=direction_target,
                    multistep_pred=multistep_pred,
                    multistep_target=y_multistep_batch
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(1, num_batches)
    
    def compute_metrics(self, X_val, y_val_original):
        """Compute evaluation metrics"""
        self.model.eval()
        
        with torch.no_grad():
            preds = self.model(X_val).squeeze().cpu().numpy()
            y_true = y_val_original.cpu().numpy().squeeze()
        
        # Handle scalar predictions
        if np.isscalar(preds):
            preds = np.array([preds])
        if np.isscalar(y_true):
            y_true = np.array([y_true])
        
        # Inverse transform
        num_features = X_val.shape[2]
        preds_full = np.zeros((len(preds), num_features))
        preds_full[:, 0] = preds
        preds_inverse = self.scaler.inverse_transform(preds_full)[:, 0]
        
        y_true_original = self.scaler.inverse_transform(
            np.hstack([y_true.reshape(-1, 1), np.zeros((len(y_true), num_features-1))])
        )[:, 0]
        
        # Metrics
        mae = np.mean(np.abs(preds_inverse - y_true_original))
        mape = np.mean(np.abs((y_true_original - preds_inverse) / (np.abs(y_true_original) + 1e-8))) * 100
        rmse = np.sqrt(np.mean((preds_inverse - y_true_original) ** 2))
        
        # Direction accuracy
        true_dir = np.sign(np.diff(y_true_original))
        pred_dir = np.sign(np.diff(preds_inverse))
        dir_acc = np.mean(true_dir == pred_dir) * 100
        
        return {
            'MAE': mae,
            'MAPE': mape,
            'RMSE': rmse,
            'Direction_Accuracy': dir_acc
        }
    
    def train(
        self,
        symbol: str,
        epochs: int = 100,
        batch_size: int = 16,
        early_stopping_patience: int = 20
    ):
        """Full training loop"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Training TFT V3 Multi-Step Forecaster: {symbol}")
        logger.info(f"{'='*80}")
        
        # Fetch and prepare data
        logger.info(f"[1/5] Fetching data for {symbol}...")
        fetcher = TFTDataFetcher()
        df = fetcher.fetch_ohlcv_binance(f"{symbol}/USDT", timeframe='1h', limit=5000)
        
        if df is None or df.empty:
            logger.error(f"Failed to fetch data for {symbol}")
            return
        
        logger.info(f"[2/5] Preparing training/validation sets...")
        train_data, val_data = self.prepare_data(df)
        
        if train_data is None:
            logger.error("Failed to prepare data")
            return
        
        X_train, y_train, y_train_multistep = train_data
        X_val, y_val, y_val_multistep = val_data
        
        logger.info(f"  Train samples: {len(X_train)}")
        logger.info(f"  Val samples: {len(X_val)}")
        logger.info(f"  Batch size: {batch_size}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"[3/5] Starting training loop...")
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(
                X_train, y_train, y_train_multistep,
                batch_size=batch_size
            )
            
            # Validation
            val_loss = self.validate(X_val, y_val, y_val_multistep)
            
            self.train_history['loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs} | "
                       f"Train Loss: {train_loss:.6f} | "
                       f"Val Loss: {val_loss:.6f}")
            
            # Early stopping and checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                os.makedirs('models/saved_models', exist_ok=True)
                model_path = f'models/saved_models/{symbol}_tft_multistep_best.pth'
                torch.save(self.model.state_dict(), model_path)
                logger.info(f"  ✓ Best model saved: {model_path}")
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info(f"[4/5] Computing final metrics...")
        metrics = self.compute_metrics(X_val, y_val)
        self.metrics_history = metrics
        
        logger.info(f"\n{'='*80}")
        logger.info(f"FINAL METRICS - {symbol}")
        logger.info(f"{'='*80}")
        logger.info(f"MAE:                   {metrics['MAE']:.6f} USD")
        logger.info(f"MAPE:                  {metrics['MAPE']:.4f}%")
        logger.info(f"RMSE:                  {metrics['RMSE']:.6f} USD")
        logger.info(f"Direction Accuracy:    {metrics['Direction_Accuracy']:.2f}%")
        logger.info(f"{'='*80}")
        
        # Save metrics
        logger.info(f"[5/5] Saving training history...")
        os.makedirs('models/training_logs', exist_ok=True)
        
        with open(f'models/training_logs/{symbol}_metrics_v1.2.json', 'w') as f:
            json.dump({
                'symbol': symbol,
                'version': 'v1.2_multistep',
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'train_history': {
                    'losses': self.train_history['loss'],
                    'val_losses': self.train_history['val_loss']
                }
            }, f, indent=2)
        
        logger.info(f"Training complete!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train TFT V3 Multi-Step Forecaster')
    parser.add_argument('--symbol', type=str, default='SOL', help='Crypto symbol')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size (default: 16 for GPU memory)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-size', type=int, default=128, help='Hidden size (reduced for GPU memory)')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of transformer layers (reduced for GPU memory)')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--forecast-steps', type=int, default=5, help='Steps to forecast ahead')
    parser.add_argument('--lookback', type=int, default=60, help='Lookback period')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Clear GPU cache before training
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Model initialization - GPU memory optimized
    model = TemporalFusionTransformerV3EnhancedOptimized(
        input_size=44,
        hidden_size=args.hidden_size,  # Reduced: 128 instead of 256
        num_heads=8,
        num_layers=args.num_layers,    # Reduced: 2 instead of 3
        dropout=args.dropout,
        output_size=1,
        forecast_steps=args.forecast_steps,
        use_direction_head=True,
        use_multistep_head=True
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    loss_fn = EnhancedOptimizedLoss(device=str(device), use_direction_loss=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    trainer = MultiStepTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        symbol=args.symbol,
        lookback=args.lookback,
        forecast_steps=args.forecast_steps
    )
    
    trainer.train(
        symbol=args.symbol,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=20
    )


if __name__ == '__main__':
    main()

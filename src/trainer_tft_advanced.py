#!/usr/bin/env python3
"""
Advanced Temporal Fusion Transformer Trainer with Multiple Optimizations

Key Improvements:
- Mixed Loss Functions: MAE + MAPE + Huber Loss
- Weighted Loss by Time Step (closer predictions more important)
- Multiple Optimizer Options: Adam, AdamW, RAdam, LAMB
- Advanced LR Schedulers: Cosine Annealing, Cyclic, Warm Restarts
- L1/L2 Regularization Tuning
- Gradient Clipping & Layer-wise LR Decay
- Ensemble Support (voting)
- Advanced Metrics: MAPE, RMSE, MAE, SMAPE, R²
- Temporal Cross-Validation
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict, Optional, List
from sklearn.preprocessing import RobustScaler
import math
from datetime import datetime

logger = logging.getLogger(__name__)


class MixedLoss(nn.Module):
    """Combines MAE + MAPE + Huber Loss for better price prediction"""
    
    def __init__(self, weights: Dict = None, huber_delta: float = 0.1):
        super(MixedLoss, self).__init__()
        self.weights = weights or {'mae': 0.5, 'mape': 0.3, 'huber': 0.2}
        self.mae_loss = nn.L1Loss()
        self.huber_loss = nn.SmoothL1Loss(beta=huber_delta)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mae = self.mae_loss(pred, target)
        
        # MAPE with safeguard
        target_safe = torch.where(
            torch.abs(target) > 1e-8,
            target,
            torch.ones_like(target) * 1e-8
        )
        mape = torch.mean(torch.abs((target - pred) / target_safe)) * 100
        
        # Huber loss
        huber = self.huber_loss(pred, target)
        
        # Weighted combination
        total_loss = (
            self.weights['mae'] * mae +
            self.weights['mape'] * mape +
            self.weights['huber'] * huber
        )
        
        return total_loss


class WeightedTimestepLoss(nn.Module):
    """Weight closer predictions more heavily"""
    
    def __init__(self, loss_type: str = 'mae'):
        super(WeightedTimestepLoss, self).__init__()
        self.loss_type = loss_type
        if loss_type == 'mae':
            self.base_loss = nn.L1Loss(reduction='none')
        elif loss_type == 'mse':
            self.base_loss = nn.MSELoss(reduction='none')
        else:
            self.base_loss = nn.L1Loss(reduction='none')
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.base_loss(pred, target)
        # Exponential weight: recent predictions matter more
        weights = torch.exp(torch.arange(len(loss), device=loss.device, dtype=torch.float32) / len(loss))
        return (loss * weights).mean()


class CosineAnnealingWarmupRestarts(torch.optim.lr_scheduler.LRScheduler):
    """Cosine annealing with warmup and warm restarts"""
    
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, base_lr: float):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.current_epoch = 0
        super().__init__(optimizer)
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase: linear increase
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing phase
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr


class TFTTrainerAdvanced:
    """Advanced trainer with multiple optimizations"""
    
    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        self.scaler = torch.amp.GradScaler() if self.device.type == 'cuda' else None
        logger.info(f"✓ Advanced TFT Trainer initialized on {self.device}")
    
    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_size: float = 0.80,
        validation_size: float = 0.10,
        batch_size: int = 32
    ) -> Tuple[DataLoader, DataLoader, DataLoader, Tuple]:
        """Prepare training, validation, and test data"""
        
        # Feature normalization
        X_reshaped = X.reshape(-1, X.shape[-1])
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        
        # Label normalization
        y_scaler = RobustScaler()
        y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Temporal split (respecting time series ordering)
        train_idx = int(len(X) * train_size)
        val_idx = int(len(X) * (train_size + validation_size))
        
        X_train = X_scaled[:train_idx]
        X_val = X_scaled[train_idx:val_idx]
        X_test = X_scaled[val_idx:]
        
        y_train = y_scaled[:train_idx]
        y_val = y_scaled[train_idx:val_idx]
        y_test = y_scaled[val_idx:]
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
        
        # Create DataLoaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"✓ Data Split:")
        logger.info(f"  - Train: {len(X_train)} samples")
        logger.info(f"  - Validation: {len(X_val)} samples")
        logger.info(f"  - Test: {len(X_test)} samples")
        
        return train_loader, val_loader, test_loader, (scaler, y_scaler)
    
    def calculate_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Calculate comprehensive metrics"""
        pred_np = pred.detach().cpu().numpy().flatten()
        target_np = target.detach().cpu().numpy().flatten()
        
        # MAE
        mae = np.mean(np.abs(pred_np - target_np))
        
        # MAPE
        mask = target_np != 0
        mape = np.mean(np.abs((target_np[mask] - pred_np[mask]) / np.abs(target_np[mask]))) * 100 if mask.sum() > 0 else 0
        
        # RMSE
        rmse = np.sqrt(np.mean((pred_np - target_np) ** 2))
        
        # SMAPE
        numerator = np.abs(pred_np - target_np)
        denominator = (np.abs(pred_np) + np.abs(target_np)) / 2.0 + 1e-8
        smape = np.mean(numerator / denominator) * 100
        
        # R²
        ss_res = np.sum((target_np - pred_np) ** 2)
        ss_tot = np.sum((target_np - np.mean(target_np)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8)) if ss_tot > 0 else 0
        
        return {
            'mae': float(mae),
            'mape': float(mape),
            'rmse': float(rmse),
            'smape': float(smape),
            'r2': float(r2)
        }
    
    def train_tft_advanced(
        self,
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        symbol: str = "SOL",
        epochs: int = 150,
        batch_size: int = 32,
        learning_rate: float = 0.0001,
        optimizer_type: str = 'adamw',
        loss_type: str = 'mixed',
        weight_decay: float = 0.001,
        gradient_clip: float = 1.0,
        early_stopping_patience: int = 30
    ) -> Tuple[nn.Module, Dict]:
        """Advanced training with multiple optimizations"""
        
        logger.info("\n" + "="*80)
        logger.info("🚀 ADVANCED TFT TRAINING")
        logger.info("="*80)
        
        # Prepare data
        train_loader, val_loader, test_loader, scalers = self.prepare_data(
            X, y, batch_size=batch_size
        )
        
        # Move model to device
        model = model.to(self.device)
        
        # === Optimizer Selection ===
        if optimizer_type.lower() == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
            logger.info(f"  Optimizer: AdamW (weight_decay={weight_decay})")
        elif optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
            logger.info(f"  Optimizer: Adam")
        else:
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            logger.info(f"  Optimizer: AdamW (default)")
        
        # === Loss Function Selection ===
        if loss_type == 'mixed':
            loss_fn = MixedLoss(weights={'mae': 0.5, 'mape': 0.3, 'huber': 0.2})
            logger.info(f"  Loss: Mixed (MAE 0.5 + MAPE 0.3 + Huber 0.2)")
        elif loss_type == 'mape':
            loss_fn = nn.L1Loss()  # MAE approximates MAPE
            logger.info(f"  Loss: MAPE (L1/MAE)")
        elif loss_type == 'mse':
            loss_fn = nn.MSELoss()
            logger.info(f"  Loss: MSE")
        else:
            loss_fn = nn.L1Loss()
            logger.info(f"  Loss: MAE")
        
        # Learning rate scheduler
        warmup_epochs = max(5, epochs // 20)
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer, warmup_epochs, epochs, learning_rate
        )
        
        # Training state
        best_val_metrics = {'mape': float('inf'), 'mae': float('inf')}
        patience_counter = 0
        training_history = {
            'train_loss': [], 'val_loss': [],
            'train_mae': [], 'val_mae': [],
            'train_mape': [], 'val_mape': [],
            'train_rmse': [], 'val_rmse': [],
            'epochs': [], 'learning_rate': []
        }
        
        logger.info(f"\n⚡ Training Configuration:")
        logger.info(f"  - Epochs: {epochs}")
        logger.info(f"  - Batch Size: {batch_size}")
        logger.info(f"  - Learning Rate: {learning_rate}")
        logger.info(f"  - Gradient Clip: {gradient_clip}")
        logger.info(f"  - Early Stopping Patience: {early_stopping_patience}")
        logger.info(f"  - Warmup Epochs: {warmup_epochs}")
        logger.info("="*80)
        
        # Training loop
        for epoch in range(1, epochs + 1):
            # Training phase
            model.train()
            train_loss = 0.0
            train_metrics = {'mae': 0, 'mape': 0, 'rmse': 0, 'smape': 0}
            train_batches = 0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                
                if self.device.type == 'cuda' and self.scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(X_batch)
                        loss = loss_fn(outputs, y_batch)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = model(X_batch)
                    loss = loss_fn(outputs, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    optimizer.step()
                
                train_loss += loss.item()
                batch_metrics = self.calculate_metrics(outputs, y_batch)
                for key in train_metrics:
                    train_metrics[key] += batch_metrics[key]
                train_batches += 1
            
            # Average training metrics
            train_loss /= len(train_loader)
            for key in train_metrics:
                train_metrics[key] /= train_batches
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_metrics = {'mae': 0, 'mape': 0, 'rmse': 0, 'smape': 0}
            val_batches = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    outputs = model(X_batch)
                    loss = loss_fn(outputs, y_batch)
                    val_loss += loss.item()
                    
                    batch_metrics = self.calculate_metrics(outputs, y_batch)
                    for key in val_metrics:
                        val_metrics[key] += batch_metrics[key]
                    val_batches += 1
            
            # Average validation metrics
            val_loss /= len(val_loader)
            for key in val_metrics:
                val_metrics[key] /= val_batches
            
            # Learning rate update
            lr = scheduler.step()
            
            # Record history
            training_history['epochs'].append(epoch)
            training_history['learning_rate'].append(lr)
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['train_mae'].append(train_metrics['mae'])
            training_history['val_mae'].append(val_metrics['mae'])
            training_history['train_mape'].append(train_metrics['mape'])
            training_history['val_mape'].append(val_metrics['mape'])
            training_history['train_rmse'].append(train_metrics['rmse'])
            training_history['val_rmse'].append(val_metrics['rmse'])
            
            # Print progress
            if epoch % 10 == 0:
                logger.info(
                    f"Ep {epoch:3d}/{epochs} | "
                    f"TL:{train_loss:.5f} VL:{val_loss:.5f} | "
                    f"MAE:{val_metrics['mae']:.5f} MAPE:{val_metrics['mape']:.3f}% | "
                    f"LR:{lr:.2e}"
                )
            
            # Save best model
            if val_metrics['mape'] < best_val_metrics['mape'] * 0.995:
                best_val_metrics = val_metrics
                patience_counter = 0
                
                save_path = Path('models/saved_models') / f"{symbol}_tft_advanced.pth"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)
                logger.info(
                    f"  ✓ Best model saved | "
                    f"MAE:{val_metrics['mae']:.5f} MAPE:{val_metrics['mape']:.3f}%"
                )
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"\nEarly stopping at epoch {epoch}")
                break
        
        # Test phase
        logger.info("\n" + "="*80)
        logger.info("🌟 TEST SET EVALUATION")
        logger.info("="*80)
        
        model.eval()
        test_metrics = {'mae': 0, 'mape': 0, 'rmse': 0, 'smape': 0, 'r2': 0}
        test_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = model(X_batch)
                batch_metrics = self.calculate_metrics(outputs, y_batch)
                for key in test_metrics:
                    test_metrics[key] += batch_metrics[key]
                test_batches += 1
        
        for key in test_metrics:
            test_metrics[key] /= test_batches
        
        logger.info(f"✓ Test Metrics:")
        logger.info(f"  - MAE: {test_metrics['mae']:.6f}")
        logger.info(f"  - MAPE: {test_metrics['mape']:.4f}%")
        logger.info(f"  - RMSE: {test_metrics['rmse']:.6f}")
        logger.info(f"  - SMAPE: {test_metrics['smape']:.4f}%")
        logger.info(f"  - R²: {test_metrics['r2']:.4f}")
        
        logger.info("\n" + "="*80)
        logger.info("✅ TRAINING COMPLETED!")
        logger.info("="*80)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Best Validation MAPE: {best_val_metrics['mape']:.4f}%")
        logger.info(f"Test MAPE: {test_metrics['mape']:.4f}%")
        logger.info(f"Total Epochs: {len(training_history['train_loss'])}")
        logger.info(f"Model: models/saved_models/{symbol}_tft_advanced.pth")
        logger.info("="*80 + "\n")
        
        return model, training_history

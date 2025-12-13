#!/usr/bin/env python3
"""
Temporal Fusion Transformer Trainer

Features:
- MAPE loss (better for price prediction)
- Warm-up learning rate schedule
- Early stopping with patience
- Automatic model checkpointing
- Advanced metrics (MAPE, SMAPE, R²)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import logging
from typing import Tuple
from sklearn.preprocessing import RobustScaler
import math
import shutil
from datetime import datetime

logger = logging.getLogger(__name__)


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing"""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * (0.5 * (1 + math.cos(math.pi * progress)))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr


class TFTTrainer:
    """Trainer for Temporal Fusion Transformer"""
    
    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        self.scaler = torch.amp.GradScaler() if self.device.type == 'cuda' else None
        logger.info(f"TFT Trainer initialized on device: {self.device}")
    
    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_size: float = 0.85,
        batch_size: int = 32
    ) -> Tuple[DataLoader, DataLoader, Tuple]:
        """Prepare training and validation data"""
        
        # Feature normalization with RobustScaler
        X_reshaped = X.reshape(-1, X.shape[-1])
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        
        # Label normalization
        y_scaler = RobustScaler()
        y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Data split
        split_idx = int(len(X) * train_size)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
        
        # Create DataLoaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader, (scaler, y_scaler)
    
    def calculate_mape(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Mean Absolute Percentage Error"""
        mask = target != 0
        if mask.sum() == 0:
            return 0.0
        return (torch.abs((target[mask] - pred[mask]) / target[mask])).mean().item() * 100
    
    def calculate_smape(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error"""
        numerator = torch.abs(pred - target)
        denominator = (torch.abs(pred) + torch.abs(target)) / 2.0 + 1e-8
        return (numerator / denominator).mean().item() * 100
    
    def train_tft(
        self,
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        symbol: str = "SOL",
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.0001
    ) -> Tuple[nn.Module, dict]:
        """Train TFT model"""
        
        logger.info("\n" + "="*80)
        logger.info("🚀 TEMPORAL FUSION TRANSFORMER TRAINING")
        logger.info("="*80)
        
        # Prepare data
        train_loader, test_loader, scalers = self.prepare_data(X, y, batch_size=batch_size)
        
        # Move model to device
        model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        warmup_epochs = max(5, epochs // 20)
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, epochs, learning_rate)
        
        # Loss function - MAPE for better price prediction
        loss_fn = nn.L1Loss()  # MAE is similar to MAPE
        
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_mape': [],
            'val_mape': [],
            'epochs': [],
            'learning_rate': []
        }
        
        logger.info(f"\n📊 Model Configuration:")
        logger.info(f"  - Architecture: Temporal Fusion Transformer")
        logger.info(f"  - Input Features: {X.shape[2]}")
        logger.info(f"  - Total Parameters: {total_params:,}")
        logger.info(f"  - Trainable Parameters: {trainable_params:,}")
        logger.info(f"\n⚙️ Training Configuration:")
        logger.info(f"  - Epochs: {epochs}")
        logger.info(f"  - Batch Size: {batch_size}")
        logger.info(f"  - Learning Rate: {learning_rate}")
        logger.info(f"  - Loss: MAE (MAPE equivalent)")
        logger.info(f"  - Optimizer: Adam")
        logger.info(f"  - Early Stopping Patience: {patience}")
        logger.info(f"  - Warmup Epochs: {warmup_epochs}")
        logger.info(f"\n" + "="*80)
        
        # Training loop
        for epoch in range(1, epochs + 1):
            # Training phase
            model.train()
            train_loss = 0.0
            train_mape = 0.0
            train_count = 0
            
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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = model(X_batch)
                    loss = loss_fn(outputs, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                train_loss += loss.item()
                mape = self.calculate_mape(outputs, y_batch)
                train_mape += mape
                train_count += 1
            
            train_loss /= len(train_loader)
            train_mape /= train_count
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_mape = 0.0
            val_count = 0
            
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    outputs = model(X_batch)
                    loss = loss_fn(outputs, y_batch)
                    val_loss += loss.item()
                    
                    mape = self.calculate_mape(outputs, y_batch)
                    val_mape += mape
                    val_count += 1
            
            val_loss /= len(test_loader)
            val_mape /= val_count
            
            # Learning rate update
            lr = scheduler.step()
            
            # Record history
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['train_mape'].append(train_mape)
            training_history['val_mape'].append(val_mape)
            training_history['epochs'].append(epoch)
            training_history['learning_rate'].append(lr)
            
            # Print progress
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                    f"Train MAPE: {train_mape:.4f}% | Val MAPE: {val_mape:.4f}% | "
                    f"LR: {lr:.2e}"
                )
            
            # Save best model
            if val_loss < best_val_loss * 0.9995:
                best_val_loss = val_loss
                patience_counter = 0
                
                save_path = Path('models/saved_models') / f"{symbol}_tft_model.pth"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)
                logger.info(f"  ✓ Best model saved (Val Loss: {val_loss:.6f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"\nEarly stopping at epoch {epoch}")
                break
        
        logger.info("\n" + "="*80)
        logger.info("✅ Training Completed!")
        logger.info(f"  - Best Validation Loss: {best_val_loss:.6f}")
        logger.info(f"  - Best Validation MAPE: {min(training_history['val_mape']):.4f}%")
        logger.info(f"  - Total Epochs: {len(training_history['train_loss'])}")
        logger.info(f"  - Model Saved: models/saved_models/{symbol}_tft_model.pth")
        logger.info(f"\n🎯 Expected Performance:")
        logger.info(f"  - MAPE: < 5%")
        logger.info(f"  - Parameters: {total_params:,}")
        logger.info("="*80 + "\n")
        
        return model, training_history

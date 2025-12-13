#!/usr/bin/env python3
"""
Simplified LSTM Model Trainer (Option A)
Focus: Stability and interpretability over complexity

Key Design:
- Single LSTM (hidden_size=128, num_layers=2)
- Minimal features: 10 core indicators only
- Moderate regularization (Dropout 0.4)
- Longer training (150+ epochs)
- Simple loss: MAE only

Expected performance: MAE 0.15-0.20
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
from sklearn.preprocessing import StandardScaler
import math
import shutil
from datetime import datetime

logger = logging.getLogger(__name__)


class SimpleLSTMModel(nn.Module):
    """Simple, stable LSTM for cryptocurrency price prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.4):
        super(SimpleLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input layer normalization
        self.input_ln = nn.LayerNorm(input_size)
        
        # Simple LSTM - bidirectional
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Layer normalization after LSTM
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Simple dense layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),
            
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # Input normalization
        x = self.input_ln(x)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        
        # Take last time step
        last_out = lstm_out[:, -1, :]
        
        # Dense prediction
        output = self.fc(last_out)
        
        return output


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


class SimpleLSTMTrainer:
    """Trainer for simplified LSTM model"""
    
    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        self.scaler = torch.amp.GradScaler() if self.device.type == 'cuda' else None
        logger.info(f"SimpleLSTM Trainer initialized on device: {self.device}")
    
    def backup_old_model(self, symbol: str, model_type: str = 'simple'):
        """Backup existing model before overwriting"""
        model_dir = Path('models/saved_models')
        backup_dir = Path('models/saved_models_backup')
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        model_name = f"{symbol}_{model_type}_model.pth"
        model_path = model_dir / model_name
        
        if model_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{symbol}_{model_type}_model_{timestamp}.pth"
            backup_path = backup_dir / backup_name
            
            try:
                shutil.copy2(model_path, backup_path)
                logger.info(f"Old model backed up to: {backup_path}")
            except Exception as e:
                logger.error(f"Failed to backup model: {e}")
    
    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_size: float = 0.85,
        batch_size: int = 32
    ) -> Tuple[DataLoader, DataLoader, Tuple]:
        """Prepare training and validation data"""
        
        # Feature normalization
        X_reshaped = X.reshape(-1, X.shape[-1])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        
        # Label normalization
        y_scaler = StandardScaler()
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
    
    def train_simple_lstm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        symbol: str = "SOL",
        epochs: int = 150,
        batch_size: int = 32,
        learning_rate: float = 0.0001,
        hidden_size: int = 128,
        dropout: float = 0.4
    ) -> Tuple[SimpleLSTMModel, dict]:
        """Train simplified LSTM model"""
        
        logger.info("\n" + "="*80)
        logger.info("🚀 SIMPLIFIED LSTM MODEL TRAINING (Option A)")
        logger.info("="*80)
        
        # Prepare data
        train_loader, test_loader, scalers = self.prepare_data(X, y, batch_size=batch_size)
        
        # Initialize model
        input_size = X.shape[-1]
        model = SimpleLSTMModel(input_size, hidden_size=hidden_size, dropout=dropout).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,  # Moderate L2 regularization
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        warmup_epochs = 10
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, epochs, learning_rate)
        
        # Loss function - simple MAE
        loss_fn = nn.L1Loss()
        
        best_val_loss = float('inf')
        patience = 30
        patience_counter = 0
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': [],
            'overfitting_ratio': [],
            'learning_rate': []
        }
        
        logger.info(f"\n📊 Model Configuration:")
        logger.info(f"  - Architecture: Simple LSTM (Bidirectional)")
        logger.info(f"  - Hidden Size: {hidden_size}")
        logger.info(f"  - Num Layers: 2")
        logger.info(f"  - Input Features: {input_size}")
        logger.info(f"  - Total Parameters: {total_params:,}")
        logger.info(f"  - Trainable Parameters: {trainable_params:,}")
        logger.info(f"\n⚙️ Training Configuration:")
        logger.info(f"  - Epochs: {epochs}")
        logger.info(f"  - Batch Size: {batch_size}")
        logger.info(f"  - Learning Rate: {learning_rate}")
        logger.info(f"  - Optimizer: Adam (weight_decay=0.01)")
        logger.info(f"  - Loss Function: MAE (L1Loss)")
        logger.info(f"  - Dropout: {dropout}")
        logger.info(f"  - Early Stopping Patience: {patience}")
        logger.info(f"\n" + "="*80)
        
        # Backup old model
        self.backup_old_model(symbol, 'simple')
        
        # Training loop
        for epoch in range(1, epochs + 1):
            # Training phase
            model.train()
            train_loss = 0.0
            
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
            
            train_loss /= len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    outputs = model(X_batch)
                    loss = loss_fn(outputs, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(test_loader)
            
            # Learning rate update
            lr = scheduler.step()
            
            # Overfitting detection
            overfitting_ratio = val_loss / train_loss if train_loss > 0 else float('inf')
            
            # Record history
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['epochs'].append(epoch)
            training_history['overfitting_ratio'].append(overfitting_ratio)
            training_history['learning_rate'].append(lr)
            
            # Print progress
            if epoch % 15 == 0:
                status = "✓ GOOD" if overfitting_ratio < 1.2 else "⚠ MODERATE" if overfitting_ratio < 1.5 else "⚠ HIGH"
                logger.info(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                    f"Ratio: {overfitting_ratio:.3f} {status} | "
                    f"LR: {lr:.2e}"
                )
            
            # Save best model
            if val_loss < best_val_loss * 0.9995:
                best_val_loss = val_loss
                patience_counter = 0
                
                save_path = Path('models/saved_models') / f"{symbol}_simple_model.pth"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"\nEarly stopping at epoch {epoch}")
                break
        
        logger.info("\n" + "="*80)
        logger.info("✅ Training Completed!")
        logger.info(f"  - Best Validation Loss: {best_val_loss:.6f}")
        logger.info(f"  - Final Overfitting Ratio: {training_history['overfitting_ratio'][-1]:.3f}")
        logger.info(f"  - Total Epochs: {len(training_history['train_loss'])}")
        logger.info(f"  - Model Saved: models/saved_models/{symbol}_simple_model.pth")
        logger.info("\n🎯 Expected Performance:")
        logger.info(f"  - MAE Target: 0.15-0.20 (vs old 0.33)")
        logger.info(f"  - Parameters: ~{total_params:,} (vs Ensemble 8.5M)")
        logger.info(f"  - Training Speed: 2-3x faster")
        logger.info("="*80 + "\n")
        
        return model, training_history

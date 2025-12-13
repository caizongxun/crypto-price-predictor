#!/usr/bin/env python3
"""
CPU-Optimized Lightweight Cryptocurrency Price Prediction Model Trainer
Designed for safe CPU training without system freezing

Key Features:
- Lightweight LSTM + GRU (no Transformer to save memory)
- Hidden size: 128 (instead of 512)
- Dropout: 0.4 (instead of 0.6)
- Batch size: 8 (instead of 16)
- Single-loss optimization (MAE only, no multi-loss)
- Automatic gradient clearing
- CPU-friendly DataLoader (num_workers=0)
- Memory monitoring
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
import psutil
import gc

logger = logging.getLogger(__name__)


class LightweightLSTMModel(nn.Module):
    """Lightweight LSTM optimized for CPU training"""
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.4):
        super(LightweightLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input layer normalization
        self.input_ln = nn.LayerNorm(input_size)
        
        # Lightweight bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output layers with moderate regularization
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.75),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        x = self.input_ln(x)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        output = self.fc(last_out)
        return output


class LightweightGRUModel(nn.Module):
    """Lightweight GRU optimized for CPU training"""
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.4):
        super(LightweightGRUModel, self).__init__()
        
        self.input_ln = nn.LayerNorm(input_size)
        
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.75),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        x = self.input_ln(x)
        gru_out, _ = self.gru(x)
        last_out = gru_out[:, -1, :]
        output = self.fc(last_out)
        return output


class LightweightEnsembleModel(nn.Module):
    """Lightweight ensemble combining LSTM + GRU"""
    def __init__(self, lstm_model, gru_model):
        super(LightweightEnsembleModel, self).__init__()
        self.lstm_model = lstm_model
        self.gru_model = gru_model
        
        # Simple fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        lstm_out = self.lstm_model(x)
        gru_out = self.gru_model(x)
        combined = torch.cat([lstm_out, gru_out], dim=1)
        output = self.fusion(combined)
        return output


class CPUOptimizedTrainer:
    """CPU-friendly trainer with memory monitoring"""
    
    def __init__(self, device: str = None):
        # Ensure device is properly converted
        if device is None:
            self.device = torch.device('cpu')  # Force CPU for safety
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Starting RAM usage: {psutil.virtual_memory().percent}%")
    
    def get_memory_usage(self):
        """Get current memory usage in GB"""
        return psutil.virtual_memory().used / (1024**3)
    
    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_size: float = 0.85,
        batch_size: int = 8
    ) -> Tuple[DataLoader, DataLoader, Tuple]:
        """Prepare data with memory-efficient normalization"""
        
        logger.info(f"\nPreparing data...")
        logger.info(f"  - Input shape: {X.shape}")
        logger.info(f"  - Output shape: {y.shape}")
        
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
        
        # Create DataLoaders with num_workers=0 for Windows compatibility
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # CRITICAL: Must be 0 on Windows
            pin_memory=False  # Disable for CPU
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # CRITICAL: Must be 0 on Windows
            pin_memory=False  # Disable for CPU
        )
        
        logger.info(f"  - Training samples: {len(X_train)}")
        logger.info(f"  - Validation samples: {len(X_test)}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - RAM after data prep: {self.get_memory_usage():.2f} GB")
        
        return train_loader, test_loader, (scaler, y_scaler)
    
    def train_lightweight_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 8,
        learning_rate: float = 0.001
    ) -> Tuple[LightweightEnsembleModel, dict]:
        """Train lightweight ensemble with CPU safety"""
        
        logger.info("\n" + "="*80)
        logger.info("CPU-OPTIMIZED LIGHTWEIGHT MODEL TRAINER")
        logger.info("="*80)
        logger.info("This version is safe for CPU training without system freeze.\n")
        
        # Prepare data
        train_loader, test_loader, scalers = self.prepare_data(X, y, batch_size=batch_size)
        
        # Initialize lightweight models
        input_size = X.shape[-1]
        lstm_model = LightweightLSTMModel(input_size, hidden_size=128, num_layers=2, dropout=0.4).to(self.device)
        gru_model = LightweightGRUModel(input_size, hidden_size=128, num_layers=2, dropout=0.4).to(self.device)
        ensemble_model = LightweightEnsembleModel(lstm_model, gru_model).to(self.device)
        
        # Optimizer with moderate regularization
        optimizer = optim.Adam(
            ensemble_model.parameters(),
            lr=learning_rate,
            weight_decay=0.0001,
            betas=(0.9, 0.999)
        )
        
        # Simple MAE loss
        loss_fn = nn.L1Loss()
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        best_val_loss = float('inf')
        patience = 30
        patience_counter = 0
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': [],
            'memory_usage': [],
            'learning_rate': []
        }
        
        logger.info(f"Training Configuration:")
        logger.info(f"  - Model: Lightweight Ensemble (LSTM-2 + GRU-2)")
        logger.info(f"  - Hidden Size: 128 | Total Parameters: ~480K")
        logger.info(f"  - Dropout: 0.4 | Weight Decay: 0.0001")
        logger.info(f"  - Loss Function: MAE (simple and memory-efficient)")
        logger.info(f"  - Optimizer: Adam (conservative settings)")
        logger.info(f"  - Batch Size: {batch_size} | Epochs: {epochs}")
        logger.info(f"  - Device: CPU (safe mode)")
        logger.info("\n" + "="*80 + "\n")
        
        for epoch in range(1, epochs + 1):
            # Training phase
            ensemble_model.train()
            train_loss = 0.0
            
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = ensemble_model(X_batch)
                loss = loss_fn(outputs, y_batch)
                
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(ensemble_model.parameters(), 1.0)
                
                optimizer.step()
                train_loss += loss.item()
                
                # Memory monitoring
                if batch_idx % 5 == 0:
                    mem_usage = self.get_memory_usage()
                    if mem_usage > 10.0:  # Warn if using > 10GB
                        logger.warning(f"High memory usage: {mem_usage:.2f} GB - Consider reducing batch size")
            
            train_loss /= len(train_loader)
            
            # Validation phase
            ensemble_model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    outputs = ensemble_model(X_batch)
                    loss = loss_fn(outputs, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(test_loader)
            
            # Learning rate update
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Memory usage
            mem_usage = self.get_memory_usage()
            
            # Record history
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['epochs'].append(epoch)
            training_history['memory_usage'].append(mem_usage)
            training_history['learning_rate'].append(current_lr)
            
            # Progress logging
            if epoch % 10 == 0 or epoch == 1:
                status = "GOOD" if val_loss < train_loss * 1.2 else "CHECK"
                logger.info(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                    f"Status: {status} | "
                    f"Mem: {mem_usage:.2f}GB | LR: {current_lr:.2e}"
                )
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"\nEarly stopping at epoch {epoch}")
                break
            
            # Memory cleanup
            if epoch % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        logger.info("\n" + "="*80)
        logger.info(f"Training completed!")
        logger.info(f"  - Best validation loss: {best_val_loss:.6f}")
        logger.info(f"  - Final memory usage: {self.get_memory_usage():.2f} GB")
        logger.info(f"  - Total epochs trained: {len(training_history['train_loss'])}")
        logger.info("="*80 + "\n")
        
        return ensemble_model, training_history

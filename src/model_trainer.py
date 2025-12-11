"""Model training module for cryptocurrency price prediction."""

import logging
import os
from typing import Dict, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """LSTM neural network for time series prediction."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2, 
                 output_size: int = 1):
        """Initialize LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_size: Number of output features
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_size)
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output predictions
        """
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take last output
        last_output = attn_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class TransformerModel(nn.Module):
    """Transformer-based model for time series prediction."""
    
    def __init__(self, input_size: int, d_model: int = 512,
                 num_heads: int = 8, num_layers: int = 4,
                 dropout: float = 0.1, output_size: int = 1):
        """Initialize Transformer model.
        
        Args:
            input_size: Number of input features
            d_model: Dimension of the model
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            output_size: Number of output features
        """
        super(TransformerModel, self).__init__()
        
        self.embedding = nn.Linear(input_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_size)
        )
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output predictions
        """
        x = self.embedding(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


class ModelTrainer:
    """Trainer for cryptocurrency prediction models."""
    
    def __init__(self, model_type: str = 'lstm', config: Dict = None):
        """Initialize trainer.
        
        Args:
            model_type: 'lstm' or 'transformer'
            config: Configuration dictionary
        """
        self.model_type = model_type
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Model type: {model_type}")
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray,
                    test_size: float = 0.2) -> Tuple:
        """Prepare data for training.
        
        Args:
            X: Feature array
            y: Target array
            test_size: Proportion of test data
            
        Returns:
            Tuple of train and test data
        """
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=False
            )
            
            # Convert to tensors
            X_train = torch.from_numpy(X_train).float().to(self.device)
            X_test = torch.from_numpy(X_test).float().to(self.device)
            y_train = torch.from_numpy(y_train).float().to(self.device)
            y_test = torch.from_numpy(y_test).float().to(self.device)
            
            logger.info(f"Data prepared - Train: {X_train.shape}, Test: {X_test.shape}")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Failed to prepare data: {e}")
            return None
    
    def create_model(self, input_size: int) -> nn.Module:
        """Create the neural network model.
        
        Args:
            input_size: Number of input features
            
        Returns:
            Initialized model
        """
        try:
            if self.model_type == 'lstm':
                self.model = LSTMModel(
                    input_size=input_size,
                    hidden_size=self.config.get('hidden_size', 128),
                    num_layers=self.config.get('num_layers', 2),
                    dropout=self.config.get('dropout', 0.2)
                )
            elif self.model_type == 'transformer':
                self.model = TransformerModel(
                    input_size=input_size,
                    d_model=self.config.get('d_model', 512),
                    num_heads=self.config.get('num_heads', 8),
                    num_layers=self.config.get('num_layers', 4)
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            self.model.to(self.device)
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 0.001)
            )
            
            logger.info(f"{self.model_type.upper()} model created")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
            return self.model
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            return None
    
    def train(self, X_train: torch.Tensor, y_train: torch.Tensor,
             X_val: torch.Tensor, y_val: torch.Tensor,
             epochs: int = 100, batch_size: int = 32,
             early_stopping_patience: int = 10) -> Dict:
        """Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            early_stopping_patience: Early stopping patience
            
        Returns:
            Training history dictionary
        """
        try:
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            history = {'train_loss': [], 'val_loss': []}
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training phase
                self.model.train()
                train_loss = 0
                for batch_X, batch_y in train_loader:
                    self.optimizer.zero_grad()
                    predictions = self.model(batch_X)
                    loss = self.criterion(predictions, batch_y.unsqueeze(1))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                
                # Validation phase
                self.model.eval()
                with torch.no_grad():
                    val_predictions = self.model(X_val)
                    val_loss = self.criterion(val_predictions, y_val.unsqueeze(1)).item()
                
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} - "
                              f"Train Loss: {train_loss:.6f}, "
                              f"Val Loss: {val_loss:.6f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model(f"models/saved_models/best_{self.model_type}_model.pth")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            
            logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")
            return history
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return None
    
    def save_model(self, path: str):
        """Save model to disk.
        
        Args:
            path: Path to save model
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.model.state_dict(), path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self, path: str, input_size: int):
        """Load model from disk.
        
        Args:
            path: Path to model file
            input_size: Input feature size
        """
        try:
            self.create_model(input_size)
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

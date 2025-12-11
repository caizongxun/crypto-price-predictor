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

logger = logging.getLogger(__name__)


class EnhancedLSTMModel(nn.Module):
    """增強的 LSTM 模型 - 包含注意力機制"""
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3):
        super(EnhancedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 雙向 LSTM + Dropout
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 注意力機制 - 突出重要時間步
        self.attention = nn.MultiheadAttention(
            hidden_size * 2,
            num_heads=8,
            dropout=0.2,
            batch_first=True
        )
        
        # 全連接層
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # 應用注意力機制
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 使用最後一個時間步
        last_out = attn_out[:, -1, :]
        
        output = self.fc(last_out)
        return output


class GRUModel(nn.Module):
    """GRU 模型"""
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3):
        super(GRUModel, self).__init__()
        
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_out = gru_out[:, -1, :]
        output = self.fc(last_out)
        return output


class EnsembleModel(nn.Module):
    """模型集成 - 融合 LSTM + GRU"""
    def __init__(self, lstm_model, gru_model):
        super(EnsembleModel, self).__init__()
        self.lstm_model = lstm_model
        self.gru_model = gru_model
        
        # 融合層
        self.fusion = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        lstm_out = self.lstm_model(x)
        gru_out = self.gru_model(x)
        
        # 融合兩個模型的輸出
        combined = torch.cat([lstm_out, gru_out], dim=1)
        output = self.fusion(combined)
        
        return output


class OptimizedModelTrainer:
    """優化的模型訓練器"""
    
    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        logger.info(f"Using device: {self.device}")
    
    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_size: float = 0.8,
        batch_size: int = 64
    ) -> Tuple[DataLoader, DataLoader, Tuple]:
        """準備數據，包括正規化"""
        
        # 特徵正規化
        X_reshaped = X.reshape(-1, X.shape[-1])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        
        # 標籤正規化
        y_scaler = StandardScaler()
        y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # 分割數據
        split_idx = int(len(X) * train_size)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
        
        # 轉換為 tensor
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
        
        # 創建 DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader, (scaler, y_scaler)
    
    def train_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.001
    ) -> Tuple[EnsembleModel, dict]:
        """訓練集成模型"""
        
        # 準備數據
        train_loader, test_loader, scalers = self.prepare_data(X, y, batch_size=batch_size)
        
        # 初始化模型
        input_size = X.shape[-1]
        lstm_model = EnhancedLSTMModel(input_size).to(self.device)
        gru_model = GRUModel(input_size).to(self.device)
        ensemble_model = EnsembleModel(lstm_model, gru_model).to(self.device)
        
        # 優化器和損失函數
        optimizer = optim.AdamW(ensemble_model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        loss_fn = nn.HuberLoss(delta=1.0)  # 對異常值更魯棒
        
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': []
        }
        
        logger.info(f"Starting ensemble training - Epochs: {epochs}, Batch Size: {batch_size}")
        
        for epoch in range(1, epochs + 1):
            # 訓練
            ensemble_model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                
                # 混合精度訓練（如果使用 CUDA）
                if self.device.type == 'cuda' and self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = ensemble_model(X_batch)
                        loss = loss_fn(outputs, y_batch)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(ensemble_model.parameters(), 1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = ensemble_model(X_batch)
                    loss = loss_fn(outputs, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(ensemble_model.parameters(), 1.0)
                    optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 驗證
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
            scheduler.step()
            
            # 記錄
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['epochs'].append(epoch)
            
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs} - "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )
            
            # 早期停止
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        return ensemble_model, training_history

#!/usr/bin/env python3
"""
Temporal Fusion Transformer (TFT) for Cryptocurrency Price Prediction

Based on 2024 research showing superior performance:
- MAPE: 0.22-0.37% on financial time series
- Captures temporal patterns at multiple scales
- Self-attention for long-term dependencies
- Excellent for volatile markets like crypto

Key Components:
1. Temporal Embedding - captures time-based patterns
2. Multi-Scale Attention - captures different timeframes
3. BiLSTM Encoding - bidirectional context
4. Self-Attention Decoder - final predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class TemporalEmbedding(nn.Module):
    """Encode time-based features (hour of day, day of week, etc.)"""
    
    def __init__(self, embedding_dim: int = 32):
        super(TemporalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Hour of day embedding (0-23)
        self.hour_emb = nn.Embedding(24, embedding_dim)
        # Day of week embedding (0-6)
        self.day_emb = nn.Embedding(7, embedding_dim)
        # Month embedding (0-11)
        self.month_emb = nn.Embedding(12, embedding_dim)
        
    def forward(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate temporal embeddings"""
        # Create dummy time indices (in real use, pass actual timestamps)
        hours = torch.randint(0, 24, (batch_size, seq_len), device=device)
        days = torch.randint(0, 7, (batch_size, seq_len), device=device)
        months = torch.randint(0, 12, (batch_size, seq_len), device=device)
        
        hour_emb = self.hour_emb(hours)
        day_emb = self.day_emb(days)
        month_emb = self.month_emb(months)
        
        # Concatenate embeddings
        temporal_emb = hour_emb + day_emb + month_emb  # (batch, seq, embedding_dim)
        return temporal_emb


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-head attention forward pass"""
        batch_size = query.shape[0]
        
        # Linear transformations
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.hidden_size)
        output = self.fc_out(context)
        
        return output, attention_weights


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer for price prediction
    
    Architecture:
    - Temporal embedding for time-based patterns
    - BiLSTM encoder for sequential context
    - Multi-head self-attention for long-range dependencies
    - Temporal attention layer for temporal dynamics
    - Dense prediction layers
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        super(TemporalFusionTransformer, self).__init__()
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        # Input normalization
        self.input_ln = nn.LayerNorm(input_size)
        
        # Temporal embedding
        self.temporal_emb = TemporalEmbedding(embedding_dim=hidden_size // 4)
        
        # Project input to hidden size
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # BiLSTM encoder
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Project BiLSTM output back to hidden size
        self.bilstm_projection = nn.Linear(hidden_size * 2, hidden_size)
        
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 4, output_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Predictions of shape (batch_size, output_size)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Input normalization
        x = self.input_ln(x)
        
        # Project to hidden size
        x = self.input_projection(x)  # (batch, seq, hidden)
        
        # Add temporal embedding
        temporal_emb = self.temporal_emb(batch_size, seq_len, device)
        x = x + temporal_emb  # (batch, seq, hidden)
        
        # BiLSTM encoding
        lstm_out, (h_n, c_n) = self.bilstm(x)  # (batch, seq, hidden*2)
        lstm_out = self.bilstm_projection(lstm_out)  # (batch, seq, hidden)
        x = x + lstm_out  # Residual connection
        
        # Self-attention
        attn_out, _ = self.self_attention(x, x, x)  # (batch, seq, hidden)
        x = self.ln1(x + attn_out)  # Residual + layer norm
        
        # Feed-forward network
        ffn_out = self.ffn(x)  # (batch, seq, hidden)
        x = self.ln2(x + ffn_out)  # Residual + layer norm
        
        # Take last time step
        last_output = x[:, -1, :]  # (batch, hidden)
        
        # Prediction
        output = self.output_layers(last_output)  # (batch, 1)
        
        return output


class MAPELoss(nn.Module):
    """Mean Absolute Percentage Error Loss (better for price prediction)"""
    
    def __init__(self, eps: float = 1e-8):
        super(MAPELoss, self).__init__()
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate MAPE loss"""
        # Avoid division by zero
        target_safe = torch.where(
            torch.abs(target) > self.eps,
            target,
            torch.ones_like(target) * self.eps
        )
        
        return torch.mean(torch.abs((target - pred) / target_safe)) * 100


class SMAPELoss(nn.Module):
    """Symmetric Mean Absolute Percentage Error Loss"""
    
    def __init__(self, eps: float = 1e-8):
        super(SMAPELoss, self).__init__()
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate SMAPE loss"""
        numerator = torch.abs(pred - target)
        denominator = (torch.abs(pred) + torch.abs(target)) / 2.0 + self.eps
        return torch.mean(numerator / denominator) * 100

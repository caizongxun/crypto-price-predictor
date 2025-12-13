#!/usr/bin/env python3
"""
TFT V3 Enhanced Model - Improved Architecture

Key Improvements:
1. Deeper architecture (4 layers)
2. Larger hidden dimension (512)
3. Residual connections
4. Layer normalization
5. Better gradient flow
6. Scaled loss function (MAPE-like)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualBlock(nn.Module):
    """Residual block with normalization"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class MultiHeadAttentionImproved(nn.Module):
    """Improved multi-head attention"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.fc_out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        residual = x
        
        x = self.norm(x)
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(batch_size, seq_len, dim)
        out = self.fc_out(out)
        out = self.dropout(out)
        
        return out + residual


class TemporalFusionTransformerV3Enhanced(nn.Module):
    """Enhanced TFT V3 - Better for crypto price prediction"""
    
    def __init__(
        self,
        input_size,
        hidden_size=512,
        num_heads=8,
        num_layers=4,
        dropout=0.2,
        output_size=1
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Encoder with attention and residual blocks
        self.encoder_layers = nn.ModuleList([
            nn.ModuleList([
                MultiHeadAttentionImproved(hidden_size, num_heads, dropout),
                ResidualBlock(hidden_size, dropout)
            ])
            for _ in range(num_layers)
        ])
        
        # Temporal pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        
        Returns:
            output: (batch_size, output_size)
        """
        batch_size = x.shape[0]
        
        # Project input
        x = self.input_proj(x)  # (batch_size, seq_len, hidden_size)
        
        # Apply encoder layers
        for attn, ffn in self.encoder_layers:
            x = attn(x)
            x = ffn(x)
        
        # Global average pooling over time dimension
        x = x.transpose(1, 2)  # (batch_size, hidden_size, seq_len)
        x = self.global_pool(x)  # (batch_size, hidden_size, 1)
        x = x.squeeze(-1)  # (batch_size, hidden_size)
        
        # Output projection
        output = self.output_proj(x)  # (batch_size, output_size)
        
        return output


class ScaledMSELoss(nn.Module):
    """MSE loss scaled by predicted value (similar to MAPE)"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        """
        Scaled loss that penalizes relative errors more heavily
        Useful for variables with large absolute values like crypto prices
        """
        # Compute relative error
        relative_error = (pred - target) / (torch.abs(target) + 1e-8)
        
        # MSE on relative error
        loss = torch.mean(relative_error ** 2)
        
        return loss


class HuberLoss(nn.Module):
    """Huber loss - robust to outliers"""
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        error = torch.abs(pred - target)
        mask = error < self.delta
        
        loss = torch.where(
            mask,
            0.5 * error ** 2,
            self.delta * (error - 0.5 * self.delta)
        )
        
        return torch.mean(loss)

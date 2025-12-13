#!/usr/bin/env python3
"""
Enhanced Temporal Fusion Transformer with Architectural Improvements

Key Enhancements:
- BatchNormalization in addition to LayerNormalization
- GELU activation with SiLU alternatives
- Improved Residual Connections (with proper scaling)
- Gradient flow optimization
- Ensemble-ready architecture
- Quantization support
- Pruning-friendly design
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class ImprovedTemporalEmbedding(nn.Module):
    """Enhanced temporal embedding with normalization"""
    
    def __init__(self, input_dim: int = 8, embedding_dim: int = 32, use_batch_norm: bool = True):
        super(ImprovedTemporalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(input_dim, embedding_dim)
        
        if use_batch_norm:
            self.norm = nn.BatchNorm1d(embedding_dim)
        else:
            self.norm = nn.LayerNorm(embedding_dim)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, input_dim)
        batch_size, seq_len, _ = x.shape
        
        x = self.linear(x)  # (batch, seq, embedding_dim)
        
        # Apply normalization
        if isinstance(self.norm, nn.BatchNorm1d):
            x = x.view(-1, self.embedding_dim)
            x = self.norm(x)
            x = x.view(batch_size, seq_len, self.embedding_dim)
        else:
            x = self.norm(x)
        
        x = self.dropout(x)
        return x


class ImprovedMultiHeadAttention(nn.Module):
    """Enhanced multi-head attention with better numerical stability"""
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super(ImprovedMultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale_factor = np.sqrt(self.head_dim)
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Layer norm before attention
        self.ln = nn.LayerNorm(hidden_size)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.shape[0]
        
        # Layer norm before projection
        query = self.ln(query)
        
        # Linear projections
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, seq, dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale_factor
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Attention weights with numerical stability
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Context vector
        context = torch.matmul(attention_weights, V)  # (batch, heads, seq, dim)
        context = context.transpose(1, 2).contiguous()  # (batch, seq, heads, dim)
        context = context.view(batch_size, -1, self.hidden_size)  # (batch, seq, hidden_size)
        
        output = self.fc_out(context)
        
        return output, attention_weights


class ImprovedFeedForwardNetwork(nn.Module):
    """Enhanced FFN with better architecture"""
    
    def __init__(self, hidden_size: int, expansion: int = 4, dropout: float = 0.2):
        super(ImprovedFeedForwardNetwork, self).__init__()
        
        self.linear1 = nn.Linear(hidden_size, hidden_size * expansion)
        self.linear2 = nn.Linear(hidden_size * expansion, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout * 0.5)
        
        # Use GELU by default, can switch to SiLU
        self.activation = nn.GELU()
        
        # Layer norm
        self.ln = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-normalization
        x_norm = self.ln(x)
        
        # Feed-forward
        out = self.linear1(x_norm)
        out = self.activation(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.dropout2(out)
        
        # Residual connection with scaling
        return x + out * 0.95


class ResidualConnection(nn.Module):
    """Proper residual connection with dimension matching"""
    
    def __init__(self, hidden_size: int):
        super(ResidualConnection, self).__init__()
        self.scale = nn.Parameter(torch.tensor(0.95))
    
    def forward(self, residual: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        return residual + output * self.scale


class TemporalFusionTransformerEnhanced(nn.Module):
    """Enhanced Temporal Fusion Transformer
    
    Improvements:
    - Better normalization strategy
    - Improved attention mechanism
    - Enhanced residual connections
    - Better activation functions
    - Gradient flow optimization
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
        ffn_expansion: int = 4,
        use_batch_norm: bool = True
    ):
        super(TemporalFusionTransformerEnhanced, self).__init__()
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        # Input processing
        self.input_ln = nn.LayerNorm(input_size)
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.input_dropout = nn.Dropout(dropout * 0.5)
        
        # Temporal embedding
        self.temporal_emb = ImprovedTemporalEmbedding(
            input_dim=input_size,
            embedding_dim=hidden_size,
            use_batch_norm=use_batch_norm
        )
        
        # BiLSTM encoder with layer norm
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.bilstm_projection = nn.Linear(hidden_size * 2, hidden_size)
        self.bilstm_ln = nn.LayerNorm(hidden_size)
        
        # Self-attention layers (stacked)
        self.attention_layers = nn.ModuleList([
            ImprovedMultiHeadAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Feed-forward layers
        self.ffn_layers = nn.ModuleList([
            ImprovedFeedForwardNetwork(
                hidden_size=hidden_size,
                expansion=ffn_expansion,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Residual connections with proper scaling
        self.residual_connections = nn.ModuleList([
            ResidualConnection(hidden_size)
            for _ in range(num_layers * 2)  # For both attention and FFN
        ])
        
        # Output layers with improved design
        self.output_ln = nn.LayerNorm(hidden_size)
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 4, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with improved architecture
        
        Args:
            x: (batch_size, seq_len, input_size)
        
        Returns:
            (batch_size, output_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input normalization and projection
        x = self.input_ln(x)
        x_proj = self.input_projection(x)
        x_proj = self.input_dropout(x_proj)
        
        # Temporal embedding
        x_temporal = self.temporal_emb(x)
        
        # Combine projection and temporal embedding
        x = x_proj + x_temporal
        
        # BiLSTM encoding with residual
        lstm_out, _ = self.bilstm(x)
        lstm_out = self.bilstm_projection(lstm_out)
        lstm_out = self.bilstm_ln(lstm_out)
        x = x + lstm_out  # Residual connection
        
        # Stacked attention and FFN layers
        res_idx = 0
        for attn_layer, ffn_layer in zip(self.attention_layers, self.ffn_layers):
            # Self-attention with residual
            attn_out, _ = attn_layer(x, x, x)
            x = self.residual_connections[res_idx](x, attn_out)
            res_idx += 1
            
            # FFN with residual
            ffn_out = ffn_layer(x)
            x = ffn_out  # FFN already has residual built-in
            res_idx += 1
        
        # Output projection
        x = self.output_ln(x)
        
        # Take last time step
        last_output = x[:, -1, :]
        
        # Final prediction
        output = self.output_layers(last_output)
        
        return output


class EnsembleTemporalFusionTransformer(nn.Module):
    """Ensemble of TFT models for better predictions"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_models: int = 3,
        **kwargs
    ):
        super(EnsembleTemporalFusionTransformer, self).__init__()
        
        self.num_models = num_models
        self.models = nn.ModuleList([
            TemporalFusionTransformerEnhanced(
                input_size=input_size,
                hidden_size=hidden_size,
                **kwargs
            )
            for _ in range(num_models)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with ensemble voting"""
        outputs = torch.stack([
            model(x)
            for model in self.models
        ], dim=1)  # (batch, num_models, output_size)
        
        # Ensemble: average predictions
        return outputs.mean(dim=1)
    
    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        """Get predictions from all models"""
        return torch.stack([
            model(x)
            for model in self.models
        ], dim=1)

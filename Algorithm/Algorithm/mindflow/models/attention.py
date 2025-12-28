"""
MindFlow Attention Mechanisms
Self-attention and multi-head attention for capturing user behavior patterns.
Optimized for CPU inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, heads, seq_len, d_k)
            key: (batch, heads, seq_len, d_k)
            value: (batch, heads, seq_len, d_v)
            mask: Optional mask (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
        
        Returns:
            output: Attended values (batch, heads, seq_len, d_v)
            attention_weights: Attention probabilities (batch, heads, seq_len, seq_len)
        """
        d_k = query.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    Allows the model to jointly attend to information from different 
    representation subspaces at different positions.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, seq_len, d_model)
            key: (batch, seq_len, d_model)
            value: (batch, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head
        # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)
        
        # Apply attention
        attended, attention_weights = self.attention(q, k, v, mask)
        
        # Reshape and project back
        # (batch, num_heads, seq_len, d_v) -> (batch, seq_len, d_model)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(attended)
        
        return output, attention_weights


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # GELU works better than ReLU for this
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class AttentionBlock(nn.Module):
    """
    Complete attention block with:
    - Multi-head self-attention
    - Feed-forward network
    - Layer normalization (pre-norm for better training)
    - Residual connections
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        d_ff = d_ff or d_model * 4  # Default FFN dimension
        
        # Layer normalizations (pre-norm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        # Pre-norm self-attention with residual
        normed = self.norm1(x)
        attended, attention_weights = self.attention(normed, normed, normed, mask)
        x = x + self.dropout(attended)
        
        # Pre-norm FFN with residual
        normed = self.norm2(x)
        ff_out = self.ffn(normed)
        x = x + self.dropout(ff_out)
        
        return x, attention_weights


class AttentionEncoder(nn.Module):
    """
    Stack of attention blocks forming the encoder.
    Captures hierarchical patterns in user behavior sequences.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            AttentionBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            output: (batch, seq_len, d_model)
            all_attention_weights: List of attention weights from each layer
        """
        all_attention_weights = []
        
        for layer in self.layers:
            x, attention_weights = layer(x, mask)
            all_attention_weights.append(attention_weights)
        
        x = self.final_norm(x)
        
        return x, all_attention_weights

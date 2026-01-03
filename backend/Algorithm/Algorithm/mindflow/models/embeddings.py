"""
MindFlow Embedding Layers
Converts discrete user/content/action IDs into dense vector representations.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence position awareness.
    Helps the model understand temporal ordering of actions.
    """
    
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Handle both even and odd d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1] if len(div_term) > pe[:, 1::2].size(1) else div_term[:pe[:, 1::2].size(1)])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TimeEmbedding(nn.Module):
    """
    Time-aware embedding that captures temporal patterns.
    Encodes hour of day, day of week, and recency.
    """
    
    def __init__(self, embedding_dim: int, num_hours: int = 24, num_days: int = 7):
        super().__init__()
        
        # Hour of day embedding
        self.hour_embedding = nn.Embedding(num_hours, embedding_dim // 2)
        
        # Day of week embedding
        self.day_embedding = nn.Embedding(num_days, embedding_dim // 2)
        
        # Recency encoding (how long ago the action happened)
        self.recency_proj = nn.Linear(1, embedding_dim)
    
    def forward(
        self, 
        hour: torch.Tensor, 
        day: torch.Tensor, 
        recency: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hour: Hour of day (0-23), shape (batch, seq_len)
            day: Day of week (0-6), shape (batch, seq_len)
            recency: Time since action (normalized), shape (batch, seq_len, 1)
        Returns:
            Time embedding of shape (batch, seq_len, embedding_dim)
        """
        hour_emb = self.hour_embedding(hour)
        day_emb = self.day_embedding(day)
        
        time_emb = torch.cat([hour_emb, day_emb], dim=-1)
        
        if recency is not None:
            recency_emb = self.recency_proj(recency)
            time_emb = time_emb + recency_emb
        
        return time_emb


class EmbeddingLayer(nn.Module):
    """
    Main embedding layer that combines all entity embeddings.
    Creates a unified representation of user-content-action-time tuples.
    """
    
    def __init__(
        self,
        num_users: int,
        num_contents: int,
        num_action_types: int,
        user_embedding_dim: int = 64,
        content_embedding_dim: int = 64,
        action_embedding_dim: int = 32,
        time_embedding_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.user_embedding_dim = user_embedding_dim
        self.content_embedding_dim = content_embedding_dim
        self.action_embedding_dim = action_embedding_dim
        self.time_embedding_dim = time_embedding_dim
        
        # Entity embeddings
        self.user_embedding = nn.Embedding(num_users, user_embedding_dim, padding_idx=0)
        self.content_embedding = nn.Embedding(num_contents, content_embedding_dim, padding_idx=0)
        self.action_embedding = nn.Embedding(num_action_types, action_embedding_dim, padding_idx=0)
        
        # Time embedding
        self.time_embedding = TimeEmbedding(time_embedding_dim)
        
        # Positional encoding for sequence
        total_dim = user_embedding_dim + content_embedding_dim + action_embedding_dim + time_embedding_dim
        self.positional_encoding = PositionalEncoding(total_dim, dropout=dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(total_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.user_embedding.weight[1:])  # Skip padding idx
        nn.init.xavier_uniform_(self.content_embedding.weight[1:])
        nn.init.xavier_uniform_(self.action_embedding.weight[1:])
    
    @property
    def output_dim(self) -> int:
        """Total output dimension."""
        return (self.user_embedding_dim + 
                self.content_embedding_dim + 
                self.action_embedding_dim + 
                self.time_embedding_dim)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        content_ids: torch.Tensor,
        action_types: torch.Tensor,
        hours: torch.Tensor,
        days: torch.Tensor,
        recency: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Create unified embeddings from user-content-action-time tuples.
        
        Args:
            user_ids: User IDs, shape (batch, seq_len) or (batch,)
            content_ids: Content IDs, shape (batch, seq_len)
            action_types: Action type IDs, shape (batch, seq_len)
            hours: Hour of day, shape (batch, seq_len)
            days: Day of week, shape (batch, seq_len)
            recency: Time since action, shape (batch, seq_len, 1)
        
        Returns:
            Fused embedding of shape (batch, seq_len, total_embedding_dim)
        """
        # Handle user_ids - broadcast if single user per sequence
        if user_ids.dim() == 1:
            user_ids = user_ids.unsqueeze(1).expand(-1, content_ids.size(1))
        
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        content_emb = self.content_embedding(content_ids)
        action_emb = self.action_embedding(action_types)
        time_emb = self.time_embedding(hours, days, recency)
        
        # Concatenate all embeddings
        combined = torch.cat([user_emb, content_emb, action_emb, time_emb], dim=-1)
        
        # Add positional encoding
        combined = self.positional_encoding(combined)
        
        # Normalize and dropout
        combined = self.layer_norm(combined)
        combined = self.dropout(combined)
        
        return combined
    
    def get_user_embedding(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Get just the user embeddings."""
        return self.user_embedding(user_ids)
    
    def get_content_embedding(self, content_ids: torch.Tensor) -> torch.Tensor:
        """Get just the content embeddings."""
        return self.content_embedding(content_ids)

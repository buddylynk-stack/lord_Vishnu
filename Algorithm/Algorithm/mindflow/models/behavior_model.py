"""
MindFlow Behavior Model
The main neural network that predicts user engagement and behavior.
This is the "brain" of the recommendation system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .embeddings import EmbeddingLayer
from .attention import AttentionEncoder


class PredictionHead(nn.Module):
    """
    Multi-task prediction head for various engagement signals.
    Predicts: engagement score, click probability, watch time.
    """
    
    def __init__(self, d_model: int, num_tasks: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.num_tasks = num_tasks
        
        # Shared hidden layer
        self.shared = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Task-specific heads
        self.engagement_head = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )
        
        self.click_head = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),  # Probability output
        )
        
        self.watch_time_head = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.ReLU(),  # Watch time is non-negative
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, d_model)
        
        Returns:
            Dictionary with predictions for each task
        """
        shared_features = self.shared(x)
        
        return {
            'engagement': self.engagement_head(shared_features).squeeze(-1),
            'click_prob': self.click_head(shared_features).squeeze(-1),
            'watch_time': self.watch_time_head(shared_features).squeeze(-1),
        }


class ContentScorer(nn.Module):
    """
    Scores content items for a given user context.
    Used for ranking/recommendation.
    """
    
    def __init__(self, d_model: int, content_dim: int):
        super().__init__()
        
        # Project both to same space for scoring
        self.context_proj = nn.Linear(d_model, d_model)
        self.content_proj = nn.Linear(content_dim, d_model)
        
        # Scoring network
        self.scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
    
    def forward(
        self, 
        context: torch.Tensor, 
        content_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Score content items given user context.
        
        Args:
            context: User context (batch, d_model)
            content_embeddings: Content embeddings (batch, num_items, content_dim)
        
        Returns:
            Scores for each content item (batch, num_items)
        """
        # Project embeddings
        context_proj = self.context_proj(context).unsqueeze(1)  # (batch, 1, d_model)
        content_proj = self.content_proj(content_embeddings)    # (batch, num_items, d_model)
        
        # Expand context to match content
        context_expanded = context_proj.expand(-1, content_proj.size(1), -1)
        
        # Concatenate and score
        combined = torch.cat([context_expanded, content_proj], dim=-1)
        scores = self.scorer(combined).squeeze(-1)
        
        return scores


class MindFlowModel(nn.Module):
    """
    🧠 MindFlow - The Complete Recommendation Model
    
    A neural network that understands user behavior patterns and predicts
    engagement for content recommendations. Designed for fast CPU inference.
    
    Architecture:
    1. Embedding Layer: Converts IDs to dense vectors
    2. Attention Encoder: Captures behavior patterns
    3. Context Pooling: Creates user state representation
    4. Prediction Heads: Multi-task engagement prediction
    5. Content Scorer: Ranks content for recommendations
    """
    
    def __init__(
        self,
        num_users: int = 10000,
        num_contents: int = 50000,
        num_action_types: int = 10,
        user_embedding_dim: int = 64,
        content_embedding_dim: int = 64,
        action_embedding_dim: int = 32,
        time_embedding_dim: int = 16,
        hidden_dim: int = 256,
        num_attention_heads: int = 4,
        num_attention_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embeddings = EmbeddingLayer(
            num_users=num_users,
            num_contents=num_contents,
            num_action_types=num_action_types,
            user_embedding_dim=user_embedding_dim,
            content_embedding_dim=content_embedding_dim,
            action_embedding_dim=action_embedding_dim,
            time_embedding_dim=time_embedding_dim,
            dropout=dropout,
        )
        
        # Project embeddings to hidden dimension
        embedding_dim = self.embeddings.output_dim
        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        
        # Attention encoder
        self.encoder = AttentionEncoder(
            d_model=hidden_dim,
            num_heads=num_attention_heads,
            num_layers=num_attention_layers,
            dropout=dropout,
        )
        
        # Prediction head
        self.predictor = PredictionHead(hidden_dim, dropout=dropout)
        
        # Content scorer for recommendations
        self.content_scorer = ContentScorer(hidden_dim, content_embedding_dim)
        
        # Store config for ONNX export
        self.config = {
            'num_users': num_users,
            'num_contents': num_contents,
            'num_action_types': num_action_types,
            'hidden_dim': hidden_dim,
        }
    
    def get_user_context(
        self,
        user_ids: torch.Tensor,
        content_ids: torch.Tensor,
        action_types: torch.Tensor,
        hours: torch.Tensor,
        days: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Process user's action history to get context representation.
        
        Returns:
            context: User context vector (batch, hidden_dim)
            attention_weights: Attention weights from encoder
        """
        # Get embeddings
        embeddings = self.embeddings(
            user_ids=user_ids,
            content_ids=content_ids,
            action_types=action_types,
            hours=hours,
            days=days,
        )
        
        # Project to hidden dimension
        hidden = self.input_projection(embeddings)
        
        # Encode with attention (no mask needed for self-attention)
        encoded, attention_weights = self.encoder(hidden)
        
        # Pool to get single context vector (mean pooling)
        if mask is not None:
            # Masked mean pooling - handle different mask shapes
            if mask.dim() == 2:
                mask_expanded = mask.unsqueeze(-1).float()
            else:
                mask_expanded = mask.float()
            context = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            context = encoded.mean(dim=1)
        
        return context, attention_weights
    
    def forward(
        self,
        user_ids: torch.Tensor,
        content_ids: torch.Tensor,
        action_types: torch.Tensor,
        hours: torch.Tensor,
        days: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training - predicts engagement metrics.
        
        Args:
            user_ids: (batch,) or (batch, seq_len)
            content_ids: (batch, seq_len) 
            action_types: (batch, seq_len)
            hours: (batch, seq_len)
            days: (batch, seq_len)
            mask: Optional padding mask (batch, seq_len)
        
        Returns:
            Dictionary with engagement predictions
        """
        # Get user context
        context, attention_weights = self.get_user_context(
            user_ids, content_ids, action_types, hours, days, mask
        )
        
        # Predict engagement
        predictions = self.predictor(context)
        predictions['attention_weights'] = attention_weights
        
        return predictions
    
    def score_contents(
        self,
        user_ids: torch.Tensor,
        history_content_ids: torch.Tensor,
        history_action_types: torch.Tensor,
        history_hours: torch.Tensor,
        history_days: torch.Tensor,
        candidate_content_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Score candidate contents for recommendation.
        
        Args:
            user_ids: (batch,)
            history_*: User's past interaction history
            candidate_content_ids: Content to score (batch, num_candidates)
            mask: Mask for history sequence
        
        Returns:
            Scores for each candidate (batch, num_candidates)
        """
        # Get user context from history
        context, _ = self.get_user_context(
            user_ids, history_content_ids, history_action_types,
            history_hours, history_days, mask
        )
        
        # Get candidate content embeddings
        candidate_embeddings = self.embeddings.get_content_embedding(candidate_content_ids)
        
        # Score candidates
        scores = self.content_scorer(context, candidate_embeddings)
        
        return scores
    
    def get_recommendations(
        self,
        user_ids: torch.Tensor,
        history_content_ids: torch.Tensor,
        history_action_types: torch.Tensor,
        history_hours: torch.Tensor,
        history_days: torch.Tensor,
        candidate_content_ids: torch.Tensor,
        top_k: int = 10,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k recommendations for users.
        
        Returns:
            top_indices: Indices of top-k content (batch, top_k)
            top_scores: Scores of top-k content (batch, top_k)
        """
        scores = self.score_contents(
            user_ids, history_content_ids, history_action_types,
            history_hours, history_days, candidate_content_ids, mask
        )
        
        top_scores, top_indices = torch.topk(scores, min(top_k, scores.size(-1)), dim=-1)
        
        return top_indices, top_scores


def create_model_from_config(config) -> MindFlowModel:
    """Factory function to create model from config."""
    return MindFlowModel(
        num_users=config.num_users,
        num_contents=config.num_contents,
        num_action_types=config.num_action_types,
        user_embedding_dim=config.user_embedding_dim,
        content_embedding_dim=config.content_embedding_dim,
        action_embedding_dim=config.action_embedding_dim,
        time_embedding_dim=config.time_embedding_dim,
        hidden_dim=config.hidden_dim,
        num_attention_heads=config.num_attention_heads,
        num_attention_layers=config.num_attention_layers,
        dropout=config.dropout,
    )

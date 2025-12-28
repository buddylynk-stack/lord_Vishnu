"""
MindFlow Configuration
All hyperparameters and settings for the recommendation algorithm.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    """Neural network architecture configuration."""
    
    # Embedding dimensions
    user_embedding_dim: int = 64
    content_embedding_dim: int = 64
    action_embedding_dim: int = 32
    time_embedding_dim: int = 16
    
    # Vocabulary sizes (can be expanded based on data)
    num_users: int = 10000
    num_contents: int = 50000
    num_action_types: int = 10  # like, share, comment, view, etc.
    
    # Attention configuration
    hidden_dim: int = 256
    num_attention_heads: int = 4
    num_attention_layers: int = 2
    attention_dropout: float = 0.1
    
    # Prediction heads
    num_prediction_tasks: int = 3  # engagement, click, watch_time
    
    # Regularization
    dropout: float = 0.2
    
    @property
    def total_embedding_dim(self) -> int:
        """Total dimension after concatenating all embeddings."""
        return (self.user_embedding_dim + 
                self.content_embedding_dim + 
                self.action_embedding_dim + 
                self.time_embedding_dim)


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    
    # Basic training
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    
    # Scheduler
    use_scheduler: bool = True
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    
    # Early stopping
    early_stopping_patience: int = 10
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 5
    
    # Data
    sequence_length: int = 20  # Number of past actions to consider
    train_split: float = 0.8
    
    # Device
    device: str = "cpu"
    num_workers: int = 0  # For CPU training
    
    # Logging
    log_every_n_steps: int = 100


@dataclass
class InferenceConfig:
    """ONNX inference configuration."""
    
    # ONNX Runtime settings
    use_gpu: bool = False
    num_threads: int = 4
    
    # Caching
    cache_embeddings: bool = True
    cache_size: int = 10000
    
    # Batching
    max_batch_size: int = 128
    
    # Model paths
    onnx_model_path: str = "mindflow.onnx"


@dataclass
class DataConfig:
    """Data generation and processing configuration."""
    
    # Synthetic data generation
    num_synthetic_users: int = 1000
    num_synthetic_contents: int = 5000
    num_synthetic_interactions: int = 100000
    
    # Action types
    action_types: List[str] = field(default_factory=lambda: [
        "view", "like", "share", "comment", "save",
        "skip", "hide", "report", "follow", "unfollow"
    ])
    
    # Time features
    time_buckets: int = 24  # Hours in a day


# Default configurations
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_INFERENCE_CONFIG = InferenceConfig()
DEFAULT_DATA_CONFIG = DataConfig()

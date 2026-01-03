"""MindFlow - Intelligent Social Media Recommendation Algorithm"""

from .config import (
    ModelConfig,
    TrainingConfig, 
    InferenceConfig,
    DataConfig,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    DEFAULT_INFERENCE_CONFIG,
    DEFAULT_DATA_CONFIG,
)
from .inference import MindFlowInference

__version__ = "1.0.0"
__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "InferenceConfig", 
    "DataConfig",
    "DEFAULT_MODEL_CONFIG",
    "DEFAULT_TRAINING_CONFIG",
    "DEFAULT_INFERENCE_CONFIG",
    "DEFAULT_DATA_CONFIG",
    "MindFlowInference",
]

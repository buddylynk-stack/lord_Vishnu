"""
MindFlow Utility Functions
Helper functions for logging, metrics, and serialization.
"""

import logging
import os
import json
import time
from typing import Dict, Any, Optional
from functools import wraps

import numpy as np


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger("mindflow")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger


def timer(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"⏱️  {func.__name__} took {(end - start) * 1000:.2f}ms")
        return result
    return wrapper


class MetricsTracker:
    """Track and compute training/evaluation metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.values: Dict[str, list] = {}
    
    def update(self, metrics: Dict[str, float]):
        """Update metrics with new values."""
        for key, value in metrics.items():
            if key not in self.values:
                self.values[key] = []
            self.values[key].append(value)
    
    def compute(self) -> Dict[str, float]:
        """Compute average of all metrics."""
        return {key: np.mean(values) for key, values in self.values.items()}
    
    def get_latest(self, key: str) -> Optional[float]:
        """Get the latest value of a metric."""
        if key in self.values and self.values[key]:
            return self.values[key][-1]
        return None


def compute_ndcg(predictions: np.ndarray, targets: np.ndarray, k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain at K."""
    # Get top-k predictions
    top_k_indices = np.argsort(predictions)[::-1][:k]
    
    # DCG
    dcg = 0.0
    for i, idx in enumerate(top_k_indices):
        dcg += targets[idx] / np.log2(i + 2)
    
    # Ideal DCG
    ideal_order = np.argsort(targets)[::-1][:k]
    idcg = 0.0
    for i, idx in enumerate(ideal_order):
        idcg += targets[idx] / np.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def compute_hit_rate(predictions: np.ndarray, targets: np.ndarray, k: int = 10) -> float:
    """Compute Hit Rate at K (percentage of relevant items in top-k)."""
    top_k_indices = np.argsort(predictions)[::-1][:k]
    hits = np.sum(targets[top_k_indices] > 0)
    total_relevant = min(k, np.sum(targets > 0))
    
    if total_relevant == 0:
        return 0.0
    
    return hits / total_relevant


def save_json(data: Dict[str, Any], filepath: str):
    """Save dictionary to JSON file."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def ensure_dir(path: str):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
    else:
        return f"{seconds // 3600:.0f}h {(seconds % 3600) // 60:.0f}m"


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.should_stop

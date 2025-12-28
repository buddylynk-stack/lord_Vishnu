"""
MindFlow Production Inference API
High-performance inference for production deployment.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False


class MindFlowInference:
    """
    Production-ready inference engine for MindFlow.
    
    Usage:
        engine = MindFlowInference('mindflow.onnx')
        
        predictions = engine.predict(
            user_id=123,
            content_history=[101, 102, 103, ...],  # Last 20 content IDs
            action_history=[0, 1, 0, ...],          # Action types
            hour_history=[10, 11, 12, ...],         # Hours
            day_history=[1, 1, 1, ...],             # Days
        )
        
        print(predictions)
        # {'engagement': 0.65, 'click_prob': 0.42, 'watch_time': 25.3}
    """
    
    SEQUENCE_LENGTH = 20
    
    def __init__(
        self,
        model_path: str,
        num_threads: int = 4,
        enable_optimization: bool = True,
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to ONNX model file
            num_threads: CPU threads for inference
            enable_optimization: Enable ONNX Runtime optimizations
        """
        if not ONNXRUNTIME_AVAILABLE:
            raise ImportError("onnxruntime required: pip install onnxruntime")
        
        # Session options
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads
        
        if enable_optimization:
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Create session
        self.session = ort.InferenceSession(
            model_path,
            sess_options,
            providers=['CPUExecutionProvider'],
        )
        
        # Warmup
        self._warmup()
    
    def _warmup(self, iterations: int = 5):
        """Warmup for consistent latency."""
        dummy = {
            'user_ids': np.array([1], dtype=np.int64),
            'content_ids': np.zeros((1, self.SEQUENCE_LENGTH), dtype=np.int64),
            'action_types': np.zeros((1, self.SEQUENCE_LENGTH), dtype=np.int64),
            'hours': np.zeros((1, self.SEQUENCE_LENGTH), dtype=np.int64),
            'days': np.zeros((1, self.SEQUENCE_LENGTH), dtype=np.int64),
        }
        for _ in range(iterations):
            self.session.run(None, dummy)
    
    def _pad_sequence(self, seq: List[int], default: int = 0) -> np.ndarray:
        """Pad or truncate sequence to SEQUENCE_LENGTH."""
        arr = np.full(self.SEQUENCE_LENGTH, default, dtype=np.int64)
        seq = seq[-self.SEQUENCE_LENGTH:]  # Take last N items
        arr[-len(seq):] = seq
        return arr
    
    def predict(
        self,
        user_id: int,
        content_history: List[int],
        action_history: List[int],
        hour_history: List[int],
        day_history: List[int],
    ) -> Dict[str, float]:
        """
        Get predictions for a single user.
        
        Args:
            user_id: User ID
            content_history: List of recent content IDs (max 20)
            action_history: List of action types (0-9)
            hour_history: List of hours (0-23)
            day_history: List of days (0-6)
        
        Returns:
            Dictionary with engagement, click_prob, watch_time
        """
        inputs = {
            'user_ids': np.array([user_id], dtype=np.int64),
            'content_ids': self._pad_sequence(content_history).reshape(1, -1),
            'action_types': self._pad_sequence(action_history).reshape(1, -1),
            'hours': self._pad_sequence(hour_history).reshape(1, -1),
            'days': self._pad_sequence(day_history).reshape(1, -1),
        }
        
        outputs = self.session.run(None, inputs)
        
        return {
            'engagement': float(outputs[0][0]),
            'click_prob': float(outputs[1][0]),
            'watch_time': float(outputs[2][0]),
        }
    
    def predict_batch(
        self,
        user_ids: np.ndarray,
        content_ids: np.ndarray,
        action_types: np.ndarray,
        hours: np.ndarray,
        days: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Batch prediction for multiple users.
        
        Args:
            All arrays should have shape (batch_size,) or (batch_size, 20)
        
        Returns:
            Dictionary with arrays for each prediction
        """
        inputs = {
            'user_ids': user_ids.astype(np.int64),
            'content_ids': content_ids.astype(np.int64),
            'action_types': action_types.astype(np.int64),
            'hours': hours.astype(np.int64),
            'days': days.astype(np.int64),
        }
        
        outputs = self.session.run(None, inputs)
        
        return {
            'engagement': outputs[0],
            'click_prob': outputs[1],
            'watch_time': outputs[2],
        }
    
    def benchmark(self, iterations: int = 1000) -> Dict[str, float]:
        """
        Benchmark inference performance.
        
        Returns:
            Dictionary with timing statistics
        """
        dummy = {
            'user_ids': np.array([1], dtype=np.int64),
            'content_ids': np.random.randint(1, 1000, (1, 20)).astype(np.int64),
            'action_types': np.random.randint(0, 10, (1, 20)).astype(np.int64),
            'hours': np.random.randint(0, 24, (1, 20)).astype(np.int64),
            'days': np.random.randint(0, 7, (1, 20)).astype(np.int64),
        }
        
        # Warmup
        for _ in range(10):
            self.session.run(None, dummy)
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.session.run(None, dummy)
            times.append((time.perf_counter() - start) * 1000)
        
        times = np.array(times)
        
        return {
            'mean_ms': float(np.mean(times)),
            'p50_ms': float(np.percentile(times, 50)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)),
            'throughput_per_sec': float(1000 / np.mean(times)),
        }

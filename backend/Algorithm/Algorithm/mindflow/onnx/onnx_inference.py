"""
ONNX Inference Engine for MindFlow
Fast inference using ONNX Runtime optimized for CPU.
"""

import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from functools import lru_cache

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False


class ONNXInferenceEngine:
    """
    High-performance ONNX inference engine.
    
    Features:
    - Session management with optimizations
    - Batch prediction
    - Caching for embeddings
    - Warm-up for consistent latency
    """
    
    def __init__(
        self,
        model_path: str,
        num_threads: int = 4,
        use_gpu: bool = False,
        enable_profiling: bool = False,
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to ONNX model
            num_threads: Number of CPU threads
            use_gpu: Use GPU if available
            enable_profiling: Enable performance profiling
        """
        if not ONNXRUNTIME_AVAILABLE:
            raise ImportError("onnxruntime not installed. Run: pip install onnxruntime")
        
        self.model_path = model_path
        self.num_threads = num_threads
        
        # Session options
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        if enable_profiling:
            sess_options.enable_profiling = True
        
        # Providers
        providers = ['CPUExecutionProvider']
        if use_gpu:
            providers.insert(0, 'CUDAExecutionProvider')
        
        # Create session
        self.session = ort.InferenceSession(
            model_path,
            sess_options,
            providers=providers,
        )
        
        # Get input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        # Warm-up
        self._warm_up()
        
        print(f"🚀 ONNX Inference Engine initialized")
        print(f"   Model: {os.path.basename(model_path)}")
        print(f"   Threads: {num_threads}")
        print(f"   Provider: {self.session.get_providers()[0]}")
    
    def _warm_up(self, iterations: int = 3):
        """Warm up the model for consistent latency."""
        dummy_inputs = {
            'user_ids': np.array([1], dtype=np.int64),
            'content_ids': np.random.randint(1, 100, (1, 20), dtype=np.int64),
            'action_types': np.random.randint(0, 10, (1, 20), dtype=np.int64),
            'hours': np.random.randint(0, 24, (1, 20), dtype=np.int64),
            'days': np.random.randint(0, 7, (1, 20), dtype=np.int64),
        }
        
        for _ in range(iterations):
            self.session.run(None, dummy_inputs)
    
    def predict(
        self,
        user_ids: np.ndarray,
        content_ids: np.ndarray,
        action_types: np.ndarray,
        hours: np.ndarray,
        days: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Run prediction on input data.
        
        Args:
            user_ids: (batch,) user IDs
            content_ids: (batch, seq_len) content IDs
            action_types: (batch, seq_len) action types
            hours: (batch, seq_len) hours
            days: (batch, seq_len) days
        
        Returns:
            Dictionary with predictions
        """
        # Ensure correct dtypes
        inputs = {
            'user_ids': user_ids.astype(np.int64),
            'content_ids': content_ids.astype(np.int64),
            'action_types': action_types.astype(np.int64),
            'hours': hours.astype(np.int64),
            'days': days.astype(np.int64),
        }
        
        # Run inference
        outputs = self.session.run(None, inputs)
        
        return {
            'engagement': outputs[0],
            'click_prob': outputs[1],
            'watch_time': outputs[2],
        }
    
    def predict_single(
        self,
        user_id: int,
        content_ids: List[int],
        action_types: List[int],
        hours: List[int],
        days: List[int],
    ) -> Dict[str, float]:
        """
        Predict for a single user.
        
        Returns:
            Dictionary with scalar predictions
        """
        result = self.predict(
            user_ids=np.array([user_id]),
            content_ids=np.array([content_ids]),
            action_types=np.array([action_types]),
            hours=np.array([hours]),
            days=np.array([days]),
        )
        
        return {
            'engagement': float(result['engagement'][0]),
            'click_prob': float(result['click_prob'][0]),
            'watch_time': float(result['watch_time'][0]),
        }
    
    def benchmark(
        self,
        batch_size: int = 1,
        sequence_length: int = 20,
        iterations: int = 100,
        warmup: int = 10,
    ) -> Dict[str, float]:
        """
        Benchmark inference performance.
        
        Returns:
            Dictionary with timing statistics
        """
        # Create test inputs
        inputs = {
            'user_ids': np.random.randint(1, 1000, (batch_size,), dtype=np.int64),
            'content_ids': np.random.randint(1, 5000, (batch_size, sequence_length), dtype=np.int64),
            'action_types': np.random.randint(0, 10, (batch_size, sequence_length), dtype=np.int64),
            'hours': np.random.randint(0, 24, (batch_size, sequence_length), dtype=np.int64),
            'days': np.random.randint(0, 7, (batch_size, sequence_length), dtype=np.int64),
        }
        
        # Warm-up
        for _ in range(warmup):
            self.session.run(None, inputs)
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.session.run(None, inputs)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        times = np.array(times)
        
        results = {
            'batch_size': batch_size,
            'iterations': iterations,
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'p50_ms': float(np.percentile(times, 50)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)),
            'throughput_per_sec': float(1000 / np.mean(times) * batch_size),
        }
        
        return results
    
    def print_benchmark(self, **kwargs):
        """Run and print benchmark results."""
        results = self.benchmark(**kwargs)
        
        print(f"\n📊 Benchmark Results (batch={results['batch_size']}, n={results['iterations']})")
        print(f"   Mean:    {results['mean_ms']:.3f} ms")
        print(f"   Std:     {results['std_ms']:.3f} ms")
        print(f"   P50:     {results['p50_ms']:.3f} ms")
        print(f"   P95:     {results['p95_ms']:.3f} ms")
        print(f"   P99:     {results['p99_ms']:.3f} ms")
        print(f"   Throughput: {results['throughput_per_sec']:.0f} samples/sec")
        
        return results


class CachedInferenceEngine(ONNXInferenceEngine):
    """
    Inference engine with LRU caching for repeated predictions.
    Useful when the same user context is queried multiple times.
    """
    
    def __init__(self, model_path: str, cache_size: int = 1000, **kwargs):
        super().__init__(model_path, **kwargs)
        self.cache_size = cache_size
        self._cache: Dict[str, Dict[str, float]] = {}
    
    def _make_cache_key(
        self,
        user_id: int,
        content_ids: tuple,
        action_types: tuple,
    ) -> str:
        """Create cache key from inputs."""
        return f"{user_id}_{hash(content_ids)}_{hash(action_types)}"
    
    def predict_cached(
        self,
        user_id: int,
        content_ids: List[int],
        action_types: List[int],
        hours: List[int],
        days: List[int],
    ) -> Dict[str, float]:
        """Predict with caching."""
        cache_key = self._make_cache_key(
            user_id, 
            tuple(content_ids), 
            tuple(action_types)
        )
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        result = self.predict_single(
            user_id, content_ids, action_types, hours, days
        )
        
        # Cache with LRU eviction
        if len(self._cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[cache_key] = result
        return result
    
    def clear_cache(self):
        """Clear the prediction cache."""
        self._cache.clear()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ONNX Inference Benchmark")
    parser.add_argument("--model", type=str, default="mindflow.onnx")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    
    args = parser.parse_args()
    
    if args.benchmark:
        engine = ONNXInferenceEngine(args.model)
        engine.print_benchmark(
            batch_size=args.batch_size,
            iterations=args.iterations,
        )

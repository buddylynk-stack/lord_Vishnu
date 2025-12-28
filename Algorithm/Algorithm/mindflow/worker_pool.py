"""
MindFlow Worker Pool
Multi-process inference for maximum CPU utilization.
"""

import os
import time
import multiprocessing as mp
from typing import List, Dict, Optional, Callable
from queue import Empty
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class InferenceWorker:
    """Single inference worker process."""
    
    def __init__(self, model_path: str, worker_id: int):
        self.model_path = model_path
        self.worker_id = worker_id
        self.session = None
    
    def initialize(self):
        """Initialize ONNX session (called in worker process)."""
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options,
            providers=['CPUExecutionProvider'],
        )
        
        # Warmup
        dummy = {
            'user_ids': np.array([1], dtype=np.int64),
            'content_ids': np.ones((1, 20), dtype=np.int64),
            'action_types': np.zeros((1, 20), dtype=np.int64),
            'hours': np.zeros((1, 20), dtype=np.int64),
            'days': np.zeros((1, 20), dtype=np.int64),
        }
        for _ in range(3):
            self.session.run(None, dummy)
    
    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run inference."""
        outputs = self.session.run(None, inputs)
        return {
            'engagement': outputs[0],
            'click_prob': outputs[1],
            'watch_time': outputs[2],
        }


def worker_process(
    model_path: str,
    worker_id: int,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    shutdown_event: mp.Event,
):
    """Worker process main loop."""
    worker = InferenceWorker(model_path, worker_id)
    worker.initialize()
    
    while not shutdown_event.is_set():
        try:
            request_id, inputs = input_queue.get(timeout=0.1)
            start = time.perf_counter()
            result = worker.predict(inputs)
            latency = (time.perf_counter() - start) * 1000
            output_queue.put((request_id, result, latency, worker_id))
        except Empty:
            continue
        except Exception as e:
            output_queue.put((request_id, {'error': str(e)}, 0, worker_id))


class WorkerPool:
    """
    Multi-process worker pool for parallel inference.
    
    Usage:
        pool = WorkerPool('mindflow.onnx', num_workers=4)
        pool.start()
        
        results = pool.predict_batch([...])
        
        pool.shutdown()
    """
    
    def __init__(self, model_path: str, num_workers: int = None):
        self.model_path = model_path
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)
        
        self.workers: List[mp.Process] = []
        self.input_queue: Optional[mp.Queue] = None
        self.output_queue: Optional[mp.Queue] = None
        self.shutdown_event: Optional[mp.Event] = None
        
        self._request_counter = 0
        self._running = False
    
    def start(self):
        """Start worker processes."""
        if self._running:
            return
        
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.shutdown_event = mp.Event()
        
        for i in range(self.num_workers):
            p = mp.Process(
                target=worker_process,
                args=(
                    self.model_path,
                    i,
                    self.input_queue,
                    self.output_queue,
                    self.shutdown_event,
                ),
            )
            p.start()
            self.workers.append(p)
        
        self._running = True
        print(f"🚀 Started {self.num_workers} inference workers")
    
    def shutdown(self):
        """Shutdown all workers."""
        if not self._running:
            return
        
        self.shutdown_event.set()
        
        for p in self.workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        
        self.workers.clear()
        self._running = False
        print("👋 Worker pool shutdown")
    
    def predict_batch(
        self,
        requests: List[Dict[str, np.ndarray]],
        timeout: float = 30.0,
    ) -> List[Dict]:
        """
        Distribute predictions across workers.
        
        Args:
            requests: List of input dictionaries
            timeout: Max time to wait for all results
        
        Returns:
            List of result dictionaries
        """
        if not self._running:
            raise RuntimeError("Worker pool not started")
        
        # Submit all requests
        request_ids = []
        for inputs in requests:
            request_id = self._request_counter
            self._request_counter += 1
            request_ids.append(request_id)
            self.input_queue.put((request_id, inputs))
        
        # Collect results
        results = {}
        start = time.time()
        
        while len(results) < len(requests):
            if time.time() - start > timeout:
                raise TimeoutError(f"Timeout waiting for results")
            
            try:
                request_id, result, latency, worker_id = self.output_queue.get(timeout=0.5)
                result['latency_ms'] = latency
                result['worker_id'] = worker_id
                results[request_id] = result
            except Empty:
                continue
        
        # Return in original order
        return [results[rid] for rid in request_ids]
    
    def benchmark(self, num_requests: int = 1000) -> Dict:
        """Benchmark worker pool performance."""
        # Create test inputs
        requests = []
        for _ in range(num_requests):
            requests.append({
                'user_ids': np.array([1], dtype=np.int64),
                'content_ids': np.random.randint(1, 1000, (1, 20)).astype(np.int64),
                'action_types': np.random.randint(0, 10, (1, 20)).astype(np.int64),
                'hours': np.random.randint(0, 24, (1, 20)).astype(np.int64),
                'days': np.random.randint(0, 7, (1, 20)).astype(np.int64),
            })
        
        # Run benchmark
        start = time.perf_counter()
        results = self.predict_batch(requests)
        total_time = time.perf_counter() - start
        
        latencies = [r['latency_ms'] for r in results]
        
        return {
            'num_requests': num_requests,
            'num_workers': self.num_workers,
            'total_time_s': total_time,
            'throughput_per_sec': num_requests / total_time,
            'avg_latency_ms': np.mean(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Worker Pool Benchmark")
    parser.add_argument('--model', type=str, default='mindflow.onnx')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--requests', type=int, default=1000)
    
    args = parser.parse_args()
    
    pool = WorkerPool(args.model, num_workers=args.workers)
    pool.start()
    
    print(f"\n📊 Benchmarking with {args.requests} requests...")
    results = pool.benchmark(args.requests)
    
    print(f"\n⚡ Results:")
    print(f"   Workers: {results['num_workers']}")
    print(f"   Throughput: {results['throughput_per_sec']:.0f} req/sec")
    print(f"   Avg Latency: {results['avg_latency_ms']:.2f} ms")
    print(f"   P95 Latency: {results['p95_latency_ms']:.2f} ms")
    
    pool.shutdown()

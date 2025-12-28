#!/usr/bin/env python3
"""
MindFlow Inference Demo
Demonstrates the trained model in production mode.
"""

import argparse
import os
import sys
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_demo(model_path: str):
    """Run production demo with ONNX model."""
    import onnxruntime as ort
    
    print(f"📦 Loading model: {model_path}")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    print("✅ Model loaded!")
    
    # Simulate user behavior history
    print("\n📱 Simulating user session...")
    
    # User's recent 20 interactions
    user_input = {
        'user_ids': np.array([1], dtype=np.int64),
        'content_ids': np.array([[
            101, 205, 150, 300, 102, 210, 155, 400, 103, 220,
            160, 350, 104, 230, 165, 500, 105, 240, 170, 450
        ]], dtype=np.int64),
        'action_types': np.array([[
            0, 1, 1, 0, 0, 1, 2, 0, 0, 1,  # 0=view, 1=like, 2=share
            1, 0, 0, 1, 0, 0, 0, 1, 1, 0
        ]], dtype=np.int64),
        'hours': np.array([[
            9, 9, 10, 10, 12, 12, 14, 14, 18, 18,
            19, 19, 20, 20, 21, 21, 22, 22, 23, 23
        ]], dtype=np.int64),
        'days': np.array([[
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        ]], dtype=np.int64),
    }
    
    # Run inference
    outputs = session.run(None, user_input)
    
    print("\n📊 Model Predictions:")
    print(f"   ├─ Engagement Score:  {outputs[0][0]*100:.1f}%")
    print(f"   ├─ Click Probability: {outputs[1][0]*100:.1f}%")
    print(f"   └─ Watch Time:        {outputs[2][0]:.1f}s")
    
    # Benchmark
    print("\n⚡ Performance Benchmark:")
    
    # Warmup
    for _ in range(10):
        session.run(None, user_input)
    
    # Benchmark single prediction
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        session.run(None, user_input)
        times.append((time.perf_counter() - start) * 1000)
    
    print(f"   ├─ Latency (avg):  {np.mean(times):.2f} ms")
    print(f"   ├─ Latency (p95):  {np.percentile(times, 95):.2f} ms")
    print(f"   ├─ Latency (p99):  {np.percentile(times, 99):.2f} ms")
    print(f"   └─ Throughput:     {1000/np.mean(times):.0f} predictions/sec")
    
    # Batch benchmark
    print("\n📈 Batch Performance:")
    for batch_size in [1, 8, 32, 64]:
        batch_input = {
            'user_ids': np.ones(batch_size, dtype=np.int64),
            'content_ids': np.random.randint(1, 1000, (batch_size, 20)).astype(np.int64),
            'action_types': np.random.randint(0, 10, (batch_size, 20)).astype(np.int64),
            'hours': np.random.randint(0, 24, (batch_size, 20)).astype(np.int64),
            'days': np.random.randint(0, 7, (batch_size, 20)).astype(np.int64),
        }
        
        # Warmup
        for _ in range(5):
            session.run(None, batch_input)
        
        times = []
        for _ in range(100):
            start = time.perf_counter()
            session.run(None, batch_input)
            times.append((time.perf_counter() - start) * 1000)
        
        throughput = batch_size * 1000 / np.mean(times)
        print(f"   Batch {batch_size:2}: {np.mean(times):.2f}ms ({throughput:.0f}/sec)")


def main():
    parser = argparse.ArgumentParser(description="🧠 MindFlow Production Demo")
    parser.add_argument('--model', type=str, default='mindflow.onnx', help='ONNX model path')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧠 MindFlow Production Demo")
    print("=" * 60)
    
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        print("\n📖 To create a model:")
        print("   1. python train.py --epochs 50")
        print("   2. python export_onnx.py --checkpoint models/best_model.pt")
        sys.exit(1)
    
    run_demo(args.model)
    
    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

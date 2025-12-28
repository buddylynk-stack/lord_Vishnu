#!/usr/bin/env python3
"""
MindFlow ONNX Export Script
Export trained models to production-ready ONNX format.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from mindflow.models.behavior_model import MindFlowModel


class ONNXExportWrapper(torch.nn.Module):
    """Wrapper for clean ONNX export."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, user_ids, content_ids, action_types, hours, days):
        outputs = self.model(
            user_ids=user_ids,
            content_ids=content_ids,
            action_types=action_types,
            hours=hours,
            days=days,
        )
        return outputs['engagement'], outputs['click_prob'], outputs['watch_time']


def export_model(checkpoint_path: str, output_path: str, validate: bool = True):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        checkpoint_path: Path to PyTorch checkpoint (.pt file)
        output_path: Output path for ONNX model
        validate: Whether to validate the exported model
    """
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', {})
    
    print(f"   Config: {config}")
    
    # Create and load model
    model = MindFlowModel(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Wrap for export
    wrapper = ONNXExportWrapper(model)
    wrapper.eval()
    
    # Create dummy inputs
    seq_len = 20
    dummy = (
        torch.tensor([1]),
        torch.randint(1, 100, (1, seq_len)),
        torch.randint(0, 10, (1, seq_len)),
        torch.randint(0, 24, (1, seq_len)),
        torch.randint(0, 7, (1, seq_len)),
    )
    
    # Export
    print(f"🔄 Exporting to ONNX...")
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        input_names=['user_ids', 'content_ids', 'action_types', 'hours', 'days'],
        output_names=['engagement', 'click_prob', 'watch_time'],
        dynamic_axes={
            'user_ids': {0: 'batch'},
            'content_ids': {0: 'batch'},
            'action_types': {0: 'batch'},
            'hours': {0: 'batch'},
            'days': {0: 'batch'},
            'engagement': {0: 'batch'},
            'click_prob': {0: 'batch'},
            'watch_time': {0: 'batch'},
        },
        opset_version=14,
    )
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Exported to: {output_path}")
    print(f"📦 Model size: {size_mb:.2f} MB")
    
    # Validate
    if validate:
        print(f"🔍 Validating...")
        try:
            import onnx
            model_onnx = onnx.load(output_path)
            onnx.checker.check_model(model_onnx)
            print("✅ ONNX validation passed!")
        except Exception as e:
            print(f"⚠️ Validation warning: {e}")
        
        # Test inference
        try:
            import onnxruntime as ort
            import numpy as np
            import time
            
            session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
            
            test_input = {
                'user_ids': np.array([1], dtype=np.int64),
                'content_ids': np.random.randint(1, 100, (1, seq_len)).astype(np.int64),
                'action_types': np.random.randint(0, 10, (1, seq_len)).astype(np.int64),
                'hours': np.random.randint(0, 24, (1, seq_len)).astype(np.int64),
                'days': np.random.randint(0, 7, (1, seq_len)).astype(np.int64),
            }
            
            # Warmup and benchmark
            for _ in range(3):
                session.run(None, test_input)
            
            times = []
            for _ in range(100):
                start = time.perf_counter()
                session.run(None, test_input)
                times.append((time.perf_counter() - start) * 1000)
            
            avg_time = np.mean(times)
            print(f"⚡ Inference: {avg_time:.2f}ms ({1000/avg_time:.0f} predictions/sec)")
            
        except ImportError:
            print("⚠️ Install onnxruntime for inference testing")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="📦 Export MindFlow to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python export_onnx.py --checkpoint models/model.pt
  python export_onnx.py --checkpoint models/model.pt --output production/mindflow.onnx
        """
    )
    
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, default='mindflow.onnx', help='Output ONNX path')
    parser.add_argument('--no-validate', action='store_true', help='Skip validation')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("📦 MindFlow ONNX Export")
    print("=" * 60)
    
    export_model(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        validate=not args.no_validate,
    )
    
    print("\n" + "=" * 60)
    print("✅ Export complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

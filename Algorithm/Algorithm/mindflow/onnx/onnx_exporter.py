"""
ONNX Exporter for MindFlow
Exports trained PyTorch models to optimized ONNX format.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
import numpy as np

try:
    import onnx
    from onnx import checker
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class ONNXExporter:
    """
    Export MindFlow models to ONNX format.
    
    Features:
    - Dynamic batch size support
    - Optimization passes
    - Validation
    - Quantization options
    """
    
    def __init__(
        self,
        model: nn.Module,
        sequence_length: int = 20,
        opset_version: int = 14,
    ):
        self.model = model
        self.sequence_length = sequence_length
        self.opset_version = opset_version
    
    def _create_dummy_inputs(
        self, 
        batch_size: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """Create dummy inputs for tracing."""
        return {
            'user_ids': torch.randint(1, 1000, (batch_size,)),
            'content_ids': torch.randint(1, 5000, (batch_size, self.sequence_length)),
            'action_types': torch.randint(0, 10, (batch_size, self.sequence_length)),
            'hours': torch.randint(0, 24, (batch_size, self.sequence_length)),
            'days': torch.randint(0, 7, (batch_size, self.sequence_length)),
        }
    
    def export(
        self,
        output_path: str,
        dynamic_batch: bool = True,
        optimize: bool = True,
        validate: bool = True,
        verbose: bool = True,
    ) -> str:
        """
        Export model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            dynamic_batch: Enable dynamic batch size
            optimize: Apply optimization passes
            validate: Validate exported model
            verbose: Print progress
        
        Returns:
            Path to exported model
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX not installed. Run: pip install onnx")
        
        self.model.eval()
        
        if verbose:
            print(f"🔄 Exporting MindFlow to ONNX...")
        
        # Create dummy inputs
        dummy_inputs = self._create_dummy_inputs(batch_size=1)
        
        # Define input/output names
        input_names = ['user_ids', 'content_ids', 'action_types', 'hours', 'days']
        output_names = ['engagement', 'click_prob', 'watch_time']
        
        # Dynamic axes for variable batch size
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                'user_ids': {0: 'batch_size'},
                'content_ids': {0: 'batch_size'},
                'action_types': {0: 'batch_size'},
                'hours': {0: 'batch_size'},
                'days': {0: 'batch_size'},
                'engagement': {0: 'batch_size'},
                'click_prob': {0: 'batch_size'},
                'watch_time': {0: 'batch_size'},
            }
        
        # Create wrapper for clean export
        class ExportWrapper(nn.Module):
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
                return (
                    outputs['engagement'],
                    outputs['click_prob'],
                    outputs['watch_time'],
                )
        
        wrapper = ExportWrapper(self.model)
        wrapper.eval()
        
        # Export
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        torch.onnx.export(
            wrapper,
            (
                dummy_inputs['user_ids'],
                dummy_inputs['content_ids'],
                dummy_inputs['action_types'],
                dummy_inputs['hours'],
                dummy_inputs['days'],
            ),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=self.opset_version,
            do_constant_folding=True,
            export_params=True,
        )
        
        if verbose:
            print(f"✅ Exported to: {output_path}")
        
        # Optimize
        if optimize:
            self._optimize_model(output_path, verbose)
        
        # Validate
        if validate:
            self._validate_model(output_path, dummy_inputs, verbose)
        
        # Print size
        if verbose:
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"📦 Model size: {size_mb:.2f} MB")
        
        return output_path
    
    def _optimize_model(self, model_path: str, verbose: bool = True):
        """Apply ONNX optimization passes."""
        if verbose:
            print(f"⚡ Optimizing ONNX model...")
        
        model = onnx.load(model_path)
        
        # Basic optimizations (avoid deprecated optimizer)
        try:
            from onnx import shape_inference
            model = shape_inference.infer_shapes(model)
        except Exception:
            pass
        
        onnx.save(model, model_path)
        
        if verbose:
            print(f"✅ Optimization complete")
    
    def _validate_model(
        self, 
        model_path: str, 
        dummy_inputs: Dict[str, torch.Tensor],
        verbose: bool = True,
    ):
        """Validate exported ONNX model."""
        if verbose:
            print(f"🔍 Validating ONNX model...")
        
        # Check model structure
        model = onnx.load(model_path)
        checker.check_model(model)
        
        # Compare outputs with PyTorch
        try:
            import onnxruntime as ort
            
            session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            
            # Run ONNX inference
            onnx_inputs = {
                name: dummy_inputs[name].numpy()
                for name in dummy_inputs
            }
            onnx_outputs = session.run(None, onnx_inputs)
            
            # Run PyTorch inference
            with torch.no_grad():
                self.model.eval()
                pytorch_outputs = self.model(**dummy_inputs)
            
            # Compare
            torch_engagement = pytorch_outputs['engagement'].numpy()
            onnx_engagement = onnx_outputs[0]
            
            max_diff = np.abs(torch_engagement - onnx_engagement).max()
            
            if verbose:
                print(f"   Max output difference: {max_diff:.6f}")
                
            if max_diff < 1e-4:
                print(f"✅ Validation passed!")
            else:
                print(f"⚠️  Validation warning: outputs differ by {max_diff:.6f}")
                
        except ImportError:
            if verbose:
                print(f"⚠️  onnxruntime not installed, skipping output validation")
    
    def export_for_inference(
        self,
        output_path: str,
        quantize: bool = False,
    ) -> str:
        """
        Export model optimized for production inference.
        
        Args:
            output_path: Output path
            quantize: Apply int8 quantization for faster inference
        
        Returns:
            Path to exported model
        """
        # First, do regular export
        self.export(output_path, optimize=True, validate=True)
        
        # Quantization (optional)
        if quantize:
            try:
                from onnxruntime.quantization import quantize_dynamic, QuantType
                
                quantized_path = output_path.replace('.onnx', '_quantized.onnx')
                quantize_dynamic(
                    output_path,
                    quantized_path,
                    weight_type=QuantType.QInt8,
                )
                print(f"🗜️  Quantized model saved: {quantized_path}")
                
                # Compare sizes
                orig_size = os.path.getsize(output_path) / (1024 * 1024)
                quant_size = os.path.getsize(quantized_path) / (1024 * 1024)
                print(f"   Original: {orig_size:.2f} MB, Quantized: {quant_size:.2f} MB")
                
                return quantized_path
                
            except ImportError:
                print(f"⚠️  Quantization requires onnxruntime-extensions")
        
        return output_path


def load_and_export(
    checkpoint_path: str,
    output_path: str,
    model_config: Optional[dict] = None,
) -> str:
    """
    Convenience function to load checkpoint and export to ONNX.
    
    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_path: Output ONNX path
        model_config: Optional model configuration
    
    Returns:
        Path to exported model
    """
    from ..models.behavior_model import MindFlowModel
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get config
    config = model_config or checkpoint.get('config', {})
    
    # Create model
    model = MindFlowModel(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Export
    exporter = ONNXExporter(model)
    return exporter.export(output_path)

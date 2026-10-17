"""Model optimization utilities for edge-aware tracker deployment.

This sub-package provides tools to compress and accelerate deep-learning
trackers so they can be evaluated under realistic edge-device constraints.

Components
----------
quantization
    Post-training quantization (INT8 / FP16) via ONNX Runtime.
pruning
    Magnitude-based structured pruning for PyTorch Conv2d backbones.
onnx_export
    Export PyTorch tracker modules to ONNX format with shape validation.

Quick start::

    from eovot.optimization import ModelQuantizer, QuantizationConfig
    from eovot.optimization import MagnitudePruner, PruningConfig
    from eovot.optimization import ONNXExporter, ONNXExportConfig

Requires optional backends:
    - ``onnxruntime`` (quantization, inference benchmarking, ONNX export validation)
    - ``torch``       (pruning, ONNX export)
"""

from .quantization import QuantizationConfig, ModelQuantizer, QuantizationBenchmark
from .pruning import PruningConfig, MagnitudePruner, PruningAnalyzer
from .onnx_export import ONNXExportConfig, ONNXExporter

__all__ = [
    "QuantizationConfig",
    "ModelQuantizer",
    "QuantizationBenchmark",
    "PruningConfig",
    "MagnitudePruner",
    "PruningAnalyzer",
    "ONNXExportConfig",
    "ONNXExporter",
]

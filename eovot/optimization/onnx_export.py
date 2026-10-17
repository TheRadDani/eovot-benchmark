"""ONNX export pipeline for PyTorch-based EOVOT trackers.

Converts a PyTorch ``nn.Module`` to ONNX format with:
  - Dynamic batch / spatial-size axes for flexible deployment
  - Shape validation via ONNX checker and a dry-run ONNX Runtime session
  - Optional graph simplification via ``onnxsim``
  - A structured export report covering file size, IR version, and
    node counts

Typical usage
-------------
::

    import torch
    from eovot.optimization.onnx_export import ONNXExporter, ONNXExportConfig

    backbone = MyTrackerBackbone()
    backbone.eval()

    cfg = ONNXExportConfig(
        input_shape=(1, 3, 128, 128),
        dynamic_axes={"input": {0: "batch", 2: "H", 3: "W"}},
        opset=17,
    )
    exporter = ONNXExporter(cfg)
    report = exporter.export(backbone, "tracker_backbone.onnx")
    print(report)

Requires ``torch`` (and ``onnx`` / ``onnxruntime`` for validation).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ONNXExportConfig:
    """Parameters for the ONNX export pipeline.

    Attributes
    ----------
    input_shape:
        Shape of the dummy input tensor used for tracing, e.g.
        ``(1, 3, 128, 128)`` for a batch-1 RGB image.
    input_names:
        Names to assign to the ONNX graph input nodes.
    output_names:
        Names to assign to the ONNX graph output nodes.
    dynamic_axes:
        Mapping of node name → {axis_index: axis_name} for dynamic
        dimensions.  Example: ``{"input": {0: "batch", 2: "H", 3: "W"}}``.
    opset:
        ONNX opset version.  Opset 17 is recommended for broad
        compatibility with ONNX Runtime ≥ 1.14 and TensorRT ≥ 8.5.
    simplify:
        Run ``onnxsim`` on the exported graph to fold constants and
        remove redundant ops.  Requires ``onnxsim`` to be installed.
    validate:
        Check the exported model with ``onnx.checker`` and a dry-run
        ONNX Runtime session on a random input.  Strongly recommended.
    """

    input_shape: Tuple[int, ...] = (1, 3, 128, 128)
    input_names: List[str] = field(default_factory=lambda: ["input"])
    output_names: List[str] = field(default_factory=lambda: ["output"])
    dynamic_axes: Optional[Dict] = None
    opset: int = 17
    simplify: bool = False
    validate: bool = True

    def __post_init__(self) -> None:
        if len(self.input_shape) < 2:
            raise ValueError("input_shape must have at least 2 dimensions")
        if self.opset < 11:
            raise ValueError("opset must be >= 11; many modern ops require at least opset 11")


@dataclass
class ExportReport:
    """Structured summary of an ONNX export operation.

    Attributes
    ----------
    output_path:      Path of the exported ONNX file.
    export_time_s:    Wall-clock seconds for the torch.onnx.export call.
    file_size_mb:     File size of the exported model in megabytes.
    onnx_ir_version:  ONNX IR version embedded in the protobuf.
    opset_version:    ONNX opset the model was exported at.
    num_nodes:        Total node count in the ONNX graph.
    input_shape:      Input tensor shape used during tracing.
    validation_ok:    True if validate=True and all checks passed.
    simplification_applied: True if onnxsim was successfully run.
    """

    output_path: str
    export_time_s: float
    file_size_mb: float
    onnx_ir_version: int
    opset_version: int
    num_nodes: int
    input_shape: Tuple[int, ...]
    validation_ok: bool = False
    simplification_applied: bool = False

    def to_dict(self) -> Dict:
        return {
            "output_path": self.output_path,
            "export_time_s": round(self.export_time_s, 3),
            "file_size_mb": round(self.file_size_mb, 3),
            "onnx_ir_version": self.onnx_ir_version,
            "opset_version": self.opset_version,
            "num_nodes": self.num_nodes,
            "input_shape": list(self.input_shape),
            "validation_ok": self.validation_ok,
            "simplification_applied": self.simplification_applied,
        }

    def __str__(self) -> str:
        status = "OK" if self.validation_ok else "SKIPPED/FAILED"
        return (
            f"ExportReport({os.path.basename(self.output_path)} "
            f"{self.file_size_mb:.2f} MB, "
            f"opset={self.opset_version}, "
            f"nodes={self.num_nodes}, "
            f"validation={status}, "
            f"export_time={self.export_time_s:.2f}s)"
        )


class ONNXExporter:
    """Export a PyTorch module to ONNX with optional validation and simplification.

    Parameters
    ----------
    config:
        Export configuration.  Defaults to standard settings suitable
        for a single-input, single-output feature extractor.
    """

    def __init__(self, config: Optional[ONNXExportConfig] = None) -> None:
        self.config = config or ONNXExportConfig()
        self._torch_available = self._probe_torch()
        self._onnx_available = self._probe_onnx()
        self._ort_available = self._probe_ort()
        self._onnxsim_available = self._probe_onnxsim()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(self, model: "torch.nn.Module", output_path: str) -> ExportReport:
        """Export *model* to ONNX and return an :class:`ExportReport`.

        Parameters
        ----------
        model:       ``nn.Module`` in evaluation mode.  The caller is
                     responsible for calling ``model.eval()`` beforehand.
        output_path: Destination ``.onnx`` file path.

        Returns
        -------
        :class:`ExportReport` with size, node count, and validation result.

        Raises
        ------
        ImportError
            If ``torch`` is not installed.
        RuntimeError
            If ``config.validate=True`` and the exported model fails
            ``onnx.checker`` or ONNX Runtime validation.
        """
        self._require_torch()
        import torch

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        dummy = torch.randn(*self.config.input_shape)

        t0 = time.perf_counter()
        torch.onnx.export(
            model,
            dummy,
            output_path,
            input_names=self.config.input_names,
            output_names=self.config.output_names,
            dynamic_axes=self.config.dynamic_axes,
            opset_version=self.config.opset,
            do_constant_folding=True,
            export_params=True,
        )
        export_time = time.perf_counter() - t0

        file_mb = os.path.getsize(output_path) / (1024 ** 2)

        ir_version, num_nodes = self._inspect_onnx(output_path)

        simplification_applied = False
        if self.config.simplify:
            simplification_applied = self._try_simplify(output_path)

        validation_ok = False
        if self.config.validate:
            validation_ok = self._validate(output_path)

        return ExportReport(
            output_path=output_path,
            export_time_s=export_time,
            file_size_mb=file_mb,
            onnx_ir_version=ir_version,
            opset_version=self.config.opset,
            num_nodes=num_nodes,
            input_shape=self.config.input_shape,
            validation_ok=validation_ok,
            simplification_applied=simplification_applied,
        )

    def validate_only(self, onnx_path: str) -> bool:
        """Run validation checks on an already-exported ONNX file.

        Parameters
        ----------
        onnx_path: Path to an existing ``.onnx`` file.

        Returns
        -------
        ``True`` if all enabled checks pass, ``False`` if a check fails
        or required libraries (``onnx``, ``onnxruntime``) are absent.
        """
        return self._validate(onnx_path)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _inspect_onnx(self, path: str) -> Tuple[int, int]:
        """Return (ir_version, node_count) by reading the ONNX protobuf."""
        if not self._onnx_available:
            return 0, 0
        import onnx
        model = onnx.load(path)
        return model.ir_version, len(model.graph.node)

    def _validate(self, path: str) -> bool:
        """Run onnx.checker and a dry-run ORT session."""
        try:
            if self._onnx_available:
                import onnx
                onnx.checker.check_model(path)
            if self._ort_available:
                import numpy as np
                import onnxruntime as ort
                session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                in_name = session.get_inputs()[0].name
                dummy_np = np.random.randn(*self.config.input_shape).astype(np.float32)
                session.run(None, {in_name: dummy_np})
            return True
        except Exception:
            return False

    def _try_simplify(self, path: str) -> bool:
        """Attempt in-place graph simplification via onnxsim."""
        if not self._onnxsim_available:
            return False
        try:
            import onnx
            import onnxsim
            model = onnx.load(path)
            simplified, ok = onnxsim.simplify(model)
            if ok:
                import onnx as _onnx
                _onnx.save(simplified, path)
            return ok
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Dependency probes
    # ------------------------------------------------------------------

    @staticmethod
    def _probe_torch() -> bool:
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _probe_onnx() -> bool:
        try:
            import onnx  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _probe_ort() -> bool:
        try:
            import onnxruntime  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _probe_onnxsim() -> bool:
        try:
            import onnxsim  # noqa: F401
            return True
        except ImportError:
            return False

    def _require_torch(self) -> None:
        if not self._torch_available:
            raise ImportError(
                "torch is required for ONNX export. Install with: pip install torch"
            )

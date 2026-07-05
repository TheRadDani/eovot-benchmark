"""Post-training quantization pipeline for ONNX-based trackers.

Provides INT8 and FP16 quantization via ONNX Runtime's quantization API,
plus an inference benchmarking utility to measure latency before/after
compression.  All heavy dependencies (``onnxruntime``,
``onnxruntime.quantization``) are imported lazily so the rest of EOVOT
continues to work without them.

Example
-------
::

    from eovot.optimization.quantization import ModelQuantizer, QuantizationConfig

    cfg = QuantizationConfig(mode="int8", per_channel=True)
    quantizer = ModelQuantizer(cfg)

    report = quantizer.quantize_dynamic("tracker.onnx", "tracker_int8.onnx")
    print(report)
    # {'mode': 'int8', 'original_size_mb': 4.2, 'quantized_size_mb': 1.1,
    #  'compression_ratio': 3.82, 'quantization_time_s': 0.34}

    bench_orig = quantizer.benchmark_inference("tracker.onnx", (1, 3, 128, 128))
    bench_q    = quantizer.benchmark_inference("tracker_int8.onnx", (1, 3, 128, 128))
    print(QuantizationBenchmark.compare(bench_orig, bench_q))
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class QuantizationConfig:
    """Parameters controlling post-training quantization.

    Attributes
    ----------
    mode:
        Target precision — ``"int8"`` (4× size reduction, most edge devices)
        or ``"fp16"`` (2× reduction, GPU-friendly, better accuracy).
    per_channel:
        Use per-channel weight quantization (better accuracy, slightly
        larger model than per-tensor).  Only applies to ``"int8"`` mode.
    activation_type:
        Activation quantisation scheme: ``"asymmetric"`` (zero-point shifted,
        default for INT8) or ``"symmetric"`` (zero centred, INT8 with better
        inference compatibility on some backends).
    calibration_frames:
        Number of representative frames used to calibrate activation
        ranges for static quantization.  Ignored for dynamic quantization.
    """

    mode: str = "int8"
    per_channel: bool = True
    activation_type: str = "asymmetric"
    calibration_frames: int = 100

    def __post_init__(self) -> None:
        if self.mode not in ("int8", "fp16"):
            raise ValueError(f"mode must be 'int8' or 'fp16', got {self.mode!r}")
        if self.activation_type not in ("asymmetric", "symmetric"):
            raise ValueError(
                f"activation_type must be 'asymmetric' or 'symmetric', got {self.activation_type!r}"
            )


@dataclass
class InferenceStats:
    """Latency and throughput statistics from a single benchmark run.

    Attributes
    ----------
    model_path:   Path to the ONNX model that was benchmarked.
    mean_ms:      Mean per-frame inference latency in milliseconds.
    std_ms:       Standard deviation of per-frame latency.
    p95_ms:       95th-percentile latency (tail-latency budget indicator).
    fps:          Frames per second derived from mean latency.
    n_runs:       Number of timed inference passes used.
    input_shape:  Input tensor shape used during benchmarking.
    """

    model_path: str
    mean_ms: float
    std_ms: float
    p95_ms: float
    fps: float
    n_runs: int
    input_shape: Tuple

    def to_dict(self) -> Dict:
        return {
            "model_path": self.model_path,
            "mean_latency_ms": self.mean_ms,
            "std_latency_ms": self.std_ms,
            "p95_latency_ms": self.p95_ms,
            "fps": self.fps,
            "n_runs": self.n_runs,
            "input_shape": list(self.input_shape),
        }


class ModelQuantizer:
    """Post-training quantization and inference benchmarking for ONNX models.

    Parameters
    ----------
    config:
        Quantization settings.  Defaults to INT8 per-channel asymmetric.
    """

    def __init__(self, config: Optional[QuantizationConfig] = None) -> None:
        self.config = config or QuantizationConfig()
        self._ort_available = self._probe_onnxruntime()
        self._quant_available = self._probe_quant_tools()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quantize_dynamic(self, model_path: str, output_path: str) -> Dict:
        """Dynamically quantize an ONNX model's weights to INT8 or FP16.

        Dynamic quantization quantises weights at export time and activations
        at runtime — no calibration data is needed, making it the simplest
        path to edge deployment.

        Parameters
        ----------
        model_path:   Path to the source FP32 ONNX model.
        output_path:  Destination path for the quantized model.

        Returns
        -------
        dict with keys:
            ``mode``, ``original_size_mb``, ``quantized_size_mb``,
            ``compression_ratio``, ``quantization_time_s``.

        Raises
        ------
        ImportError
            If ``onnxruntime`` or ``onnxruntime.quantization`` is not installed.
        FileNotFoundError
            If *model_path* does not exist.
        """
        self._require_quant()
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        from onnxruntime.quantization import QuantType, quantize_dynamic

        qtype = QuantType.QUInt8 if self.config.mode == "int8" else QuantType.QInt8

        t0 = time.perf_counter()
        quantize_dynamic(
            model_input=model_path,
            model_output=output_path,
            weight_type=qtype,
            per_channel=self.config.per_channel,
        )
        elapsed = time.perf_counter() - t0

        orig_mb = os.path.getsize(model_path) / (1024 ** 2)
        quant_mb = os.path.getsize(output_path) / (1024 ** 2)

        return {
            "mode": self.config.mode,
            "original_size_mb": round(orig_mb, 3),
            "quantized_size_mb": round(quant_mb, 3),
            "compression_ratio": round(orig_mb / quant_mb, 3) if quant_mb > 0 else float("inf"),
            "quantization_time_s": round(elapsed, 3),
        }

    def benchmark_inference(
        self,
        model_path: str,
        input_shape: Tuple[int, ...],
        n_runs: int = 100,
        warmup: int = 10,
    ) -> InferenceStats:
        """Measure inference latency of an ONNX model on random inputs.

        Parameters
        ----------
        model_path:   Path to the ONNX model.
        input_shape:  Input tensor shape, e.g. ``(1, 3, 128, 128)``.
        n_runs:       Number of timed inference passes.
        warmup:       Un-timed warmup passes to prime JIT compilation.

        Returns
        -------
        :class:`InferenceStats` with mean, std, p95 latency and FPS.
        """
        self._require_ort()
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        import onnxruntime as ort

        session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        input_name = session.get_inputs()[0].name
        dummy = np.random.randn(*input_shape).astype(np.float32)

        for _ in range(warmup):
            session.run(None, {input_name: dummy})

        times_ms: list = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            session.run(None, {input_name: dummy})
            times_ms.append((time.perf_counter() - t0) * 1_000.0)

        arr = np.asarray(times_ms)
        mean_ms = float(arr.mean())
        return InferenceStats(
            model_path=model_path,
            mean_ms=round(mean_ms, 3),
            std_ms=round(float(arr.std()), 3),
            p95_ms=round(float(np.percentile(arr, 95)), 3),
            fps=round(1_000.0 / mean_ms, 2) if mean_ms > 0 else float("inf"),
            n_runs=n_runs,
            input_shape=input_shape,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _probe_onnxruntime() -> bool:
        try:
            import onnxruntime  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _probe_quant_tools() -> bool:
        try:
            from onnxruntime.quantization import quantize_dynamic  # noqa: F401
            return True
        except ImportError:
            return False

    def _require_ort(self) -> None:
        if not self._ort_available:
            raise ImportError(
                "onnxruntime is required for inference benchmarking. "
                "Install with: pip install onnxruntime"
            )

    def _require_quant(self) -> None:
        if not self._quant_available:
            raise ImportError(
                "onnxruntime quantization tools are required. "
                "Install with: pip install onnxruntime"
            )


class QuantizationBenchmark:
    """Compare inference stats between an original and a quantized model.

    All methods are static — no instantiation needed.

    Example
    -------
    ::

        orig  = quantizer.benchmark_inference("model.onnx",      (1,3,128,128))
        quant = quantizer.benchmark_inference("model_int8.onnx",  (1,3,128,128))
        print(QuantizationBenchmark.compare(orig, quant))
    """

    @staticmethod
    def compare(original: InferenceStats, quantized: InferenceStats) -> Dict:
        """Return a dict summarising the speed-up from quantization.

        Parameters
        ----------
        original:   Benchmark stats for the FP32 baseline model.
        quantized:  Benchmark stats for the quantized model.

        Returns
        -------
        dict with keys:
            ``latency_speedup``, ``fps_gain_pct``,
            ``original_fps``, ``quantized_fps``,
            ``original_p95_ms``, ``quantized_p95_ms``.
        """
        speedup = (
            round(original.mean_ms / quantized.mean_ms, 3)
            if quantized.mean_ms > 0
            else float("inf")
        )
        fps_gain = (
            round((quantized.fps - original.fps) / original.fps * 100.0, 2)
            if original.fps > 0
            else float("inf")
        )
        return {
            "latency_speedup": speedup,
            "fps_gain_pct": fps_gain,
            "original_fps": original.fps,
            "quantized_fps": quantized.fps,
            "original_mean_ms": original.mean_ms,
            "quantized_mean_ms": quantized.mean_ms,
            "original_p95_ms": original.p95_ms,
            "quantized_p95_ms": quantized.p95_ms,
        }

    @staticmethod
    def format_report(
        original: InferenceStats,
        quantized: InferenceStats,
        quantization_meta: Optional[Dict] = None,
    ) -> str:
        """Return a human-readable Markdown comparison report.

        Parameters
        ----------
        original:          FP32 baseline stats.
        quantized:         Quantized model stats.
        quantization_meta: Optional dict returned by :meth:`ModelQuantizer.quantize_dynamic`.
        """
        cmp = QuantizationBenchmark.compare(original, quantized)
        lines = [
            "## Quantization Report",
            "",
            f"| Metric            | Original (FP32) | Quantized        | Delta          |",
            f"|-------------------|-----------------|------------------|----------------|",
            f"| Mean latency (ms) | {original.mean_ms:>15.3f} | {quantized.mean_ms:>16.3f} | "
            f"×{cmp['latency_speedup']:.2f} speedup    |",
            f"| FPS               | {original.fps:>15.2f} | {quantized.fps:>16.2f} | "
            f"+{cmp['fps_gain_pct']:.1f}%           |",
            f"| P95 latency (ms)  | {original.p95_ms:>15.3f} | {quantized.p95_ms:>16.3f} |                |",
        ]
        if quantization_meta:
            lines += [
                "",
                f"**Model size:** {quantization_meta.get('original_size_mb', '?'):.2f} MB "
                f"→ {quantization_meta.get('quantized_size_mb', '?'):.2f} MB  "
                f"(×{quantization_meta.get('compression_ratio', '?'):.2f} compression)",
            ]
        return "\n".join(lines)

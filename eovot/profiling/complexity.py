"""Tracker complexity analysis: parameters, estimated FLOPs, and memory bandwidth.

This module provides :class:`TrackerComplexityProfiler`, which estimates the
computational complexity of any EOVOT tracker and produces a
:class:`ComplexityReport` comparable alongside benchmark accuracy and efficiency
results.

Three complexity axes are measured:

1. **Parameter count** — total stored numerical values (PyTorch tensors or
   NumPy arrays in the tracker's internal state).
2. **Estimated FLOPs** — analytic formulas for classical correlation-filter and
   optical-flow trackers; hook-based counting for PyTorch ``nn.Module`` models;
   resolution-proportional fallback for unknown types.
3. **Memory bandwidth** — bytes transferred per frame: frame read + search
   region read + response map write.  Memory-bound trackers may not meet
   real-time constraints on embedded DRAM even when FLOPs counts are low.

Edge deployment context::

    Raspberry Pi 4 DRAM bandwidth: ~4 GB/s
    STM32H7 DRAM bandwidth:         ~480 MB/s
    Trackers reading > 1 MB/frame will struggle at real-time rates on MCUs.

Example::

    from eovot.profiling.complexity import TrackerComplexityProfiler
    from eovot.trackers.kcf import KCFTracker
    from eovot.trackers.mosse import MOSSETracker

    profiler = TrackerComplexityProfiler()

    for tracker in [MOSSETracker(), KCFTracker()]:
        report = profiler.profile(tracker, frame_size=(320, 240))
        print(report)

    # Multi-resolution sweep
    reports = profiler.profile_multi_resolution(MOSSETracker())
    for res, report in reports.items():
        print(f"{res}: {report.estimated_flops_m:.2f}M FLOPs")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ComplexityReport:
    """Complexity summary for one tracker at one resolution.

    Attributes:
        tracker_name:                     Human-readable identifier.
        frame_width:                      Frame width used during profiling (px).
        frame_height:                     Frame height used during profiling (px).
        param_count:                      Total stored parameters (learnable
                                          weights or filter arrays).
        param_memory_bytes:               Bytes required to store all params
                                          (assumes ``bytes_per_param`` per value).
        estimated_flops:                  Approximate floating-point ops per frame.
                                          ``None`` when no estimate is available.
        memory_bandwidth_bytes_per_frame: Estimated DRAM bytes transferred per
                                          processed frame (read + write).
        complexity_class:                 Categorical label computed from FLOPs:
                                          ``"light"`` (<1M), ``"medium"`` (1–50M),
                                          ``"heavy"`` (>50M), or ``"unknown"``.
        notes:                            Caveats about estimation method.
    """

    tracker_name: str
    frame_width: int
    frame_height: int
    param_count: int
    param_memory_bytes: int
    estimated_flops: Optional[int]
    memory_bandwidth_bytes_per_frame: int
    complexity_class: str
    notes: str = ""

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def param_memory_kb(self) -> float:
        """Parameter storage in kibibytes."""
        return self.param_memory_bytes / 1024.0

    @property
    def estimated_flops_m(self) -> Optional[float]:
        """Estimated FLOPs in millions, or ``None``."""
        return self.estimated_flops / 1e6 if self.estimated_flops is not None else None

    @property
    def memory_bandwidth_kb_per_frame(self) -> float:
        """Memory bandwidth in kibibytes per frame."""
        return self.memory_bandwidth_bytes_per_frame / 1024.0

    @property
    def pixels(self) -> int:
        """Frame area in pixels."""
        return self.frame_width * self.frame_height

    @property
    def flops_per_pixel(self) -> Optional[float]:
        """Estimated FLOPs divided by frame area, or ``None``."""
        if self.estimated_flops is None or self.pixels == 0:
            return None
        return self.estimated_flops / self.pixels

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """Return a plain dict for JSON / CSV export."""
        return {
            "tracker_name": self.tracker_name,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "param_count": self.param_count,
            "param_memory_bytes": self.param_memory_bytes,
            "param_memory_kb": round(self.param_memory_kb, 2),
            "estimated_flops": self.estimated_flops,
            "estimated_flops_m": (
                round(self.estimated_flops_m, 4)
                if self.estimated_flops_m is not None else None
            ),
            "memory_bandwidth_bytes_per_frame": self.memory_bandwidth_bytes_per_frame,
            "memory_bandwidth_kb_per_frame": round(self.memory_bandwidth_kb_per_frame, 2),
            "complexity_class": self.complexity_class,
            "flops_per_pixel": (
                round(self.flops_per_pixel, 4)
                if self.flops_per_pixel is not None else None
            ),
            "notes": self.notes,
        }

    def __str__(self) -> str:
        flops_str = (
            f"{self.estimated_flops_m:.2f}M"
            if self.estimated_flops_m is not None
            else "N/A"
        )
        return (
            f"ComplexityReport[{self.tracker_name}  "
            f"{self.frame_width}×{self.frame_height}]\n"
            f"  Parameters  : {self.param_count:,}  "
            f"({self.param_memory_kb:.1f} KB)\n"
            f"  Est. FLOPs  : {flops_str}  [{self.complexity_class}]\n"
            f"  Mem BW/frame: {self.memory_bandwidth_kb_per_frame:.1f} KB\n"
            f"  Notes       : {self.notes or '—'}"
        )


class TrackerComplexityProfiler:
    """Estimate the computational complexity of EOVOT trackers.

    Works with classical OpenCV-based trackers and PyTorch deep trackers
    via introspection.  No GPU is required and no real video data is needed
    — the profiler synthesises a random test frame.

    Args:
        bytes_per_param: Storage bytes assumed per scalar parameter.  Default:
                         ``4`` (float32).  Use ``2`` for half-precision models.

    Typical usage::

        profiler = TrackerComplexityProfiler()
        report   = profiler.profile(KCFTracker(), frame_size=(640, 480))
        print(report)
    """

    def __init__(self, bytes_per_param: int = 4) -> None:
        self.bytes_per_param = bytes_per_param

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(
        self,
        tracker: Any,
        frame_size: Tuple[int, int] = (320, 240),
        bbox: Optional[Tuple[float, float, float, float]] = None,
        n_warmup_frames: int = 5,
    ) -> ComplexityReport:
        """Profile *tracker* at the given *frame_size* and return a report.

        The tracker is initialized with a synthetic frame and run for
        ``n_warmup_frames`` update steps so that its internal state (filter
        arrays, model weights) is fully populated before counting parameters.

        Args:
            tracker:         Any :class:`~eovot.trackers.base.BaseTracker`.
            frame_size:      ``(width, height)`` in pixels.  Default: ``(320, 240)``.
            bbox:            Init bounding box ``(x, y, w, h)``.  Defaults to a
                             centred box occupying 25 % of the frame.
            n_warmup_frames: Update steps before counting parameters.  Default: ``5``.

        Returns:
            Populated :class:`ComplexityReport`.
        """
        w, h = frame_size
        if bbox is None:
            bbox = (w * 0.25, h * 0.25, w * 0.25, h * 0.25)

        frame = self._make_test_frame(w, h)
        tracker.initialize(frame, bbox)
        for _ in range(n_warmup_frames):
            tracker.update(frame)

        param_count, param_bytes, notes = self._count_params(tracker)
        flops = self._estimate_flops(tracker, w, h)
        bw = self._estimate_memory_bandwidth(w, h, bbox)
        cls = _classify_flops(flops)

        return ComplexityReport(
            tracker_name=getattr(tracker, "name", type(tracker).__name__),
            frame_width=w,
            frame_height=h,
            param_count=param_count,
            param_memory_bytes=param_bytes,
            estimated_flops=flops,
            memory_bandwidth_bytes_per_frame=bw,
            complexity_class=cls,
            notes=notes,
        )

    def profile_multi_resolution(
        self,
        tracker: Any,
        resolutions: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[Tuple[int, int], ComplexityReport]:
        """Profile *tracker* at several standard resolutions.

        Useful for understanding how FLOPs and memory bandwidth scale with
        input size — critical for choosing the right resolution on a given
        edge device.

        Args:
            tracker:     Tracker to profile.
            resolutions: List of ``(width, height)`` pairs.  Defaults to
                         four standard edge-device resolutions:
                         160×120, 320×240, 640×480, 1280×720.

        Returns:
            Ordered dict mapping each ``(width, height)`` to its report.
        """
        if resolutions is None:
            resolutions = [(160, 120), (320, 240), (640, 480), (1280, 720)]

        reports: Dict[Tuple[int, int], ComplexityReport] = {}
        for res in resolutions:
            rw, rh = res
            init_bbox = (rw * 0.25, rh * 0.25, rw * 0.25, rh * 0.25)
            reports[res] = self.profile(tracker, frame_size=res, bbox=init_bbox)
        return reports

    def compare(
        self,
        trackers: List[Any],
        frame_size: Tuple[int, int] = (320, 240),
    ) -> List[ComplexityReport]:
        """Profile a list of trackers at the same resolution for comparison.

        Args:
            trackers:   List of trackers to compare.
            frame_size: Resolution used for all trackers.

        Returns:
            List of :class:`ComplexityReport`, one per tracker, sorted by
            estimated FLOPs in ascending order (``None`` FLOPs last).
        """
        reports = [self.profile(t, frame_size=frame_size) for t in trackers]
        return sorted(
            reports,
            key=lambda r: (r.estimated_flops is None, r.estimated_flops or 0),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_test_frame(w: int, h: int) -> np.ndarray:
        """Generate a deterministic synthetic BGR frame."""
        rng = np.random.default_rng(seed=0)
        return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

    def _count_params(self, tracker: Any) -> Tuple[int, int, str]:
        """Count stored parameters in *tracker*.

        Tries PyTorch ``nn.Module`` first, then scans NumPy arrays in
        ``__dict__``.

        Returns:
            ``(param_count, bytes_used, estimation_notes)``
        """
        # PyTorch path
        try:
            import torch.nn as nn
            for attr in ("model", "_model", "net", "_net", "backbone", "siamese"):
                candidate = getattr(tracker, attr, None)
                if isinstance(candidate, nn.Module):
                    count = sum(p.numel() for p in candidate.parameters())
                    return count, count * self.bytes_per_param, "PyTorch nn.Module"
        except ImportError:
            pass

        # NumPy array scan
        total = 0
        for v in vars(tracker).values():
            if isinstance(v, np.ndarray):
                total += v.size
        note = "NumPy array scan" if total > 0 else "no internal arrays detected"
        return total, total * self.bytes_per_param, note

    def _estimate_flops(self, tracker: Any, w: int, h: int) -> Optional[int]:
        """Return per-frame FLOPs estimate for *tracker* at resolution ``w×h``.

        Uses analytic formulas for known classical tracker families.  Falls
        back to a PyTorch conv-hook trace for deep models, then to a generic
        10-ops-per-pixel estimate.
        """
        n = w * h
        name = type(tracker).__name__.lower()

        # Correlation-filter family: two 2D FFTs + element-wise product
        # FFT complexity: O(N log₂ N) with radix-2 — coefficient ≈5
        fft_flops = int(5 * n * max(1.0, np.log2(n + 1)))
        if any(k in name for k in ("mosse", "kcf", "dsst", "csrt", "dcf")):
            return 2 * fft_flops + n

        if any(k in name for k in ("camshift", "meanshift")):
            # Histogram (256 bins) + backprojection + mean-shift iterations
            return int(20 * n)

        if any(k in name for k in ("mil", "boosting", "adaboost")):
            # 100 weak classifiers, 10 spatial patches each
            return int(100 * 10 * n)

        if any(k in name for k in ("lk", "lucaskanade", "opticalflow", "medianflow")):
            # Lucas-Kanade: 200 keypoints, 3×3 spatial patch, ~15 iterations
            return int(200 * 9 * 15)

        if any(k in name for k in ("kalman", "particle")):
            # State-space prediction: low cost, dominated by feature extraction
            return int(5 * n)

        # PyTorch model hook trace
        pytorch_flops = self._pytorch_flops(tracker, w, h)
        if pytorch_flops is not None:
            return pytorch_flops

        # Generic fallback
        return int(10 * n)

    def _pytorch_flops(self, tracker: Any, w: int, h: int) -> Optional[int]:
        """Estimate FLOPs for a PyTorch tracker via forward-pass conv hooks."""
        try:
            import torch
            import torch.nn as nn

            model: Optional[nn.Module] = None
            for attr in ("model", "_model", "net", "backbone"):
                candidate = getattr(tracker, attr, None)
                if isinstance(candidate, nn.Module):
                    model = candidate
                    break
            if model is None:
                return None

            flops_acc = [0]

            def _conv_hook(module: nn.Module, inp: Any, out: Any) -> None:
                if isinstance(module, nn.Conv2d):
                    out_h, out_w = out.shape[-2], out.shape[-1]
                    k_h, k_w = module.kernel_size
                    c_in = module.in_channels
                    c_out = module.out_channels
                    macs = out_h * out_w * k_h * k_w * (c_in // module.groups) * c_out
                    flops_acc[0] += 2 * macs  # multiply-accumulate ×2 for flops

            handles = [m.register_forward_hook(_conv_hook) for m in model.modules()]
            model.eval()
            with torch.no_grad():
                try:
                    model(torch.zeros(1, 3, h, w))
                except Exception:
                    pass
            for hdl in handles:
                hdl.remove()

            return flops_acc[0] if flops_acc[0] > 0 else None
        except (ImportError, Exception):
            return None

    @staticmethod
    def _estimate_memory_bandwidth(
        w: int,
        h: int,
        bbox: Tuple[float, float, float, float],
    ) -> int:
        """Estimate bytes transferred per frame.

        Accounts for: full frame read (BGR uint8), search region read
        (2× template linear padding), response map write (float32).
        """
        frame_bytes = w * h * 3  # BGR uint8
        bw, bh = bbox[2], bbox[3]
        template_px = int(bw * bh)
        # Typical correlation filter search region: 2.5× template side
        search_px = int((bw * 2.5) * (bh * 2.5))
        search_bytes = search_px * 3
        response_bytes = template_px * 4  # float32 response map
        return frame_bytes + search_bytes + response_bytes


def _classify_flops(flops: Optional[int]) -> str:
    """Return a categorical complexity label for *flops*."""
    if flops is None:
        return "unknown"
    if flops < 1_000_000:
        return "light"
    if flops < 50_000_000:
        return "medium"
    return "heavy"

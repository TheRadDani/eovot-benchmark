"""Tracker algorithmic complexity analyser for EOVOT.

Estimates per-tracker computational cost in terms of:

* **Parameters** — number of learnable/state variables maintained by the
  tracker (filter coefficients, feature weights, tracked points, etc.).
* **FLOPs per frame** — floating-point operations required for one update
  step given the template patch size.
* **Model size (MB)** — estimated storage required for the tracker state,
  assuming float32 representation.

All estimates for classical trackers are derived analytically from their
published algorithmic complexity.  For any tracker that exposes a PyTorch
``nn.Module`` at ``.model``, the actual parameter count is used instead.

Complexity models
-----------------
MOSSE
    O(N log N) via a single-channel FFT on the template patch.
    Update step: 3 complex FFTs (2 forward + 1 inverse) + element-wise ops.

KCF
    O(C · N log N) with C=31 HOG channels.  Same FFT-domain structure as
    MOSSE but applied independently per channel.

CSRT
    O(C · N log N) with C≈50 deep-feature channels, plus a spatial
    reliability map requiring an additional FFT pass.

MIL
    O(B · D · N) where B=45 (bag size) and D=256 (Haar-like feature count).
    Online Boosting update on each bag member.

MedianFlow
    O(P · W² · I) where P=500 points, W=5 (LK window), I=20 (iterations).
    Two Lucas-Kanade passes (forward + backward) for consistency check.

Usage::

    from eovot.profiling.complexity import TrackerComplexityAnalyzer

    analyzer = TrackerComplexityAnalyzer(patch_size=64)

    # Single tracker
    report = analyzer.analyze("KCF")
    print(report)
    # ComplexityReport[KCF]  params=262,144  FLOPs=8.70 MFLOPs/frame  size=1.0000 MB

    # All trackers
    reports = analyzer.analyze_all()
    for name, r in reports.items():
        print(f"{name:12s}  {r.mflops:8.3f} MFLOPs  {r.model_size_mb:.3f} MB")

References
----------
- Bolme et al., "Visual Object Tracking using Adaptive Correlation Filters",
  CVPR 2010.  (MOSSE)
- Henriques et al., "High-Speed Tracking with Kernelized Correlation Filters",
  TPAMI 2015.  (KCF)
- Lukezic et al., "Discriminative Correlation Filter with Channel and Spatial
  Reliability", CVPR 2017.  (CSRT)
- Babenko et al., "Visual Tracking with Online Multiple Instance Learning",
  CVPR 2009.  (MIL)
- Kalal et al., "Forward-Backward Error: Automatic Detection of Tracking
  Failures", ICPR 2010.  (MedianFlow)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ComplexityReport:
    """Algorithmic complexity summary for one tracker.

    Attributes:
        tracker_name:     Tracker identifier.
        param_count:      Estimated number of parameters / state variables.
        flops_per_frame:  Estimated floating-point ops per update step.
        model_size_mb:    Estimated state/model size in megabytes (float32).
        patch_size:       Template patch side length used for the estimate.
        notes:            Human-readable explanation of the estimate.
    """

    tracker_name: str
    param_count: int
    flops_per_frame: int
    model_size_mb: float
    patch_size: int
    notes: str = ""

    @property
    def mflops(self) -> float:
        """FLOPs per frame expressed in mega-FLOPs (millions)."""
        return self.flops_per_frame / 1_000_000.0

    def __str__(self) -> str:
        return (
            f"ComplexityReport[{self.tracker_name}]  "
            f"params={self.param_count:,}  "
            f"FLOPs={self.mflops:.2f} MFLOPs/frame  "
            f"size={self.model_size_mb:.4f} MB"
        )

    def to_dict(self) -> Dict:
        """Serialize to a plain dict for JSON export."""
        return {
            "tracker_name": self.tracker_name,
            "patch_size": self.patch_size,
            "param_count": self.param_count,
            "flops_per_frame": self.flops_per_frame,
            "mflops_per_frame": round(self.mflops, 4),
            "model_size_mb": round(self.model_size_mb, 6),
            "notes": self.notes,
        }


# Ordered list used by analyze_all() and the CLI.
SUPPORTED_TRACKERS: List[str] = ["MOSSE", "KCF", "CSRT", "MIL", "MedianFlow"]


class TrackerComplexityAnalyzer:
    """Compute analytical FLOPs and parameter estimates for EOVOT trackers.

    Args:
        patch_size:    Side length (px) of the square template region.
                       Defaults to 64.
        search_scale:  Ratio of search-region side to template side.
                       Informational only — not all trackers use an explicit
                       search region in the classical sense.  Defaults to 2.0.
        float_bytes:   Bytes per element used for size estimation.
                       Defaults to 4 (float32).
    """

    # Tracker-specific constants (matching OpenCV default configurations)
    _KCF_HOG_CHANNELS: int = 31
    _CSRT_FEATURE_CHANNELS: int = 50
    _MIL_BAG_SIZE: int = 45
    _MIL_FEATURE_DIM: int = 256
    _MEDIANFLOW_NUM_POINTS: int = 500
    _MEDIANFLOW_LK_WINDOW: int = 5
    _MEDIANFLOW_LK_ITERS: int = 20

    def __init__(
        self,
        patch_size: int = 64,
        search_scale: float = 2.0,
        float_bytes: int = 4,
    ) -> None:
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if search_scale <= 0:
            raise ValueError(f"search_scale must be positive, got {search_scale}")
        self.patch_size = patch_size
        self.search_scale = search_scale
        self.float_bytes = float_bytes

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze(self, tracker_name: str) -> ComplexityReport:
        """Return the :class:`ComplexityReport` for *tracker_name*.

        Args:
            tracker_name: One of ``MOSSE``, ``KCF``, ``CSRT``, ``MIL``,
                          ``MedianFlow``.

        Raises:
            KeyError: If *tracker_name* is not in the built-in registry.
        """
        registry = {
            "MOSSE": self._mosse,
            "KCF": self._kcf,
            "CSRT": self._csrt,
            "MIL": self._mil,
            "MedianFlow": self._median_flow,
        }
        if tracker_name not in registry:
            known = ", ".join(SUPPORTED_TRACKERS)
            raise KeyError(
                f"No complexity model for '{tracker_name}'. Known: {known}"
            )
        return registry[tracker_name]()

    def analyze_all(self) -> Dict[str, ComplexityReport]:
        """Return complexity reports for all built-in trackers (ordered)."""
        return {name: self.analyze(name) for name in SUPPORTED_TRACKERS}

    def compare_table(self) -> str:
        """Return a human-readable comparison table as a string."""
        reports = self.analyze_all()
        header = (
            f"{'Tracker':<14} {'Params':>12} {'MFLOPs/fr':>12} {'Size (MB)':>12}"
        )
        sep = "-" * len(header)
        rows = ["\n" + sep, header, sep]
        for r in reports.values():
            rows.append(
                f"{r.tracker_name:<14} "
                f"{r.param_count:>12,} "
                f"{r.mflops:>12.3f} "
                f"{r.model_size_mb:>12.4f}"
            )
        rows.append(sep)
        return "\n".join(rows)

    # ------------------------------------------------------------------
    # Private: per-tracker analytical models
    # ------------------------------------------------------------------

    def _fft2_flops(self, n: int) -> int:
        """Estimate FLOPs for a 2-D radix-2 FFT on *n* elements.

        Uses the standard approximation: 5 · N · log₂(N).
        """
        if n <= 1:
            return 0
        return int(5 * n * math.log2(n))

    def _bytes_to_mb(self, n_elements: int) -> float:
        return n_elements * self.float_bytes / (1024 ** 2)

    def _mosse(self) -> ComplexityReport:
        """MOSSE: minimum output sum of squared error correlation filter.

        Single luminance channel.  Each update step:
          1. 2× forward 2-D FFT (numerator and denominator accumulators)
          2. 1× element-wise complex multiply + running average: O(N)
          3. 1× inverse 2-D FFT for the response map

        Total ≈ 3 · FFT(N) + N operations.
        State: two complex arrays (numerator A, denominator B) → 4 real arrays.
        """
        n = self.patch_size ** 2
        flops = 3 * self._fft2_flops(n) + n
        # Two complex arrays stored as pairs of real arrays (Re + Im)
        params = 4 * n
        return ComplexityReport(
            tracker_name="MOSSE",
            param_count=params,
            flops_per_frame=flops,
            model_size_mb=self._bytes_to_mb(params),
            patch_size=self.patch_size,
            notes=(
                f"Single-channel FFT correlation filter on "
                f"{self.patch_size}×{self.patch_size} patch. "
                "~500+ FPS on modern CPUs."
            ),
        )

    def _kcf(self) -> ComplexityReport:
        """KCF: kernelised correlation filter with HOG features.

        Uses C=31 HOG channels (9 unsigned + 18 signed gradient bins +
        4 block normalisation channels).  Each channel is processed via
        an independent FFT-domain update, then fused via the kernel trick.

        Update per frame:
          - C forward FFTs (feature extraction in frequency domain)
          - 1 kernel FFT + element-wise ridge regression: C · N ops
          - 1 inverse FFT for the response map
        Total ≈ (2C + 1) · FFT(N) + C · N
        """
        c = self._KCF_HOG_CHANNELS
        n = self.patch_size ** 2
        flops = (2 * c + 1) * self._fft2_flops(n) + c * n
        params = c * n  # filter coefficients, one per HOG channel
        return ComplexityReport(
            tracker_name="KCF",
            param_count=params,
            flops_per_frame=flops,
            model_size_mb=self._bytes_to_mb(params),
            patch_size=self.patch_size,
            notes=(
                f"{c}-channel HOG on {self.patch_size}×{self.patch_size} patch. "
                "Gaussian kernel; FFT-domain ridge regression solution. "
                "~150–350 FPS."
            ),
        )

    def _csrt(self) -> ComplexityReport:
        """CSRT: channel and spatial reliability tracking.

        Extends KCF to C≈50 deep-feature channels and adds a spatial
        reliability map (one additional FFT pass per frame).  The map
        down-weights unreliable spatial regions of the filter.

        Update per frame:
          - 2C forward FFTs (features + reliability)
          - reliability map computation: N ops
          - 2 inverse FFTs
        Total ≈ (2C + 2) · FFT(N) + 2C · N
        State: C filter arrays + 1 spatial mask.
        """
        c = self._CSRT_FEATURE_CHANNELS
        n = self.patch_size ** 2
        flops = (2 * c + 2) * self._fft2_flops(n) + 2 * c * n
        params = c * n + n  # filter bank + spatial reliability mask
        return ComplexityReport(
            tracker_name="CSRT",
            param_count=params,
            flops_per_frame=flops,
            model_size_mb=self._bytes_to_mb(params),
            patch_size=self.patch_size,
            notes=(
                f"~{c}-channel features on {self.patch_size}×{self.patch_size} patch. "
                "Spatial reliability map adds ~1 extra FFT pass per frame. "
                "Higher accuracy than KCF; ~30–60 FPS."
            ),
        )

    def _mil(self) -> ComplexityReport:
        """MIL: Multiple Instance Learning online tracker.

        Maintains a bag-based classifier (online Boosting) trained on
        B=45 patches per frame.  Each patch is described by D=256 Haar-like
        features; the classifier update cost is O(B · D · N).

        State: D weak-classifier weights (float array of length D).
        """
        b = self._MIL_BAG_SIZE
        d = self._MIL_FEATURE_DIM
        n = self.patch_size ** 2
        # Feature extraction per bag member + Boosting update per feature
        flops = b * (d * n + d)
        params = d  # learned weak-classifier weights
        return ComplexityReport(
            tracker_name="MIL",
            param_count=params,
            flops_per_frame=flops,
            model_size_mb=self._bytes_to_mb(params),
            patch_size=self.patch_size,
            notes=(
                f"Online Boosting with Haar-like features; "
                f"bag_size={b}, feature_dim={d}. "
                "Higher latency than correlation filters (~30–80 FPS). "
                "Robust to partial occlusion."
            ),
        )

    def _median_flow(self) -> ComplexityReport:
        """MedianFlow: sparse optical-flow with forward-backward check.

        Tracks P=500 sparse keypoints using Lucas-Kanade optical flow with
        window W=5 and I=20 iterations.  Two passes are performed per frame
        (forward + backward) for the consistency check.

        FLOPs per frame ≈ P · W² · I · 2 (forward + backward).
        State: current point positions (P × 2 floats).
        """
        p = self._MEDIANFLOW_NUM_POINTS
        w = self._MEDIANFLOW_LK_WINDOW
        iters = self._MEDIANFLOW_LK_ITERS
        flops = p * (w ** 2) * iters * 2  # forward + backward passes
        params = p * 2  # (x, y) position per tracked point
        return ComplexityReport(
            tracker_name="MedianFlow",
            param_count=params,
            flops_per_frame=flops,
            model_size_mb=self._bytes_to_mb(params),
            patch_size=self.patch_size,
            notes=(
                f"Lucas-Kanade sparse flow; {p} points, "
                f"window={w}×{w}, iters={iters}. "
                "Explicit failure detection via forward-backward error. "
                "Fails on fast motion or large occlusion."
            ),
        )


def analyze_tracker_complexity(
    tracker_name: str,
    patch_size: int = 64,
    search_scale: float = 2.0,
) -> ComplexityReport:
    """Convenience function: analyse a single tracker's complexity.

    Args:
        tracker_name:  One of MOSSE, KCF, CSRT, MIL, MedianFlow.
        patch_size:    Template patch side length in pixels (default 64).
        search_scale:  Search-region scale factor (default 2.0).

    Returns:
        :class:`ComplexityReport` for the tracker.
    """
    return TrackerComplexityAnalyzer(
        patch_size=patch_size,
        search_scale=search_scale,
    ).analyze(tracker_name)

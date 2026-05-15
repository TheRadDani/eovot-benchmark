"""Compute-complexity profiling for visual object trackers.

Estimates theoretical FLOPs per forward pass for EOVOT's built-in trackers
and combines them with empirical latency measurements to produce hardware
efficiency metrics.  This is critical for edge-deployment decisions: knowing
that tracker A delivers the same FPS as tracker B while requiring 10× fewer
FLOPs reveals a fundamental efficiency advantage that FPS alone cannot show.

Module structure
----------------
- :func:`mosse_flops` / :func:`kcf_flops` — analytic estimates for
  specific correlation-filter trackers.
- :func:`correlation_filter_flops` — generic parameterised estimator for any
  FFT-based tracker.
- :func:`siamese_tracker_flops` — lightweight estimate for Siamese-network
  trackers (SiamRPN, NanoTracker, etc.).
- :class:`ComputeProfile` — data container for a tracker's compute profile.
- :class:`ComputeProfiler` — registry and comparison interface.

All estimates are *theoretical lower bounds*: memory-access costs, cache
misses, Python/OpenCV overhead, and framework dispatch are not modelled.
Use these numbers for relative comparisons, not for absolute prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Data container ────────────────────────────────────────────────────────────


@dataclass
class ComputeProfile:
    """FLOP complexity profile for a single tracker forward pass.

    Attributes:
        tracker_name: Identifier matching the tracker's class name.
        patch_size: Template / search patch size ``(height, width)`` in pixels.
        flops_per_frame: Estimated FLOPs for one tracker update call.
        params_count: Learnable parameter count (DL trackers only; ``None``
            for classical filter trackers).
        compute_note: Human-readable derivation note for audit purposes.
    """

    tracker_name: str
    patch_size: Tuple[int, int]
    flops_per_frame: float
    params_count: Optional[int] = None
    compute_note: str = ""

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def mega_flops(self) -> float:
        """FLOPs per frame expressed in MFLOPs."""
        return self.flops_per_frame / 1e6

    @property
    def giga_flops(self) -> float:
        """FLOPs per frame expressed in GFLOPs."""
        return self.flops_per_frame / 1e9

    def throughput_gflops_per_sec(self, fps: float) -> float:
        """Effective GFLOPs/s at observed *fps* throughput.

        Useful for comparing compute efficiency across devices: a tracker
        achieving 1 GFLOPs/s on an ARM Cortex-A53 is doing more with less
        than one achieving the same rate on a high-end x86 core.

        Args:
            fps: Empirically measured frames-per-second.

        Returns:
            GFLOPs per second.
        """
        return self.flops_per_frame * fps / 1e9

    def flops_per_pixel(self) -> float:
        """FLOPs per pixel of the patch (resolution-normalised complexity)."""
        h, w = self.patch_size
        n_pixels = h * w
        return self.flops_per_frame / n_pixels if n_pixels > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "tracker_name": self.tracker_name,
            "patch_size": list(self.patch_size),
            "flops_per_frame": self.flops_per_frame,
            "mega_flops_per_frame": round(self.mega_flops, 4),
            "giga_flops_per_frame": round(self.giga_flops, 6),
            "params_count": self.params_count,
            "compute_note": self.compute_note,
        }

    def __str__(self) -> str:
        params_str = f"  params={self.params_count:,}" if self.params_count else ""
        return (
            f"ComputeProfile[{self.tracker_name}] "
            f"{self.mega_flops:.3f} MFLOPs/frame  "
            f"patch={self.patch_size[0]}×{self.patch_size[1]}"
            f"{params_str}"
        )


# ── FFT cost primitives ───────────────────────────────────────────────────────


def _fft2d_flops(height: int, width: int) -> float:
    """Estimated FLOPs for a single 2-D real-valued FFT of size ``(H, W)``.

    Uses the Cooley-Tukey O(N log₂ N) approximation:

        FLOPs(1-D FFT of size N) ≈ 5 · N · log₂(N)

    A 2-D FFT decomposes into H row-wise 1-D FFTs of size W followed by
    W column-wise 1-D FFTs of size H.
    """
    if height <= 0 or width <= 0:
        return 0.0
    row_ffts = height * 5.0 * width * math.log2(max(width, 2))
    col_ffts = width * 5.0 * height * math.log2(max(height, 2))
    return row_ffts + col_ffts


# ── Tracker-specific estimators ───────────────────────────────────────────────


def mosse_flops(patch_h: int = 64, patch_w: int = 64) -> ComputeProfile:
    """Estimate FLOPs per frame for a MOSSE tracker.

    MOSSE maintains a correlation filter in the Fourier domain and performs
    per-frame updates via the following operations:

    1. Extract + pre-process patch — 3 · H · W ops (normalisation, windowing)
    2. Forward FFT of target template G — fft2d(H, W)
    3. Forward FFT of input patch F — fft2d(H, W)
    4. Element-wise complex product G* ⊙ F — 6 · H · W real ops
    5. Element-wise magnitude squared |F|² — 4 · H · W real ops
    6. Inverse FFT of response map — fft2d(H, W)
    7. EMA update of numerator + denominator — 2 · H · W ops

    Args:
        patch_h: Patch / template height in pixels.
        patch_w: Patch / template width in pixels.

    Returns:
        :class:`ComputeProfile` with FLOPs estimate.
    """
    n = patch_h * patch_w
    pre_process = 3.0 * n
    fft_g = _fft2d_flops(patch_h, patch_w)
    fft_f = _fft2d_flops(patch_h, patch_w)
    complex_product = 6.0 * n
    mag_sq = 4.0 * n
    ifft_resp = _fft2d_flops(patch_h, patch_w)
    ema = 2.0 * n

    total = pre_process + fft_g + fft_f + complex_product + mag_sq + ifft_resp + ema
    note = (
        f"MOSSE {patch_h}×{patch_w}: pre-process + 3 FFTs "
        f"+ element-wise ops + EMA update"
    )
    return ComputeProfile(
        tracker_name="mosse",
        patch_size=(patch_h, patch_w),
        flops_per_frame=total,
        compute_note=note,
    )


def kcf_flops(
    patch_h: int = 64,
    patch_w: int = 64,
    num_features: int = 1,
) -> ComputeProfile:
    """Estimate FLOPs per frame for a KCF tracker.

    KCF extends MOSSE by computing a Gaussian kernel response via cross-
    correlation in the Fourier domain and using multi-channel HOG features.
    Additional cost over MOSSE per frame:

    1. Feature extraction — 8 · C · H · W ops (HOG-like, C channels)
    2. Kernel correlation FFT pair — 2 × fft2d(H, W)
    3. Element-wise kernel computation — 6 · H · W ops
    4. Response localisation (argmax) — H · W comparisons

    Args:
        patch_h: Patch height in pixels.
        patch_w: Patch width in pixels.
        num_features: Number of feature channels (1 = raw grayscale).

    Returns:
        :class:`ComputeProfile` with FLOPs estimate.
    """
    n = patch_h * patch_w
    base = mosse_flops(patch_h, patch_w).flops_per_frame

    feature_cost = 8.0 * num_features * n
    kernel_fft1 = _fft2d_flops(patch_h, patch_w)
    kernel_fft2 = _fft2d_flops(patch_h, patch_w)
    kernel_eltwise = 6.0 * n
    argmax = float(n)

    total = base + feature_cost + kernel_fft1 + kernel_fft2 + kernel_eltwise + argmax
    note = (
        f"KCF {patch_h}×{patch_w}: MOSSE base + {num_features}ch HOG features "
        f"+ Gaussian kernel correlation"
    )
    return ComputeProfile(
        tracker_name="kcf",
        patch_size=(patch_h, patch_w),
        flops_per_frame=total,
        params_count=None,
        compute_note=note,
    )


def correlation_filter_flops(
    tracker_name: str,
    patch_h: int,
    patch_w: int,
    num_fft_passes: int = 4,
    feature_channels: int = 1,
) -> ComputeProfile:
    """Generic FLOP estimate for any correlation-filter-based tracker.

    Parameterises the pipeline depth (number of FFT passes) and feature
    richness (channels), making it applicable to CSRT, MedianFlow, MIL,
    and custom filters without modelling them individually.

    Args:
        tracker_name: Human-readable tracker identifier.
        patch_h: Patch height in pixels.
        patch_w: Patch width in pixels.
        num_fft_passes: Number of 2-D FFT passes per frame (default 4).
        feature_channels: Number of feature channels used.

    Returns:
        :class:`ComputeProfile` with FLOPs estimate.
    """
    n = patch_h * patch_w
    fft_cost = num_fft_passes * _fft2d_flops(patch_h, patch_w)
    eltwise_cost = feature_channels * 12.0 * n
    argmax_cost = float(n)
    total = fft_cost + eltwise_cost + argmax_cost

    note = (
        f"{tracker_name} {patch_h}×{patch_w}: {num_fft_passes} FFT passes, "
        f"{feature_channels} feature channel(s)"
    )
    return ComputeProfile(
        tracker_name=tracker_name,
        patch_size=(patch_h, patch_w),
        flops_per_frame=total,
        compute_note=note,
    )


def siamese_tracker_flops(
    tracker_name: str,
    backbone_flops: float,
    search_area_factor: float = 4.0,
    rpn_channels: int = 256,
    num_anchors: int = 5,
    feature_map_size: Tuple[int, int] = (7, 7),
) -> ComputeProfile:
    """Estimate FLOPs per frame for a Siamese-network tracker.

    The template branch runs once at initialisation; only the search branch
    runs each frame.  A lightweight RPN head (classification + regression)
    is added on top of the cross-correlation feature map.

    Args:
        tracker_name: Human-readable tracker identifier.
        backbone_flops: FLOPs for one backbone forward pass on the template.
        search_area_factor: Search-area is this multiple of the template area
            (default 4× for SiamRPN-style trackers).
        rpn_channels: Feature channels in the RPN head.
        num_anchors: Anchors per spatial location in the RPN.
        feature_map_size: Output feature map spatial size ``(H, W)``
            (default ``(7, 7)`` for SiamRPN).

    Returns:
        :class:`ComputeProfile` with FLOPs estimate.  ``patch_size`` is set
        to the standard SiamRPN template size ``(127, 127)``.
    """
    search_flops = backbone_flops * search_area_factor
    fh, fw = feature_map_size
    xcorr_flops = fh * fw * rpn_channels * rpn_channels
    rpn_cls = fh * fw * num_anchors * rpn_channels * 2
    rpn_reg = fh * fw * num_anchors * rpn_channels * 4
    total = search_flops + xcorr_flops + rpn_cls + rpn_reg

    note = (
        f"{tracker_name}: backbone {backbone_flops / 1e6:.1f} MFLOPs "
        f"× {search_area_factor}× search + RPN ({fh}×{fw}, "
        f"{rpn_channels}ch, {num_anchors} anchors)"
    )
    return ComputeProfile(
        tracker_name=tracker_name,
        patch_size=(127, 127),
        flops_per_frame=total,
        compute_note=note,
    )


# ── Registry and comparison interface ─────────────────────────────────────────


class ComputeProfiler:
    """Registry and convenience interface for tracker compute profiles.

    Provides analytic profiles for EOVOT's built-in trackers and allows
    registration of custom profiles for external or DL-based trackers.

    Built-in trackers
    -----------------
    ``"mosse"`` — MOSSE correlation filter
    ``"kcf"`` — Kernelized Correlation Filter
    ``"csrt"`` — Channel-Spatial Reliability Tracking (6 FFT passes, 8ch)
    ``"medianflow"`` — Lucas-Kanade optical flow (2 FFT passes, 1ch)
    ``"mil"`` — Multiple Instance Learning AdaBoost (4 FFT passes, 1ch)

    Example::

        profiler = ComputeProfiler()

        # Single profile
        profile = profiler.profile("kcf", patch_size=(64, 64))
        print(f"KCF: {profile.mega_flops:.3f} MFLOPs/frame")

        # Combine with empirical FPS (e.g. from Profiler)
        gflops_s = profile.throughput_gflops_per_sec(fps=200.0)
        print(f"Effective: {gflops_s:.4f} GFLOPs/s")

        # Compare all built-ins
        print(profiler.comparison_table(patch_size=(64, 64)))
    """

    _BUILTIN_NAMES = frozenset({"mosse", "kcf", "csrt", "medianflow", "mil"})

    # Per-tracker overrides for the generic estimator
    _CSRT_FFT_PASSES = 6
    _CSRT_CHANNELS = 8
    _MEDIANFLOW_FFT_PASSES = 2
    _MIL_FFT_PASSES = 4

    def __init__(self) -> None:
        self._custom: Dict[str, ComputeProfile] = {}

    def register(self, name: str, profile: ComputeProfile) -> None:
        """Register a custom compute profile under *name*.

        Use this to add profiles for DL-based or external trackers that are
        not covered by the built-in analytic estimators.

        Args:
            name: Tracker identifier (case-insensitive lookup key).
            profile: Pre-built :class:`ComputeProfile` instance.
        """
        self._custom[name.lower()] = profile

    def profile(
        self,
        tracker_name: str,
        patch_size: Tuple[int, int] = (64, 64),
        **kwargs,
    ) -> ComputeProfile:
        """Return a :class:`ComputeProfile` for *tracker_name*.

        Built-in trackers are estimated analytically.  For unrecognised
        trackers a generic correlation-filter estimate is returned.

        Args:
            tracker_name: Tracker identifier (case-insensitive).
            patch_size: Template / search patch ``(height, width)`` in pixels.
            **kwargs: Passed to the underlying estimator (e.g.
                ``num_features`` for KCF, ``num_fft_passes`` / ``feature_channels``
                for the generic estimator).

        Returns:
            :class:`ComputeProfile` for the tracker.
        """
        name = tracker_name.lower()
        h, w = patch_size

        if name in self._custom:
            return self._custom[name]

        if name == "mosse":
            return mosse_flops(h, w)

        if name == "kcf":
            return kcf_flops(h, w, num_features=kwargs.get("num_features", 1))

        if name == "csrt":
            return correlation_filter_flops(
                "csrt", h, w, self._CSRT_FFT_PASSES, self._CSRT_CHANNELS
            )

        if name == "medianflow":
            return correlation_filter_flops(
                "medianflow", h, w, self._MEDIANFLOW_FFT_PASSES, 1
            )

        if name == "mil":
            return correlation_filter_flops(
                "mil", h, w, self._MIL_FFT_PASSES, 1
            )

        # Fallback: generic estimate with caller-supplied parameters
        return correlation_filter_flops(
            tracker_name,
            h,
            w,
            num_fft_passes=kwargs.get("num_fft_passes", 4),
            feature_channels=kwargs.get("feature_channels", 1),
        )

    def all_builtin_profiles(
        self, patch_size: Tuple[int, int] = (64, 64)
    ) -> List[ComputeProfile]:
        """Return profiles for all built-in trackers at *patch_size*.

        Args:
            patch_size: Patch size ``(H, W)`` in pixels.

        Returns:
            List of :class:`ComputeProfile` instances sorted by ascending FLOPs.
        """
        profiles = [self.profile(n, patch_size) for n in sorted(self._BUILTIN_NAMES)]
        return sorted(profiles, key=lambda p: p.flops_per_frame)

    def comparison_table(
        self,
        tracker_names: Optional[List[str]] = None,
        patch_size: Tuple[int, int] = (64, 64),
        show_fps_at: Optional[float] = None,
    ) -> str:
        """Return a Markdown table comparing compute profiles.

        Args:
            tracker_names: Names of trackers to compare.  Defaults to all
                built-in trackers when ``None``.
            patch_size: Patch size for all estimates.
            show_fps_at: If provided, add a ``GFLOPs/s`` column computed at
                this empirical FPS value.

        Returns:
            Formatted Markdown table string, sorted by ascending MFLOPs.
        """
        if tracker_names is None:
            profiles = self.all_builtin_profiles(patch_size)
        else:
            profiles = [self.profile(n, patch_size) for n in tracker_names]
            profiles.sort(key=lambda p: p.flops_per_frame)

        if show_fps_at is not None:
            header = (
                "| Tracker | MFLOPs/frame | Patch | GFLOPs/s |\n"
                "|---------|-------------:|------:|---------:|\n"
            )
            rows = [
                f"| {p.tracker_name} | {p.mega_flops:.3f} "
                f"| {p.patch_size[0]}×{p.patch_size[1]} "
                f"| {p.throughput_gflops_per_sec(show_fps_at):.4f} |"
                for p in profiles
            ]
        else:
            header = (
                "| Tracker | MFLOPs/frame | Patch |\n"
                "|---------|-------------:|------:|\n"
            )
            rows = [
                f"| {p.tracker_name} | {p.mega_flops:.3f} "
                f"| {p.patch_size[0]}×{p.patch_size[1]} |"
                for p in profiles
            ]

        return header + "\n".join(rows)

    def efficiency_rank(
        self,
        tracker_fps_map: Dict[str, float],
        patch_size: Tuple[int, int] = (64, 64),
    ) -> List[Tuple[str, float, float, float]]:
        """Rank trackers by compute efficiency (FPS per MFLOPs).

        A tracker that achieves more FPS while using fewer FLOPs is more
        efficient — it extracts more throughput per unit of compute, which
        is the key metric for edge deployment on power-constrained hardware.

        Args:
            tracker_fps_map: Mapping from tracker name to observed FPS.
            patch_size: Patch size for FLOPs estimates.

        Returns:
            List of ``(tracker_name, fps, mega_flops, fps_per_mflop)`` tuples,
            sorted by descending efficiency (fps / mflop).
        """
        records = []
        for name, fps in tracker_fps_map.items():
            p = self.profile(name, patch_size)
            mf = p.mega_flops
            efficiency = fps / mf if mf > 0.0 else float("inf")
            records.append((name, fps, mf, efficiency))
        records.sort(key=lambda r: r[3], reverse=True)
        return records


# re-export for direct import convenience
__all__ = [
    "ComputeProfile",
    "ComputeProfiler",
    "mosse_flops",
    "kcf_flops",
    "correlation_filter_flops",
    "siamese_tracker_flops",
]

# Optional import guard — avoid hard failure when typing stubs are missing
from typing import Optional  # noqa: E402  (already imported at top in 3.11+)

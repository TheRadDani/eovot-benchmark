"""Accuracy–Efficiency Tradeoff (AET) metrics for edge-aware tracker evaluation.

On edge devices the key question is not just *how accurate* a tracker is,
but *how accurate* it is *at a given computational budget*.  The
Accuracy–Efficiency Tradeoff (AET) curve captures this by sweeping the
frame-skip rate (a proxy for computational load) and recording the
resulting mean IoU.

The AET-AUC scalar summarises the curve: a higher value means the tracker
degrades gracefully under temporal sub-sampling — a desirable property for
deployment on Raspberry Pi, Jetson Nano, or MCU platforms.

Usage
-----
::

    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.frame_skip import FrameSkipWrapper
    from eovot.metrics.efficiency import AETCurve, AETPoint

    points = [
        AETPoint(skip_rate=1, mean_iou=0.52, fps=480.0),
        AETPoint(skip_rate=2, mean_iou=0.50, fps=820.0),
        AETPoint(skip_rate=4, mean_iou=0.44, fps=1400.0),
    ]
    curve = AETCurve(tracker_name="MOSSE", points=points)
    print(curve.auc)            # → AUC in [0, 1]
    print(curve.fps_gain(4))    # → relative FPS gain at skip_rate=4
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class AETPoint:
    """A single operating point on the Accuracy–Efficiency Tradeoff curve.

    Attributes:
        skip_rate: Frame-skip rate used to produce this measurement.
            ``1`` = no skipping (baseline).
        mean_iou:  Mean IoU across all evaluated sequences at this skip rate.
        fps:       Effective throughput in frames per second.  For a wrapped
            tracker this equals ``base_fps × skip_rate`` (skipped frames are
            free), subject to hardware variability.
        memory_mb: Peak memory usage in MiB (optional).
    """

    skip_rate: int
    mean_iou: float
    fps: float
    memory_mb: Optional[float] = None

    def to_dict(self) -> Dict:
        d = {
            "skip_rate": self.skip_rate,
            "mean_iou": round(self.mean_iou, 4),
            "fps": round(self.fps, 2),
        }
        if self.memory_mb is not None:
            d["memory_mb"] = round(self.memory_mb, 2)
        return d


@dataclass
class AETCurve:
    """Accuracy–Efficiency Tradeoff curve for a tracker under varying skip rates.

    The curve is built from a list of :class:`AETPoint` objects ordered by
    increasing *skip_rate* (equivalently, increasing FPS).

    The AUC is computed by integrating mean IoU over normalised FPS, so that
    a tracker maintaining accuracy across the full throughput range scores
    close to 1.0, while one that collapses at the first skip scores near 0.

    Attributes:
        tracker_name: Identifier of the evaluated tracker.
        points:       Ordered list of operating points (sorted by skip_rate).
    """

    tracker_name: str
    points: List[AETPoint] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.points = sorted(self.points, key=lambda p: p.skip_rate)

    # ------------------------------------------------------------------
    # Scalar summaries
    # ------------------------------------------------------------------

    @property
    def auc(self) -> float:
        """Area Under the AET Curve, normalised to ``[0, 1]``.

        Integrates mean IoU over normalised FPS (0 = baseline, 1 = max
        measured FPS).  Uses the trapezoidal rule.

        Returns 0.0 if fewer than 2 points are available.
        """
        if len(self.points) < 2:
            return self.points[0].mean_iou if self.points else 0.0

        ious = np.array([p.mean_iou for p in self.points], dtype=np.float64)
        fps  = np.array([p.fps      for p in self.points], dtype=np.float64)

        fps_range = fps[-1] - fps[0]
        if fps_range <= 0:
            return float(ious.mean())

        fps_norm = (fps - fps[0]) / fps_range  # 0 … 1
        _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # type: ignore[attr-defined]
        return float(_trapz(ious, fps_norm))

    @property
    def baseline(self) -> Optional[AETPoint]:
        """Operating point with ``skip_rate=1`` (no skipping), or ``None``."""
        for p in self.points:
            if p.skip_rate == 1:
                return p
        return None

    def fps_gain(self, skip_rate: int) -> Optional[float]:
        """Relative FPS speedup at *skip_rate* compared to the baseline.

        Args:
            skip_rate: Target skip rate to query.

        Returns:
            ``fps(skip_rate) / fps(1)`` or ``None`` if the baseline or the
            requested skip rate is not in :attr:`points`.
        """
        base = self.baseline
        if base is None or base.fps <= 0:
            return None
        for p in self.points:
            if p.skip_rate == skip_rate:
                return p.fps / base.fps
        return None

    def iou_drop(self, skip_rate: int) -> Optional[float]:
        """Absolute IoU drop at *skip_rate* relative to the no-skip baseline.

        Args:
            skip_rate: Target skip rate to query.

        Returns:
            ``iou(1) - iou(skip_rate)``, a non-negative float, or ``None``
            if the required points are missing.
        """
        base = self.baseline
        if base is None:
            return None
        for p in self.points:
            if p.skip_rate == skip_rate:
                return base.mean_iou - p.mean_iou
        return None

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """Serialise to a plain dict suitable for JSON export."""
        return {
            "tracker_name": self.tracker_name,
            "auc": round(self.auc, 4),
            "points": [p.to_dict() for p in self.points],
        }

    def summary_table(self) -> str:
        """Return a Markdown table string summarising the AET curve."""
        lines = [
            f"### AET Curve — {self.tracker_name}",
            "",
            f"AUC = **{self.auc:.4f}**",
            "",
            "| skip_rate | mean_IoU | FPS   | FPS gain | IoU drop |",
            "|----------:|--------:|------:|---------:|---------:|",
        ]
        for p in self.points:
            gain = self.fps_gain(p.skip_rate)
            drop = self.iou_drop(p.skip_rate)
            gain_s = f"{gain:.2f}×" if gain is not None else "—"
            drop_s = f"{drop:.4f}"  if drop is not None else "—"
            lines.append(
                f"| {p.skip_rate:>9d} "
                f"| {p.mean_iou:>8.4f} "
                f"| {p.fps:>5.1f} "
                f"| {gain_s:>8s} "
                f"| {drop_s:>8s} |"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"AETCurve(tracker={self.tracker_name!r}, "
            f"points={len(self.points)}, auc={self.auc:.4f})"
        )


def build_aet_curve(
    tracker_name: str,
    skip_rates: List[int],
    mean_ious: List[float],
    fps_values: List[float],
    memory_mb: Optional[List[float]] = None,
) -> AETCurve:
    """Convenience constructor for :class:`AETCurve` from parallel lists.

    Args:
        tracker_name: Tracker identifier.
        skip_rates:   List of skip rate values (e.g. ``[1, 2, 4, 8]``).
        mean_ious:    Corresponding mean IoU values.
        fps_values:   Corresponding FPS values.
        memory_mb:    Optional list of peak memory values (MiB).

    Returns:
        :class:`AETCurve` with one :class:`AETPoint` per entry.

    Raises:
        ValueError: If input lists have different lengths.
    """
    n = len(skip_rates)
    if not (len(mean_ious) == n == len(fps_values)):
        raise ValueError(
            "skip_rates, mean_ious, and fps_values must have the same length."
        )
    mem_list: List[Optional[float]] = memory_mb if memory_mb is not None else [None] * n
    points = [
        AETPoint(
            skip_rate=skip_rates[i],
            mean_iou=mean_ious[i],
            fps=fps_values[i],
            memory_mb=mem_list[i],
        )
        for i in range(n)
    ]
    return AETCurve(tracker_name=tracker_name, points=points)

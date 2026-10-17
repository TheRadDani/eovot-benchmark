"""Multi-resolution sweep evaluator for accuracy-latency Pareto analysis.

Reducing input resolution is the most effective single-knob optimisation for
edge tracker deployment: halving frame dimensions typically triples throughput
with only a modest IoU penalty.  The inflection point of this tradeoff is
tracker-specific — correlation-filter trackers (MOSSE, KCF) tolerate
downscaling well, while appearance-model trackers degrade sharply.

This module provides:

:class:`ResolutionWrapper`
    Wraps any :class:`~eovot.trackers.base.BaseTracker` with automatic
    frame rescaling.  Predictions are mapped back to original coordinates so
    the wrapper is transparent to the benchmark engine and metrics pipeline.

:class:`ScalePoint`
    Accuracy and throughput summary at one resolution scale.

:class:`ResolutionSweepResult`
    All :class:`ScalePoint` objects from a sweep, with Pareto-front flags set.

:class:`ResolutionSweepEvaluator`
    Runs the full benchmark at each candidate scale, identifies the Pareto
    front in (mIoU, FPS) space, and provides helpers to select the optimal
    operating point for a given throughput budget.

Example::

    from eovot.benchmark.resolution_sweep import ResolutionSweepEvaluator
    from eovot.trackers.mosse import MOSSETracker
    from eovot.datasets.synthetic import SyntheticDataset

    dataset   = SyntheticDataset(num_sequences=3, num_frames=60)
    evaluator = ResolutionSweepEvaluator(scales=[0.25, 0.5, 0.75, 1.0])
    result    = evaluator.evaluate(MOSSETracker, dataset, dataset_name="Synthetic")

    print(evaluator.to_markdown_table(result))

    # Find best resolution that meets a 30-FPS budget
    point = evaluator.best_for_fps_target(result, fps_target=30.0)
    if point:
        print(f"Optimal scale for ≥30 FPS: {point.scale:.2f}x  mIoU={point.mean_iou:.4f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import cv2
import numpy as np

from ..datasets.base import BaseDataset
from ..trackers.base import BaseTracker, BBox
from .engine import BenchmarkEngine, BenchmarkResult


# ---------------------------------------------------------------------------
# Resolution wrapper
# ---------------------------------------------------------------------------

class ResolutionWrapper(BaseTracker):
    """Wraps a :class:`~eovot.trackers.base.BaseTracker` with frame rescaling.

    Frames are rescaled by *scale* before being forwarded to the inner tracker.
    Predicted bounding boxes are rescaled back to original-frame coordinates
    before being returned, making the wrapper transparent to the benchmark engine.

    Args:
        tracker: Pre-constructed inner tracker instance.
        scale:   Linear scale factor applied to both spatial dimensions.
                 ``0.5`` halves width and height; ``1.0`` is a no-op.
                 Must be in ``(0, 2]``.
    """

    def __init__(self, tracker: BaseTracker, scale: float) -> None:
        if not (0.0 < scale <= 2.0):
            raise ValueError(f"scale must be in (0, 2], got {scale!r}")
        super().__init__(name=f"{tracker.name}@{scale:.2f}x")
        self._inner = tracker
        self.scale = scale

    # ------------------------------------------------------------------
    # Internal coordinate helpers
    # ------------------------------------------------------------------

    def _scale_frame(self, frame: np.ndarray) -> np.ndarray:
        if abs(self.scale - 1.0) < 1e-9:
            return frame
        h, w = frame.shape[:2]
        new_w = max(1, int(round(w * self.scale)))
        new_h = max(1, int(round(h * self.scale)))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def _bbox_to_scaled(self, bbox: BBox) -> BBox:
        x, y, w, h = bbox
        s = self.scale
        return (x * s, y * s, w * s, h * s)

    def _bbox_to_original(self, bbox: BBox) -> BBox:
        x, y, w, h = bbox
        s = self.scale
        return (x / s, y / s, w / s, h / s)

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Rescale *frame* and *bbox*, then initialise the inner tracker."""
        self._inner.initialize(self._scale_frame(frame), self._bbox_to_scaled(bbox))

    def update(self, frame: np.ndarray) -> BBox:
        """Track on a rescaled frame; return box in original-frame coordinates."""
        small_box = self._inner.update(self._scale_frame(frame))
        return self._bbox_to_original(small_box)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ScalePoint:
    """Accuracy and throughput summary at one resolution scale.

    Attributes:
        scale:            Linear scale factor (e.g. ``0.5`` for half resolution).
        mean_iou:         Mean IoU across all evaluated frames.
        mean_fps:         Mean throughput in frames per second.
        peak_memory_mb:   Peak RSS memory footprint in megabytes.
        success_auc:      Area under the success curve, or ``None`` if not computed.
        precision_auc:    Normalised precision-curve AUC, or ``None`` if not computed.
        benchmark_result: Full :class:`~eovot.benchmark.engine.BenchmarkResult`
                          produced at this scale (excluded from repr for brevity).
    """

    scale: float
    mean_iou: float
    mean_fps: float
    peak_memory_mb: float
    success_auc: Optional[float]
    precision_auc: Optional[float]
    benchmark_result: BenchmarkResult = field(repr=False)
    _on_pareto_front: bool = field(default=False, init=False, repr=False)

    @property
    def on_pareto_front(self) -> bool:
        """``True`` if no other scale dominates both mIoU and FPS."""
        return self._on_pareto_front

    @on_pareto_front.setter
    def on_pareto_front(self, value: bool) -> None:
        self._on_pareto_front = value

    def __str__(self) -> str:
        pareto = "✓" if self.on_pareto_front else " "
        sauc = f"{self.success_auc:.4f}" if self.success_auc is not None else "—"
        return (
            f"[{pareto}] scale={self.scale:.2f}x  "
            f"mIoU={self.mean_iou:.4f}  FPS={self.mean_fps:.1f}  "
            f"AUC={sauc}  mem={self.peak_memory_mb:.0f} MB"
        )


@dataclass
class ResolutionSweepResult:
    """Full sweep result: one :class:`ScalePoint` per evaluated resolution.

    Attributes:
        tracker_name: Base tracker identifier (without scale suffix).
        dataset_name: Dataset the sweep was evaluated on.
        points:       Scale points sorted by scale (ascending).
    """

    tracker_name: str
    dataset_name: str
    points: List[ScalePoint] = field(default_factory=list)

    @property
    def pareto_front(self) -> List[ScalePoint]:
        """Scale points that lie on the accuracy-throughput Pareto front."""
        return [p for p in self.points if p.on_pareto_front]

    def best_iou(self) -> Optional[ScalePoint]:
        """Scale with the highest mean IoU."""
        return max(self.points, key=lambda p: p.mean_iou) if self.points else None

    def best_fps(self) -> Optional[ScalePoint]:
        """Scale with the highest mean FPS."""
        return max(self.points, key=lambda p: p.mean_fps) if self.points else None

    def at_scale(self, scale: float, tol: float = 0.01) -> Optional[ScalePoint]:
        """Return the :class:`ScalePoint` closest to *scale*, within *tol*."""
        for p in self.points:
            if abs(p.scale - scale) <= tol:
                return p
        return None


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ResolutionSweepEvaluator:
    """Evaluate a tracker at multiple input resolutions and identify the Pareto front.

    For each scale in ``scales`` a fresh :class:`ResolutionWrapper` is built
    around a new tracker instance and the full benchmark is run.  The Pareto
    front in ``(mIoU, FPS)`` space is computed, flagging scales where no other
    resolution simultaneously achieves higher accuracy *and* throughput.

    Args:
        scales:        Linear scale factors to sweep.  ``1.0`` is full resolution.
                       Default: ``[0.25, 0.5, 0.75, 1.0]``.
        max_sequences: Maximum sequences to evaluate per scale run.  ``None``
                       uses the full dataset.  Useful for quick sweeps.
        tdp_watts:     If provided, enables energy profiling in the engine.
        verbose:       Print per-scale benchmark progress.  Default: ``True``.
    """

    DEFAULT_SCALES: List[float] = [0.25, 0.5, 0.75, 1.0]

    def __init__(
        self,
        scales: Optional[List[float]] = None,
        max_sequences: Optional[int] = None,
        tdp_watts: Optional[float] = None,
        verbose: bool = True,
    ) -> None:
        self.scales = sorted(set(scales or self.DEFAULT_SCALES))
        self.max_sequences = max_sequences
        self.tdp_watts = tdp_watts
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        tracker_cls: Type[BaseTracker],
        dataset: BaseDataset,
        tracker_kwargs: Optional[Dict] = None,
        dataset_name: str = "unknown",
    ) -> ResolutionSweepResult:
        """Run the benchmark at each scale and return a :class:`ResolutionSweepResult`.

        A fresh tracker instance is created for every scale to avoid any state
        leakage between runs.

        Args:
            tracker_cls:    Tracker *class* (not instance) to instantiate per scale.
            dataset:        Dataset to evaluate on.
            tracker_kwargs: Kwargs forwarded to ``tracker_cls(**kwargs)`` at each scale.
            dataset_name:   Human-readable label used in reports.

        Returns:
            :class:`ResolutionSweepResult` with Pareto-front flags set and points
            sorted by scale (ascending).
        """
        kwargs: Dict = tracker_kwargs or {}
        engine = BenchmarkEngine(verbose=self.verbose, tdp_watts=self.tdp_watts)

        # Determine the tracker's base name from a throwaway instance.
        base_name = tracker_cls(**kwargs).name

        points: List[ScalePoint] = []

        for scale in self.scales:
            if self.verbose:
                print(f"\n{'─' * 50}")
                print(f"Resolution sweep — scale {scale:.2f}x  ({base_name})")
                print(f"{'─' * 50}")

            wrapped = ResolutionWrapper(tracker=tracker_cls(**kwargs), scale=scale)
            result = engine.run(
                wrapped,
                dataset,
                dataset_name=dataset_name,
                max_sequences=self.max_sequences,
            )
            points.append(
                ScalePoint(
                    scale=scale,
                    mean_iou=result.mean_iou,
                    mean_fps=result.mean_fps,
                    peak_memory_mb=result.peak_memory_mb,
                    success_auc=result.mean_success_auc,
                    precision_auc=result.mean_precision_auc,
                    benchmark_result=result,
                )
            )

        sweep = ResolutionSweepResult(
            tracker_name=base_name,
            dataset_name=dataset_name,
            points=sorted(points, key=lambda p: p.scale),
        )
        self._mark_pareto_front(sweep.points)
        return sweep

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def best_for_fps_target(
        self, result: ResolutionSweepResult, fps_target: float
    ) -> Optional[ScalePoint]:
        """Return the highest-IoU scale that meets *fps_target*.

        Among all scale points where ``mean_fps >= fps_target``, selects the
        one with the highest ``mean_iou``.

        Args:
            result:     Output of :meth:`evaluate`.
            fps_target: Minimum required FPS (e.g. ``30.0`` for real-time).

        Returns:
            Best :class:`ScalePoint` meeting the FPS constraint, or ``None``
            if no scale reaches the target.
        """
        eligible = [p for p in result.points if p.mean_fps >= fps_target]
        if not eligible:
            return None
        return max(eligible, key=lambda p: p.mean_iou)

    def iou_retention(
        self, result: ResolutionSweepResult, reference_scale: float = 1.0
    ) -> Dict[float, float]:
        """Compute IoU retention ratio relative to a reference scale.

        ``retention = mean_iou_at_scale / mean_iou_at_reference``.  A value
        of ``0.90`` at ``scale=0.5`` means the tracker retains 90 % of its
        full-resolution accuracy at half resolution.

        Args:
            result:          Output of :meth:`evaluate`.
            reference_scale: Scale to use as 100 % baseline.  Default: ``1.0``.

        Returns:
            Mapping ``{scale: retention_ratio}``.
            Returns an empty dict if the reference scale is not in the sweep.
        """
        ref = result.at_scale(reference_scale)
        if ref is None or ref.mean_iou == 0.0:
            return {}
        return {
            p.scale: round(p.mean_iou / ref.mean_iou, 4)
            for p in result.points
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def to_markdown_table(self, result: ResolutionSweepResult) -> str:
        """Format the sweep result as a Markdown table.

        Args:
            result: Output of :meth:`evaluate`.

        Returns:
            Multi-line Markdown string ready to embed in reports or READMEs.
        """
        lines = [
            f"## Resolution Sweep: {result.tracker_name} on {result.dataset_name}",
            "",
            "| Scale | mIoU | FPS | Mem (MB) | Success AUC | Precision AUC | Pareto |",
            "|------:|-----:|----:|---------:|------------:|--------------:|:------:|",
        ]
        for p in result.points:
            sauc = f"{p.success_auc:.4f}" if p.success_auc is not None else "—"
            pauc = f"{p.precision_auc:.4f}" if p.precision_auc is not None else "—"
            pareto = "✓" if p.on_pareto_front else ""
            lines.append(
                f"| {p.scale:.2f} | {p.mean_iou:.4f} | {p.mean_fps:.1f} "
                f"| {p.peak_memory_mb:.0f} | {sauc} | {pauc} | {pareto} |"
            )
        return "\n".join(lines)

    def to_summary_dict(self, result: ResolutionSweepResult) -> dict:
        """Convert the sweep result to a JSON-serialisable plain dict.

        Args:
            result: Output of :meth:`evaluate`.

        Returns:
            Dict with ``tracker_name``, ``dataset_name``, and ``scales`` list.
        """
        return {
            "tracker_name": result.tracker_name,
            "dataset_name": result.dataset_name,
            "scales": [
                {
                    "scale": p.scale,
                    "mean_iou": round(p.mean_iou, 4),
                    "mean_fps": round(p.mean_fps, 2),
                    "peak_memory_mb": round(p.peak_memory_mb, 1),
                    "success_auc": round(p.success_auc, 4) if p.success_auc is not None else None,
                    "precision_auc": (
                        round(p.precision_auc, 4) if p.precision_auc is not None else None
                    ),
                    "on_pareto_front": p.on_pareto_front,
                }
                for p in result.points
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mark_pareto_front(points: List[ScalePoint]) -> None:
        """Flag Pareto-optimal points in (mIoU, FPS) space (in-place).

        Point A dominates point B iff ``A.mean_iou >= B.mean_iou`` AND
        ``A.mean_fps >= B.mean_fps`` with at least one strict inequality.
        Non-dominated points are Pareto-optimal.
        """
        for i, candidate in enumerate(points):
            dominated = any(
                (
                    other.mean_iou >= candidate.mean_iou
                    and other.mean_fps >= candidate.mean_fps
                    and (other.mean_iou > candidate.mean_iou or other.mean_fps > candidate.mean_fps)
                )
                for j, other in enumerate(points)
                if j != i
            )
            candidate.on_pareto_front = not dominated

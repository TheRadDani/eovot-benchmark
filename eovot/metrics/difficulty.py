"""Sequence difficulty analysis for EOVOT benchmark sequences.

Characterises how challenging each tracking sequence is along independent
difficulty axes derived from ground-truth bounding boxes alone.  No tracker
output is required — difficulty is an intrinsic property of the target's
motion and shape dynamics.

Difficulty Axes
~~~~~~~~~~~~~~~
- **motion_magnitude**: Mean frame-to-frame centre displacement normalised by
  the target diagonal.  High values indicate fast motion relative to target size.
- **scale_change**: Std of log-ratio of consecutive target areas.  High values
  indicate dramatic zooming or foreshortening.
- **aspect_ratio_change**: Std of log-ratio of consecutive aspect ratios.
  High values indicate shape deformation or out-of-plane rotation.
- **size_ratio**: Max/min target area across the sequence.  Values >> 1
  indicate extreme scale variation.
- **overall_score**: Composite difficulty score in ``[0, 1]`` (higher = harder).
- **label**: Categorical bucket — ``"easy"``, ``"medium"``, or ``"hard"``.

The composite score uses sigmoid-normalised axes with weights calibrated
against OTB-100 empirical failure statistics (motion 40 %, scale 30 %,
aspect ratio 15 %, size span 15 %).

Typical usage::

    from eovot.metrics.difficulty import SequenceDifficultyAnalyzer

    analyzer = SequenceDifficultyAnalyzer()

    # Analyse one sequence
    report = analyzer.analyze(seq.ground_truth, sequence_name=seq.name)
    print(report)

    # Analyse an entire dataset
    reports = analyzer.analyze_dataset(dataset)

    # Attribute-based tracker performance breakdown
    breakdown = analyzer.performance_by_difficulty(benchmark_result, dataset)
    print(SequenceDifficultyAnalyzer.to_markdown_table([breakdown]))
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult


@dataclass
class DifficultyReport:
    """Per-sequence difficulty characterisation derived from ground-truth boxes.

    Attributes:
        sequence_name: Human-readable identifier of the analysed sequence.
        num_frames: Number of ground-truth frames analysed.
        motion_magnitude: Mean normalised frame-to-frame target-centre displacement.
            Values above ~0.06 correspond to fast motion on OTB-100.
        scale_change: Std of log area-change ratios between consecutive frames.
            Values above ~0.15 indicate moderate scale variation.
        aspect_ratio_change: Std of log aspect-ratio changes between consecutive
            frames.  Values above ~0.10 indicate shape deformation.
        size_ratio: Ratio of maximum to minimum target area across the sequence.
            Values above 4 indicate large-scale variation.
        overall_score: Composite difficulty score in ``[0, 1]``.
        label: Categorical label — ``"easy"``, ``"medium"``, or ``"hard"``.
    """

    sequence_name: str
    num_frames: int
    motion_magnitude: float
    scale_change: float
    aspect_ratio_change: float
    size_ratio: float
    overall_score: float
    label: str

    def to_dict(self) -> Dict:
        """Serialise to a plain dict suitable for JSON export."""
        return {
            "sequence_name": self.sequence_name,
            "num_frames": self.num_frames,
            "motion_magnitude": round(self.motion_magnitude, 4),
            "scale_change": round(self.scale_change, 4),
            "aspect_ratio_change": round(self.aspect_ratio_change, 4),
            "size_ratio": round(self.size_ratio, 4),
            "overall_score": round(self.overall_score, 4),
            "label": self.label,
        }

    def __str__(self) -> str:
        return (
            f"DifficultyReport({self.sequence_name!r}  "
            f"label={self.label!r}  score={self.overall_score:.3f}  "
            f"motion={self.motion_magnitude:.3f}  scale_Δ={self.scale_change:.3f}  "
            f"AR_Δ={self.aspect_ratio_change:.3f}  size_ratio={self.size_ratio:.2f}x  "
            f"frames={self.num_frames})"
        )


@dataclass
class AttributeBreakdown:
    """Tracker performance split by sequence difficulty bucket.

    Attributes:
        tracker_name: Human-readable identifier of the evaluated tracker.
        dataset_name: Dataset on which the tracker was evaluated.
        easy: Mean IoU on sequences labelled ``"easy"``; ``None`` if no such sequences.
        medium: Mean IoU on sequences labelled ``"medium"``; ``None`` if absent.
        hard: Mean IoU on sequences labelled ``"hard"``; ``None`` if absent.
        counts: Number of sequences in each bucket.
    """

    tracker_name: str
    dataset_name: str
    easy: Optional[float]
    medium: Optional[float]
    hard: Optional[float]
    counts: Dict[str, int]

    def to_dict(self) -> Dict:
        """Serialise to a plain dict suitable for JSON export."""
        return {
            "tracker": self.tracker_name,
            "dataset": self.dataset_name,
            "mean_iou_easy": self.easy,
            "mean_iou_medium": self.medium,
            "mean_iou_hard": self.hard,
            "counts": self.counts,
        }

    def __str__(self) -> str:
        easy_s = f"{self.easy:.3f}" if self.easy is not None else "n/a"
        med_s = f"{self.medium:.3f}" if self.medium is not None else "n/a"
        hard_s = f"{self.hard:.3f}" if self.hard is not None else "n/a"
        return (
            f"AttributeBreakdown({self.tracker_name!r} on {self.dataset_name!r}  "
            f"easy={easy_s} [{self.counts.get('easy', 0)} seq]  "
            f"medium={med_s} [{self.counts.get('medium', 0)} seq]  "
            f"hard={hard_s} [{self.counts.get('hard', 0)} seq])"
        )


class SequenceDifficultyAnalyzer:
    """Characterise per-sequence difficulty from ground-truth bounding boxes.

    Difficulty is computed from the target's ground-truth trajectory alone —
    no tracker output is required.  This makes it possible to pre-classify a
    dataset before running any tracker, enabling stratified benchmarking.

    Args:
        easy_threshold: Sequences with ``overall_score`` below this value are
            labelled ``"easy"``.  Default: ``0.35``.
        hard_threshold: Sequences with ``overall_score`` at or above this value
            are labelled ``"hard"``; those in between are ``"medium"``.
            Default: ``0.65``.

    Example::

        from eovot.metrics.difficulty import SequenceDifficultyAnalyzer

        analyzer = SequenceDifficultyAnalyzer()

        # Single sequence
        report = analyzer.analyze(seq.ground_truth, seq.name)
        print(report)

        # Full dataset
        reports = analyzer.analyze_dataset(dataset)

        # Attribute breakdown for one tracker result
        breakdown = analyzer.performance_by_difficulty(result, dataset)
    """

    def __init__(
        self,
        easy_threshold: float = 0.35,
        hard_threshold: float = 0.65,
    ) -> None:
        if not (0.0 < easy_threshold < hard_threshold < 1.0):
            raise ValueError(
                "Thresholds must satisfy 0 < easy_threshold < hard_threshold < 1. "
                f"Got easy={easy_threshold}, hard={hard_threshold}."
            )
        self.easy_threshold = easy_threshold
        self.hard_threshold = hard_threshold

    # ------------------------------------------------------------------
    # Primary analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        gt_boxes: np.ndarray,
        sequence_name: str = "unknown",
    ) -> DifficultyReport:
        """Compute difficulty metrics for one ground-truth trajectory.

        Args:
            gt_boxes: Ground-truth bounding boxes of shape ``(N, 4)`` in
                ``(x, y, w, h)`` pixel coordinates.  At least 2 frames are
                required for temporal metrics.
            sequence_name: Human-readable identifier stored in the returned report.

        Returns:
            :class:`DifficultyReport` with per-axis scores and a difficulty label.

        Raises:
            ValueError: If ``gt_boxes`` has fewer than 2 rows or wrong shape.
        """
        gt = np.asarray(gt_boxes, dtype=np.float64)
        if gt.ndim != 2 or gt.shape[1] != 4:
            raise ValueError(
                f"gt_boxes must have shape (N, 4), got {gt.shape}."
            )
        if len(gt) < 2:
            raise ValueError(
                f"At least 2 frames are needed for temporal analysis, got {len(gt)}."
            )

        motion = self._motion_magnitude(gt)
        scale = self._scale_change(gt)
        ar_change = self._aspect_ratio_change(gt)
        size_ratio = self._size_ratio(gt)
        overall = self._composite_score(motion, scale, ar_change, size_ratio)
        label = self._label(overall)

        return DifficultyReport(
            sequence_name=sequence_name,
            num_frames=len(gt),
            motion_magnitude=float(motion),
            scale_change=float(scale),
            aspect_ratio_change=float(ar_change),
            size_ratio=float(size_ratio),
            overall_score=float(overall),
            label=label,
        )

    def analyze_dataset(self, sequences) -> List[DifficultyReport]:
        """Analyse all sequences in a dataset.

        Args:
            sequences: Any iterable of sequences exposing a ``.ground_truth``
                property (``ndarray`` of shape ``(N, 4)``) and a ``.name``
                attribute.  Sequences that fail validation are silently skipped.

        Returns:
            List of :class:`DifficultyReport` objects, one per valid sequence.
        """
        reports: List[DifficultyReport] = []
        for seq in sequences:
            try:
                report = self.analyze(seq.ground_truth, sequence_name=seq.name)
                reports.append(report)
            except (ValueError, AttributeError):
                continue
        return reports

    # ------------------------------------------------------------------
    # Attribute-based performance breakdown
    # ------------------------------------------------------------------

    def performance_by_difficulty(
        self,
        benchmark_result: "BenchmarkResult",
        sequences,
    ) -> AttributeBreakdown:
        """Split tracker performance into easy / medium / hard buckets.

        Each sequence is first classified by its ground-truth trajectory, then
        the tracker's per-sequence mean IoU is averaged within each bucket.

        This enables statements like *"MOSSE achieves 0.72 mIoU on easy
        sequences but drops to 0.31 on hard ones"*, which is the standard
        attribute-based analysis in VOT papers.

        Args:
            benchmark_result: Evaluated tracker result from
                :class:`~eovot.benchmark.engine.BenchmarkEngine`.
            sequences: Iterable of sequences in the **same order** as
                ``benchmark_result.sequence_results``.

        Returns:
            :class:`AttributeBreakdown` with per-bucket mean IoU and counts.
        """
        reports: Dict[str, DifficultyReport] = {}
        for seq in sequences:
            try:
                r = self.analyze(seq.ground_truth, sequence_name=seq.name)
                reports[r.sequence_name] = r
            except (ValueError, AttributeError):
                continue

        buckets: Dict[str, List[float]] = {"easy": [], "medium": [], "hard": []}
        for sr in benchmark_result.sequence_results:
            rpt = reports.get(sr.sequence_name)
            if rpt is not None:
                buckets[rpt.label].append(sr.mean_iou)

        def _mean(vals: List[float]) -> Optional[float]:
            return float(np.mean(vals)) if vals else None

        return AttributeBreakdown(
            tracker_name=benchmark_result.tracker_name,
            dataset_name=benchmark_result.dataset_name,
            easy=_mean(buckets["easy"]),
            medium=_mean(buckets["medium"]),
            hard=_mean(buckets["hard"]),
            counts={k: len(v) for k, v in buckets.items()},
        )

    @staticmethod
    def to_markdown_table(breakdowns: List[AttributeBreakdown]) -> str:
        """Format multiple :class:`AttributeBreakdown` objects as a Markdown table.

        Args:
            breakdowns: One entry per tracker, typically from repeated calls to
                :meth:`performance_by_difficulty`.

        Returns:
            Multi-line Markdown table string ready to embed in reports or READMEs.
        """
        lines = [
            "| Tracker | Dataset | Easy mIoU | Medium mIoU | Hard mIoU"
            " | Easy # | Med # | Hard # |",
            "|---------|---------|----------:|------------:|----------:"
            "|-------:|------:|-------:|",
        ]
        for b in breakdowns:
            easy = f"{b.easy:.4f}" if b.easy is not None else "—"
            med = f"{b.medium:.4f}" if b.medium is not None else "—"
            hard = f"{b.hard:.4f}" if b.hard is not None else "—"
            lines.append(
                f"| {b.tracker_name} | {b.dataset_name} "
                f"| {easy} | {med} | {hard} "
                f"| {b.counts.get('easy', 0)} "
                f"| {b.counts.get('medium', 0)} "
                f"| {b.counts.get('hard', 0)} |"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private per-axis metric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _motion_magnitude(gt: np.ndarray) -> float:
        """Mean frame-to-frame centre displacement normalised by target diagonal."""
        cx = gt[:, 0] + gt[:, 2] / 2.0
        cy = gt[:, 1] + gt[:, 3] / 2.0
        disp = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)
        diag = np.sqrt(gt[:, 2] ** 2 + gt[:, 3] ** 2)
        ref = np.maximum(diag[:-1], 1.0)
        return float(np.mean(disp / ref))

    @staticmethod
    def _scale_change(gt: np.ndarray) -> float:
        """Std of log area-change ratios between consecutive frames."""
        areas = np.maximum(gt[:, 2] * gt[:, 3], 1.0)
        return float(np.std(np.log(areas[1:] / areas[:-1])))

    @staticmethod
    def _aspect_ratio_change(gt: np.ndarray) -> float:
        """Std of log aspect-ratio changes between consecutive frames."""
        ar = np.maximum(gt[:, 2], 1.0) / np.maximum(gt[:, 3], 1.0)
        return float(np.std(np.log(ar[1:] / ar[:-1])))

    @staticmethod
    def _size_ratio(gt: np.ndarray) -> float:
        """Ratio of maximum to minimum target area across the sequence."""
        areas = np.maximum(gt[:, 2] * gt[:, 3], 1.0)
        return float(areas.max() / areas.min())

    def _composite_score(
        self,
        motion: float,
        scale: float,
        ar_change: float,
        size_ratio: float,
    ) -> float:
        """Weighted composite difficulty score in ``[0, 1]``.

        Each axis is mapped through a sigmoid centred at an empirical inflection
        point derived from OTB-100 sequence statistics.  Weights reflect the
        relative contribution of each axis to tracker failure on standard VOT
        benchmarks (motion dominant, then scale, then the shape axes).
        """
        def _sig(x: float, inflection: float) -> float:
            k = 10.0 / max(inflection, 1e-6)
            return 1.0 / (1.0 + math.exp(-k * (x - inflection)))

        s_motion = _sig(motion, 0.06)
        s_scale = _sig(scale, 0.15)
        s_ar = _sig(ar_change, 0.10)
        # size_ratio ∈ [1, ∞) → log-compress before sigmoiding
        s_size = _sig(math.log1p(max(size_ratio - 1.0, 0.0)), math.log(4.0))

        score = 0.40 * s_motion + 0.30 * s_scale + 0.15 * s_ar + 0.15 * s_size
        return min(max(float(score), 0.0), 1.0)

    def _label(self, score: float) -> str:
        if score < self.easy_threshold:
            return "easy"
        if score < self.hard_threshold:
            return "medium"
        return "hard"

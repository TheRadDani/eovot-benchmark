"""Attribute-stratified performance breakdown.

This module computes per-attribute success AUC and mean IoU, enabling
researchers to identify which tracking challenges most affect a given
tracker — the standard analysis methodology in VOT papers.

Example::

    from eovot.analysis import AttributeBreakdown
    from eovot.benchmark.engine import BenchmarkResult

    breakdown = AttributeBreakdown()

    # Single tracker, single sequence
    result = breakdown.compute(seq_result.ground_truths, ious=seq_result.ious)
    for attr, r in result.items():
        print(f"{attr}: mIoU={r.mean_iou:.3f}  AUC={r.success_auc:.3f}")

    # Compare multiple trackers across all sequences in a BenchmarkResult
    comparison = breakdown.from_benchmark_results(
        benchmark_results={"MOSSE": mosse_result, "KCF": kcf_result},
    )
    print(comparison.to_markdown())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .attributes import AttributeDetector, SequenceAttribute


# IoU thresholds used for success-curve AUC (standard VOT protocol).
_DEFAULT_THRESHOLDS = np.linspace(0.0, 1.0, 101)
_MIN_ATTRIBUTE_FRAMES = 5   # skip attribute if too few frames are flagged


@dataclass
class BreakdownResult:
    """Per-attribute performance summary for one tracker.

    Attributes:
        attribute:    Attribute name string.
        num_frames:   Number of frames where this attribute is active.
        mean_iou:     Mean IoU on attribute-flagged frames.
        success_auc:  AUC of the success curve on attribute-flagged frames.
    """

    attribute: str
    num_frames: int
    mean_iou: float
    success_auc: float

    def __str__(self) -> str:
        return (
            f"BreakdownResult[{self.attribute}] "
            f"frames={self.num_frames}  "
            f"mIoU={self.mean_iou:.4f}  "
            f"AUC={self.success_auc:.4f}"
        )


@dataclass
class TrackerAttributeComparison:
    """Attribute breakdown results for multiple trackers.

    Attributes:
        tracker_breakdowns: ``{tracker_name: {attribute: BreakdownResult}}``.
    """

    tracker_breakdowns: Dict[str, Dict[str, BreakdownResult]] = field(
        default_factory=dict
    )

    def to_markdown(self) -> str:
        """Render a Markdown table: rows = attributes, columns = trackers.

        Returns:
            Multi-line Markdown string ready to paste into a README or paper.
        """
        if not self.tracker_breakdowns:
            return "_No results._"

        trackers = list(self.tracker_breakdowns.keys())
        # Collect all attributes that appear in at least one tracker.
        all_attrs: List[str] = []
        seen: set = set()
        for bd in self.tracker_breakdowns.values():
            for attr in bd:
                if attr not in seen:
                    all_attrs.append(attr)
                    seen.add(attr)

        # Header
        header = "| Attribute | " + " | ".join(trackers) + " |\n"
        sep = "|-----------|" + "|".join(["-------:"] * len(trackers)) + "|\n"

        rows = []
        for attr in all_attrs:
            cells = []
            for t in trackers:
                r = self.tracker_breakdowns[t].get(attr)
                if r is not None:
                    cells.append(f"{r.success_auc:.3f}")
                else:
                    cells.append("—")
            rows.append(f"| {attr} | " + " | ".join(cells) + " |")

        return header + sep + "\n".join(rows)

    def to_dict(self) -> Dict:
        """Serialise to a nested plain-Python dict for JSON export."""
        out: Dict = {}
        for tracker, bd in self.tracker_breakdowns.items():
            out[tracker] = {
                attr: {
                    "attribute": r.attribute,
                    "num_frames": r.num_frames,
                    "mean_iou": round(r.mean_iou, 4),
                    "success_auc": round(r.success_auc, 4),
                }
                for attr, r in bd.items()
            }
        return out


class AttributeBreakdown:
    """Compute attribute-stratified performance for one or more trackers.

    Args:
        thresholds: IoU thresholds for the success curve.
                    Default: 101 evenly-spaced values from 0 to 1.
        detector:   :class:`~eovot.analysis.attributes.AttributeDetector`
                    instance.  Constructed with default parameters if
                    ``None``.
        min_frames: Minimum number of attribute-active frames required
                    to include the attribute in the output.
                    Default: ``5``.
    """

    def __init__(
        self,
        thresholds: Optional[np.ndarray] = None,
        detector: Optional[AttributeDetector] = None,
        min_frames: int = _MIN_ATTRIBUTE_FRAMES,
    ) -> None:
        self.thresholds = thresholds if thresholds is not None else _DEFAULT_THRESHOLDS
        self.detector = detector or AttributeDetector()
        self.min_frames = min_frames

    # ------------------------------------------------------------------
    # Single-sequence / single-tracker analysis
    # ------------------------------------------------------------------

    def compute(
        self,
        gt_boxes: np.ndarray,
        ious: np.ndarray,
    ) -> Dict[str, BreakdownResult]:
        """Compute per-attribute breakdown for a single sequence.

        Args:
            gt_boxes: ``(T, 4)`` ground-truth boxes ``(x, y, w, h)``.
            ious:     ``(T,)`` per-frame IoU values.

        Returns:
            Dict mapping attribute name → :class:`BreakdownResult`.
            Always includes ``"Overall"`` as the first key.
        """
        gt = np.asarray(gt_boxes, dtype=np.float64)
        iou_arr = np.asarray(ious, dtype=np.float64)
        T = min(len(gt), len(iou_arr))
        gt, iou_arr = gt[:T], iou_arr[:T]

        profiles = self.detector.detect(gt)
        results: Dict[str, BreakdownResult] = {}

        overall_mask = np.ones(T, dtype=bool)
        results["Overall"] = self._compute_for_mask(iou_arr, overall_mask, "Overall")

        for attr, profile in profiles.items():
            if profile.num_frames >= self.min_frames:
                results[attr.value] = self._compute_for_mask(
                    iou_arr, profile.per_frame_flags, attr.value
                )

        return results

    # ------------------------------------------------------------------
    # Multi-sequence / multi-tracker analysis
    # ------------------------------------------------------------------

    def from_benchmark_results(
        self,
        benchmark_results: Dict[str, "BenchmarkResult"],  # noqa: F821
    ) -> TrackerAttributeComparison:
        """Aggregate attribute breakdowns across all sequences for each tracker.

        Accepts a dict of :class:`~eovot.benchmark.engine.BenchmarkResult`
        objects (one per tracker) and returns a
        :class:`TrackerAttributeComparison` with mean per-attribute AUC
        pooled across all sequences.

        Args:
            benchmark_results: ``{tracker_name: BenchmarkResult}``.

        Returns:
            :class:`TrackerAttributeComparison` ready for table rendering.
        """
        tracker_breakdowns: Dict[str, Dict[str, BreakdownResult]] = {}

        for tracker_name, bench_result in benchmark_results.items():
            # Aggregate per-attribute IoU arrays across all sequences.
            attr_ious: Dict[str, List[float]] = {"Overall": []}

            for seq_result in bench_result.sequence_results:
                if seq_result.ground_truths is None or len(seq_result.ious) == 0:
                    continue

                gt = seq_result.ground_truths
                iou_arr = seq_result.ious
                T = min(len(gt), len(iou_arr))
                profiles = self.detector.detect(gt[:T])

                attr_ious["Overall"].extend(iou_arr[:T].tolist())

                for attr, profile in profiles.items():
                    key = attr.value
                    if key not in attr_ious:
                        attr_ious[key] = []
                    masked = iou_arr[:T][profile.per_frame_flags[:T]]
                    attr_ious[key].extend(masked.tolist())

            # Convert pooled IoU lists to BreakdownResult.
            breakdown: Dict[str, BreakdownResult] = {}
            for attr_name, iou_list in attr_ious.items():
                arr = np.array(iou_list, dtype=np.float64)
                if len(arr) >= self.min_frames:
                    mask = np.ones(len(arr), dtype=bool)
                    breakdown[attr_name] = self._compute_for_mask(arr, mask, attr_name)

            tracker_breakdowns[tracker_name] = breakdown

        return TrackerAttributeComparison(tracker_breakdowns=tracker_breakdowns)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_for_mask(
        self,
        ious: np.ndarray,
        mask: np.ndarray,
        attribute: str,
    ) -> BreakdownResult:
        """Compute mean IoU and success AUC for frames selected by *mask*."""
        selected = ious[mask]
        mean_iou = float(selected.mean()) if len(selected) else 0.0

        if len(selected) == 0:
            auc = 0.0
        else:
            rates = np.array([(selected > t).mean() for t in self.thresholds])
            try:
                _trapz = np.trapezoid   # NumPy >= 2.0
            except AttributeError:
                _trapz = np.trapz       # NumPy < 2.0
            auc = float(_trapz(rates, self.thresholds))

        return BreakdownResult(
            attribute=attribute,
            num_frames=int(mask.sum()),
            mean_iou=mean_iou,
            success_auc=auc,
        )

"""Challenge attribute analysis for visual object tracking.

Provides automatic detection of standard VOT challenge attributes from
ground-truth trajectories and per-attribute accuracy breakdown.

Challenge attributes follow the VOT benchmark taxonomy and can be derived
purely from the ground-truth bounding-box sequence without access to raw
frames.  This enables attribute-level analysis on any dataset that ships
standard ``(x, y, w, h)`` ground-truth annotations.

Detectable attributes
---------------------
- **FastMotion** (FM) — target centre moves more than a velocity threshold.
- **ScaleVariation** (SV) — bounding-box area changes significantly between
  consecutive frames.
- **LowResolution** (LR) — target occupies fewer than ``lr_area_threshold``
  pixels in the image.
- **AspectRatioChange** (ARC) — aspect ratio (w/h) changes more than
  ``arc_threshold`` fraction between consecutive frames.
- **OutOfView** (OV) — target centre is near or outside the frame boundary.
  Requires ``frame_width`` / ``frame_height`` to be provided.

Typical usage::

    from eovot.metrics.challenge import ChallengeDetector, ChallengeAnalyzer

    # 1 — Label every frame of a sequence
    detector = ChallengeDetector(frame_width=640, frame_height=480)
    labels = detector.label_sequence(ground_truth_bboxes)  # shape (N, 5) bool

    # 2 — Compute per-attribute accuracy breakdown from benchmark IoU arrays
    analyzer = ChallengeAnalyzer()
    report = analyzer.compute_breakdown(
        predictions=preds_array,       # (N, 4) float
        ground_truths=gt_array,        # (N, 4) float
        attribute_labels=labels,
        tracker_name="KCF",
        sequence_name="car1",
    )
    for entry in report:
        print(entry)

    # 3 — Aggregate across a full benchmark run
    benchmark_report = analyzer.aggregate(reports_list)
    print(analyzer.to_markdown_table(benchmark_report))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .accuracy import MetricsEngine

# ---------------------------------------------------------------------------
# Attribute constants
# ---------------------------------------------------------------------------

FM = "FastMotion"
SV = "ScaleVariation"
LR = "LowResolution"
ARC = "AspectRatioChange"
OV = "OutOfView"

#: Canonical ordering used for column indices in the label matrix.
ATTRIBUTES: Tuple[str, ...] = (FM, SV, LR, ARC, OV)

_ATTR_INDEX: Dict[str, int] = {a: i for i, a in enumerate(ATTRIBUTES)}


# ---------------------------------------------------------------------------
# Per-frame label matrix
# ---------------------------------------------------------------------------

@dataclass
class SequenceAttributeLabels:
    """Boolean attribute labels for every frame of a tracking sequence.

    Attributes:
        sequence_name: Name of the originating sequence.
        labels: Boolean array of shape ``(N, 5)`` where ``N`` is the
            number of frames and columns correspond to
            ``(FastMotion, ScaleVariation, LowResolution, AspectRatioChange,
            OutOfView)`` in that order (see :data:`ATTRIBUTES`).
        frame_width: Frame width used for out-of-view detection, or ``None``.
        frame_height: Frame height used for out-of-view detection, or ``None``.
    """

    sequence_name: str
    labels: np.ndarray          # shape (N, 5), dtype bool
    frame_width: Optional[int]
    frame_height: Optional[int]

    def active_frames(self, attribute: str) -> np.ndarray:
        """Return the frame indices where *attribute* is active.

        Args:
            attribute: One of the strings in :data:`ATTRIBUTES`.

        Returns:
            Integer array of frame indices.

        Raises:
            KeyError: If *attribute* is not a known challenge attribute.
        """
        if attribute not in _ATTR_INDEX:
            raise KeyError(
                f"Unknown attribute '{attribute}'. Choose from {ATTRIBUTES}."
            )
        return np.where(self.labels[:, _ATTR_INDEX[attribute]])[0]

    def coverage(self) -> Dict[str, float]:
        """Fraction of frames active for each attribute.

        Returns:
            Dict mapping attribute name → fraction in ``[0, 1]``.
        """
        n = max(len(self.labels), 1)
        return {a: float(self.labels[:, i].sum()) / n for i, a in enumerate(ATTRIBUTES)}

    def __repr__(self) -> str:
        cov = ", ".join(f"{a}={v:.2f}" for a, v in self.coverage().items())
        return f"SequenceAttributeLabels(seq={self.sequence_name!r}, {cov})"


# ---------------------------------------------------------------------------
# Challenge detector
# ---------------------------------------------------------------------------

class ChallengeDetector:
    """Derive per-frame challenge attribute labels from a GT bbox trajectory.

    Args:
        fm_velocity_px: Pixel displacement threshold for FastMotion.
            Default: ``20.0`` px.
        fm_fraction: Displacement as a fraction of bbox diagonal as an
            alternative FM criterion. Either condition triggers FM.
            Default: ``0.2`` (20% of diagonal).
        sv_area_ratio: Fractional area change threshold for ScaleVariation.
            A ratio change of more than this between consecutive frames
            triggers SV. Default: ``0.25`` (25% area change).
        lr_area_threshold: Target area (in pixels²) below which LowResolution
            is triggered. Default: ``400`` px² (roughly 20×20 px).
        arc_ratio_threshold: Fractional aspect-ratio change threshold for
            AspectRatioChange. Default: ``0.4`` (40% change).
        ov_border_fraction: Fraction of the frame dimension used to define
            the "near-boundary" zone for OutOfView detection.
            Default: ``0.05`` (centre within 5% of frame edge triggers OV).
        frame_width: Frame width in pixels. Required for OV detection;
            if ``None``, OV column is all-False.
        frame_height: Frame height in pixels. Required for OV detection;
            if ``None``, OV column is all-False.
    """

    def __init__(
        self,
        fm_velocity_px: float = 20.0,
        fm_fraction: float = 0.20,
        sv_area_ratio: float = 0.25,
        lr_area_threshold: float = 400.0,
        arc_ratio_threshold: float = 0.40,
        ov_border_fraction: float = 0.05,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> None:
        self.fm_velocity_px = fm_velocity_px
        self.fm_fraction = fm_fraction
        self.sv_area_ratio = sv_area_ratio
        self.lr_area_threshold = lr_area_threshold
        self.arc_ratio_threshold = arc_ratio_threshold
        self.ov_border_fraction = ov_border_fraction
        self.frame_width = frame_width
        self.frame_height = frame_height

    def label_sequence(
        self,
        ground_truth: np.ndarray,
        sequence_name: str = "",
    ) -> SequenceAttributeLabels:
        """Compute per-frame challenge attribute labels for a GT trajectory.

        Args:
            ground_truth: ``(N, 4)`` float array of GT bboxes in ``(x, y, w, h)``
                format.  Must have at least one row.
            sequence_name: Optional name stored in the returned labels object.

        Returns:
            :class:`SequenceAttributeLabels` with a ``(N, 5)`` boolean array.

        Raises:
            ValueError: If *ground_truth* is not a 2-D array with 4 columns.
        """
        gt = np.asarray(ground_truth, dtype=np.float64)
        if gt.ndim != 2 or gt.shape[1] != 4:
            raise ValueError(
                f"ground_truth must be shape (N, 4), got {gt.shape}"
            )

        n = len(gt)
        labels = np.zeros((n, len(ATTRIBUTES)), dtype=bool)

        cx = gt[:, 0] + gt[:, 2] / 2.0
        cy = gt[:, 1] + gt[:, 3] / 2.0
        areas = gt[:, 2] * gt[:, 3]
        aspect = np.where(gt[:, 3] > 0, gt[:, 2] / np.maximum(gt[:, 3], 1e-6), 1.0)
        diag = np.sqrt(gt[:, 2] ** 2 + gt[:, 3] ** 2)

        # --- FastMotion ---
        if n > 1:
            dcx = np.diff(cx)
            dcy = np.diff(cy)
            vel = np.sqrt(dcx ** 2 + dcy ** 2)
            # Either absolute velocity or fractional-of-diagonal triggers FM
            fm_abs = vel > self.fm_velocity_px
            fm_frac = vel > self.fm_fraction * diag[:-1]
            fm_flag = np.zeros(n, dtype=bool)
            fm_flag[1:] = fm_abs | fm_frac
            labels[:, _ATTR_INDEX[FM]] = fm_flag

        # --- ScaleVariation ---
        if n > 1:
            area_ratio = np.abs(areas[1:] / np.maximum(areas[:-1], 1e-6) - 1.0)
            sv_flag = np.zeros(n, dtype=bool)
            sv_flag[1:] = area_ratio > self.sv_area_ratio
            labels[:, _ATTR_INDEX[SV]] = sv_flag

        # --- LowResolution ---
        labels[:, _ATTR_INDEX[LR]] = areas < self.lr_area_threshold

        # --- AspectRatioChange ---
        if n > 1:
            ar_change = np.abs(aspect[1:] / np.maximum(aspect[:-1], 1e-6) - 1.0)
            arc_flag = np.zeros(n, dtype=bool)
            arc_flag[1:] = ar_change > self.arc_ratio_threshold
            labels[:, _ATTR_INDEX[ARC]] = arc_flag

        # --- OutOfView ---
        if self.frame_width is not None and self.frame_height is not None:
            bx = self.ov_border_fraction * self.frame_width
            by = self.ov_border_fraction * self.frame_height
            ov_x = (cx < bx) | (cx > self.frame_width - bx)
            ov_y = (cy < by) | (cy > self.frame_height - by)
            labels[:, _ATTR_INDEX[OV]] = ov_x | ov_y

        return SequenceAttributeLabels(
            sequence_name=sequence_name,
            labels=labels,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
        )


# ---------------------------------------------------------------------------
# Per-attribute accuracy result
# ---------------------------------------------------------------------------

@dataclass
class AttributeAccuracy:
    """Accuracy metrics for one challenge attribute across one or more sequences.

    Attributes:
        attribute: The challenge attribute name (e.g. ``"FastMotion"``).
        tracker_name: Identifier of the evaluated tracker.
        mean_iou: Mean IoU over all frames where the attribute is active.
        success_auc: AUC of the success curve over attribute-active frames.
        precision_auc: Normalised AUC of the precision curve (attribute-active).
        n_frames: Number of frames where this attribute was active.
    """

    attribute: str
    tracker_name: str
    mean_iou: float
    success_auc: float
    precision_auc: float
    n_frames: int

    def __str__(self) -> str:
        return (
            f"AttributeAccuracy({self.attribute!r} | {self.tracker_name!r}  "
            f"mIoU={self.mean_iou:.4f}  "
            f"AUC={self.success_auc:.4f}  "
            f"n={self.n_frames})"
        )


# ---------------------------------------------------------------------------
# Challenge analyzer
# ---------------------------------------------------------------------------

class ChallengeAnalyzer:
    """Compute per-challenge-attribute accuracy breakdowns from benchmark data.

    Uses the same vectorised :class:`~eovot.metrics.accuracy.MetricsEngine`
    pipeline as the core benchmark engine, applied to the subset of frames
    where each attribute is active.

    Example::

        detector = ChallengeDetector(frame_width=1920, frame_height=1080)
        labels = detector.label_sequence(gt_array, sequence_name="seq1")

        analyzer = ChallengeAnalyzer()
        report = analyzer.compute_breakdown(preds, gt, labels, tracker_name="KCF")
        for entry in report:
            print(entry)

        # Aggregate multiple sequences
        all_reports = [report_seq1, report_seq2]
        agg = analyzer.aggregate(all_reports)
        print(analyzer.to_markdown_table(agg))
    """

    def __init__(self) -> None:
        self._metrics = MetricsEngine()

    def compute_breakdown(
        self,
        predictions: np.ndarray,
        ground_truths: np.ndarray,
        attribute_labels: SequenceAttributeLabels,
        tracker_name: str = "",
    ) -> List[AttributeAccuracy]:
        """Compute accuracy metrics for each active challenge attribute.

        Only attributes that are active on at least one frame are included
        in the result.

        Args:
            predictions:      ``(N, 4)`` predicted bboxes ``(x, y, w, h)``.
            ground_truths:    ``(N, 4)`` GT bboxes ``(x, y, w, h)``.
            attribute_labels: Frame-level boolean labels from
                :class:`ChallengeDetector`.
            tracker_name:     Human-readable tracker identifier.

        Returns:
            List of :class:`AttributeAccuracy` — one entry per active attribute,
            in the same order as :data:`ATTRIBUTES`.
        """
        preds = np.asarray(predictions, dtype=np.float64)
        gts = np.asarray(ground_truths, dtype=np.float64)
        n = min(len(preds), len(gts), len(attribute_labels.labels))

        results: List[AttributeAccuracy] = []
        for attr in ATTRIBUTES:
            idx = _ATTR_INDEX[attr]
            active = attribute_labels.labels[:n, idx]
            frames = np.where(active)[0]

            if len(frames) == 0:
                continue

            sub_preds = preds[frames]
            sub_gts = gts[frames]

            metrics = self._metrics.compute_all(sub_preds, sub_gts)
            results.append(AttributeAccuracy(
                attribute=attr,
                tracker_name=tracker_name,
                mean_iou=metrics.mean_iou,
                success_auc=metrics.success_auc,
                precision_auc=metrics.precision_auc,
                n_frames=int(len(frames)),
            ))

        return results

    def aggregate(
        self,
        reports: List[List[AttributeAccuracy]],
    ) -> Dict[str, AttributeAccuracy]:
        """Weighted-average accuracy across multiple per-sequence reports.

        Frames are weighted by count so that long sequences dominate the
        average proportionally to their contribution.

        Args:
            reports: Output of repeated :meth:`compute_breakdown` calls, one
                per sequence.

        Returns:
            Dict mapping attribute name → aggregated :class:`AttributeAccuracy`.
            Only attributes that appear in at least one report are included.
        """
        accumulated: Dict[str, List[AttributeAccuracy]] = {}
        for report in reports:
            for entry in report:
                accumulated.setdefault(entry.attribute, []).append(entry)

        aggregated: Dict[str, AttributeAccuracy] = {}
        for attr, entries in accumulated.items():
            total = sum(e.n_frames for e in entries)
            if total == 0:
                continue
            w_iou = sum(e.mean_iou * e.n_frames for e in entries) / total
            w_sauc = sum(e.success_auc * e.n_frames for e in entries) / total
            w_pauc = sum(e.precision_auc * e.n_frames for e in entries) / total
            tracker = entries[0].tracker_name
            aggregated[attr] = AttributeAccuracy(
                attribute=attr,
                tracker_name=tracker,
                mean_iou=w_iou,
                success_auc=w_sauc,
                precision_auc=w_pauc,
                n_frames=total,
            )
        return aggregated

    def to_markdown_table(
        self,
        aggregated: Dict[str, AttributeAccuracy],
        tracker_name: str = "",
    ) -> str:
        """Format aggregated challenge results as a Markdown table.

        Args:
            aggregated: Output of :meth:`aggregate`.
            tracker_name: Override for the tracker column header.

        Returns:
            Multi-line Markdown table string ready for embedding in reports.
        """
        name = tracker_name or (
            next(iter(aggregated.values())).tracker_name if aggregated else ""
        )
        lines = [
            f"### Challenge Attribute Breakdown — {name}",
            "",
            "| Attribute | mIoU | Success AUC | Precision AUC | # Frames |",
            "|-----------|-----:|------------:|--------------:|---------:|",
        ]
        for attr in ATTRIBUTES:
            if attr not in aggregated:
                continue
            e = aggregated[attr]
            lines.append(
                f"| {attr} "
                f"| {e.mean_iou:.4f} "
                f"| {e.success_auc:.4f} "
                f"| {e.precision_auc:.4f} "
                f"| {e.n_frames} |"
            )
        return "\n".join(lines)

    def compare_trackers(
        self,
        tracker_aggregates: Dict[str, Dict[str, AttributeAccuracy]],
    ) -> str:
        """Build a side-by-side Markdown comparison table for multiple trackers.

        Args:
            tracker_aggregates: Mapping ``{tracker_name: aggregate_dict}`` where
                each inner dict is the output of :meth:`aggregate`.

        Returns:
            Markdown table with one row per attribute and one mIoU column per
            tracker.
        """
        trackers = list(tracker_aggregates.keys())
        header = "| Attribute | " + " | ".join(f"mIoU {t}" for t in trackers) + " |"
        sep = "|-----------|" + "|".join(["-----:"] * len(trackers)) + "|"
        lines = ["### Challenge Attribute Comparison", "", header, sep]

        for attr in ATTRIBUTES:
            row_vals = []
            has_any = False
            for t in trackers:
                agg = tracker_aggregates[t]
                if attr in agg:
                    row_vals.append(f"{agg[attr].mean_iou:.4f}")
                    has_any = True
                else:
                    row_vals.append("—")
            if has_any:
                lines.append(f"| {attr} | " + " | ".join(row_vals) + " |")

        return "\n".join(lines)

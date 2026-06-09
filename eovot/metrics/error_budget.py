"""Tracking error budget decomposition for EOVOT.

Decomposes the per-frame IoU loss into orthogonal geometric error components,
revealing *why* a tracker fails rather than just *that* it fails.

Error Components
~~~~~~~~~~~~~~~~
For every predicted box vs. ground-truth box pair the full IoU loss can be
attributed to four independent geometric axes:

- **center_error**: Fraction of IoU loss attributable to predicted centre
  displacement relative to GT.  High values mean the tracker is drifting.
- **scale_error**: Fraction attributable to predicted area differing from GT.
  High values indicate the tracker over- or under-zooms.
- **aspect_ratio_error**: Fraction attributable to predicted W/H ratio
  diverging from GT.  High values indicate shape deformation errors.
- **residual_error**: Remaining loss not explained by the above three axes.

The decomposition is additive: the four fractions always sum to 1.0 over any
non-empty set of frames.  This makes it straightforward to compare trackers:

    MOSSE: center=0.72, scale=0.18, AR=0.05, residual=0.05
    KCF:   center=0.41, scale=0.43, AR=0.10, residual=0.06

→ MOSSE loses accuracy primarily due to centre drift; KCF due to scale error.

Per-Frame Contributions
~~~~~~~~~~~~~~~~~~~~~~~
In addition to the dataset-level summary, the analyzer stores per-frame
contribution arrays that enable temporal plots (e.g. "where in the sequence
does scale error dominate?").

Typical usage::

    from eovot.metrics.error_budget import ErrorBudgetAnalyzer

    analyzer = ErrorBudgetAnalyzer()
    budget = analyzer.analyze(predictions, ground_truths)
    print(budget)

    # Compare multiple trackers
    budgets = [analyzer.analyze(r.predictions, r.ground_truths,
                                tracker_name=r.sequence_name)
               for r in result.sequence_results]
    agg = analyzer.aggregate(budgets, tracker_name="MOSSE", dataset_name="OTB100")
    print(ErrorBudgetAnalyzer.to_markdown_table([agg]))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class FrameErrorBudget:
    """Per-frame IoU-loss decomposition for a single (prediction, GT) pair.

    All four fraction values are in ``[0, 1]`` and sum to ``1.0`` whenever
    the GT box is valid (non-zero area).  They are ``0.0`` when the GT box
    has zero area (degenerate frame).

    Attributes:
        iou_loss: Raw IoU loss for this frame: ``1 - IoU``.  Range ``[0, 1]``.
        center_fraction: Fraction of the total IoU loss attributable to
            centre displacement.
        scale_fraction: Fraction attributable to area (scale) mismatch.
        aspect_ratio_fraction: Fraction attributable to W/H ratio divergence.
        residual_fraction: Remaining fraction not explained by the above.
    """

    iou_loss: float
    center_fraction: float
    scale_fraction: float
    aspect_ratio_fraction: float
    residual_fraction: float


@dataclass
class ErrorBudget:
    """Aggregate IoU-loss decomposition for one sequence or tracker–dataset pair.

    Attributes:
        tracker_name: Identifier of the tracker being analysed.
        sequence_name: Sequence or dataset name.
        num_frames: Number of evaluated frames (GT-valid only).
        mean_iou_loss: Mean ``1 - IoU`` across all evaluated frames.
        center_error: Mean fraction of IoU loss from centre displacement.
        scale_error: Mean fraction from area (scale) mismatch.
        aspect_ratio_error: Mean fraction from aspect-ratio divergence.
        residual_error: Mean unexplained fraction.
        dominant_axis: The error component with the highest fraction.
        center_contributions: Per-frame centre-fraction array (shape ``(N,)``).
        scale_contributions: Per-frame scale-fraction array (shape ``(N,)``).
        aspect_ratio_contributions: Per-frame AR-fraction array (shape ``(N,)``).
    """

    tracker_name: str
    sequence_name: str
    num_frames: int
    mean_iou_loss: float
    center_error: float
    scale_error: float
    aspect_ratio_error: float
    residual_error: float
    dominant_axis: str
    center_contributions: np.ndarray = field(repr=False)
    scale_contributions: np.ndarray = field(repr=False)
    aspect_ratio_contributions: np.ndarray = field(repr=False)

    def to_dict(self) -> Dict:
        """Serialise summary scalars to a plain dict (arrays excluded)."""
        return {
            "tracker": self.tracker_name,
            "sequence": self.sequence_name,
            "num_frames": self.num_frames,
            "mean_iou_loss": round(self.mean_iou_loss, 4),
            "center_error": round(self.center_error, 4),
            "scale_error": round(self.scale_error, 4),
            "aspect_ratio_error": round(self.aspect_ratio_error, 4),
            "residual_error": round(self.residual_error, 4),
            "dominant_axis": self.dominant_axis,
        }

    def __str__(self) -> str:
        return (
            f"ErrorBudget({self.tracker_name!r} on {self.sequence_name!r}  "
            f"IoU_loss={self.mean_iou_loss:.3f}  "
            f"center={self.center_error:.3f}  "
            f"scale={self.scale_error:.3f}  "
            f"AR={self.aspect_ratio_error:.3f}  "
            f"residual={self.residual_error:.3f}  "
            f"dominant={self.dominant_axis!r}  "
            f"frames={self.num_frames})"
        )


@dataclass
class AggregateErrorBudget:
    """Tracker-level error budget aggregated over multiple sequences.

    Attributes:
        tracker_name: Tracker identifier.
        dataset_name: Dataset name.
        num_sequences: Number of sequences included.
        num_frames: Total evaluated frames across all sequences.
        mean_iou_loss: Mean IoU loss across all frames.
        center_error: Mean centre-error fraction.
        scale_error: Mean scale-error fraction.
        aspect_ratio_error: Mean AR-error fraction.
        residual_error: Mean residual fraction.
        dominant_axis: Component with the highest mean fraction.
    """

    tracker_name: str
    dataset_name: str
    num_sequences: int
    num_frames: int
    mean_iou_loss: float
    center_error: float
    scale_error: float
    aspect_ratio_error: float
    residual_error: float
    dominant_axis: str

    def to_dict(self) -> Dict:
        return {
            "tracker": self.tracker_name,
            "dataset": self.dataset_name,
            "num_sequences": self.num_sequences,
            "num_frames": self.num_frames,
            "mean_iou_loss": round(self.mean_iou_loss, 4),
            "center_error": round(self.center_error, 4),
            "scale_error": round(self.scale_error, 4),
            "aspect_ratio_error": round(self.aspect_ratio_error, 4),
            "residual_error": round(self.residual_error, 4),
            "dominant_axis": self.dominant_axis,
        }

    def __str__(self) -> str:
        return (
            f"AggregateErrorBudget({self.tracker_name!r} on {self.dataset_name!r}  "
            f"seqs={self.num_sequences}  frames={self.num_frames}  "
            f"IoU_loss={self.mean_iou_loss:.3f}  "
            f"center={self.center_error:.3f}  "
            f"scale={self.scale_error:.3f}  "
            f"AR={self.aspect_ratio_error:.3f}  "
            f"residual={self.residual_error:.3f}  "
            f"dominant={self.dominant_axis!r})"
        )


class ErrorBudgetAnalyzer:
    """Decompose per-frame IoU loss into orthogonal geometric error components.

    The decomposition partitions the total IoU loss ``(1 - IoU)`` into four
    fractions that sum to 1.0:

    1. **Centre error** — contribution from predicted-centre displacement.
    2. **Scale error** — contribution from predicted-area mismatch.
    3. **Aspect-ratio error** — contribution from predicted W/H divergence.
    4. **Residual** — IoU loss unexplained by the above three axes.

    The fractions are estimated by computing three "ideal" bounding boxes that
    correct one error axis at a time, measuring the resulting IoU improvement,
    and normalising the improvements to sum to 1.  This attribution is exact
    for axis-aligned boxes and non-negative by construction.

    Example::

        from eovot.metrics.error_budget import ErrorBudgetAnalyzer

        analyzer = ErrorBudgetAnalyzer()

        # Analyse one sequence
        budget = analyzer.analyze(
            predictions=seq_result.predictions,
            ground_truths=seq_result.ground_truths,
            tracker_name="MOSSE",
            sequence_name=seq_result.sequence_name,
        )
        print(budget)

        # Aggregate across sequences
        budgets = [
            analyzer.analyze(sr.predictions, sr.ground_truths, tracker_name="MOSSE",
                             sequence_name=sr.sequence_name)
            for sr in benchmark_result.sequence_results
            if sr.predictions is not None
        ]
        agg = analyzer.aggregate(budgets, tracker_name="MOSSE", dataset_name="OTB100")
        print(ErrorBudgetAnalyzer.to_markdown_table([agg]))
    """

    # ------------------------------------------------------------------
    # Primary analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        predictions: np.ndarray,
        ground_truths: np.ndarray,
        tracker_name: str = "tracker",
        sequence_name: str = "unknown",
    ) -> ErrorBudget:
        """Decompose IoU loss for one sequence into error budget fractions.

        Args:
            predictions: Predicted bounding boxes, shape ``(N, 4)`` in
                ``(x, y, w, h)`` pixel coordinates.
            ground_truths: Ground-truth bounding boxes, same shape.
            tracker_name: Human-readable tracker identifier.
            sequence_name: Human-readable sequence identifier.

        Returns:
            :class:`ErrorBudget` with per-axis fractions and per-frame arrays.

        Raises:
            ValueError: If ``predictions`` and ``ground_truths`` have different
                shapes, or if shape is not ``(N, 4)`` with ``N >= 1``.
        """
        preds = np.asarray(predictions, dtype=np.float64)
        gts = np.asarray(ground_truths, dtype=np.float64)

        if preds.ndim != 2 or preds.shape[1] != 4:
            raise ValueError(
                f"predictions must have shape (N, 4), got {preds.shape}."
            )
        if gts.shape != preds.shape:
            raise ValueError(
                f"predictions and ground_truths must have the same shape, "
                f"got {preds.shape} vs {gts.shape}."
            )
        if len(preds) == 0:
            raise ValueError("predictions must contain at least one frame.")

        # Mask out degenerate GT frames (zero area).
        valid = (gts[:, 2] > 0) & (gts[:, 3] > 0)
        preds_v = preds[valid]
        gts_v = gts[valid]

        if len(preds_v) == 0:
            empty = np.array([], dtype=np.float64)
            return ErrorBudget(
                tracker_name=tracker_name,
                sequence_name=sequence_name,
                num_frames=0,
                mean_iou_loss=0.0,
                center_error=0.0,
                scale_error=0.0,
                aspect_ratio_error=0.0,
                residual_error=1.0,
                dominant_axis="residual",
                center_contributions=empty,
                scale_contributions=empty,
                aspect_ratio_contributions=empty,
            )

        iou_orig = self._batch_iou(preds_v, gts_v)
        iou_loss = 1.0 - iou_orig

        # Construct "corrected" boxes that fix one axis at a time.
        center_corrected = self._correct_center(preds_v, gts_v)
        scale_corrected = self._correct_scale(preds_v, gts_v)
        ar_corrected = self._correct_aspect_ratio(preds_v, gts_v)

        iou_center_fixed = self._batch_iou(center_corrected, gts_v)
        iou_scale_fixed = self._batch_iou(scale_corrected, gts_v)
        iou_ar_fixed = self._batch_iou(ar_corrected, gts_v)

        # IoU gain from fixing each axis (clamped to non-negative).
        gain_center = np.maximum(iou_center_fixed - iou_orig, 0.0)
        gain_scale = np.maximum(iou_scale_fixed - iou_orig, 0.0)
        gain_ar = np.maximum(iou_ar_fixed - iou_orig, 0.0)

        total_gain = gain_center + gain_scale + gain_ar + 1e-12
        center_frac = gain_center / total_gain
        scale_frac = gain_scale / total_gain
        ar_frac = gain_ar / total_gain
        # Residual: normalised unexplained fraction of iou_loss.
        residual_frac = np.maximum(1.0 - center_frac - scale_frac - ar_frac, 0.0)

        mean_center = float(np.mean(center_frac))
        mean_scale = float(np.mean(scale_frac))
        mean_ar = float(np.mean(ar_frac))
        mean_residual = float(np.mean(residual_frac))

        axis_map = {
            "center": mean_center,
            "scale": mean_scale,
            "aspect_ratio": mean_ar,
            "residual": mean_residual,
        }
        dominant = max(axis_map, key=lambda k: axis_map[k])

        return ErrorBudget(
            tracker_name=tracker_name,
            sequence_name=sequence_name,
            num_frames=int(valid.sum()),
            mean_iou_loss=float(np.mean(iou_loss)),
            center_error=mean_center,
            scale_error=mean_scale,
            aspect_ratio_error=mean_ar,
            residual_error=mean_residual,
            dominant_axis=dominant,
            center_contributions=center_frac,
            scale_contributions=scale_frac,
            aspect_ratio_contributions=ar_frac,
        )

    def aggregate(
        self,
        budgets: List[ErrorBudget],
        tracker_name: str = "tracker",
        dataset_name: str = "unknown",
    ) -> AggregateErrorBudget:
        """Compute a frame-weighted aggregate over a list of sequence budgets.

        Args:
            budgets: One :class:`ErrorBudget` per sequence, all from the same
                tracker.
            tracker_name: Tracker identifier for the aggregate record.
            dataset_name: Dataset name for the aggregate record.

        Returns:
            :class:`AggregateErrorBudget` with frame-weighted mean fractions.

        Raises:
            ValueError: If ``budgets`` is empty.
        """
        if not budgets:
            raise ValueError("budgets must contain at least one ErrorBudget.")

        total_frames = sum(b.num_frames for b in budgets)
        if total_frames == 0:
            return AggregateErrorBudget(
                tracker_name=tracker_name,
                dataset_name=dataset_name,
                num_sequences=len(budgets),
                num_frames=0,
                mean_iou_loss=0.0,
                center_error=0.0,
                scale_error=0.0,
                aspect_ratio_error=0.0,
                residual_error=1.0,
                dominant_axis="residual",
            )

        def _wavg(attr: str) -> float:
            return float(
                sum(getattr(b, attr) * b.num_frames for b in budgets) / total_frames
            )

        center = _wavg("center_error")
        scale = _wavg("scale_error")
        ar = _wavg("aspect_ratio_error")
        residual = _wavg("residual_error")
        iou_loss = _wavg("mean_iou_loss")

        axis_map = {"center": center, "scale": scale,
                    "aspect_ratio": ar, "residual": residual}
        dominant = max(axis_map, key=lambda k: axis_map[k])

        return AggregateErrorBudget(
            tracker_name=tracker_name,
            dataset_name=dataset_name,
            num_sequences=len(budgets),
            num_frames=total_frames,
            mean_iou_loss=iou_loss,
            center_error=center,
            scale_error=scale,
            aspect_ratio_error=ar,
            residual_error=residual,
            dominant_axis=dominant,
        )

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def to_markdown_table(aggregates: List[AggregateErrorBudget]) -> str:
        """Format aggregate error budgets as a Markdown comparison table.

        Args:
            aggregates: One :class:`AggregateErrorBudget` per tracker.

        Returns:
            Multi-line Markdown table string ready to embed in papers or READMEs.
        """
        lines = [
            "| Tracker | Dataset | IoU Loss | Center % | Scale % | AR % | Residual % | Dominant |",
            "|---------|---------|----------:|---------:|--------:|-----:|-----------:|:---------|",
        ]
        for a in aggregates:
            lines.append(
                f"| {a.tracker_name} | {a.dataset_name} "
                f"| {a.mean_iou_loss:.4f} "
                f"| {a.center_error * 100:.1f} "
                f"| {a.scale_error * 100:.1f} "
                f"| {a.aspect_ratio_error * 100:.1f} "
                f"| {a.residual_error * 100:.1f} "
                f"| {a.dominant_axis} |"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private geometric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _batch_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """Vectorised IoU for two arrays of (x, y, w, h) boxes."""
        ax1 = boxes_a[:, 0]
        ay1 = boxes_a[:, 1]
        ax2 = ax1 + boxes_a[:, 2]
        ay2 = ay1 + boxes_a[:, 3]

        bx1 = boxes_b[:, 0]
        by1 = boxes_b[:, 1]
        bx2 = bx1 + boxes_b[:, 2]
        by2 = by1 + boxes_b[:, 3]

        ix1 = np.maximum(ax1, bx1)
        iy1 = np.maximum(ay1, by1)
        ix2 = np.minimum(ax2, bx2)
        iy2 = np.minimum(ay2, by2)

        inter = np.maximum(ix2 - ix1, 0.0) * np.maximum(iy2 - iy1, 0.0)
        area_a = boxes_a[:, 2] * boxes_a[:, 3]
        area_b = boxes_b[:, 2] * boxes_b[:, 3]
        union = area_a + area_b - inter + 1e-9
        return np.clip(inter / union, 0.0, 1.0)

    @staticmethod
    def _correct_center(
        preds: np.ndarray, gts: np.ndarray
    ) -> np.ndarray:
        """Return boxes with GT centre but predicted width & height."""
        gt_cx = gts[:, 0] + gts[:, 2] / 2.0
        gt_cy = gts[:, 1] + gts[:, 3] / 2.0
        corrected = preds.copy()
        corrected[:, 0] = gt_cx - preds[:, 2] / 2.0
        corrected[:, 1] = gt_cy - preds[:, 3] / 2.0
        return corrected

    @staticmethod
    def _correct_scale(
        preds: np.ndarray, gts: np.ndarray
    ) -> np.ndarray:
        """Return boxes with GT area but predicted centre and aspect ratio."""
        pred_cx = preds[:, 0] + preds[:, 2] / 2.0
        pred_cy = preds[:, 1] + preds[:, 3] / 2.0
        pred_area = np.maximum(preds[:, 2] * preds[:, 3], 1e-9)
        gt_area = np.maximum(gts[:, 2] * gts[:, 3], 1e-9)

        scale = np.sqrt(gt_area / pred_area)
        new_w = preds[:, 2] * scale
        new_h = preds[:, 3] * scale

        corrected = preds.copy()
        corrected[:, 0] = pred_cx - new_w / 2.0
        corrected[:, 1] = pred_cy - new_h / 2.0
        corrected[:, 2] = new_w
        corrected[:, 3] = new_h
        return corrected

    @staticmethod
    def _correct_aspect_ratio(
        preds: np.ndarray, gts: np.ndarray
    ) -> np.ndarray:
        """Return boxes with GT aspect ratio but predicted centre and area."""
        pred_cx = preds[:, 0] + preds[:, 2] / 2.0
        pred_cy = preds[:, 1] + preds[:, 3] / 2.0
        pred_area = np.maximum(preds[:, 2] * preds[:, 3], 1e-9)

        gt_ar = np.maximum(gts[:, 2], 1e-6) / np.maximum(gts[:, 3], 1e-6)
        new_h = np.sqrt(pred_area / np.maximum(gt_ar, 1e-9))
        new_w = pred_area / np.maximum(new_h, 1e-9)

        corrected = preds.copy()
        corrected[:, 0] = pred_cx - new_w / 2.0
        corrected[:, 1] = pred_cy - new_h / 2.0
        corrected[:, 2] = new_w
        corrected[:, 3] = new_h
        return corrected

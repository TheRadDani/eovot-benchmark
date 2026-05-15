"""Per-attribute performance analysis for VOT benchmarks.

Standard benchmarks (LaSOT, OTB, GOT-10k) annotate sequences with challenging
conditions such as occlusion, fast motion, and scale variation.  Reporting
accuracy *per attribute* is required by most VOT challenge papers and reveals
failure modes that a single scalar mIoU hides.

This module provides:
- :class:`AttributeAnalyzer` — compute per-attribute IoU / success / precision
- :func:`derive_fast_motion_mask` — auto-derive fast-motion labels from GT boxes
- :func:`derive_scale_variation_mask` — auto-derive scale-variation labels
- :func:`derive_low_resolution_mask` — auto-derive low-resolution labels
- :class:`AttributeResult` / :class:`AttributeAnalysis` — result containers
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .accuracy import AccuracyMetrics, MetricsEngine


# ── Standard attribute vocabularies ──────────────────────────────────────────

#: Attribute names used by the LaSOT benchmark (per-sequence level annotations).
LASOT_ATTRIBUTES: Tuple[str, ...] = (
    "illumination_variation",
    "occlusion",
    "deformation",
    "motion_blur",
    "fast_motion",
    "in_plane_rotation",
    "out_of_plane_rotation",
    "out_of_view",
    "background_clutter",
    "low_resolution",
)

#: Attribute names used by the OTB benchmark (sequence-level tags).
OTB_ATTRIBUTES: Tuple[str, ...] = (
    "illumination_variation",
    "scale_variation",
    "occlusion",
    "deformation",
    "motion_blur",
    "fast_motion",
    "in_plane_rotation",
    "out_of_plane_rotation",
    "out_of_view",
    "background_clutter",
    "low_resolution",
)

#: Attribute names that can be auto-derived from ground-truth geometry alone.
AUTO_DERIVABLE_ATTRIBUTES: Tuple[str, ...] = (
    "fast_motion",
    "scale_variation",
    "low_resolution",
)


# ── Result containers ─────────────────────────────────────────────────────────


@dataclass
class AttributeResult:
    """Accuracy metrics for a single challenging attribute."""

    attribute: str
    num_frames: int
    mean_iou: float
    success_auc: float
    precision_auc: float

    def to_dict(self) -> dict:
        return {
            "attribute": self.attribute,
            "num_frames": self.num_frames,
            "mean_iou": round(self.mean_iou, 4),
            "success_auc": round(self.success_auc, 4),
            "precision_auc": round(self.precision_auc, 4),
        }


@dataclass
class AttributeAnalysis:
    """Aggregated per-attribute results for one tracker on a dataset.

    Attributes:
        per_attribute: Mapping from attribute name to its metrics.
        overall: Aggregate metrics across all frames (attribute-agnostic).
    """

    per_attribute: Dict[str, AttributeResult] = field(default_factory=dict)
    overall: Optional[AccuracyMetrics] = None

    # ── Human-readable output ─────────────────────────────────────────────────

    def summary_table(self) -> str:
        """Return a Markdown table sorted descending by mIoU."""
        if not self.per_attribute:
            return "No attribute data available."

        rows = sorted(
            self.per_attribute.values(),
            key=lambda r: r.mean_iou,
            reverse=True,
        )
        header = (
            "| Attribute | Frames | mIoU | Success AUC | Precision AUC |\n"
            "|-----------|-------:|-----:|------------:|--------------:|\n"
        )
        lines = [
            f"| {r.attribute} | {r.num_frames} | {r.mean_iou:.4f} "
            f"| {r.success_auc:.4f} | {r.precision_auc:.4f} |"
            for r in rows
        ]
        return header + "\n".join(lines)

    def worst_attributes(self, n: int = 3) -> List[AttributeResult]:
        """Return the *n* most challenging attributes (lowest mIoU)."""
        return sorted(self.per_attribute.values(), key=lambda r: r.mean_iou)[:n]

    def best_attributes(self, n: int = 3) -> List[AttributeResult]:
        """Return the *n* easiest attributes (highest mIoU)."""
        return sorted(
            self.per_attribute.values(), key=lambda r: r.mean_iou, reverse=True
        )[:n]

    def to_dict(self) -> dict:
        d: dict = {
            "per_attribute": {k: v.to_dict() for k, v in self.per_attribute.items()}
        }
        if self.overall is not None:
            d["overall"] = {
                "mean_iou": round(self.overall.mean_iou, 4),
                "success_auc": round(self.overall.success_auc, 4),
                "precision_auc": round(self.overall.precision_auc, 4),
            }
        return d


# ── Geometry-based mask derivation ───────────────────────────────────────────


def derive_fast_motion_mask(
    ground_truth: np.ndarray,
    velocity_threshold: float = 20.0,
) -> np.ndarray:
    """Per-frame fast-motion binary mask derived from ground-truth box centres.

    A frame is labelled fast-motion when the centre displacement from the
    previous frame exceeds *velocity_threshold* pixels.  The first frame is
    always ``False`` (no previous frame to compare against).

    Args:
        ground_truth: ``(N, 4)`` array of boxes in ``(x, y, w, h)`` format.
        velocity_threshold: Displacement threshold in pixels.

    Returns:
        Boolean ``(N,)`` array.
    """
    gt = np.asarray(ground_truth, dtype=np.float64)
    cx = gt[:, 0] + gt[:, 2] / 2.0
    cy = gt[:, 1] + gt[:, 3] / 2.0
    centers = np.stack([cx, cy], axis=1)
    displacements = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    mask = np.concatenate([[False], displacements > velocity_threshold])
    return mask.astype(bool)


def derive_scale_variation_mask(
    ground_truth: np.ndarray,
    ratio_threshold: float = 0.25,
) -> np.ndarray:
    """Per-frame scale-variation binary mask derived from ground-truth box areas.

    A frame is labelled scale-variation when the absolute relative change in
    box area with respect to the initial frame exceeds *ratio_threshold*.

    Args:
        ground_truth: ``(N, 4)`` array of boxes in ``(x, y, w, h)`` format.
        ratio_threshold: Relative area deviation threshold (e.g. 0.25 = 25 %).

    Returns:
        Boolean ``(N,)`` array.
    """
    gt = np.asarray(ground_truth, dtype=np.float64)
    areas = gt[:, 2] * gt[:, 3]
    init_area = float(areas[0]) if areas[0] > 0.0 else 1.0
    ratios = np.abs(areas / init_area - 1.0)
    return (ratios > ratio_threshold).astype(bool)


def derive_low_resolution_mask(
    ground_truth: np.ndarray,
    area_threshold: float = 400.0,
) -> np.ndarray:
    """Per-frame low-resolution binary mask derived from ground-truth box areas.

    A frame is labelled low-resolution when the bounding-box area is below
    *area_threshold* pixels² (default 400 ≈ 20 × 20 pixels).

    Args:
        ground_truth: ``(N, 4)`` array of boxes in ``(x, y, w, h)`` format.
        area_threshold: Area threshold in pixels².

    Returns:
        Boolean ``(N,)`` array.
    """
    gt = np.asarray(ground_truth, dtype=np.float64)
    areas = gt[:, 2] * gt[:, 3]
    return (areas < area_threshold).astype(bool)


# ── Core analyzer ─────────────────────────────────────────────────────────────


class AttributeAnalyzer:
    """Compute per-attribute accuracy metrics for a single tracker.

    Supports two complementary workflows:

    **Explicit masks** — caller provides per-frame boolean masks for each
    attribute (e.g. loaded from LaSOT annotation files or OTB metadata).

    **Auto-derived masks** — when no attribute annotations are available,
    :meth:`auto_derive_masks` generates geometry-based masks for
    ``fast_motion``, ``scale_variation``, and ``low_resolution``.

    Both workflows produce an :class:`AttributeAnalysis` that can be printed
    as a Markdown table or serialised to JSON.

    Example — explicit masks::

        analyzer = AttributeAnalyzer()
        masks = {
            "occlusion": np.array([False, True, True, False, ...]),
            "fast_motion": np.array([False, False, True, True, ...]),
        }
        analysis = analyzer.analyze_sequence(predictions, ground_truth, masks)
        print(analysis.summary_table())

    Example — auto-derived masks::

        analyzer = AttributeAnalyzer()
        masks = analyzer.auto_derive_masks(ground_truth)
        analysis = analyzer.analyze_sequence(predictions, ground_truth, masks)
        print(analysis.summary_table())
    """

    def __init__(self) -> None:
        self._engine = MetricsEngine()

    # ── Single-sequence API ───────────────────────────────────────────────────

    def analyze_sequence(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        attribute_masks: Dict[str, np.ndarray],
    ) -> AttributeAnalysis:
        """Compute per-attribute metrics for a single sequence.

        Args:
            predictions: ``(N, 4)`` predicted bounding boxes ``(x, y, w, h)``.
            ground_truth: ``(N, 4)`` ground-truth bounding boxes ``(x, y, w, h)``.
            attribute_masks: Mapping from attribute name to boolean ``(N,)`` mask.

        Returns:
            :class:`AttributeAnalysis` with per-attribute and overall metrics.
        """
        predictions = np.asarray(predictions, dtype=np.float32)
        ground_truth = np.asarray(ground_truth, dtype=np.float32)

        n = min(len(predictions), len(ground_truth))
        predictions = predictions[:n]
        ground_truth = ground_truth[:n]

        overall = self._engine.compute_all(predictions, ground_truth)
        per_attr: Dict[str, AttributeResult] = {}

        for attr_name, mask in attribute_masks.items():
            mask = np.asarray(mask, dtype=bool)
            if len(mask) > n:
                mask = mask[:n]
            n_frames = int(mask.sum())
            if n_frames < 2:
                continue

            pred_sub = predictions[mask]
            gt_sub = ground_truth[mask]
            metrics = self._engine.compute_all(pred_sub, gt_sub)

            per_attr[attr_name] = AttributeResult(
                attribute=attr_name,
                num_frames=n_frames,
                mean_iou=metrics.mean_iou,
                success_auc=metrics.success_auc,
                precision_auc=metrics.precision_auc,
            )

        return AttributeAnalysis(per_attribute=per_attr, overall=overall)

    # ── Multi-sequence (benchmark) API ────────────────────────────────────────

    def analyze_benchmark(
        self,
        sequence_predictions: List[np.ndarray],
        sequence_ground_truths: List[np.ndarray],
        sequence_attribute_masks: List[Dict[str, np.ndarray]],
    ) -> AttributeAnalysis:
        """Aggregate per-attribute metrics across multiple sequences.

        Frames for each attribute are *pooled* across all sequences, and then
        metrics are recomputed on the combined pool.  This matches the standard
        LaSOT / OTB evaluation protocol.

        Args:
            sequence_predictions: List of ``(N_i, 4)`` prediction arrays.
            sequence_ground_truths: List of ``(N_i, 4)`` ground-truth arrays.
            sequence_attribute_masks: Per-sequence attribute mask dicts.

        Returns:
            :class:`AttributeAnalysis` aggregated over all sequences.
        """
        pooled_pred: Dict[str, List[np.ndarray]] = {}
        pooled_gt: Dict[str, List[np.ndarray]] = {}
        all_pred_list: List[np.ndarray] = []
        all_gt_list: List[np.ndarray] = []

        for preds, gts, masks in zip(
            sequence_predictions, sequence_ground_truths, sequence_attribute_masks
        ):
            preds = np.asarray(preds, dtype=np.float32)
            gts = np.asarray(gts, dtype=np.float32)
            n = min(len(preds), len(gts))
            preds, gts = preds[:n], gts[:n]
            all_pred_list.append(preds)
            all_gt_list.append(gts)

            for attr_name, mask in masks.items():
                mask = np.asarray(mask, dtype=bool)
                if len(mask) > n:
                    mask = mask[:n]
                if mask.sum() < 2:
                    continue
                pooled_pred.setdefault(attr_name, []).append(preds[mask])
                pooled_gt.setdefault(attr_name, []).append(gts[mask])

        overall_pred = np.concatenate(all_pred_list, axis=0)
        overall_gt = np.concatenate(all_gt_list, axis=0)
        overall = self._engine.compute_all(overall_pred, overall_gt)

        per_attr: Dict[str, AttributeResult] = {}
        for attr_name in pooled_pred:
            combined_pred = np.concatenate(pooled_pred[attr_name], axis=0)
            combined_gt = np.concatenate(pooled_gt[attr_name], axis=0)
            metrics = self._engine.compute_all(combined_pred, combined_gt)
            per_attr[attr_name] = AttributeResult(
                attribute=attr_name,
                num_frames=len(combined_pred),
                mean_iou=metrics.mean_iou,
                success_auc=metrics.success_auc,
                precision_auc=metrics.precision_auc,
            )

        return AttributeAnalysis(per_attribute=per_attr, overall=overall)

    # ── Mask derivation helpers ───────────────────────────────────────────────

    def auto_derive_masks(
        self,
        ground_truth: np.ndarray,
        fast_motion_threshold: float = 20.0,
        scale_variation_threshold: float = 0.25,
        low_resolution_threshold: float = 400.0,
    ) -> Dict[str, np.ndarray]:
        """Auto-derive per-frame attribute masks from ground-truth geometry.

        Useful when explicit attribute annotations are unavailable.  Produces
        three geometry-based masks:

        - ``fast_motion`` — large inter-frame centre displacement
        - ``scale_variation`` — significant area change from initial frame
        - ``low_resolution`` — bounding box smaller than threshold

        Args:
            ground_truth: ``(N, 4)`` ground-truth boxes ``(x, y, w, h)``.
            fast_motion_threshold: Centre displacement threshold (px).
            scale_variation_threshold: Relative area deviation threshold.
            low_resolution_threshold: Minimum box area (px²) for normal res.

        Returns:
            Dict mapping attribute name → boolean ``(N,)`` mask.
        """
        gt = np.asarray(ground_truth, dtype=np.float64)
        return {
            "fast_motion": derive_fast_motion_mask(gt, fast_motion_threshold),
            "scale_variation": derive_scale_variation_mask(gt, scale_variation_threshold),
            "low_resolution": derive_low_resolution_mask(gt, low_resolution_threshold),
        }

    def extract_otb_attributes(
        self,
        sequence_tags: List[str],
        num_frames: int,
    ) -> Dict[str, np.ndarray]:
        """Convert OTB sequence-level attribute tags to per-frame boolean masks.

        OTB annotates at the sequence level; this method broadcasts each tag
        into an all-``True`` frame-level mask so it can be used with
        :meth:`analyze_sequence`.

        Args:
            sequence_tags: OTB attribute strings, e.g. ``["occlusion", "fast_motion"]``.
            num_frames: Total number of frames in the sequence.

        Returns:
            Dict mapping normalised attribute name → all-True ``(N,)`` mask.
        """
        masks: Dict[str, np.ndarray] = {}
        for tag in sequence_tags:
            normalised = tag.lower().replace(" ", "_")
            masks[normalised] = np.ones(num_frames, dtype=bool)
        return masks

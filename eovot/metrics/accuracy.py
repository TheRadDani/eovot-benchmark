"""Accuracy metrics for visual object tracking.

Implements the standard VOT evaluation protocol used by OTB, GOT-10k, and
LaSOT benchmarks:

- **IoU** (Intersection over Union) per frame
- **Success Curve** — fraction of frames with IoU above a threshold,
  swept from 0 to 1; AUC summarises the curve as a scalar.
- **Precision Curve** — fraction of frames whose predicted centre is
  within a pixel-distance threshold of the ground-truth centre,
  swept from 0 to 50 px; AUC at 20 px is the canonical scalar.
- **Normalized Precision Curve** — precision curve with distances
  normalized by the GT box diagonal, making it resolution-agnostic.
  Used by GOT-10k and LaSOT evaluation protocols.
- **AccuracyMetrics** dataclass that bundles all scalars together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Bounding box: (x, y, width, height)
BBox = Tuple[float, float, float, float]


def iou(pred: BBox, gt: BBox) -> float:
    """Compute Intersection-over-Union between two axis-aligned boxes.

    Args:
        pred: Predicted box ``(x, y, w, h)``.
        gt:   Ground-truth box ``(x, y, w, h)``.

    Returns:
        IoU in ``[0, 1]``.  Returns 0 when either box has zero area.
    """
    px, py, pw, ph = pred
    gx, gy, gw, gh = gt

    if pw <= 0 or ph <= 0 or gw <= 0 or gh <= 0:
        return 0.0

    ix1 = max(px, gx)
    iy1 = max(py, gy)
    ix2 = min(px + pw, gx + gw)
    iy2 = min(py + ph, gy + gh)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = pw * ph + gw * gh - inter
    return float(inter / union) if union > 0 else 0.0


def center_distance(pred: BBox, gt: BBox) -> float:
    """Euclidean distance between the centres of two boxes.

    Args:
        pred: Predicted box ``(x, y, w, h)``.
        gt:   Ground-truth box ``(x, y, w, h)``.

    Returns:
        Distance in pixels.
    """
    px, py, pw, ph = pred
    gx, gy, gw, gh = gt
    dx = (px + pw / 2) - (gx + gw / 2)
    dy = (py + ph / 2) - (gy + gh / 2)
    return float(np.sqrt(dx * dx + dy * dy))


def normalized_center_distance(pred: BBox, gt: BBox) -> float:
    """Centre-distance normalized by the GT box diagonal length.

    Dividing by the diagonal makes the metric independent of image resolution
    and object size, enabling fair cross-dataset comparisons.

    Args:
        pred: Predicted box ``(x, y, w, h)``.
        gt:   Ground-truth box ``(x, y, w, h)``.

    Returns:
        Normalized distance in ``[0, ∞)``.  Returns 0 when the GT box is
        degenerate (zero area).
    """
    _, _, gw, gh = gt
    diagonal = float(np.sqrt(gw * gw + gh * gh))
    if diagonal <= 0:
        return 0.0
    return center_distance(pred, gt) / diagonal


@dataclass
class AccuracyMetrics:
    """Scalar accuracy summary for a tracker on a dataset or sequence."""

    mean_iou: float
    """Mean IoU across all evaluated frames."""

    success_auc: float
    """Area Under the Success Curve (IoU thresholds 0 → 1)."""

    precision_auc: float
    """Normalised AUC of the Precision Curve (distance thresholds 0 → 50 px)."""

    norm_precision_auc: Optional[float] = None
    """AUC of the Normalized Precision Curve (thresholds 0 → 0.5 × GT diagonal).
    Resolution-agnostic; used by GOT-10k and LaSOT protocols."""

    def __str__(self) -> str:
        parts = (
            f"mIoU={self.mean_iou:.4f}, "
            f"success_AUC={self.success_auc:.4f}, "
            f"precision_AUC={self.precision_auc:.4f}"
        )
        if self.norm_precision_auc is not None:
            parts += f", norm_precision_AUC={self.norm_precision_auc:.4f}"
        return f"AccuracyMetrics({parts})"


class MetricsEngine:
    """Compute standard VOT accuracy metrics from prediction/GT arrays.

    All methods accept numpy arrays of shape ``(N, 4)`` where each row is
    a bounding box in ``(x, y, w, h)`` format.

    Example::

        engine = MetricsEngine()
        ious   = engine.batch_iou(preds, gts)
        result = engine.compute_all(preds, gts)
        print(result.success_auc)
    """

    def batch_iou(self, preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
        """Vectorised per-frame IoU.

        Args:
            preds: ``(N, 4)`` array of predicted boxes.
            gts:   ``(N, 4)`` array of ground-truth boxes.

        Returns:
            ``(N,)`` float array of IoU values.
        """
        n = min(len(preds), len(gts))
        result = np.empty(n, dtype=np.float64)
        for i in range(n):
            result[i] = iou(tuple(preds[i]), tuple(gts[i]))  # type: ignore[arg-type]
        return result

    def success_curve(
        self,
        ious: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Success curve: fraction of frames with IoU > threshold.

        Args:
            ious:       Per-frame IoU values, shape ``(N,)``.
            thresholds: IoU thresholds to sweep (default: 0, 0.01, …, 1).

        Returns:
            ``(thresholds, success_rates)`` — both shape ``(T,)``.
        """
        if thresholds is None:
            thresholds = np.linspace(0.0, 1.0, 101)
        rates = np.array([(ious > t).mean() for t in thresholds])
        return thresholds, rates

    def precision_curve(
        self,
        preds: np.ndarray,
        gts: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Precision curve: fraction of frames with centre-dist < threshold.

        Args:
            preds:      ``(N, 4)`` predicted boxes.
            gts:        ``(N, 4)`` ground-truth boxes.
            thresholds: Distance thresholds in pixels (default: 0 … 50).

        Returns:
            ``(thresholds, precision_rates)`` — both shape ``(T,)``.
        """
        if thresholds is None:
            thresholds = np.linspace(0.0, 50.0, 51)
        n = min(len(preds), len(gts))
        dists = np.array(
            [center_distance(tuple(preds[i]), tuple(gts[i])) for i in range(n)]  # type: ignore[arg-type]
        )
        rates = np.array([(dists < t).mean() for t in thresholds])
        return thresholds, rates

    def normalized_precision_curve(
        self,
        preds: np.ndarray,
        gts: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normalized precision curve: fraction of frames with normalized centre-dist < threshold.

        Distances are divided by the GT box diagonal, making the metric
        independent of image resolution and object scale.  Standard threshold
        range is 0 → 0.5 (i.e. half a GT-box diagonal), following the
        GOT-10k and LaSOT evaluation protocols.

        Args:
            preds:      ``(N, 4)`` predicted boxes.
            gts:        ``(N, 4)`` ground-truth boxes.
            thresholds: Normalized distance thresholds (default: 0 … 0.5,
                        101 points).

        Returns:
            ``(thresholds, precision_rates)`` — both shape ``(T,)``.
        """
        if thresholds is None:
            thresholds = np.linspace(0.0, 0.5, 101)
        n = min(len(preds), len(gts))
        norm_dists = np.array(
            [
                normalized_center_distance(
                    tuple(preds[i]), tuple(gts[i])  # type: ignore[arg-type]
                )
                for i in range(n)
            ]
        )
        rates = np.array([(norm_dists < t).mean() for t in thresholds])
        return thresholds, rates

    def compute_all(
        self,
        preds: np.ndarray,
        gts: np.ndarray,
    ) -> AccuracyMetrics:
        """Compute mean IoU, success AUC, precision AUC, and normalized precision AUC.

        Args:
            preds: ``(N, 4)`` predicted boxes.
            gts:   ``(N, 4)`` ground-truth boxes.

        Returns:
            :class:`AccuracyMetrics` with all scalar summaries populated.
        """
        ious = self.batch_iou(preds, gts)

        _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # type: ignore[attr-defined]

        thr_iou, sr = self.success_curve(ious)
        success_auc = float(_trapz(sr, thr_iou))

        thr_dist, pr = self.precision_curve(preds, gts)
        prec_auc = float(_trapz(pr, thr_dist) / thr_dist[-1]) if thr_dist[-1] > 0 else 0.0

        thr_norm, npr = self.normalized_precision_curve(preds, gts)
        norm_prec_auc = float(_trapz(npr, thr_norm) / thr_norm[-1]) if thr_norm[-1] > 0 else 0.0

        return AccuracyMetrics(
            mean_iou=float(ious.mean()),
            success_auc=success_auc,
            precision_auc=prec_auc,
            norm_precision_auc=norm_prec_auc,
        )

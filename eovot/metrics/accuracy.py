"""Accuracy metrics for visual object tracking.

Implements the standard VOT evaluation protocol used by OTB, GOT-10k, and
LaSOT benchmarks:

- **IoU** (Intersection over Union) per frame
- **Success Curve** — fraction of frames with IoU above a threshold,
  swept from 0 to 1; AUC summarises the curve as a scalar.
- **Precision Curve** — fraction of frames whose predicted centre is
  within a pixel-distance threshold of the ground-truth centre,
  swept from 0 to 50 px; AUC at 20 px is the canonical scalar.
- **Normalized Precision Curve** — like precision but distance is normalized
  by ``sqrt(gt_w * gt_h)``, making it resolution-independent (LaSOT protocol).
  Thresholds sweep 0 → 0.5; AUC at 0.20 is the canonical scalar.
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
    """Center distance normalized by the square root of the GT box area.

    This is the LaSOT normalized precision metric, which removes the
    dependence on image resolution and target scale:

        NCD = center_distance(pred, gt) / sqrt(gt_w * gt_h)

    Args:
        pred: Predicted box ``(x, y, w, h)``.
        gt:   Ground-truth box ``(x, y, w, h)``.

    Returns:
        Normalized distance (dimensionless).  Returns 0.0 when the GT
        box has zero area.
    """
    _, _, gw, gh = gt
    gt_scale = float(np.sqrt(gw * gh))
    if gt_scale <= 0:
        return 0.0
    return center_distance(pred, gt) / gt_scale


@dataclass
class AccuracyMetrics:
    """Scalar accuracy summary for a tracker on a dataset or sequence."""

    mean_iou: float
    """Mean IoU across all evaluated frames."""

    success_auc: float
    """Area Under the Success Curve (IoU thresholds 0 → 1)."""

    precision_auc: float
    """Normalised AUC of the Precision Curve (distance thresholds 0 → 50 px)."""

    np_auc: float
    """Normalised AUC of the Normalized Precision Curve (LaSOT protocol,
    distance thresholds 0 → 0.5 relative to sqrt(gt_w * gt_h))."""

    def __str__(self) -> str:
        return (
            f"AccuracyMetrics("
            f"mIoU={self.mean_iou:.4f}, "
            f"success_AUC={self.success_auc:.4f}, "
            f"precision_AUC={self.precision_auc:.4f}, "
            f"np_AUC={self.np_auc:.4f})"
        )


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
        """Normalized precision curve (LaSOT protocol).

        Distances are normalized by ``sqrt(gt_w * gt_h)`` before thresholding,
        making the metric invariant to image resolution and target scale.
        Thresholds sweep 0 → 0.5; AUC at threshold 0.20 is the canonical scalar.

        Args:
            preds:      ``(N, 4)`` predicted boxes.
            gts:        ``(N, 4)`` ground-truth boxes.
            thresholds: Normalized distance thresholds (default: 0 … 0.5).

        Returns:
            ``(thresholds, precision_rates)`` — both shape ``(T,)``.
        """
        if thresholds is None:
            thresholds = np.linspace(0.0, 0.5, 51)
        n = min(len(preds), len(gts))
        dists = np.array(
            [normalized_center_distance(tuple(preds[i]), tuple(gts[i])) for i in range(n)]  # type: ignore[arg-type]
        )
        rates = np.array([(dists < t).mean() for t in thresholds])
        return thresholds, rates

    def compute_all(
        self,
        preds: np.ndarray,
        gts: np.ndarray,
    ) -> AccuracyMetrics:
        """Compute all scalar accuracy summaries in one call.

        Computes mean IoU, success AUC, precision AUC (OTB/VOT protocol),
        and normalized precision AUC (LaSOT protocol).

        Args:
            preds: ``(N, 4)`` predicted boxes.
            gts:   ``(N, 4)`` ground-truth boxes.

        Returns:
            :class:`AccuracyMetrics` with all scalar summaries populated.
        """
        ious = self.batch_iou(preds, gts)

        try:
            _trapz = np.trapezoid  # numpy ≥ 2.0
        except AttributeError:
            _trapz = np.trapz  # numpy < 2.0

        thr_iou, sr = self.success_curve(ious)
        success_auc = float(_trapz(sr, thr_iou))

        thr_dist, pr = self.precision_curve(preds, gts)
        prec_auc = float(_trapz(pr, thr_dist) / thr_dist[-1]) if thr_dist[-1] > 0 else 0.0

        thr_np, npr = self.normalized_precision_curve(preds, gts)
        np_auc = float(_trapz(npr, thr_np) / thr_np[-1]) if thr_np[-1] > 0 else 0.0

        return AccuracyMetrics(
            mean_iou=float(ious.mean()),
            success_auc=success_auc,
            precision_auc=prec_auc,
            np_auc=np_auc,
        )

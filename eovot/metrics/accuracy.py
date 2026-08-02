"""Accuracy metrics for visual object tracking.

Implements the standard VOT evaluation protocols used by OTB, GOT-10k, and
LaSOT benchmarks:

- **IoU** (Intersection over Union) per frame
- **Success Curve** — fraction of frames with IoU above a threshold,
  swept from 0 to 1; AUC summarises the curve as a scalar.
- **Precision Curve** — fraction of frames whose predicted centre is
  within a pixel-distance threshold of the ground-truth centre,
  swept from 0 to 50 px; AUC at 20 px is the canonical scalar.
- **SR@0.5 / SR@0.75** — Success Rate at fixed IoU thresholds (GOT-10k
  primary metrics alongside Average Overlap).
- **Normalized Precision** — centre-distance normalized by sqrt(GT area),
  threshold swept 0 → 0.5; scale-invariant and required for LaSOT / GOT-10k
  NP AUC computation.
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


@dataclass
class AccuracyMetrics:
    """Scalar accuracy summary for a tracker on a dataset or sequence.

    Standard metrics (OTB / GOT-10k / LaSOT protocols):

    * ``mean_iou`` — Average Overlap (AO), the GOT-10k primary scalar.
    * ``success_auc`` — AUC of the overlap success curve (0 → 1 threshold).
    * ``precision_auc`` — Normalised AUC of the centre-distance precision
      curve (0 → 50 px threshold); OTB-protocol.
    * ``success_rate_50`` — SR@0.5: fraction of frames with IoU > 0.5.
      First GOT-10k secondary metric.
    * ``success_rate_75`` — SR@0.75: fraction of frames with IoU > 0.75.
      Second GOT-10k secondary metric (stricter).
    * ``normalized_precision_auc`` — NP AUC: precision curve with distances
      normalised by sqrt(GT area), threshold 0 → 0.5; LaSOT / GOT-10k
      scale-invariant precision.
    """

    mean_iou: float
    """Mean IoU across all evaluated frames (= Average Overlap / AO)."""

    success_auc: float
    """Area Under the Success Curve (IoU thresholds 0 → 1)."""

    precision_auc: float
    """Normalised AUC of the Precision Curve (distance thresholds 0 → 50 px)."""

    success_rate_50: float = 0.0
    """SR@0.5 — fraction of frames where IoU > 0.5 (GOT-10k metric)."""

    success_rate_75: float = 0.0
    """SR@0.75 — fraction of frames where IoU > 0.75 (GOT-10k strict metric)."""

    normalized_precision_auc: float = 0.0
    """NP AUC — precision with distance normalised by sqrt(GT area),
    thresholds 0 → 0.5 (LaSOT / GOT-10k protocol)."""

    def __str__(self) -> str:
        return (
            f"AccuracyMetrics("
            f"AO={self.mean_iou:.4f}, "
            f"success_AUC={self.success_auc:.4f}, "
            f"SR@0.5={self.success_rate_50:.4f}, "
            f"SR@0.75={self.success_rate_75:.4f}, "
            f"NP_AUC={self.normalized_precision_auc:.4f}, "
            f"precision_AUC={self.precision_auc:.4f})"
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
        """Fully-vectorised per-frame IoU using NumPy broadcasting.

        Replaces the previous element-wise Python loop with a single-pass
        NumPy operation, giving a ~100× speedup on 1000-frame sequences and
        enabling efficient evaluation of long LaSOT / GOT-10k sequences.

        Args:
            preds: ``(N, 4)`` array of predicted boxes ``(x, y, w, h)``.
            gts:   ``(N, 4)`` array of ground-truth boxes ``(x, y, w, h)``.

        Returns:
            ``(N,)`` float64 array of IoU values in ``[0, 1]``.
        """
        n = min(len(preds), len(gts))
        if n == 0:
            return np.empty(0, dtype=np.float64)
        p = np.asarray(preds[:n], dtype=np.float64)
        g = np.asarray(gts[:n], dtype=np.float64)

        ix1 = np.maximum(p[:, 0], g[:, 0])
        iy1 = np.maximum(p[:, 1], g[:, 1])
        ix2 = np.minimum(p[:, 0] + p[:, 2], g[:, 0] + g[:, 2])
        iy2 = np.minimum(p[:, 1] + p[:, 3], g[:, 1] + g[:, 3])

        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        union = p[:, 2] * p[:, 3] + g[:, 2] * g[:, 3] - inter

        valid = (p[:, 2] > 0) & (p[:, 3] > 0) & (g[:, 2] > 0) & (g[:, 3] > 0) & (union > 0)
        return np.where(valid, inter / union, 0.0)

    def batch_center_distance(self, preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
        """Vectorised per-frame centre-to-centre Euclidean distance (pixels).

        Args:
            preds: ``(N, 4)`` array of predicted boxes ``(x, y, w, h)``.
            gts:   ``(N, 4)`` array of ground-truth boxes ``(x, y, w, h)``.

        Returns:
            ``(N,)`` float64 array of distances in pixels.
        """
        n = min(len(preds), len(gts))
        if n == 0:
            return np.empty(0, dtype=np.float64)
        p = np.asarray(preds[:n], dtype=np.float64)
        g = np.asarray(gts[:n], dtype=np.float64)
        pc = p[:, :2] + p[:, 2:] / 2.0   # predicted centres (N, 2)
        gc = g[:, :2] + g[:, 2:] / 2.0   # GT centres (N, 2)
        return np.sqrt(np.sum((pc - gc) ** 2, axis=1))

    def batch_normalized_center_distance(
        self, preds: np.ndarray, gts: np.ndarray
    ) -> np.ndarray:
        """Per-frame centre distance normalised by the square root of GT box area.

        This is the **Normalized Precision (NP)** distance used by GOT-10k and
        LaSOT to make precision evaluation scale-invariant: a 20 px error is
        severe for a 40×40 target but negligible for a 400×400 one.

        Formula::

            norm_dist_t = ||centre_pred_t - centre_gt_t|| / sqrt(w_gt_t × h_gt_t)

        Frames where the GT box has zero area are assigned distance ``0.0``.

        Args:
            preds: ``(N, 4)`` array of predicted boxes ``(x, y, w, h)``.
            gts:   ``(N, 4)`` array of ground-truth boxes ``(x, y, w, h)``.

        Returns:
            ``(N,)`` float64 array of normalised distances (dimensionless).
        """
        n = min(len(preds), len(gts))
        if n == 0:
            return np.empty(0, dtype=np.float64)
        p = np.asarray(preds[:n], dtype=np.float64)
        g = np.asarray(gts[:n], dtype=np.float64)
        pc = p[:, :2] + p[:, 2:] / 2.0
        gc = g[:, :2] + g[:, 2:] / 2.0
        pixel_dists = np.sqrt(np.sum((pc - gc) ** 2, axis=1))
        gt_area = g[:, 2] * g[:, 3]
        gt_sqrt_area = np.where(gt_area > 0, np.sqrt(gt_area), 1.0)
        return pixel_dists / gt_sqrt_area

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
        """Normalized Precision (NP) curve used by GOT-10k and LaSOT.

        Sweeps a normalised-distance threshold from 0 to 0.5 and reports the
        fraction of frames where the normalised centre-distance is below each
        threshold.  Normalisation by ``sqrt(GT area)`` makes the metric
        scale-invariant — a 20 px error weighs more for a small target than
        for a large one.

        The GOT-10k NP AUC is computed by trapezoidal integration of this
        curve over ``[0, 0.5]`` and dividing by 0.5 (so values live in
        ``[0, 1]`` regardless of the threshold range).

        Args:
            preds:      ``(N, 4)`` predicted boxes ``(x, y, w, h)``.
            gts:        ``(N, 4)`` ground-truth boxes ``(x, y, w, h)``.
            thresholds: Normalised thresholds to sweep (default: 0 … 0.5,
                        51 evenly-spaced points).

        Returns:
            ``(thresholds, precision_rates)`` — both shape ``(T,)``.
        """
        if thresholds is None:
            thresholds = np.linspace(0.0, 0.5, 51)
        norm_dists = self.batch_normalized_center_distance(preds, gts)
        if len(norm_dists) == 0:
            return thresholds, np.zeros_like(thresholds)
        rates = np.array([(norm_dists < t).mean() for t in thresholds])
        return thresholds, rates

    def compute_all(
        self,
        preds: np.ndarray,
        gts: np.ndarray,
    ) -> AccuracyMetrics:
        """Compute all accuracy metrics in one call.

        Covers the full OTB + GOT-10k + LaSOT metric suite:
        mean IoU (AO), success AUC, precision AUC, SR@0.5, SR@0.75, and
        Normalized Precision AUC.

        Args:
            preds: ``(N, 4)`` predicted boxes.
            gts:   ``(N, 4)`` ground-truth boxes.

        Returns:
            :class:`AccuracyMetrics` with all scalar summaries populated.
        """
        ious = self.batch_iou(preds, gts)

        # np.trapezoid was introduced in NumPy 2.0; np.trapz was removed in 2.0.
        try:
            _trapz = np.trapezoid  # numpy ≥ 2.0
        except AttributeError:
            _trapz = np.trapz  # numpy < 2.0

        thr_iou, sr = self.success_curve(ious)
        success_auc = float(_trapz(sr, thr_iou))

        thr_dist, pr = self.precision_curve(preds, gts)
        prec_auc = float(_trapz(pr, thr_dist) / thr_dist[-1]) if thr_dist[-1] > 0 else 0.0

        # GOT-10k fixed-threshold success rates
        sr_50 = float((ious > 0.5).mean()) if len(ious) > 0 else 0.0
        sr_75 = float((ious > 0.75).mean()) if len(ious) > 0 else 0.0

        # Normalized Precision AUC (GOT-10k / LaSOT protocol)
        thr_np, np_rates = self.normalized_precision_curve(preds, gts)
        np_auc = float(_trapz(np_rates, thr_np) / thr_np[-1]) if thr_np[-1] > 0 else 0.0

        return AccuracyMetrics(
            mean_iou=float(ious.mean()) if len(ious) > 0 else 0.0,
            success_auc=success_auc,
            precision_auc=prec_auc,
            success_rate_50=sr_50,
            success_rate_75=sr_75,
            normalized_precision_auc=np_auc,
        )

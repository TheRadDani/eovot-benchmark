"""Accuracy metrics for visual object tracking.

Implements the standard VOT evaluation protocol used by OTB, GOT-10k, and
LaSOT benchmarks:

- **IoU** (Intersection over Union) per frame
- **Success Curve** — fraction of frames with IoU above a threshold,
  swept from 0 to 1; AUC summarises the curve as a scalar.
- **Precision Curve** — fraction of frames whose predicted centre is
  within a pixel-distance threshold of the ground-truth centre,
  swept from 0 to 50 px; AUC at 20 px is the canonical scalar.
- **Normalized Precision Curve** — like precision curve but distance is
  normalized by sqrt(gt_w * gt_h), making it scale-invariant.  Used by
  GOT-10k and LaSOT.  Canonical threshold: 0.2 (NormPrec@0.2).
- **Success Rate at fixed thresholds** — SR@0.5 and SR@0.75 are the
  official GOT-10k protocol scalars alongside AO (average overlap = mIoU).
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


def normalized_center_error(pred: BBox, gt: BBox) -> float:
    """Scale-invariant centre distance normalized by the ground-truth box size.

    Divides the Euclidean centre distance by ``sqrt(gt_w * gt_h)`` so that
    the score is independent of target scale.  This is the per-frame error
    used to build the *Normalized Precision Curve* in the GOT-10k and LaSOT
    evaluation protocols.

    Args:
        pred: Predicted box ``(x, y, w, h)``.
        gt:   Ground-truth box ``(x, y, w, h)``.

    Returns:
        Normalized error in ``[0, ∞)``.  Returns ``float("inf")`` when the
        ground-truth box has zero area (degenerate annotation).
    """
    px, py, pw, ph = pred
    gx, gy, gw, gh = gt

    gt_scale = float(np.sqrt(gw * gh))
    if gt_scale <= 0.0:
        return float("inf")

    dx = (px + pw / 2) - (gx + gw / 2)
    dy = (py + ph / 2) - (gy + gh / 2)
    return float(np.sqrt(dx * dx + dy * dy)) / gt_scale


@dataclass
class AccuracyMetrics:
    """Scalar accuracy summary for a tracker on a dataset or sequence.

    Covers the full GOT-10k evaluation protocol (AO = ``mean_iou``,
    SR@0.5 = ``sr_05``, SR@0.75 = ``sr_075``) as well as the OTB
    success/precision AUC and the LaSOT/GOT-10k normalized precision AUC.
    """

    mean_iou: float
    """Mean IoU across all evaluated frames.  Equivalent to GOT-10k AO."""

    success_auc: float
    """Area Under the Success Curve (IoU thresholds 0 → 1)."""

    precision_auc: float
    """Normalised AUC of the Precision Curve (distance thresholds 0 → 50 px)."""

    norm_precision_auc: float
    """Normalised AUC of the Normalized Precision Curve (thresholds 0 → 0.5).

    Uses scale-invariant centre error.  NormPrec at threshold 0.2 is the
    canonical single-number summary used by GOT-10k and LaSOT.
    """

    sr_05: float
    """Success Rate at IoU threshold 0.5 — GOT-10k protocol SR₀.₅."""

    sr_075: float
    """Success Rate at IoU threshold 0.75 — GOT-10k protocol SR₀.₇₅."""

    def __str__(self) -> str:
        return (
            f"AccuracyMetrics("
            f"mIoU/AO={self.mean_iou:.4f}, "
            f"SR@0.5={self.sr_05:.4f}, "
            f"SR@0.75={self.sr_075:.4f}, "
            f"success_AUC={self.success_auc:.4f}, "
            f"precision_AUC={self.precision_auc:.4f}, "
            f"norm_prec_AUC={self.norm_precision_auc:.4f})"
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
        print(result.sr_05)       # GOT-10k SR@0.5
        print(result.sr_075)      # GOT-10k SR@0.75
        print(result.norm_precision_auc)  # LaSOT/GOT-10k NormPrec AUC
    """

    def batch_iou(self, preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
        """Vectorised per-frame IoU.

        Args:
            preds: ``(N, 4)`` array of predicted boxes ``(x, y, w, h)``.
            gts:   ``(N, 4)`` array of ground-truth boxes ``(x, y, w, h)``.

        Returns:
            ``(N,)`` float array of IoU values in ``[0, 1]``.
        """
        n = min(len(preds), len(gts))
        px, py, pw, ph = preds[:n, 0], preds[:n, 1], preds[:n, 2], preds[:n, 3]
        gx, gy, gw, gh = gts[:n, 0], gts[:n, 1], gts[:n, 2], gts[:n, 3]

        ix1 = np.maximum(px, gx)
        iy1 = np.maximum(py, gy)
        ix2 = np.minimum(px + pw, gx + gw)
        iy2 = np.minimum(py + ph, gy + gh)

        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        union = pw * ph + gw * gh - inter

        valid = (pw > 0) & (ph > 0) & (gw > 0) & (gh > 0) & (union > 0)
        result = np.zeros(n, dtype=np.float64)
        result[valid] = inter[valid] / union[valid]
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
        """Normalized precision curve: fraction of frames with norm-error < threshold.

        The *normalized centre error* divides the pixel centre distance by
        ``sqrt(gt_w * gt_h)``, making it invariant to target scale.  This is
        the standard evaluation metric used by GOT-10k and LaSOT.

        The canonical single-number summary is **NormPrec@0.2** — the rate at
        the 0.2 threshold, read off the returned arrays with
        ``np.interp(0.2, thresholds, rates)``.

        Args:
            preds:      ``(N, 4)`` predicted boxes ``(x, y, w, h)``.
            gts:        ``(N, 4)`` ground-truth boxes ``(x, y, w, h)``.
            thresholds: Normalized distance thresholds (default: 0 … 0.5,
                        51 evenly-spaced points).

        Returns:
            ``(thresholds, precision_rates)`` — both shape ``(T,)``.
        """
        if thresholds is None:
            thresholds = np.linspace(0.0, 0.5, 51)
        n = min(len(preds), len(gts))
        norm_errs = np.array(
            [normalized_center_error(tuple(preds[i]), tuple(gts[i])) for i in range(n)]  # type: ignore[arg-type]
        )
        # Frames with degenerate GT (inf error) never count as precise.
        finite_mask = np.isfinite(norm_errs)
        rates = np.zeros(len(thresholds), dtype=np.float64)
        if finite_mask.any():
            fe = norm_errs[finite_mask]
            rates = np.array([(norm_errs < t).mean() for t in thresholds])
        return thresholds, rates

    def sr_at_threshold(self, ious: np.ndarray, threshold: float) -> float:
        """Fraction of frames with IoU at or above a fixed threshold.

        This is the **Success Rate** metric used by GOT-10k:

        * SR₀.₅  — ``sr_at_threshold(ious, 0.5)``
        * SR₀.₇₅ — ``sr_at_threshold(ious, 0.75)``

        Args:
            ious:      Per-frame IoU array, shape ``(N,)``.
            threshold: IoU threshold in ``[0, 1]``.

        Returns:
            Float in ``[0, 1]``.
        """
        if len(ious) == 0:
            return 0.0
        return float((ious >= threshold).mean())

    def compute_all(
        self,
        preds: np.ndarray,
        gts: np.ndarray,
    ) -> AccuracyMetrics:
        """Compute all accuracy metrics in one call.

        Computes and returns:
        - ``mean_iou`` / GOT-10k AO
        - ``success_auc`` (OTB)
        - ``precision_auc`` (OTB, pixel distance)
        - ``norm_precision_auc`` (GOT-10k / LaSOT, scale-invariant)
        - ``sr_05`` — GOT-10k SR₀.₅
        - ``sr_075`` — GOT-10k SR₀.₇₅

        Args:
            preds: ``(N, 4)`` predicted boxes.
            gts:   ``(N, 4)`` ground-truth boxes.

        Returns:
            :class:`AccuracyMetrics` with all scalar summaries populated.
        """
        ious = self.batch_iou(preds, gts)

        try:
            _trapz = np.trapezoid  # numpy >= 2.0
        except AttributeError:
            _trapz = np.trapz  # numpy < 2.0

        thr_iou, sr = self.success_curve(ious)
        success_auc = float(_trapz(sr, thr_iou))

        thr_dist, pr = self.precision_curve(preds, gts)
        prec_auc = float(_trapz(pr, thr_dist) / thr_dist[-1]) if thr_dist[-1] > 0 else 0.0

        thr_norm, npr = self.normalized_precision_curve(preds, gts)
        norm_prec_auc = (
            float(_trapz(npr, thr_norm) / thr_norm[-1]) if thr_norm[-1] > 0 else 0.0
        )

        return AccuracyMetrics(
            mean_iou=float(ious.mean()) if len(ious) else 0.0,
            success_auc=success_auc,
            precision_auc=prec_auc,
            norm_precision_auc=norm_prec_auc,
            sr_05=self.sr_at_threshold(ious, 0.5),
            sr_075=self.sr_at_threshold(ious, 0.75),
        )

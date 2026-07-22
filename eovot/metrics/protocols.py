"""Protocol-specific evaluation metrics for standard VOT benchmarks.

Each major benchmark publishes results under a slightly different evaluation
protocol.  Using the wrong scalar for comparison leads to non-reproducible
numbers that cannot be directly contrasted with published leaderboards.
This module provides first-class implementations of the protocols used by
the three dominant public benchmarks:

GOT-10k
    Reports **AO** (Average Overlap, i.e. mean IoU), **SR₀.₅** (Success
    Rate at IoU ≥ 0.5), and **SR₀.₇₅** (Success Rate at IoU ≥ 0.75).
    These are the official submission metrics for the GOT-10k server.

LaSOT
    Reports **Success AUC** (area under the overlap success curve, same as
    OTB), **Normalized Precision** (NP — centre error normalized by target
    size, not pixel distance), and **Normalized Precision AUC**.

OTB
    Reports **Success AUC** and a single-number **Precision Score** — the
    fraction of frames whose centre error is < 20 px (the canonical OTB
    threshold, not the full AUC).

All functions accept ``(N, 4)`` arrays of predicted and GT bounding boxes in
``(x, y, w, h)`` format and return plain Python dicts so results can be
merged into any existing result dict.

Typical usage::

    from eovot.metrics.protocols import GOT10kMetrics, LaSOTMetrics, OTBMetrics

    got10k = GOT10kMetrics()
    result  = got10k.compute(preds, gts)
    print(result)
    # {'ao': 0.4812, 'sr_50': 0.6421, 'sr_75': 0.2109}

    otb = OTBMetrics()
    result = otb.compute(preds, gts)
    print(result)
    # {'success_auc': 0.531, 'precision_score': 0.762}

    # Protocol dispatcher — choose protocol by name from a config:
    engine = ProtocolMetricsEngine()
    result = engine.compute("got10k", preds, gts)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

# Reuse the vectorised IoU and centre-distance already in the accuracy module.
from .accuracy import MetricsEngine as _MetricsEngine


# ---------------------------------------------------------------------------
# Internal helpers (not part of the public API)
# ---------------------------------------------------------------------------

def _iou_array(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """Return per-frame IoU array, aligned to the shorter sequence."""
    _me = _MetricsEngine()
    return _me.batch_iou(preds, gts)


def _center_distances(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """Return per-frame centre-to-centre Euclidean distance (pixels)."""
    _me = _MetricsEngine()
    return _me.batch_center_distance(preds, gts)


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # type: ignore[attr-defined]
    return float(fn(y, x))


def _success_curve(
    ious: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    rates = np.array([(ious >= t).mean() for t in thresholds])
    return thresholds, rates


# ---------------------------------------------------------------------------
# GOT-10k protocol
# ---------------------------------------------------------------------------

class GOT10kMetrics:
    """Official GOT-10k evaluation protocol metrics.

    The GOT-10k benchmark server accepts submissions and reports three
    scalars:

    **AO** — Average Overlap (mean IoU across all frames).

    **SR₀.₅** — Success Rate at IoU ≥ 0.5: fraction of frames with
    overlap ≥ 0.5.

    **SR₀.₇₅** — Success Rate at IoU ≥ 0.75: fraction of frames with
    overlap ≥ 0.75.

    References:
        Ming-Hsuan Yang et al., "GOT-10k: A Large High-Diversity
        Benchmark for Generic Object Tracking in the Wild", TPAMI 2021.
    """

    def compute(
        self,
        preds: np.ndarray,
        gts: np.ndarray,
    ) -> Dict[str, float]:
        """Compute GOT-10k protocol scalars.

        Args:
            preds: Predicted boxes, shape ``(N, 4)`` in ``(x, y, w, h)``.
            gts:   Ground-truth boxes, shape ``(N, 4)`` in ``(x, y, w, h)``.

        Returns:
            Dict with keys ``"ao"``, ``"sr_50"``, ``"sr_75"``.
        """
        ious = _iou_array(preds, gts)
        if len(ious) == 0:
            return {"ao": 0.0, "sr_50": 0.0, "sr_75": 0.0}

        ao = float(ious.mean())
        sr_50 = float((ious >= 0.5).mean())
        sr_75 = float((ious >= 0.75).mean())

        return {"ao": round(ao, 4), "sr_50": round(sr_50, 4), "sr_75": round(sr_75, 4)}

    def compute_multi_sequence(
        self,
        sequence_preds: list,
        sequence_gts: list,
    ) -> Dict[str, float]:
        """Compute GOT-10k metrics averaged across multiple sequences.

        GOT-10k averages AO and SR per sequence first, then averages across
        sequences (not globally across all frames).

        Args:
            sequence_preds: List of ``(N_i, 4)`` prediction arrays.
            sequence_gts:   List of ``(N_i, 4)`` ground-truth arrays.

        Returns:
            Dict with keys ``"ao"``, ``"sr_50"``, ``"sr_75"``.
        """
        if not sequence_preds:
            return {"ao": 0.0, "sr_50": 0.0, "sr_75": 0.0}

        per_seq = [self.compute(p, g) for p, g in zip(sequence_preds, sequence_gts)]
        ao = float(np.mean([r["ao"] for r in per_seq]))
        sr50 = float(np.mean([r["sr_50"] for r in per_seq]))
        sr75 = float(np.mean([r["sr_75"] for r in per_seq]))
        return {"ao": round(ao, 4), "sr_50": round(sr50, 4), "sr_75": round(sr75, 4)}


# ---------------------------------------------------------------------------
# LaSOT protocol
# ---------------------------------------------------------------------------

class LaSOTMetrics:
    """LaSOT evaluation protocol metrics.

    LaSOT uses three evaluation scalars:

    **Success AUC** — area under the overlap success curve (same as OTB).

    **Normalized Precision** (**NP**) at threshold 0.5 — fraction of
    frames where the centre error, *normalized by the square root of the
    GT box area*, is < 0.5.  This removes sensitivity to target scale.

    **Normalized Precision AUC** — area under the NP curve swept from
    0 to 0.5 (the LaSOT standard range), divided by 0.5 to normalize to
    [0, 1].

    References:
        Fan et al., "LaSOT: A High-quality Benchmark for Large-scale
        Single Object Tracking", CVPR 2019.
    """

    # Threshold at which the canonical NP scalar is reported.
    NP_THRESHOLD: float = 0.5

    def _normalized_center_errors(
        self,
        preds: np.ndarray,
        gts: np.ndarray,
    ) -> np.ndarray:
        """Per-frame centre error normalized by sqrt(GT area)."""
        n = min(len(preds), len(gts))
        if n == 0:
            return np.empty(0)
        p = np.asarray(preds[:n], dtype=np.float64)
        g = np.asarray(gts[:n], dtype=np.float64)

        # Pixel centre distances
        pc = p[:, :2] + p[:, 2:] / 2.0
        gc = g[:, :2] + g[:, 2:] / 2.0
        pixel_dists = np.sqrt(np.sum((pc - gc) ** 2, axis=1))

        # Normalise by sqrt of GT box area (avoid divide-by-zero)
        gt_areas = g[:, 2] * g[:, 3]
        safe_areas = np.maximum(gt_areas, 1.0)
        return pixel_dists / np.sqrt(safe_areas)

    def compute(
        self,
        preds: np.ndarray,
        gts: np.ndarray,
    ) -> Dict[str, float]:
        """Compute LaSOT protocol scalars.

        Args:
            preds: Predicted boxes, shape ``(N, 4)`` in ``(x, y, w, h)``.
            gts:   Ground-truth boxes, shape ``(N, 4)`` in ``(x, y, w, h)``.

        Returns:
            Dict with keys ``"success_auc"``, ``"norm_precision"``,
            ``"norm_precision_auc"``.
        """
        ious = _iou_array(preds, gts)
        if len(ious) == 0:
            return {"success_auc": 0.0, "norm_precision": 0.0, "norm_precision_auc": 0.0}

        # Success AUC
        thr_iou, sr = _success_curve(ious)
        success_auc = _trapz(sr, thr_iou)

        # Normalized precision curve (0 → NP_THRESHOLD range)
        norm_errs = self._normalized_center_errors(preds, gts)
        np_thresholds = np.linspace(0.0, self.NP_THRESHOLD, 51)
        np_rates = np.array([(norm_errs < t).mean() for t in np_thresholds])
        norm_precision = float((norm_errs < self.NP_THRESHOLD).mean())
        norm_precision_auc = _trapz(np_rates, np_thresholds) / self.NP_THRESHOLD

        return {
            "success_auc": round(success_auc, 4),
            "norm_precision": round(norm_precision, 4),
            "norm_precision_auc": round(norm_precision_auc, 4),
        }


# ---------------------------------------------------------------------------
# OTB protocol
# ---------------------------------------------------------------------------

class OTBMetrics:
    """OTB (One-pass Tracking Benchmark) evaluation protocol metrics.

    OTB reports two scalars:

    **Success AUC** — area under the overlap success curve (IoU swept
    from 0 → 1).

    **Precision Score** — fraction of frames whose predicted centre is
    within **20 pixels** of the GT centre.  This is the canonical OTB
    threshold; results quoted with other thresholds are not directly
    comparable to OTB numbers.

    References:
        Wu et al., "Object Tracking Benchmark", TPAMI 2015.
    """

    # The canonical OTB centre-distance threshold (pixels).
    PRECISION_THRESHOLD_PX: float = 20.0

    def compute(
        self,
        preds: np.ndarray,
        gts: np.ndarray,
    ) -> Dict[str, float]:
        """Compute OTB protocol scalars.

        Args:
            preds: Predicted boxes, shape ``(N, 4)`` in ``(x, y, w, h)``.
            gts:   Ground-truth boxes, shape ``(N, 4)`` in ``(x, y, w, h)``.

        Returns:
            Dict with keys ``"success_auc"`` and ``"precision_score_20px"``.
        """
        ious = _iou_array(preds, gts)
        if len(ious) == 0:
            return {"success_auc": 0.0, "precision_score_20px": 0.0}

        thr_iou, sr = _success_curve(ious)
        success_auc = _trapz(sr, thr_iou)

        dists = _center_distances(preds, gts)
        precision_score = float((dists < self.PRECISION_THRESHOLD_PX).mean())

        return {
            "success_auc": round(success_auc, 4),
            "precision_score_20px": round(precision_score, 4),
        }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

#: Maps short protocol names to their metric class.
PROTOCOL_REGISTRY: Dict[str, type] = {
    "got10k": GOT10kMetrics,
    "lasot": LaSOTMetrics,
    "otb": OTBMetrics,
}


class ProtocolMetricsEngine:
    """Compute protocol-specific metrics from a config-driven name string.

    This lets experiment configs and CLI flags specify a protocol by name
    (``"got10k"``, ``"lasot"``, ``"otb"``) and get the correct scalars
    without importing each class directly.

    Example::

        engine = ProtocolMetricsEngine()
        got = engine.compute("got10k", preds, gts)
        otb = engine.compute("otb",    preds, gts)

    Attributes:
        PROTOCOLS: Sorted list of available protocol names.
    """

    PROTOCOLS = sorted(PROTOCOL_REGISTRY)

    def compute(
        self,
        protocol: str,
        preds: np.ndarray,
        gts: np.ndarray,
    ) -> Dict[str, float]:
        """Compute metrics for *protocol* from prediction/GT arrays.

        Args:
            protocol: One of ``"got10k"``, ``"lasot"``, ``"otb"``.
            preds:    ``(N, 4)`` predicted boxes in ``(x, y, w, h)``.
            gts:      ``(N, 4)`` ground-truth boxes in ``(x, y, w, h)``.

        Returns:
            Dict of protocol-specific scalars.

        Raises:
            ValueError: If *protocol* is not in :data:`PROTOCOL_REGISTRY`.
        """
        key = protocol.lower()
        cls = PROTOCOL_REGISTRY.get(key)
        if cls is None:
            raise ValueError(
                f"Unknown protocol '{protocol}'. "
                f"Available: {self.PROTOCOLS}"
            )
        return cls().compute(preds, gts)

    def compute_all(
        self,
        preds: np.ndarray,
        gts: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics for all registered protocols at once.

        Args:
            preds: ``(N, 4)`` predicted boxes.
            gts:   ``(N, 4)`` ground-truth boxes.

        Returns:
            Dict mapping protocol name → metrics dict.
        """
        return {proto: self.compute(proto, preds, gts) for proto in self.PROTOCOLS}

"""Deployment-Readiness Score (DRS) for edge tracker evaluation.

Standard benchmarks evaluate trackers on a single axis — accuracy (mIoU,
success AUC) or speed (FPS).  EOVOT's thesis is that edge deployment requires
jointly satisfying *four* constraints:

1. **Accuracy** — the tracker must follow the target reliably (success AUC).
2. **Efficiency** — the tracker must run within device throughput/memory limits
   (Edge Efficiency Score).
3. **Robustness** — the tracker must recover from failures, not accumulate drift
   (EAO, survival rate from :class:`~eovot.metrics.robustness.RobustnessAnalyzer`).
4. **Temporal consistency** — the tracker's bounding-box stream must be smooth
   enough for downstream consumers: robot controllers, surveillance pipelines,
   AR overlays (smoothness score from
   :class:`~eovot.metrics.temporal.TemporalConsistencyAnalyzer`).

DRS formulation
---------------
The Deployment-Readiness Score aggregates these four components into one
comparable scalar via a *weighted geometric mean*:

.. math::

    \\text{DRS} = A^{w_a} \\cdot E^{w_e} \\cdot R^{w_r} \\cdot S^{w_s}

where :math:`A, E, R, S \\in [0, 1]` are normalised component scores and
:math:`w_a + w_e + w_r + w_s = 1`.  A geometric mean is chosen over an
arithmetic mean because it enforces that *no single weak dimension can be
compensated by strength elsewhere* — a tracker with zero robustness has
DRS = 0 regardless of its accuracy.

Component normalisation
-----------------------
* :math:`A` — success AUC (already in :math:`[0, 1]`).
* :math:`E` — EES normalised to :math:`[0, 1]` by :math:`E / (1 + E)` (the
  EES has no fixed upper bound, so a soft saturation is used).
* :math:`R` — mean EAO from :class:`RobustnessAnalyzer` (already in
  :math:`[0, 1]`).
* :math:`S` — smoothness score from :class:`TemporalConsistencyAnalyzer`
  (already in :math:`(0, 1]`).

Default weights reflect the EOVOT project's deployment focus:
accuracy 35 %, efficiency 30 %, robustness 20 %, smoothness 15 %.
Weights are configurable per evaluation context.

Typical usage::

    from eovot.metrics.deployment import DeploymentReadinessAnalyzer, DeploymentReadinessScore

    analyzer = DeploymentReadinessAnalyzer(memory_budget_mb=512.0)
    score = analyzer.compute(
        benchmark_result,
        sequence_predictions={seq_name: preds_array, ...},
    )
    print(score)
    # DeploymentReadinessScore[MOSSE] DRS=0.4231  A=0.521 E=0.389 R=0.612 S=0.743
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .accuracy import MetricsEngine
from .efficiency import EfficiencyMetricsEngine
from .robustness import RobustnessAnalyzer
from .temporal import TemporalConsistencyAnalyzer


@dataclass
class DeploymentReadinessScore:
    """Four-component Deployment-Readiness Score for one tracker/dataset pair.

    Attributes:
        tracker_name:   Human-readable tracker identifier.
        dataset_name:   Dataset the tracker was evaluated on.
        drs:            Composite Deployment-Readiness Score in ``[0, 1]``.
        accuracy:       Normalised accuracy component in ``[0, 1]``
                        (success AUC).
        efficiency:     Normalised efficiency component in ``[0, 1]``
                        (soft-normalised EES).
        robustness:     Normalised robustness component in ``[0, 1]``
                        (mean EAO).
        smoothness:     Normalised smoothness component in ``(0, 1]``
                        (temporal consistency smoothness score).
        weights:        ``(w_a, w_e, w_r, w_s)`` weights used to compute DRS.
        mean_fps:       Mean tracker throughput (frames / second).
        peak_memory_mb: Peak RSS memory footprint (megabytes).
        mean_iou:       Mean IoU across all evaluated frames.
        ees:            Raw Edge Efficiency Score (before normalisation).
        mean_eao:       Mean Expected Average Overlap across sequences.
    """

    tracker_name: str
    dataset_name: str
    drs: float
    accuracy: float
    efficiency: float
    robustness: float
    smoothness: float
    weights: Tuple[float, float, float, float]
    mean_fps: float
    peak_memory_mb: float
    mean_iou: float
    ees: float
    mean_eao: float

    def __str__(self) -> str:
        return (
            f"DeploymentReadinessScore[{self.tracker_name} on {self.dataset_name}] "
            f"DRS={self.drs:.4f}  "
            f"A={self.accuracy:.3f}  E={self.efficiency:.3f}  "
            f"R={self.robustness:.3f}  S={self.smoothness:.3f}  "
            f"FPS={self.mean_fps:.1f}  mIoU={self.mean_iou:.4f}"
        )

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON export."""
        return {
            "tracker_name": self.tracker_name,
            "dataset_name": self.dataset_name,
            "drs": round(self.drs, 6),
            "accuracy": round(self.accuracy, 6),
            "efficiency": round(self.efficiency, 6),
            "robustness": round(self.robustness, 6),
            "smoothness": round(self.smoothness, 6),
            "weights": {
                "accuracy": self.weights[0],
                "efficiency": self.weights[1],
                "robustness": self.weights[2],
                "smoothness": self.weights[3],
            },
            "mean_fps": round(self.mean_fps, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "mean_iou": round(self.mean_iou, 6),
            "ees": round(self.ees, 6),
            "mean_eao": round(self.mean_eao, 6),
        }


class DeploymentReadinessAnalyzer:
    """Compute Deployment-Readiness Scores from :class:`~eovot.benchmark.engine.BenchmarkResult` objects.

    Args:
        memory_budget_mb: Acceptable peak-memory ceiling for edge deployment
            (megabytes).  Passed to the internal
            :class:`~eovot.metrics.efficiency.EfficiencyMetricsEngine`.
            Default: ``512.0``.
        weight_accuracy:   DRS weight for the accuracy component.  Default: ``0.35``.
        weight_efficiency: DRS weight for the efficiency component.  Default: ``0.30``.
        weight_robustness: DRS weight for the robustness component.  Default: ``0.20``.
        weight_smoothness: DRS weight for the temporal-consistency component.
            Default: ``0.15``.

    Raises:
        ValueError: If the four weights do not sum to approximately 1.0
            (tolerance: 1e-6).
    """

    def __init__(
        self,
        memory_budget_mb: float = 512.0,
        weight_accuracy: float = 0.35,
        weight_efficiency: float = 0.30,
        weight_robustness: float = 0.20,
        weight_smoothness: float = 0.15,
    ) -> None:
        total = weight_accuracy + weight_efficiency + weight_robustness + weight_smoothness
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"DRS weights must sum to 1.0, got {total:.6f} "
                f"(w_a={weight_accuracy}, w_e={weight_efficiency}, "
                f"w_r={weight_robustness}, w_s={weight_smoothness})"
            )
        self.memory_budget_mb = memory_budget_mb
        self.weights = (weight_accuracy, weight_efficiency, weight_robustness, weight_smoothness)

        self._eff_engine = EfficiencyMetricsEngine(memory_budget_mb=memory_budget_mb)
        self._rob_analyzer = RobustnessAnalyzer()
        self._temporal_analyzer = TemporalConsistencyAnalyzer()
        self._metrics_engine = MetricsEngine()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        result: "BenchmarkResult",
        sequence_predictions: Optional[Dict[str, np.ndarray]] = None,
    ) -> DeploymentReadinessScore:
        """Compute the DRS for a single tracker/dataset result.

        Args:
            result: A :class:`~eovot.benchmark.engine.BenchmarkResult`
                containing per-sequence metrics and profiling data.
            sequence_predictions: Optional mapping ``{sequence_name: preds}``
                where ``preds`` is an ``(N, 4)`` array of predicted bounding
                boxes.  When provided, the temporal-consistency score is
                computed from the actual prediction trajectories.  When
                absent, the smoothness score is estimated from the stored
                per-frame IoU arrays as a proxy (less accurate but always
                available).

        Returns:
            :class:`DeploymentReadinessScore` with all components populated.
        """
        w_a, w_e, w_r, w_s = self.weights

        # --- Component 1: Accuracy ---
        accuracy = self._compute_accuracy(result)

        # --- Component 2: Efficiency ---
        raw_ees = self._eff_engine.edge_efficiency_score(
            mean_iou=result.mean_iou,
            fps=result.mean_fps,
            peak_memory_mb=result.peak_memory_mb,
        )
        # Soft-normalise EES to [0, 1]: x / (1 + x)
        efficiency = raw_ees / (1.0 + raw_ees) if raw_ees >= 0 else 0.0

        # --- Component 3: Robustness ---
        mean_eao = self._compute_robustness(result)
        robustness = float(np.clip(mean_eao, 0.0, 1.0))

        # --- Component 4: Temporal Consistency (Smoothness) ---
        smoothness = self._compute_smoothness(result, sequence_predictions)

        # --- DRS: weighted geometric mean ---
        components = [accuracy, efficiency, robustness, smoothness]
        weights = list(self.weights)
        drs = self._geometric_mean(components, weights)

        return DeploymentReadinessScore(
            tracker_name=result.tracker_name,
            dataset_name=result.dataset_name,
            drs=drs,
            accuracy=accuracy,
            efficiency=efficiency,
            robustness=robustness,
            smoothness=smoothness,
            weights=self.weights,
            mean_fps=result.mean_fps,
            peak_memory_mb=result.peak_memory_mb,
            mean_iou=result.mean_iou,
            ees=raw_ees,
            mean_eao=mean_eao,
        )

    def rank(
        self,
        results: List["BenchmarkResult"],
        sequence_predictions: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    ) -> List[DeploymentReadinessScore]:
        """Compute DRS for a list of benchmark results, sorted best-first.

        Args:
            results: One :class:`~eovot.benchmark.engine.BenchmarkResult`
                per tracker/dataset combination.
            sequence_predictions: Optional outer mapping
                ``{tracker_name: {sequence_name: preds}}``.

        Returns:
            List of :class:`DeploymentReadinessScore` sorted by DRS descending.
        """
        scores: List[DeploymentReadinessScore] = []
        for r in results:
            sp = None
            if sequence_predictions is not None:
                sp = sequence_predictions.get(r.tracker_name)
            scores.append(self.compute(r, sequence_predictions=sp))
        scores.sort(key=lambda s: s.drs, reverse=True)
        return scores

    def to_markdown_table(self, scores: List[DeploymentReadinessScore]) -> str:
        """Format ranked DRS scores as a Markdown table.

        Args:
            scores: Output of :meth:`rank` or a list of
                :class:`DeploymentReadinessScore` objects.

        Returns:
            Multi-line Markdown string ready to embed in reports or READMEs.
        """
        header = (
            "| Rank | Tracker | Dataset | DRS | Accuracy | Efficiency | "
            "Robustness | Smoothness | mIoU | FPS | Mem (MB) |"
        )
        sep = (
            "|------|---------|---------|----:|---------:|-----------:|"
            "----------:|-----------:|-----:|----:|---------:|"
        )
        lines = [header, sep]
        for rank, s in enumerate(scores, start=1):
            lines.append(
                f"| {rank} | {s.tracker_name} | {s.dataset_name} "
                f"| {s.drs:.4f} | {s.accuracy:.4f} | {s.efficiency:.4f} "
                f"| {s.robustness:.4f} | {s.smoothness:.4f} "
                f"| {s.mean_iou:.4f} | {s.mean_fps:.1f} | {s.peak_memory_mb:.1f} |"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _geometric_mean(values: List[float], weights: List[float]) -> float:
        """Weighted geometric mean: prod(v_i ^ w_i).

        Returns 0 if any component is zero (geometric mean enforces joint
        performance).
        """
        result = 1.0
        for v, w in zip(values, weights):
            clamped = max(0.0, min(1.0, float(v)))
            if clamped <= 0.0:
                return 0.0
            result *= clamped ** w
        return float(result)

    def _compute_accuracy(self, result: "BenchmarkResult") -> float:
        """Return the best available accuracy scalar from the result."""
        # Prefer success AUC (most informative), fall back to mean IoU
        sauc = result.mean_success_auc
        if sauc is not None:
            return float(np.clip(sauc, 0.0, 1.0))
        return float(np.clip(result.mean_iou, 0.0, 1.0))

    def _compute_robustness(self, result: "BenchmarkResult") -> float:
        """Compute mean EAO across all sequences in the result."""
        from ..benchmark.engine import BenchmarkResult  # avoid circular at module level
        eaos: List[float] = []
        for sr in result.sequence_results:
            rob = self._rob_analyzer.analyze_sequence(
                sr.ious,
                tracker_name=result.tracker_name,
                sequence_name=sr.sequence_name,
            )
            eaos.append(rob.eao)
        return float(np.mean(eaos)) if eaos else 0.0

    def _compute_smoothness(
        self,
        result: "BenchmarkResult",
        sequence_predictions: Optional[Dict[str, np.ndarray]],
    ) -> float:
        """Return mean smoothness score across all sequences.

        When ``sequence_predictions`` is supplied the temporal analyzer runs
        on the actual trajectories.  When absent, the IoU-based proxy
        ``1 / (1 + iou_std)`` is used (penalises jittery IoU streams).
        """
        smoothness_scores: List[float] = []
        for sr in result.sequence_results:
            if sequence_predictions is not None and sr.sequence_name in sequence_predictions:
                preds = sequence_predictions[sr.sequence_name]
                tc = self._temporal_analyzer.analyze(
                    preds,
                    tracker_name=result.tracker_name,
                    sequence_name=sr.sequence_name,
                )
                smoothness_scores.append(tc.smoothness_score)
            else:
                # IoU-variance proxy: stable tracker ≈ low variance IoU stream
                iou_std = float(sr.ious.std()) if len(sr.ious) > 1 else 0.0
                proxy = 1.0 / (1.0 + iou_std * 5.0)
                smoothness_scores.append(proxy)
        return float(np.mean(smoothness_scores)) if smoothness_scores else 0.5

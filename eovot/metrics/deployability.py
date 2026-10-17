"""Composite edge-deployability scoring for EOVOT tracker comparison.

Standard VOT benchmarks evaluate trackers along a single axis (accuracy).
EOVOT extends this with efficiency, robustness, and temporal consistency —
but exposing four separate metric objects creates a decision burden: which
tracker to pick for a given edge deployment scenario?

This module resolves that burden with a single composite scalar:

    Deployability Score (DS)
    ~~~~~~~~~~~~~~~~~~~~~~~~

    DS = w_acc × A_norm + w_eff × E_norm + w_rob × R_norm + w_smooth × S_norm

Where each dimension is normalised to [0, 1] within the comparison set:

* **Accuracy** (A) — success AUC from
  :class:`~eovot.metrics.accuracy.MetricsEngine` (falls back to mean IoU).
* **Efficiency** (E) — Edge Efficiency Score from
  :class:`~eovot.metrics.efficiency.EfficiencyMetricsEngine` (EES = IoU ×
  log1p(FPS) / memory_factor), normalised by the max EES in the set.
* **Robustness** (R) — EAO × survival_rate from
  :class:`~eovot.metrics.robustness.RobustnessAnalyzer`.
* **Temporal Smoothness** (S) — smoothness_score from
  :class:`~eovot.metrics.temporal.TemporalConsistencyAnalyzer`.

Default weights are equal (0.25 each).  Domain experts can up-weight the
dimensions that matter most for their deployment target:

* Edge robotics with strict real-time requirements → ``w_eff=0.4, w_smooth=0.3``
* Surveillance with long-duration inference → ``w_rob=0.4``
* Research benchmarking → equal weights (default)

Pareto-Front in DS-space
~~~~~~~~~~~~~~~~~~~~~~~~
The DS is a *single scalar* and can mislead when its components trade off
against each other.  :meth:`DeployabilityScoreEngine.pareto_front` therefore
also identifies Pareto-optimal trackers in the ``(accuracy, efficiency)``
2-D objective space — the same approach used in the peer-reviewed VOT-Nano
and PETS workshop papers.

Typical usage::

    from eovot.metrics.deployability import DeployabilityScoreEngine

    engine = DeployabilityScoreEngine(memory_budget_mb=512.0)
    scores = engine.score_results(benchmark_results)
    for entry in scores:
        print(entry)
    print(engine.to_markdown_table(scores))

Research note:
    The Deployability Score is EOVOT's original multi-dimensional ranking
    metric.  It is not part of the standard VOT, OTB, or GOT-10k protocols.
    When reporting results in papers, always report the individual component
    scores (A, E, R, S) alongside DS so reviewers can inspect the trade-offs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DeployabilityEntry:
    """Per-tracker deployability summary.

    Attributes:
        tracker_name: Human-readable tracker identifier.
        dataset_name: Dataset on which the tracker was evaluated.

        accuracy_score: Normalised accuracy component in ``[0, 1]``.
            Raw value: success AUC (or mean IoU as fallback).
        efficiency_score: Normalised efficiency component in ``[0, 1]``.
            Raw value: Edge Efficiency Score (EES).
        robustness_score: Normalised robustness component in ``[0, 1]``.
            Raw value: EAO × mean survival rate (estimated from IoU arrays).
        smoothness_score: Normalised temporal smoothness component in ``[0, 1]``.
            Raw value: mean smoothness score from TemporalConsistencyAnalyzer.

        weight_accuracy: Weight applied to the accuracy component.
        weight_efficiency: Weight applied to the efficiency component.
        weight_robustness: Weight applied to the robustness component.
        weight_smoothness: Weight applied to the smoothness component.

        deployability_score: Composite DS scalar in ``[0, 1]``.

        mean_iou: Raw mean IoU (for reference).
        fps: Raw mean FPS (for reference).
        peak_memory_mb: Raw peak RSS memory in MB (for reference).
        success_auc: Raw success AUC, or ``None`` when not computed.
        ees: Raw Edge Efficiency Score (before normalisation).

        on_pareto_front: ``True`` when no other tracker in the comparison set
            dominates this one in both (accuracy, efficiency) simultaneously.
    """

    tracker_name: str
    dataset_name: str

    # Normalised component scores
    accuracy_score: float
    efficiency_score: float
    robustness_score: float
    smoothness_score: float

    # Weights (stored for auditability)
    weight_accuracy: float
    weight_efficiency: float
    weight_robustness: float
    weight_smoothness: float

    # Composite
    deployability_score: float

    # Raw reference metrics
    mean_iou: float
    fps: float
    peak_memory_mb: float
    success_auc: Optional[float]
    ees: float

    on_pareto_front: bool = field(default=False)

    def component_dict(self) -> Dict[str, float]:
        """Return the four normalised component scores as a dict."""
        return {
            "accuracy": round(self.accuracy_score, 4),
            "efficiency": round(self.efficiency_score, 4),
            "robustness": round(self.robustness_score, 4),
            "smoothness": round(self.smoothness_score, 4),
        }

    def to_dict(self) -> Dict:
        """Serialise to a JSON-safe dict for result archiving."""
        return {
            "tracker": self.tracker_name,
            "dataset": self.dataset_name,
            "deployability_score": round(self.deployability_score, 4),
            "components": self.component_dict(),
            "weights": {
                "accuracy": self.weight_accuracy,
                "efficiency": self.weight_efficiency,
                "robustness": self.weight_robustness,
                "smoothness": self.weight_smoothness,
            },
            "raw": {
                "mean_iou": round(self.mean_iou, 4),
                "fps": round(self.fps, 2),
                "peak_memory_mb": round(self.peak_memory_mb, 2),
                "success_auc": round(self.success_auc, 4) if self.success_auc is not None else None,
                "ees": round(self.ees, 4),
            },
            "on_pareto_front": self.on_pareto_front,
        }

    def __str__(self) -> str:
        pareto = " ✓" if self.on_pareto_front else ""
        return (
            f"DeployabilityEntry({self.tracker_name}  "
            f"DS={self.deployability_score:.4f}  "
            f"[acc={self.accuracy_score:.3f}  eff={self.efficiency_score:.3f}  "
            f"rob={self.robustness_score:.3f}  smooth={self.smoothness_score:.3f}]"
            f"{pareto})"
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DeployabilityScoreEngine:
    """Compute composite edge-deployability scores for tracker comparison.

    All four metric dimensions are extracted from
    :class:`~eovot.benchmark.engine.BenchmarkResult` objects and combined
    into a single scalar using configurable weights.  Scores are normalised
    within the comparison set so the best tracker on each dimension receives
    a component score of 1.0.

    Args:
        weight_accuracy: Weight for the accuracy dimension (success AUC / mIoU).
            Default: ``0.25``.
        weight_efficiency: Weight for the efficiency dimension (EES).
            Default: ``0.25``.
        weight_robustness: Weight for the robustness dimension (EAO × survival).
            Default: ``0.25``.
        weight_smoothness: Weight for the temporal smoothness dimension.
            Default: ``0.25``.
        memory_budget_mb: Memory ceiling in MB used for EES computation.
            Default: ``512.0``.
        failure_threshold: IoU below which a frame counts as a tracking failure,
            used for robustness estimation.  Default: ``0.1``.

    Raises:
        ValueError: If weights are negative or do not sum to a positive value.

    Example::

        from eovot.metrics.deployability import DeployabilityScoreEngine

        engine = DeployabilityScoreEngine(
            weight_accuracy=0.30,
            weight_efficiency=0.30,
            weight_robustness=0.25,
            weight_smoothness=0.15,
            memory_budget_mb=512.0,
        )
        scores = engine.score_results(benchmark_results)
        print(engine.to_markdown_table(scores))
    """

    def __init__(
        self,
        weight_accuracy: float = 0.25,
        weight_efficiency: float = 0.25,
        weight_robustness: float = 0.25,
        weight_smoothness: float = 0.25,
        memory_budget_mb: float = 512.0,
        failure_threshold: float = 0.1,
    ) -> None:
        if any(w < 0 for w in [weight_accuracy, weight_efficiency, weight_robustness, weight_smoothness]):
            raise ValueError("All weights must be non-negative.")
        total_w = weight_accuracy + weight_efficiency + weight_robustness + weight_smoothness
        if total_w <= 0:
            raise ValueError("Weights must sum to a positive value.")
        if memory_budget_mb <= 0:
            raise ValueError(f"memory_budget_mb must be positive, got {memory_budget_mb}.")
        if not 0 < failure_threshold < 1:
            raise ValueError(f"failure_threshold must be in (0, 1), got {failure_threshold}.")

        # Normalise weights so they sum to 1.0
        self.weight_accuracy = weight_accuracy / total_w
        self.weight_efficiency = weight_efficiency / total_w
        self.weight_robustness = weight_robustness / total_w
        self.weight_smoothness = weight_smoothness / total_w
        self.memory_budget_mb = memory_budget_mb
        self.failure_threshold = failure_threshold

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def score_results(
        self,
        results: Sequence["BenchmarkResult"],
    ) -> List[DeployabilityEntry]:
        """Compute deployability scores for a collection of benchmark results.

        Each result represents one tracker evaluated on one dataset.  Results
        are ranked by DS (highest first) and Pareto flags are set for the
        accuracy–efficiency frontier.

        Args:
            results: One :class:`~eovot.benchmark.engine.BenchmarkResult` per
                tracker / dataset combination.  Must contain at least one result.

        Returns:
            List of :class:`DeployabilityEntry` sorted by DS (highest first),
            with ``on_pareto_front`` flags set.

        Raises:
            ValueError: If *results* is empty.
        """
        if not results:
            raise ValueError("score_results() requires at least one BenchmarkResult.")

        raw_acc = [self._raw_accuracy(r) for r in results]
        raw_eff = [self._raw_efficiency(r) for r in results]
        raw_rob = [self._raw_robustness(r) for r in results]
        raw_smooth = [self._raw_smoothness(r) for r in results]

        norm_acc = _normalise(raw_acc)
        norm_eff = _normalise(raw_eff)
        norm_rob = _normalise(raw_rob)
        norm_smooth = _normalise(raw_smooth)

        entries: List[DeployabilityEntry] = []
        for i, result in enumerate(results):
            ds = (
                self.weight_accuracy * norm_acc[i]
                + self.weight_efficiency * norm_eff[i]
                + self.weight_robustness * norm_rob[i]
                + self.weight_smoothness * norm_smooth[i]
            )
            sauc = result.mean_success_auc  # may be None
            entries.append(
                DeployabilityEntry(
                    tracker_name=result.tracker_name,
                    dataset_name=result.dataset_name,
                    accuracy_score=norm_acc[i],
                    efficiency_score=norm_eff[i],
                    robustness_score=norm_rob[i],
                    smoothness_score=norm_smooth[i],
                    weight_accuracy=self.weight_accuracy,
                    weight_efficiency=self.weight_efficiency,
                    weight_robustness=self.weight_robustness,
                    weight_smoothness=self.weight_smoothness,
                    deployability_score=float(ds),
                    mean_iou=result.mean_iou,
                    fps=result.mean_fps,
                    peak_memory_mb=result.peak_memory_mb,
                    success_auc=sauc,
                    ees=raw_eff[i],
                )
            )

        self._mark_pareto_front(entries)
        entries.sort(key=lambda e: e.deployability_score, reverse=True)
        return entries

    # ------------------------------------------------------------------
    # Pareto-front computation
    # ------------------------------------------------------------------

    def _mark_pareto_front(self, entries: List[DeployabilityEntry]) -> None:
        """Mark entries on the accuracy–efficiency Pareto front (in-place).

        Tracker A dominates tracker B iff A.mean_iou >= B.mean_iou AND
        A.ees >= B.ees with at least one strict inequality.
        """
        for i, candidate in enumerate(entries):
            dominated = False
            for j, other in enumerate(entries):
                if i == j:
                    continue
                if (
                    other.mean_iou >= candidate.mean_iou
                    and other.ees >= candidate.ees
                    and (other.mean_iou > candidate.mean_iou or other.ees > candidate.ees)
                ):
                    dominated = True
                    break
            candidate.on_pareto_front = not dominated

    def pareto_front(
        self, results: Sequence["BenchmarkResult"]
    ) -> List[DeployabilityEntry]:
        """Return only the Pareto-optimal trackers in (accuracy, efficiency) space.

        Args:
            results: Benchmark results to analyse.

        Returns:
            Subset of :meth:`score_results` entries where ``on_pareto_front`` is True,
            sorted by DS (highest first).
        """
        all_entries = self.score_results(results)
        return [e for e in all_entries if e.on_pareto_front]

    # ------------------------------------------------------------------
    # Raw dimension extractors
    # ------------------------------------------------------------------

    def _raw_accuracy(self, result: "BenchmarkResult") -> float:
        """Extract the accuracy scalar from a BenchmarkResult.

        Preference order: success AUC → mean IoU.
        """
        sauc = result.mean_success_auc
        return float(sauc) if sauc is not None else result.mean_iou

    def _raw_efficiency(self, result: "BenchmarkResult") -> float:
        """Compute the Edge Efficiency Score for a BenchmarkResult.

        EES = mean_iou × log1p(fps) / (1 + peak_memory_mb / memory_budget_mb)
        """
        if result.mean_fps <= 0 or result.mean_iou < 0:
            return 0.0
        memory_factor = 1.0 + result.peak_memory_mb / self.memory_budget_mb
        return float(result.mean_iou * math.log1p(result.mean_fps) / memory_factor)

    def _raw_robustness(self, result: "BenchmarkResult") -> float:
        """Estimate a robustness scalar from per-sequence IoU arrays.

        Computes EAO (mean IoU after a 5-frame burn-in) × mean survival rate
        (fraction of frames with IoU >= failure_threshold), averaged across
        all sequences in the result.

        Returns a value in [0, 1].
        """
        from .robustness import RobustnessAnalyzer

        analyzer = RobustnessAnalyzer(
            failure_threshold=self.failure_threshold,
            burn_in_frames=5,
        )
        if not result.sequence_results:
            return 0.0

        eao_list: List[float] = []
        survival_list: List[float] = []
        for sr in result.sequence_results:
            r = analyzer.analyze_sequence(sr.ious)
            eao_list.append(r.eao)
            survival_list.append(r.survival_rate)

        mean_eao = float(np.mean(eao_list))
        mean_survival = float(np.mean(survival_list))
        # Combined robustness: geometric-mean of EAO and survival rate
        # Geometric mean penalises severe failure in either dimension.
        return float(math.sqrt(max(0.0, mean_eao) * max(0.0, mean_survival)))

    def _raw_smoothness(self, result: "BenchmarkResult") -> float:
        """Estimate mean temporal smoothness across all sequences.

        Returns 0.0 when no prediction arrays are stored (legacy results).
        """
        from .temporal import TemporalConsistencyAnalyzer

        analyzer = TemporalConsistencyAnalyzer()
        if not result.sequence_results:
            return 0.0

        scores: List[float] = []
        for sr in result.sequence_results:
            if sr.predictions is not None and len(sr.predictions) >= 2:
                tc = analyzer.analyze(sr.predictions)
                scores.append(tc.smoothness_score)

        return float(np.mean(scores)) if scores else 0.0

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def to_markdown_table(
        self,
        entries: List[DeployabilityEntry],
        title: str = "EOVOT Deployability Ranking",
    ) -> str:
        """Format a deployability ranking as a Markdown table.

        Args:
            entries: Output of :meth:`score_results`, already sorted by DS.
            title: Table heading string.

        Returns:
            Multi-line Markdown string ready for embedding in reports.

        Example output::

            ## EOVOT Deployability Ranking

            | Rank | Tracker | DS | Accuracy | Efficiency | Robustness | Smoothness | FPS | mIoU | Pareto |
            |------|---------|---:|---------:|-----------:|-----------:|-----------:|----:|-----:|:------:|
            | 1    | CSRT    | 0.812 | 0.950  | 0.721      | 0.888      | 0.689      | 32.1 | 0.71 | ✓ |
        """
        lines = [
            f"## {title}\n",
            (
                "| Rank | Tracker | Dataset | DS | Accuracy | Efficiency | "
                "Robustness | Smoothness | FPS | mIoU | Pareto |"
            ),
            (
                "|------|---------|---------|---:|---------:|-----------:|"
                "-----------:|-----------:|----:|-----:|:------:|"
            ),
        ]
        for rank, e in enumerate(entries, start=1):
            pareto = "✓" if e.on_pareto_front else ""
            lines.append(
                f"| {rank} | {e.tracker_name} | {e.dataset_name} "
                f"| {e.deployability_score:.4f} "
                f"| {e.accuracy_score:.3f} "
                f"| {e.efficiency_score:.3f} "
                f"| {e.robustness_score:.3f} "
                f"| {e.smoothness_score:.3f} "
                f"| {e.fps:.1f} "
                f"| {e.mean_iou:.4f} "
                f"| {pareto} |"
            )
        lines.append("")
        lines.append(
            f"*Weights: accuracy={self.weight_accuracy:.2f}  "
            f"efficiency={self.weight_efficiency:.2f}  "
            f"robustness={self.weight_robustness:.2f}  "
            f"smoothness={self.weight_smoothness:.2f}  "
            f"(normalised within comparison set)*"
        )
        return "\n".join(lines)

    def to_summary_dict(
        self, entries: List[DeployabilityEntry]
    ) -> List[Dict]:
        """Serialise all entries to a list of plain dicts (for JSON export).

        Args:
            entries: Output of :meth:`score_results`.

        Returns:
            List of dicts, one per tracker, suitable for ``json.dump``.
        """
        return [e.to_dict() for e in entries]

    def dimension_weights(self) -> Dict[str, float]:
        """Return the (normalised) weight for each dimension.

        Returns:
            Dict with keys ``"accuracy"``, ``"efficiency"``,
            ``"robustness"``, ``"smoothness"``.
        """
        return {
            "accuracy": self.weight_accuracy,
            "efficiency": self.weight_efficiency,
            "robustness": self.weight_robustness,
            "smoothness": self.weight_smoothness,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise(values: List[float]) -> List[float]:
    """Min-max normalise a list of floats to [0, 1].

    When all values are identical (zero range), returns a list of 1.0s so
    that no tracker is penalised for a degenerate comparison set.

    Args:
        values: Raw metric values.

    Returns:
        Normalised values in [0, 1].
    """
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float64)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return [1.0] * len(values)
    return list((arr - mn) / (mx - mn))

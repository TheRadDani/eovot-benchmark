"""Edge efficiency analysis for EOVOT benchmark results.

Provides tools for reasoning about the accuracy-efficiency trade-off that is
central to edge deployment of visual object trackers:

- **Pareto frontier**: trackers that are not dominated on any combination of
  accuracy (mIoU) and throughput (FPS).
- **Edge Fitness Score**: a single weighted scalar that combines all relevant
  metrics into a deployability score for a target device profile.
- **Hardware constraint filtering**: quickly identify which trackers can run
  within a device's latency, memory, and energy budget.
- **Leaderboard ranking**: rank trackers by fitness and format the result.

All methods accept standard EOVOT result dicts produced by
:meth:`~eovot.benchmark.engine.BenchmarkResult.to_dict`::

    {
        "summary": {
            "tracker":           str,
            "mean_iou":          float,
            "mean_fps":          float,
            "peak_memory_mb":    float,
            "mean_latency_ms":   float,  # optional
            "mean_energy_per_frame_mj": float,  # optional
        },
        "sequences": [...]
    }

Typical usage::

    from eovot.metrics.efficiency import EfficiencyAnalyzer

    analyzer = EfficiencyAnalyzer()

    # Which trackers are Pareto-optimal on IoU vs. FPS?
    frontier = analyzer.pareto_frontier(results)

    # Rank by weighted edge fitness (higher = more deployable)
    ranked = analyzer.rank_trackers(results)
    for score in ranked:
        print(score)

    # Filter to trackers that fit within a Raspberry Pi 4 budget
    feasible = analyzer.filter_by_constraints(
        results,
        max_latency_ms=33.0,   # ≤ 30 FPS minimum
        max_memory_mb=512.0,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EdgeFitnessScore:
    """Composite deployability score for a single tracker.

    Attributes:
        tracker_name:      Tracker identifier.
        fitness:           Weighted composite score in ``[0, 1]``.  Higher is
                           more deployable on the target edge device profile.
        is_pareto_optimal: ``True`` when this tracker is on the Pareto frontier
                           of accuracy vs. throughput among the compared set.
        violates_constraints: ``True`` when the tracker exceeds at least one
                              hardware constraint passed to
                              :meth:`EfficiencyAnalyzer.filter_by_constraints`.
        component_scores:  Normalised per-metric scores used in the weighted
                           sum (keys match the weight dict).
    """

    tracker_name: str
    fitness: float
    is_pareto_optimal: bool = False
    violates_constraints: bool = False
    component_scores: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        pareto_tag = " [Pareto]" if self.is_pareto_optimal else ""
        violates_tag = " [violates constraints]" if self.violates_constraints else ""
        return (
            f"EdgeFitnessScore[{self.tracker_name}] "
            f"fitness={self.fitness:.4f}{pareto_tag}{violates_tag}"
        )


class EfficiencyAnalyzer:
    """Analyse accuracy-efficiency trade-offs across a set of benchmark results.

    Args:
        accuracy_weight:  Weight for mIoU in the fitness score.
        fps_weight:       Weight for throughput (FPS).
        memory_weight:    Weight for memory efficiency (penalises high peak RAM).
        energy_weight:    Weight for energy efficiency (penalises high mJ/frame).
                          Only applied when energy data is present in results.

    The four weights are re-normalised to sum to 1.0 internally, so you can
    pass any positive values (e.g. ``accuracy_weight=2, fps_weight=1``).

    Example::

        analyzer = EfficiencyAnalyzer(accuracy_weight=0.5, fps_weight=0.3,
                                      memory_weight=0.2)
        scores = analyzer.rank_trackers(results)
    """

    def __init__(
        self,
        accuracy_weight: float = 0.40,
        fps_weight: float = 0.30,
        memory_weight: float = 0.20,
        energy_weight: float = 0.10,
    ) -> None:
        if any(w < 0 for w in (accuracy_weight, fps_weight, memory_weight, energy_weight)):
            raise ValueError("All weights must be non-negative.")
        total = accuracy_weight + fps_weight + memory_weight + energy_weight
        if total <= 0:
            raise ValueError("At least one weight must be positive.")
        self.accuracy_weight = accuracy_weight / total
        self.fps_weight = fps_weight / total
        self.memory_weight = memory_weight / total
        self.energy_weight = energy_weight / total

    # ------------------------------------------------------------------
    # Pareto frontier
    # ------------------------------------------------------------------

    def pareto_frontier(self, results: List[Dict[str, Any]]) -> List[str]:
        """Return tracker names that lie on the accuracy-FPS Pareto frontier.

        A tracker is Pareto-optimal when no other tracker in the set achieves
        *both* higher mIoU *and* higher FPS simultaneously.  These trackers
        represent the best accuracy-efficiency operating points — moving away
        from any Pareto-optimal tracker costs accuracy or speed.

        Args:
            results: List of EOVOT result dicts.

        Returns:
            List of tracker names on the frontier, sorted by FPS ascending.
        """
        summaries = [r.get("summary", {}) for r in results]
        names = [s.get("tracker", f"tracker_{i}") for i, s in enumerate(summaries)]
        ious = np.array([float(s.get("mean_iou", 0.0)) for s in summaries])
        fps = np.array([float(s.get("mean_fps", 0.0)) for s in summaries])

        n = len(names)
        on_frontier = []
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                # j dominates i if j is at least as good on both axes AND
                # strictly better on at least one.
                if ious[j] >= ious[i] and fps[j] >= fps[i]:
                    if ious[j] > ious[i] or fps[j] > fps[i]:
                        dominated = True
                        break
            if not dominated:
                on_frontier.append((fps[i], names[i]))

        on_frontier.sort(key=lambda x: x[0])
        return [name for _, name in on_frontier]

    # ------------------------------------------------------------------
    # Edge fitness score
    # ------------------------------------------------------------------

    def compute_fitness(
        self,
        result: Dict[str, Any],
        all_results: List[Dict[str, Any]],
    ) -> EdgeFitnessScore:
        """Compute the normalised edge fitness score for a single tracker.

        Each metric is min-max normalised over ``all_results`` so that the
        scores are always in ``[0, 1]`` and directly comparable.  Higher is
        better for all components:

        - ``accuracy``: mIoU normalised to ``[0, 1]``.
        - ``throughput``: FPS normalised to ``[0, 1]``.
        - ``memory_eff``: ``1 - (peak_mb / max_peak_mb)`` — penalises large RAM.
        - ``energy_eff``: ``1 - (mj / max_mj)`` — penalises high energy draw;
          omitted (weight redistributed) when energy data is unavailable.

        Args:
            result:      The result dict whose fitness to compute.
            all_results: Full set of results used for normalisation.

        Returns:
            :class:`EdgeFitnessScore` with fitness and per-component scores.
        """
        summaries = [r.get("summary", {}) for r in all_results]

        ious_all = np.array([float(s.get("mean_iou", 0.0)) for s in summaries])
        fps_all = np.array([float(s.get("mean_fps", 0.0)) for s in summaries])
        mem_all = np.array([float(s.get("peak_memory_mb", 0.0)) for s in summaries])

        has_energy = any(s.get("mean_energy_per_frame_mj") is not None for s in summaries)
        if has_energy:
            energy_all = np.array(
                [float(s.get("mean_energy_per_frame_mj", 0.0)) for s in summaries]
            )
        else:
            energy_all = None

        s = result.get("summary", {})
        tracker_name = s.get("tracker", "unknown")

        acc_score = _minmax_norm(float(s.get("mean_iou", 0.0)), ious_all)
        fps_score = _minmax_norm(float(s.get("mean_fps", 0.0)), fps_all)
        # Memory efficiency: lower memory → higher score
        mem_score = 1.0 - _minmax_norm(float(s.get("peak_memory_mb", 0.0)), mem_all)

        components: Dict[str, float] = {
            "accuracy": acc_score,
            "throughput": fps_score,
            "memory_efficiency": mem_score,
        }

        if has_energy and energy_all is not None:
            e_val = float(s.get("mean_energy_per_frame_mj", 0.0))
            e_score = 1.0 - _minmax_norm(e_val, energy_all)
            components["energy_efficiency"] = e_score
            # Include all four weights
            fitness = (
                self.accuracy_weight * acc_score
                + self.fps_weight * fps_score
                + self.memory_weight * mem_score
                + self.energy_weight * e_score
            )
        else:
            # Re-distribute the energy weight to accuracy and FPS proportionally
            total_w = self.accuracy_weight + self.fps_weight + self.memory_weight
            w_acc = self.accuracy_weight / total_w if total_w > 0 else 1 / 3
            w_fps = self.fps_weight / total_w if total_w > 0 else 1 / 3
            w_mem = self.memory_weight / total_w if total_w > 0 else 1 / 3
            fitness = w_acc * acc_score + w_fps * fps_score + w_mem * mem_score

        pareto_names = self.pareto_frontier(all_results)
        is_pareto = tracker_name in pareto_names

        return EdgeFitnessScore(
            tracker_name=tracker_name,
            fitness=round(float(fitness), 4),
            is_pareto_optimal=is_pareto,
            violates_constraints=False,
            component_scores={k: round(v, 4) for k, v in components.items()},
        )

    def rank_trackers(self, results: List[Dict[str, Any]]) -> List[EdgeFitnessScore]:
        """Rank all trackers by edge fitness (highest first).

        Args:
            results: List of EOVOT result dicts.

        Returns:
            List of :class:`EdgeFitnessScore` objects sorted by
            :attr:`~EdgeFitnessScore.fitness` descending.
        """
        scores = [self.compute_fitness(r, results) for r in results]
        scores.sort(key=lambda s: s.fitness, reverse=True)
        return scores

    # ------------------------------------------------------------------
    # Hardware constraint filtering
    # ------------------------------------------------------------------

    def filter_by_constraints(
        self,
        results: List[Dict[str, Any]],
        max_latency_ms: Optional[float] = None,
        max_memory_mb: Optional[float] = None,
        min_fps: Optional[float] = None,
        max_energy_mj: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Return only the results that satisfy all specified hardware constraints.

        Constraints are ANDed together; any ``None`` constraint is ignored.
        Typical device profiles:

        - Raspberry Pi 4: ``max_latency_ms=33, max_memory_mb=512``
        - Jetson Nano:    ``max_latency_ms=20, max_memory_mb=2048``
        - Desktop CPU:    ``min_fps=60``

        Args:
            results:        List of EOVOT result dicts.
            max_latency_ms: Maximum acceptable mean latency per frame.
            max_memory_mb:  Maximum acceptable peak memory usage.
            min_fps:        Minimum acceptable throughput.
            max_energy_mj:  Maximum acceptable mean energy per frame (mJ).

        Returns:
            Subset of ``results`` satisfying all constraints, preserving order.
        """
        feasible = []
        for r in results:
            s = r.get("summary", {})
            if max_latency_ms is not None:
                latency = float(s.get("mean_latency_ms", 0.0))
                if latency > max_latency_ms:
                    continue
            if max_memory_mb is not None:
                mem = float(s.get("peak_memory_mb", 0.0))
                if mem > max_memory_mb:
                    continue
            if min_fps is not None:
                fps = float(s.get("mean_fps", 0.0))
                if fps < min_fps:
                    continue
            if max_energy_mj is not None:
                energy = s.get("mean_energy_per_frame_mj")
                if energy is not None and float(energy) > max_energy_mj:
                    continue
            feasible.append(r)
        return feasible

    def score_with_constraints(
        self,
        results: List[Dict[str, Any]],
        max_latency_ms: Optional[float] = None,
        max_memory_mb: Optional[float] = None,
        min_fps: Optional[float] = None,
        max_energy_mj: Optional[float] = None,
    ) -> List[EdgeFitnessScore]:
        """Rank all trackers, flagging those that violate hardware constraints.

        Unlike :meth:`filter_by_constraints`, this method scores *all* trackers
        but marks violating ones via
        :attr:`~EdgeFitnessScore.violates_constraints`.  Useful for comparing
        performance across a heterogeneous device fleet.

        Args:
            results: List of EOVOT result dicts.
            max_latency_ms, max_memory_mb, min_fps, max_energy_mj:
                Same semantics as :meth:`filter_by_constraints`.

        Returns:
            All :class:`EdgeFitnessScore` objects sorted by fitness; violating
            trackers carry ``violates_constraints=True``.
        """
        feasible_set = {
            r.get("summary", {}).get("tracker", "")
            for r in self.filter_by_constraints(
                results, max_latency_ms, max_memory_mb, min_fps, max_energy_mj
            )
        }
        scores = self.rank_trackers(results)
        for sc in scores:
            sc.violates_constraints = sc.tracker_name not in feasible_set
        return scores


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _minmax_norm(value: float, arr: np.ndarray) -> float:
    """Min-max normalise *value* into ``[0, 1]`` using the range of *arr*.

    Returns 0.5 when all values in *arr* are identical (avoids division by zero).
    """
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return 0.5
    return float((value - lo) / (hi - lo))

"""Multi-objective tracker ranking with hypervolume indicator.

Extends the single-scalar :class:`~eovot.metrics.efficiency.EfficiencyMetricsEngine`
with a principled multi-objective framework that:

1. **Normalises** each metric (IoU, FPS, memory, energy) to [0, 1] across the
   tracker cohort so objectives are dimensionally comparable.
2. **Weights** normalised metrics with user-supplied importance coefficients and
   produces a *Weighted Composite Score* (WCS) per tracker.
3. **Computes the 2-D Hypervolume Indicator** (HV) for any pair of objectives,
   quantifying the area of the objective space dominated by the Pareto front.
   HV is the gold-standard scalar for Pareto front quality in the
   multi-objective optimisation literature.
4. **Ranks crowding distance** on the Pareto front so clusters of similar
   trackers are de-ranked, promoting diversity in deployment recommendations.

Why this matters for edge benchmarking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Edge deployment decisions rarely optimise a single metric.  A researcher
comparing MOSSE vs. KCF vs. CSRT faces a 3- or 4-dimensional trade-off:
higher IoU at the cost of more latency and memory.  WCS + HV give a
mathematically sound, reproducible way to collapse that trade-off into a
ranked list, while the Pareto front reveals the true frontier.

Typical usage::

    from eovot.metrics.multi_objective import MultiObjectiveRanker, ObjectiveWeights

    weights = ObjectiveWeights(iou=0.5, fps=0.3, memory=0.2)
    ranker  = MultiObjectiveRanker(weights)

    scores  = ranker.score_trackers(benchmark_results)
    hv      = ranker.hypervolume_2d(scores, obj_x="norm_iou", obj_y="norm_fps")

    print(ranker.to_markdown_table(scores))
    print(f"Pareto HV (IoU × FPS space): {hv:.4f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ObjectiveWeights:
    """Importance weights for each optimisation objective.

    All weights are non-negative; they are renormalised to sum to 1.0
    internally so the user does not need to pre-normalise them.

    Args:
        iou:    Weight for tracking accuracy (mean IoU or success AUC).
        fps:    Weight for throughput.
        memory: Weight for memory efficiency (lower is better → inverted
                internally before weighting).
        energy: Weight for energy efficiency (lower is better → inverted).
                Set to 0.0 (default) when energy profiling is unavailable.
    """

    iou: float = 0.5
    fps: float = 0.3
    memory: float = 0.2
    energy: float = 0.0

    def __post_init__(self) -> None:
        if any(w < 0 for w in (self.iou, self.fps, self.memory, self.energy)):
            raise ValueError("All objective weights must be non-negative.")
        total = self.iou + self.fps + self.memory + self.energy
        if total <= 0:
            raise ValueError("At least one objective weight must be positive.")
        # Renormalise to sum to 1.0.
        self.iou /= total
        self.fps /= total
        self.memory /= total
        self.energy /= total


# ---------------------------------------------------------------------------
# Per-tracker result
# ---------------------------------------------------------------------------


@dataclass
class MultiObjectiveScore:
    """Multi-objective evaluation summary for one tracker.

    All ``norm_*`` fields are in [0, 1] with *higher = better*
    (cost objectives such as memory and energy are inverted before
    normalisation so a score of 1.0 always means best in cohort).

    Attributes:
        tracker_name:       Human-readable identifier.
        dataset_name:       Dataset on which the tracker was evaluated.
        raw_iou:            Mean IoU (or success AUC) from the benchmark run.
        raw_fps:            Mean FPS from the benchmark run.
        raw_memory_mb:      Peak RSS memory footprint (MB).
        raw_energy_mj:      Mean per-frame energy (mJ), or ``None``.
        norm_iou:           Normalised accuracy score ∈ [0, 1].
        norm_fps:           Normalised throughput score ∈ [0, 1].
        norm_memory:        Normalised memory score ∈ [0, 1] (1 = least memory).
        norm_energy:        Normalised energy score ∈ [0, 1] (1 = least energy),
                            or ``None`` when energy data is absent.
        wcs:                Weighted Composite Score — dot product of normalised
                            objectives and weights.  Range: [0, 1].
        on_pareto_front:    ``True`` when no other tracker in this cohort
                            dominates both ``raw_iou`` and ``wcs``.
        crowding_distance:  Crowding distance on the (IoU, WCS) Pareto front.
                            ``inf`` for boundary Pareto-front members.
    """

    tracker_name: str
    dataset_name: str
    raw_iou: float
    raw_fps: float
    raw_memory_mb: float
    raw_energy_mj: Optional[float]
    norm_iou: float
    norm_fps: float
    norm_memory: float
    norm_energy: Optional[float]
    wcs: float
    on_pareto_front: bool = field(default=False)
    crowding_distance: float = field(default=0.0)

    def __str__(self) -> str:
        pareto = "Pareto" if self.on_pareto_front else "     -"
        energy_str = f"  E={self.raw_energy_mj:.2f}mJ" if self.raw_energy_mj is not None else ""
        return (
            f"MultiObjectiveScore({self.tracker_name}  WCS={self.wcs:.4f}  "
            f"mIoU={self.raw_iou:.4f}  FPS={self.raw_fps:.1f}  "
            f"mem={self.raw_memory_mb:.1f}MB{energy_str}  {pareto})"
        )


# ---------------------------------------------------------------------------
# Ranker
# ---------------------------------------------------------------------------


class MultiObjectiveRanker:
    """Rank a cohort of trackers using configurable objective weights.

    Args:
        weights: :class:`ObjectiveWeights` controlling the relative importance
            of IoU, FPS, memory, and (optionally) energy.  Weights are
            normalised to sum to 1.0.

    Example::

        ranker = MultiObjectiveRanker(ObjectiveWeights(iou=0.5, fps=0.3, memory=0.2))
        scores = ranker.score_trackers(benchmark_results)
        hv = ranker.hypervolume_2d(scores, obj_x="norm_iou", obj_y="norm_fps")
        print(ranker.to_markdown_table(scores))
    """

    def __init__(self, weights: Optional[ObjectiveWeights] = None) -> None:
        self.weights = weights or ObjectiveWeights()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def score_trackers(
        self, results: List["BenchmarkResult"]
    ) -> List[MultiObjectiveScore]:
        """Compute multi-objective scores for a collection of benchmark results.

        The pipeline is:
        1. Extract raw metric vectors from each result.
        2. Normalise each metric to [0, 1] across the cohort (min-max).
        3. Invert cost objectives (memory, energy) so higher = better.
        4. Compute WCS as a weighted dot product of normalised scores.
        5. Mark Pareto-optimal trackers (dominated in IoU × WCS space).
        6. Compute crowding distance on the Pareto front.

        Args:
            results: One :class:`~eovot.benchmark.engine.BenchmarkResult` per
                tracker (or tracker / dataset combination).

        Returns:
            List of :class:`MultiObjectiveScore`, sorted by WCS descending.
        """
        if not results:
            return []

        # -- 1. Extract raw metrics ------------------------------------------
        ious = np.array([r.mean_iou for r in results], dtype=np.float64)
        fpss = np.array([r.mean_fps for r in results], dtype=np.float64)
        mems = np.array([r.peak_memory_mb for r in results], dtype=np.float64)

        energies: Optional[np.ndarray] = None
        raw_energies = [r.mean_energy_per_frame_mj for r in results]
        if any(e is not None for e in raw_energies):
            energies = np.array(
                [e if e is not None else np.nan for e in raw_energies],
                dtype=np.float64,
            )

        # -- 2+3. Normalise, invert cost objectives -------------------------
        norm_iou = _minmax_normalize(ious)
        norm_fps = _minmax_normalize(fpss)
        norm_mem = 1.0 - _minmax_normalize(mems)  # lower memory → higher score

        norm_energy: Optional[np.ndarray] = None
        if energies is not None:
            # Replace NaN with cohort mean so missing energy doesn't bias.
            valid_mask = ~np.isnan(energies)
            filled = energies.copy()
            if valid_mask.any():
                filled[~valid_mask] = float(np.nanmean(energies))
            norm_energy = 1.0 - _minmax_normalize(filled)

        # -- 4. Weighted Composite Score ------------------------------------
        w = self.weights
        wcs = w.iou * norm_iou + w.fps * norm_fps + w.memory * norm_mem
        if norm_energy is not None:
            wcs += w.energy * norm_energy

        # -- 5+6. Assemble score objects ------------------------------------
        scores: List[MultiObjectiveScore] = []
        for i, r in enumerate(results):
            scores.append(
                MultiObjectiveScore(
                    tracker_name=r.tracker_name,
                    dataset_name=r.dataset_name,
                    raw_iou=float(ious[i]),
                    raw_fps=float(fpss[i]),
                    raw_memory_mb=float(mems[i]),
                    raw_energy_mj=raw_energies[i],
                    norm_iou=float(norm_iou[i]),
                    norm_fps=float(norm_fps[i]),
                    norm_memory=float(norm_mem[i]),
                    norm_energy=float(norm_energy[i]) if norm_energy is not None else None,
                    wcs=float(wcs[i]),
                )
            )

        self._mark_pareto_front(scores)
        self._compute_crowding_distance(scores)
        scores.sort(key=lambda s: s.wcs, reverse=True)
        return scores

    # ------------------------------------------------------------------
    # Hypervolume indicator
    # ------------------------------------------------------------------

    def hypervolume_2d(
        self,
        scores: List[MultiObjectiveScore],
        obj_x: str = "norm_iou",
        obj_y: str = "norm_fps",
        ref_point: Optional[Tuple[float, float]] = None,
    ) -> float:
        """Compute the 2-D Hypervolume Indicator for a pair of objectives.

        The hypervolume is the area of the region in objective space that is
        dominated by at least one point in the input set and bounded by the
        reference point.  Higher HV = better Pareto front quality.

        This implementation uses the sweep-line algorithm (O(n log n)),
        which is exact for 2-D fronts.

        Args:
            scores:    Output of :meth:`score_trackers`.
            obj_x:     Attribute name of the first objective (x-axis).
                Must be an attribute of :class:`MultiObjectiveScore`.
                Default: ``"norm_iou"``.
            obj_y:     Attribute name of the second objective (y-axis).
                Default: ``"norm_fps"``.
            ref_point: Reference (nadir) point ``(x_ref, y_ref)``.
                Must be dominated by all points on the Pareto front.
                Default: ``(0.0, 0.0)`` — valid when both objectives are
                normalised to [0, 1] with higher = better.

        Returns:
            Non-negative float: the hypervolume in units of the objective
            space.  Returns 0.0 for an empty input.
        """
        if not scores:
            return 0.0
        if ref_point is None:
            ref_point = (0.0, 0.0)

        points = np.array(
            [(getattr(s, obj_x), getattr(s, obj_y)) for s in scores],
            dtype=np.float64,
        )
        return float(_hypervolume_2d_sweep(points, ref_point))

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def sensitivity_analysis(
        self,
        results: List["BenchmarkResult"],
        weight_grid_steps: int = 5,
    ) -> Dict[str, List[Tuple[ObjectiveWeights, int]]]:
        """Analyse how tracker rankings change with varying IoU/FPS weights.

        Sweeps IoU weight from 0 to 1 (and FPS = 1 - IoU, memory = 0)
        in ``weight_grid_steps`` steps, computing the rank of each tracker
        at each weight combination.

        Args:
            results:          Benchmark results for all trackers.
            weight_grid_steps: Number of evenly-spaced IoU-weight values.

        Returns:
            Dict mapping tracker name → list of ``(weights, rank)`` tuples,
            one per weight setting evaluated.
        """
        tracker_ranks: Dict[str, List[Tuple[ObjectiveWeights, int]]] = {
            r.tracker_name: [] for r in results
        }
        for alpha in np.linspace(0.0, 1.0, weight_grid_steps):
            w = ObjectiveWeights(iou=float(alpha), fps=float(1.0 - alpha), memory=0.0)
            ranker = MultiObjectiveRanker(w)
            scores = ranker.score_trackers(results)
            for rank, score in enumerate(scores, start=1):
                tracker_ranks[score.tracker_name].append((w, rank))
        return tracker_ranks

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def to_markdown_table(self, scores: List[MultiObjectiveScore]) -> str:
        """Format ranked multi-objective scores as a Markdown table.

        Args:
            scores: Output of :meth:`score_trackers` (already sorted by WCS).

        Returns:
            Multi-line Markdown table string with rank, tracker, raw metrics,
            normalised scores, WCS, and Pareto membership.
        """
        has_energy = any(s.raw_energy_mj is not None for s in scores)
        header_cols = (
            "| Rank | Tracker | Dataset | mIoU | FPS | Mem (MB)"
            + ("| Energy (mJ/fr)" if has_energy else "")
            + " | WCS | Pareto |"
        )
        sep_cols = (
            "|------|---------|---------|-----:|----:|---------:"
            + ("|--------------:" if has_energy else "")
            + "|----:|:------:|"
        )
        lines = [header_cols, sep_cols]
        for rank, s in enumerate(scores, start=1):
            pareto = "✓" if s.on_pareto_front else ""
            energy_cell = (
                f"| {s.raw_energy_mj:.2f} " if has_energy else ""
            )
            lines.append(
                f"| {rank} | {s.tracker_name} | {s.dataset_name} "
                f"| {s.raw_iou:.4f} | {s.raw_fps:.1f} | {s.raw_memory_mb:.1f} "
                f"{energy_cell}| {s.wcs:.4f} | {pareto} |"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mark_pareto_front(scores: List[MultiObjectiveScore]) -> None:
        """Mark Pareto-optimal entries in (raw_iou, wcs) space — in-place."""
        for i, cand in enumerate(scores):
            dominated = False
            for j, other in enumerate(scores):
                if i == j:
                    continue
                if (
                    other.raw_iou >= cand.raw_iou
                    and other.wcs >= cand.wcs
                    and (other.raw_iou > cand.raw_iou or other.wcs > cand.wcs)
                ):
                    dominated = True
                    break
            cand.on_pareto_front = not dominated

    @staticmethod
    def _compute_crowding_distance(scores: List[MultiObjectiveScore]) -> None:
        """Assign crowding distance on the Pareto front — in-place.

        Crowding distance measures how isolated a Pareto-front point is from
        its neighbours.  Boundary points receive ``inf``, interior points
        receive the sum of normalised gaps in each objective.  Higher
        crowding distance = better spread = preferred when ranks are tied.
        """
        pareto = [s for s in scores if s.on_pareto_front]
        if len(pareto) < 2:
            for s in pareto:
                s.crowding_distance = float("inf")
            return

        for obj in ("raw_iou", "wcs"):
            vals = np.array([getattr(s, obj) for s in pareto])
            order = np.argsort(vals)
            span = vals[order[-1]] - vals[order[0]]

            pareto[order[0]].crowding_distance = float("inf")
            pareto[order[-1]].crowding_distance = float("inf")

            if span <= 0:
                continue
            for k in range(1, len(order) - 1):
                gap = (vals[order[k + 1]] - vals[order[k - 1]]) / span
                if pareto[order[k]].crowding_distance != float("inf"):
                    pareto[order[k]].crowding_distance += gap
                else:
                    pareto[order[k]].crowding_distance = gap


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def _minmax_normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]; returns 0.5 for constant arrays."""
    lo, hi = arr.min(), arr.max()
    span = hi - lo
    if span < 1e-12:
        return np.full_like(arr, 0.5, dtype=np.float64)
    return (arr - lo) / span


def _hypervolume_2d_sweep(
    points: np.ndarray, ref: Tuple[float, float]
) -> float:
    """Exact 2-D hypervolume via a one-pass sweep-line algorithm.

    Computes the area dominated by *points* (shape ``(N, 2)``) relative to
    the reference (nadir) point *ref*.  Points that do not dominate *ref*
    are filtered out before the sweep.

    Algorithm: sort points by x descending, sweep from right to left
    accumulating rectangular contributions in y.

    Args:
        points: ``(N, 2)`` array of (x, y) objective values (higher = better).
        ref:    Reference (nadir) point; must be dominated by at least one
                point in *points* for a non-zero result.

    Returns:
        Hypervolume as a non-negative float.
    """
    rx, ry = ref
    # Keep only points that dominate the reference.
    mask = (points[:, 0] > rx) & (points[:, 1] > ry)
    pts = points[mask]
    if len(pts) == 0:
        return 0.0

    # Build the non-dominated front (keep points with maximal y for any x).
    pts = pts[np.argsort(pts[:, 0])[::-1]]   # sort by x descending
    front: List[Tuple[float, float]] = []
    best_y = float("-inf")
    for x, y in pts:
        if y > best_y:
            front.append((x, y))
            best_y = y

    # Sweep right to left: each point contributes a rectangle.
    hv = 0.0
    prev_x = float("-inf")
    for i, (x, y) in enumerate(front):
        if i == len(front) - 1:
            width = x - rx
        else:
            width = x - front[i + 1][0]
        height = y - ry
        if width > 0 and height > 0:
            hv += width * height
        prev_x = x

    return hv

"""Edge efficiency scoring and Pareto trade-off analysis for EOVOT.

This module implements the core research contribution of EOVOT: evaluating
trackers not only by accuracy but by their *deployment feasibility* on
edge hardware.  Three hardware dimensions are combined into a single
composite efficiency score:

* **FPS score** — what fraction of the target frame rate is achieved
  (e.g. target = 30 FPS for real-time; score = 1.0 if FPS ≥ target)
* **Memory score** — how far the tracker is from the memory budget
  (score = 1 − mem/max_mem; zero when budget is exceeded)
* **Energy score** — fraction of the energy budget remaining per frame
  (only included when energy profiling was enabled)

The composite score is a weighted combination of the three, normalised to
``[0, 1]``.  Weights default to 0.5 / 0.3 / 0.2 (FPS / memory / energy),
reflecting that real-time throughput is the most critical constraint for
edge deployment, followed by memory footprint.

**Pareto frontier analysis** identifies the set of trackers that are not
dominated in both accuracy (mIoU) and efficiency — i.e. the optimal
operating points for edge deployment.

Typical usage::

    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.metrics.efficiency import EdgeEfficiencyAnalyzer

    engine = BenchmarkEngine(tdp_watts=10.0)   # Jetson Nano TDP
    results = [engine.run(t, dataset) for t in trackers]

    analyzer = EdgeEfficiencyAnalyzer(target_fps=30.0, max_memory_mb=512.0)
    scores = analyzer.analyze(results)
    frontier = analyzer.pareto_frontier(scores)

    print(analyzer.ranking_table(scores))
    for s in frontier:
        print(f"  Pareto-optimal: {s.tracker_name}  mIoU={s.mean_iou:.3f}  eff={s.composite_score:.3f}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class EdgeEfficiencyScore:
    """Multi-dimensional edge deployment score for a single tracker.

    All score components are in ``[0, 1]`` where 1 is optimal.

    Attributes:
        tracker_name:       Identifier of the tracker.
        fps_score:          Fraction of target FPS achieved (capped at 1.0).
        memory_score:       Memory efficiency: ``1 − peak_mem / max_mem``.
        energy_score:       Energy efficiency: ``1 − energy_per_frame / max_energy``.
                            ``0.0`` when energy profiling was not enabled.
        composite_score:    Weighted combination of the above, normalised to
                            ``[0, 1]``.
        mean_iou:           Tracker accuracy (mean IoU across sequences).
        mean_fps:           Raw mean throughput in frames per second.
        peak_memory_mb:     Raw peak memory footprint in MiB.
        energy_per_frame_mj: Raw mean energy per frame in milli-Joules, or
                            ``None`` if energy profiling was not enabled.
        has_energy:         ``True`` when energy data was available.
    """

    tracker_name: str
    fps_score: float
    memory_score: float
    energy_score: float
    composite_score: float
    mean_iou: float
    mean_fps: float
    peak_memory_mb: float
    energy_per_frame_mj: Optional[float]
    has_energy: bool

    def __str__(self) -> str:
        energy_str = (
            f"  energy={self.energy_per_frame_mj:.2f} mJ/fr"
            if self.has_energy and self.energy_per_frame_mj is not None
            else ""
        )
        return (
            f"EdgeEfficiencyScore[{self.tracker_name}] "
            f"mIoU={self.mean_iou:.4f}  eff={self.composite_score:.4f}  "
            f"fps={self.mean_fps:.1f}  mem={self.peak_memory_mb:.1f} MB"
            f"{energy_str}"
        )

    def to_dict(self) -> Dict:
        d: Dict = {
            "tracker": self.tracker_name,
            "mean_iou": round(self.mean_iou, 4),
            "composite_score": round(self.composite_score, 4),
            "fps_score": round(self.fps_score, 4),
            "memory_score": round(self.memory_score, 4),
            "mean_fps": round(self.mean_fps, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
        }
        if self.has_energy and self.energy_per_frame_mj is not None:
            d["energy_score"] = round(self.energy_score, 4)
            d["energy_per_frame_mj"] = round(self.energy_per_frame_mj, 4)
        return d


class EdgeEfficiencyAnalyzer:
    """Compute edge deployment efficiency scores and identify Pareto-optimal trackers.

    Hardware constraints are expressed as targets/budgets:

    * ``target_fps`` — desired minimum throughput for the application
      (e.g. 30 FPS for real-time video; 10 FPS for robotics).
    * ``max_memory_mb`` — total available RAM / VRAM on the edge device
      (e.g. 512 MB for Jetson Nano, 4096 MB for Raspberry Pi 4).
    * ``max_energy_mj`` — energy budget per frame in milli-Joules; only
      relevant when energy profiling was enabled (``tdp_watts`` set in
      :class:`~eovot.benchmark.engine.BenchmarkEngine`).

    Args:
        target_fps:    Target throughput in frames per second.  Default: ``30.0``.
        max_memory_mb: Memory budget in MiB.  Default: ``512.0``.
        max_energy_mj: Energy budget per frame in milli-Joules.  ``None`` disables
            energy from the composite score.  Default: ``None``.
        fps_weight:    Weight of FPS in the composite score.  Default: ``0.5``.
        memory_weight: Weight of memory efficiency.  Default: ``0.3``.
        energy_weight: Weight of energy efficiency (ignored when energy data absent).
            Default: ``0.2``.

    Example::

        analyzer = EdgeEfficiencyAnalyzer(
            target_fps=30.0,
            max_memory_mb=512.0,
            max_energy_mj=5.0,      # Jetson Nano budget
            fps_weight=0.5,
            memory_weight=0.3,
            energy_weight=0.2,
        )

        scores = analyzer.analyze([mosse_result, kcf_result, csrt_result])
        frontier = analyzer.pareto_frontier(scores)
        print(analyzer.ranking_table(scores))
    """

    def __init__(
        self,
        target_fps: float = 30.0,
        max_memory_mb: float = 512.0,
        max_energy_mj: Optional[float] = None,
        fps_weight: float = 0.5,
        memory_weight: float = 0.3,
        energy_weight: float = 0.2,
    ) -> None:
        if target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {target_fps}")
        if max_memory_mb <= 0:
            raise ValueError(f"max_memory_mb must be positive, got {max_memory_mb}")
        if max_energy_mj is not None and max_energy_mj <= 0:
            raise ValueError(f"max_energy_mj must be positive, got {max_energy_mj}")
        if not (fps_weight >= 0 and memory_weight >= 0 and energy_weight >= 0):
            raise ValueError("All weights must be non-negative")
        if fps_weight + memory_weight == 0:
            raise ValueError("At least fps_weight or memory_weight must be non-zero")

        self.target_fps = target_fps
        self.max_memory_mb = max_memory_mb
        self.max_energy_mj = max_energy_mj
        self.fps_weight = fps_weight
        self.memory_weight = memory_weight
        self.energy_weight = energy_weight

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, result) -> EdgeEfficiencyScore:
        """Compute the edge efficiency score for a single tracker result.

        Args:
            result: :class:`~eovot.benchmark.engine.BenchmarkResult` from
                :meth:`~eovot.benchmark.engine.BenchmarkEngine.run`.

        Returns:
            :class:`EdgeEfficiencyScore` with all component scores populated.
        """
        fps = result.mean_fps
        mem = result.peak_memory_mb
        energy_mj = result.mean_energy_per_frame_mj

        fps_score = min(fps / self.target_fps, 1.0)
        memory_score = max(0.0, 1.0 - mem / self.max_memory_mb)

        has_energy = (
            energy_mj is not None
            and self.max_energy_mj is not None
        )
        if has_energy:
            energy_score = max(0.0, 1.0 - energy_mj / self.max_energy_mj)
            total_weight = self.fps_weight + self.memory_weight + self.energy_weight
            composite = (
                fps_score * self.fps_weight
                + memory_score * self.memory_weight
                + energy_score * self.energy_weight
            ) / total_weight
        else:
            energy_score = 0.0
            total_weight = self.fps_weight + self.memory_weight
            composite = (
                fps_score * self.fps_weight
                + memory_score * self.memory_weight
            ) / total_weight

        return EdgeEfficiencyScore(
            tracker_name=result.tracker_name,
            fps_score=fps_score,
            memory_score=memory_score,
            energy_score=energy_score,
            composite_score=composite,
            mean_iou=result.mean_iou,
            mean_fps=fps,
            peak_memory_mb=mem,
            energy_per_frame_mj=energy_mj,
            has_energy=has_energy,
        )

    def analyze(self, results: list) -> List[EdgeEfficiencyScore]:
        """Compute efficiency scores for a list of benchmark results.

        Args:
            results: List of :class:`~eovot.benchmark.engine.BenchmarkResult`
                objects, one per tracker.

        Returns:
            List of :class:`EdgeEfficiencyScore` in the same order as input.
        """
        return [self.score(r) for r in results]

    # ------------------------------------------------------------------
    # Pareto frontier
    # ------------------------------------------------------------------

    def pareto_frontier(
        self, scores: List[EdgeEfficiencyScore]
    ) -> List[EdgeEfficiencyScore]:
        """Find the Pareto-optimal trackers in accuracy–efficiency space.

        A tracker A *dominates* tracker B when A is at least as good in both
        accuracy (mIoU) and efficiency (composite score), and strictly better
        in at least one dimension.  The Pareto frontier is the set of all
        non-dominated trackers — the optimal operating points for deployment.

        Args:
            scores: List of :class:`EdgeEfficiencyScore` objects.

        Returns:
            Subset of ``scores`` containing only the non-dominated trackers,
            sorted by mIoU descending.
        """
        n = len(scores)
        dominated = [False] * n

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                iou_i, eff_i = scores[i].mean_iou, scores[i].composite_score
                iou_j, eff_j = scores[j].mean_iou, scores[j].composite_score
                if (
                    iou_j >= iou_i
                    and eff_j >= eff_i
                    and (iou_j > iou_i or eff_j > eff_i)
                ):
                    dominated[i] = True
                    break

        frontier = [s for s, d in zip(scores, dominated) if not d]
        return sorted(frontier, key=lambda s: s.mean_iou, reverse=True)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def ranking_table(self, scores: List[EdgeEfficiencyScore]) -> str:
        """Format scores as a Markdown table ranked by composite efficiency.

        Trackers are sorted by composite efficiency score (descending).
        mIoU is included so the accuracy–efficiency trade-off is visible.

        Args:
            scores: List of :class:`EdgeEfficiencyScore` objects.

        Returns:
            Multi-line Markdown string suitable for README or paper.
        """
        ranked = sorted(scores, key=lambda s: s.composite_score, reverse=True)

        has_energy = any(s.has_energy for s in scores)
        if has_energy:
            header = (
                "| Rank | Tracker | mIoU | Efficiency | FPS Score | "
                "Mem Score | Energy Score | FPS | Mem (MB) |\n"
                "|------|---------|-----:|-----------:|----------:|"
                "----------:|-------------:|----:|---------:|"
            )
            rows = []
            for rank, s in enumerate(ranked, 1):
                rows.append(
                    f"| {rank} | {s.tracker_name} | {s.mean_iou:.4f} "
                    f"| {s.composite_score:.4f} | {s.fps_score:.3f} "
                    f"| {s.memory_score:.3f} | {s.energy_score:.3f} "
                    f"| {s.mean_fps:.1f} | {s.peak_memory_mb:.1f} |"
                )
        else:
            header = (
                "| Rank | Tracker | mIoU | Efficiency | FPS Score | "
                "Mem Score | FPS | Mem (MB) |\n"
                "|------|---------|-----:|-----------:|----------:|"
                "----------:|----:|---------:|"
            )
            rows = []
            for rank, s in enumerate(ranked, 1):
                rows.append(
                    f"| {rank} | {s.tracker_name} | {s.mean_iou:.4f} "
                    f"| {s.composite_score:.4f} | {s.fps_score:.3f} "
                    f"| {s.memory_score:.3f} "
                    f"| {s.mean_fps:.1f} | {s.peak_memory_mb:.1f} |"
                )

        return header + "\n" + "\n".join(rows)

    def to_dataframe(self, scores: List[EdgeEfficiencyScore]):
        """Convert scores to a pandas DataFrame for further analysis.

        Args:
            scores: List of :class:`EdgeEfficiencyScore` objects.

        Returns:
            ``pandas.DataFrame`` with one row per tracker, sorted by
            composite efficiency score descending.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install pandas"
            ) from exc

        rows = [s.to_dict() for s in scores]
        df = pd.DataFrame(rows)
        return df.sort_values("composite_score", ascending=False).reset_index(drop=True)

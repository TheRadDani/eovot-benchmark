"""Pareto frontier analysis for tracker accuracy–efficiency tradeoff.

Identifies which trackers are Pareto-optimal — no other tracker is
simultaneously more accurate *and* faster — and provides a composite
score based on the normalised geometric mean of both axes.

Typical usage::

    from eovot.analysis.pareto import ParetoAnalyzer

    analyzer = ParetoAnalyzer()
    points = analyzer.analyze(
        {"MOSSE": mosse_result, "KCF": kcf_result, "CSRT": csrt_result}
    )
    print(analyzer.to_markdown(points))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class ParetoPoint:
    """A tracker mapped to its accuracy and efficiency values.

    Attributes:
        tracker:    Tracker identifier.
        accuracy:   Accuracy metric value (higher is better).
        efficiency: Efficiency metric value (higher is better, e.g. FPS).
        is_pareto:  ``True`` when no other tracker dominates this one on
                    *both* axes simultaneously.
        score:      Composite score in ``[0, 1]`` — geometric mean of the
                    values normalised by the best tracker on each axis.
    """

    tracker: str
    accuracy: float
    efficiency: float
    is_pareto: bool
    score: float = field(default=0.0)


class ParetoAnalyzer:
    """Identify and describe the Pareto frontier of accuracy vs efficiency.

    Given a collection of tracker result dicts the analyser determines
    which trackers are Pareto-optimal and assigns a composite score.  The
    accuracy–efficiency frontier is useful for selecting the right tracker
    for a given deployment constraint (e.g. "best accuracy under 30 ms/fr").

    Example::

        from eovot.analysis.pareto import ParetoAnalyzer

        analyzer = ParetoAnalyzer()
        results = {
            "MOSSE": mosse_result_dict,
            "KCF":   kcf_result_dict,
            "CSRT":  csrt_result_dict,
        }
        points = analyzer.analyze(results)
        print(analyzer.to_markdown(points))
    """

    def analyze(
        self,
        results: Dict[str, Dict],
        accuracy_metric: str = "mean_iou",
        efficiency_metric: str = "mean_fps",
    ) -> List[ParetoPoint]:
        """Compute Pareto-optimality for a collection of tracker results.

        Args:
            results:           Mapping of tracker name → result dict from
                               ``BenchmarkResult.to_dict()``.
            accuracy_metric:   Key inside ``"summary"`` for the accuracy axis.
                               Default: ``"mean_iou"``.
            efficiency_metric: Key inside ``"summary"`` for the efficiency
                               axis.  Default: ``"mean_fps"``.

        Returns:
            List of :class:`ParetoPoint` sorted by accuracy (descending).
        """
        if not results:
            return []

        points: List[ParetoPoint] = []
        for name, result in results.items():
            summary = result.get("summary", result)
            acc = float(summary.get(accuracy_metric, 0.0))
            eff = float(summary.get(efficiency_metric, 0.0))
            points.append(ParetoPoint(name, acc, eff, is_pareto=False))

        # Domination: q dominates p if q >= p on both axes with strict > on one.
        dominated: set = set()
        for i, p in enumerate(points):
            for j, q in enumerate(points):
                if i == j:
                    continue
                if q.accuracy >= p.accuracy and q.efficiency >= p.efficiency:
                    if q.accuracy > p.accuracy or q.efficiency > p.efficiency:
                        dominated.add(i)
                        break

        max_acc = max(p.accuracy for p in points) or 1.0
        max_eff = max(p.efficiency for p in points) or 1.0

        for i, p in enumerate(points):
            p.is_pareto = i not in dominated
            norm_acc = p.accuracy / max_acc
            norm_eff = p.efficiency / max_eff
            p.score = round(math.sqrt(norm_acc * norm_eff), 4)

        return sorted(points, key=lambda p: p.accuracy, reverse=True)

    def pareto_scores(self, results: Dict[str, Dict]) -> Dict[str, float]:
        """Composite Pareto score for each tracker in ``[0, 1]``.

        The score is the geometric mean of accuracy and efficiency
        normalised by the best tracker on each axis.  Higher is better.

        Args:
            results: Mapping of tracker name → result dict.

        Returns:
            Dict mapping tracker name → score.
        """
        return {p.tracker: p.score for p in self.analyze(results)}

    def efficiency_frontier(
        self,
        results: Dict[str, Dict],
        accuracy_metric: str = "mean_iou",
        efficiency_metric: str = "mean_fps",
    ) -> List[Tuple[float, float, str]]:
        """Return the Pareto frontier as ``(efficiency, accuracy, name)`` triples.

        Sorted by efficiency (ascending) for easy plotting of the
        accuracy–efficiency curve.

        Args:
            results:           Mapping of tracker name → result dict.
            accuracy_metric:   Accuracy axis key.
            efficiency_metric: Efficiency axis key.

        Returns:
            List of ``(efficiency, accuracy, tracker_name)`` triples for
            Pareto-optimal points only.
        """
        points = self.analyze(results, accuracy_metric, efficiency_metric)
        frontier = [
            (p.efficiency, p.accuracy, p.tracker)
            for p in points
            if p.is_pareto
        ]
        return sorted(frontier, key=lambda t: t[0])

    def to_markdown(self, points: List[ParetoPoint]) -> str:
        """Render the Pareto analysis as a Markdown table.

        Args:
            points: Output of :meth:`analyze`.

        Returns:
            Multi-line Markdown string.
        """
        lines = [
            "| Rank | Tracker | Accuracy | Efficiency (FPS) | Score | Pareto |",
            "|-----:|---------|----------|-----------------:|------:|:------:|",
        ]
        for rank, p in enumerate(points, start=1):
            flag = "✓" if p.is_pareto else ""
            lines.append(
                f"| {rank} | {p.tracker} | {p.accuracy:.4f} "
                f"| {p.efficiency:.1f} | {p.score:.4f} | {flag} |"
            )
        return "\n".join(lines)

"""Pareto frontier analysis for speed-accuracy tradeoff in edge tracking.

In edge deployment, no single metric tells the full story: a tracker with
exceptional accuracy that runs at 2 FPS is unusable on a 30-FPS camera, and
a tracker that runs at 500 FPS but tracks nothing is equally worthless.  The
Pareto frontier is the formal tool for navigating this tradeoff — it identifies
the subset of trackers where no competitor is strictly better in *both*
dimensions simultaneously.

:class:`ParetoFrontierAnalyzer` accepts a list of benchmark results, extracts
a configurable accuracy metric (mIoU, success AUC, EAO) and throughput (FPS),
and computes:

- **Pareto-optimal trackers** — the efficient frontier.
- **Edge Deployment Score (EDS)** — harmonic mean of normalised FPS and
  normalised accuracy; a single scalar that balances both objectives without
  choosing an arbitrary trade-off weight.
- **Dominated trackers** — those strictly worse than at least one Pareto point.
- A ranked summary table in Markdown or CSV.

Typical usage::

    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kcf import KCFTracker
    from eovot.analysis.pareto import ParetoFrontierAnalyzer

    ds      = SyntheticDataset(num_sequences=10)
    engine  = BenchmarkEngine(verbose=False)
    results = [
        engine.run(MOSSETracker(), ds, dataset_name="Synthetic"),
        engine.run(KCFTracker(),   ds, dataset_name="Synthetic"),
    ]

    analyzer = ParetoFrontierAnalyzer(accuracy_metric="success_auc")
    report   = analyzer.analyze(results)

    print(report.to_markdown())
    report.to_csv("pareto.csv")

    for tp in report.pareto_points:
        print(f"{tp.tracker_name}: FPS={tp.fps:.1f} acc={tp.accuracy:.4f} EDS={tp.eds:.4f}")
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TrackerPoint:
    """One tracker's position in speed-accuracy space."""

    tracker_name: str
    fps: float
    accuracy: float
    eds: float
    """Edge Deployment Score — harmonic mean of normalised FPS and normalised accuracy.

    EDS ∈ [0, 1]; higher is better.  The normalisation is relative to the
    range observed in the input results, so EDS is meaningful only within one
    :class:`ParetoFrontierAnalyzer` call.
    """
    is_pareto_optimal: bool
    raw_summary: Dict[str, Any]


@dataclass
class ParetoReport:
    """Full Pareto analysis result."""

    accuracy_metric: str
    all_points: List[TrackerPoint]

    @property
    def pareto_points(self) -> List[TrackerPoint]:
        """Trackers on the Pareto frontier, sorted by FPS ascending."""
        pts = [p for p in self.all_points if p.is_pareto_optimal]
        return sorted(pts, key=lambda p: p.fps)

    @property
    def dominated_points(self) -> List[TrackerPoint]:
        """Trackers dominated by at least one Pareto-optimal tracker."""
        return [p for p in self.all_points if not p.is_pareto_optimal]

    def best_by_eds(self) -> Optional[TrackerPoint]:
        """Return the tracker with the highest Edge Deployment Score."""
        if not self.all_points:
            return None
        return max(self.all_points, key=lambda p: p.eds)

    def optimal_for_fps_budget(self, min_fps: float) -> Optional[TrackerPoint]:
        """Return the most accurate Pareto-optimal tracker that meets *min_fps*.

        Args:
            min_fps: Minimum required throughput in frames-per-second.

        Returns:
            The Pareto-optimal tracker with the highest accuracy among those
            that satisfy ``fps >= min_fps``, or ``None`` if none qualify.
        """
        candidates = [p for p in self.pareto_points if p.fps >= min_fps]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.accuracy)

    def to_markdown(self, title: str = "Pareto Frontier Analysis") -> str:
        """Render a Markdown table of all trackers ranked by EDS.

        Pareto-optimal rows are annotated with a ✓ marker in the *Pareto*
        column so the frontier is immediately visible.

        Args:
            title: Heading printed above the table.

        Returns:
            Multi-line Markdown string.
        """
        lines: List[str] = [f"## {title}", ""]
        acc_label = _metric_display_name(self.accuracy_metric)
        lines.append(
            f"| Rank | Tracker | FPS | {acc_label} | EDS | Pareto |"
        )
        lines.append("|---:|:---|---:|---:|---:|:---:|")

        sorted_pts = sorted(self.all_points, key=lambda p: p.eds, reverse=True)
        for rank, pt in enumerate(sorted_pts, start=1):
            pareto_mark = "✓" if pt.is_pareto_optimal else ""
            lines.append(
                f"| {rank} | {pt.tracker_name} "
                f"| {pt.fps:.1f} "
                f"| {pt.accuracy:.4f} "
                f"| {pt.eds:.4f} "
                f"| {pareto_mark} |"
            )

        lines.append("")
        lines.append(
            f"*EDS = Edge Deployment Score (harmonic mean of normalised FPS and "
            f"normalised {acc_label}).  ✓ = Pareto-optimal.*"
        )
        return "\n".join(lines)

    def to_csv(self, path: Union[str, Path]) -> Path:
        """Write the full analysis to a CSV file.

        Args:
            path: Destination path.  A ``.csv`` extension is appended if absent.

        Returns:
            Resolved :class:`~pathlib.Path` of the written file.
        """
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".csv")
        p.parent.mkdir(parents=True, exist_ok=True)

        acc_label = _metric_display_name(self.accuracy_metric)
        sorted_pts = sorted(self.all_points, key=lambda p: p.eds, reverse=True)

        with open(p, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["rank", "tracker", "fps", acc_label, "eds", "pareto_optimal"])
            for rank, pt in enumerate(sorted_pts, start=1):
                writer.writerow(
                    [rank, pt.tracker_name, f"{pt.fps:.3f}",
                     f"{pt.accuracy:.6f}", f"{pt.eds:.6f}",
                     str(pt.is_pareto_optimal)]
                )
        return p


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class ParetoFrontierAnalyzer:
    """Identify the Pareto frontier in speed-accuracy space for a set of trackers.

    Args:
        accuracy_metric: Which accuracy scalar to use as the y-axis.
            Supported: ``"mean_iou"``, ``"success_auc"``, ``"mean_eao"``.
            Default: ``"success_auc"``.
        fps_key: Key in the result summary dict for throughput.
            Default: ``"mean_fps"``.

    Example::

        analyzer = ParetoFrontierAnalyzer(accuracy_metric="mean_eao")
        report   = analyzer.analyze(results)
        best     = report.optimal_for_fps_budget(min_fps=30.0)
        print(f"Best tracker at ≥30 FPS: {best.tracker_name} EAO={best.accuracy:.4f}")
    """

    def __init__(
        self,
        accuracy_metric: str = "success_auc",
        fps_key: str = "mean_fps",
    ) -> None:
        self.accuracy_metric = accuracy_metric
        self.fps_key = fps_key

    def analyze(
        self,
        results: Sequence[Union[Any, Dict[str, Any]]],
    ) -> ParetoReport:
        """Compute the Pareto frontier and Edge Deployment Scores.

        Args:
            results: Sequence of :class:`~eovot.benchmark.engine.BenchmarkResult`
                objects or their ``to_dict()`` / ``summary()`` dicts.  At least
                two results are needed for a meaningful comparison, though a
                single-tracker call is handled gracefully.

        Returns:
            :class:`ParetoReport` containing all tracker points and the frontier.

        Raises:
            ValueError: If *results* is empty.
        """
        if not results:
            raise ValueError("ParetoFrontierAnalyzer.analyze: results list is empty.")

        summaries = [self._extract_summary(r) for r in results]
        raw_fps = np.array(
            [float(s.get(self.fps_key, 0.0)) for s in summaries], dtype=np.float64
        )
        raw_acc = np.array(
            [float(s.get(self.accuracy_metric, 0.0)) for s in summaries], dtype=np.float64
        )
        tracker_names = [
            s.get("tracker") or s.get("tracker_name", f"tracker_{i}")
            for i, s in enumerate(summaries)
        ]

        norm_fps = _normalize(raw_fps)
        norm_acc = _normalize(raw_acc)
        eds_arr = _harmonic_mean_pair(norm_fps, norm_acc)

        pareto_mask = _pareto_frontier_mask(raw_fps, raw_acc)

        points: List[TrackerPoint] = []
        for i, name in enumerate(tracker_names):
            points.append(
                TrackerPoint(
                    tracker_name=name,
                    fps=float(raw_fps[i]),
                    accuracy=float(raw_acc[i]),
                    eds=float(eds_arr[i]),
                    is_pareto_optimal=bool(pareto_mask[i]),
                    raw_summary=summaries[i],
                )
            )

        return ParetoReport(accuracy_metric=self.accuracy_metric, all_points=points)

    @staticmethod
    def _extract_summary(result: Any) -> Dict[str, Any]:
        if isinstance(result, dict):
            return result.get("summary", result)
        if hasattr(result, "summary"):
            return result.summary()
        raise TypeError(
            f"ParetoFrontierAnalyzer: unsupported result type {type(result).__name__}."
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pareto_frontier_mask(fps: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """Return a boolean mask: True for each point not dominated by any other.

    A point (f_i, a_i) is dominated if there exists (f_j, a_j) such that
    ``f_j >= f_i`` and ``a_j >= a_i`` with at least one strict inequality.

    Args:
        fps: 1-D array of throughput values (higher is better).
        acc: 1-D array of accuracy values (higher is better).

    Returns:
        Boolean array of the same length; True = Pareto-optimal.
    """
    n = len(fps)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if fps[j] >= fps[i] and acc[j] >= acc[i]:
                if fps[j] > fps[i] or acc[j] > acc[i]:
                    mask[i] = False
                    break
    return mask


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise *arr* to [0, 1].  Returns all-zeros if range is 0."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / (hi - lo)


def _harmonic_mean_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise harmonic mean of two arrays; 0 when either element is 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(
            (a > 0) & (b > 0),
            2.0 * a * b / (a + b),
            0.0,
        )
    return h.astype(np.float64)


def _metric_display_name(key: str) -> str:
    names = {
        "mean_iou": "mIoU",
        "success_auc": "Succ. AUC",
        "mean_eao": "EAO",
        "precision_auc": "Prec. AUC",
        "normalized_precision_auc": "Norm. Prec. AUC",
    }
    return names.get(key, key)

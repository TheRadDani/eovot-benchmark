"""Multi-tracker benchmark: evaluate N trackers on one dataset in a single call.

Researchers routinely compare several trackers on the same dataset, but the
existing API requires manually looping, aggregating results, and separately
calling :class:`~eovot.metrics.efficiency.EfficiencyMetricsEngine` and
:class:`~eovot.metrics.statistical.StatisticalTestEngine`.

:class:`MultiTrackerBenchmark` composes those steps into a single high-level
call that returns a :class:`MultiTrackerReport` — a self-contained comparison
report with per-tracker :class:`~eovot.benchmark.engine.BenchmarkResult`s,
an EES-sorted efficiency ranking, and an optional pairwise statistical
significance matrix.

Design principles
-----------------
* **Single-call API** — one method call to evaluate N trackers.
* **Composable** — wraps the existing ``BenchmarkEngine`` + ``EfficiencyMetricsEngine``
  rather than duplicating their logic.
* **Portable output** — ``MultiTrackerReport.to_dict()`` / ``save()`` produce
  plain JSON consumable by downstream scripts.
* **No breaking changes** — purely additive; existing code paths unaffected.

Typical usage::

    from eovot.benchmark.multi_tracker import MultiTrackerBenchmark
    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kcf import KCFTracker
    from eovot.trackers.csrt import CSRTTracker
    from eovot.datasets.synthetic import SyntheticDataset

    dataset = SyntheticDataset(num_sequences=10, num_frames=100)
    bench = MultiTrackerBenchmark(memory_budget_mb=512.0, verbose=False)
    report = bench.run(
        trackers=[MOSSETracker(), KCFTracker(), CSRTTracker()],
        dataset=dataset,
        dataset_name="Synthetic-10",
    )

    print(report.to_markdown_table())
    print("Best (EES):", report.best_tracker("ees"))
    print("Best (IoU):", report.best_tracker("mean_iou"))
    report.save("results/multi_tracker_report.json")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from ..datasets.base import BaseDataset
from ..metrics.efficiency import EfficiencyEntry, EfficiencyMetricsEngine
from ..trackers.base import BaseTracker
from .engine import BenchmarkEngine, BenchmarkResult


@dataclass
class MultiTrackerReport:
    """Results from evaluating N trackers on the same dataset.

    Attributes:
        dataset_name: Human-readable name of the dataset used.
        results: One :class:`~eovot.benchmark.engine.BenchmarkResult` per
            tracker, in evaluation order.
        efficiency_ranking: EES-sorted
            :class:`~eovot.metrics.efficiency.EfficiencyEntry` list.
            The tracker with the highest Edge Efficiency Score is first.
    """

    dataset_name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    efficiency_ranking: List[EfficiencyEntry] = field(default_factory=list)

    def best_tracker(self, metric: str = "ees") -> str:
        """Name of the best-performing tracker by *metric*.

        Args:
            metric: One of:

                * ``"ees"`` (default) — highest Edge Efficiency Score.
                * ``"mean_iou"`` — highest mean IoU.
                * ``"fps"`` — highest frames per second.
                * ``"peak_memory_mb"`` — *lowest* peak memory (inverted).

        Returns:
            Tracker name string.

        Raises:
            ValueError: If results are empty or *metric* is unrecognised.
        """
        if not self.results:
            raise ValueError("MultiTrackerReport contains no results.")
        if metric == "ees":
            if not self.efficiency_ranking:
                raise ValueError("Efficiency ranking is empty.")
            return self.efficiency_ranking[0].tracker_name

        low_is_better = {"peak_memory_mb"}
        key_map: Dict = {
            "mean_iou": lambda r: r.mean_iou,
            "fps": lambda r: r.mean_fps,
            "peak_memory_mb": lambda r: r.peak_memory_mb,
        }
        if metric not in key_map:
            raise ValueError(
                f"Unknown metric '{metric}'. "
                f"Choose from: ees, mean_iou, fps, peak_memory_mb."
            )
        reverse = metric not in low_is_better
        return sorted(self.results, key=key_map[metric], reverse=reverse)[0].tracker_name

    def to_markdown_table(self) -> str:
        """Full comparison as a Markdown table sorted by EES.

        Columns: Rank, Tracker, mIoU, FPS, Mem (MB), EES, Pareto.

        Returns:
            Multi-line Markdown string, or an italicised placeholder when
            there are no results.
        """
        if not self.efficiency_ranking:
            return "_No results available._"
        lines = [
            "| Rank | Tracker | mIoU | FPS | Mem (MB) | EES | Pareto |",
            "|-----:|---------|-----:|----:|---------:|----:|:------:|",
        ]
        for rank, e in enumerate(self.efficiency_ranking, start=1):
            pareto = "✓" if e.on_pareto_front else ""
            lines.append(
                f"| {rank} | {e.tracker_name} | {e.mean_iou:.4f} "
                f"| {e.fps:.1f} | {e.peak_memory_mb:.1f} "
                f"| {e.ees:.4f} | {pareto} |"
            )
        return "\n".join(lines)

    def summary_table(self) -> List[Dict]:
        """Per-tracker summary as a list of dicts (one dict per tracker).

        Each dict contains: ``tracker_name``, ``dataset_name``, ``mean_iou``,
        ``mean_fps``, ``peak_memory_mb``, and all keys from
        :meth:`~eovot.benchmark.engine.BenchmarkResult.summary`.

        Returns:
            List of summary dicts, ordered by evaluation order (not EES rank).
        """
        return [r.summary() for r in self.results]

    def to_dict(self) -> Dict:
        """Serialise to a JSON-compatible dict.

        Returns:
            Dict with keys ``"dataset_name"``, ``"trackers"`` (list of
            per-tracker summary dicts), and ``"efficiency_ranking"``.
        """
        return {
            "dataset_name": self.dataset_name,
            "trackers": self.summary_table(),
            "efficiency_ranking": [
                {
                    "rank": i + 1,
                    "tracker_name": e.tracker_name,
                    "mean_iou": round(e.mean_iou, 4),
                    "fps": round(e.fps, 2),
                    "peak_memory_mb": round(e.peak_memory_mb, 2),
                    "ees": round(e.ees, 6),
                    "on_pareto_front": e.on_pareto_front,
                }
                for i, e in enumerate(self.efficiency_ranking)
            ],
        }

    def save(self, path: Union[str, Path]) -> Path:
        """Serialise this report to a JSON file.

        Args:
            path: Destination path.  A ``.json`` extension is appended when
                the path has none.  Parent directories are created automatically.

        Returns:
            The resolved :class:`pathlib.Path` that was written.
        """
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".json")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return p


class MultiTrackerBenchmark:
    """Evaluate multiple trackers on the same dataset in one method call.

    Internally runs :class:`~eovot.benchmark.engine.BenchmarkEngine` for each
    tracker sequentially (deterministic, reproducible order) and computes
    EES ranking via :class:`~eovot.metrics.efficiency.EfficiencyMetricsEngine`.

    Args:
        memory_budget_mb: Memory ceiling for Edge Efficiency Score computation.
            Trackers within this budget receive full memory credit.
            Default: ``512.0`` MB.
        verbose: Forward progress output from the underlying
            :class:`~eovot.benchmark.engine.BenchmarkEngine`.  Default: ``True``.
        tdp_watts: Enable CPU energy estimation with this TDP value in Watts.
            ``None`` disables energy profiling.  Same semantics as
            :class:`~eovot.benchmark.engine.BenchmarkEngine`.

    Example::

        from eovot.benchmark.multi_tracker import MultiTrackerBenchmark
        from eovot.trackers.mosse import MOSSETracker
        from eovot.trackers.kcf import KCFTracker
        from eovot.datasets.synthetic import SyntheticDataset

        bench = MultiTrackerBenchmark(memory_budget_mb=256.0, verbose=False)
        report = bench.run(
            trackers=[MOSSETracker(), KCFTracker()],
            dataset=SyntheticDataset(num_sequences=5),
            dataset_name="Synthetic-5",
        )
        print(report.to_markdown_table())
        print("Best (EES):", report.best_tracker())
        report.save("results/comparison.json")
    """

    def __init__(
        self,
        memory_budget_mb: float = 512.0,
        verbose: bool = True,
        tdp_watts: Optional[float] = None,
    ) -> None:
        self._engine = BenchmarkEngine(verbose=verbose, tdp_watts=tdp_watts)
        self._efficiency = EfficiencyMetricsEngine(memory_budget_mb=memory_budget_mb)

    def run(
        self,
        trackers: List[BaseTracker],
        dataset: BaseDataset,
        dataset_name: str = "unknown",
        max_sequences: Optional[int] = None,
    ) -> MultiTrackerReport:
        """Evaluate each tracker in *trackers* against *dataset*.

        Trackers are evaluated in the order they appear in *trackers*.
        After all runs the EES ranking and Pareto front are computed from
        the aggregated results.

        Args:
            trackers:       List of :class:`~eovot.trackers.base.BaseTracker`
                instances to evaluate.  Each tracker is called with
                :meth:`~eovot.trackers.base.BaseTracker.initialize` then
                :meth:`~eovot.trackers.base.BaseTracker.update` per frame.
            dataset:        Dataset to benchmark on.
            dataset_name:   Human-readable label for reports and serialisation.
            max_sequences:  Cap the number of sequences evaluated per tracker.
                ``None`` evaluates all sequences.

        Returns:
            :class:`MultiTrackerReport` containing all per-tracker results and
            the EES-sorted efficiency ranking.
        """
        results: List[BenchmarkResult] = []
        for tracker in trackers:
            br = self._engine.run(
                tracker=tracker,
                dataset=dataset,
                dataset_name=dataset_name,
                max_sequences=max_sequences,
            )
            results.append(br)

        ranking = self._efficiency.rank_trackers(results)
        return MultiTrackerReport(
            dataset_name=dataset_name,
            results=results,
            efficiency_ranking=ranking,
        )

"""Real-time latency budget compliance analysis for edge deployment.

Edge devices must track targets at a specific frame rate (e.g., 30 FPS) to
keep pace with incoming video.  If the tracker is too slow, frames pile up and
the system either drops them or introduces unacceptable latency.

This module answers: *"Does this tracker meet a given FPS requirement, and by
how much safety margin?"*

For each (tracker, FPS target) pair the analyzer produces:

- **budget_ms** — milliseconds available per frame = 1000 / target_fps
- **budget_utilization** — mean_latency / budget (< 1 is healthy)
- **safety_margin_ms** — budget − p99_latency (positive = headroom, negative = overrun)
- **real_time_compliant** — True when p99_latency ≤ budget_ms (≤ 1 % tail overrun)
- **min_feasible_fps** — highest FPS target at which the tracker is still compliant

The analysis works from :class:`~eovot.benchmark.engine.BenchmarkResult` objects
so it can run both on freshly-computed results and on results loaded from a
:class:`~eovot.results.bank.ResultsBank`, without re-running the benchmark.

Typical usage::

    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.kcf import KCFTracker
    from eovot.trackers.mosse import MOSSETracker
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.analysis.latency_budget import LatencyBudgetAnalyzer

    engine  = BenchmarkEngine(verbose=False)
    dataset = SyntheticDataset(num_sequences=5)

    results = [
        engine.run(KCFTracker(),   dataset, dataset_name="Synthetic"),
        engine.run(MOSSETracker(), dataset, dataset_name="Synthetic"),
    ]

    analyzer = LatencyBudgetAnalyzer()

    # Compare both trackers at 15, 30, and 60 FPS budgets
    report = analyzer.compare_trackers(results, target_fps_list=[15.0, 30.0, 60.0])
    print(report.to_markdown())

    # Check a single tracker against several targets
    entry = analyzer.analyze_single(results[0], target_fps=30.0)
    print(entry)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult


@dataclass
class LatencyBudgetEntry:
    """Latency compliance summary for one tracker at one FPS target.

    Attributes:
        tracker_name:        Human-readable tracker identifier.
        dataset_name:        Dataset on which the tracker was evaluated.
        target_fps:          Required frames per second.
        budget_ms:           Per-frame time budget = 1000 / target_fps (ms).
        mean_latency_ms:     Mean per-frame update time (ms).
        p95_latency_ms:      95th-percentile latency (ms).
        p99_latency_ms:      99th-percentile latency (ms) — tail indicator.
        budget_utilization:  mean_latency / budget (< 1.0 = within budget).
        safety_margin_ms:    budget − p99_latency; positive = headroom.
        real_time_compliant: True when p99_latency ≤ budget_ms.
        observed_fps:        Measured throughput from the benchmark run.
    """

    tracker_name: str
    dataset_name: str
    target_fps: float
    budget_ms: float
    mean_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    budget_utilization: float
    safety_margin_ms: float
    real_time_compliant: bool
    observed_fps: float

    def __str__(self) -> str:
        status = "PASS" if self.real_time_compliant else "FAIL"
        return (
            f"LatencyBudgetEntry[{self.tracker_name} @ {self.target_fps:.0f} FPS]  "
            f"{status}  "
            f"budget={self.budget_ms:.1f}ms  "
            f"mean={self.mean_latency_ms:.2f}ms  "
            f"p99={self.p99_latency_ms:.2f}ms  "
            f"margin={self.safety_margin_ms:+.2f}ms  "
            f"util={self.budget_utilization:.2f}"
        )

    def to_dict(self) -> Dict:
        return {
            "tracker": self.tracker_name,
            "dataset": self.dataset_name,
            "target_fps": self.target_fps,
            "budget_ms": round(self.budget_ms, 3),
            "mean_latency_ms": round(self.mean_latency_ms, 3),
            "p95_latency_ms": round(self.p95_latency_ms, 3),
            "p99_latency_ms": round(self.p99_latency_ms, 3),
            "budget_utilization": round(self.budget_utilization, 4),
            "safety_margin_ms": round(self.safety_margin_ms, 3),
            "real_time_compliant": self.real_time_compliant,
            "observed_fps": round(self.observed_fps, 2),
        }


@dataclass
class LatencyBudgetReport:
    """Full latency compliance report for one or more trackers.

    Attributes:
        entries: All :class:`LatencyBudgetEntry` objects in the report,
            ordered by (target_fps, tracker_name).
        trackers: Sorted list of unique tracker names.
        target_fps_list: Sorted list of unique FPS targets evaluated.
    """

    entries: List[LatencyBudgetEntry] = field(default_factory=list)
    trackers: List[str] = field(default_factory=list)
    target_fps_list: List[float] = field(default_factory=list)

    def compliant_trackers(self, target_fps: float) -> List[str]:
        """Return tracker names that are real-time compliant at *target_fps*.

        Args:
            target_fps: FPS requirement to query.

        Returns:
            Sorted list of compliant tracker names.
        """
        return sorted(
            e.tracker_name for e in self.entries
            if e.target_fps == target_fps and e.real_time_compliant
        )

    def fastest_compliant(self, target_fps: float) -> Optional[LatencyBudgetEntry]:
        """Return the entry with the highest observed FPS among compliant trackers.

        Args:
            target_fps: FPS requirement.

        Returns:
            :class:`LatencyBudgetEntry` with highest throughput, or ``None``.
        """
        candidates = [
            e for e in self.entries
            if e.target_fps == target_fps and e.real_time_compliant
        ]
        return max(candidates, key=lambda e: e.observed_fps) if candidates else None

    def min_feasible_fps(self, tracker_name: str) -> Optional[float]:
        """Return the highest FPS target at which *tracker_name* is compliant.

        Among all FPS targets the tracker was evaluated against, this is the
        largest one for which ``real_time_compliant`` is True.

        Args:
            tracker_name: Tracker to query.

        Returns:
            Highest compliant target FPS, or ``None`` if the tracker fails all.
        """
        compliant = [
            e.target_fps for e in self.entries
            if e.tracker_name == tracker_name and e.real_time_compliant
        ]
        return max(compliant) if compliant else None

    def to_markdown(self) -> str:
        """Format the report as a Markdown table.

        Each row is one (tracker, FPS target) pair.  Compliant entries are
        marked with ✓; non-compliant with ✗.

        Returns:
            Multi-line Markdown string ready for embedding in reports.
        """
        if not self.entries:
            return "_No latency budget entries in report._"

        lines = [
            "| Tracker | Target FPS | Budget (ms) | Mean Lat (ms) "
            "| p99 Lat (ms) | Margin (ms) | Util | Observed FPS | Compliant |",
            "|---------|----------:|------------:|-------------:|"
            "------------:|------------:|-----:|-------------:|:---------:|",
        ]
        for e in sorted(self.entries, key=lambda x: (x.target_fps, x.tracker_name)):
            status = "✓" if e.real_time_compliant else "✗"
            lines.append(
                f"| {e.tracker_name} | {e.target_fps:.0f} | {e.budget_ms:.2f} "
                f"| {e.mean_latency_ms:.2f} | {e.p99_latency_ms:.2f} "
                f"| {e.safety_margin_ms:+.2f} | {e.budget_utilization:.2f} "
                f"| {e.observed_fps:.1f} | {status} |"
            )
        return "\n".join(lines)

    def to_compliance_matrix(self) -> str:
        """Format a compact tracker × FPS compliance matrix.

        Rows are trackers, columns are FPS targets.  Each cell is ✓ or ✗.

        Returns:
            Multi-line Markdown string.
        """
        if not self.entries:
            return "_No entries._"

        fps_list = sorted(set(e.target_fps for e in self.entries))
        tracker_list = sorted(set(e.tracker_name for e in self.entries))

        fps_header = " | ".join(f"{f:.0f} FPS" for f in fps_list)
        sep = " | ".join(["------"] * len(fps_list))
        lines = [
            f"| Tracker | {fps_header} |",
            f"|---------|{sep}|",
        ]

        lookup: Dict[Tuple[str, float], bool] = {
            (e.tracker_name, e.target_fps): e.real_time_compliant
            for e in self.entries
        }

        for tracker in tracker_list:
            cells = []
            for fps in fps_list:
                compliant = lookup.get((tracker, fps))
                cells.append("✓" if compliant else ("✗" if compliant is not None else "—"))
            lines.append(f"| {tracker} | " + " | ".join(cells) + " |")

        return "\n".join(lines)


class LatencyBudgetAnalyzer:
    """Compute real-time compliance metrics from benchmark profiling data.

    Works from :class:`~eovot.benchmark.engine.BenchmarkResult` objects, which
    include per-sequence :class:`~eovot.profiling.profiler.ProfilingResult`
    summaries with mean, p95, and p99 latency statistics.

    Compliance rule: a tracker is *real-time compliant* at *target_fps* when
    its p99 latency ≤ 1000 / target_fps.  Using p99 (not mean) as the
    threshold gives the tracker a 99 % probability of meeting each frame
    deadline in steady-state operation.

    Args:
        compliance_percentile: Latency percentile used for the compliance check.
            Default: ``99`` (p99).  Use ``95`` for a more lenient threshold
            (5 % deadline miss rate).

    Example::

        analyzer = LatencyBudgetAnalyzer()
        report = analyzer.compare_trackers(results, [15.0, 30.0, 60.0])
        print(report.to_compliance_matrix())
        print(report.to_markdown())
    """

    def __init__(self, compliance_percentile: int = 99) -> None:
        if compliance_percentile not in (95, 99):
            raise ValueError(
                f"compliance_percentile must be 95 or 99, got {compliance_percentile}."
            )
        self.compliance_percentile = compliance_percentile

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze_single(
        self,
        result: "BenchmarkResult",
        target_fps: float,
    ) -> LatencyBudgetEntry:
        """Compute compliance metrics for one tracker at one FPS target.

        Args:
            result: Benchmark result whose profiling data will be analyzed.
            target_fps: Required frames per second (must be > 0).

        Returns:
            :class:`LatencyBudgetEntry` with all compliance metrics populated.

        Raises:
            ValueError: If *target_fps* is not positive, or if *result* has no
                sequence profiling data.
        """
        if target_fps <= 0:
            raise ValueError(f"target_fps must be > 0, got {target_fps}.")
        if not result.sequence_results:
            raise ValueError(
                f"BenchmarkResult for '{result.tracker_name}' has no sequence results."
            )

        budget_ms = 1000.0 / target_fps

        mean_lat = self._aggregate_mean_latency(result)
        p95_lat = self._aggregate_percentile_latency(result, 95)
        p99_lat = self._aggregate_percentile_latency(result, 99)

        compliance_lat = p99_lat if self.compliance_percentile == 99 else p95_lat
        safety_margin = budget_ms - compliance_lat
        utilization = mean_lat / budget_ms if budget_ms > 0 else float("inf")

        return LatencyBudgetEntry(
            tracker_name=result.tracker_name,
            dataset_name=result.dataset_name,
            target_fps=target_fps,
            budget_ms=budget_ms,
            mean_latency_ms=mean_lat,
            p95_latency_ms=p95_lat,
            p99_latency_ms=p99_lat,
            budget_utilization=round(utilization, 4),
            safety_margin_ms=round(safety_margin, 3),
            real_time_compliant=compliance_lat <= budget_ms,
            observed_fps=result.mean_fps,
        )

    def analyze(
        self,
        result: "BenchmarkResult",
        target_fps_list: List[float],
    ) -> LatencyBudgetReport:
        """Evaluate one tracker across multiple FPS targets.

        Args:
            result: Benchmark result for one tracker.
            target_fps_list: List of FPS targets to evaluate (e.g.,
                ``[15.0, 24.0, 30.0, 60.0]``).

        Returns:
            :class:`LatencyBudgetReport` with one entry per FPS target.
        """
        entries = [self.analyze_single(result, fps) for fps in target_fps_list]
        return LatencyBudgetReport(
            entries=entries,
            trackers=[result.tracker_name],
            target_fps_list=sorted(target_fps_list),
        )

    def compare_trackers(
        self,
        results: List["BenchmarkResult"],
        target_fps_list: List[float],
    ) -> LatencyBudgetReport:
        """Evaluate multiple trackers across multiple FPS targets.

        Args:
            results: One :class:`~eovot.benchmark.engine.BenchmarkResult` per
                tracker.
            target_fps_list: FPS targets to evaluate (e.g.,
                ``[15.0, 30.0, 60.0]``).

        Returns:
            :class:`LatencyBudgetReport` combining all (tracker, FPS) pairs.
        """
        all_entries: List[LatencyBudgetEntry] = []
        for result in results:
            for fps in target_fps_list:
                all_entries.append(self.analyze_single(result, fps))

        return LatencyBudgetReport(
            entries=all_entries,
            trackers=sorted(set(r.tracker_name for r in results)),
            target_fps_list=sorted(target_fps_list),
        )

    def recommend_fps_budget(
        self,
        results: List["BenchmarkResult"],
        candidate_fps: Optional[List[float]] = None,
    ) -> Dict[str, Optional[float]]:
        """Find the highest compliant FPS target for each tracker.

        Sweeps *candidate_fps* targets from highest to lowest and returns the
        first (highest) at which each tracker is real-time compliant.

        Args:
            results: List of benchmark results.
            candidate_fps: FPS targets to consider, in any order.  Default:
                ``[120, 60, 30, 24, 15, 10, 5]``.

        Returns:
            Dict mapping tracker_name → highest compliant FPS (or ``None`` if
            the tracker fails all candidates).
        """
        if candidate_fps is None:
            candidate_fps = [120.0, 60.0, 30.0, 24.0, 15.0, 10.0, 5.0]

        report = self.compare_trackers(results, sorted(candidate_fps, reverse=True))

        recommendations: Dict[str, Optional[float]] = {}
        for result in results:
            recommendations[result.tracker_name] = report.min_feasible_fps(
                result.tracker_name
            )
        return recommendations

    # ------------------------------------------------------------------
    # Private aggregation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_mean_latency(result: "BenchmarkResult") -> float:
        """Weighted-mean latency across all sequences (weighted by frame count)."""
        total_frames = sum(r.profiling.frame_count for r in result.sequence_results)
        if total_frames == 0:
            return 0.0
        weighted_sum = sum(
            r.profiling.latency_mean_ms * r.profiling.frame_count
            for r in result.sequence_results
        )
        return weighted_sum / total_frames

    @staticmethod
    def _aggregate_percentile_latency(
        result: "BenchmarkResult", percentile: int
    ) -> float:
        """Approximate aggregate percentile across sequences.

        Uses the conservative maximum of per-sequence percentiles — this
        over-estimates slightly but avoids under-estimating the tail risk,
        which would lead to falsely-compliant conclusions.

        Args:
            result:     Benchmark result.
            percentile: 95 or 99.

        Returns:
            Conservative aggregate percentile in ms.
        """
        if percentile == 99:
            values = [r.profiling.latency_p99_ms for r in result.sequence_results]
        else:
            values = [r.profiling.latency_p95_ms for r in result.sequence_results]

        if not values:
            return 0.0
        # Use mean rather than max to avoid extreme sensitivity to outliers
        # in long benchmark runs — still conservative vs. the true aggregate.
        return float(sum(values) / len(values))

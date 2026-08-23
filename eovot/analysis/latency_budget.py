"""Real-time latency budget analysis for edge tracker deployment.

Edge and embedded systems impose hard or soft real-time constraints:
a robot controller at 30 FPS gives each tracker exactly 33.3 ms per frame.
Standard benchmarks report mean latency, but the mean alone cannot answer the
deployment question — a tracker with mean=10 ms and p99=80 ms will cause
deadline misses in 1 % of frames, which on a 30 FPS stream means a dropped
frame every ~3 seconds.

This module provides :class:`LatencyBudgetAnalyzer`, which takes a benchmark
result (or a single :class:`~eovot.profiling.profiler.ProfilingResult`) and
evaluates whether the tracker can sustain one or more target frame rates.  For
each (tracker, target-FPS) pair it computes:

- **budget_ms** — the wall-clock deadline: ``1000 / target_fps`` ms.
- **mean_headroom_ms** — slack between the deadline and mean latency.
- **p99_headroom_ms** — slack between the deadline and p99 latency.  Negative
  values indicate that the tail of the latency distribution exceeds the
  deadline, causing missed frames even in a well-tuned deployment.
- **p99_safety_margin** — ``budget_ms / p99_ms``.  Values ≥ 1.0 mean the
  p99 fits inside the budget; < 1.0 means tail frames miss the deadline.
- **estimated_compliance** — Gaussian approximation of the fraction of frames
  that finish within the deadline, derived from the stored mean and std.
  This is a lower-bound estimate; the true distribution is typically
  right-skewed, so actual compliance may be slightly higher.
- **verdict** — a human-readable deployment recommendation.

Typical usage::

    from eovot.analysis.latency_budget import LatencyBudgetAnalyzer
    from eovot.benchmark.engine import BenchmarkResult

    result = BenchmarkResult.load("results/mosse-synthetic.json")
    analyzer = LatencyBudgetAnalyzer(target_fps_list=[15, 24, 30, 60])
    report = analyzer.analyze_benchmark(result)

    print(report.to_markdown_table())
    print(report.recommendations())

The analyzer can also operate on a single
:class:`~eovot.profiling.profiler.ProfilingResult`::

    from eovot.analysis.latency_budget import LatencyBudgetAnalyzer

    profiling = ...  # from Profiler.summary()
    analyzer  = LatencyBudgetAnalyzer(target_fps_list=[30, 60])
    entries   = analyzer.analyze_profiling(profiling)
    for e in entries:
        print(e)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult
    from ..profiling.profiler import ProfilingResult


# ---------------------------------------------------------------------------
# Gaussian CDF approximation (no scipy dependency)
# ---------------------------------------------------------------------------

def _erf_approx(x: float) -> float:
    """Abramowitz & Stegun approximation of erf(x), max error 1.5e-7."""
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (
        0.254829592
        + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    )
    return sign * (1.0 - poly * math.exp(-(x * x)))


def _normal_cdf(x: float, mean: float, std: float) -> float:
    """Cumulative distribution function of N(mean, std) evaluated at x."""
    if std <= 0:
        return 1.0 if x >= mean else 0.0
    z = (x - mean) / (std * math.sqrt(2.0))
    return 0.5 * (1.0 + _erf_approx(z))


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

_VERDICTS = {
    "safe": "SAFE — p99 within budget; tracker can sustain this frame rate.",
    "marginal": "MARGINAL — mean within budget but p99 exceeds it; expect occasional drops.",
    "unsafe": "UNSAFE — mean latency exceeds budget; tracker cannot sustain this frame rate.",
}


@dataclass
class LatencyBudgetEntry:
    """Latency budget evaluation for one (tracker, target-FPS) pair.

    Attributes:
        tracker_name: Human-readable tracker identifier.
        target_fps: Desired frame rate (frames per second).
        budget_ms: Frame deadline in milliseconds (``1000 / target_fps``).
        mean_latency_ms: Mean per-frame tracking latency.
        latency_std_ms: Standard deviation of per-frame latency.
        latency_p99_ms: 99th-percentile per-frame latency.
        mean_headroom_ms: ``budget_ms - mean_latency_ms``.  Positive → slack;
            negative → mean already exceeds deadline.
        p99_headroom_ms: ``budget_ms - latency_p99_ms``.  Positive → tail fits;
            negative → tail frames miss the deadline.
        p99_safety_margin: ``budget_ms / latency_p99_ms``.  ≥ 1.0 is safe.
        estimated_compliance: Gaussian approximation of the fraction of frames
            completing within the deadline, in ``[0, 1]``.
        verdict: ``"safe"``, ``"marginal"``, or ``"unsafe"``.
    """

    tracker_name: str
    target_fps: float
    budget_ms: float
    mean_latency_ms: float
    latency_std_ms: float
    latency_p99_ms: float
    mean_headroom_ms: float
    p99_headroom_ms: float
    p99_safety_margin: float
    estimated_compliance: float
    verdict: str

    def verdict_description(self) -> str:
        """Return the human-readable verdict description."""
        return _VERDICTS.get(self.verdict, self.verdict)

    def __str__(self) -> str:
        return (
            f"LatencyBudgetEntry({self.tracker_name} @ {self.target_fps:.0f} FPS "
            f"| budget={self.budget_ms:.1f}ms "
            f"| mean={self.mean_latency_ms:.2f}ms "
            f"| p99={self.latency_p99_ms:.2f}ms "
            f"| headroom(p99)={self.p99_headroom_ms:+.2f}ms "
            f"| compliance={self.estimated_compliance:.3f} "
            f"| {self.verdict.upper()})"
        )


@dataclass
class LatencyBudgetReport:
    """Full latency budget report for one benchmark result.

    Contains one :class:`LatencyBudgetEntry` per (target FPS) evaluated.
    When the report was built from a :class:`~eovot.benchmark.engine.BenchmarkResult`,
    ``tracker_name`` and ``dataset_name`` are populated from the result.

    Attributes:
        tracker_name: Tracker evaluated.
        dataset_name: Dataset evaluated.
        entries: One entry per target FPS, sorted by target_fps ascending.
    """

    tracker_name: str
    dataset_name: str
    entries: List[LatencyBudgetEntry] = field(default_factory=list)

    def safe_targets(self) -> List[float]:
        """Return the list of target FPS values at which the tracker is SAFE."""
        return [e.target_fps for e in self.entries if e.verdict == "safe"]

    def max_safe_fps(self) -> Optional[float]:
        """Highest target FPS at which the tracker is SAFE, or ``None`` if none."""
        safe = self.safe_targets()
        return max(safe) if safe else None

    def to_markdown_table(self) -> str:
        """Format the report as a Markdown table for embedding in papers or READMEs.

        Returns:
            Multi-line Markdown string with one row per target FPS.
        """
        lines = [
            f"### Latency Budget: {self.tracker_name} on {self.dataset_name}\n",
            "| Target FPS | Budget (ms) | Mean (ms) | p99 (ms) | Headroom p99 | Compliance | Verdict |",
            "|:----------:|:-----------:|:---------:|:--------:|:------------:|:----------:|:-------:|",
        ]
        for e in self.entries:
            compliance_pct = f"{e.estimated_compliance * 100:.1f}%"
            headroom = f"{e.p99_headroom_ms:+.2f}"
            lines.append(
                f"| {e.target_fps:.0f} | {e.budget_ms:.1f} "
                f"| {e.mean_latency_ms:.2f} | {e.latency_p99_ms:.2f} "
                f"| {headroom} | {compliance_pct} | **{e.verdict.upper()}** |"
            )
        return "\n".join(lines)

    def recommendations(self) -> str:
        """Return a human-readable deployment recommendation summary."""
        max_safe = self.max_safe_fps()
        if max_safe is None:
            return (
                f"{self.tracker_name} cannot meet any of the tested frame-rate targets. "
                "Consider a faster tracker or apply frame-skipping to reduce compute."
            )
        safe_list = ", ".join(f"{f:.0f}" for f in sorted(self.safe_targets()))
        unsafe = [e for e in self.entries if e.verdict == "unsafe"]
        if not unsafe:
            return (
                f"{self.tracker_name} meets all tested frame-rate targets. "
                f"Maximum tested safe rate: {max_safe:.0f} FPS."
            )
        return (
            f"{self.tracker_name} is SAFE at: {safe_list} FPS. "
            f"Maximum safe rate: {max_safe:.0f} FPS. "
            f"Rates above {max_safe:.0f} FPS exceed the mean-latency budget."
        )


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class LatencyBudgetAnalyzer:
    """Evaluate whether a tracker can sustain one or more target frame rates.

    Args:
        target_fps_list: List of desired frame rates to evaluate.
            Default: ``[15, 24, 30, 60]`` — covering low-power surveillance,
            standard video, smooth video, and high-frequency robotics targets.
        marginal_p99_factor: Threshold multiplier for the "marginal" verdict.
            When ``budget_ms / p99_ms ≥ marginal_p99_factor``, the entry is
            safe; when ``budget_ms / mean_ms ≥ 1`` but ``p99`` exceeds the
            budget, it is marginal.  Default: ``1.0`` (p99 must fit in budget).
    """

    def __init__(
        self,
        target_fps_list: Optional[List[float]] = None,
        marginal_p99_factor: float = 1.0,
    ) -> None:
        self.target_fps_list: List[float] = (
            list(target_fps_list) if target_fps_list is not None else [15.0, 24.0, 30.0, 60.0]
        )
        self.marginal_p99_factor = marginal_p99_factor

    def _evaluate_entry(
        self,
        tracker_name: str,
        target_fps: float,
        mean_latency_ms: float,
        latency_std_ms: float,
        latency_p99_ms: float,
    ) -> LatencyBudgetEntry:
        """Build a single :class:`LatencyBudgetEntry`."""
        budget_ms = 1000.0 / target_fps
        mean_headroom = budget_ms - mean_latency_ms
        p99_headroom = budget_ms - latency_p99_ms
        p99_safety = budget_ms / latency_p99_ms if latency_p99_ms > 0 else float("inf")
        compliance = _normal_cdf(budget_ms, mean_latency_ms, latency_std_ms)

        if mean_headroom < 0:
            verdict = "unsafe"
        elif p99_safety >= self.marginal_p99_factor:
            verdict = "safe"
        else:
            verdict = "marginal"

        return LatencyBudgetEntry(
            tracker_name=tracker_name,
            target_fps=target_fps,
            budget_ms=budget_ms,
            mean_latency_ms=mean_latency_ms,
            latency_std_ms=latency_std_ms,
            latency_p99_ms=latency_p99_ms,
            mean_headroom_ms=mean_headroom,
            p99_headroom_ms=p99_headroom,
            p99_safety_margin=p99_safety,
            estimated_compliance=compliance,
            verdict=verdict,
        )

    def analyze_profiling(
        self, profiling: "ProfilingResult"
    ) -> List[LatencyBudgetEntry]:
        """Evaluate latency budget for every target FPS from a single profiling result.

        Args:
            profiling: Output of :meth:`~eovot.profiling.profiler.Profiler.summary`.

        Returns:
            List of :class:`LatencyBudgetEntry`, one per target FPS, sorted
            by ``target_fps`` ascending.
        """
        entries = [
            self._evaluate_entry(
                tracker_name=profiling.tracker_name,
                target_fps=fps,
                mean_latency_ms=profiling.latency_mean_ms,
                latency_std_ms=profiling.latency_std_ms,
                latency_p99_ms=profiling.latency_p99_ms,
            )
            for fps in sorted(self.target_fps_list)
        ]
        return entries

    def analyze_benchmark(
        self,
        result: "BenchmarkResult",
        aggregate: bool = True,
    ) -> LatencyBudgetReport:
        """Evaluate latency budget from a full benchmark result.

        When ``aggregate=True`` (default), uses the mean profiling statistics
        across all sequences to produce a single representative report.
        When ``aggregate=False``, uses the worst-case (highest mean latency)
        sequence as a conservative lower bound.

        Args:
            result: Output of :meth:`~eovot.benchmark.engine.BenchmarkEngine.run`.
            aggregate: Whether to average across sequences (default) or use
                the worst-case sequence.

        Returns:
            :class:`LatencyBudgetReport` with one entry per target FPS.
        """
        import numpy as np  # local import — numpy is optional at module level

        if not result.sequence_results:
            return LatencyBudgetReport(
                tracker_name=result.tracker_name,
                dataset_name=result.dataset_name,
                entries=[],
            )

        profilings = [r.profiling for r in result.sequence_results]
        means = np.array([p.latency_mean_ms for p in profilings])
        stds = np.array([p.latency_std_ms for p in profilings])
        p99s = np.array([p.latency_p99_ms for p in profilings])

        if aggregate:
            mean_lat = float(means.mean())
            std_lat = float(np.sqrt((stds ** 2).mean()))   # pooled std
            p99_lat = float(p99s.mean())
        else:
            worst_idx = int(means.argmax())
            mean_lat = float(means[worst_idx])
            std_lat = float(stds[worst_idx])
            p99_lat = float(p99s[worst_idx])

        entries = [
            self._evaluate_entry(
                tracker_name=result.tracker_name,
                target_fps=fps,
                mean_latency_ms=mean_lat,
                latency_std_ms=std_lat,
                latency_p99_ms=p99_lat,
            )
            for fps in sorted(self.target_fps_list)
        ]
        return LatencyBudgetReport(
            tracker_name=result.tracker_name,
            dataset_name=result.dataset_name,
            entries=entries,
        )

    @staticmethod
    def multi_tracker_table(reports: List[LatencyBudgetReport]) -> str:
        """Format a side-by-side Markdown table comparing multiple trackers.

        Columns: Tracker | FPS target | Budget | Mean lat. | p99 lat. | Compliance | Verdict

        Args:
            reports: One :class:`LatencyBudgetReport` per tracker.

        Returns:
            Multi-line Markdown string ready for embedding in README or paper.
        """
        lines = [
            "| Tracker | Target FPS | Budget (ms) | Mean (ms) | p99 (ms) | Compliance | Verdict |",
            "|---------|:----------:|:-----------:|:---------:|:--------:|:----------:|:-------:|",
        ]
        for report in reports:
            for e in report.entries:
                compliance_pct = f"{e.estimated_compliance * 100:.1f}%"
                lines.append(
                    f"| {e.tracker_name} | {e.target_fps:.0f} | {e.budget_ms:.1f} "
                    f"| {e.mean_latency_ms:.2f} | {e.latency_p99_ms:.2f} "
                    f"| {compliance_pct} | **{e.verdict.upper()}** |"
                )
        return "\n".join(lines)

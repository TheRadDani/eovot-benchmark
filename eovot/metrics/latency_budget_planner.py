"""SLA compliance analysis for edge tracker deployment.

Real-time tracking on edge hardware comes with hard latency budgets:
a system running at 30 FPS must complete each inference within ~33 ms.
A tracker whose mean latency is 20 ms can still violate that budget on
5 % of frames if its latency distribution has a heavy tail.

This module answers the deployment question that raw p95/p99 numbers
do not directly answer:

    "If my budget is **B ms/frame**, what fraction of frames will my
    tracker complete within that budget?"

and its inverse:

    "What budget do I need to meet a **K %** compliance target?"

Approach
--------
:class:`~eovot.profiling.profiler.ProfilingResult` stores five aggregate
statistics for the latency distribution: *mean*, *std*, *p95*, *p99*, and
*CV* (coefficient of variation).  The raw per-frame samples are not
persisted in JSON files, so the planner fits a **log-normal distribution**
to those statistics.  Log-normal is well-motivated for per-frame inference
latency: it is strictly positive, right-skewed (matches occasional stalls),
and determined by just two parameters (μ, σ of the underlying normal).

The two log-normal parameters are estimated by matching the empirical mean
and standard deviation::

    σ² = ln(1 + (std / mean)²)   # from coefficient of variation
    μ  = ln(mean) - σ² / 2

This gives a continuous CDF from which compliance rates and budget
quantiles can be derived in closed form — no scipy required.

Typical usage::

    from eovot.metrics.latency_budget_planner import LatencyBudgetPlanner

    planner = LatencyBudgetPlanner()

    # What fraction of frames meet a 33 ms budget?
    result = planner.compute_compliance(profiling_result, budget_ms=33.0)
    print(f"Compliance at 33 ms: {result.compliance_rate:.1%}")

    # What budget do I need for 99 % compliance?
    budget = planner.find_min_budget(profiling_result, target_rate=0.99)
    print(f"Budget needed for 99 % SLA: {budget:.1f} ms")

    # Compare multiple trackers side by side.
    table = planner.to_markdown_table(
        planner.compare_trackers(benchmark_results, budget_ms=33.0)
    )
    print(table)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult
    from ..profiling.profiler import ProfilingResult


# ---------------------------------------------------------------------------
# Log-normal helpers (no scipy dependency)
# ---------------------------------------------------------------------------

def _lognormal_params(mean_ms: float, std_ms: float) -> tuple:
    """Fit log-normal (μ, σ) from empirical mean and std.

    Returns:
        (mu, sigma) for the underlying normal distribution, or ``None``
        when the distribution cannot be fitted (zero std or mean).
    """
    if mean_ms <= 0 or std_ms < 0:
        return (None, None)
    if std_ms == 0.0:
        # Degenerate: all frames take exactly mean_ms — deterministic.
        return (math.log(mean_ms), 0.0)
    cv_sq = (std_ms / mean_ms) ** 2
    sigma_sq = math.log(1.0 + cv_sq)
    mu = math.log(mean_ms) - sigma_sq / 2.0
    return (mu, math.sqrt(sigma_sq))


def _erf_approx(x: float) -> float:
    """Abramowitz & Stegun rational approximation of erf(x) — max error 1.5e-7."""
    sign = 1.0 if x >= 0.0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = (
        0.254829592 * t
        - 0.284496736 * t ** 2
        + 1.421413741 * t ** 3
        - 1.453152027 * t ** 4
        + 1.061405429 * t ** 5
    )
    return sign * (1.0 - poly * math.exp(-(x * x)))


def _lognormal_cdf(x: float, mu: float, sigma: float) -> float:
    """Log-normal CDF: P(X ≤ x)."""
    if x <= 0.0:
        return 0.0
    if sigma == 0.0:
        return 1.0 if math.exp(mu) <= x else 0.0
    z = (math.log(x) - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + _erf_approx(z))


def _lognormal_quantile(p: float, mu: float, sigma: float) -> float:
    """Log-normal quantile function (inverse CDF) via bisection.

    Returns the value x such that P(X ≤ x) ≈ p.
    """
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return float("inf")
    if sigma == 0.0:
        return math.exp(mu)

    # Bisection over a wide range: [1e-6, 10 × 99.9th percentile estimate].
    lo = 1e-6
    hi = math.exp(mu + 4.0 * sigma)  # well past the 99.9th percentile
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if _lognormal_cdf(mid, mu, sigma) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SLAComplianceResult:
    """SLA compliance summary for one tracker at one latency budget.

    Attributes:
        tracker_name: Human-readable tracker identifier.
        budget_ms: The latency budget in milliseconds (one frame must
            complete within this window to count as compliant).
        compliance_rate: Estimated fraction of frames that meet the budget
            (range ``[0, 1]``).
        violation_rate: ``1 - compliance_rate``.
        min_budget_99pct_ms: Estimated minimum budget (ms) for 99 %
            compliance — the "near-worst-case" operating point.
        min_budget_999pct_ms: Estimated minimum budget (ms) for 99.9 %
            compliance — the "tail-risk" operating point.
        mean_latency_ms: Mean per-frame latency (from ProfilingResult).
        p99_latency_ms: 99th-percentile per-frame latency.
        fps_equivalent: ``1000 / budget_ms`` — the FPS target the budget
            corresponds to.
        distribution: ``"lognormal"`` or ``"deterministic"`` depending on
            whether a distribution could be fitted.
    """

    tracker_name: str
    budget_ms: float
    compliance_rate: float
    violation_rate: float
    min_budget_99pct_ms: float
    min_budget_999pct_ms: float
    mean_latency_ms: float
    p99_latency_ms: float
    fps_equivalent: float
    distribution: str = "lognormal"

    def __str__(self) -> str:
        return (
            f"SLACompliance[{self.tracker_name}  "
            f"budget={self.budget_ms:.1f}ms ({self.fps_equivalent:.0f}fps)]  "
            f"compliance={self.compliance_rate:.1%}  "
            f"p99={self.p99_latency_ms:.1f}ms  "
            f"budget_for_99%={self.min_budget_99pct_ms:.1f}ms  "
            f"budget_for_99.9%={self.min_budget_999pct_ms:.1f}ms"
        )


# ---------------------------------------------------------------------------
# Main planner
# ---------------------------------------------------------------------------

class LatencyBudgetPlanner:
    """Estimate SLA compliance for tracker latency distributions.

    Uses a log-normal model fitted to ProfilingResult aggregate statistics
    (mean, std) to answer deployment questions about latency budget compliance
    without needing raw per-frame latency samples.

    Args:
        fallback_percentile_factor: When std is unavailable (zero), how many
            multiples of the mean to use as the 99th-percentile estimate.
            Default: ``1.5`` (99th-percentile ≈ 1.5 × mean, a conservative
            assumption for jitter-free trackers).
    """

    def __init__(self, fallback_percentile_factor: float = 1.5) -> None:
        self._fallback_factor = fallback_percentile_factor

    # ------------------------------------------------------------------
    # Core analysis methods
    # ------------------------------------------------------------------

    def compute_compliance(
        self,
        profiling: "ProfilingResult",
        budget_ms: float,
    ) -> SLAComplianceResult:
        """Compute SLA compliance for one tracker at a given latency budget.

        Args:
            profiling: Per-tracker profiling summary from
                :class:`~eovot.profiling.profiler.Profiler`.
            budget_ms: Latency budget per frame in milliseconds (e.g. 33.3
                for a 30-FPS system, 16.7 for 60 FPS).

        Returns:
            :class:`SLAComplianceResult` with compliance rate and budget
            recommendations.

        Raises:
            ValueError: If ``budget_ms`` is not positive.
        """
        if budget_ms <= 0:
            raise ValueError(f"budget_ms must be positive, got {budget_ms}.")

        mean_ms = profiling.latency_mean_ms
        std_ms = profiling.latency_std_ms
        p99_ms = profiling.latency_p99_ms

        mu, sigma = _lognormal_params(mean_ms, std_ms)

        if mu is None:
            # Cannot fit a distribution — degenerate case
            compliance = 1.0 if mean_ms <= budget_ms else 0.0
            return SLAComplianceResult(
                tracker_name=profiling.tracker_name,
                budget_ms=budget_ms,
                compliance_rate=compliance,
                violation_rate=1.0 - compliance,
                min_budget_99pct_ms=mean_ms * self._fallback_factor,
                min_budget_999pct_ms=mean_ms * self._fallback_factor,
                mean_latency_ms=mean_ms,
                p99_latency_ms=p99_ms,
                fps_equivalent=1000.0 / budget_ms,
                distribution="deterministic",
            )

        compliance = _lognormal_cdf(budget_ms, mu, sigma)
        b99 = _lognormal_quantile(0.99, mu, sigma)
        b999 = _lognormal_quantile(0.999, mu, sigma)

        # Cross-validate fitted p99 against empirical p99; take the more
        # conservative (larger) value to avoid underestimating tail risk.
        if p99_ms > 0:
            b99 = max(b99, p99_ms)
            b999 = max(b999, p99_ms * 1.1)

        return SLAComplianceResult(
            tracker_name=profiling.tracker_name,
            budget_ms=budget_ms,
            compliance_rate=compliance,
            violation_rate=1.0 - compliance,
            min_budget_99pct_ms=b99,
            min_budget_999pct_ms=b999,
            mean_latency_ms=mean_ms,
            p99_latency_ms=p99_ms,
            fps_equivalent=1000.0 / budget_ms,
        )

    def find_min_budget(
        self,
        profiling: "ProfilingResult",
        target_rate: float = 0.99,
    ) -> float:
        """Find the minimum latency budget (ms) achieving a target compliance rate.

        Args:
            profiling: Per-tracker profiling summary.
            target_rate: Desired compliance rate in ``(0, 1)``.  Default: 0.99.

        Returns:
            Minimum budget in milliseconds such that
            ``compliance_rate >= target_rate``.

        Raises:
            ValueError: If ``target_rate`` is not in ``(0, 1)``.
        """
        if not (0.0 < target_rate < 1.0):
            raise ValueError(f"target_rate must be in (0, 1), got {target_rate}.")

        mean_ms = profiling.latency_mean_ms
        std_ms = profiling.latency_std_ms
        mu, sigma = _lognormal_params(mean_ms, std_ms)

        if mu is None:
            return mean_ms * self._fallback_factor

        budget = _lognormal_quantile(target_rate, mu, sigma)

        # Cross-validate against stored empirical quantiles.
        if target_rate >= 0.99 and profiling.latency_p99_ms > 0:
            budget = max(budget, profiling.latency_p99_ms)
        if target_rate >= 0.95 and profiling.latency_p95_ms > 0:
            budget = max(budget, profiling.latency_p95_ms)

        return budget

    def compare_trackers(
        self,
        results: List["BenchmarkResult"],
        budget_ms: float,
    ) -> List[SLAComplianceResult]:
        """Compare multiple trackers on SLA compliance at a shared budget.

        Aggregates per-sequence ProfilingResults into a single combined
        distribution per tracker (using the mean of means as the effective
        mean, and the RMS of per-sequence stds as the effective std), then
        applies :meth:`compute_compliance` to each tracker.

        Args:
            results: One :class:`~eovot.benchmark.engine.BenchmarkResult`
                per tracker/dataset combination.
            budget_ms: Shared latency budget in milliseconds.

        Returns:
            List of :class:`SLAComplianceResult` sorted by compliance rate
            (highest first — best tracker for the given budget comes first).
        """
        import numpy as np

        compliance_list: List[SLAComplianceResult] = []
        for br in results:
            seqs = br.sequence_results
            if not seqs:
                continue
            means = np.array([s.profiling.latency_mean_ms for s in seqs])
            stds = np.array([s.profiling.latency_std_ms for s in seqs])
            p99s = np.array([s.profiling.latency_p99_ms for s in seqs])
            p95s = np.array([s.profiling.latency_p95_ms for s in seqs])

            # Combined distribution: pool all sequences as one effective profiling.
            from ..profiling.profiler import ProfilingResult
            combined = ProfilingResult(
                tracker_name=br.tracker_name,
                frame_count=sum(s.profiling.frame_count for s in seqs),
                fps=float(np.mean([s.profiling.fps for s in seqs])),
                latency_mean_ms=float(means.mean()),
                latency_std_ms=float(np.sqrt(np.mean(stds ** 2))),  # RMS
                latency_p95_ms=float(p95s.max()),
                latency_p99_ms=float(p99s.max()),
                latency_cv=float(
                    np.sqrt(np.mean(stds ** 2)) / means.mean()
                    if means.mean() > 0 else 0.0
                ),
                peak_memory_mb=br.peak_memory_mb,
            )
            compliance_list.append(self.compute_compliance(combined, budget_ms))

        compliance_list.sort(key=lambda r: r.compliance_rate, reverse=True)
        return compliance_list

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def to_markdown_table(self, results: List[SLAComplianceResult]) -> str:
        """Format an SLA comparison table as a Markdown string.

        Columns: tracker, budget (ms / fps), compliance %, violation %,
        mean latency, p99 latency, budget for 99 %, budget for 99.9 %.

        Args:
            results: Output of :meth:`compare_trackers` or a list built
                from repeated :meth:`compute_compliance` calls.

        Returns:
            Multi-line Markdown table, ready to embed in reports or READMEs.
        """
        header = (
            "| Tracker | Budget (ms) | FPS Target | Compliance | Violations "
            "| Mean Lat. | p99 Lat. | Bgt@99% | Bgt@99.9% |"
        )
        sep = (
            "|---------|------------:|-----------:|-----------:|-----------:"
            "|----------:|---------:|--------:|----------:|"
        )
        lines = [header, sep]
        for r in results:
            lines.append(
                f"| {r.tracker_name} "
                f"| {r.budget_ms:.1f} ms "
                f"| {r.fps_equivalent:.0f} FPS "
                f"| {r.compliance_rate:.1%} "
                f"| {r.violation_rate:.2%} "
                f"| {r.mean_latency_ms:.2f} ms "
                f"| {r.p99_latency_ms:.2f} ms "
                f"| {r.min_budget_99pct_ms:.2f} ms "
                f"| {r.min_budget_999pct_ms:.2f} ms |"
            )
        return "\n".join(lines)

    def deployment_report(
        self,
        results: List[SLAComplianceResult],
        fps_targets: Optional[List[float]] = None,
    ) -> str:
        """Generate a human-readable deployment recommendation report.

        For each FPS target, reports which trackers are compliant (≥ 95 %)
        and flags those that are marginal (90–95 %) or non-compliant (< 90 %).

        Args:
            results: One :class:`SLAComplianceResult` per tracker.
            fps_targets: FPS targets to evaluate (e.g. ``[15, 30, 60]``).
                Default: ``[15, 24, 30, 60]``.

        Returns:
            Multi-line plain-text report string.
        """
        if fps_targets is None:
            fps_targets = [15.0, 24.0, 30.0, 60.0]

        # Map tracker name → SLAComplianceResult for reuse.
        by_name = {r.tracker_name: r for r in results}

        lines = ["## Deployment Feasibility Report", ""]
        for fps in fps_targets:
            budget = 1000.0 / fps
            lines.append(f"### {fps:.0f} FPS (budget = {budget:.1f} ms/frame)")
            compliant = []
            marginal = []
            failing = []
            for name, r in by_name.items():
                # Recompute compliance for THIS budget using the stored distribution.
                mean_ms = r.mean_latency_ms
                # Use the original result's budget to infer distribution params.
                p99_ms = r.p99_latency_ms
                # Quick estimate: scale from stored budget compliance.
                if mean_ms <= 0:
                    continue
                # Recompute from mean+p99 as rough sigma estimate.
                std_est = max((p99_ms - mean_ms) / 2.326, 0.0)  # 2.326 ≈ Φ⁻¹(0.99) for normal
                mu, sigma = _lognormal_params(mean_ms, max(std_est, mean_ms * 0.01))
                if mu is None:
                    rate = 1.0 if mean_ms <= budget else 0.0
                else:
                    rate = _lognormal_cdf(budget, mu, sigma)
                if rate >= 0.95:
                    compliant.append((name, rate))
                elif rate >= 0.90:
                    marginal.append((name, rate))
                else:
                    failing.append((name, rate))

            if compliant:
                lines.append("  Compliant (≥ 95 %):")
                for name, rate in sorted(compliant, key=lambda x: -x[1]):
                    lines.append(f"    ✓ {name:16s}  {rate:.1%}")
            if marginal:
                lines.append("  Marginal (90–95 %):")
                for name, rate in sorted(marginal, key=lambda x: -x[1]):
                    lines.append(f"    ~ {name:16s}  {rate:.1%}")
            if failing:
                lines.append("  Non-compliant (< 90 %):")
                for name, rate in sorted(failing, key=lambda x: -x[1]):
                    lines.append(f"    ✗ {name:16s}  {rate:.1%}")
            lines.append("")
        return "\n".join(lines)

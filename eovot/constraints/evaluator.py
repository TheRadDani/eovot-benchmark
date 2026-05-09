"""Edge constraint evaluator for EOVOT benchmark results.

:class:`ConstraintEvaluator` compares measured profiling data from a
:class:`~eovot.benchmark.engine.BenchmarkResult` against the thresholds
defined in an :class:`~eovot.constraints.profiles.EdgeProfile` and
produces a per-constraint pass/fail report with signed margins.

This answers the core EOVOT research question:
**"Is this tracker deployable on this device?"**

Typical usage::

    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.constraints.evaluator import ConstraintEvaluator
    from eovot.constraints.profiles import RASPBERRY_PI_4, JETSON_NANO

    engine  = BenchmarkEngine(verbose=False, tdp_watts=6.0)
    result  = engine.run(tracker, dataset, dataset_name="OTB100")

    ev = ConstraintEvaluator()
    report = ev.evaluate(result, RASPBERRY_PI_4)
    print(report.summary())
    # → "Overall: DEPLOYABLE" or "NOT DEPLOYABLE"

    table = ev.markdown_table([report], RASPBERRY_PI_4)
    print(table)  # paste into README / paper
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..benchmark.engine import BenchmarkResult
from .profiles import EdgeProfile


@dataclass
class ConstraintCheck:
    """Result of evaluating a single hardware constraint.

    Attributes:
        constraint:  Name of the constraint (e.g. ``"min_fps"``).
        passed:      ``True`` when the tracker satisfies this constraint.
        measured:    Metric value observed in the benchmark.
        limit:       Threshold from the :class:`~eovot.constraints.profiles.EdgeProfile`.
        margin:      Signed headroom.  Positive means passing with room to spare;
                     negative means the constraint is violated by that amount.
                     For lower-bound constraints (FPS) ``margin = measured - limit``.
                     For upper-bound constraints (memory, latency, energy)
                     ``margin = limit - measured``.
    """

    constraint: str
    passed: bool
    measured: float
    limit: float
    margin: float

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.constraint}: "
            f"measured={self.measured:.3f}, "
            f"limit={self.limit:.3f}, "
            f"margin={self.margin:+.3f}"
        )


@dataclass
class ConstraintReport:
    """Complete deployability assessment for one tracker on one edge device.

    Attributes:
        tracker_name: Name of the evaluated tracker.
        profile_name: Name of the edge device profile.
        checks:       Individual :class:`ConstraintCheck` results.
        overall_pass: ``True`` only when *every* checked constraint is satisfied.
        missing_data: Constraints that could not be checked because the
                      corresponding benchmark metric was not collected
                      (e.g. energy requires ``tdp_watts`` to be set).
    """

    tracker_name: str
    profile_name: str
    checks: List[ConstraintCheck] = field(default_factory=list)
    overall_pass: bool = False
    missing_data: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable multi-line summary block."""
        verdict = "DEPLOYABLE" if self.overall_pass else "NOT DEPLOYABLE"
        lines = [
            f"Constraint Report: {self.tracker_name} → {self.profile_name}",
            f"Overall: {verdict}",
            "-" * 55,
        ]
        for check in self.checks:
            lines.append(f"  {check}")
        for name in self.missing_data:
            lines.append(f"  [SKIP] {name}: metric not available")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Serialise to a plain dict for JSON export."""
        return {
            "tracker_name": self.tracker_name,
            "profile_name": self.profile_name,
            "overall_pass": self.overall_pass,
            "checks": [
                {
                    "constraint": c.constraint,
                    "passed": c.passed,
                    "measured": round(c.measured, 4),
                    "limit": round(c.limit, 4),
                    "margin": round(c.margin, 4),
                }
                for c in self.checks
            ],
            "missing_data": self.missing_data,
        }


class ConstraintEvaluator:
    """Assess whether benchmark results satisfy edge deployment constraints.

    Checks four constraints derived from :class:`~eovot.constraints.profiles.EdgeProfile`:

    1. **FPS** — lower bound: tracker must be fast enough for the target FPS.
    2. **Peak memory** — upper bound: must not exceed device RAM budget.
    3. **Latency** — upper bound: derived from FPS (``1000 / fps`` ms per frame).
    4. **Energy per frame** — upper bound: checked only when the profile
       defines ``max_energy_mj_per_frame`` *and* energy was profiled in the
       benchmark (i.e. ``tdp_watts`` was set on the engine).

    Example::

        from eovot.constraints.profiles import RASPBERRY_PI_4, PREDEFINED_PROFILES
        from eovot.constraints.evaluator import ConstraintEvaluator

        ev = ConstraintEvaluator()

        # Single tracker, single device
        report = ev.evaluate(result, RASPBERRY_PI_4)
        print(report.summary())

        # Multiple trackers, multiple devices — returns a 2-D list
        for profile in PREDEFINED_PROFILES.values():
            reports = ev.evaluate_many(results, profile)
            print(ev.markdown_table(reports, profile))
    """

    def evaluate(
        self,
        result: BenchmarkResult,
        profile: EdgeProfile,
    ) -> ConstraintReport:
        """Check one benchmark result against one edge profile.

        Args:
            result:  Aggregated result from
                     :class:`~eovot.benchmark.engine.BenchmarkEngine`.
            profile: Target device constraints.

        Returns:
            :class:`ConstraintReport` with per-constraint pass/fail details
            and an ``overall_pass`` flag.
        """
        report = ConstraintReport(
            tracker_name=result.tracker_name,
            profile_name=profile.name,
        )

        fps = result.mean_fps
        report.checks.append(ConstraintCheck(
            constraint="min_fps",
            passed=fps >= profile.min_fps,
            measured=fps,
            limit=profile.min_fps,
            margin=fps - profile.min_fps,
        ))

        mem = result.peak_memory_mb
        report.checks.append(ConstraintCheck(
            constraint="max_memory_mb",
            passed=mem <= profile.max_memory_mb,
            measured=mem,
            limit=profile.max_memory_mb,
            margin=profile.max_memory_mb - mem,
        ))

        # Latency is derived from FPS; avoids requiring a separate profiling field.
        latency_ms = 1000.0 / fps if fps > 0 else float("inf")
        report.checks.append(ConstraintCheck(
            constraint="max_latency_ms",
            passed=latency_ms <= profile.max_latency_ms,
            measured=latency_ms,
            limit=profile.max_latency_ms,
            margin=profile.max_latency_ms - latency_ms,
        ))

        if profile.max_energy_mj_per_frame is not None:
            energy = result.mean_energy_per_frame_mj
            if energy is not None:
                report.checks.append(ConstraintCheck(
                    constraint="max_energy_mj_per_frame",
                    passed=energy <= profile.max_energy_mj_per_frame,
                    measured=energy,
                    limit=profile.max_energy_mj_per_frame,
                    margin=profile.max_energy_mj_per_frame - energy,
                ))
            else:
                report.missing_data.append(
                    "max_energy_mj_per_frame (enable tdp_watts in BenchmarkEngine)"
                )

        report.overall_pass = all(c.passed for c in report.checks)
        return report

    def evaluate_many(
        self,
        results: List[BenchmarkResult],
        profile: EdgeProfile,
    ) -> List[ConstraintReport]:
        """Evaluate multiple trackers against the same edge profile.

        Args:
            results: One :class:`~eovot.benchmark.engine.BenchmarkResult`
                     per tracker.
            profile: Target device constraints shared by all evaluations.

        Returns:
            List of :class:`ConstraintReport` objects in the same order as
            *results*.
        """
        return [self.evaluate(r, profile) for r in results]

    @staticmethod
    def markdown_table(
        reports: List[ConstraintReport],
        profile: EdgeProfile,
    ) -> str:
        """Format constraint reports as a Markdown table.

        Produces a table suitable for README files, paper appendices, or
        the experiment leaderboard.

        Args:
            reports: Reports from :meth:`evaluate` or :meth:`evaluate_many`.
            profile: The edge profile used (supplies column headers).

        Returns:
            Multi-line Markdown string.
        """
        lines = [
            f"## Edge Constraint Compliance: {profile.name}\n",
            f"*{profile.description}*\n",
            (
                f"| Tracker "
                f"| FPS (≥{profile.min_fps:.0f}) "
                f"| Mem MB (≤{profile.max_memory_mb:.0f}) "
                f"| Latency ms (≤{profile.max_latency_ms:.0f}) "
                + (
                    f"| Energy mJ/fr (≤{profile.max_energy_mj_per_frame:.0f}) "
                    if profile.max_energy_mj_per_frame is not None
                    else ""
                )
                + "| Deployable |"
            ),
            (
                "|---------|-----------|---------|------------|"
                + ("------------|" if profile.max_energy_mj_per_frame is not None else "")
                + "-----------|"
            ),
        ]

        for rpt in reports:
            checks: Dict[str, Optional[ConstraintCheck]] = {
                c.constraint: c for c in rpt.checks
            }

            def _cell(key: str) -> str:
                c = checks.get(key)
                if c is None:
                    return "N/A"
                mark = "✓" if c.passed else "✗"
                return f"{c.measured:.1f} {mark}"

            row = (
                f"| {rpt.tracker_name} "
                f"| {_cell('min_fps')} "
                f"| {_cell('max_memory_mb')} "
                f"| {_cell('max_latency_ms')} "
            )
            if profile.max_energy_mj_per_frame is not None:
                row += f"| {_cell('max_energy_mj_per_frame')} "
            row += f"| {'**YES**' if rpt.overall_pass else 'NO'} |"
            lines.append(row)

        return "\n".join(lines)

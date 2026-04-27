"""Edge deployment feasibility analysis for EOVOT benchmark results.

Given a :class:`~eovot.benchmark.engine.BenchmarkResult` and a
:class:`~eovot.profiling.hardware_profiles.HardwareProfile`, the
:class:`EdgeDeploymentAnalyzer` checks whether the tracker meets the
device's FPS, memory, and (optionally) energy constraints and produces a
structured :class:`EdgeDeploymentReport` with per-constraint pass/fail
scores, margin percentages, and a letter grade.

Typical usage::

    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.profiling.hardware_profiles import get_profile
    from eovot.reporting.edge_report import EdgeDeploymentAnalyzer

    result = BenchmarkEngine(tdp_watts=10.0).run(tracker, dataset)

    analyzer = EdgeDeploymentAnalyzer(energy_budget_mj_per_frame=0.5)
    report   = analyzer.analyze(result, get_profile("jetson-nano"))

    print(report.overall_grade)   # e.g. "B"
    print(report.is_deployable)   # True / False
    print(report.to_markdown())

    # Compare multiple trackers on one device:
    reports = analyzer.compare([result_mosse, result_kcf], get_profile("rpi4"))
    print(EdgeDeploymentAnalyzer.leaderboard(reports))
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..benchmark.engine import BenchmarkResult
from ..profiling.hardware_profiles import HardwareProfile

# Grade thresholds: minimum headroom (%) across all passing constraints
_GRADE_A_MARGIN = 50.0
_GRADE_B_MARGIN = 20.0


@dataclass
class ConstraintScore:
    """Pass/fail verdict and margin for a single deployment constraint.

    Args:
        metric: Human-readable constraint name (e.g., ``"FPS"``).
        required: Threshold value the tracker must meet.
        measured: Value observed in the benchmark result.
        passed: Whether the constraint is satisfied.
        margin_pct: Positive = headroom over the requirement;
            negative = how far the tracker exceeds the limit.
    """

    metric: str
    required: float
    measured: float
    passed: bool
    margin_pct: float

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        sign = "+" if self.margin_pct >= 0 else ""
        return (
            f"[{status}] {self.metric}: "
            f"measured={self.measured:.2f}, "
            f"required={self.required:.2f} "
            f"({sign}{self.margin_pct:.1f}%)"
        )


@dataclass
class EdgeDeploymentReport:
    """Deployment feasibility summary for one tracker / hardware target pair.

    Produced by :meth:`EdgeDeploymentAnalyzer.analyze`.  All constraint
    scores are stored as :class:`ConstraintScore` instances; the
    :attr:`overall_grade` and :attr:`is_deployable` properties aggregate
    them into human-readable verdicts.
    """

    tracker_name: str
    dataset_name: str
    profile: HardwareProfile
    fps_score: ConstraintScore
    memory_score: ConstraintScore
    energy_score: Optional[ConstraintScore]

    @property
    def is_deployable(self) -> bool:
        """``True`` only when *all* active constraints are satisfied."""
        checks = [self.fps_score.passed, self.memory_score.passed]
        if self.energy_score is not None:
            checks.append(self.energy_score.passed)
        return all(checks)

    @property
    def overall_grade(self) -> str:
        """Letter grade summarising deployment readiness.

        * **A** — all constraints pass with ≥50% margin.
        * **B** — all constraints pass with ≥20% margin.
        * **C** — all constraints pass but with thin margin (<20%).
        * **D** — exactly one constraint fails.
        * **F** — two or more constraints fail.
        """
        scores = [s for s in (self.fps_score, self.memory_score, self.energy_score) if s is not None]
        failed = sum(1 for s in scores if not s.passed)

        if failed >= 2:
            return "F"
        if failed == 1:
            return "D"

        min_margin = min(s.margin_pct for s in scores)
        if min_margin >= _GRADE_A_MARGIN:
            return "A"
        if min_margin >= _GRADE_B_MARGIN:
            return "B"
        return "C"

    def summary(self) -> Dict:
        """Return a flat dict suitable for JSON export."""
        d: Dict = {
            "tracker": self.tracker_name,
            "dataset": self.dataset_name,
            "hardware_profile": self.profile.name,
            "deployable": self.is_deployable,
            "grade": self.overall_grade,
            "fps_required": self.profile.target_fps,
            "fps_measured": round(self.fps_score.measured, 2),
            "fps_margin_pct": round(self.fps_score.margin_pct, 1),
            "memory_limit_mb": self.profile.memory_limit_mb,
            "memory_measured_mb": round(self.memory_score.measured, 2),
            "memory_margin_pct": round(self.memory_score.margin_pct, 1),
        }
        if self.energy_score is not None:
            d["energy_budget_mj_per_frame"] = self.energy_score.required
            d["energy_measured_mj_per_frame"] = round(self.energy_score.measured, 4)
            d["energy_margin_pct"] = round(self.energy_score.margin_pct, 1)
        return d

    def to_markdown(self) -> str:
        """Render the full report as a Markdown block."""
        deployable_str = "YES" if self.is_deployable else "NO"
        lines = [
            f"## Edge Deployment Report",
            f"",
            f"**Tracker**: {self.tracker_name}  ",
            f"**Dataset**: {self.dataset_name}  ",
            f"**Target device**: {self.profile.name}  ",
            f"**Device notes**: {self.profile.description}  ",
            f"",
            f"**Deployable**: {deployable_str}  ",
            f"**Overall grade**: {self.overall_grade}",
            f"",
            f"### Constraint Checks",
            f"",
            f"| Constraint | Required | Measured | Margin | Status |",
            f"|-----------|---------|---------|--------|--------|",
        ]
        for score in (self.fps_score, self.memory_score, self.energy_score):
            if score is None:
                continue
            status = "PASS" if score.passed else "FAIL"
            sign = "+" if score.margin_pct >= 0 else ""
            lines.append(
                f"| {score.metric} | {score.required:.2f} | {score.measured:.2f} "
                f"| {sign}{score.margin_pct:.1f}% | {status} |"
            )
        lines.append("")
        return "\n".join(lines)

    def save(self, path: str) -> None:
        """Write the Markdown report to *path*, creating parent dirs if needed."""
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_markdown())


class EdgeDeploymentAnalyzer:
    """Evaluate whether a :class:`~eovot.benchmark.engine.BenchmarkResult`
    meets a :class:`~eovot.profiling.hardware_profiles.HardwareProfile`'s
    deployment constraints.

    Args:
        energy_budget_mj_per_frame: Maximum allowed energy per frame in
            milli-Joules.  When ``None`` (default) no energy constraint is
            applied, even if energy data is present in the result.
    """

    def __init__(self, energy_budget_mj_per_frame: Optional[float] = None) -> None:
        self._energy_budget = energy_budget_mj_per_frame

    def analyze(
        self,
        result: BenchmarkResult,
        profile: HardwareProfile,
    ) -> EdgeDeploymentReport:
        """Produce an :class:`EdgeDeploymentReport` for *result* on *profile*.

        Args:
            result: Output from
                :class:`~eovot.benchmark.engine.BenchmarkEngine`.
            profile: Target hardware specification.

        Returns:
            :class:`EdgeDeploymentReport` with per-constraint pass/fail
            scores, margins, and an overall letter grade.
        """
        return EdgeDeploymentReport(
            tracker_name=result.tracker_name,
            dataset_name=result.dataset_name,
            profile=profile,
            fps_score=self._score_fps(result.mean_fps, profile.target_fps),
            memory_score=self._score_memory(
                result.peak_memory_mb, float(profile.memory_limit_mb)
            ),
            energy_score=self._score_energy(result.mean_energy_per_frame_mj),
        )

    def compare(
        self,
        results: List[BenchmarkResult],
        profile: HardwareProfile,
    ) -> List[EdgeDeploymentReport]:
        """Analyze multiple trackers on one hardware profile.

        Returns reports sorted by grade (A best), then FPS margin descending.

        Args:
            results: List of :class:`~eovot.benchmark.engine.BenchmarkResult`
                objects, one per tracker.
            profile: Shared hardware target.

        Returns:
            Sorted list of :class:`EdgeDeploymentReport` objects.
        """
        reports = [self.analyze(r, profile) for r in results]
        reports.sort(key=lambda r: (_GRADE_ORDER[r.overall_grade], -r.fps_score.margin_pct))
        return reports

    # ------------------------------------------------------------------
    # Per-constraint scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _score_fps(measured: float, required: float) -> ConstraintScore:
        margin_pct = (measured - required) / required * 100.0
        return ConstraintScore(
            metric="FPS",
            required=required,
            measured=measured,
            passed=measured >= required,
            margin_pct=margin_pct,
        )

    @staticmethod
    def _score_memory(measured_mb: float, limit_mb: float) -> ConstraintScore:
        margin_pct = (limit_mb - measured_mb) / limit_mb * 100.0
        return ConstraintScore(
            metric="Memory (MiB)",
            required=limit_mb,
            measured=measured_mb,
            passed=measured_mb <= limit_mb,
            margin_pct=margin_pct,
        )

    def _score_energy(self, measured_mj: Optional[float]) -> Optional[ConstraintScore]:
        if measured_mj is None or self._energy_budget is None:
            return None
        margin_pct = (self._energy_budget - measured_mj) / self._energy_budget * 100.0
        return ConstraintScore(
            metric="Energy (mJ/frame)",
            required=self._energy_budget,
            measured=measured_mj,
            passed=measured_mj <= self._energy_budget,
            margin_pct=margin_pct,
        )

    # ------------------------------------------------------------------
    # Leaderboard helper
    # ------------------------------------------------------------------

    @staticmethod
    def leaderboard(reports: List[EdgeDeploymentReport]) -> str:
        """Render a Markdown leaderboard table from a list of reports.

        Reports are sorted by grade (A → F), then FPS margin descending.

        Args:
            reports: List of :class:`EdgeDeploymentReport` objects,
                typically from :meth:`compare`.

        Returns:
            Multi-line Markdown string containing the leaderboard table.
        """
        sorted_reports = sorted(
            reports,
            key=lambda r: (_GRADE_ORDER[r.overall_grade], -r.fps_score.margin_pct),
        )
        lines = [
            "| Rank | Tracker | FPS | FPS Margin | Memory (MiB) | Memory Margin | Grade | Deployable |",
            "|-----|--------|-----|-----------|-------------|--------------|-------|-----------|",
        ]
        for rank, r in enumerate(sorted_reports, start=1):
            fps_sign = "+" if r.fps_score.margin_pct >= 0 else ""
            mem_sign = "+" if r.memory_score.margin_pct >= 0 else ""
            deployable = "YES" if r.is_deployable else "NO"
            lines.append(
                f"| {rank} "
                f"| {r.tracker_name} "
                f"| {r.fps_score.measured:.1f} "
                f"| {fps_sign}{r.fps_score.margin_pct:.1f}% "
                f"| {r.memory_score.measured:.0f} "
                f"| {mem_sign}{r.memory_score.margin_pct:.1f}% "
                f"| {r.overall_grade} "
                f"| {deployable} |"
            )
        return "\n".join(lines)


_GRADE_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}

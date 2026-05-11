"""Hardware-aware deployment advisor for tracker selection.

Given one or more :class:`~eovot.benchmark.engine.BenchmarkResult` objects and
a set of :class:`~eovot.profiling.hardware_profiles.HardwareProfile` targets,
the :class:`DeploymentAdvisor` scores each (tracker, device) pair against four
deployment constraints:

- **FPS**: measured mean FPS vs device target FPS
- **Latency**: measured mean latency vs device latency budget
- **Memory**: measured peak memory vs device RAM limit
- **Power**: measured mean power (if energy was profiled) vs device power budget

Each constraint is scored 0–1, where 1 means the constraint is comfortably met.
A tracker is ``deployable`` on a device only when *all hard constraints* are
satisfied (FPS ≥ target, latency ≤ budget, memory ≤ RAM).  Power is treated as
a soft constraint that lowers the overall score but does not block deployment.

Usage::

    from eovot.profiling.deployment_advisor import DeploymentAdvisor
    from eovot.profiling.hardware_profiles import PROFILES

    advisor = DeploymentAdvisor()
    scores = advisor.rank(benchmark_results, PROFILES["jetson_nano"])
    print(advisor.report_markdown(scores, PROFILES["jetson_nano"]))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .hardware_profiles import HardwareProfile


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ConstraintScore:
    """Score for a single deployment constraint.

    Attributes:
        name:      Constraint name (e.g. ``"fps"``).
        measured:  Observed value from benchmark.
        required:  Device constraint threshold.
        score:     Normalised score in ``[0, 1]``.
        passed:    Whether the hard constraint is satisfied.
    """

    name: str
    measured: float
    required: float
    score: float
    passed: bool

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "measured": round(self.measured, 4),
            "required": round(self.required, 4),
            "score": round(self.score, 4),
            "passed": self.passed,
        }


@dataclass
class DeploymentScore:
    """Deployment evaluation for one (tracker, device) pair.

    Attributes:
        tracker_name:          Name of the tracker.
        profile_name:          Name of the hardware profile.
        constraints:           Individual constraint scores.
        overall_score:         Weighted aggregate score in ``[0, 1]``.
        deployable:            True when all hard constraints are met.
        violations:            List of constraint names that failed.
        recommendation:        Human-readable deployment verdict.
    """

    tracker_name: str
    profile_name: str
    constraints: List[ConstraintScore] = field(default_factory=list)
    overall_score: float = 0.0
    deployable: bool = False
    violations: List[str] = field(default_factory=list)
    recommendation: str = ""

    def constraint(self, name: str) -> Optional[ConstraintScore]:
        for c in self.constraints:
            if c.name == name:
                return c
        return None

    def to_dict(self) -> Dict:
        return {
            "tracker_name": self.tracker_name,
            "profile_name": self.profile_name,
            "overall_score": round(self.overall_score, 4),
            "deployable": self.deployable,
            "violations": self.violations,
            "recommendation": self.recommendation,
            "constraints": [c.to_dict() for c in self.constraints],
        }


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _fps_score(measured_fps: float, target_fps: float) -> ConstraintScore:
    """Higher FPS is better; score = min(measured/target, 1)."""
    ratio = measured_fps / target_fps if target_fps > 0 else 0.0
    score = min(ratio, 1.0)
    return ConstraintScore(
        name="fps",
        measured=measured_fps,
        required=target_fps,
        score=score,
        passed=measured_fps >= target_fps,
    )


def _latency_score(measured_ms: float, budget_ms: float) -> ConstraintScore:
    """Lower latency is better; score = min(budget/measured, 1)."""
    if measured_ms <= 0:
        score, passed = 1.0, True
    else:
        ratio = budget_ms / measured_ms
        score = min(ratio, 1.0)
        passed = measured_ms <= budget_ms
    return ConstraintScore(
        name="latency",
        measured=measured_ms,
        required=budget_ms,
        score=score,
        passed=passed,
    )


def _memory_score(measured_mb: float, limit_mb: float) -> ConstraintScore:
    """Lower memory is better; score = min(limit/measured, 1)."""
    if measured_mb <= 0:
        score, passed = 1.0, True
    else:
        ratio = limit_mb / measured_mb
        score = min(ratio, 1.0)
        passed = measured_mb <= limit_mb
    return ConstraintScore(
        name="memory",
        measured=measured_mb,
        required=limit_mb,
        score=score,
        passed=passed,
    )


def _power_score(
    mean_power_w: Optional[float], budget_w: float
) -> Optional[ConstraintScore]:
    """Returns None if energy data is unavailable."""
    if mean_power_w is None:
        return None
    if mean_power_w <= 0:
        return ConstraintScore(
            name="power", measured=0.0, required=budget_w,
            score=1.0, passed=True,
        )
    ratio = budget_w / mean_power_w
    score = min(ratio, 1.0)
    return ConstraintScore(
        name="power",
        measured=mean_power_w,
        required=budget_w,
        score=score,
        passed=mean_power_w <= budget_w,
    )


# Weights for the overall score aggregation
_WEIGHTS: Dict[str, float] = {
    "fps": 0.35,
    "latency": 0.35,
    "memory": 0.20,
    "power": 0.10,
}


def _overall_score(constraints: List[ConstraintScore]) -> float:
    """Weighted average of available constraint scores."""
    total_weight = 0.0
    weighted_sum = 0.0
    available = {c.name for c in constraints}
    for c in constraints:
        w = _WEIGHTS.get(c.name, 0.0)
        if c.name == "power" and "power" not in available:
            continue
        weighted_sum += w * c.score
        total_weight += w
    if total_weight == 0:
        return 0.0
    # Re-normalise if power is absent so remaining weights sum to 1
    return weighted_sum / total_weight


def _recommendation(score: DeploymentScore, profile: HardwareProfile) -> str:
    if score.deployable:
        if score.overall_score >= 0.85:
            return (
                f"Highly recommended for {profile.display_name}. "
                "All constraints met with comfortable margin."
            )
        elif score.overall_score >= 0.60:
            return (
                f"Suitable for {profile.display_name}. "
                "Constraints met; some headroom is limited."
            )
        else:
            return (
                f"Marginally deployable on {profile.display_name}. "
                "Constraints met but operating near limits — test on device."
            )
    else:
        viol_str = ", ".join(score.violations)
        return (
            f"Not recommended for {profile.display_name}. "
            f"Constraint(s) violated: {viol_str}."
        )


# ---------------------------------------------------------------------------
# DeploymentAdvisor
# ---------------------------------------------------------------------------


class DeploymentAdvisor:
    """Score and rank trackers against hardware deployment constraints.

    Args:
        memory_safety_factor: Fraction of device RAM treated as usable
            (default ``0.80`` — leaves 20% for OS and system processes).

    Example::

        advisor = DeploymentAdvisor()
        scores = advisor.rank(results, PROFILES["raspberry_pi_4"])
        for s in scores:
            print(s.tracker_name, s.overall_score, s.deployable)
        print(advisor.report_markdown(scores, PROFILES["raspberry_pi_4"]))
    """

    def __init__(self, memory_safety_factor: float = 0.80) -> None:
        if not 0 < memory_safety_factor <= 1.0:
            raise ValueError(
                f"memory_safety_factor must be in (0, 1], got {memory_safety_factor}"
            )
        self.memory_safety_factor = memory_safety_factor

    def score(
        self,
        result,  # BenchmarkResult — avoid circular import
        profile: HardwareProfile,
    ) -> DeploymentScore:
        """Evaluate a single benchmark result against a hardware profile.

        Args:
            result:  A :class:`~eovot.benchmark.engine.BenchmarkResult`.
            profile: Target :class:`HardwareProfile`.

        Returns:
            :class:`DeploymentScore` with per-constraint breakdown.
        """
        measured_fps = result.mean_fps
        measured_latency = np.mean(
            [sr.profiling.latency_mean_ms for sr in result.sequence_results]
        )
        measured_memory = result.peak_memory_mb
        usable_memory = profile.memory_mb * self.memory_safety_factor

        # Derive mean power from energy data if available
        mean_power: Optional[float] = None
        with_energy = [
            sr.energy for sr in result.sequence_results if sr.energy is not None
        ]
        if with_energy:
            mean_power = float(np.mean([e.mean_power_w for e in with_energy]))

        # Compute per-constraint scores
        c_fps = _fps_score(measured_fps, profile.target_fps)
        c_lat = _latency_score(float(measured_latency), profile.latency_budget_ms)
        c_mem = _memory_score(measured_memory, usable_memory)
        constraints: List[ConstraintScore] = [c_fps, c_lat, c_mem]

        c_pwr = _power_score(mean_power, profile.power_budget_w)
        if c_pwr is not None:
            constraints.append(c_pwr)

        # Hard violations
        hard = [c_fps, c_lat, c_mem]
        violations = [c.name for c in hard if not c.passed]
        deployable = len(violations) == 0

        overall = _overall_score(constraints)

        ds = DeploymentScore(
            tracker_name=result.tracker_name,
            profile_name=profile.name,
            constraints=constraints,
            overall_score=overall,
            deployable=deployable,
            violations=violations,
        )
        ds.recommendation = _recommendation(ds, profile)
        return ds

    def rank(
        self,
        results: List,  # List[BenchmarkResult]
        profile: HardwareProfile,
    ) -> List[DeploymentScore]:
        """Score and sort all trackers for a given device.

        Deployable trackers are ranked by overall score (descending), followed
        by non-deployable trackers.

        Args:
            results: List of :class:`~eovot.benchmark.engine.BenchmarkResult`.
            profile: Target :class:`HardwareProfile`.

        Returns:
            Sorted list of :class:`DeploymentScore` objects.
        """
        scores = [self.score(r, profile) for r in results]
        deployable = sorted(
            [s for s in scores if s.deployable],
            key=lambda s: s.overall_score,
            reverse=True,
        )
        not_deployable = sorted(
            [s for s in scores if not s.deployable],
            key=lambda s: s.overall_score,
            reverse=True,
        )
        return deployable + not_deployable

    def multi_profile_summary(
        self,
        results: List,  # List[BenchmarkResult]
        profiles: List[HardwareProfile],
    ) -> Dict[str, List[DeploymentScore]]:
        """Rank all trackers against multiple profiles.

        Args:
            results:  List of benchmark results.
            profiles: List of target hardware profiles.

        Returns:
            Mapping from profile name to ranked list of deployment scores.
        """
        return {p.name: self.rank(results, p) for p in profiles}

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report_markdown(
        self,
        scores: List[DeploymentScore],
        profile: HardwareProfile,
    ) -> str:
        """Render a deployment report for one device as Markdown.

        Args:
            scores:  Ranked list returned by :meth:`rank`.
            profile: The hardware profile used.

        Returns:
            Markdown string suitable for README sections or wiki pages.
        """
        lines = [
            f"## Deployment Report: {profile.display_name}\n",
            f"> {profile.description}  ",
            f"> TDP: {profile.tdp_watts} W | RAM: {profile.memory_mb} MB | "
            f"Target FPS: {profile.target_fps} | Latency budget: "
            f"{profile.latency_budget_ms} ms\n",
            "### Tracker Rankings\n",
            "| Rank | Tracker | Score | FPS | Latency (ms) | Memory (MB) | Deployable | Violations |",
            "|------|---------|-------|-----|--------------|-------------|------------|------------|",
        ]

        for rank, s in enumerate(scores, 1):
            dep = "✓" if s.deployable else "✗"
            viol = ", ".join(s.violations) if s.violations else "—"
            c_fps = s.constraint("fps")
            c_lat = s.constraint("latency")
            c_mem = s.constraint("memory")
            fps_str = f"{c_fps.measured:.1f}" if c_fps else "—"
            lat_str = f"{c_lat.measured:.1f}" if c_lat else "—"
            mem_str = f"{c_mem.measured:.1f}" if c_mem else "—"
            lines.append(
                f"| {rank} | {s.tracker_name} | {s.overall_score:.3f} "
                f"| {fps_str} | {lat_str} | {mem_str} | {dep} | {viol} |"
            )

        lines.append("\n### Recommendations\n")
        for s in scores:
            icon = "✅" if s.deployable else "❌"
            lines.append(f"- {icon} **{s.tracker_name}**: {s.recommendation}")

        return "\n".join(lines)

    def report_multi_profile_markdown(
        self,
        summary: Dict[str, List[DeploymentScore]],
        profiles: List[HardwareProfile],
    ) -> str:
        """Render a cross-device compatibility matrix as Markdown.

        Rows are trackers; columns are devices.  Cells show the overall score
        and a ✓/✗ deployability indicator.

        Args:
            summary:  Output of :meth:`multi_profile_summary`.
            profiles: Profiles in the desired column order.

        Returns:
            Markdown string with the compatibility matrix.
        """
        # Collect all tracker names (preserve per-device ranking order)
        seen: Dict[str, None] = {}
        for scores in summary.values():
            for s in scores:
                seen[s.tracker_name] = None
        tracker_names = list(seen.keys())

        # Header
        profile_headers = " | ".join(p.display_name for p in profiles)
        sep_cols = " | ".join(["---"] * len(profiles))
        lines = [
            "## Cross-Device Deployment Compatibility\n",
            f"| Tracker | {profile_headers} |",
            f"|---------|{sep_cols}|",
        ]

        for tracker in tracker_names:
            cells: List[str] = []
            for p in profiles:
                p_scores = summary.get(p.name, [])
                match = next((s for s in p_scores if s.tracker_name == tracker), None)
                if match is None:
                    cells.append("—")
                elif match.deployable:
                    cells.append(f"✓ {match.overall_score:.2f}")
                else:
                    cells.append(f"✗ {match.overall_score:.2f}")
            lines.append(f"| {tracker} | " + " | ".join(cells) + " |")

        return "\n".join(lines)

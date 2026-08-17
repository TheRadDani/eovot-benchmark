"""Deployment readiness checker for EOVOT benchmark results.

Bridges the gap between benchmark numbers and real-world deployment decisions.
Given a :class:`~eovot.benchmark.engine.BenchmarkResult` and a
:class:`DeploymentProfile` that captures the requirements of a target
deployment scenario, :class:`DeploymentChecker` produces a
:class:`DeploymentReport` with:

- Per-metric GO / NO-GO assessment and headroom ratio
- Aggregate deployment risk score (0 = safe, 1 = undeployable)
- Ranked list of bottleneck metrics (worst violations first)
- Human-readable Markdown summary ready for research papers or READMEs

Typical usage::

    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.kcf import KCFTracker
    from eovot.analysis.deployment import DeploymentChecker, DeploymentProfile

    engine  = BenchmarkEngine(verbose=False)
    dataset = SyntheticDataset(num_sequences=5, num_frames=100)
    result  = engine.run(KCFTracker(), dataset, dataset_name="Synthetic")

    profile = DeploymentProfile(
        target_fps=30.0,
        memory_budget_mb=256.0,
        max_latency_mean_ms=33.3,
        max_latency_p99_ms=50.0,
        min_iou=0.5,
        device_name="Raspberry Pi 4",
    )

    checker = DeploymentChecker()
    report  = checker.check(result, profile)
    print(report)
    print(report.to_markdown())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult


@dataclass
class DeploymentProfile:
    """Target hardware and SLA requirements for a deployment scenario.

    All thresholds are **requirements** — a tracker must satisfy every
    configured requirement to receive a GO verdict.  ``None`` means the
    requirement is not configured and will not be checked.

    Args:
        target_fps: Minimum acceptable mean FPS (e.g. ``30.0``).
        memory_budget_mb: Maximum acceptable peak memory usage in MB.
        max_latency_mean_ms: Maximum acceptable mean per-frame latency (ms).
        max_latency_p99_ms: Maximum acceptable 99th-percentile latency (ms);
            enforces tail-latency SLAs critical for real-time systems.
        min_iou: Minimum acceptable mean IoU across all evaluated sequences.
        min_success_auc: Minimum acceptable mean success-curve AUC.
        device_name: Optional human-readable label for the deployment target
            (e.g. ``"Jetson Nano"``).  Used in reports only.
    """

    target_fps: Optional[float] = None
    memory_budget_mb: Optional[float] = None
    max_latency_mean_ms: Optional[float] = None
    max_latency_p99_ms: Optional[float] = None
    min_iou: Optional[float] = None
    min_success_auc: Optional[float] = None
    device_name: str = "unspecified"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_name": self.device_name,
            "target_fps": self.target_fps,
            "memory_budget_mb": self.memory_budget_mb,
            "max_latency_mean_ms": self.max_latency_mean_ms,
            "max_latency_p99_ms": self.max_latency_p99_ms,
            "min_iou": self.min_iou,
            "min_success_auc": self.min_success_auc,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DeploymentProfile":
        return cls(
            target_fps=d.get("target_fps"),
            memory_budget_mb=d.get("memory_budget_mb"),
            max_latency_mean_ms=d.get("max_latency_mean_ms"),
            max_latency_p99_ms=d.get("max_latency_p99_ms"),
            min_iou=d.get("min_iou"),
            min_success_auc=d.get("min_success_auc"),
            device_name=d.get("device_name", "unspecified"),
        )


@dataclass
class MetricAssessment:
    """Assessment of one metric against its deployment requirement.

    Attributes:
        name: Human-readable metric label (e.g. ``"Mean FPS"``).
        required: Required threshold value from the :class:`DeploymentProfile`.
        actual: Observed value from the :class:`~eovot.benchmark.engine.BenchmarkResult`.
        passes: ``True`` if the actual value satisfies the requirement.
        higher_is_better: ``True`` for FPS / IoU; ``False`` for latency / memory.
        headroom: Signed ratio measuring margin beyond the requirement.
            Positive = spare capacity; negative = deficit.

            For higher-is-better metrics: ``(actual - required) / required``
            For lower-is-better metrics: ``(required - actual) / required``
    """

    name: str
    required: float
    actual: float
    passes: bool
    higher_is_better: bool
    headroom: float

    def __str__(self) -> str:
        status = "PASS" if self.passes else "FAIL"
        sign = "+" if self.headroom >= 0 else ""
        return (
            f"[{status}] {self.name}: actual={self.actual:.3g}  "
            f"required={'≥' if self.higher_is_better else '≤'}{self.required:.3g}  "
            f"headroom={sign}{self.headroom:.1%}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "required": self.required,
            "actual": self.actual,
            "passes": self.passes,
            "higher_is_better": self.higher_is_better,
            "headroom": round(self.headroom, 4),
        }


@dataclass
class DeploymentReport:
    """Full deployment readiness report for one tracker / deployment-profile pair.

    Attributes:
        tracker_name: Name of the evaluated tracker.
        dataset_name: Name of the dataset used for evaluation.
        profile: The :class:`DeploymentProfile` against which the tracker was assessed.
        assessments: Per-metric assessments, ranked by headroom (worst first).
        deployable: ``True`` iff all metric assessments pass.
        risk_score: Scalar in ``[0, 1]`` summarising deployment risk.
            ``0.0`` means all requirements are satisfied with large margins.
            ``1.0`` means at least one metric is critically insufficient.
            Computed as the mean of clipped deficit fractions across failing
            metrics; passing metrics contribute 0.
        bottlenecks: Subset of ``assessments`` that do not pass, ordered by
            severity (largest deficit first).
    """

    tracker_name: str
    dataset_name: str
    profile: DeploymentProfile
    assessments: List[MetricAssessment] = field(default_factory=list)
    deployable: bool = False
    risk_score: float = 0.0
    bottlenecks: List[MetricAssessment] = field(default_factory=list)

    def __str__(self) -> str:
        verdict = "GO" if self.deployable else "NO-GO"
        lines = [
            f"DeploymentReport [{verdict}] — {self.tracker_name} on {self.dataset_name}",
            f"  Target device : {self.profile.device_name}",
            f"  Risk score    : {self.risk_score:.3f}  (0=safe, 1=undeployable)",
            "  Assessments:",
        ]
        for a in self.assessments:
            lines.append(f"    {a}")
        if self.bottlenecks:
            lines.append("  Bottlenecks (worst first):")
            for b in self.bottlenecks:
                lines.append(f"    ► {b.name}  deficit={abs(b.headroom):.1%}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the full report to a JSON-compatible dict."""
        return {
            "tracker_name": self.tracker_name,
            "dataset_name": self.dataset_name,
            "profile": self.profile.to_dict(),
            "deployable": self.deployable,
            "risk_score": round(self.risk_score, 4),
            "assessments": [a.to_dict() for a in self.assessments],
            "bottlenecks": [b.name for b in self.bottlenecks],
        }

    def save(self, path: str) -> Path:
        """Write the report to a JSON file.

        Args:
            path: Destination path.  A ``.json`` extension is appended when
                the path has none.

        Returns:
            Resolved :class:`pathlib.Path` of the written file.
        """
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".json")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return p

    def to_markdown(self) -> str:
        """Render the report as a Markdown string.

        Returns a block suitable for embedding in a README, GitHub issue, or
        research paper supplementary materials section.

        Example output::

            ## Deployment Report: KCF on Synthetic → Raspberry Pi 4

            **Verdict:** ✅ GO  |  **Risk score:** 0.000

            | Metric | Required | Actual | Status | Headroom |
            |--------|---------|--------|--------|---------|
            | Mean FPS | ≥ 30.0 | 158.3 | ✅ | +427.7% |
            ...
        """
        verdict_icon = "✅ GO" if self.deployable else "❌ NO-GO"
        lines = [
            f"## Deployment Report: {self.tracker_name} on {self.dataset_name}"
            f" → {self.profile.device_name}",
            "",
            f"**Verdict:** {verdict_icon}  |  **Risk score:** {self.risk_score:.3f}",
            "",
            "| Metric | Required | Actual | Status | Headroom |",
            "|--------|---------|--------|:------:|--------:|",
        ]
        for a in self.assessments:
            icon = "✅" if a.passes else "❌"
            op = "≥" if a.higher_is_better else "≤"
            sign = "+" if a.headroom >= 0 else ""
            lines.append(
                f"| {a.name} | {op} {a.required:.3g} | {a.actual:.3g} "
                f"| {icon} | {sign}{a.headroom:.1%} |"
            )
        if self.bottlenecks:
            lines += [
                "",
                "### Bottlenecks (action required)",
                "",
            ]
            for b in self.bottlenecks:
                deficit = abs(b.headroom)
                op = "≥" if b.higher_is_better else "≤"
                lines.append(
                    f"- **{b.name}**: actual `{b.actual:.3g}` vs required `{op} {b.required:.3g}` "
                    f"— deficit **{deficit:.1%}**"
                )
        return "\n".join(lines)


class DeploymentChecker:
    """Evaluate a :class:`~eovot.benchmark.engine.BenchmarkResult` against a
    :class:`DeploymentProfile` and produce a :class:`DeploymentReport`.

    The checker is stateless — call :meth:`check` repeatedly with different
    results or profiles.

    Example::

        checker = DeploymentChecker()
        report  = checker.check(benchmark_result, profile)
        if not report.deployable:
            print("Cannot deploy:", [b.name for b in report.bottlenecks])
    """

    def check(
        self,
        result: "BenchmarkResult",
        profile: DeploymentProfile,
    ) -> DeploymentReport:
        """Assess *result* against *profile* and return a :class:`DeploymentReport`.

        Args:
            result:  A :class:`~eovot.benchmark.engine.BenchmarkResult` produced
                by :class:`~eovot.benchmark.engine.BenchmarkEngine`.
            profile: Target deployment requirements.

        Returns:
            :class:`DeploymentReport` with per-metric assessments, a GO/NO-GO
            verdict, risk score, and ranked bottleneck list.
        """
        assessments: List[MetricAssessment] = []

        # FPS
        if profile.target_fps is not None:
            assessments.append(
                self._assess(
                    name="Mean FPS",
                    required=profile.target_fps,
                    actual=result.mean_fps,
                    higher_is_better=True,
                )
            )

        # Peak memory
        if profile.memory_budget_mb is not None:
            assessments.append(
                self._assess(
                    name="Peak Memory (MB)",
                    required=profile.memory_budget_mb,
                    actual=result.peak_memory_mb,
                    higher_is_better=False,
                )
            )

        # Mean latency
        if profile.max_latency_mean_ms is not None:
            mean_lat = float(
                sum(r.profiling.latency_mean_ms for r in result.sequence_results)
                / len(result.sequence_results)
            )
            assessments.append(
                self._assess(
                    name="Mean Latency (ms)",
                    required=profile.max_latency_mean_ms,
                    actual=mean_lat,
                    higher_is_better=False,
                )
            )

        # p99 latency
        if profile.max_latency_p99_ms is not None:
            p99_vals = [r.profiling.latency_p99_ms for r in result.sequence_results]
            max_p99 = float(max(p99_vals)) if p99_vals else 0.0
            assessments.append(
                self._assess(
                    name="p99 Latency (ms)",
                    required=profile.max_latency_p99_ms,
                    actual=max_p99,
                    higher_is_better=False,
                )
            )

        # Mean IoU
        if profile.min_iou is not None:
            assessments.append(
                self._assess(
                    name="Mean IoU",
                    required=profile.min_iou,
                    actual=result.mean_iou,
                    higher_is_better=True,
                )
            )

        # Success AUC
        if profile.min_success_auc is not None:
            sauc = result.mean_success_auc
            if sauc is None:
                sauc = 0.0
            assessments.append(
                self._assess(
                    name="Success AUC",
                    required=profile.min_success_auc,
                    actual=sauc,
                    higher_is_better=True,
                )
            )

        # Sort assessments: failing first, then by absolute headroom ascending
        assessments.sort(key=lambda a: (a.passes, a.headroom))

        deployable = all(a.passes for a in assessments)
        bottlenecks = [a for a in assessments if not a.passes]
        risk_score = self._compute_risk(bottlenecks)

        return DeploymentReport(
            tracker_name=result.tracker_name,
            dataset_name=result.dataset_name,
            profile=profile,
            assessments=assessments,
            deployable=deployable,
            risk_score=risk_score,
            bottlenecks=bottlenecks,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _assess(
        name: str,
        required: float,
        actual: float,
        higher_is_better: bool,
    ) -> MetricAssessment:
        if required == 0:
            headroom = 0.0
            passes = actual >= 0 if higher_is_better else actual <= 0
        elif higher_is_better:
            passes = actual >= required
            headroom = (actual - required) / required
        else:
            passes = actual <= required
            headroom = (required - actual) / required
        return MetricAssessment(
            name=name,
            required=required,
            actual=actual,
            passes=passes,
            higher_is_better=higher_is_better,
            headroom=headroom,
        )

    @staticmethod
    def _compute_risk(bottlenecks: List[MetricAssessment]) -> float:
        """Risk score in [0, 1]: mean clipped deficit fraction across failing metrics.

        A deficit fraction of -0.5 (50 % short of required) is clipped to 1.0 at
        the extreme — a tracker that achieves zero FPS on a 30 FPS target is
        undeployable (risk = 1.0), not just risky.
        """
        if not bottlenecks:
            return 0.0
        deficits = [min(1.0, abs(b.headroom)) for b in bottlenecks]
        return float(sum(deficits) / len(deficits))

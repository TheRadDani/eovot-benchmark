"""Deployment Suitability Scorer for edge-device tracker selection.

Projects host-machine benchmark results onto a target edge device via
:class:`~eovot.profiling.device_sim.DeviceSimulator`, then scores each
tracker across four deployment-relevant dimensions:

* **Accuracy** — mean IoU (or success AUC when available)
* **Throughput** — estimated FPS relative to a configurable minimum and target
* **Memory** — headroom within the device's RAM ceiling
* **Energy** — estimated millijoules per frame relative to target throughput

All dimension scores are normalised to ``[0, 1]`` and combined via a
user-configurable weight dict into a single composite ``total_score``.
Hard constraints (OOM, below-minimum FPS) disqualify a tracker entirely;
disqualified trackers still receive a score so that *partial* deployments
can be ranked when no candidate fully satisfies the constraints.

Example::

    from eovot.deployment.scorer import DeploymentConstraints, DeploymentSuitabilityScorer

    # Require ≥ 15 FPS, target 30 FPS, on a Raspberry Pi 4
    constraints = DeploymentConstraints(min_fps=15.0, target_fps=30.0)
    scorer = DeploymentSuitabilityScorer(device="rpi4", constraints=constraints)

    scores = scorer.rank([result_mosse, result_kcf, result_csrt])
    for s in scores:
        print(s)

    print(scorer.to_markdown_table(scores))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..benchmark.engine import BenchmarkResult
from ..profiling.device_sim import DeviceSimulator
from ..profiling.profiler import ProfilingResult


@dataclass
class DeploymentConstraints:
    """Hard and soft constraints that define a deployment scenario.

    Attributes:
        min_fps: Hard minimum FPS on the target device.  Trackers below this
            threshold are disqualified.  Default: ``10.0``.
        target_fps: Soft FPS target used for throughput and energy scoring.
            A tracker achieving this FPS earns full marks on both dimensions.
            Defaults to ``3 × min_fps`` when ``None``.
        max_memory_mb: Optional hard memory ceiling in megabytes.  When set,
            overrides the device's physical RAM limit for disqualification
            (useful when an OS or co-process consumes part of device RAM).
            Defaults to ``None`` (use device limit).
        min_accuracy: Soft accuracy lower bound (mean IoU or success AUC).
            Trackers below this value receive a zero accuracy score but are
            *not* disqualified.  Default: ``0.0``.
        sustained_seconds: Duration of continuous tracking used by the thermal
            model in :class:`~eovot.profiling.device_sim.DeviceSimulator`.
            ``0.0`` (default) models a cold-start burst scenario.
        weights: Per-dimension scoring weights.  Keys: ``"accuracy"``,
            ``"throughput"``, ``"memory"``, ``"energy"``.  Weights are
            normalised to sum to 1.0 internally.

    Example::

        # Latency-critical UAV scenario: prioritise speed over accuracy
        c = DeploymentConstraints(
            min_fps=20.0,
            target_fps=30.0,
            weights={"accuracy": 0.20, "throughput": 0.50, "memory": 0.20, "energy": 0.10},
        )
    """

    min_fps: float = 10.0
    target_fps: Optional[float] = None
    max_memory_mb: Optional[float] = None
    min_accuracy: float = 0.0
    sustained_seconds: float = 0.0
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "accuracy": 0.35,
            "throughput": 0.30,
            "memory": 0.20,
            "energy": 0.15,
        }
    )

    def effective_target_fps(self) -> float:
        """Return ``target_fps`` or ``3 × min_fps`` if not explicitly set."""
        return self.target_fps if self.target_fps is not None else 3.0 * self.min_fps


@dataclass
class DeploymentScore:
    """Scored suitability of a single tracker for a specific edge device.

    Attributes:
        tracker_name: Name of the evaluated tracker.
        device_name: Short device key (e.g. ``"rpi4"``).
        total_score: Weighted composite score in ``[0, 1]``.  Higher is better.
        deployment_ready: ``True`` when all hard constraints are satisfied
            (memory fits, FPS above minimum).
        disqualified: ``True`` when any hard constraint is violated.
        disqualify_reason: Human-readable explanation, or ``""`` if not disqualified.
        accuracy_score: Normalised accuracy component in ``[0, 1]``.
        throughput_score: Normalised throughput component in ``[0, 1]``.
        memory_score: Normalised memory-headroom component in ``[0, 1]``.
        energy_score: Normalised energy-efficiency component in ``[0, 1]``.
        mean_accuracy: Raw accuracy value used for scoring (IoU or success AUC).
        estimated_fps: Projected FPS on the target device.
        estimated_memory_mb: Expected peak memory on the device.
        estimated_energy_mj_per_frame: Projected energy cost per frame.
        thermal_state: Thermal condition — ``"nominal"``, ``"transitioning"``,
            or ``"throttled"``.
    """

    tracker_name: str
    device_name: str
    total_score: float
    deployment_ready: bool
    disqualified: bool
    disqualify_reason: str
    accuracy_score: float
    throughput_score: float
    memory_score: float
    energy_score: float
    mean_accuracy: float
    estimated_fps: float
    estimated_memory_mb: float
    estimated_energy_mj_per_frame: float
    thermal_state: str

    def __str__(self) -> str:
        ready = "READY" if self.deployment_ready else f"DQ({self.disqualify_reason})"
        return (
            f"DeploymentScore[{self.tracker_name} → {self.device_name}]  "
            f"total={self.total_score:.3f}  "
            f"acc={self.accuracy_score:.3f}  "
            f"fps={self.throughput_score:.3f}({self.estimated_fps:.1f} FPS)  "
            f"mem={self.memory_score:.3f}({self.estimated_memory_mb:.0f} MB)  "
            f"energy={self.energy_score:.3f}  "
            f"thermal={self.thermal_state}  "
            f"[{ready}]"
        )

    def to_dict(self) -> Dict:
        """Return a plain dict suitable for JSON serialisation."""
        return {
            "tracker_name": self.tracker_name,
            "device_name": self.device_name,
            "total_score": round(self.total_score, 4),
            "deployment_ready": self.deployment_ready,
            "disqualified": self.disqualified,
            "disqualify_reason": self.disqualify_reason,
            "accuracy_score": round(self.accuracy_score, 4),
            "throughput_score": round(self.throughput_score, 4),
            "memory_score": round(self.memory_score, 4),
            "energy_score": round(self.energy_score, 4),
            "mean_accuracy": round(self.mean_accuracy, 4),
            "estimated_fps": round(self.estimated_fps, 2),
            "estimated_memory_mb": round(self.estimated_memory_mb, 1),
            "estimated_energy_mj_per_frame": round(self.estimated_energy_mj_per_frame, 3),
            "thermal_state": self.thermal_state,
        }


class DeploymentSuitabilityScorer:
    """Rank trackers for deployment on a specific edge device.

    Combines accuracy metrics from a :class:`~eovot.benchmark.engine.BenchmarkResult`
    with device-simulation projections from
    :class:`~eovot.profiling.device_sim.DeviceSimulator` to produce a
    multi-dimensional suitability score for each tracker.

    Args:
        device: Target device key (e.g. ``"rpi4"``, ``"jetson_nano"``).
            Must be registered in :data:`~eovot.profiling.device_sim.KNOWN_DEVICES`
            or added via
            :meth:`~eovot.profiling.device_sim.DeviceSimulator.register_device`
            on the ``sim`` attribute.
        constraints: Deployment scenario constraints.  Defaults to
            :class:`DeploymentConstraints` with all-default values.
        host_calibration_factor: Passed to
            :class:`~eovot.profiling.device_sim.DeviceSimulator`.  Correct for
            host machines that are faster (> 1.0) or slower (< 1.0) than the
            reference Intel Core i7 class.

    Attributes:
        sim: The underlying :class:`~eovot.profiling.device_sim.DeviceSimulator`
            instance — use this to register custom device profiles.

    Example::

        scorer = DeploymentSuitabilityScorer(
            device="jetson_nano",
            constraints=DeploymentConstraints(min_fps=15.0, target_fps=25.0),
        )
        scores = scorer.rank([mosse_result, kcf_result, csrt_result])
        print(scorer.to_markdown_table(scores))
    """

    def __init__(
        self,
        device: str,
        constraints: Optional[DeploymentConstraints] = None,
        host_calibration_factor: float = 1.0,
    ) -> None:
        self._device = device
        self._constraints = constraints if constraints is not None else DeploymentConstraints()
        self.sim = DeviceSimulator(host_calibration_factor=host_calibration_factor)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _aggregate_profiling(self, result: BenchmarkResult) -> ProfilingResult:
        """Build a representative ProfilingResult from aggregate BenchmarkResult metrics."""
        fps = result.mean_fps if result.sequence_results else 0.0
        lat_ms = 1_000.0 / fps if fps > 0 else 0.0
        frame_count = sum(
            r.profiling.frame_count for r in result.sequence_results
        )
        return ProfilingResult(
            tracker_name=result.tracker_name,
            frame_count=frame_count,
            fps=fps,
            latency_mean_ms=lat_ms,
            latency_std_ms=0.0,
            latency_p95_ms=lat_ms,
            peak_memory_mb=result.peak_memory_mb if result.sequence_results else 0.0,
        )

    def _extract_accuracy(self, result: BenchmarkResult) -> float:
        """Return the best available accuracy scalar: success AUC > mean IoU."""
        if not result.sequence_results:
            return 0.0
        sauc = result.mean_success_auc
        if sauc is not None:
            return float(sauc)
        return float(result.mean_iou)

    # ------------------------------------------------------------------
    # Dimension scorers (normalised to [0, 1])
    # ------------------------------------------------------------------

    @staticmethod
    def _score_accuracy(acc: float, min_accuracy: float) -> float:
        """Linear accuracy score.  Returns 0 when below ``min_accuracy``."""
        if acc < min_accuracy:
            return 0.0
        return float(np.clip(acc, 0.0, 1.0))

    @staticmethod
    def _score_throughput(fps: float, min_fps: float, target_fps: float) -> float:
        """Throughput score: 0 at or below ``min_fps``, 1 at ``target_fps``."""
        if fps <= min_fps:
            return 0.0
        span = max(target_fps - min_fps, 1e-6)
        return float(min((fps - min_fps) / span, 1.0))

    @staticmethod
    def _score_memory(est_mb: float, limit_mb: float) -> float:
        """Memory headroom score: 1 when near-empty, 0 when at the limit."""
        return float(np.clip(1.0 - est_mb / max(limit_mb, 1.0), 0.0, 1.0))

    @staticmethod
    def _score_energy(
        est_energy_mj: float,
        device_tdp_watts: float,
        target_fps: float,
    ) -> float:
        """Energy efficiency score relative to the expected cost at ``target_fps``.

        The reference energy is the estimated millijoules per frame if the
        tracker were running exactly at ``target_fps``.  A tracker achieving
        higher FPS (lower latency) earns score > 1, clamped to 1.0; a slower
        tracker earns a proportionally lower score.
        """
        target_latency_ms = 1_000.0 / max(target_fps, 1e-6)
        ref_energy_mj = (
            device_tdp_watts
            * DeviceSimulator.CPU_UTIL_FRACTION
            * target_latency_ms
        )
        return float(np.clip(ref_energy_mj / max(est_energy_mj, 1e-9), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, result: BenchmarkResult) -> DeploymentScore:
        """Score a single tracker for deployment on the configured device.

        Args:
            result: Benchmark output from
                :class:`~eovot.benchmark.engine.BenchmarkEngine`.

        Returns:
            :class:`DeploymentScore` with all component and composite scores.

        Raises:
            KeyError: If the configured device name is not registered.
        """
        c = self._constraints
        target_fps = c.effective_target_fps()

        # Project onto target device
        prof = self._aggregate_profiling(result)
        sim_result = self.sim.simulate(prof, self._device, c.sustained_seconds)
        device_profile = self.sim.get_profile(self._device)

        # Determine effective memory ceiling (user override or device limit)
        mem_limit = (
            c.max_memory_mb
            if c.max_memory_mb is not None
            else device_profile.memory_limit_mb
        )

        # Hard constraint checks
        disqualified = False
        disqualify_reason = ""
        if sim_result.estimated_memory_mb > mem_limit:
            disqualified = True
            disqualify_reason = (
                f"OOM: tracker needs {sim_result.estimated_memory_mb:.0f} MB, "
                f"limit is {mem_limit:.0f} MB"
            )
        elif sim_result.estimated_fps < c.min_fps:
            disqualified = True
            disqualify_reason = (
                f"below min FPS: {sim_result.estimated_fps:.1f} < {c.min_fps:.1f}"
            )

        # Component scores
        acc_val = self._extract_accuracy(result)
        acc_s = self._score_accuracy(acc_val, c.min_accuracy)
        tput_s = self._score_throughput(sim_result.estimated_fps, c.min_fps, target_fps)
        mem_s = self._score_memory(sim_result.estimated_memory_mb, mem_limit)
        energy_s = self._score_energy(
            sim_result.estimated_energy_mj_per_frame,
            device_profile.tdp_watts,
            target_fps,
        )

        # Weighted composite (weights normalised to sum 1)
        w = c.weights
        w_total = sum(w.values()) or 1.0
        total = (
            w.get("accuracy", 0.0) * acc_s
            + w.get("throughput", 0.0) * tput_s
            + w.get("memory", 0.0) * mem_s
            + w.get("energy", 0.0) * energy_s
        ) / w_total

        return DeploymentScore(
            tracker_name=result.tracker_name,
            device_name=self._device,
            total_score=float(np.clip(total, 0.0, 1.0)),
            deployment_ready=not disqualified,
            disqualified=disqualified,
            disqualify_reason=disqualify_reason,
            accuracy_score=acc_s,
            throughput_score=tput_s,
            memory_score=mem_s,
            energy_score=energy_s,
            mean_accuracy=acc_val,
            estimated_fps=sim_result.estimated_fps,
            estimated_memory_mb=sim_result.estimated_memory_mb,
            estimated_energy_mj_per_frame=sim_result.estimated_energy_mj_per_frame,
            thermal_state=sim_result.thermal_state,
        )

    def rank(self, results: List[BenchmarkResult]) -> List[DeploymentScore]:
        """Score and rank multiple trackers for the configured device.

        Deployment-ready trackers are listed before disqualified ones.
        Within each group, trackers are sorted by ``total_score`` (highest first).

        Args:
            results: List of :class:`~eovot.benchmark.engine.BenchmarkResult`,
                one per tracker.

        Returns:
            List of :class:`DeploymentScore` sorted by suitability.
        """
        scores = [self.score(r) for r in results]
        scores.sort(key=lambda s: (s.disqualified, -s.total_score))
        return scores

    def to_markdown_table(self, scores: List[DeploymentScore]) -> str:
        """Format ranked deployment scores as a Markdown table.

        Args:
            scores: Output of :meth:`rank` or a list of :meth:`score` calls.

        Returns:
            Multi-line Markdown string ready for README or report embedding.

        Example output::

            | Rank | Tracker | Score | Acc  | Thpt | Mem  | Enrg | FPS  | Mem(MB) | Status   |
            |------|---------|------:|-----:|-----:|-----:|-----:|-----:|--------:|----------|
            |    1 | MOSSE   | 0.712 | 0.61 | 0.88 | 0.91 | 0.82 | 24.3 |    52   | READY    |
        """
        header = (
            "| Rank | Tracker | Score | Acc | Thpt | Mem | Enrg"
            " | Est. FPS | Mem (MB) | Status |\n"
            "|------|---------|------:|----:|-----:|----:|-----:"
            "|---------:|---------:|--------|\n"
        )
        rows: List[str] = []
        for rank, s in enumerate(scores, start=1):
            status = "READY" if s.deployment_ready else f"DQ: {s.disqualify_reason}"
            rows.append(
                f"| {rank:>4} "
                f"| {s.tracker_name} "
                f"| {s.total_score:.3f} "
                f"| {s.accuracy_score:.3f} "
                f"| {s.throughput_score:.3f} "
                f"| {s.memory_score:.3f} "
                f"| {s.energy_score:.3f} "
                f"| {s.estimated_fps:>8.1f} "
                f"| {s.estimated_memory_mb:>8.0f} "
                f"| {status} |"
            )
        return header + "\n".join(rows)

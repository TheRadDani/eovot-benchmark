"""Multi-tracker edge deployment analysis report.

Bridges the gap between ``BenchmarkEngine`` results and ``DeviceSimulator``
by projecting every tracker's profiling data onto every built-in edge device
in a single call.  The output answers the practitioner's core question:

    *"Given these benchmark results, which tracker should I deploy on each
    of my target devices?"*

Computation overview
--------------------
1. For each ``BenchmarkResult``, compute an *aggregate* ``ProfilingResult``
   that summarises all sequences into a single representative timing profile.
2. Feed that aggregate into ``DeviceSimulator.simulate_all`` to obtain
   projected latency, FPS, memory, and energy per device.
3. Compute a *device-specific* Edge Efficiency Score (EES) using the
   projected FPS instead of the host-measured FPS::

       EES_device = mean_iou × log1p(device_fps) / (1 + mem_mb / budget_mb)

4. On each device, identify the Pareto front in the (mean_iou, EES_device)
   objective space and recommend the tracker with the highest device-EES.

Typical usage::

    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.reporting.deployment_report import DeploymentReportEngine

    engine = BenchmarkEngine()
    results = [engine.run(t, dataset) for t in trackers]

    report_engine = DeploymentReportEngine()
    report = report_engine.analyze(results)

    print(report_engine.to_markdown(report))
    report_engine.save(report, output_dir="results/deployment")
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..profiling.device_sim import DeviceSimulator, DeviceSimResult
from ..profiling.profiler import ProfilingResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DeviceTrackerEntry:
    """Performance of one tracker projected onto one edge device.

    Attributes:
        tracker_name: Human-readable tracker identifier.
        device_name: Short device key (e.g. ``"rpi4"``).
        display_name: Human-readable device label.
        mean_iou: Mean IoU measured on the host benchmark.
        device_fps: Projected frames per second on the target device.
        device_latency_ms: Projected mean per-frame latency (ms).
        peak_memory_mb: Peak RSS memory footprint (algorithm-determined).
        memory_limit_mb: Total device RAM.
        fits_in_memory: Whether the tracker fits within device memory.
        device_ees: Device-specific Edge Efficiency Score.
        thermal_state: ``"nominal"``, ``"transitioning"``, or ``"throttled"``.
        energy_mj_per_frame: Projected energy consumption per frame (mJ).
        on_pareto_front: ``True`` if no other tracker dominates this one
            on this device in both mIoU and device_ees.
        is_recommended: ``True`` for the highest device_ees tracker on this
            device that also fits in memory.
    """

    tracker_name: str
    device_name: str
    display_name: str
    mean_iou: float
    device_fps: float
    device_latency_ms: float
    peak_memory_mb: float
    memory_limit_mb: float
    fits_in_memory: bool
    device_ees: float
    thermal_state: str
    energy_mj_per_frame: float
    on_pareto_front: bool = field(default=False)
    is_recommended: bool = field(default=False)


@dataclass
class DeploymentReport:
    """Full deployment analysis across all trackers and devices.

    Attributes:
        dataset_name: Name of the dataset used for benchmarking.
        sustained_seconds: Sustained-load duration used for thermal modelling.
        devices: Ordered list of device keys included in the report.
        entries: All (tracker, device) pairs; length = n_trackers × n_devices.
        recommendations: Mapping ``device_name → tracker_name`` for the
            highest device-EES memory-feasible tracker on each device.
    """

    dataset_name: str
    sustained_seconds: float
    devices: List[str]
    entries: List[DeviceTrackerEntry] = field(default_factory=list)
    recommendations: Dict[str, str] = field(default_factory=dict)

    def entries_for_device(self, device_name: str) -> List[DeviceTrackerEntry]:
        """Return all entries for a specific device, sorted by device_ees desc."""
        out = [e for e in self.entries if e.device_name == device_name]
        out.sort(key=lambda e: e.device_ees, reverse=True)
        return out

    def entries_for_tracker(self, tracker_name: str) -> List[DeviceTrackerEntry]:
        """Return all device entries for a specific tracker."""
        return [e for e in self.entries if e.tracker_name == tracker_name]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DeploymentReportEngine:
    """Analyse benchmark results across edge devices and produce deployment guidance.

    Args:
        memory_budget_mb: Memory ceiling (MB) for EES computation.
            Trackers within budget receive full memory credit.
            Default: ``512.0`` MB.
        sustained_seconds: Duration of continuous tracking on the device,
            used for thermal throttling modelling.
            ``0.0`` (default) models a cold-start burst scenario.
        host_calibration_factor: Passed to :class:`DeviceSimulator` to correct
            for a host that is faster/slower than the reference i7 class.
            Default: ``1.0``.
        device_names: Explicit list of device keys to include.  If ``None``
            all built-in devices are used.
    """

    def __init__(
        self,
        memory_budget_mb: float = 512.0,
        sustained_seconds: float = 0.0,
        host_calibration_factor: float = 1.0,
        device_names: Optional[List[str]] = None,
    ) -> None:
        if memory_budget_mb <= 0:
            raise ValueError("memory_budget_mb must be positive.")
        self.memory_budget_mb = memory_budget_mb
        self.sustained_seconds = sustained_seconds
        self._sim = DeviceSimulator(host_calibration_factor=host_calibration_factor)
        self._device_names: List[str] = (
            device_names if device_names is not None else self._sim.list_devices()
        )

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def analyze(
        self,
        results: "List[BenchmarkResult]",  # noqa: F821 — resolved at runtime
    ) -> DeploymentReport:
        """Run the full deployment analysis.

        Args:
            results: One :class:`~eovot.benchmark.engine.BenchmarkResult` per
                tracker, all evaluated on the same dataset.

        Returns:
            :class:`DeploymentReport` with all entries populated and
            ``recommendations`` filled.

        Raises:
            ValueError: If *results* is empty.
        """
        if not results:
            raise ValueError("results must contain at least one BenchmarkResult.")

        dataset_name = results[0].dataset_name
        entries: List[DeviceTrackerEntry] = []

        for bench in results:
            agg_profiling = _aggregate_profiling(bench)
            sim_results: List[DeviceSimResult] = self._sim.simulate_all(
                agg_profiling,
                sustained_seconds=self.sustained_seconds,
                device_names=self._device_names,
            )
            for sim in sim_results:
                ees = _device_ees(bench.mean_iou, sim.estimated_fps, sim.estimated_memory_mb, self.memory_budget_mb)
                entries.append(
                    DeviceTrackerEntry(
                        tracker_name=bench.tracker_name,
                        device_name=sim.device_name,
                        display_name=sim.display_name,
                        mean_iou=bench.mean_iou,
                        device_fps=sim.estimated_fps,
                        device_latency_ms=sim.estimated_latency_ms,
                        peak_memory_mb=sim.estimated_memory_mb,
                        memory_limit_mb=sim.memory_limit_mb,
                        fits_in_memory=sim.fits_in_memory,
                        device_ees=ees,
                        thermal_state=sim.thermal_state,
                        energy_mj_per_frame=sim.estimated_energy_mj_per_frame,
                    )
                )

        report = DeploymentReport(
            dataset_name=dataset_name,
            sustained_seconds=self.sustained_seconds,
            devices=list(self._device_names),
            entries=entries,
        )
        self._mark_pareto(report)
        self._pick_recommendations(report)
        return report

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def to_markdown(self, report: DeploymentReport) -> str:
        """Render the deployment report as a multi-section Markdown document.

        Produces one table per device showing all trackers ranked by
        device-specific EES, plus a summary recommendations table.

        Args:
            report: Output of :meth:`analyze`.

        Returns:
            Multi-line Markdown string ready for embedding in a README or paper.
        """
        lines = [
            "# EOVOT Edge Deployment Analysis Report\n",
            f"**Dataset:** {report.dataset_name}  ",
            f"**Thermal scenario:** {'sustained ' + str(int(report.sustained_seconds)) + 's' if report.sustained_seconds > 0 else 'cold-start (burst)'}  ",
            f"**Memory budget:** {self.memory_budget_mb:.0f} MB\n",
            "---\n",
        ]

        # Per-device tables
        for device_key in report.devices:
            device_entries = report.entries_for_device(device_key)
            if not device_entries:
                continue
            display = device_entries[0].display_name
            mem_limit = device_entries[0].memory_limit_mb
            lines.append(f"## {display} (RAM: {mem_limit:.0f} MB)\n")
            lines.append(
                "| Rank | Tracker | mIoU | FPS | Latency (ms) | Mem (MB) | Fits? | EES | Thermal | mJ/frame | Pareto |"
            )
            lines.append(
                "|------|---------|-----:|----:|-------------:|---------:|:-----:|----:|---------|----------|:------:|"
            )
            for rank, e in enumerate(device_entries, start=1):
                fits = "✓" if e.fits_in_memory else "✗ OOM"
                pareto = "✓" if e.on_pareto_front else ""
                rec = " ★" if e.is_recommended else ""
                lines.append(
                    f"| {rank} | {e.tracker_name}{rec} "
                    f"| {e.mean_iou:.4f} "
                    f"| {e.device_fps:.1f} "
                    f"| {e.device_latency_ms:.1f} "
                    f"| {e.peak_memory_mb:.0f} "
                    f"| {fits} "
                    f"| {e.device_ees:.4f} "
                    f"| {e.thermal_state} "
                    f"| {e.energy_mj_per_frame:.3f} "
                    f"| {pareto} |"
                )
            lines.append("")

        # Recommendation summary
        lines.append("---\n")
        lines.append("## Recommended Tracker per Device\n")
        lines.append("| Device | Recommended Tracker | Reason |")
        lines.append("|--------|--------------------:|--------|")
        for device_key in report.devices:
            rec_tracker = report.recommendations.get(device_key, "none (all OOM)")
            device_entries = report.entries_for_device(device_key)
            display = device_entries[0].display_name if device_entries else device_key
            if rec_tracker != "none (all OOM)":
                matched = next((e for e in device_entries if e.tracker_name == rec_tracker), None)
                reason = (
                    f"highest EES ({matched.device_ees:.4f}), "
                    f"{matched.device_fps:.1f} FPS, "
                    f"fits in memory"
                ) if matched else "highest device EES"
            else:
                reason = "all trackers exceed device RAM"
            lines.append(f"| {display} | {rec_tracker} | {reason} |")
        lines.append("")

        return "\n".join(lines)

    def to_dict(self, report: DeploymentReport) -> dict:
        """Serialise the report to a JSON-compatible dict.

        Args:
            report: Output of :meth:`analyze`.

        Returns:
            Nested dict with keys ``"metadata"``, ``"recommendations"``,
            and ``"per_device"`` (one entry per device).
        """
        per_device: Dict[str, list] = {}
        for device_key in report.devices:
            per_device[device_key] = [
                {
                    "tracker": e.tracker_name,
                    "mean_iou": round(e.mean_iou, 4),
                    "device_fps": round(e.device_fps, 2),
                    "device_latency_ms": round(e.device_latency_ms, 3),
                    "peak_memory_mb": round(e.peak_memory_mb, 1),
                    "memory_limit_mb": e.memory_limit_mb,
                    "fits_in_memory": e.fits_in_memory,
                    "device_ees": round(e.device_ees, 4),
                    "thermal_state": e.thermal_state,
                    "energy_mj_per_frame": round(e.energy_mj_per_frame, 4),
                    "on_pareto_front": e.on_pareto_front,
                    "is_recommended": e.is_recommended,
                }
                for e in report.entries_for_device(device_key)
            ]
        return {
            "metadata": {
                "dataset": report.dataset_name,
                "sustained_seconds": report.sustained_seconds,
                "memory_budget_mb": self.memory_budget_mb,
                "devices": report.devices,
            },
            "recommendations": report.recommendations,
            "per_device": per_device,
        }

    def save(self, report: DeploymentReport, output_dir: str = "results/deployment") -> Dict[str, Path]:
        """Write the Markdown report and JSON data to *output_dir*.

        Args:
            report: Output of :meth:`analyze`.
            output_dir: Directory to write files into (created if absent).

        Returns:
            ``{"markdown": Path(...), "json": Path(...)}`` mapping.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        md_path = out / "deployment_report.md"
        md_path.write_text(self.to_markdown(report), encoding="utf-8")

        json_path = out / "deployment_report.json"
        with open(json_path, "w") as fh:
            json.dump(self.to_dict(report), fh, indent=2)

        return {"markdown": md_path, "json": json_path}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _mark_pareto(self, report: DeploymentReport) -> None:
        """Set ``on_pareto_front`` for each entry in (mIoU, device_ees) space."""
        for device_key in report.devices:
            device_entries = [e for e in report.entries if e.device_name == device_key]
            for i, cand in enumerate(device_entries):
                dominated = False
                for j, other in enumerate(device_entries):
                    if i == j:
                        continue
                    if (
                        other.mean_iou >= cand.mean_iou
                        and other.device_ees >= cand.device_ees
                        and (other.mean_iou > cand.mean_iou or other.device_ees > cand.device_ees)
                    ):
                        dominated = True
                        break
                cand.on_pareto_front = not dominated

    def _pick_recommendations(self, report: DeploymentReport) -> None:
        """Set ``is_recommended`` and populate ``report.recommendations``."""
        for device_key in report.devices:
            device_entries = report.entries_for_device(device_key)
            feasible = [e for e in device_entries if e.fits_in_memory]
            if not feasible:
                report.recommendations[device_key] = "none (all OOM)"
                continue
            best = max(feasible, key=lambda e: e.device_ees)
            best.is_recommended = True
            report.recommendations[device_key] = best.tracker_name


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _aggregate_profiling(bench: "BenchmarkResult") -> ProfilingResult:  # noqa: F821
    """Build a single representative ProfilingResult from a BenchmarkResult.

    Collects all per-sequence latency statistics and the peak memory across
    all sequences to produce a host-level summary suitable for device projection.

    Args:
        bench: Benchmark result containing one or more sequence results.

    Returns:
        :class:`~eovot.profiling.profiler.ProfilingResult` aggregating all
        sequence-level profiling data.
    """
    seqs = bench.sequence_results
    if not seqs:
        raise ValueError(f"BenchmarkResult for '{bench.tracker_name}' has no sequence results.")

    # Reconstruct individual frame latencies from summary stats via normal approx.
    # We use the mean and std already stored; this is sufficient for device projection.
    all_means = np.array([s.profiling.latency_mean_ms for s in seqs])
    all_stds = np.array([s.profiling.latency_std_ms for s in seqs])
    all_counts = np.array([s.profiling.frame_count for s in seqs], dtype=np.float64)

    total_frames = int(all_counts.sum())
    # Grand mean: weighted average of per-sequence means
    grand_mean_ms = float(np.average(all_means, weights=all_counts))
    # Grand std: pooled standard deviation
    grand_var = float(
        np.average(all_stds ** 2 + (all_means - grand_mean_ms) ** 2, weights=all_counts)
    )
    grand_std_ms = math.sqrt(max(grand_var, 0.0))
    # p95: assume normal distribution for the aggregate
    grand_p95_ms = grand_mean_ms + 1.645 * grand_std_ms

    peak_mem = bench.peak_memory_mb
    fps = 1_000.0 / grand_mean_ms if grand_mean_ms > 0 else 0.0

    return ProfilingResult(
        tracker_name=bench.tracker_name,
        frame_count=total_frames,
        fps=fps,
        latency_mean_ms=grand_mean_ms,
        latency_std_ms=grand_std_ms,
        latency_p95_ms=grand_p95_ms,
        peak_memory_mb=peak_mem,
    )


def _device_ees(
    mean_iou: float,
    device_fps: float,
    peak_memory_mb: float,
    memory_budget_mb: float,
) -> float:
    """Compute the device-specific Edge Efficiency Score.

    Uses the same formula as :class:`~eovot.metrics.efficiency.EfficiencyMetricsEngine`
    but with the device-projected FPS rather than the host-measured FPS::

        EES = mean_iou × log1p(device_fps) / (1 + peak_memory_mb / memory_budget_mb)

    Args:
        mean_iou: Mean IoU on the host benchmark, in ``[0, 1]``.
        device_fps: Projected FPS on the target edge device.
        peak_memory_mb: Peak RSS memory footprint.
        memory_budget_mb: Memory ceiling for the penalty denominator.

    Returns:
        EES scalar ≥ 0.  Returns ``0.0`` for non-positive fps or negative IoU.
    """
    if device_fps <= 0 or mean_iou < 0:
        return 0.0
    return (mean_iou * math.log1p(device_fps)) / (1.0 + peak_memory_mb / memory_budget_mb)

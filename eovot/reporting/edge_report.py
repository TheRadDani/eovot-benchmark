"""End-to-end edge deployment report generator for EOVOT.

EOVOT's key research claim is that trackers must be evaluated across *both*
accuracy and hardware efficiency.  :class:`EdgeDeploymentReporter` operationalises
this by taking a completed :class:`~eovot.benchmark.engine.BenchmarkResult` and
producing a single, self-contained Markdown report that covers:

1. **Accuracy summary** — mean IoU, success AUC, precision AUC per tracker.
2. **Host profiling** — FPS, latency (mean ± std, p95), peak memory.
3. **Edge device projection** — FPS, latency, memory, thermal state, and energy
   per frame on every built-in edge device (or a custom subset), projected via
   :class:`~eovot.profiling.device_sim.DeviceSimulator`.
4. **Efficiency ranking** — Edge Efficiency Score (EES) and Pareto-front flag,
   computed by :class:`~eovot.metrics.efficiency.EfficiencyMetricsEngine`.
5. **Deployment verdict** — per-device go/no-go recommendation based on minimum
   FPS and memory constraints supplied by the caller.

The report is designed to be pasted directly into a paper supplementary section
or README.  All sections are clearly delimited so individual tables can be
extracted programmatically.

Typical usage::

    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.mosse import MOSSETracker
    from eovot.reporting.edge_report import EdgeDeploymentReporter

    ds = SyntheticDataset(num_sequences=5, num_frames=100)
    engine = BenchmarkEngine(verbose=False, tdp_watts=15.0)
    result = engine.run(MOSSETracker(), ds, dataset_name="Synthetic")

    reporter = EdgeDeploymentReporter(
        min_fps=10.0,
        memory_budget_mb=512.0,
        sustained_seconds=60.0,
    )
    report_md = reporter.generate(result)
    print(report_md)

    # Save to disk
    reporter.save(result, path="results/mosse_edge_report.md")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

from ..metrics.efficiency import EfficiencyMetricsEngine
from ..profiling.device_sim import DeviceSimResult, DeviceSimulator

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult


# ---------------------------------------------------------------------------
# Per-device deployment verdict
# ---------------------------------------------------------------------------

@dataclass
class DeploymentVerdict:
    """Go/no-go recommendation for one tracker × device combination.

    Attributes:
        device_name: Short device identifier (e.g. ``"rpi4"``).
        display_name: Human-readable device label.
        tracker_name: Name of the tracker under evaluation.
        deployable: ``True`` when the projected FPS ≥ ``min_fps`` AND
            the tracker fits in device memory.
        reason: One-line explanation of the verdict.
        estimated_fps: Projected FPS on the device (after thermal throttle).
        memory_ok: Whether the tracker fits within the device memory limit.
    """

    device_name: str
    display_name: str
    tracker_name: str
    deployable: bool
    reason: str
    estimated_fps: float
    memory_ok: bool

    def __str__(self) -> str:
        status = "✓ DEPLOY" if self.deployable else "✗ SKIP"
        return (
            f"[{status}] {self.display_name} — "
            f"{self.estimated_fps:.1f} FPS — {self.reason}"
        )


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

class EdgeDeploymentReporter:
    """Generate a comprehensive edge deployment report from a benchmark result.

    Args:
        min_fps: Minimum acceptable FPS on the target device for the tracker
            to receive a "deployable" verdict.  Default: ``10.0``.
        memory_budget_mb: Memory budget for the Edge Efficiency Score.
            Default: ``512.0`` MB.
        sustained_seconds: Sustained tracking duration used for thermal
            throttling model in :class:`~eovot.profiling.device_sim.DeviceSimulator`.
            Default: ``60.0`` seconds.
        device_names: Subset of device keys to include.  If ``None``, all
            built-in devices are used.
        host_calibration_factor: Passed to :class:`~eovot.profiling.device_sim.DeviceSimulator`
            to correct for host machines faster or slower than the i7 reference.
            Default: ``1.0``.
    """

    def __init__(
        self,
        min_fps: float = 10.0,
        memory_budget_mb: float = 512.0,
        sustained_seconds: float = 60.0,
        device_names: Optional[List[str]] = None,
        host_calibration_factor: float = 1.0,
    ) -> None:
        if min_fps <= 0:
            raise ValueError(f"min_fps must be positive, got {min_fps}.")
        self.min_fps = min_fps
        self.memory_budget_mb = memory_budget_mb
        self.sustained_seconds = sustained_seconds
        self.device_names = device_names
        self._sim = DeviceSimulator(host_calibration_factor=host_calibration_factor)
        self._eff = EfficiencyMetricsEngine(memory_budget_mb=memory_budget_mb)

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def generate(self, result: "BenchmarkResult") -> str:
        """Produce the full Markdown edge deployment report.

        Args:
            result: Completed :class:`~eovot.benchmark.engine.BenchmarkResult`.

        Returns:
            Multi-section Markdown string containing all analysis tables.
        """
        sections = [
            self._header(result),
            self._accuracy_section(result),
            self._host_profiling_section(result),
            self._device_projection_section(result),
            self._efficiency_section(result),
            self._verdict_section(result),
            self._footer(result),
        ]
        return "\n\n".join(sections)

    def save(self, result: "BenchmarkResult", path: str = "edge_report.md") -> Path:
        """Write the Markdown report to *path*.

        Args:
            result: Completed benchmark result.
            path: Output file path.

        Returns:
            :class:`pathlib.Path` of the written file.
        """
        md = self.generate(result)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        return out

    def to_dict(self, result: "BenchmarkResult") -> dict:
        """Export all report data as a serialisable dict (for JSON export).

        Args:
            result: Completed benchmark result.

        Returns:
            Dict with keys ``accuracy``, ``host_profiling``,
            ``device_projections``, ``efficiency``, ``verdicts``.
        """
        sim_results = self._run_simulation(result)
        eff_entries = self._eff.rank_trackers([result])
        verdicts = self._compute_verdicts(result, sim_results)

        return {
            "tracker": result.tracker_name,
            "dataset": result.dataset_name,
            "accuracy": {
                "mean_iou": round(result.mean_iou, 6),
                "success_auc": round(result.mean_success_auc or result.mean_iou, 6),
                "precision_auc": round(result.mean_precision_auc or 0.0, 6),
            },
            "host_profiling": {
                "mean_fps": round(result.mean_fps, 2),
                "peak_memory_mb": round(result.peak_memory_mb, 2),
            },
            "device_projections": self._sim.to_summary_dict(sim_results),
            "efficiency": [
                {
                    "tracker": e.tracker_name,
                    "ees": round(e.ees, 6),
                    "on_pareto_front": e.on_pareto_front,
                }
                for e in eff_entries
            ],
            "verdicts": [
                {
                    "device": v.device_name,
                    "deployable": v.deployable,
                    "estimated_fps": round(v.estimated_fps, 2),
                    "reason": v.reason,
                }
                for v in verdicts
            ],
        }

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------

    def _run_simulation(self, result: "BenchmarkResult") -> List[DeviceSimResult]:
        """Project the first sequence's profiling result across all target devices.

        Uses the aggregate FPS/latency from the benchmark result converted to
        a synthetic :class:`~eovot.profiling.profiler.ProfilingResult`.
        """
        from ..profiling.profiler import ProfilingResult
        import numpy as np

        # Collect all per-frame latencies across all sequences for a more
        # representative aggregate profiling result.
        all_latencies = []
        for sr in result.sequence_results:
            # Re-derive per-frame latencies from the mean and std stored in
            # ProfilingResult.  This avoids storing the raw latency list in
            # BenchmarkResult while still giving the simulator a usable input.
            mean_lat = sr.profiling.latency_mean_ms
            std_lat = sr.profiling.latency_std_ms
            n = sr.profiling.frame_count
            if n > 0:
                rng = np.random.default_rng(42)
                lats = rng.normal(mean_lat, std_lat, n)
                all_latencies.extend(lats.clip(min=0.1).tolist())

        if not all_latencies:
            all_latencies = [1000.0 / max(result.mean_fps, 0.01)]

        arr = np.array(all_latencies)
        mean_ms = float(arr.mean())
        agg_profiling = ProfilingResult(
            tracker_name=result.tracker_name,
            frame_count=len(arr),
            fps=1000.0 / mean_ms if mean_ms > 0 else 0.0,
            latency_mean_ms=mean_ms,
            latency_std_ms=float(arr.std()),
            latency_p95_ms=float(np.percentile(arr, 95)),
            peak_memory_mb=result.peak_memory_mb,
        )
        return self._sim.simulate_all(
            agg_profiling,
            sustained_seconds=self.sustained_seconds,
            device_names=self.device_names,
        )

    def _compute_verdicts(
        self,
        result: "BenchmarkResult",
        sim_results: List[DeviceSimResult],
    ) -> List[DeploymentVerdict]:
        verdicts = []
        for sim in sim_results:
            fps_ok = sim.estimated_fps >= self.min_fps
            mem_ok = sim.fits_in_memory
            deployable = fps_ok and mem_ok

            if deployable:
                reason = (
                    f"FPS={sim.estimated_fps:.1f} ≥ {self.min_fps:.0f} threshold; "
                    f"memory fits ({sim.estimated_memory_mb:.0f}/{sim.memory_limit_mb:.0f} MB)"
                )
            elif not fps_ok and not mem_ok:
                reason = (
                    f"FPS too low ({sim.estimated_fps:.1f} < {self.min_fps:.0f}) "
                    f"and OOM ({sim.estimated_memory_mb:.0f} > {sim.memory_limit_mb:.0f} MB)"
                )
            elif not fps_ok:
                reason = (
                    f"FPS too low: {sim.estimated_fps:.1f} < {self.min_fps:.0f} required"
                )
            else:
                reason = (
                    f"OOM: {sim.estimated_memory_mb:.0f} MB > {sim.memory_limit_mb:.0f} MB limit"
                )

            verdicts.append(DeploymentVerdict(
                device_name=sim.device_name,
                display_name=sim.display_name,
                tracker_name=result.tracker_name,
                deployable=deployable,
                reason=reason,
                estimated_fps=sim.estimated_fps,
                memory_ok=mem_ok,
            ))
        return verdicts

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    @staticmethod
    def _header(result: "BenchmarkResult") -> str:
        return (
            f"# EOVOT Edge Deployment Report\n\n"
            f"**Tracker:** {result.tracker_name}  \n"
            f"**Dataset:** {result.dataset_name}  \n"
            f"**Sequences evaluated:** {len(result.sequence_results)}"
        )

    @staticmethod
    def _accuracy_section(result: "BenchmarkResult") -> str:
        sauc = result.mean_success_auc
        pauc = result.mean_precision_auc
        lines = [
            "## Accuracy Metrics\n",
            "| Metric | Value |",
            "|--------|------:|",
            f"| Mean IoU | {result.mean_iou:.4f} |",
        ]
        if sauc is not None:
            lines.append(f"| Success AUC | {sauc:.4f} |")
        if pauc is not None:
            lines.append(f"| Precision AUC | {pauc:.4f} |")
        return "\n".join(lines)

    @staticmethod
    def _host_profiling_section(result: "BenchmarkResult") -> str:
        import numpy as np
        fps_values = [r.profiling.fps for r in result.sequence_results]
        lat_values = [r.profiling.latency_mean_ms for r in result.sequence_results]
        lat_p95 = [r.profiling.latency_p95_ms for r in result.sequence_results]

        lines = [
            "## Host Machine Profiling\n",
            "| Metric | Value |",
            "|--------|------:|",
            f"| Mean FPS | {float(np.mean(fps_values)):.1f} |",
            f"| Mean latency (ms) | {float(np.mean(lat_values)):.2f} |",
            f"| p95 latency (ms) | {float(np.mean(lat_p95)):.2f} |",
            f"| Peak memory (MB) | {result.peak_memory_mb:.1f} |",
        ]
        if result.total_energy_j is not None:
            lines.append(f"| Total energy (J) | {result.total_energy_j:.4f} |")
        if result.mean_energy_per_frame_mj is not None:
            lines.append(
                f"| Mean energy/frame (mJ) | {result.mean_energy_per_frame_mj:.4f} |"
            )
        return "\n".join(lines)

    def _device_projection_section(self, result: "BenchmarkResult") -> str:
        sim_results = self._run_simulation(result)
        table = self._sim.to_markdown_table(sim_results)
        return f"## Edge Device Projections\n\n*Sustained load: {self.sustained_seconds:.0f} s*\n\n{table}"

    def _efficiency_section(self, result: "BenchmarkResult") -> str:
        entries = self._eff.rank_trackers([result])
        table = self._eff.to_markdown_table(entries)
        return (
            f"## Edge Efficiency Score\n\n"
            f"*Memory budget: {self.memory_budget_mb:.0f} MB*\n\n"
            f"{table}"
        )

    def _verdict_section(self, result: "BenchmarkResult") -> str:
        sim_results = self._run_simulation(result)
        verdicts = self._compute_verdicts(result, sim_results)
        deployable = [v for v in verdicts if v.deployable]
        not_deployable = [v for v in verdicts if not v.deployable]

        lines = [
            "## Deployment Verdicts\n",
            f"*Minimum FPS threshold: {self.min_fps:.0f} FPS  |  "
            f"Memory budget: {self.memory_budget_mb:.0f} MB*\n",
            "| Device | FPS | Memory | Verdict | Reason |",
            "|--------|----:|:------:|:-------:|--------|",
        ]
        for v in verdicts:
            status = "✓ Deploy" if v.deployable else "✗ Skip"
            mem = "✓" if v.memory_ok else "✗"
            lines.append(
                f"| {v.display_name} "
                f"| {v.estimated_fps:.1f} "
                f"| {mem} "
                f"| {status} "
                f"| {v.reason} |"
            )

        summary_parts = []
        if deployable:
            names = ", ".join(v.display_name for v in deployable)
            summary_parts.append(f"**{len(deployable)} deployable:** {names}")
        if not_deployable:
            summary_parts.append(f"**{len(not_deployable)} not deployable.**")
        lines.append("\n" + "  \n".join(summary_parts))
        return "\n".join(lines)

    @staticmethod
    def _footer(result: "BenchmarkResult") -> str:
        return (
            "---\n"
            f"*Generated by EOVOT EdgeDeploymentReporter — "
            f"{result.tracker_name} × {result.dataset_name}*"
        )

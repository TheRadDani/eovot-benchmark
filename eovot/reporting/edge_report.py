"""Edge deployment report generator for EOVOT.

Combines benchmark results (accuracy + profiling) with :class:`DeviceSimulator`
projections to answer the central edge-deployment question:

    **"Which trackers can run on which edge devices, and at what cost?"**

The main output is an :class:`EdgeDeploymentReport` containing a *deployment
matrix* — a table of trackers × devices annotated with:

* Estimated FPS on each device
* Estimated energy per frame (mJ)
* Memory feasibility (fits / OOM)
* Deployability verdict against user-supplied FPS and memory constraints

Typical usage::

    from eovot.reporting.edge_report import EdgeDeploymentReport

    report = EdgeDeploymentReport.from_result_dicts(
        result_dicts,          # list of BenchmarkResult.to_dict() outputs
        min_fps=15.0,          # deployability threshold
        sustained_seconds=60.0,
    )

    print(report.to_markdown())
    report.save(output_dir="results/edge_report")

The :mod:`scripts.simulate_edge` CLI script wraps this class for command-line use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..profiling.device_sim import DeviceSimulator, DeviceSimResult, KNOWN_DEVICES
from ..profiling.profiler import ProfilingResult


# ---------------------------------------------------------------------------
# Cell: one (tracker, device) pair
# ---------------------------------------------------------------------------

@dataclass
class DeploymentCell:
    """Performance estimate for one tracker on one edge device.

    Attributes:
        tracker_name: Human-readable tracker identifier.
        device_name: Device key (e.g. ``"rpi4"``).
        device_display: Human-readable device label.
        host_fps: Measured FPS on the benchmark host.
        estimated_fps: Projected FPS on the target device.
        estimated_latency_ms: Projected mean per-frame latency (ms).
        estimated_memory_mb: Memory footprint (same as host; algorithm-bound).
        memory_limit_mb: Device RAM ceiling.
        fits_in_memory: Whether the tracker fits within the device RAM.
        estimated_energy_mj: Estimated energy per frame (milli-Joules).
        thermal_state: ``"nominal"``, ``"transitioning"``, or ``"throttled"``.
        deployable: Whether estimated FPS ≥ ``min_fps`` and ``fits_in_memory``.
        mean_iou: Mean IoU from the benchmark result (accuracy proxy).
    """

    tracker_name: str
    device_name: str
    device_display: str
    host_fps: float
    estimated_fps: float
    estimated_latency_ms: float
    estimated_memory_mb: float
    memory_limit_mb: float
    fits_in_memory: bool
    estimated_energy_mj: float
    thermal_state: str
    deployable: bool
    mean_iou: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "tracker": self.tracker_name,
            "device": self.device_name,
            "device_display": self.device_display,
            "host_fps": round(self.host_fps, 2),
            "estimated_fps": round(self.estimated_fps, 2),
            "estimated_latency_ms": round(self.estimated_latency_ms, 2),
            "estimated_memory_mb": round(self.estimated_memory_mb, 1),
            "memory_limit_mb": self.memory_limit_mb,
            "fits_in_memory": self.fits_in_memory,
            "estimated_energy_mj_per_frame": round(self.estimated_energy_mj, 4),
            "thermal_state": self.thermal_state,
            "deployable": self.deployable,
            "mean_iou": round(self.mean_iou, 4),
        }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class EdgeDeploymentReport:
    """Deployment analysis for a set of trackers across edge devices.

    Attributes:
        tracker_names: Ordered list of tracker names in the report.
        device_names: Ordered list of device keys in the report.
        cells: Flat list of :class:`DeploymentCell` objects (one per tracker×device pair).
        min_fps: FPS threshold used for the deployability verdict.
        sustained_seconds: Sustained-load duration used for thermal modelling.
    """

    tracker_names: List[str]
    device_names: List[str]
    cells: List[DeploymentCell] = field(default_factory=list)
    min_fps: float = 10.0
    sustained_seconds: float = 0.0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_result_dicts(
        cls,
        result_dicts: List[Dict],
        min_fps: float = 10.0,
        sustained_seconds: float = 0.0,
        device_names: Optional[List[str]] = None,
        host_calibration_factor: float = 1.0,
    ) -> "EdgeDeploymentReport":
        """Build a report from benchmark result dicts.

        Args:
            result_dicts: List of ``BenchmarkResult.to_dict()`` outputs.
            min_fps: Minimum FPS required for a deployment to be marked
                feasible.  Default ``10.0`` (practical real-time floor).
            sustained_seconds: Seconds of sustained operation to model
                thermal throttling.  ``0.0`` models a cold-start burst.
            device_names: Subset of device keys to include.  If ``None``,
                all built-in devices are included.
            host_calibration_factor: Passed to :class:`DeviceSimulator`.
                Adjust if the benchmark host is not an Intel Core i7 class CPU.

        Returns:
            :class:`EdgeDeploymentReport` populated with all cells.
        """
        sim = DeviceSimulator(host_calibration_factor=host_calibration_factor)
        target_devices = device_names or sim.list_devices()

        tracker_names: List[str] = []
        cells: List[DeploymentCell] = []

        for rd in result_dicts:
            summary = rd.get("summary", rd)
            tracker_name = summary.get("tracker_name") or summary.get("tracker", "unknown")
            if tracker_name not in tracker_names:
                tracker_names.append(tracker_name)

            mean_iou = float(summary.get("mean_iou", 0.0))
            host_fps = float(summary.get("mean_fps", 1.0))
            peak_mem = float(summary.get("peak_memory_mb", 0.0))
            latency_ms = (1000.0 / host_fps) if host_fps > 0 else float("inf")

            prof = ProfilingResult(
                tracker_name=tracker_name,
                frame_count=1,
                fps=host_fps,
                latency_mean_ms=latency_ms,
                latency_std_ms=0.0,
                latency_p95_ms=latency_ms,
                peak_memory_mb=peak_mem,
            )

            for device_key in target_devices:
                sim_result: DeviceSimResult = sim.simulate(
                    prof, device_key, sustained_seconds=sustained_seconds
                )
                deployable = (
                    sim_result.estimated_fps >= min_fps
                    and sim_result.fits_in_memory
                )
                cell = DeploymentCell(
                    tracker_name=tracker_name,
                    device_name=device_key,
                    device_display=sim_result.display_name,
                    host_fps=host_fps,
                    estimated_fps=sim_result.estimated_fps,
                    estimated_latency_ms=sim_result.estimated_latency_ms,
                    estimated_memory_mb=sim_result.estimated_memory_mb,
                    memory_limit_mb=sim_result.memory_limit_mb,
                    fits_in_memory=sim_result.fits_in_memory,
                    estimated_energy_mj=sim_result.estimated_energy_mj_per_frame,
                    thermal_state=sim_result.thermal_state,
                    deployable=deployable,
                    mean_iou=mean_iou,
                )
                cells.append(cell)

        return cls(
            tracker_names=tracker_names,
            device_names=target_devices,
            cells=cells,
            min_fps=min_fps,
            sustained_seconds=sustained_seconds,
        )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_cell(self, tracker_name: str, device_name: str) -> Optional[DeploymentCell]:
        """Return the cell for a specific (tracker, device) pair."""
        for cell in self.cells:
            if cell.tracker_name == tracker_name and cell.device_name == device_name:
                return cell
        return None

    def deployable_combinations(self) -> List[Tuple[str, str]]:
        """Return list of ``(tracker_name, device_name)`` pairs marked deployable."""
        return [
            (c.tracker_name, c.device_name)
            for c in self.cells
            if c.deployable
        ]

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def to_markdown(self) -> str:
        """Render the deployment matrix as a Markdown string.

        The table has one row per tracker and one column group per device.
        Each cell shows ``FPS / mJ`` and a deployability icon (✓ or ✗).

        Returns:
            Multi-line Markdown string suitable for GitHub, reports, or paper appendix.
        """
        lines: List[str] = []

        lines.append("# EOVOT Edge Deployment Matrix\n")
        lines.append(f"**Min FPS threshold:** {self.min_fps:.0f} FPS  ")
        lines.append(
            f"**Sustained load:** "
            f"{'cold-start' if self.sustained_seconds <= 0 else f'{self.sustained_seconds:.0f}s'}  "
        )
        lines.append(f"**Trackers evaluated:** {len(self.tracker_names)}  ")
        lines.append(f"**Devices simulated:** {len(self.device_names)}  \n")

        # --- Deployment matrix table ---
        lines.append("## Deployment Matrix\n")
        lines.append("> ✓ = deployable (FPS ≥ threshold, fits in RAM)  ✗ = not deployable\n")

        # Get display names for header
        device_displays: Dict[str, str] = {}
        for cell in self.cells:
            device_displays[cell.device_name] = cell.device_display

        # Header
        device_cols = [device_displays.get(d, d) for d in self.device_names]
        header = "| Tracker | mIoU |" + "".join(
            f" {_truncate(d, 22)} |" for d in device_cols
        )
        sep = "|---------|------|" + "".join("-" * (len(_truncate(d, 22)) + 2) + "|" for d in device_cols)

        lines.append(header)
        lines.append(sep)

        for tracker in self.tracker_names:
            # Find mIoU (same for all devices for this tracker)
            miou = 0.0
            for c in self.cells:
                if c.tracker_name == tracker:
                    miou = c.mean_iou
                    break
            row = f"| {tracker:<7} | {miou:.3f} |"
            for device in self.device_names:
                cell = self.get_cell(tracker, device)
                if cell is None:
                    row += " — |"
                else:
                    icon = "✓" if cell.deployable else "✗"
                    mem_flag = "" if cell.fits_in_memory else " OOM"
                    row += f" {icon} {cell.estimated_fps:.0f} FPS{mem_flag} |"
            lines.append(row)

        lines.append("")

        # --- Per-device detail tables ---
        lines.append("## Per-Device Detail\n")
        for device in self.device_names:
            display = device_displays.get(device, device)
            device_cells = [c for c in self.cells if c.device_name == device]
            if not device_cells:
                continue
            lines.append(f"### {display}\n")
            lines.append(
                "| Tracker | Host FPS | Est. FPS | Latency (ms) | Mem (MB) | Fits? | mJ/frame | Thermal | Deploy? |"
            )
            lines.append(
                "|---------|----------|----------|--------------|----------|:-----:|----------|---------|---------|"
            )
            for cell in sorted(device_cells, key=lambda c: c.estimated_fps, reverse=True):
                fits = "✓" if cell.fits_in_memory else "✗"
                deploy = "✓" if cell.deployable else "✗"
                lines.append(
                    f"| {cell.tracker_name} "
                    f"| {cell.host_fps:.1f} "
                    f"| {cell.estimated_fps:.1f} "
                    f"| {cell.estimated_latency_ms:.1f} "
                    f"| {cell.estimated_memory_mb:.0f} / {cell.memory_limit_mb:.0f} "
                    f"| {fits} "
                    f"| {cell.estimated_energy_mj:.3f} "
                    f"| {cell.thermal_state} "
                    f"| {deploy} |"
                )
            lines.append("")

        # --- Summary ---
        n_deployable = len(self.deployable_combinations())
        n_total = len(self.tracker_names) * len(self.device_names)
        lines.append("## Summary\n")
        lines.append(
            f"**{n_deployable} / {n_total}** tracker-device combinations are deployable "
            f"at ≥{self.min_fps:.0f} FPS.\n"
        )
        if self.deployable_combinations():
            lines.append("**Deployable combinations:**\n")
            for tracker, device in self.deployable_combinations():
                cell = self.get_cell(tracker, device)
                lines.append(
                    f"- {tracker} → {device_displays.get(device, device)}: "
                    f"{cell.estimated_fps:.0f} FPS, {cell.estimated_energy_mj:.3f} mJ/frame"
                )
        lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Serialize the full report to a dict (for JSON export)."""
        return {
            "min_fps_threshold": self.min_fps,
            "sustained_seconds": self.sustained_seconds,
            "trackers": self.tracker_names,
            "devices": self.device_names,
            "deployable_combinations": self.deployable_combinations(),
            "cells": [c.to_dict() for c in self.cells],
        }

    def save(self, output_dir: str = "results/edge_report", prefix: str = "edge_report") -> Dict[str, Path]:
        """Save Markdown and JSON reports to *output_dir*.

        Args:
            output_dir: Directory where files are written (created if absent).
            prefix: Filename prefix for both output files.

        Returns:
            Dict mapping ``"markdown"`` and ``"json"`` to their saved paths.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        md_path = out / f"{prefix}.md"
        md_path.write_text(self.to_markdown(), encoding="utf-8")

        json_path = out / f"{prefix}.json"
        json_path.write_text(
            json.dumps(self.to_dict(), indent=2), encoding="utf-8"
        )

        return {"markdown": md_path, "json": json_path}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _truncate(s: str, max_len: int) -> str:
    return s if len(s) <= max_len else s[: max_len - 1] + "…"

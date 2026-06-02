"""Device deployment analysis layer for EOVOT experiments.

After benchmarking trackers on the host machine, this module projects each
tracker's ProfilingResult onto all (or a selected subset of) edge devices
using DeviceSimulator, producing per-device FPS, latency, energy, and OOM
estimates in a single pass.

Typical usage — called automatically by ExperimentRunner when
``experiment.device_simulation.enabled: true`` is set in the config, but
also usable standalone::

    from eovot.experiment.device_report import DeviceReport
    from eovot.profiling.profiler import ProfilingResult

    # prof_map: dict of tracker_name → ProfilingResult
    report = DeviceReport(devices=["rpi4", "jetson_nano"], sustained_seconds=60.0)
    sim_results = report.run(prof_map)

    print(report.to_markdown(sim_results))
    json_data = report.to_summary_dicts(sim_results)
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..profiling.device_sim import DeviceSimResult, DeviceSimulator
from ..profiling.profiler import ProfilingResult


class DeviceReport:
    """Project multiple trackers' ProfilingResults onto an edge device fleet.

    Args:
        devices: Device keys to simulate (e.g. ``["rpi4", "jetson_nano"]``).
            ``None`` (default) uses all built-in profiles.
        sustained_seconds: Duration of continuous tracking in seconds, used
            for thermal-throttling modelling.  ``0.0`` = cold-start burst.
        host_calibration_factor: Correction factor for non-reference hosts.
            Values > 1.0 mean the benchmark host is faster than the i7
            reference used to calibrate the built-in speed factors.
    """

    def __init__(
        self,
        devices: Optional[List[str]] = None,
        sustained_seconds: float = 0.0,
        host_calibration_factor: float = 1.0,
    ) -> None:
        self._sim = DeviceSimulator(host_calibration_factor=host_calibration_factor)
        self._devices = devices
        self._sustained = sustained_seconds

    def run(
        self,
        profiling_by_tracker: Dict[str, ProfilingResult],
    ) -> Dict[str, List[DeviceSimResult]]:
        """Simulate all trackers across the device fleet.

        Args:
            profiling_by_tracker: Mapping of tracker name → ProfilingResult.

        Returns:
            Mapping of tracker name → list of DeviceSimResult, sorted by
            estimated FPS (highest first) for each tracker.
        """
        return {
            name: self._sim.simulate_all(
                prof,
                sustained_seconds=self._sustained,
                device_names=self._devices,
            )
            for name, prof in profiling_by_tracker.items()
        }

    def to_markdown(
        self,
        sim_results: Dict[str, List[DeviceSimResult]],
    ) -> str:
        """Render per-tracker device deployment tables as Markdown.

        Args:
            sim_results: Output of :meth:`run`.

        Returns:
            Multi-section Markdown string ready to write to ``device_report.md``.
        """
        sections: List[str] = ["# EOVOT Device Deployment Report\n"]
        for tracker_name, results in sim_results.items():
            sections.append(f"## {tracker_name}\n")
            sections.append(self._sim.to_markdown_table(results))
            sections.append("")
        return "\n".join(sections)

    def to_summary_dicts(
        self,
        sim_results: Dict[str, List[DeviceSimResult]],
    ) -> Dict[str, List[dict]]:
        """Convert all simulation results to plain dicts for JSON export.

        Args:
            sim_results: Output of :meth:`run`.

        Returns:
            Same structure with each DeviceSimResult replaced by a plain dict.
        """
        return {
            tracker: self._sim.to_summary_dict(results)
            for tracker, results in sim_results.items()
        }

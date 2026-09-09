"""Edge deployment scorecard for tracker × device feasibility analysis.

Combines benchmark profiling results with hardware-projected performance
(via :class:`~eovot.profiling.device_sim.DeviceSimulator`) to produce a
structured, publication-ready deployment matrix.

For each (tracker, device) pair the scorecard computes:

* **Projected FPS** — host FPS scaled by the device's CPU speed factor.
* **Memory feasibility** — whether peak RSS fits within the device's RAM.
* **Energy per hour** — Wh consumed running the tracker continuously.
* **Deployment Score** — a scalar in ``[0, 1]`` balancing real-time
  feasibility, memory efficiency, and energy frugality.
* **Deployment Tier** — qualitative label: ``READY``, ``MARGINAL``, or
  ``NOT FEASIBLE``.

Typical usage::

    from eovot.reporting.edge_scorecard import EdgeScorecard
    from eovot.profiling.device_sim import DeviceSimulator

    scorecard = EdgeScorecard(
        target_fps=25.0,
        devices=["rpi4", "jetson_nano", "coral_board"],
    )
    rows = scorecard.evaluate(benchmark_results)
    print(scorecard.to_markdown(rows))
    scorecard.to_json(rows, "results/edge_scorecard.json")
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult

from ..profiling.device_sim import DeviceSimulator


_TIERS = {
    "READY": "Tracker meets real-time target and memory budget on this device.",
    "MARGINAL": "Tracker is close to the real-time target; may work with optimisation.",
    "NOT FEASIBLE": "Tracker cannot meet real-time target or exceeds device memory.",
}


@dataclass
class ScorecardRow:
    """One row of the deployment scorecard (one tracker + one device)."""

    tracker_name: str
    device_name: str
    dataset_name: str

    # Accuracy (from original benchmark, device-independent)
    mean_iou: float

    # Projected performance on target device
    projected_fps: float
    projected_latency_ms: float
    memory_ok: bool            # peak RSS fits within device RAM
    energy_wh_per_hour: float  # watt-hours consumed per hour of operation

    # Composite deployment score [0, 1]
    deployment_score: float

    # Qualitative tier
    tier: str  # "READY" | "MARGINAL" | "NOT FEASIBLE"

    def __str__(self) -> str:
        mem = "OK" if self.memory_ok else "FAIL"
        return (
            f"[{self.tier:12s}] {self.tracker_name:12s} on {self.device_name:15s}  "
            f"mIoU={self.mean_iou:.3f}  FPS={self.projected_fps:.1f}  "
            f"E={self.energy_wh_per_hour:.3f} Wh/h  mem={mem}  "
            f"score={self.deployment_score:.3f}"
        )


class EdgeScorecard:
    """Evaluate trackers against a fleet of edge devices.

    Args:
        target_fps:       Minimum acceptable throughput (frames per second)
            for a tracker to be labelled ``READY``.
        devices:          List of device names to evaluate against.  Must
            match keys in
            :attr:`~eovot.profiling.device_sim.DeviceSimulator._profiles`.
            Pass ``None`` to evaluate against all built-in devices.
        marginal_fps_ratio: Fraction of ``target_fps`` below which a
            tracker is ``NOT FEASIBLE`` (default: 0.5).  Between this
            ratio and 1.0 the tier is ``MARGINAL``.
        energy_weight:    Weight applied to the energy efficiency term in
            the Deployment Score formula.  Range ``[0, 1]``.  Default: 0.2.
        sustained_seconds: Duration (s) passed to the device simulator
            for thermal-throttle modelling.  Default: 60.0.
    """

    def __init__(
        self,
        target_fps: float = 25.0,
        devices: Optional[List[str]] = None,
        marginal_fps_ratio: float = 0.5,
        energy_weight: float = 0.2,
        sustained_seconds: float = 60.0,
    ) -> None:
        if target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {target_fps}")
        if not (0.0 <= marginal_fps_ratio < 1.0):
            raise ValueError("marginal_fps_ratio must be in [0, 1)")
        if not (0.0 <= energy_weight <= 1.0):
            raise ValueError("energy_weight must be in [0, 1]")

        self.target_fps = target_fps
        self.marginal_fps_ratio = marginal_fps_ratio
        self.energy_weight = energy_weight
        self.sustained_seconds = sustained_seconds

        self._sim = DeviceSimulator()

        if devices is None:
            self._devices = list(self._sim._profiles.keys())
        else:
            unknown = [d for d in devices if d not in self._sim._profiles]
            if unknown:
                raise ValueError(f"Unknown devices: {unknown}. Available: {list(self._sim._profiles)}")
            self._devices = devices

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    def evaluate(self, results: List["BenchmarkResult"]) -> List[ScorecardRow]:
        """Compute the deployment scorecard for a list of benchmark results.

        Args:
            results: One :class:`~eovot.benchmark.engine.BenchmarkResult`
                per tracker (typically from
                :class:`~eovot.benchmark.engine.BenchmarkEngine`).

        Returns:
            List of :class:`ScorecardRow` objects, ordered by
            (tracker_name, device_name).
        """
        rows: List[ScorecardRow] = []
        for result in results:
            seq_profiling = [s.profiling for s in result.sequence_results]
            if not seq_profiling:
                continue

            import numpy as np
            from ..profiling.profiler import ProfilingResult

            # Build a representative ProfilingResult from sequence averages
            host_profiling = ProfilingResult(
                tracker_name=result.tracker_name,
                frame_count=sum(p.frame_count for p in seq_profiling),
                fps=float(np.mean([p.fps for p in seq_profiling])),
                latency_mean_ms=float(np.mean([p.latency_mean_ms for p in seq_profiling])),
                latency_std_ms=float(np.mean([p.latency_std_ms for p in seq_profiling])),
                latency_p95_ms=float(np.mean([p.latency_p95_ms for p in seq_profiling])),
                latency_p99_ms=float(np.mean([p.latency_p99_ms for p in seq_profiling])),
                latency_cv=float(np.mean([p.latency_cv for p in seq_profiling])),
                peak_memory_mb=float(np.max([p.peak_memory_mb for p in seq_profiling])),
            )

            for device_name in self._devices:
                row = self._evaluate_pair(result, host_profiling, device_name)
                rows.append(row)

        rows.sort(key=lambda r: (r.tracker_name, r.device_name))
        return rows

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_markdown(self, rows: List[ScorecardRow]) -> str:
        """Render the scorecard as a Markdown table.

        Args:
            rows: Output of :meth:`evaluate`.

        Returns:
            Multi-line Markdown string.
        """
        header = (
            "| Tracker | Device | mIoU | Proj. FPS | Latency (ms) "
            "| Mem | Energy (Wh/h) | Score | Tier |\n"
            "|---------|--------|-----:|----------:|-------------:"
            "|:---:|--------------:|------:|------|\n"
        )
        lines = [header]
        for r in rows:
            mem = "OK" if r.memory_ok else "FAIL"
            lines.append(
                f"| {r.tracker_name} | {r.device_name} "
                f"| {r.mean_iou:.3f} | {r.projected_fps:.1f} "
                f"| {r.projected_latency_ms:.1f} | {mem} "
                f"| {r.energy_wh_per_hour:.3f} "
                f"| {r.deployment_score:.3f} | {r.tier} |\n"
            )
        return "".join(lines)

    def to_json(self, rows: List[ScorecardRow], path: str) -> None:
        """Write the scorecard to a JSON file.

        Args:
            rows: Output of :meth:`evaluate`.
            path: Destination file path (created/overwritten).
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "scorecard": [
                {
                    "tracker": r.tracker_name,
                    "device": r.device_name,
                    "dataset": r.dataset_name,
                    "mean_iou": round(r.mean_iou, 4),
                    "projected_fps": round(r.projected_fps, 2),
                    "projected_latency_ms": round(r.projected_latency_ms, 2),
                    "memory_ok": r.memory_ok,
                    "energy_wh_per_hour": round(r.energy_wh_per_hour, 4),
                    "deployment_score": round(r.deployment_score, 4),
                    "tier": r.tier,
                }
                for r in rows
            ],
            "config": {
                "target_fps": self.target_fps,
                "marginal_fps_ratio": self.marginal_fps_ratio,
                "energy_weight": self.energy_weight,
                "sustained_seconds": self.sustained_seconds,
                "devices": self._devices,
            },
        }
        out.write_text(json.dumps(data, indent=2))

    def to_csv(self, rows: List[ScorecardRow], path: str) -> None:
        """Write the scorecard to a CSV file.

        Args:
            rows: Output of :meth:`evaluate`.
            path: Destination file path (created/overwritten).
        """
        import csv
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "tracker", "device", "dataset", "mean_iou",
            "projected_fps", "projected_latency_ms", "memory_ok",
            "energy_wh_per_hour", "deployment_score", "tier",
        ]
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({
                    "tracker": r.tracker_name,
                    "device": r.device_name,
                    "dataset": r.dataset_name,
                    "mean_iou": r.mean_iou,
                    "projected_fps": r.projected_fps,
                    "projected_latency_ms": r.projected_latency_ms,
                    "memory_ok": r.memory_ok,
                    "energy_wh_per_hour": r.energy_wh_per_hour,
                    "deployment_score": r.deployment_score,
                    "tier": r.tier,
                })

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_pair(
        self,
        result: "BenchmarkResult",
        host_profiling,
        device_name: str,
    ) -> ScorecardRow:
        """Compute a single scorecard row for one tracker × device pair."""
        sim_result = self._sim.simulate(
            host_profiling,
            device_name,
            sustained_seconds=self.sustained_seconds,
        )

        projected_fps = sim_result.estimated_fps
        projected_latency_ms = sim_result.estimated_latency_ms
        memory_ok = sim_result.fits_in_memory

        # Convert mJ/frame × fps × 3600 s/h ÷ 1000 mJ/Wh = Wh/h
        energy_wh_per_hour = (
            sim_result.estimated_energy_mj_per_frame * projected_fps * 3.6
        )

        score = self._deployment_score(
            projected_fps=projected_fps,
            memory_ok=memory_ok,
            energy_wh_per_hour=energy_wh_per_hour,
        )

        tier = self._assign_tier(projected_fps, memory_ok)

        return ScorecardRow(
            tracker_name=result.tracker_name,
            device_name=device_name,
            dataset_name=result.dataset_name,
            mean_iou=result.mean_iou,
            projected_fps=projected_fps,
            projected_latency_ms=projected_latency_ms,
            memory_ok=memory_ok,
            energy_wh_per_hour=energy_wh_per_hour,
            deployment_score=score,
            tier=tier,
        )

    def _deployment_score(
        self,
        projected_fps: float,
        memory_ok: bool,
        energy_wh_per_hour: float,
    ) -> float:
        """Compute the Deployment Score for one (tracker, device) pair.

        Formula::

            fps_term    = min(projected_fps / target_fps, 1.0)
            mem_factor  = 1.0 if memory_ok else 0.0
            energy_term = exp(-energy_wh_per_hour)   # [0, 1]; lower energy → higher score
            score       = (1 - w) * fps_term * mem_factor + w * energy_term

        where ``w = energy_weight``.
        """
        fps_term = min(projected_fps / self.target_fps, 1.0) if self.target_fps > 0 else 0.0
        mem_factor = 1.0 if memory_ok else 0.0
        energy_term = math.exp(-energy_wh_per_hour)
        w = self.energy_weight
        return (1.0 - w) * fps_term * mem_factor + w * energy_term

    def _assign_tier(self, projected_fps: float, memory_ok: bool) -> str:
        """Assign a qualitative deployment tier."""
        if not memory_ok:
            return "NOT FEASIBLE"
        ratio = projected_fps / self.target_fps
        if ratio >= 1.0:
            return "READY"
        if ratio >= self.marginal_fps_ratio:
            return "MARGINAL"
        return "NOT FEASIBLE"

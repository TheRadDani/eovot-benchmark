"""Multi-tracker experiment sweep engine for EOVOT.

Runs a configurable grid of trackers against one or more datasets and
aggregates the results into a ranked comparison table.  This is the
recommended entry point for systematic benchmarking sessions (e.g.
comparing all classical trackers on OTB-100 in a single command).

The sweep is intentionally simple: it runs each tracker sequentially so
that profiling numbers (latency, memory, energy) are not contaminated by
parallel process contention.  For large sweeps, launch multiple sweep
processes on different machines and merge the resulting JSON files.

Example (Python API)
--------------------
::

    from eovot.benchmark.sweep import SweepConfig, SweepRunner
    from eovot.datasets.base import OTBDataset
    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kcf import KCFTracker

    config = SweepConfig(
        name="classical-otb-sweep",
        trackers={"MOSSE": MOSSETracker, "KCF": KCFTracker},
        dataset=OTBDataset("/data/OTB100"),
        dataset_name="OTB100",
        max_sequences=10,
    )
    runner = SweepRunner()
    result = runner.run(config)
    print(result.summary_table())

Example (CLI)
-------------
::

    python scripts/run_sweep.py \\
        --config configs/experiments/classical_sweep.yaml
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import numpy as np

from ..datasets.base import BaseDataset
from ..trackers.base import BaseTracker
from .engine import BenchmarkEngine, BenchmarkResult


@dataclass
class SweepConfig:
    """Parameters for a multi-tracker benchmark sweep.

    Attributes
    ----------
    name:
        Identifier used in output filenames and report headers.
    trackers:
        Mapping of tracker name → tracker class.  Instances are created
        with no arguments; pass pre-constructed instances via
        :attr:`tracker_instances` if you need custom hyperparameters.
    dataset:
        An already-constructed :class:`~eovot.datasets.base.BaseDataset`
        instance.  Build it before calling :meth:`SweepRunner.run`.
    dataset_name:
        Human-readable label used in reports (e.g. ``"OTB100"``).
    max_sequences:
        Cap on the number of sequences evaluated per tracker.  ``None``
        evaluates all sequences.
    tdp_watts:
        TDP for CPU energy estimation.  ``None`` disables energy profiling.
    verbose:
        Print per-sequence progress for each tracker.
    output_dir:
        Directory where the sweep JSON report is written.
    tracker_instances:
        Optional list of pre-built tracker objects.  When provided,
        overrides *trackers*.
    """

    name: str
    dataset: BaseDataset
    dataset_name: str
    trackers: Dict[str, Type[BaseTracker]] = field(default_factory=dict)
    tracker_instances: List[BaseTracker] = field(default_factory=list)
    max_sequences: Optional[int] = None
    tdp_watts: Optional[float] = None
    verbose: bool = True
    output_dir: str = "results/"


@dataclass
class SweepResult:
    """Aggregated results from a multi-tracker sweep.

    Attributes
    ----------
    sweep_name:
        Name from the originating :class:`SweepConfig`.
    results:
        One :class:`~eovot.benchmark.engine.BenchmarkResult` per tracker.
    """

    sweep_name: str
    results: List[BenchmarkResult] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def ranking(self, sort_by: str = "mean_iou") -> List[Dict[str, Any]]:
        """Rank trackers by a scalar metric.

        Args:
            sort_by: Column to sort by.  One of ``"mean_iou"``,
                ``"mean_fps"``, ``"peak_memory_mb"``.
                IoU and FPS sort descending (higher is better);
                memory sorts ascending (lower is better).

        Returns:
            List of summary dicts, sorted by *sort_by*.
        """
        rows = [r.summary() for r in self.results]
        descending = sort_by not in {"peak_memory_mb", "mean_latency_ms"}
        rows.sort(key=lambda x: x.get(sort_by, 0), reverse=descending)
        return rows

    def summary_table(self, sort_by: str = "mean_iou") -> str:
        """Return a Markdown comparison table ranked by *sort_by*.

        Args:
            sort_by: Primary sort column (see :meth:`ranking`).

        Returns:
            Multi-line Markdown string, suitable for README embedding or
            GitHub PR descriptions.
        """
        rows = self.ranking(sort_by=sort_by)
        if not rows:
            return "_No results._"

        # Determine which optional columns are present
        has_energy = any("total_energy_j" in r for r in rows)
        has_dist = any("mean_center_distance_px" in r for r in rows)

        header_parts = ["Rank", "Tracker", "mIoU", "FPS", "Peak Mem (MB)"]
        if has_dist:
            header_parts.append("Ctr-Dist (px)")
        if has_energy:
            header_parts.append("Energy/frame (mJ)")

        sep = ["---"] * len(header_parts)
        lines = [
            "| " + " | ".join(header_parts) + " |",
            "| " + " | ".join(sep) + " |",
        ]

        for rank, row in enumerate(rows, 1):
            cells = [
                str(rank),
                row.get("tracker", "?"),
                str(row.get("mean_iou", "N/A")),
                str(row.get("mean_fps", "N/A")),
                str(row.get("peak_memory_mb", "N/A")),
            ]
            if has_dist:
                cells.append(str(row.get("mean_center_distance_px", "N/A")))
            if has_energy:
                cells.append(str(row.get("mean_energy_per_frame_mj", "N/A")))
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the full sweep to a nested dict for JSON export."""
        return {
            "sweep_name": self.sweep_name,
            "ranking": self.ranking(),
            "trackers": [r.to_dict() for r in self.results],
        }

    def save(self, output_dir: str) -> str:
        """Write the sweep result to ``<output_dir>/<sweep_name>.json``.

        Args:
            output_dir: Directory path (created if it does not exist).

        Returns:
            Absolute path of the written file.
        """
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{self.sweep_name}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return os.path.abspath(path)


class SweepRunner:
    """Execute a :class:`SweepConfig` and return a :class:`SweepResult`.

    The runner creates a fresh :class:`~eovot.benchmark.engine.BenchmarkEngine`
    for each tracker so that profiling state does not leak across runs.

    Example::

        runner = SweepRunner()
        result = runner.run(config)
        print(result.summary_table())
    """

    def run(self, config: SweepConfig) -> SweepResult:
        """Run all trackers defined in *config* and return aggregated results.

        Args:
            config: Sweep configuration.

        Returns:
            :class:`SweepResult` containing one :class:`BenchmarkResult`
            per tracker.
        """
        sweep_result = SweepResult(sweep_name=config.name)

        # Resolve tracker list — explicit instances take priority
        trackers: List[BaseTracker] = list(config.tracker_instances)
        for name, cls in config.trackers.items():
            trackers.append(cls())

        if not trackers:
            raise ValueError(
                "SweepConfig must define at least one tracker via "
                "'trackers' or 'tracker_instances'."
            )

        n_trackers = len(trackers)
        for i, tracker in enumerate(trackers, 1):
            if config.verbose:
                print(
                    f"\n[Sweep {config.name}] "
                    f"Tracker {i}/{n_trackers}: {tracker.name}"
                )

            engine = BenchmarkEngine(
                verbose=config.verbose,
                tdp_watts=config.tdp_watts,
            )
            result = engine.run(
                tracker=tracker,
                dataset=config.dataset,
                dataset_name=config.dataset_name,
                max_sequences=config.max_sequences,
            )
            sweep_result.results.append(result)

        if config.verbose:
            print(f"\n{'=' * 60}")
            print(f"  Sweep complete: {config.name}")
            print(f"{'=' * 60}")
            print(sweep_result.summary_table())

        return sweep_result

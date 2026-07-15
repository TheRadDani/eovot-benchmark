"""Full analysis pipeline for EOVOT.

Orchestrates every metric module in a single call so researchers don't
have to manually chain BenchmarkEngine, RobustnessAnalyzer,
TemporalConsistencyAnalyzer, AttributeAnalyzer, EfficiencyMetricsEngine,
and DeviceSimulator.

The pipeline produces a unified result dict that includes:

- Per-tracker accuracy metrics (IoU, success AUC, precision AUC)
- Robustness analysis (EAO, failure count, survival rate, recovery lag)
- Temporal consistency (smoothness score, jitter, velocity outlier ratio)
- Per-attribute performance breakdown (scale variation, fast motion, etc.)
- Edge Efficiency Score and Pareto-front ranking
- Device simulation table (optional)

All outputs are saved to a structured directory tree and returned as a
plain dict for programmatic post-processing or JSON export.

Example::

    from eovot.analysis.pipeline import FullAnalysisPipeline
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.registry import build_tracker

    trackers = [build_tracker("MOSSE"), build_tracker("KCF")]
    dataset  = SyntheticDataset(num_sequences=5, num_frames=80)

    pipeline = FullAnalysisPipeline(output_dir="results/full_analysis")
    report   = pipeline.run(trackers, dataset, dataset_name="Synthetic")

    print(report["leaderboard_md"])
    print(report["efficiency"]["markdown_table"])
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..benchmark.engine import BenchmarkEngine, BenchmarkResult
from ..datasets.base import BaseDataset
from ..metrics.attributes import AttributeAnalyzer, AttributePerformanceTable
from ..metrics.efficiency import EfficiencyEntry, EfficiencyMetricsEngine
from ..metrics.robustness import RobustnessAnalyzer
from ..metrics.temporal import TemporalConsistencyAnalyzer
from ..profiling.device_sim import DeviceSimulator
from ..reporting.reporter import BenchmarkReporter
from ..trackers.base import BaseTracker


class FullAnalysisPipeline:
    """Run a complete EOVOT analysis: benchmarking + all metric modules.

    Wraps :class:`~eovot.benchmark.engine.BenchmarkEngine` and automatically
    applies every analysis module after each tracker finishes, eliminating the
    boilerplate of manually chaining individual analyzers.

    Args:
        output_dir: Root directory for all output files.  A subdirectory is
            created per tracker.  Default: ``"results/full_analysis"``.
        verbose: Print per-sequence progress and analysis summaries.
            Default: ``True``.
        tdp_watts: TDP for host-machine energy estimation (passed to
            :class:`~eovot.benchmark.engine.BenchmarkEngine`).  ``None``
            disables energy profiling.  Default: ``None``.
        memory_budget_mb: Memory ceiling for
            :class:`~eovot.metrics.efficiency.EfficiencyMetricsEngine`.
            Default: ``512.0`` MB.
        device_names: List of device keys for
            :class:`~eovot.profiling.device_sim.DeviceSimulator`.  Pass
            ``None`` to simulate all built-in devices.  Pass ``[]`` to skip
            device simulation entirely.  Default: ``None`` (all devices).
        sustained_seconds: Duration of continuous tracking used for thermal
            throttling modelling in device simulation.  Default: ``60.0``.
        robustness_failure_threshold: IoU threshold below which a frame is a
            tracking failure.  Default: ``0.1``.
        max_sequences: Evaluate only the first N sequences per tracker.
            Default: ``None`` (all sequences).

    Example::

        pipeline = FullAnalysisPipeline(
            output_dir="results/paper_run",
            tdp_watts=15.0,
            device_names=["rpi4", "jetson_nano"],
        )
        report = pipeline.run(
            trackers=[MOSSETracker(), KCFTracker(), CSRTTracker()],
            dataset=SyntheticDataset(num_sequences=10, num_frames=100),
            dataset_name="Synthetic-Linear",
        )
        print(report["leaderboard_md"])
    """

    def __init__(
        self,
        output_dir: str = "results/full_analysis",
        verbose: bool = True,
        tdp_watts: Optional[float] = None,
        memory_budget_mb: float = 512.0,
        device_names: Optional[List[str]] = None,
        sustained_seconds: float = 60.0,
        robustness_failure_threshold: float = 0.1,
        max_sequences: Optional[int] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.tdp_watts = tdp_watts
        self.memory_budget_mb = memory_budget_mb
        # None  → simulate all built-in devices
        # []    → skip device simulation entirely
        self.device_names = device_names
        self.sustained_seconds = sustained_seconds
        self.robustness_failure_threshold = robustness_failure_threshold
        self.max_sequences = max_sequences

        self._engine = BenchmarkEngine(verbose=verbose, tdp_watts=tdp_watts)
        self._robustness = RobustnessAnalyzer(
            failure_threshold=robustness_failure_threshold
        )
        self._temporal = TemporalConsistencyAnalyzer()
        self._attribute = AttributeAnalyzer()
        self._efficiency = EfficiencyMetricsEngine(
            memory_budget_mb=memory_budget_mb
        )
        self._device_sim: Optional[DeviceSimulator] = (
            None if device_names == [] else DeviceSimulator()
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        trackers: List[BaseTracker],
        dataset: BaseDataset,
        dataset_name: str = "dataset",
    ) -> Dict[str, Any]:
        """Execute the full analysis pipeline on a list of trackers.

        Args:
            trackers: One or more :class:`~eovot.trackers.base.BaseTracker`
                instances to evaluate.
            dataset: :class:`~eovot.datasets.base.BaseDataset` to benchmark on.
            dataset_name: Human-readable dataset label used in output files
                and leaderboard tables.

        Returns:
            Nested dict with the following top-level keys:

            * ``"tracker_results"`` — list of per-tracker report dicts, each
              containing ``"benchmark"``, ``"robustness"``, ``"temporal"``,
              ``"attributes"``, and optionally ``"device_sim"`` sections.
            * ``"efficiency"`` — EES ranking and Pareto front across all
              trackers.
            * ``"leaderboard_md"`` — Markdown leaderboard table (accuracy +
              efficiency combined, ranked by EES).
            * ``"output_dir"`` — absolute path of the results directory.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        benchmark_results: List[BenchmarkResult] = []
        tracker_reports: List[Dict[str, Any]] = []

        for tracker in trackers:
            report = self._run_single(tracker, dataset, dataset_name)
            tracker_reports.append(report)
            benchmark_results.append(report["_benchmark_obj"])

        eff_entries = self._efficiency.rank_trackers(benchmark_results)
        efficiency_section = self._format_efficiency(eff_entries)

        leaderboard_md = self._build_leaderboard_md(tracker_reports, eff_entries)
        lb_path = self.output_dir / "leaderboard.md"
        lb_path.write_text(leaderboard_md, encoding="utf-8")

        clean_reports = [
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in tracker_reports
        ]

        output: Dict[str, Any] = {
            "tracker_results": clean_reports,
            "efficiency": efficiency_section,
            "leaderboard_md": leaderboard_md,
            "output_dir": str(self.output_dir.resolve()),
        }

        with open(self.output_dir / "full_analysis.json", "w") as fh:
            json.dump(output, fh, indent=2, default=_json_default)

        if self.verbose:
            print(f"\n{'='*60}")
            print("  FULL ANALYSIS COMPLETE")
            print(f"  Results -> {self.output_dir}")
            print(f"{'='*60}")
            print(leaderboard_md)

        return output

    # ------------------------------------------------------------------
    # Per-tracker analysis
    # ------------------------------------------------------------------

    def _run_single(
        self,
        tracker: BaseTracker,
        dataset: BaseDataset,
        dataset_name: str,
    ) -> Dict[str, Any]:
        """Benchmark one tracker and run all metric analyses."""
        result = self._engine.run(
            tracker=tracker,
            dataset=dataset,
            dataset_name=dataset_name,
            max_sequences=self.max_sequences,
        )

        tracker_dir = self.output_dir / tracker.name
        tracker_dir.mkdir(parents=True, exist_ok=True)

        reporter = BenchmarkReporter(output_dir=str(tracker_dir))
        reporter.save_all(result.to_dict(), name=f"{tracker.name}-{dataset_name}")

        # --- Robustness ---
        seq_ious = {r.sequence_name: r.ious for r in result.sequence_results}
        rob = self._robustness.analyze_benchmark(
            seq_ious, tracker_name=tracker.name
        )

        # --- Temporal consistency ---
        seq_preds = {
            r.sequence_name: r.predictions
            for r in result.sequence_results
            if r.predictions is not None
        }
        temp = self._temporal.analyze_benchmark(
            seq_preds, tracker_name=tracker.name
        )

        # --- Attribute breakdown ---
        attr_table: AttributePerformanceTable = self._attribute.breakdown(result)

        # --- Device simulation (optional) ---
        device_section: Optional[List[Dict]] = None
        if self._device_sim is not None and result.sequence_results:
            first_prof = result.sequence_results[0].profiling
            sim_results = self._device_sim.simulate_all(
                result=first_prof,
                sustained_seconds=self.sustained_seconds,
                device_names=self.device_names,
            )
            device_section = self._device_sim.to_summary_dict(sim_results)

        tracker_analysis: Dict[str, Any] = {
            "tracker_name": tracker.name,
            "dataset_name": dataset_name,
            "benchmark": result.summary(),
            "robustness": rob["aggregate"],
            "temporal": temp["aggregate"],
            "attributes": attr_table.to_dict(),
        }
        if device_section is not None:
            tracker_analysis["device_sim"] = device_section

        with open(tracker_dir / "analysis.json", "w") as fh:
            json.dump(
                {k: v for k, v in tracker_analysis.items() if not k.startswith("_")},
                fh,
                indent=2,
                default=_json_default,
            )

        tracker_analysis["_benchmark_obj"] = result
        return tracker_analysis

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_efficiency(self, entries: List[EfficiencyEntry]) -> Dict[str, Any]:
        """Convert efficiency entries to a JSON-serialisable dict."""
        return {
            "memory_budget_mb": self.memory_budget_mb,
            "ranking": [
                {
                    "rank": i + 1,
                    "tracker": e.tracker_name,
                    "dataset": e.dataset_name,
                    "mean_iou": round(e.mean_iou, 4),
                    "fps": round(e.fps, 2),
                    "peak_memory_mb": round(e.peak_memory_mb, 1),
                    "ees": round(e.ees, 4),
                    "pareto": e.on_pareto_front,
                }
                for i, e in enumerate(entries)
            ],
            "markdown_table": self._efficiency.to_markdown_table(entries),
        }

    def _build_leaderboard_md(
        self,
        tracker_reports: List[Dict[str, Any]],
        eff_entries: List[EfficiencyEntry],
    ) -> str:
        """Build a combined accuracy + efficiency Markdown leaderboard."""
        ees_map = {e.tracker_name: e for e in eff_entries}
        rows = []
        for r in tracker_reports:
            bm = r.get("benchmark", {})
            rob = r.get("robustness", {})
            temp = r.get("temporal", {})
            name = r.get("tracker_name", "?")
            ee = ees_map.get(name)
            rows.append({
                "tracker": name,
                "mIoU": float(bm.get("mean_iou", 0.0)),
                "success_auc": float(bm.get("success_auc", 0.0)),
                "fps": float(bm.get("mean_fps", 0.0)),
                "mem_mb": float(bm.get("peak_memory_mb", 0.0)),
                "eao": float(rob.get("mean_eao", 0.0)),
                "failures": int(rob.get("total_failures", 0)),
                "smoothness": float(temp.get("mean_smoothness_score", 0.0)),
                "ees": ee.ees if ee else 0.0,
                "pareto": ee.on_pareto_front if ee else False,
            })

        rows.sort(key=lambda x: x["ees"], reverse=True)

        lines = [
            "# EOVOT Full Analysis Leaderboard\n",
            "| Rank | Tracker | mIoU | AUC | FPS | Mem(MB) | EAO | Failures | Smoothness | EES | Pareto |",
            "|------|---------|-----:|----:|----:|--------:|----:|---------:|-----------:|----:|:------:|",
        ]
        for rank, row in enumerate(rows, start=1):
            pareto = "✓" if row["pareto"] else ""
            lines.append(
                f"| {rank} | {row['tracker']} "
                f"| {row['mIoU']:.4f} | {row['success_auc']:.4f} "
                f"| {row['fps']:.1f} | {row['mem_mb']:.1f} "
                f"| {row['eao']:.4f} | {row['failures']} "
                f"| {row['smoothness']:.4f} | {row['ees']:.4f} | {pareto} |"
            )
        lines.append("")
        return "\n".join(lines)


# ------------------------------------------------------------------
# JSON serialisation helper
# ------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """JSON fallback for non-standard types (numpy scalars, dataclasses, etc.)."""
    if hasattr(obj, "item"):       # numpy scalar
        return obj.item()
    if hasattr(obj, "tolist"):     # numpy array
        return obj.tolist()
    if hasattr(obj, "to_dict"):    # EOVOT result dataclasses
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)

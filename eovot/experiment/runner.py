"""Systematic multi-tracker experiment runner for EOVOT.

:class:`ExperimentRunner` executes a set of trackers against a dataset,
captures a reproducibility snapshot, ranks results in a leaderboard, and
saves everything to a structured output directory.  Interrupted runs can
be resumed with ``resume=True``.

Typical usage::

    import yaml
    from eovot.experiment.runner import ExperimentRunner

    with open("configs/experiments/multi_tracker.yaml") as f:
        config = yaml.safe_load(f)

    runner = ExperimentRunner(output_dir="results/experiments", resume=True)
    output = runner.run_from_config(config)
    print(output["leaderboard"])

Config schema::

    experiment:
      name: "classical-comparison"   # used as subdirectory name
      seed: 42                        # captured in snapshot; not used for seeding
      tdp_watts: null                 # float or null; enables energy profiling

    dataset:
      loader: OTBDataset              # OTBDataset | GOT10kDataset | LaSOTDataset
      root: /data/OTB100
      name: OTB100                    # human-readable label in reports
      split: val                      # for GOT10k / LaSOT
      max_sequences: null             # int or null

    trackers:
      - name: MOSSE
        params: {}
      - name: KCF
        params:
          learning_rate: 0.125
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..benchmark.engine import BenchmarkEngine
from ..reporting.reporter import BenchmarkReporter
from .snapshot import ReproducibilitySnapshot


class ExperimentRunner:
    """Run multiple trackers on a dataset in a reproducible, resumable way.

    Args:
        output_dir: Root directory where all experiment outputs are written.
            A subdirectory named after ``experiment.name`` is created inside.
        verbose:    Print per-sequence benchmark progress.  Default ``True``.
        resume:     When ``True``, skip any tracker whose per-tracker JSON
                    result file already exists in the experiment directory.
                    Useful for continuing interrupted long runs.
    """

    def __init__(
        self,
        output_dir: str = "results/experiments",
        verbose: bool = True,
        resume: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.resume = resume

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_from_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an experiment defined by a config dict.

        The optional ``edge_profile`` section triggers edge efficiency analysis:

        .. code-block:: yaml

            edge_profile:
              devices: [rpi4, jetson_nano, jetson_xnx]  # null = all devices
              sustained_seconds: 60.0
              memory_budget_mb: 512.0

        When present, :meth:`run_from_config` additionally produces:

        * ``edge_leaderboard.md`` — EES-ranked table + per-device projected FPS
        * ``edge_projection.json`` — device simulation data for all trackers

        Args:
            config: Nested dict matching the schema described in this
                module's docstring.  Typically loaded from YAML.

        Returns:
            Dict with keys:

            * ``"metadata"`` — experiment name, snapshot, wall-clock timings.
            * ``"results"`` — list of per-tracker result dicts.
            * ``"leaderboard"`` — Markdown accuracy leaderboard.
            * ``"edge_leaderboard"`` — Markdown edge efficiency leaderboard
              (only present when ``edge_profile`` is configured).
        """
        from ..benchmark.engine import BenchmarkResult  # local import avoids circular ref

        exp_cfg = config.get("experiment", {})
        exp_name = exp_cfg.get("name", "unnamed-experiment")
        seed = exp_cfg.get("seed", None)
        tdp_watts = exp_cfg.get("tdp_watts", None)

        snapshot = ReproducibilitySnapshot.capture(seed=seed)

        exp_dir = self.output_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        dataset_cfg = config.get("dataset", {})
        tracker_cfgs = config.get("trackers", [])
        dataset_name = dataset_cfg.get("name", dataset_cfg.get("loader", "unknown"))
        max_sequences = dataset_cfg.get("max_sequences", None)
        edge_cfg: Optional[Dict[str, Any]] = config.get("edge_profile", None)

        engine = BenchmarkEngine(verbose=self.verbose, tdp_watts=tdp_watts)
        reporter = BenchmarkReporter(output_dir=str(exp_dir))

        all_results: List[Dict] = []
        all_benchmark_results: List[BenchmarkResult] = []
        run_timings: Dict[str, float] = {}

        for tracker_cfg in tracker_cfgs:
            tracker_name = tracker_cfg["name"]
            result_path = exp_dir / f"{tracker_name}-{dataset_name}.json"

            if self.resume and result_path.exists():
                if self.verbose:
                    print(f"[resume] Skipping {tracker_name} — result found at {result_path}")
                with open(result_path) as fh:
                    all_results.append(json.load(fh))
                # No BenchmarkResult available for resumed runs — edge projection skipped
                continue

            tracker = self._build_tracker(tracker_cfg)
            dataset = self._build_dataset(dataset_cfg)

            t0 = time.perf_counter()
            result = engine.run(
                tracker=tracker,
                dataset=dataset,
                dataset_name=dataset_name,
                max_sequences=max_sequences,
            )
            run_timings[tracker_name] = round(time.perf_counter() - t0, 3)

            result_dict = result.to_dict()
            reporter.save_all(result_dict, name=f"{tracker_name}-{dataset_name}")
            all_results.append(result_dict)
            all_benchmark_results.append(result)

        leaderboard = self._build_leaderboard(all_results)
        (exp_dir / "leaderboard.md").write_text(leaderboard, encoding="utf-8")

        output: Dict[str, Any] = {
            "metadata": {
                "experiment": exp_name,
                "reproducibility": snapshot.to_dict(),
                "run_timings_s": run_timings,
            },
            "results": all_results,
            "leaderboard": leaderboard,
        }

        # Edge efficiency analysis — only when edge_profile is configured AND
        # we have live BenchmarkResult objects (not resumed from disk).
        edge_leaderboard: Optional[str] = None
        if edge_cfg is not None and all_benchmark_results:
            edge_leaderboard = self._build_edge_leaderboard(
                all_benchmark_results, edge_cfg, exp_dir
            )
            output["edge_leaderboard"] = edge_leaderboard

        with open(exp_dir / "metadata.json", "w") as fh:
            json.dump(output["metadata"], fh, indent=2)

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"  EXPERIMENT COMPLETE: {exp_name}")
            print(f"  Results → {exp_dir}")
            print(f"{'=' * 60}")
            print(leaderboard)
            if edge_leaderboard:
                print(edge_leaderboard)

        return output

    # ------------------------------------------------------------------
    # Leaderboard generation
    # ------------------------------------------------------------------

    def _build_leaderboard(
        self,
        results: List[Dict],
        memory_budget_mb: float = 512.0,
    ) -> str:
        """Build a Markdown leaderboard ranked by mIoU (descending) with EES column.

        The Edge Efficiency Score (EES) column is added alongside accuracy and
        throughput metrics so the leaderboard reflects EOVOT's core thesis —
        edge suitability requires evaluating accuracy **and** efficiency jointly.

        Args:
            results: List of result dicts from
                :meth:`~eovot.benchmark.engine.BenchmarkResult.to_dict`.
            memory_budget_mb: Memory ceiling for EES computation.  Default 512 MB.

        Returns:
            Multi-line Markdown string ready for writing to a ``.md`` file.
        """
        import math  # local import keeps module import order clean

        if not results:
            return "No results to display.\n"

        def _ees(miou: float, fps: float, mem_mb: float) -> float:
            if fps <= 0 or miou < 0:
                return 0.0
            return (miou * math.log1p(fps)) / (1.0 + mem_mb / memory_budget_mb)

        rows = []
        for r in results:
            s = r.get("summary", {})
            miou = float(s.get("mean_iou", 0.0))
            fps = float(s.get("mean_fps", 0.0))
            mem_mb = float(s.get("peak_memory_mb", 0.0))
            rows.append(
                {
                    "tracker": s.get("tracker", "?"),
                    "dataset": s.get("dataset", "?"),
                    "mIoU": miou,
                    "fps": fps,
                    "mem_mb": mem_mb,
                    "ees": _ees(miou, fps, mem_mb),
                    "n_seq": int(s.get("num_sequences", 0)),
                }
            )

        rows.sort(key=lambda x: x["mIoU"], reverse=True)

        lines = [
            "# EOVOT Experiment Leaderboard\n",
            "| Rank | Tracker | Dataset | mIoU | FPS | Mem (MB) | EES | Sequences |",
            "|------|---------|---------|-----:|----:|---------:|----:|----------:|",
        ]
        for rank, row in enumerate(rows, start=1):
            lines.append(
                f"| {rank} | {row['tracker']} | {row['dataset']} "
                f"| {row['mIoU']:.4f} | {row['fps']:.1f} "
                f"| {row['mem_mb']:.1f} | {row['ees']:.4f} | {row['n_seq']} |"
            )
        lines.append("")
        return "\n".join(lines)

    def _build_edge_leaderboard(
        self,
        benchmark_results: "List",
        edge_cfg: Dict[str, Any],
        exp_dir: "Path",
    ) -> str:
        """Build edge efficiency leaderboard + per-device projection tables.

        Uses :class:`~eovot.metrics.efficiency.EfficiencyMetricsEngine` for EES
        ranking and :class:`~eovot.profiling.device_sim.DeviceSimulator` to
        project each tracker's host-measured latency onto target edge hardware.

        Saves two files into *exp_dir*:

        * ``edge_leaderboard.md`` — EES-ranked table + device projection tables
        * ``edge_projection.json`` — raw projection data for downstream analysis

        Args:
            benchmark_results: Live :class:`~eovot.benchmark.engine.BenchmarkResult`
                objects (not available for resumed runs).
            edge_cfg: The ``edge_profile`` section from the experiment config.
            exp_dir: Experiment output directory (:class:`pathlib.Path`).

        Returns:
            The generated Markdown string (also written to ``edge_leaderboard.md``).
        """
        from ..metrics.efficiency import EfficiencyMetricsEngine
        from ..profiling.device_sim import DeviceSimulator
        from ..profiling.profiler import ProfilingResult

        memory_budget_mb = float(edge_cfg.get("memory_budget_mb", 512.0))
        sustained_seconds = float(edge_cfg.get("sustained_seconds", 0.0))
        device_names: Optional[List[str]] = edge_cfg.get("devices", None)

        eff_engine = EfficiencyMetricsEngine(memory_budget_mb=memory_budget_mb)
        sim = DeviceSimulator()

        ranking = eff_engine.rank_trackers(benchmark_results)

        sections: List[str] = [
            "# EOVOT Edge Efficiency Leaderboard\n",
            f"> Memory budget: {memory_budget_mb:.0f} MB  "
            f"| Sustained load: {sustained_seconds:.0f} s\n",
            "## EES Ranking\n",
            eff_engine.to_markdown_table(ranking),
            "",
        ]

        all_projection_data: Dict[str, Any] = {}

        for result in benchmark_results:
            # Build a representative ProfilingResult from mean sequence stats
            seq_results = result.sequence_results
            if not seq_results:
                continue

            import numpy as np
            mean_lat = float(np.mean([s.profiling.latency_mean_ms for s in seq_results]))
            mean_fps = float(np.mean([s.profiling.fps for s in seq_results]))
            peak_mem = float(np.max([s.profiling.peak_memory_mb for s in seq_results]))
            std_lat = float(np.std([s.profiling.latency_mean_ms for s in seq_results]))
            p95_lat = float(np.percentile(
                [s.profiling.latency_mean_ms for s in seq_results], 95
            ))
            frame_count = sum(s.profiling.frame_count for s in seq_results)

            prof_result = ProfilingResult(
                tracker_name=result.tracker_name,
                frame_count=frame_count,
                fps=mean_fps,
                latency_mean_ms=mean_lat,
                latency_std_ms=std_lat,
                latency_p95_ms=p95_lat,
                peak_memory_mb=peak_mem,
            )

            device_sims = sim.simulate_all(
                prof_result,
                sustained_seconds=sustained_seconds,
                device_names=device_names,
            )

            sections.append(f"\n### {result.tracker_name} — Device Projection\n")
            sections.append(sim.to_markdown_table(device_sims))
            sections.append("")

            all_projection_data[result.tracker_name] = sim.to_summary_dict(device_sims)

        edge_md = "\n".join(sections)
        (exp_dir / "edge_leaderboard.md").write_text(edge_md, encoding="utf-8")
        with open(exp_dir / "edge_projection.json", "w") as fh:
            json.dump(all_projection_data, fh, indent=2)

        return edge_md

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_dataset(cfg: Dict):
        """Instantiate a dataset from a config dict."""
        from ..datasets.base import OTBDataset
        from ..datasets.got10k import GOT10kDataset
        from ..datasets.lasot import LaSOTDataset
        from ..datasets.synthetic import SyntheticDataset

        loader_name = cfg.get("loader", "OTBDataset")

        if loader_name == "SyntheticDataset":
            frame_size = cfg.get("frame_size", [320, 240])
            bbox_size = cfg.get("bbox_size", [40, 40])
            return SyntheticDataset(
                num_sequences=cfg.get("num_sequences", 10),
                num_frames=cfg.get("num_frames", 100),
                frame_size=tuple(frame_size),
                bbox_size=tuple(bbox_size),
                motion=cfg.get("motion", "linear"),
                seed=cfg.get("seed", 0),
            )

        loaders = {
            "OTBDataset": OTBDataset,
            "GOT10kDataset": GOT10kDataset,
            "LaSOTDataset": LaSOTDataset,
        }
        if loader_name not in loaders:
            raise ValueError(
                f"Unknown dataset loader '{loader_name}'. "
                f"Available: {['SyntheticDataset'] + list(loaders)}"
            )
        cls = loaders[loader_name]
        root = cfg["root"]

        if loader_name == "OTBDataset":
            return cls(root=root)
        split = cfg.get("split", "val")
        max_seq = cfg.get("max_sequences", None)
        return cls(root=root, split=split, max_sequences=max_seq)

    @staticmethod
    def _build_tracker(cfg: Dict):
        """Instantiate a tracker from a config dict."""
        from ..trackers.csrt import CSRTTracker
        from ..trackers.kcf import KCFTracker
        from ..trackers.median_flow import MedianFlowTracker
        from ..trackers.mil import MILTracker
        from ..trackers.mosse import MOSSETracker

        registry = {
            "MOSSE": MOSSETracker,
            "KCF": KCFTracker,
            "CSRT": CSRTTracker,
            "MIL": MILTracker,
            "MedianFlow": MedianFlowTracker,
        }
        name = cfg["name"]
        if name not in registry:
            raise ValueError(
                f"Unknown tracker '{name}'. Available: {list(registry)}"
            )
        params = cfg.get("params", {}) or {}
        return registry[name](**params)

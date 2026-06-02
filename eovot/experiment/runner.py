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

import numpy as np

from ..benchmark.engine import BenchmarkEngine, BenchmarkResult
from ..profiling.profiler import ProfilingResult
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

        Args:
            config: Nested dict matching the schema described in this
                module's docstring.  Typically loaded from YAML.

        Returns:
            Dict with three keys:

            * ``"metadata"`` — experiment name, reproducibility snapshot,
              per-tracker wall-clock timings.
            * ``"results"`` — list of per-tracker result dicts (same format
              as :meth:`~eovot.benchmark.engine.BenchmarkResult.to_dict`).
            * ``"leaderboard"`` — Markdown string ranking trackers by mIoU.
        """
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

        engine = BenchmarkEngine(verbose=self.verbose, tdp_watts=tdp_watts)
        reporter = BenchmarkReporter(output_dir=str(exp_dir))

        all_results: List[Dict] = []
        benchmark_results: Dict[str, BenchmarkResult] = {}
        run_timings: Dict[str, float] = {}

        for tracker_cfg in tracker_cfgs:
            tracker_name = tracker_cfg["name"]
            result_path = exp_dir / f"{tracker_name}-{dataset_name}.json"

            if self.resume and result_path.exists():
                if self.verbose:
                    print(f"[resume] Skipping {tracker_name} — result found at {result_path}")
                with open(result_path) as fh:
                    all_results.append(json.load(fh))
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
            benchmark_results[tracker_name] = result

        leaderboard = self._build_leaderboard(all_results)
        leaderboard_path = exp_dir / "leaderboard.md"
        leaderboard_path.write_text(leaderboard, encoding="utf-8")

        device_report_output: Optional[Dict] = None
        device_sim_cfg = exp_cfg.get("device_simulation", {})
        if device_sim_cfg.get("enabled", False):
            device_report_output = self._run_device_report(
                all_results=all_results,
                benchmark_results=benchmark_results,
                device_sim_cfg=device_sim_cfg,
                exp_dir=exp_dir,
            )

        metadata: Dict[str, Any] = {
            "experiment": exp_name,
            "reproducibility": snapshot.to_dict(),
            "run_timings_s": run_timings,
        }
        with open(exp_dir / "metadata.json", "w") as fh:
            json.dump(metadata, fh, indent=2)

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"  EXPERIMENT COMPLETE: {exp_name}")
            print(f"  Results → {exp_dir}")
            print(f"{'=' * 60}")
            print(leaderboard)

        output: Dict[str, Any] = {
            "metadata": metadata,
            "results": all_results,
            "leaderboard": leaderboard,
        }
        if device_report_output is not None:
            output["device_report"] = device_report_output
        return output

    # ------------------------------------------------------------------
    # Device deployment report
    # ------------------------------------------------------------------

    def _run_device_report(
        self,
        all_results: List[Dict],
        benchmark_results: Dict[str, BenchmarkResult],
        device_sim_cfg: Dict[str, Any],
        exp_dir: Path,
    ) -> Dict[str, Any]:
        """Project tracker profiling results onto an edge device fleet.

        Uses live BenchmarkResult objects when available (trackers that ran
        in this session) and reconstructs ProfilingResult from saved JSON for
        any resumed tracker.

        Args:
            all_results: List of result dicts (may include resumed trackers).
            benchmark_results: Live BenchmarkResult objects for this session.
            device_sim_cfg: ``experiment.device_simulation`` config sub-dict.
            exp_dir: Experiment output directory for writing report files.

        Returns:
            Dict with ``"markdown_path"`` and ``"json_path"`` keys.
        """
        from .device_report import DeviceReport

        dr = DeviceReport(
            devices=device_sim_cfg.get("devices"),
            sustained_seconds=float(device_sim_cfg.get("sustained_seconds", 0.0)),
            host_calibration_factor=float(
                device_sim_cfg.get("host_calibration_factor", 1.0)
            ),
        )

        # Build profiling map — prefer live objects, fall back to JSON reconstruction.
        profiling_map: Dict[str, ProfilingResult] = {}
        for r in all_results:
            name = r.get("summary", {}).get("tracker", "")
            if not name:
                continue
            if name in benchmark_results:
                try:
                    profiling_map[name] = benchmark_results[name].to_profiling_result()
                except ValueError:
                    pass
            else:
                prof = self._profiling_from_dict(r, name)
                if prof is not None:
                    profiling_map[name] = prof

        if not profiling_map:
            return {}

        sim_results = dr.run(profiling_map)

        md_path = exp_dir / "device_report.md"
        md_path.write_text(dr.to_markdown(sim_results), encoding="utf-8")

        json_path = exp_dir / "device_report.json"
        with open(json_path, "w") as fh:
            json.dump(dr.to_summary_dicts(sim_results), fh, indent=2)

        if self.verbose:
            print(f"\n  Device report → {md_path}")

        return {
            "markdown_path": str(md_path),
            "json_path": str(json_path),
        }

    @staticmethod
    def _profiling_from_dict(
        result_dict: Dict, tracker_name: str
    ) -> Optional[ProfilingResult]:
        """Reconstruct an approximate ProfilingResult from a saved JSON dict.

        Used for resumed trackers whose live BenchmarkResult is unavailable.
        Latency is estimated from the per-sequence mean_latency_ms values
        stored in the JSON.

        Args:
            result_dict: Dict in the format produced by BenchmarkResult.to_dict().
            tracker_name: Name to embed in the returned ProfilingResult.

        Returns:
            Reconstructed ProfilingResult, or None if the dict lacks the
            required fields.
        """
        summary = result_dict.get("summary", {})
        seqs = result_dict.get("sequences", [])
        if not seqs:
            return None
        latencies = [
            float(s["mean_latency_ms"]) for s in seqs if "mean_latency_ms" in s
        ]
        if not latencies:
            return None
        mean_lat = float(np.mean(latencies))
        return ProfilingResult(
            tracker_name=tracker_name,
            frame_count=len(seqs) * 100,  # rough estimate; frame_count not in summary
            fps=1_000.0 / mean_lat if mean_lat > 0 else 0.0,
            latency_mean_ms=mean_lat,
            latency_std_ms=float(np.std(latencies)),
            latency_p95_ms=float(np.percentile(latencies, 95)),
            peak_memory_mb=float(summary.get("peak_memory_mb", 0.0)),
        )

    # ------------------------------------------------------------------
    # Leaderboard generation
    # ------------------------------------------------------------------

    def _build_leaderboard(self, results: List[Dict]) -> str:
        """Build a Markdown leaderboard ranked by success AUC (descending).

        Success AUC is the standard primary scalar for VOT benchmarks (OTB,
        GOT-10k, LaSOT).  Falls back to mIoU when success AUC is absent.

        Args:
            results: List of result dicts from
                :meth:`~eovot.benchmark.engine.BenchmarkResult.to_dict`.

        Returns:
            Multi-line Markdown string ready for writing to a ``.md`` file.
        """
        if not results:
            return "No results to display.\n"

        rows = []
        for r in results:
            s = r.get("summary", {})
            miou = float(s.get("mean_iou", 0.0))
            sauc = float(s.get("success_auc", miou))   # fall back to mIoU
            pauc = float(s.get("precision_auc", 0.0))
            rows.append(
                {
                    "tracker": s.get("tracker", "?"),
                    "dataset": s.get("dataset", "?"),
                    "mIoU": miou,
                    "success_auc": sauc,
                    "precision_auc": pauc,
                    "fps": float(s.get("mean_fps", 0.0)),
                    "mem_mb": float(s.get("peak_memory_mb", 0.0)),
                    "n_seq": int(s.get("num_sequences", 0)),
                }
            )

        rows.sort(key=lambda x: x["success_auc"], reverse=True)

        lines = [
            "# EOVOT Experiment Leaderboard\n",
            "| Rank | Tracker | Dataset | mIoU | Success AUC | Precision AUC | FPS | Mem (MB) | Sequences |",
            "|------|---------|---------|-----:|------------:|--------------:|----:|---------:|----------:|",
        ]
        for rank, row in enumerate(rows, start=1):
            lines.append(
                f"| {rank} | {row['tracker']} | {row['dataset']} "
                f"| {row['mIoU']:.4f} | {row['success_auc']:.4f} "
                f"| {row['precision_auc']:.4f} | {row['fps']:.1f} "
                f"| {row['mem_mb']:.1f} | {row['n_seq']} |"
            )
        lines.append("")
        return "\n".join(lines)

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

        loaders = {
            "OTBDataset": OTBDataset,
            "GOT10kDataset": GOT10kDataset,
            "LaSOTDataset": LaSOTDataset,
            "SyntheticDataset": SyntheticDataset,
        }
        loader_name = cfg.get("loader", "OTBDataset")
        if loader_name not in loaders:
            raise ValueError(
                f"Unknown dataset loader '{loader_name}'. "
                f"Available: {list(loaders)}"
            )
        cls = loaders[loader_name]

        if loader_name == "SyntheticDataset":
            frame_size = cfg.get("frame_size", [320, 240])
            bbox_size = cfg.get("bbox_size", [40, 40])
            return cls(
                num_sequences=cfg.get("num_sequences", 10),
                num_frames=cfg.get("num_frames", 100),
                frame_size=tuple(frame_size),
                bbox_size=tuple(bbox_size),
                motion=cfg.get("motion", "linear"),
                seed=cfg.get("seed", 42),
            )

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

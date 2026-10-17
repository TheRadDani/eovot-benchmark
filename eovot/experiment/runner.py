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

        leaderboard = self._build_leaderboard(all_results)
        leaderboard_path = exp_dir / "leaderboard.md"
        leaderboard_path.write_text(leaderboard, encoding="utf-8")

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

        return {
            "metadata": metadata,
            "results": all_results,
            "leaderboard": leaderboard,
        }

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
        from ..datasets.challenge import ChallengeDataset

        loaders = {
            "OTBDataset": OTBDataset,
            "GOT10kDataset": GOT10kDataset,
            "LaSOTDataset": LaSOTDataset,
            "SyntheticDataset": SyntheticDataset,
            "ChallengeDataset": ChallengeDataset,
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

        if loader_name == "ChallengeDataset":
            frame_size = cfg.get("frame_size", [320, 240])
            bbox_size = cfg.get("bbox_size", [40, 40])
            return cls(
                challenge=cfg.get("challenge", "occlusion"),
                num_sequences=cfg.get("num_sequences", 10),
                num_frames=cfg.get("num_frames", 150),
                frame_size=tuple(frame_size),
                bbox_size=tuple(bbox_size),
                seed=cfg.get("seed", 42),
                n_distractors=cfg.get("n_distractors", 3),
            )

        root = cfg["root"]
        if loader_name == "OTBDataset":
            return cls(root=root)
        split = cfg.get("split", "val")
        max_seq = cfg.get("max_sequences", None)
        return cls(root=root, split=split, max_sequences=max_seq)

    @staticmethod
    def _build_tracker(cfg: Dict):
        """Instantiate a tracker from a config dict.

        Delegates to the single shared registry in
        :mod:`eovot.trackers.registry` so every tracker registered there
        (including new ones) is automatically available here too.
        """
        from ..trackers.registry import build_tracker

        params = cfg.get("params", {}) or {}
        return build_tracker(cfg["name"], **params)

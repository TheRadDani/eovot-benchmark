"""Multi-dataset cross-evaluation engine for EOVOT.

Addresses a critical gap in the framework: running the same set of trackers
across multiple datasets and aggregating results into a unified, weighted
cross-dataset leaderboard.

Standard VOT benchmark practice requires evaluating trackers on several
datasets (OTB, GOT-10k, LaSOT) to demonstrate generalisation; a tracker
that tops one dataset's leaderboard may fail on another.  This module makes
that workflow first-class in EOVOT.

Cross-Dataset Aggregation
~~~~~~~~~~~~~~~~~~~~~~~~~
Metrics are aggregated per tracker using **frame-count weighting** so that
long sequences (e.g. LaSOT at ~2 500 frames) contribute proportionally more
than short ones (OTB at ~500 frames):

    M_agg = Σ_i (M_i × F_i) / Σ_i F_i

where ``F_i`` is the total number of frames evaluated on dataset ``i``.

Typical usage::

    from eovot.benchmark.multi_dataset import MultiDatasetEvaluator
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.registry import build_tracker

    ds1 = (SyntheticDataset(num_sequences=3, num_frames=80, motion="linear"),  "Syn-Linear")
    ds2 = (SyntheticDataset(num_sequences=3, num_frames=80, motion="random"),  "Syn-Random")

    evaluator = MultiDatasetEvaluator(verbose=True)
    result = evaluator.run(
        trackers=[build_tracker("MOSSE"), build_tracker("KCF")],
        datasets=[ds1, ds2],
    )
    print(result.cross_dataset_table())
    result.save("results/multi_dataset.json")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ..datasets.base import BaseDataset
from ..trackers.base import BaseTracker
from .engine import BenchmarkEngine, BenchmarkResult


@dataclass
class CrossDatasetEntry:
    """Per-tracker aggregated summary across all evaluated datasets.

    Attributes:
        tracker_name: Human-readable tracker identifier.
        dataset_names: Ordered list of dataset names included in the aggregate.
        mean_iou: Frame-count-weighted mean IoU across datasets.
        mean_fps: Frame-count-weighted mean FPS across datasets.
        peak_memory_mb: Maximum peak RSS across all dataset runs.
        mean_success_auc: Weighted success-curve AUC, or ``None`` if not
            available for all datasets.
        mean_precision_auc: Weighted precision-curve AUC, or ``None`` likewise.
        mean_energy_per_frame_mj: Weighted mean per-frame energy in mJ,
            or ``None`` when energy profiling was not enabled.
        total_frames: Total number of frames evaluated across all datasets.
        per_dataset: ``{dataset_name: summary_dict}`` breakdown.
    """

    tracker_name: str
    dataset_names: List[str]
    mean_iou: float
    mean_fps: float
    peak_memory_mb: float
    mean_success_auc: Optional[float]
    mean_precision_auc: Optional[float]
    mean_energy_per_frame_mj: Optional[float]
    total_frames: int
    per_dataset: Dict[str, Dict]

    def __str__(self) -> str:
        return (
            f"CrossDatasetEntry({self.tracker_name})  "
            f"datasets={len(self.dataset_names)}  "
            f"mIoU={self.mean_iou:.4f}  FPS={self.mean_fps:.1f}  "
            f"frames={self.total_frames}"
        )


@dataclass
class MultiDatasetResult:
    """Complete results from a multi-dataset evaluation run.

    Attributes:
        benchmark_results: One :class:`~eovot.benchmark.engine.BenchmarkResult`
            per (tracker, dataset) pair, in evaluation order.
    """

    benchmark_results: List[BenchmarkResult] = field(default_factory=list)

    def aggregate(self) -> List[CrossDatasetEntry]:
        """Build per-tracker cross-dataset summaries using frame-count weighting.

        Returns:
            List of :class:`CrossDatasetEntry` sorted by weighted mIoU
            descending (best tracker first).
        """
        by_tracker: Dict[str, List[BenchmarkResult]] = {}
        for r in self.benchmark_results:
            by_tracker.setdefault(r.tracker_name, []).append(r)

        entries: List[CrossDatasetEntry] = []
        for tracker_name, results in sorted(by_tracker.items()):
            frame_counts = [
                sum(len(sr.ious) for sr in r.sequence_results) for r in results
            ]
            total = sum(frame_counts)
            if total == 0:
                weights = [1.0 / len(results)] * len(results)
            else:
                weights = [f / total for f in frame_counts]

            def _wavg(values: list) -> float:
                return float(sum(w * v for w, v in zip(weights, values)))

            mean_iou = _wavg([r.mean_iou for r in results])
            mean_fps = _wavg([r.mean_fps for r in results])
            peak_mem = float(max(r.peak_memory_mb for r in results))

            mean_sauc: Optional[float] = None
            if all(r.mean_success_auc is not None for r in results):
                mean_sauc = _wavg([r.mean_success_auc for r in results])  # type: ignore[arg-type]

            mean_pauc: Optional[float] = None
            if all(r.mean_precision_auc is not None for r in results):
                mean_pauc = _wavg([r.mean_precision_auc for r in results])  # type: ignore[arg-type]

            mean_energy: Optional[float] = None
            if all(r.mean_energy_per_frame_mj is not None for r in results):
                mean_energy = _wavg([r.mean_energy_per_frame_mj for r in results])  # type: ignore[arg-type]

            entries.append(
                CrossDatasetEntry(
                    tracker_name=tracker_name,
                    dataset_names=[r.dataset_name for r in results],
                    mean_iou=mean_iou,
                    mean_fps=mean_fps,
                    peak_memory_mb=peak_mem,
                    mean_success_auc=mean_sauc,
                    mean_precision_auc=mean_pauc,
                    mean_energy_per_frame_mj=mean_energy,
                    total_frames=total,
                    per_dataset={r.dataset_name: r.summary() for r in results},
                )
            )

        entries.sort(key=lambda e: e.mean_iou, reverse=True)
        return entries

    def cross_dataset_table(self) -> str:
        """Format a Markdown leaderboard table for the cross-dataset evaluation.

        Columns included conditionally:
        - **Success AUC** — only when at least one entry has it.
        - **Energy (mJ/fr)** — only when energy profiling was enabled.

        Returns:
            Multi-line Markdown string ready to embed in reports or READMEs.
        """
        entries = self.aggregate()
        if not entries:
            return "_No results to display._"

        has_sauc = any(e.mean_success_auc is not None for e in entries)
        has_energy = any(e.mean_energy_per_frame_mj is not None for e in entries)

        cols = ["Rank", "Tracker", "Datasets", "mIoU", "FPS", "Mem (MB)", "Frames"]
        if has_sauc:
            cols.append("Success AUC")
        if has_energy:
            cols.append("Energy (mJ/fr)")

        _right_align = {"mIoU", "FPS", "Mem (MB)", "Frames", "Success AUC", "Energy (mJ/fr)"}
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(
            "---:" if c in _right_align else "---" for c in cols
        ) + " |"

        rows = [header, sep]
        for rank, e in enumerate(entries, 1):
            cells = [
                str(rank),
                e.tracker_name,
                " + ".join(e.dataset_names),
                f"{e.mean_iou:.4f}",
                f"{e.mean_fps:.1f}",
                f"{e.peak_memory_mb:.1f}",
                str(e.total_frames),
            ]
            if has_sauc:
                cells.append(f"{e.mean_success_auc:.4f}" if e.mean_success_auc is not None else "—")
            if has_energy:
                cells.append(
                    f"{e.mean_energy_per_frame_mj:.3f}"
                    if e.mean_energy_per_frame_mj is not None else "—"
                )
            rows.append("| " + " | ".join(cells) + " |")

        return "\n".join(rows)

    def save(self, path: Union[str, Path]) -> Path:
        """Serialise all results and the cross-dataset aggregate to JSON.

        Args:
            path: Destination path.  A ``.json`` suffix is appended when
                absent.  Parent directories are created automatically.

        Returns:
            The resolved :class:`~pathlib.Path` that was written.
        """
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".json")
        p.parent.mkdir(parents=True, exist_ok=True)

        def _fmt_entry(e: CrossDatasetEntry) -> Dict:
            d: Dict = {
                "tracker_name": e.tracker_name,
                "dataset_names": e.dataset_names,
                "total_frames": e.total_frames,
                "mean_iou": round(e.mean_iou, 4),
                "mean_fps": round(e.mean_fps, 2),
                "peak_memory_mb": round(e.peak_memory_mb, 2),
                "per_dataset": e.per_dataset,
            }
            if e.mean_success_auc is not None:
                d["mean_success_auc"] = round(e.mean_success_auc, 4)
            if e.mean_precision_auc is not None:
                d["mean_precision_auc"] = round(e.mean_precision_auc, 4)
            if e.mean_energy_per_frame_mj is not None:
                d["mean_energy_per_frame_mj"] = round(e.mean_energy_per_frame_mj, 4)
            return d

        data = {
            "aggregate": [_fmt_entry(e) for e in self.aggregate()],
            "results": [r.to_dict() for r in self.benchmark_results],
        }
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        return p


class MultiDatasetEvaluator:
    """Evaluate multiple trackers across multiple datasets in one call.

    Runs a grid of (tracker × dataset) benchmark evaluations, aggregates
    the results with frame-count weighting, and exposes a unified cross-
    dataset leaderboard.

    Args:
        verbose:    Print per-sequence progress to stdout.  Default ``True``.
        tdp_watts:  CPU TDP for energy profiling (Watts).  ``None`` disables
            energy profiling.
    """

    def __init__(
        self,
        verbose: bool = True,
        tdp_watts: Optional[float] = None,
    ) -> None:
        self.verbose = verbose
        self.tdp_watts = tdp_watts

    def run(
        self,
        trackers: List[BaseTracker],
        datasets: List[Tuple[BaseDataset, str]],
        max_sequences: Optional[int] = None,
    ) -> MultiDatasetResult:
        """Evaluate every tracker on every dataset.

        Args:
            trackers:       List of tracker instances to evaluate.
            datasets:       List of ``(dataset, dataset_name)`` pairs.
            max_sequences:  Cap on sequences evaluated per dataset; useful for
                quick smoke tests.

        Returns:
            :class:`MultiDatasetResult` containing one
            :class:`~eovot.benchmark.engine.BenchmarkResult` per
            (tracker, dataset) pair.
        """
        engine = BenchmarkEngine(verbose=self.verbose, tdp_watts=self.tdp_watts)
        multi_result = MultiDatasetResult()
        total_runs = len(trackers) * len(datasets)

        for run_idx, tracker in enumerate(trackers):
            for ds_idx, (dataset, dataset_name) in enumerate(datasets):
                run_no = run_idx * len(datasets) + ds_idx + 1
                if self.verbose:
                    print(f"\n{'='*60}")
                    print(
                        f"[{run_no}/{total_runs}] "
                        f"Tracker: {tracker.name}  |  Dataset: {dataset_name}"
                    )
                    print(f"{'='*60}")
                result = engine.run(
                    tracker=tracker,
                    dataset=dataset,
                    dataset_name=dataset_name,
                    max_sequences=max_sequences,
                )
                multi_result.benchmark_results.append(result)

        if self.verbose:
            print("\n" + "=" * 60)
            print("CROSS-DATASET LEADERBOARD")
            print("=" * 60)
            print(multi_result.cross_dataset_table())

        return multi_result

    @classmethod
    def from_config(
        cls,
        config: Dict,
        verbose: bool = True,
    ) -> MultiDatasetResult:
        """Run a multi-dataset experiment defined by a config dict (from YAML).

        Config schema::

            experiment:
              name: multi-dataset-run
              tdp_watts: null       # float or null
              max_sequences: null   # int or null (cap per dataset)

            datasets:
              - loader: OTBDataset
                root: /data/OTB100
                name: OTB100
              - loader: GOT10kDataset
                root: /data/GOT-10k
                split: val
                name: GOT-10k-val
              - loader: synthetic
                name: Synthetic
                n_sequences: 5
                n_frames: 100
                motion: linear

            trackers:
              - name: MOSSE
                params: {}
              - name: KCF
                params:
                  learning_rate: 0.125

        Args:
            config:  Nested dict typically loaded from YAML.
            verbose: Forwarded to the evaluator.

        Returns:
            :class:`MultiDatasetResult` with all results populated.

        Raises:
            ValueError: For unknown ``loader`` values.
        """
        from ..datasets.base import OTBDataset
        from ..datasets.got10k import GOT10kDataset
        from ..datasets.lasot import LaSOTDataset
        from ..datasets.synthetic import SyntheticDataset
        from ..trackers.registry import build_tracker

        exp_cfg = config.get("experiment", {})
        tdp_watts = exp_cfg.get("tdp_watts", None)
        max_sequences = exp_cfg.get("max_sequences", None)

        trackers = [
            build_tracker(t["name"], **t.get("params", {}))
            for t in config.get("trackers", [])
        ]

        datasets: List[Tuple[BaseDataset, str]] = []
        for ds_cfg in config.get("datasets", []):
            loader = ds_cfg.get("loader", "OTBDataset")
            name = ds_cfg.get("name", loader)
            if loader == "OTBDataset":
                ds: BaseDataset = OTBDataset(ds_cfg["root"])
            elif loader == "GOT10kDataset":
                ds = GOT10kDataset(ds_cfg["root"], split=ds_cfg.get("split", "val"))
            elif loader == "LaSOTDataset":
                ds = LaSOTDataset(ds_cfg["root"], split=ds_cfg.get("split", "test"))
            elif loader == "synthetic":
                ds = SyntheticDataset(
                    num_sequences=ds_cfg.get("n_sequences", 5),
                    num_frames=ds_cfg.get("n_frames", 100),
                    motion=ds_cfg.get("motion", "linear"),
                )
            else:
                raise ValueError(
                    f"Unknown dataset loader '{loader}'.  "
                    "Expected one of: OTBDataset, GOT10kDataset, LaSOTDataset, synthetic."
                )
            datasets.append((ds, name))

        evaluator = cls(verbose=verbose, tdp_watts=tdp_watts)
        return evaluator.run(trackers, datasets, max_sequences=max_sequences)

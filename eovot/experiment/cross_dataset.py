"""Cross-dataset evaluation engine for EOVOT.

Runs a single tracker across multiple datasets in one call, collecting
per-dataset :class:`~eovot.benchmark.engine.BenchmarkResult` objects and
computing aggregate statistics (mean ± std) across datasets.  This is the
standard evaluation protocol used in papers that report jointly on OTB,
LaSOT, and GOT-10k.

Example::

    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.mosse import MOSSETracker
    from eovot.experiment.cross_dataset import CrossDatasetEvaluator

    datasets = {
        "Synthetic-Linear": SyntheticDataset(motion="linear"),
        "Synthetic-Random": SyntheticDataset(motion="random"),
    }
    evaluator = CrossDatasetEvaluator(datasets=datasets)
    report = evaluator.run(MOSSETracker())
    print(report.to_markdown())
    print(report.aggregate)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..benchmark.engine import BenchmarkEngine, BenchmarkResult
from ..datasets.base import BaseDataset
from ..trackers.base import BaseTracker


@dataclass
class CrossDatasetReport:
    """Aggregated result of evaluating one tracker across N datasets.

    Attributes:
        tracker_name: Name of the evaluated tracker.
        per_dataset:  Mapping from dataset name to the summary dict
                      returned by :meth:`~eovot.benchmark.engine.BenchmarkResult.summary`.
        raw_results:  Full :class:`~eovot.benchmark.engine.BenchmarkResult` objects,
                      keyed by dataset name.  Not serialised by :meth:`to_dict`.
    """

    tracker_name: str
    per_dataset: Dict[str, Dict] = field(default_factory=dict)
    raw_results: Dict[str, BenchmarkResult] = field(default_factory=dict, repr=False)

    @property
    def aggregate(self) -> Dict:
        """Mean ± std across all datasets for the primary metrics.

        Keys follow the pattern ``<metric>_mean`` and ``<metric>_std``.
        Only metrics present in at least one per-dataset summary are included.
        """
        if not self.per_dataset:
            return {}

        core_metrics = ["mean_iou", "mean_fps", "peak_memory_mb"]
        optional_metrics = [
            "success_auc",
            "precision_auc",
            "mean_energy_per_frame_mj",
            "mean_center_distance_px",
        ]

        agg: Dict = {
            "tracker": self.tracker_name,
            "num_datasets": len(self.per_dataset),
        }

        for key in core_metrics + optional_metrics:
            vals = [
                float(s[key])
                for s in self.per_dataset.values()
                if key in s
            ]
            if vals:
                agg[f"{key}_mean"] = round(float(np.mean(vals)), 4)
                agg[f"{key}_std"] = round(float(np.std(vals)), 4)

        return agg

    def to_dict(self) -> Dict:
        """Serialise to a JSON-compatible dict."""
        return {
            "tracker": self.tracker_name,
            "per_dataset": self.per_dataset,
            "aggregate": self.aggregate,
        }

    def to_markdown(self) -> str:
        """Render the per-dataset breakdown as a Markdown table.

        The final row shows the aggregate mean ± std across all datasets,
        suitable for inclusion in a paper's results section.
        """
        if not self.per_dataset:
            return f"No results for {self.tracker_name}.\n"

        # Collect metric keys in stable insertion order (skip label fields).
        label_keys = {"tracker", "dataset", "num_sequences"}
        all_keys: List[str] = []
        seen: set = set()
        for s in self.per_dataset.values():
            for k in s:
                if k not in seen and k not in label_keys:
                    seen.add(k)
                    all_keys.append(k)

        col_names = ["Dataset"] + all_keys
        header = "| " + " | ".join(col_names) + " |"
        sep = "| " + " | ".join(["---"] * len(col_names)) + " |"
        rows = [header, sep]

        for ds_name, s in self.per_dataset.items():
            cells = [ds_name]
            for k in all_keys:
                v = s.get(k, "-")
                cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
            rows.append("| " + " | ".join(cells) + " |")

        # Aggregate row
        agg = self.aggregate
        cells = ["**mean ± std**"]
        for k in all_keys:
            mean_key = f"{k}_mean"
            std_key = f"{k}_std"
            if mean_key in agg:
                cells.append(f"{agg[mean_key]:.4f} ± {agg[std_key]:.4f}")
            else:
                cells.append("-")
        rows.append("| " + " | ".join(cells) + " |")

        title = f"## {self.tracker_name} — Cross-Dataset Results"
        return title + "\n\n" + "\n".join(rows) + "\n"


class CrossDatasetEvaluator:
    """Evaluate a tracker (or a set of trackers) across multiple datasets.

    Args:
        datasets:      Mapping from a human-readable dataset name to a
                       :class:`~eovot.datasets.base.BaseDataset` instance.
                       Must contain at least one entry.
        verbose:       Print per-sequence progress during evaluation.
                       Default ``True``.
        tdp_watts:     CPU TDP for energy estimation (see
                       :class:`~eovot.benchmark.engine.BenchmarkEngine`).
                       ``None`` disables energy profiling.
        max_sequences: Cap the number of sequences per dataset.
                       ``None`` uses the full dataset.

    Raises:
        ValueError: If *datasets* is empty.

    Example::

        evaluator = CrossDatasetEvaluator(
            datasets={
                "Synthetic-Linear": SyntheticDataset(motion="linear"),
                "Synthetic-Random": SyntheticDataset(motion="random"),
            },
            verbose=False,
        )
        report = evaluator.run(MOSSETracker())
        print(report.to_markdown())
    """

    def __init__(
        self,
        datasets: Dict[str, BaseDataset],
        verbose: bool = True,
        tdp_watts: Optional[float] = None,
        max_sequences: Optional[int] = None,
    ) -> None:
        if not datasets:
            raise ValueError(
                "datasets must not be empty — pass at least one dataset."
            )
        self.datasets = datasets
        self.verbose = verbose
        self.tdp_watts = tdp_watts
        self.max_sequences = max_sequences

    def run(self, tracker: BaseTracker) -> CrossDatasetReport:
        """Evaluate *tracker* on every registered dataset.

        Args:
            tracker: Any :class:`~eovot.trackers.base.BaseTracker` instance.

        Returns:
            :class:`CrossDatasetReport` with per-dataset summaries and
            aggregate statistics.
        """
        report = CrossDatasetReport(tracker_name=tracker.name)
        engine = BenchmarkEngine(verbose=self.verbose, tdp_watts=self.tdp_watts)

        for ds_name, dataset in self.datasets.items():
            result = engine.run(
                tracker=tracker,
                dataset=dataset,
                dataset_name=ds_name,
                max_sequences=self.max_sequences,
            )
            report.per_dataset[ds_name] = result.summary()
            report.raw_results[ds_name] = result

        return report

    def compare(self, trackers: List[BaseTracker]) -> Dict[str, CrossDatasetReport]:
        """Evaluate multiple trackers and return one report per tracker.

        Args:
            trackers: List of tracker instances.  Each tracker is evaluated
                      against every dataset in :attr:`datasets`.

        Returns:
            Dict mapping ``tracker.name`` → :class:`CrossDatasetReport`.
        """
        return {t.name: self.run(t) for t in trackers}

    def to_comparison_markdown(
        self, reports: Dict[str, CrossDatasetReport]
    ) -> str:
        """Render a multi-tracker aggregate comparison table in Markdown.

        Trackers are ranked by mean success AUC (falls back to mean mIoU
        when success AUC is unavailable).

        Args:
            reports: Dict of ``tracker_name`` → :class:`CrossDatasetReport`,
                     as returned by :meth:`compare`.

        Returns:
            Markdown string ready for a ``results.md`` file.
        """
        if not reports:
            return "No reports to compare.\n"

        rows = []
        for t_name, report in reports.items():
            agg = report.aggregate
            miou = agg.get("mean_iou_mean", 0.0)
            miou_std = agg.get("mean_iou_std", 0.0)
            sauc = agg.get("success_auc_mean", miou)  # fall back to mIoU
            fps = agg.get("mean_fps_mean", 0.0)
            mem = agg.get("peak_memory_mb_mean", 0.0)
            n_datasets = agg.get("num_datasets", 0)
            rows.append(
                {
                    "tracker": t_name,
                    "n_datasets": n_datasets,
                    "mIoU": miou,
                    "mIoU_std": miou_std,
                    "sauc": sauc,
                    "fps": fps,
                    "mem": mem,
                }
            )

        rows.sort(key=lambda r: r["sauc"], reverse=True)

        lines = [
            "## Multi-Tracker Cross-Dataset Comparison\n",
            "| Rank | Tracker | Datasets | mIoU (mean ± std) | Success AUC | FPS | Mem (MB) |",
            "|------|---------|----------|:-----------------:|------------:|----:|---------:|",
        ]
        for rank, row in enumerate(rows, 1):
            lines.append(
                f"| {rank} | {row['tracker']} | {row['n_datasets']} "
                f"| {row['mIoU']:.4f} ± {row['mIoU_std']:.4f} "
                f"| {row['sauc']:.4f} "
                f"| {row['fps']:.1f} "
                f"| {row['mem']:.1f} |"
            )
        lines.append("")
        return "\n".join(lines)

"""Cross-dataset evaluation pipeline for EOVOT.

Running a tracker on a single dataset gives a narrow view of its capabilities.
Research papers (GOT-10k, LaSOT, OTB100, TrackingNet) each use different
challenge conditions, and a tracker that excels on one may fail on another.

:class:`CrossDatasetEvaluator` solves this by automating the full
tracker × dataset matrix, computing per-combination results, and aggregating
them into a single **Aggregate Performance Score (APS)**:

    APS = Σ (weight_i × success_auc_i) / Σ weight_i

where the sum is over datasets.  The APS gives a single ranked ordering
suitable for a research paper's main table without cherry-picking datasets.

Trackers that have pre-computed JSON results (from a prior
:class:`~eovot.benchmark.engine.BenchmarkEngine` run) can be loaded from
disk and merged with newly-evaluated trackers, enabling incremental updates
without re-running everything.

Typical usage::

    from eovot.experiment.cross_eval import CrossDatasetEvaluator
    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kcf import KCFTracker
    from eovot.datasets.synthetic import SyntheticDataset

    ds1 = SyntheticDataset(num_sequences=5, motion="linear",   seed=1)
    ds2 = SyntheticDataset(num_sequences=5, motion="circular", seed=2)

    evaluator = CrossDatasetEvaluator(
        datasets={"Syn-Linear": ds1, "Syn-Circular": ds2},
        dataset_weights={"Syn-Linear": 1.0, "Syn-Circular": 1.5},
    )
    evaluator.add_tracker(MOSSETracker())
    evaluator.add_tracker(KCFTracker())

    table = evaluator.run()
    print(table.to_markdown())
    table.save("results/cross_eval.json")
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..benchmark.engine import BenchmarkEngine, BenchmarkResult
from ..datasets.base import BaseDataset
from ..trackers.base import BaseTracker


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrackerCrossResult:
    """Per-tracker cross-dataset summary produced by :class:`CrossDatasetEvaluator`.

    Attributes:
        tracker_name:       Human-readable tracker identifier.
        per_dataset:        Mapping from dataset name to its
                            :class:`~eovot.benchmark.engine.BenchmarkResult`.
        dataset_weights:    Per-dataset weights used to compute APS.
        aps:                Aggregate Performance Score (weighted mean
                            success AUC across datasets).
        mean_fps:           Weighted mean FPS across datasets.
        mean_memory_mb:     Weighted mean peak memory across datasets.
    """

    tracker_name: str
    per_dataset: Dict[str, BenchmarkResult] = field(default_factory=dict)
    dataset_weights: Dict[str, float] = field(default_factory=dict)
    aps: float = 0.0
    mean_fps: float = 0.0
    mean_memory_mb: float = 0.0

    def __str__(self) -> str:
        ds_list = ", ".join(self.per_dataset)
        return (
            f"TrackerCrossResult({self.tracker_name}  "
            f"APS={self.aps:.4f}  FPS={self.mean_fps:.1f}  "
            f"mem={self.mean_memory_mb:.1f}MB  datasets=[{ds_list}])"
        )


@dataclass
class CrossEvalTable:
    """Multi-tracker cross-dataset leaderboard produced by :class:`CrossDatasetEvaluator`.

    Attributes:
        results:      List of :class:`TrackerCrossResult`, sorted by APS
                      (highest first).
        dataset_names: Ordered list of evaluated dataset names.
    """

    results: List[TrackerCrossResult] = field(default_factory=list)
    dataset_names: List[str] = field(default_factory=list)

    def to_markdown(self, show_per_dataset: bool = True) -> str:
        """Format the leaderboard as a Markdown table.

        Args:
            show_per_dataset: When True, include a column for each dataset's
                success AUC in addition to the APS.  Default: ``True``.

        Returns:
            Multi-line Markdown string ready to paste into a README or paper.
        """
        if not self.results:
            return "_No results available._\n"

        ds_names = self.dataset_names

        # Header
        cols = ["Rank", "Tracker", "APS", "FPS", "Mem (MB)"]
        if show_per_dataset:
            cols += [f"AUC ({d})" for d in ds_names]

        header = "| " + " | ".join(cols) + " |"
        sep = "|" + "|".join(["---"] * len(cols)) + "|"
        lines = [header, sep]

        for rank, r in enumerate(self.results, start=1):
            row = [
                str(rank),
                r.tracker_name,
                f"{r.aps:.4f}",
                f"{r.mean_fps:.1f}",
                f"{r.mean_memory_mb:.1f}",
            ]
            if show_per_dataset:
                for ds in ds_names:
                    br = r.per_dataset.get(ds)
                    if br is not None:
                        sauc = br.mean_success_auc
                        row.append(f"{sauc:.4f}" if sauc is not None else "—")
                    else:
                        row.append("—")
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def to_csv(self, path: Union[str, Path]) -> Path:
        """Write the leaderboard to a CSV file.

        Columns: ``rank``, ``tracker``, ``aps``, ``fps``, ``peak_memory_mb``,
        then one column per dataset with its success AUC.

        Args:
            path: Output file path.

        Returns:
            Resolved :class:`pathlib.Path` that was written.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["rank", "tracker", "aps", "fps", "peak_memory_mb"]
        for ds in self.dataset_names:
            fieldnames.append(f"auc_{ds.lower().replace(' ', '_')}")

        with open(p, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for rank, r in enumerate(self.results, start=1):
                row: Dict[str, Any] = {
                    "rank": rank,
                    "tracker": r.tracker_name,
                    "aps": round(r.aps, 4),
                    "fps": round(r.mean_fps, 2),
                    "peak_memory_mb": round(r.mean_memory_mb, 2),
                }
                for ds in self.dataset_names:
                    key = f"auc_{ds.lower().replace(' ', '_')}"
                    br = r.per_dataset.get(ds)
                    if br is not None:
                        sauc = br.mean_success_auc
                        row[key] = round(sauc, 4) if sauc is not None else ""
                    else:
                        row[key] = ""
                writer.writerow(row)
        return p

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the leaderboard."""
        entries = []
        for rank, r in enumerate(self.results, start=1):
            entry: Dict[str, Any] = {
                "rank": rank,
                "tracker": r.tracker_name,
                "aps": round(r.aps, 4),
                "fps": round(r.mean_fps, 2),
                "peak_memory_mb": round(r.mean_memory_mb, 2),
                "datasets": {},
            }
            for ds, br in r.per_dataset.items():
                summary = br.summary()
                entry["datasets"][ds] = {
                    "mean_iou": summary.get("mean_iou"),
                    "success_auc": summary.get("success_auc"),
                    "mean_fps": summary.get("mean_fps"),
                    "peak_memory_mb": summary.get("peak_memory_mb"),
                }
            entries.append(entry)
        return {
            "dataset_names": self.dataset_names,
            "leaderboard": entries,
        }

    def save(self, path: Union[str, Path]) -> Path:
        """Serialise the leaderboard to a JSON file.

        Args:
            path: Destination file path.  A ``.json`` extension is appended
                  when the path has none.

        Returns:
            Resolved :class:`pathlib.Path` that was written.
        """
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".json")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return p


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class CrossDatasetEvaluator:
    """Evaluate a set of trackers on multiple datasets and rank by APS.

    The evaluator manages the full (tracker × dataset) evaluation matrix,
    handles skipping already-evaluated combinations when ``resume=True``,
    and returns a ranked :class:`CrossEvalTable`.

    Args:
        datasets: Mapping from dataset name (string label used in reports)
            to a :class:`~eovot.datasets.base.BaseDataset` instance.
        dataset_weights: Optional per-dataset weight for APS computation.
            Dataset names absent from this dict receive weight ``1.0``.
        verbose: Print per-sequence benchmark progress.  Default: ``True``.
        max_sequences: Limit each dataset to this many sequences.  Useful
            for quick validation runs.  ``None`` = use all sequences.
        tdp_watts: Enable CPU energy profiling with this TDP value (Watts).
            ``None`` disables energy profiling.

    Example::

        from eovot.experiment.cross_eval import CrossDatasetEvaluator
        from eovot.trackers.mosse import MOSSETracker
        from eovot.datasets.synthetic import SyntheticDataset

        evaluator = CrossDatasetEvaluator(
            datasets={"Linear": SyntheticDataset(motion="linear"),
                      "Random": SyntheticDataset(motion="random")},
            verbose=False,
        )
        evaluator.add_tracker(MOSSETracker())
        table = evaluator.run()
        print(table.to_markdown())
    """

    def __init__(
        self,
        datasets: Dict[str, BaseDataset],
        dataset_weights: Optional[Dict[str, float]] = None,
        verbose: bool = True,
        max_sequences: Optional[int] = None,
        tdp_watts: Optional[float] = None,
    ) -> None:
        if not datasets:
            raise ValueError("At least one dataset must be provided.")
        self._datasets: Dict[str, BaseDataset] = dict(datasets)
        self._weights: Dict[str, float] = dict(dataset_weights or {})
        self.verbose = verbose
        self.max_sequences = max_sequences
        self._trackers: List[BaseTracker] = []
        self._engine = BenchmarkEngine(verbose=verbose, tdp_watts=tdp_watts)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def add_tracker(self, tracker: BaseTracker) -> None:
        """Register a tracker for evaluation.

        Args:
            tracker: Any :class:`~eovot.trackers.base.BaseTracker` instance.

        Raises:
            ValueError: If a tracker with the same name is already registered.
        """
        existing = {t.name for t in self._trackers}
        if tracker.name in existing:
            raise ValueError(
                f"Tracker '{tracker.name}' is already registered. "
                "Use a different name or remove the existing entry."
            )
        self._trackers.append(tracker)

    def set_weight(self, dataset_name: str, weight: float) -> None:
        """Set the APS weight for one dataset.

        Args:
            dataset_name: Must match a key in the ``datasets`` dict.
            weight:       Non-negative scalar.  ``0.0`` excludes the dataset
                          from APS but still runs it.

        Raises:
            ValueError: If ``dataset_name`` is not registered.
        """
        if dataset_name not in self._datasets:
            raise ValueError(
                f"Dataset '{dataset_name}' not registered.  "
                f"Available: {list(self._datasets)}"
            )
        if weight < 0:
            raise ValueError(f"Weight must be ≥ 0, got {weight}.")
        self._weights[dataset_name] = weight

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        prior_results: Optional[Dict[str, Dict[str, BenchmarkResult]]] = None,
    ) -> CrossEvalTable:
        """Run all (tracker × dataset) combinations and return a ranked table.

        Args:
            prior_results: Optional pre-computed results to reuse.  Format:
                ``{tracker_name: {dataset_name: BenchmarkResult}}``.
                Combinations present here are **not re-evaluated**, enabling
                incremental updates.

        Returns:
            :class:`CrossEvalTable` with trackers ranked by APS.

        Raises:
            RuntimeError: If no trackers have been added.
        """
        if not self._trackers:
            raise RuntimeError("No trackers registered. Call add_tracker() first.")

        prior: Dict[str, Dict[str, BenchmarkResult]] = prior_results or {}
        dataset_names = list(self._datasets)

        tracker_cross_results: List[TrackerCrossResult] = []

        for tracker in self._trackers:
            per_dataset: Dict[str, BenchmarkResult] = {}

            # Reuse any prior results for this tracker
            if tracker.name in prior:
                per_dataset.update(prior[tracker.name])

            for ds_name, dataset in self._datasets.items():
                if ds_name in per_dataset:
                    if self.verbose:
                        print(f"  [skip] {tracker.name} on {ds_name} (cached)")
                    continue

                result = self._engine.run(
                    tracker,
                    dataset,
                    dataset_name=ds_name,
                    max_sequences=self.max_sequences,
                )
                per_dataset[ds_name] = result

            aps, mean_fps, mean_mem = self._aggregate(per_dataset)
            tracker_cross_results.append(
                TrackerCrossResult(
                    tracker_name=tracker.name,
                    per_dataset=per_dataset,
                    dataset_weights=dict(self._weights),
                    aps=aps,
                    mean_fps=mean_fps,
                    mean_memory_mb=mean_mem,
                )
            )

        # Sort by APS descending
        tracker_cross_results.sort(key=lambda r: r.aps, reverse=True)

        return CrossEvalTable(
            results=tracker_cross_results,
            dataset_names=dataset_names,
        )

    # ------------------------------------------------------------------
    # Class-method helpers for loading from disk
    # ------------------------------------------------------------------

    @classmethod
    def load_prior_results(
        cls,
        result_dir: Union[str, Path],
    ) -> Dict[str, Dict[str, BenchmarkResult]]:
        """Load all JSON result files from a directory into a prior-results dict.

        Expects files named ``<tracker_name>__<dataset_name>.json`` or
        ``<tracker_name>.json`` (single-dataset results).  Files produced by
        :meth:`~eovot.benchmark.engine.BenchmarkResult.save` are directly
        compatible.

        Args:
            result_dir: Directory containing JSON benchmark result files.

        Returns:
            Nested dict ``{tracker_name: {dataset_name: BenchmarkResult}}``.
        """
        result_dir = Path(result_dir)
        out: Dict[str, Dict[str, BenchmarkResult]] = {}

        for json_path in sorted(result_dir.glob("*.json")):
            try:
                br = BenchmarkResult.load(json_path)
            except Exception:
                continue
            tracker = br.tracker_name
            dataset = br.dataset_name
            out.setdefault(tracker, {})[dataset] = br

        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_weight(self, dataset_name: str) -> float:
        """Return the weight for a dataset, defaulting to 1.0."""
        return float(self._weights.get(dataset_name, 1.0))

    def _aggregate(
        self,
        per_dataset: Dict[str, BenchmarkResult],
    ) -> Tuple[float, float, float]:
        """Compute APS, weighted mean FPS, and weighted mean memory.

        APS = Σ (w_i × success_auc_i) / Σ w_i
        where datasets without a success_auc are excluded from the APS sum.

        Returns:
            (aps, mean_fps, mean_memory_mb) tuple.
        """
        aps_num = 0.0
        aps_den = 0.0
        fps_num = 0.0
        fps_den = 0.0
        mem_num = 0.0
        mem_den = 0.0

        for ds_name, br in per_dataset.items():
            w = self._get_weight(ds_name)
            sauc = br.mean_success_auc
            if sauc is not None:
                aps_num += w * sauc
                aps_den += w
            fps_num += w * br.mean_fps
            fps_den += w
            mem_num += w * br.peak_memory_mb
            mem_den += w

        aps = (aps_num / aps_den) if aps_den > 0 else 0.0
        mean_fps = (fps_num / fps_den) if fps_den > 0 else 0.0
        mean_mem = (mem_num / mem_den) if mem_den > 0 else 0.0
        return aps, mean_fps, mean_mem

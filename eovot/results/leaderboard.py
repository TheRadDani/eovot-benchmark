"""Leaderboard aggregator for EOVOT benchmark result collections.

Researchers accumulate many JSON result files from different experimental
runs, trackers, and datasets.  Without a unified aggregation layer, comparing
them requires ad-hoc scripts.  This module provides a single entry point that:

1. Loads any number of :class:`~eovot.benchmark.engine.BenchmarkResult` JSON
   files (or in-memory :class:`~eovot.benchmark.engine.BenchmarkResult` objects).
2. Groups results by dataset and ranks trackers on every configured metric.
3. Optionally compares each tracker to a named *baseline* tracker, surfacing
   delta values (Δ) so regressions and improvements are immediately visible.
4. Exports the leaderboard as a Markdown table, a CSV file, or a dict.

No database, no external dependencies beyond NumPy.

Typical usage::

    from eovot.results.leaderboard import LeaderboardAggregator

    agg = LeaderboardAggregator(metrics=["mean_iou", "mean_fps", "peak_memory_mb"])

    # Load from saved JSON files
    agg.load_dir("results/experiment-1/")
    agg.load_dir("results/experiment-2/")

    # Build and print the leaderboard
    board = agg.build(dataset="OTB100", sort_by="mean_iou")
    print(board.to_markdown())
    board.save_csv("leaderboard_otb100.csv")

    # Compare everything against MOSSE as baseline
    board_delta = agg.build(dataset="OTB100", sort_by="mean_iou", baseline="MOSSE")
    print(board_delta.to_markdown(show_delta=True))
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Row and board dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LeaderboardRow:
    """One tracker's aggregated metrics on a single dataset.

    Attributes:
        rank: Position in the leaderboard (1 = best on the sort metric).
        tracker_name: Identifier for the tracker.
        dataset_name: Dataset this result was evaluated on.
        metrics: ``{metric_name: value}`` for all configured metrics.
        deltas: ``{metric_name: delta_vs_baseline}`` when a baseline was
            specified; empty otherwise.
        num_sequences: Number of sequences in the evaluation.
    """

    rank: int
    tracker_name: str
    dataset_name: str
    metrics: Dict[str, float]
    deltas: Dict[str, float] = field(default_factory=dict)
    num_sequences: int = 0

    def metric(self, name: str) -> Optional[float]:
        """Return the value for *name*, or ``None`` if not present."""
        return self.metrics.get(name)

    def delta(self, name: str) -> Optional[float]:
        """Return the delta vs baseline for *name*, or ``None`` if absent."""
        return self.deltas.get(name)


@dataclass
class Leaderboard:
    """Ranked list of tracker results on a single dataset.

    Attributes:
        dataset_name: Dataset that all rows were evaluated on.
        sort_metric: Metric used to determine ranking order.
        higher_is_better: Direction of the sort metric.
        baseline_tracker: Name of the reference tracker for delta columns,
            or ``None`` when no baseline was requested.
        rows: Ordered list of :class:`LeaderboardRow`, rank 1 first.
        metrics: Ordered list of all metric names present in rows.
    """

    dataset_name: str
    sort_metric: str
    higher_is_better: bool
    baseline_tracker: Optional[str]
    rows: List[LeaderboardRow]
    metrics: List[str]

    def to_markdown(self, show_delta: bool = True) -> str:
        """Render the leaderboard as a Markdown table.

        Args:
            show_delta: When ``True`` (default) and a baseline was set,
                append a ``Δ`` column for each metric showing the difference
                from the baseline.  Positive deltas are improvement for
                higher-is-better metrics.

        Returns:
            Multi-line Markdown string ready to paste into a README or report.
        """
        include_delta = show_delta and bool(self.baseline_tracker)

        # Build column headers
        metric_cols = self.metrics
        headers = ["Rank", "Tracker"]
        for m in metric_cols:
            headers.append(m)
            if include_delta:
                headers.append(f"Δ {m}")
        headers.append("Sequences")

        sep = "|" + "|".join("-" * (len(h) + 2) for h in headers) + "|"
        header_row = "| " + " | ".join(headers) + " |"

        rows = []
        for row in self.rows:
            cols = [str(row.rank), row.tracker_name]
            for m in metric_cols:
                val = row.metrics.get(m)
                cols.append(f"{val:.4f}" if val is not None else "—")
                if include_delta:
                    d = row.deltas.get(m)
                    if d is None:
                        cols.append("—")
                    else:
                        sign = "+" if d >= 0 else ""
                        cols.append(f"{sign}{d:.4f}")
            cols.append(str(row.num_sequences))
            rows.append("| " + " | ".join(cols) + " |")

        title = f"### Leaderboard — {self.dataset_name} (sorted by `{self.sort_metric}`)"
        if self.baseline_tracker:
            title += f"  |  baseline: `{self.baseline_tracker}`"
        return "\n".join([title, "", header_row, sep] + rows)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a nested dict (JSON-safe)."""
        return {
            "dataset_name": self.dataset_name,
            "sort_metric": self.sort_metric,
            "higher_is_better": self.higher_is_better,
            "baseline_tracker": self.baseline_tracker,
            "metrics": self.metrics,
            "rows": [
                {
                    "rank": r.rank,
                    "tracker_name": r.tracker_name,
                    "dataset_name": r.dataset_name,
                    "num_sequences": r.num_sequences,
                    "metrics": r.metrics,
                    "deltas": r.deltas,
                }
                for r in self.rows
            ],
        }

    def save_csv(self, path: Union[str, Path]) -> Path:
        """Write the leaderboard to a CSV file.

        Args:
            path: Destination file path.  A ``.csv`` extension is appended
                when the path has none.

        Returns:
            Resolved :class:`pathlib.Path` that was written.
        """
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".csv")
        p.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["rank", "tracker_name", "dataset_name", "num_sequences"] + self.metrics
        if self.baseline_tracker:
            fieldnames += [f"delta_{m}" for m in self.metrics]

        with open(p, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in self.rows:
                entry: Dict[str, Any] = {
                    "rank": row.rank,
                    "tracker_name": row.tracker_name,
                    "dataset_name": row.dataset_name,
                    "num_sequences": row.num_sequences,
                    **row.metrics,
                }
                for m in self.metrics:
                    d = row.deltas.get(m)
                    entry[f"delta_{m}"] = "" if d is None else f"{d:.6f}"
                writer.writerow(entry)
        return p

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[LeaderboardRow]:
        return iter(self.rows)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class LeaderboardAggregator:
    """Collect BenchmarkResult objects and build ranked leaderboards.

    Args:
        metrics: Ordered list of metric keys to include in the leaderboard.
            Keys must appear in the dict returned by
            :meth:`~eovot.benchmark.engine.BenchmarkResult.summary`.
            Default: ``["mean_iou", "mean_fps", "peak_memory_mb"]``.

    Example::

        agg = LeaderboardAggregator(metrics=["mean_iou", "success_auc", "mean_fps"])
        agg.load_dir("results/")
        board = agg.build(dataset="SyntheticSet", sort_by="mean_iou")
        print(board.to_markdown())
    """

    _DEFAULT_METRICS = ["mean_iou", "mean_fps", "peak_memory_mb"]

    def __init__(self, metrics: Optional[List[str]] = None) -> None:
        self.metrics: List[str] = list(metrics or self._DEFAULT_METRICS)
        # {(tracker_name, dataset_name): summary_dict}
        self._store: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def add(self, result: "BenchmarkResult") -> None:  # type: ignore[name-defined]
        """Register an in-memory :class:`~eovot.benchmark.engine.BenchmarkResult`.

        Args:
            result: A fully-evaluated benchmark result.  Duplicate
                (tracker, dataset) pairs overwrite the previous entry.
        """
        summary = result.summary()
        tracker = summary.get("tracker") or summary.get("tracker_name", "unknown")
        dataset = summary.get("dataset") or summary.get("dataset_name", "unknown")
        self._store[(tracker, dataset)] = summary

    def load_json(self, path: Union[str, Path]) -> None:
        """Load a single JSON file produced by
        :meth:`~eovot.benchmark.engine.BenchmarkResult.save`.

        Args:
            path: Path to the JSON file.

        Raises:
            FileNotFoundError: If the file does not exist.
            KeyError: If the JSON lacks the required ``"summary"`` key.
        """
        p = Path(path)
        with open(p, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        summary = d["summary"]
        tracker = summary.get("tracker") or summary.get("tracker_name", p.stem)
        dataset = summary.get("dataset") or summary.get("dataset_name", "unknown")
        self._store[(tracker, dataset)] = summary

    def load_dir(
        self,
        directory: Union[str, Path],
        recursive: bool = False,
    ) -> int:
        """Load all ``.json`` result files from a directory.

        Args:
            directory: Path to scan for JSON files.
            recursive: When ``True``, scan subdirectories as well.
                Default: ``False``.

        Returns:
            Number of files successfully loaded.

        Raises:
            FileNotFoundError: If *directory* does not exist.
        """
        d = Path(directory)
        if not d.is_dir():
            raise FileNotFoundError(f"Directory not found: {d}")
        pattern = "**/*.json" if recursive else "*.json"
        loaded = 0
        for p in sorted(d.glob(pattern)):
            try:
                self.load_json(p)
                loaded += 1
            except (KeyError, json.JSONDecodeError):
                pass  # silently skip malformed files
        return loaded

    # ------------------------------------------------------------------
    # Building the leaderboard
    # ------------------------------------------------------------------

    def build(
        self,
        dataset: Optional[str] = None,
        sort_by: str = "mean_iou",
        higher_is_better: bool = True,
        baseline: Optional[str] = None,
    ) -> Leaderboard:
        """Build a ranked leaderboard for one dataset.

        Args:
            dataset: Dataset name to filter by.  When ``None``, all results
                are included (regardless of dataset).
            sort_by: Metric key to rank trackers by.  Must be one of
                ``self.metrics`` or any key in the stored summaries.
                Default: ``"mean_iou"``.
            higher_is_better: When ``True`` (default) higher values earn a
                lower rank number.  Set to ``False`` for latency or memory.
            baseline: Name of the tracker to use as reference for delta
                computation.  When ``None``, no delta columns are added.

        Returns:
            :class:`Leaderboard` with rows sorted by *sort_by*.

        Raises:
            ValueError: If no results are available for the requested dataset,
                or if *baseline* is specified but not found in the results.
        """
        rows_raw = self._filter(dataset)
        if not rows_raw:
            label = f"dataset='{dataset}'" if dataset else "any dataset"
            raise ValueError(
                f"No results found for {label}. "
                f"Available datasets: {self.available_datasets()}"
            )

        # Resolve baseline metrics
        baseline_metrics: Dict[str, float] = {}
        if baseline is not None:
            matched = [s for (t, _), s in rows_raw if t == baseline]
            if not matched:
                raise ValueError(
                    f"Baseline tracker '{baseline}' not found. "
                    f"Available: {[t for (t, _) in [k for k in rows_raw]]}",
                )
            baseline_metrics = {m: float(matched[0].get(m, 0.0)) for m in self.metrics}

        # Build rows
        leaderboard_rows: List[LeaderboardRow] = []
        for (tracker, ds), summary in rows_raw:
            metrics_dict: Dict[str, float] = {}
            for m in self.metrics:
                val = summary.get(m)
                if val is not None:
                    metrics_dict[m] = float(val)

            deltas: Dict[str, float] = {}
            if baseline is not None and tracker != baseline:
                for m in self.metrics:
                    if m in metrics_dict and m in baseline_metrics:
                        deltas[m] = metrics_dict[m] - baseline_metrics[m]

            leaderboard_rows.append(
                LeaderboardRow(
                    rank=0,  # assigned below
                    tracker_name=tracker,
                    dataset_name=ds,
                    metrics=metrics_dict,
                    deltas=deltas,
                    num_sequences=int(summary.get("num_sequences", 0)),
                )
            )

        # Sort and assign ranks
        leaderboard_rows.sort(
            key=lambda r: (
                r.metrics.get(sort_by, float("-inf") if higher_is_better else float("inf"))
                * (-1 if higher_is_better else 1)
            )
        )
        for i, row in enumerate(leaderboard_rows, start=1):
            row.rank = i

        # Determine final metric list (intersection of what's actually present)
        present = set()
        for r in leaderboard_rows:
            present.update(r.metrics.keys())
        final_metrics = [m for m in self.metrics if m in present]

        return Leaderboard(
            dataset_name=dataset or "all",
            sort_metric=sort_by,
            higher_is_better=higher_is_better,
            baseline_tracker=baseline,
            rows=leaderboard_rows,
            metrics=final_metrics,
        )

    def build_all(
        self,
        sort_by: str = "mean_iou",
        higher_is_better: bool = True,
        baseline: Optional[str] = None,
    ) -> Dict[str, Leaderboard]:
        """Build one leaderboard per dataset.

        Args:
            sort_by: Sort metric (see :meth:`build`).
            higher_is_better: Direction (see :meth:`build`).
            baseline: Baseline tracker name (see :meth:`build`).

        Returns:
            ``{dataset_name: Leaderboard}`` for every dataset that has results.
        """
        return {
            ds: self.build(
                dataset=ds,
                sort_by=sort_by,
                higher_is_better=higher_is_better,
                baseline=baseline,
            )
            for ds in self.available_datasets()
        }

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def available_datasets(self) -> List[str]:
        """Return sorted list of dataset names with results registered."""
        return sorted({ds for (_, ds) in self._store})

    def available_trackers(self) -> List[str]:
        """Return sorted list of tracker names with results registered."""
        return sorted({t for (t, _) in self._store})

    def __len__(self) -> int:
        """Total number of (tracker, dataset) results stored."""
        return len(self._store)

    def __repr__(self) -> str:
        return (
            f"LeaderboardAggregator("
            f"results={len(self)}, "
            f"datasets={self.available_datasets()}, "
            f"metrics={self.metrics})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter(
        self, dataset: Optional[str]
    ) -> List[Tuple[Tuple[str, str], Dict[str, Any]]]:
        """Return ``[((tracker, dataset), summary)]`` for the requested dataset."""
        if dataset is None:
            return list(self._store.items())
        return [
            (key, summary)
            for key, summary in self._store.items()
            if key[1] == dataset
        ]

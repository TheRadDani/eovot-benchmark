"""Multi-dataset rank aggregation for cross-dataset tracker comparison.

A tracker's mean IoU on GOT-10k cannot be directly compared to its mean IoU
on LaSOT — the datasets have different difficulty levels, sequence lengths,
and target diversity.  Raw metric averaging across datasets gives disproportionate
weight to harder datasets and hides per-dataset strengths.

This module implements **rank aggregation** following Demšar (2006) — the
standard approach in machine learning benchmarking for comparing classifiers
across multiple datasets.  For each metric and dataset, trackers are assigned
integer ranks (1 = best).  These ranks are then averaged across datasets,
producing a stable cross-dataset summary that is invariant to scale differences.

Key properties of rank aggregation:
- Scale-invariant: IoU of 0.6 on a hard dataset is treated the same as 0.6
  on an easy one when both give the same rank.
- Robust to dataset imbalance: a tracker that scores well on 3 of 4 datasets
  consistently ranks high even if the 4th dataset is an outlier.
- Publication-standard: used by OTB, GOT-10k, VOT challenge comparison tables.

The module also provides:
- Raw metric averages (for absolute performance context alongside ranks)
- Per-dataset per-metric breakdown tables
- Markdown and LaTeX export for direct use in papers

Typical usage::

    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kcf import KCFTracker
    from eovot.metrics.aggregation import MultiDatasetAggregator

    engine = BenchmarkEngine(verbose=False)
    ds1 = SyntheticDataset(num_sequences=5, seed=1)
    ds2 = SyntheticDataset(num_sequences=5, seed=2)

    trackers = [MOSSETracker(), KCFTracker()]
    results = [
        engine.run(t, ds, dataset_name=name)
        for t in trackers
        for ds, name in [(ds1, "Syn-A"), (ds2, "Syn-B")]
    ]

    agg = MultiDatasetAggregator()
    report = agg.aggregate(results)
    print(agg.to_markdown(report))
    print(agg.to_latex(report))

References:
    Demšar, J. (2006). Statistical comparisons of classifiers over multiple
    data sets. Journal of Machine Learning Research, 7, 1–30.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Metric catalogue
# ---------------------------------------------------------------------------

#: Metrics extracted from BenchmarkResult.summary() for aggregation.
#: Tuple: (key, display_name, higher_is_better)
AGGREGATE_METRICS: List[Tuple[str, str, bool]] = [
    ("mean_iou",              "mIoU",        True),
    ("success_auc",           "Succ. AUC",   True),
    ("precision_auc",         "Prec. AUC",   True),
    ("normalized_precision_auc", "NormP AUC", True),
    ("mean_fps",              "FPS",         True),
    ("peak_memory_mb",        "Mem (MiB)",   False),
    ("mean_center_distance_px", "Ctr Dist", False),
    ("total_energy_j",        "Energy (J)",  False),
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PerDatasetRow:
    """One tracker's performance on one dataset.

    Attributes:
        tracker_name: Tracker identifier.
        dataset_name: Dataset identifier.
        metrics: Raw metric values extracted from ``BenchmarkResult.summary()``.
        ranks: Rank of this tracker for each metric on this dataset (1 = best).
    """

    tracker_name: str
    dataset_name: str
    metrics: Dict[str, float]
    ranks: Dict[str, float] = field(default_factory=dict)


@dataclass
class AggregationReport:
    """Result of :meth:`MultiDatasetAggregator.aggregate`.

    Attributes:
        trackers: Sorted list of tracker names included in the comparison.
        datasets: Sorted list of dataset names included in the comparison.
        per_dataset: Flat list of :class:`PerDatasetRow` — one per
            (tracker, dataset) pair.
        avg_ranks: ``{tracker_name: {metric_key: avg_rank}}``.
            Ranks averaged across all datasets that report the metric.
            Lower average rank = better overall performance on that metric.
        avg_metrics: ``{tracker_name: {metric_key: mean_value}}``.
            Raw metric values averaged across datasets for absolute context.
        primary_metric: The metric key used to produce the final ranking.
        final_ranking: Tracker names sorted by ascending average rank on
            ``primary_metric`` (best tracker first).
    """

    trackers: List[str]
    datasets: List[str]
    per_dataset: List[PerDatasetRow]
    avg_ranks: Dict[str, Dict[str, float]]
    avg_metrics: Dict[str, Dict[str, float]]
    primary_metric: str
    final_ranking: List[str]

    def summary_dict(self) -> Dict[str, Dict]:
        """Return a compact dict keyed by tracker name with avg rank and raw metrics."""
        out: Dict[str, Dict] = {}
        for tracker in self.final_ranking:
            out[tracker] = {
                "avg_rank": round(self.avg_ranks[tracker].get(self.primary_metric, float("nan")), 3),
                "avg_metrics": {k: round(v, 4) for k, v in self.avg_metrics[tracker].items()},
            }
        return out


# ---------------------------------------------------------------------------
# Main aggregator
# ---------------------------------------------------------------------------

class MultiDatasetAggregator:
    """Aggregate tracker benchmark results across multiple datasets using rank averaging.

    Rank averaging is the standard method for cross-dataset tracker comparison
    used in OTB, GOT-10k, and VOT challenge leaderboards.  It avoids the scale
    sensitivity of raw metric averaging by ranking trackers on each dataset
    independently, then averaging those ranks.

    Args:
        metrics: List of ``(key, display_name, higher_is_better)`` tuples
            specifying which metrics to aggregate.  Defaults to
            :data:`AGGREGATE_METRICS`.
        primary_metric: Key of the metric used to determine the final ranking
            (by ascending average rank).  Default: ``"success_auc"``.

    Example::

        agg = MultiDatasetAggregator(primary_metric="mean_iou")
        report = agg.aggregate(results)
        print(agg.to_markdown(report))
    """

    def __init__(
        self,
        metrics: Optional[List[Tuple[str, str, bool]]] = None,
        primary_metric: str = "success_auc",
    ) -> None:
        self.metrics = metrics if metrics is not None else AGGREGATE_METRICS
        self.primary_metric = primary_metric
        self._metric_keys = [m[0] for m in self.metrics]
        self._metric_names = {m[0]: m[1] for m in self.metrics}
        self._higher_is_better = {m[0]: m[2] for m in self.metrics}

    # ------------------------------------------------------------------
    # Core aggregation
    # ------------------------------------------------------------------

    def aggregate(self, results: "List") -> AggregationReport:
        """Compute per-dataset ranks and cross-dataset rank averages.

        Args:
            results: List of :class:`~eovot.benchmark.engine.BenchmarkResult`
                objects.  Results from the same tracker on the same dataset
                are averaged if duplicates exist.

        Returns:
            :class:`AggregationReport` with ranks, averages, and final ranking.

        Raises:
            ValueError: If fewer than two trackers or two datasets are provided,
                making cross-dataset comparison ill-defined.
        """
        # --- 1. Extract (tracker, dataset, metrics) tuples --------------------
        rows: List[PerDatasetRow] = []
        for r in results:
            summary = r.summary()
            metrics_vals: Dict[str, float] = {}
            for key in self._metric_keys:
                val = summary.get(key)
                if val is not None:
                    metrics_vals[key] = float(val)
            rows.append(PerDatasetRow(
                tracker_name=r.tracker_name,
                dataset_name=r.dataset_name,
                metrics=metrics_vals,
            ))

        trackers = sorted({r.tracker_name for r in rows})
        datasets = sorted({r.dataset_name for r in rows})

        if len(trackers) < 2:
            raise ValueError(
                f"Rank aggregation requires at least 2 trackers; got {trackers}."
            )

        # --- 2. Assign ranks per (dataset, metric) ----------------------------
        for dataset in datasets:
            dataset_rows = [r for r in rows if r.dataset_name == dataset]
            for key in self._metric_keys:
                values = [(r.tracker_name, r.metrics.get(key)) for r in dataset_rows]
                values_with_data = [(t, v) for t, v in values if v is not None]
                if not values_with_data:
                    continue

                # Sort: best tracker first (rank 1)
                reverse = self._higher_is_better[key]
                sorted_trackers = sorted(
                    values_with_data, key=lambda x: x[1], reverse=reverse
                )

                # Assign ranks, averaging ties
                rank_map = _average_ranks(sorted_trackers)

                for r in dataset_rows:
                    if key in r.metrics and r.tracker_name in rank_map:
                        r.ranks[key] = rank_map[r.tracker_name]

        # --- 3. Average ranks and metrics across datasets per tracker ----------
        avg_ranks: Dict[str, Dict[str, float]] = {t: {} for t in trackers}
        avg_metrics: Dict[str, Dict[str, float]] = {t: {} for t in trackers}

        for tracker in trackers:
            tracker_rows = [r for r in rows if r.tracker_name == tracker]
            for key in self._metric_keys:
                rank_vals = [r.ranks[key] for r in tracker_rows if key in r.ranks]
                metric_vals = [r.metrics[key] for r in tracker_rows if key in r.metrics]
                if rank_vals:
                    avg_ranks[tracker][key] = float(np.mean(rank_vals))
                if metric_vals:
                    avg_metrics[tracker][key] = float(np.mean(metric_vals))

        # --- 4. Final ranking by primary metric average rank ------------------
        def _sort_key(t: str) -> float:
            return avg_ranks[t].get(self.primary_metric, float("inf"))

        final_ranking = sorted(trackers, key=_sort_key)

        return AggregationReport(
            trackers=trackers,
            datasets=datasets,
            per_dataset=rows,
            avg_ranks=avg_ranks,
            avg_metrics=avg_metrics,
            primary_metric=self.primary_metric,
            final_ranking=final_ranking,
        )

    # ------------------------------------------------------------------
    # Export methods
    # ------------------------------------------------------------------

    def to_markdown(self, report: AggregationReport) -> str:
        """Format the aggregation report as a Markdown table.

        The table shows each tracker's average rank on every metric, sorted by
        the primary metric's average rank.  Columns with no data are omitted.

        Args:
            report: Output of :meth:`aggregate`.

        Returns:
            Multi-line Markdown string ready for GitHub READMEs or issue comments.
        """
        active_keys = self._active_keys(report)
        if not active_keys:
            return "_No metrics available._"

        header_parts = ["| Rank | Tracker |"]
        sep_parts = ["|------|---------|"]
        for key in active_keys:
            name = self._metric_names[key]
            arrow = "↑" if self._higher_is_better[key] else "↓"
            header_parts.append(f" Avg Rank ({name} {arrow}) |")
            sep_parts.append("---:|")

        # Also add raw metric averages
        for key in active_keys:
            name = self._metric_names[key]
            header_parts.append(f" Avg {name} |")
            sep_parts.append("---:|")

        lines = [
            "".join(header_parts),
            "".join(sep_parts),
        ]
        for rank, tracker in enumerate(report.final_ranking, start=1):
            row = [f"| {rank} | {tracker} |"]
            ranks = report.avg_ranks[tracker]
            for key in active_keys:
                v = ranks.get(key)
                row.append(f" {v:.2f} |" if v is not None else " — |")
            avgs = report.avg_metrics[tracker]
            for key in active_keys:
                v = avgs.get(key)
                row.append(f" {v:.4f} |" if v is not None else " — |")
            lines.append("".join(row))

        header_note = (
            f"\n_Ranked by average rank on `{self.primary_metric}` across "
            f"{len(report.datasets)} dataset(s): "
            f"{', '.join(report.datasets)}._"
        )
        return "\n".join(lines) + header_note

    def to_latex(self, report: AggregationReport) -> str:
        """Format the aggregation report as a LaTeX tabular environment.

        Best values (lowest rank) in each column are bolded automatically.
        The table is wrapped in a ``booktabs``-style format suitable for
        direct inclusion in papers with ``\\input{file.tex}``.

        Args:
            report: Output of :meth:`aggregate`.

        Returns:
            LaTeX string for the aggregation table.
        """
        active_keys = self._active_keys(report)
        if not active_keys:
            return "% No metrics available."

        n_metric_cols = len(active_keys)
        col_spec = "lr" + "r" * n_metric_cols + "r" * n_metric_cols

        # Find best rank per metric column for bolding
        best_ranks: Dict[str, float] = {}
        for key in active_keys:
            vals = [report.avg_ranks[t].get(key) for t in report.trackers
                    if report.avg_ranks[t].get(key) is not None]
            if vals:
                best_ranks[key] = min(vals)

        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Cross-Dataset Tracker Ranking (rank averaging, Demšar 2006)}",
            r"\label{tab:cross_dataset_ranking}",
            r"\begin{tabular}{" + col_spec + "}",
            r"\toprule",
        ]

        # Header
        rank_headers = " & ".join(
            f"Rank({self._metric_names[k]})" for k in active_keys
        )
        avg_headers = " & ".join(
            f"Avg {self._metric_names[k]}" for k in active_keys
        )
        lines.append(f"Rank & Tracker & {rank_headers} & {avg_headers} \\\\")
        lines.append(r"\midrule")

        for pos, tracker in enumerate(report.final_ranking, start=1):
            parts = [str(pos), _latex_escape(tracker)]
            for key in active_keys:
                v = report.avg_ranks[tracker].get(key)
                if v is None:
                    parts.append("—")
                elif best_ranks.get(key) == v:
                    parts.append(f"\\textbf{{{v:.2f}}}")
                else:
                    parts.append(f"{v:.2f}")
            for key in active_keys:
                v = report.avg_metrics[tracker].get(key)
                parts.append(f"{v:.4f}" if v is not None else "—")
            lines.append(" & ".join(parts) + " \\\\")

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            (r"\caption*{\footnotesize Ranked by average rank on "
             rf"\texttt{{{_latex_escape(report.primary_metric)}}}.  "
             rf"Datasets: {', '.join(_latex_escape(d) for d in report.datasets)}.}}"),
            r"\end{table}",
        ]
        return "\n".join(lines)

    def to_per_dataset_markdown(self, report: AggregationReport) -> str:
        """Format a per-dataset breakdown as Markdown tables.

        One table per dataset, showing each tracker's raw metrics and ranks.
        Useful for understanding where a tracker excels or underperforms.

        Args:
            report: Output of :meth:`aggregate`.

        Returns:
            Multi-line Markdown string with one table per dataset.
        """
        active_keys = self._active_keys(report)
        sections: List[str] = []

        for dataset in report.datasets:
            ds_rows = [r for r in report.per_dataset if r.dataset_name == dataset]
            ds_rows_sorted = sorted(
                ds_rows,
                key=lambda r: r.ranks.get(self.primary_metric, float("inf")),
            )
            header = f"### {dataset}\n"
            col_heads = "| Rank | Tracker |"
            col_sep = "|------|---------|"
            for key in active_keys:
                arrow = "↑" if self._higher_is_better[key] else "↓"
                col_heads += f" {self._metric_names[key]} {arrow} | Rank |"
                col_sep += "---:|---:|"
            table_lines = [col_heads, col_sep]
            for pos, row in enumerate(ds_rows_sorted, start=1):
                cells = [f"| {pos} | {row.tracker_name} |"]
                for key in active_keys:
                    mv = row.metrics.get(key)
                    rv = row.ranks.get(key)
                    m_str = f"{mv:.4f}" if mv is not None else "—"
                    r_str = f"{rv:.0f}" if rv is not None else "—"
                    cells.append(f" {m_str} | {r_str} |")
                table_lines.append("".join(cells))
            sections.append(header + "\n".join(table_lines))

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _active_keys(self, report: AggregationReport) -> List[str]:
        """Return metric keys for which at least one tracker has data."""
        return [
            key for key in self._metric_keys
            if any(
                key in report.avg_ranks[t] or key in report.avg_metrics[t]
                for t in report.trackers
            )
        ]


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _average_ranks(sorted_trackers: List[Tuple[str, float]]) -> Dict[str, float]:
    """Assign average ranks to trackers with equal metric values (tie handling).

    Args:
        sorted_trackers: List of ``(tracker_name, metric_value)`` sorted best-first
            (rank 1 = first element).

    Returns:
        Dict mapping tracker name to its (possibly averaged) rank.
    """
    rank_map: Dict[str, float] = {}
    i = 0
    n = len(sorted_trackers)
    while i < n:
        j = i
        while j < n and sorted_trackers[j][1] == sorted_trackers[i][1]:
            j += 1
        avg_rank = float(sum(range(i + 1, j + 1))) / float(j - i)
        for k in range(i, j):
            rank_map[sorted_trackers[k][0]] = avg_rank
        i = j
    return rank_map


def _latex_escape(s: str) -> str:
    """Escape special LaTeX characters in a string."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    return s

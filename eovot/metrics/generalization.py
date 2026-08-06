"""Cross-dataset generalization analysis for tracker comparison.

A tracker that ranks first on OTB-100 may rank fifth on GOT-10k — not
because the datasets are equivalent, but because sequences differ in target
size, motion regime, and scene complexity.  Publication-quality benchmarks
must report how *consistent* tracker rankings are across evaluation sets.

This module provides:

``GeneralizationProfile``
    Per-tracker summary: metric values on each dataset, numerical ranks,
    mean metric, rank standard deviation (rank stability), and the best/worst
    dataset for that tracker.

``RankCorrelationTable``
    Pairwise Spearman ρ matrix between dataset tracker rankings.  An entry
    near 1.0 means two datasets agree on ranking; near 0 means rankings are
    uncorrelated, indicating dataset-specific performance factors.

``GeneralizationReport``
    Full report bundling profiles, the correlation matrix, and the mean
    inter-dataset rank correlation — a single scalar summarising how
    consistently trackers rank across the evaluation suite.

The Spearman ρ is computed from scratch using the closed-form rank-difference
formula (ρ = 1 − 6 Σd²/n(n²−1)), so there is no dependency on scipy.

Typical usage::

    from eovot.metrics.generalization import GeneralizationAnalyzer
    from eovot.benchmark.engine import BenchmarkResult

    # One list of BenchmarkResult objects per dataset
    results = {
        "OTB100": [mosse_otb, kcf_otb, csrt_otb],
        "GOT10k": [mosse_got, kcf_got, csrt_got],
    }

    analyzer = GeneralizationAnalyzer(metric="mean_iou")
    report = analyzer.analyze(results)
    print(report.to_markdown())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GeneralizationProfile:
    """Per-tracker generalization statistics across datasets.

    Attributes:
        tracker_name: Identifier for the tracker.
        datasets: Ordered list of dataset names.
        metric_values: Observed metric on each dataset, keyed by dataset name.
        ranks: 1-indexed rank of this tracker on each dataset (1 = best).
        mean_metric: Mean of ``metric_values`` across all datasets.
        rank_std: Standard deviation of ranks — lower means more consistent.
        best_dataset: Dataset where this tracker achieved its best rank.
        worst_dataset: Dataset where this tracker achieved its worst rank.
    """

    tracker_name: str
    datasets: List[str]
    metric_values: Dict[str, float]
    ranks: Dict[str, int]
    mean_metric: float
    rank_std: float
    best_dataset: str
    worst_dataset: str

    def __str__(self) -> str:
        rank_parts = "  ".join(
            f"{d}=#{self.ranks[d]}" for d in sorted(self.datasets)
        )
        return (
            f"GeneralizationProfile[{self.tracker_name}] "
            f"mean={self.mean_metric:.4f}  rank_std={self.rank_std:.2f}  "
            f"ranks: {rank_parts}"
        )


@dataclass
class RankCorrelationTable:
    """Pairwise Spearman rank-correlation matrix between dataset rankings.

    Attributes:
        datasets: Ordered list of dataset names.
        correlation_matrix: ``(D, D)`` float64 array; entry ``[i, j]`` is
            Spearman ρ between the tracker ranking on dataset ``i`` and ``j``.
            The diagonal is always 1.0.
    """

    datasets: List[str]
    correlation_matrix: np.ndarray

    def __str__(self) -> str:
        col_w = 10
        header = " " * col_w + "  ".join(f"{d[:col_w]:>{col_w}s}" for d in self.datasets)
        rows = [header]
        for i, d in enumerate(self.datasets):
            vals = "  ".join(
                f"{self.correlation_matrix[i, j]:{col_w}.3f}"
                for j in range(len(self.datasets))
            )
            rows.append(f"{d[:col_w]:>{col_w}s}  {vals}")
        return "RankCorrelationTable:\n" + "\n".join(rows)

    def to_markdown(self) -> str:
        """Format as a Markdown table."""
        header = "| Dataset | " + " | ".join(self.datasets) + " |"
        sep = "|:--------|" + "|".join(":-------:" for _ in self.datasets) + "|"
        rows = []
        for i, d in enumerate(self.datasets):
            vals = " | ".join(
                f"{self.correlation_matrix[i, j]:.3f}"
                for j in range(len(self.datasets))
            )
            rows.append(f"| {d} | {vals} |")
        return "\n".join([header, sep] + rows)


@dataclass
class GeneralizationReport:
    """Full cross-dataset generalization report.

    Attributes:
        metric: Name of the scalar metric used for ranking.
        datasets: List of datasets included in the analysis.
        profiles: Per-tracker profiles sorted by rank stability (most
            consistent first, i.e. ascending ``rank_std``).
        correlation_table: Pairwise Spearman ρ between dataset rankings.
        mean_inter_dataset_correlation: Mean of all off-diagonal Spearman ρ
            values — a single scalar summarising overall ranking stability.
    """

    metric: str
    datasets: List[str]
    profiles: List[GeneralizationProfile]
    correlation_table: RankCorrelationTable
    mean_inter_dataset_correlation: float

    def __str__(self) -> str:
        lines = [
            f"GeneralizationReport(metric={self.metric}, "
            f"datasets={self.datasets}, "
            f"mean_ρ={self.mean_inter_dataset_correlation:.3f})",
        ]
        for p in self.profiles:
            lines.append(f"  {p}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Render the full report as a Markdown document."""
        tracker_header = (
            f"| # | Tracker | Mean {self.metric} | Rank Std | Best Dataset | Worst Dataset |"
        )
        sep_col = "-" * (len(self.metric) + 12)
        tracker_sep = f"|---|---------|{sep_col}|----------|--------------|---------------|"
        tracker_rows = []
        for i, p in enumerate(self.profiles, start=1):
            tracker_rows.append(
                f"| {i} | {p.tracker_name} | {p.mean_metric:.4f} "
                f"| {p.rank_std:.2f} | {p.best_dataset} | {p.worst_dataset} |"
            )

        return "\n".join([
            f"## Tracker Generalization Report — `{self.metric}`",
            "",
            f"**Datasets evaluated:** {', '.join(self.datasets)}",
            f"**Mean inter-dataset rank correlation (Spearman ρ):** "
            f"{self.mean_inter_dataset_correlation:.3f}",
            "",
            "### Per-Tracker Generalization Profiles",
            "> Sorted by rank stability (ascending rank std). "
            "Lower rank std = more consistent across datasets.",
            "",
            tracker_header,
            tracker_sep,
            *tracker_rows,
            "",
            "### Inter-Dataset Rank Correlation (Spearman ρ)",
            "> ρ near 1 = datasets agree on ranking; near 0 = uncorrelated.",
            "",
            self.correlation_table.to_markdown(),
        ])


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class GeneralizationAnalyzer:
    """Analyze how consistently tracker rankings transfer across datasets.

    Args:
        metric: Scalar metric key used to rank trackers per dataset.  Must
            appear in the dict returned by
            :meth:`~eovot.benchmark.engine.BenchmarkResult.summary`.

            Built-in aliases (resolved automatically):
              - ``"iou"`` → ``"mean_iou"``
              - ``"fps"`` → ``"mean_fps"``
              - ``"auc"`` → ``"success_auc"``
              - ``"precision"`` → ``"precision_auc"``
              - ``"memory"`` → ``"peak_memory_mb"``

        higher_is_better: When ``True`` (default) higher metric values
            earn a lower (better) rank number.  Set to ``False`` for metrics
            where lower is better (e.g. ``"memory"`` or ``"mean_latency_ms"``).

    Example::

        from eovot.metrics.generalization import GeneralizationAnalyzer

        analyzer = GeneralizationAnalyzer(metric="success_auc")
        report = analyzer.analyze(
            {"OTB100": [mosse_otb, kcf_otb], "GOT10k": [mosse_got, kcf_got]}
        )
        print(report.to_markdown())
    """

    _ALIASES: Dict[str, str] = {
        "iou": "mean_iou",
        "fps": "mean_fps",
        "auc": "success_auc",
        "precision": "precision_auc",
        "memory": "peak_memory_mb",
    }

    def __init__(
        self,
        metric: str = "mean_iou",
        higher_is_better: bool = True,
    ) -> None:
        self.metric = self._ALIASES.get(metric, metric)
        self.higher_is_better = higher_is_better

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        results: Dict[str, List["BenchmarkResult"]],
    ) -> GeneralizationReport:
        """Compute the generalization report across all provided datasets.

        Args:
            results: Mapping ``{dataset_name: [BenchmarkResult, ...]}``.
                Each inner list contains one :class:`~eovot.benchmark.engine.BenchmarkResult`
                per tracker evaluated on that dataset.  All datasets should
                cover the same set of tracker names; missing values default
                to ``0.0`` with a warning.

        Returns:
            :class:`GeneralizationReport` with per-tracker profiles, pairwise
            Spearman rank-correlation table, and aggregate statistics.

        Raises:
            ValueError: If fewer than two datasets are provided.
        """
        if len(results) < 2:
            raise ValueError(
                f"GeneralizationAnalyzer requires at least 2 datasets, "
                f"got {len(results)}: {list(results.keys())!r}"
            )

        datasets = sorted(results.keys())
        metric_map = self._extract_metrics(datasets, results)
        tracker_names = sorted(metric_map.keys())
        n_trackers = len(tracker_names)
        n_datasets = len(datasets)

        # rank_matrix[d_idx, t_idx] = rank of tracker t on dataset d
        rank_matrix = self._build_rank_matrix(
            datasets, tracker_names, metric_map, n_datasets, n_trackers
        )

        corr_matrix = self._spearman_matrix(rank_matrix)
        corr_table = RankCorrelationTable(
            datasets=datasets, correlation_matrix=corr_matrix
        )

        profiles = self._build_profiles(
            datasets, tracker_names, metric_map, rank_matrix
        )
        # Sort by rank_std ascending (most consistent tracker first)
        profiles.sort(key=lambda p: (p.rank_std, -p.mean_metric))

        off_diag = [
            corr_matrix[i, j]
            for i in range(n_datasets)
            for j in range(n_datasets)
            if i != j
        ]
        mean_corr = float(np.mean(off_diag)) if off_diag else 0.0

        return GeneralizationReport(
            metric=self.metric,
            datasets=datasets,
            profiles=profiles,
            correlation_table=corr_table,
            mean_inter_dataset_correlation=mean_corr,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_metrics(
        self,
        datasets: List[str],
        results: Dict[str, List["BenchmarkResult"]],
    ) -> Dict[str, Dict[str, float]]:
        """Build ``{tracker_name: {dataset: metric_value}}`` from results."""
        metric_map: Dict[str, Dict[str, float]] = {}
        for ds in datasets:
            for res in results[ds]:
                summary = res.summary()
                val = summary.get(self.metric)
                if val is None:
                    import warnings
                    warnings.warn(
                        f"Metric '{self.metric}' not found for tracker "
                        f"'{res.tracker_name}' on dataset '{ds}'. "
                        f"Defaulting to 0.0. Available: {list(summary.keys())}",
                        UserWarning,
                        stacklevel=3,
                    )
                    val = 0.0
                metric_map.setdefault(res.tracker_name, {})[ds] = float(val)
        return metric_map

    def _build_rank_matrix(
        self,
        datasets: List[str],
        tracker_names: List[str],
        metric_map: Dict[str, Dict[str, float]],
        n_datasets: int,
        n_trackers: int,
    ) -> np.ndarray:
        """Return ``(n_datasets, n_trackers)`` integer rank matrix."""
        rank_matrix = np.zeros((n_datasets, n_trackers), dtype=np.float64)
        for d_idx, ds in enumerate(datasets):
            vals = np.array(
                [metric_map[t].get(ds, 0.0) for t in tracker_names],
                dtype=np.float64,
            )
            # Sort descending for "higher is better", ascending otherwise
            order = np.argsort(-vals if self.higher_is_better else vals, kind="stable")
            for rank_pos, t_idx in enumerate(order):
                rank_matrix[d_idx, t_idx] = rank_pos + 1
        return rank_matrix

    def _build_profiles(
        self,
        datasets: List[str],
        tracker_names: List[str],
        metric_map: Dict[str, Dict[str, float]],
        rank_matrix: np.ndarray,
    ) -> List[GeneralizationProfile]:
        """Construct per-tracker :class:`GeneralizationProfile` objects."""
        profiles = []
        for t_idx, tracker in enumerate(tracker_names):
            tracker_ranks = rank_matrix[:, t_idx]
            metric_vals = {ds: metric_map[tracker].get(ds, 0.0) for ds in datasets}

            best_d_idx = int(np.argmin(tracker_ranks))   # lowest rank number = best
            worst_d_idx = int(np.argmax(tracker_ranks))  # highest rank number = worst

            profiles.append(
                GeneralizationProfile(
                    tracker_name=tracker,
                    datasets=datasets,
                    metric_values=metric_vals,
                    ranks={ds: int(tracker_ranks[d]) for d, ds in enumerate(datasets)},
                    mean_metric=float(np.mean(list(metric_vals.values()))),
                    rank_std=float(tracker_ranks.std()),
                    best_dataset=datasets[best_d_idx],
                    worst_dataset=datasets[worst_d_idx],
                )
            )
        return profiles

    # ------------------------------------------------------------------
    # Spearman ρ utilities (no scipy)
    # ------------------------------------------------------------------

    @staticmethod
    def _spearman_matrix(rank_matrix: np.ndarray) -> np.ndarray:
        """Compute the pairwise Spearman ρ matrix from a rank matrix.

        Args:
            rank_matrix: ``(n_datasets, n_trackers)`` float array of ranks.

        Returns:
            Symmetric ``(n_datasets, n_datasets)`` correlation matrix.
        """
        n_datasets = rank_matrix.shape[0]
        corr = np.eye(n_datasets, dtype=np.float64)
        for i in range(n_datasets):
            for j in range(i + 1, n_datasets):
                rho = GeneralizationAnalyzer._spearman_rho(
                    rank_matrix[i], rank_matrix[j]
                )
                corr[i, j] = rho
                corr[j, i] = rho
        return corr

    @staticmethod
    def _spearman_rho(r1: np.ndarray, r2: np.ndarray) -> float:
        """Closed-form Spearman ρ for two rank vectors (no scipy required).

        Uses the formula ρ = 1 − 6Σd²/n(n²−1) where d_i = r1_i − r2_i.
        This formula is exact when there are no ties; with ties it is a
        close approximation that avoids the need for a covariance computation.

        Args:
            r1: Rank vector for dataset A, shape ``(n_trackers,)``.
            r2: Rank vector for dataset B, shape ``(n_trackers,)``.

        Returns:
            Spearman ρ clamped to ``[-1, 1]``.  Returns 0.0 when n < 2.
        """
        n = len(r1)
        if n < 2:
            return 0.0
        d_sq = np.sum((r1 - r2) ** 2)
        rho = 1.0 - (6.0 * float(d_sq)) / (n * (n * n - 1))
        return float(max(-1.0, min(1.0, rho)))

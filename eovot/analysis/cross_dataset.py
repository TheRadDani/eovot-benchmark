"""Cross-dataset generalization analysis for tracker evaluation.

A tracker that achieves top accuracy on one benchmark does not necessarily
generalize — it may overfit to that dataset's domain (target type, motion
pattern, scene complexity).  This module quantifies *rank consistency* across
multiple datasets using Spearman's rank-order correlation and derives a
per-tracker generalization score.

Architecture
------------

``CrossDatasetResult``
    Immutable container of per-tracker, per-dataset accuracy summaries
    collected by :class:`CrossDatasetEvaluator`.

``CrossDatasetEvaluator``
    Orchestrates multi-dataset evaluation: runs each tracker on each
    dataset via :class:`~eovot.benchmark.engine.BenchmarkEngine`,
    assembles a ``CrossDatasetResult``, and exposes analysis methods.

    Key methods:

    * :meth:`evaluate` — run trackers × datasets and collect results.
    * :meth:`rank_correlation_matrix` — Spearman ρ between every
      pair of datasets; measures rank consistency.
    * :meth:`generalization_scores` — per-tracker mean Spearman ρ
      across all dataset pairs; single-number generalization proxy.
    * :meth:`to_markdown_report` — human-readable cross-dataset report.

Statistical note
~~~~~~~~~~~~~~~~
Spearman's rank-order correlation is used rather than Pearson's because
accuracy distributions are rarely Gaussian and outliers (failed trackers
or degenerate sequences) must not dominate the correlation estimate.
The implementation uses a pure-NumPy ranked-correlation formula that
matches scipy.stats.spearmanr on tied data (average-tie-rank convention).

Example::

    from eovot.analysis.cross_dataset import CrossDatasetEvaluator
    from eovot.datasets.synthetic import SyntheticDataset

    datasets = {
        "SyntheticA": SyntheticDataset(num_sequences=5, seed=0),
        "SyntheticB": SyntheticDataset(num_sequences=5, seed=1),
    }
    tracker_names = ["MOSSE", "KCF"]

    evaluator = CrossDatasetEvaluator(verbose=False)
    result = evaluator.evaluate(tracker_names, datasets)

    corr = evaluator.rank_correlation_matrix(result)
    scores = evaluator.generalization_scores(result)
    print(evaluator.to_markdown_report(result))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..benchmark.engine import BenchmarkEngine, BenchmarkResult
from ..trackers.registry import build_tracker


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class DatasetSummary:
    """Accuracy + efficiency summary for one tracker on one dataset."""

    tracker_name: str
    dataset_name: str
    mean_iou: float
    success_auc: float
    mean_fps: float
    peak_memory_mb: float
    num_sequences: int

    def rank_key(self, metric: str = "success_auc") -> float:
        """Return the scalar used for ranking (higher = better)."""
        return float(getattr(self, metric, self.success_auc))


@dataclass
class CrossDatasetResult:
    """All per-tracker, per-dataset summaries from one cross-dataset run.

    Attributes:
        summaries: Nested dict ``{tracker_name: {dataset_name: DatasetSummary}}``.
        dataset_names: Ordered list of dataset names used in the evaluation.
        tracker_names: Ordered list of tracker names used in the evaluation.
    """

    summaries: Dict[str, Dict[str, DatasetSummary]] = field(default_factory=dict)
    dataset_names: List[str] = field(default_factory=list)
    tracker_names: List[str] = field(default_factory=list)

    def get(self, tracker: str, dataset: str) -> Optional[DatasetSummary]:
        return self.summaries.get(tracker, {}).get(dataset)


# ---------------------------------------------------------------------------
# Spearman correlation (pure NumPy)
# ---------------------------------------------------------------------------

def _rank_data(arr: np.ndarray) -> np.ndarray:
    """Convert a 1-D array to average-tie ranks (1-based)."""
    n = len(arr)
    order = np.argsort(arr)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and arr[order[j]] == arr[order[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman's rank correlation between two 1-D arrays.

    Uses the average-tie-rank convention so results match
    scipy.stats.spearmanr exactly.  Returns ``float('nan')`` when either
    array has zero variance (all values equal).
    """
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    rx = _rank_data(x)
    ry = _rank_data(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom == 0:
        return float("nan")
    return float(np.dot(rx, ry) / denom)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class CrossDatasetEvaluator:
    """Evaluate a set of trackers across multiple datasets and analyse rank stability.

    Args:
        verbose:       Print per-tracker per-dataset progress.  Default ``True``.
        tdp_watts:     Optional CPU TDP for energy profiling, forwarded to
                       :class:`~eovot.benchmark.engine.BenchmarkEngine`.
        rank_metric:   Metric used for ranking trackers within each dataset.
                       Must be an attribute of :class:`DatasetSummary`.
                       Default ``"success_auc"``.
        max_sequences: Cap on sequences per dataset.  ``None`` = full dataset.
    """

    def __init__(
        self,
        verbose: bool = True,
        tdp_watts: Optional[float] = None,
        rank_metric: str = "success_auc",
        max_sequences: Optional[int] = None,
    ) -> None:
        self.verbose = verbose
        self.tdp_watts = tdp_watts
        self.rank_metric = rank_metric
        self.max_sequences = max_sequences

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        tracker_names: List[str],
        datasets: Dict[str, object],
        tracker_params: Optional[Dict[str, Dict]] = None,
    ) -> CrossDatasetResult:
        """Run every tracker on every dataset and collect summaries.

        Args:
            tracker_names:  List of tracker registry keys, e.g. ``["MOSSE", "KCF"]``.
            datasets:       Mapping ``{dataset_name: BaseDataset_instance}``.
            tracker_params: Optional per-tracker kwargs forwarded to
                            :func:`~eovot.trackers.registry.build_tracker`.

        Returns:
            :class:`CrossDatasetResult` populated with all summaries.
        """
        tracker_params = tracker_params or {}
        result = CrossDatasetResult(
            dataset_names=list(datasets.keys()),
            tracker_names=list(tracker_names),
        )

        for tracker_name in tracker_names:
            result.summaries[tracker_name] = {}
            params = tracker_params.get(tracker_name, {})

            for dataset_name, dataset in datasets.items():
                if self.verbose:
                    print(f"[cross-dataset] {tracker_name} on {dataset_name} ...", flush=True)

                tracker = build_tracker(tracker_name, **params)
                engine = BenchmarkEngine(verbose=False, tdp_watts=self.tdp_watts)
                br: BenchmarkResult = engine.run(
                    tracker,
                    dataset,
                    dataset_name=dataset_name,
                    max_sequences=self.max_sequences,
                )

                summary = DatasetSummary(
                    tracker_name=tracker_name,
                    dataset_name=dataset_name,
                    mean_iou=round(br.mean_iou, 4),
                    success_auc=round(br.mean_success_auc or 0.0, 4),
                    mean_fps=round(br.mean_fps, 2),
                    peak_memory_mb=round(br.peak_memory_mb, 2),
                    num_sequences=len(br.sequence_results),
                )
                result.summaries[tracker_name][dataset_name] = summary

        return result

    # ------------------------------------------------------------------
    # Rank correlation analysis
    # ------------------------------------------------------------------

    def _ranking_vectors(
        self, result: CrossDatasetResult
    ) -> Dict[str, np.ndarray]:
        """Return per-dataset arrays of rank-metric scores, one per tracker.

        Returns:
            ``{dataset_name: np.ndarray of shape (n_trackers,)}``
        """
        vectors: Dict[str, np.ndarray] = {}
        for ds in result.dataset_names:
            scores = []
            for tr in result.tracker_names:
                s = result.get(tr, ds)
                scores.append(s.rank_key(self.rank_metric) if s is not None else 0.0)
            vectors[ds] = np.array(scores, dtype=np.float64)
        return vectors

    def rank_correlation_matrix(
        self, result: CrossDatasetResult
    ) -> Dict[Tuple[str, str], float]:
        """Compute Spearman ρ between every pair of datasets.

        A high correlation (ρ ≈ 1) means trackers that rank well on dataset A
        also rank well on dataset B — the ranking generalizes.  Low or negative
        ρ means the rankings are inconsistent: a tracker specialized for A may
        under-perform on B.

        Args:
            result: Output of :meth:`evaluate`.

        Returns:
            Dict mapping ``(dataset_A, dataset_B)`` to Spearman ρ ∈ [−1, 1].
            The matrix is symmetric; diagonal entries are 1.0.
        """
        vectors = self._ranking_vectors(result)
        names = result.dataset_names
        corr: Dict[Tuple[str, str], float] = {}
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i == j:
                    corr[(a, b)] = 1.0
                elif (b, a) in corr:
                    corr[(a, b)] = corr[(b, a)]
                else:
                    corr[(a, b)] = _spearman_rho(vectors[a], vectors[b])
        return corr

    def generalization_scores(
        self, result: CrossDatasetResult
    ) -> Dict[str, float]:
        """Compute a per-tracker generalization score.

        The score is the mean Spearman ρ between each tracker's rank on one
        dataset and its rank on every other dataset, averaged over all pairs.
        A score near 1.0 means the tracker's relative ranking is stable across
        datasets (strong generalizer); a score near 0 or below means its
        performance is dataset-specific.

        Args:
            result: Output of :meth:`evaluate`.

        Returns:
            ``{tracker_name: generalization_score}`` sorted descending.
        """
        n_trackers = len(result.tracker_names)
        if n_trackers < 2 or len(result.dataset_names) < 2:
            return {t: float("nan") for t in result.tracker_names}

        # For each tracker, collect its rank (by rank_metric) on each dataset.
        vectors = self._ranking_vectors(result)
        dataset_names = result.dataset_names

        # Build (n_datasets, n_trackers) matrix of scores.
        mat = np.stack([vectors[ds] for ds in dataset_names], axis=0)

        scores: Dict[str, float] = {}
        for ti, tracker in enumerate(result.tracker_names):
            tracker_row = mat[:, ti]  # shape (n_datasets,)
            rhos = []
            for ti2, other in enumerate(result.tracker_names):
                if ti2 == ti:
                    continue
                other_row = mat[:, ti2]
                rhos.append(_spearman_rho(tracker_row, other_row))
            # Mean ignoring NaN
            valid = [r for r in rhos if not np.isnan(r)]
            scores[tracker] = float(np.mean(valid)) if valid else float("nan")

        return dict(sorted(scores.items(), key=lambda x: x[1] if not np.isnan(x[1]) else -99, reverse=True))

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def to_markdown_report(self, result: CrossDatasetResult) -> str:
        """Generate a Markdown cross-dataset comparison report.

        The report contains:

        1. Per-dataset accuracy table (trackers as rows, datasets as columns).
        2. Spearman ρ correlation matrix between datasets.
        3. Generalization score ranking.

        Args:
            result: Output of :meth:`evaluate`.

        Returns:
            Multi-section Markdown string suitable for embedding in a README
            or experiment log.
        """
        lines: List[str] = []
        dsets = result.dataset_names
        trks = result.tracker_names

        # --- Section 1: accuracy table ---
        lines.append("## Cross-Dataset Accuracy Table\n")
        metric_label = self.rank_metric.replace("_", " ").title()
        header = "| Tracker | " + " | ".join(dsets) + " | Mean |"
        sep = "| --- | " + " | ".join(["---"] * len(dsets)) + " | --- |"
        lines.append(header)
        lines.append(sep)

        for tr in trks:
            row_vals = []
            for ds in dsets:
                s = result.get(tr, ds)
                val = s.rank_key(self.rank_metric) if s is not None else float("nan")
                row_vals.append(val)
            mean_val = float(np.nanmean(row_vals))
            cells = " | ".join(f"{v:.4f}" if not np.isnan(v) else "n/a" for v in row_vals)
            lines.append(f"| {tr} | {cells} | {mean_val:.4f} |")

        lines.append(f"\n*Metric: {metric_label}*\n")

        # --- Section 2: Spearman ρ matrix ---
        lines.append("## Dataset Rank Correlation Matrix (Spearman ρ)\n")
        corr = self.rank_correlation_matrix(result)
        header2 = "| Dataset | " + " | ".join(dsets) + " |"
        sep2 = "| --- | " + " | ".join(["---"] * len(dsets)) + " |"
        lines.append(header2)
        lines.append(sep2)

        for a in dsets:
            cells2 = " | ".join(
                f"{corr.get((a, b), float('nan')):.3f}" for b in dsets
            )
            lines.append(f"| {a} | {cells2} |")

        lines.append(
            "\n*ρ close to 1.0 = tracker rankings consistent across datasets; "
            "ρ near 0 = rankings dataset-specific.*\n"
        )

        # --- Section 3: generalization scores ---
        lines.append("## Tracker Generalization Scores\n")
        lines.append("| Tracker | Generalization Score | Interpretation |")
        lines.append("| --- | --- | --- |")

        gen_scores = self.generalization_scores(result)
        for tr, score in gen_scores.items():
            if np.isnan(score):
                interp = "insufficient data"
            elif score >= 0.7:
                interp = "strong generalizer"
            elif score >= 0.4:
                interp = "moderate generalizer"
            else:
                interp = "dataset-specific"
            score_str = f"{score:.3f}" if not np.isnan(score) else "n/a"
            lines.append(f"| {tr} | {score_str} | {interp} |")

        lines.append(
            "\n*Score = mean Spearman ρ of a tracker's cross-dataset rank vector with all other trackers.*\n"
        )

        return "\n".join(lines)

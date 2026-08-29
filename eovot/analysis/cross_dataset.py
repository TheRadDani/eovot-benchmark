"""Cross-dataset generalization analysis for EOVOT.

A tracker that achieves high IoU on OTB may fail on LaSOT's longer sequences
or GOT-10k's unseen object classes.  *Generalizability* — consistent strong
performance across diverse datasets — is a critical property for real-world
edge deployment, where the deployment domain is rarely identical to any single
benchmark.

This module quantifies generalizability from ``BenchmarkResult`` objects
already produced by ``BenchmarkEngine.run()``:

GeneralizationMatrix
~~~~~~~~~~~~~~~~~~~~
A tracker × dataset table of scalar metrics (mIoU, success AUC, FPS, EES).
Rows are trackers; columns are datasets.  Each cell holds a ``MetricBundle``
with the four core scalars so any metric can be sliced for analysis.

GeneralizationScore
~~~~~~~~~~~~~~~~~~~
A composite scalar per tracker that rewards *consistent* performance across
datasets more than peak performance on any single one::

    gen_score = mean_rank_normalized_metric - consistency_penalty

where ``mean_rank_normalized_metric`` is the mean of each metric's value
normalized to ``[0, 1]`` across trackers *on that dataset*, and
``consistency_penalty = std_dev_of_normalized_metrics`` penalises trackers
whose rank swings wildly between datasets.

Trackers with a high generalization score are reliably deployable across
unseen environments — the primary concern for an edge-device fleet.

Typical usage::

    from eovot.analysis.cross_dataset import CrossDatasetAnalyzer

    analyzer = CrossDatasetAnalyzer()

    # results_by_dataset: {dataset_name: [BenchmarkResult, ...]}
    report = analyzer.analyze(results_by_dataset)

    print(report.to_markdown_table())
    print(report.to_generalization_ranking())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MetricBundle:
    """Core scalars for one (tracker, dataset) cell in the generalization matrix."""

    mean_iou: float
    success_auc: float
    fps: float
    ees: float  # Edge Efficiency Score


@dataclass
class GeneralizationEntry:
    """Per-tracker generalization summary across all datasets.

    Attributes:
        tracker_name: Identifier for the tracker.
        datasets: Ordered list of dataset names included in this analysis.
        metrics: Mapping ``{dataset_name: MetricBundle}``.
        mean_iou_per_dataset: ``{dataset: mean_iou}`` for quick access.
        generalization_score: Composite score in ``[0, 1]``; higher = more
            consistently good across all datasets.
        consistency_penalty: Standard deviation of normalized mIoU across
            datasets; lower = more consistent.
        mean_normalized_iou: Mean of dataset-normalized mIoU values in
            ``[0, 1]`` — tracker's average rank position.
    """

    tracker_name: str
    datasets: List[str]
    metrics: Dict[str, MetricBundle]
    mean_normalized_iou: float
    consistency_penalty: float
    generalization_score: float


@dataclass
class GeneralizationReport:
    """Full cross-dataset analysis output.

    Attributes:
        entries: Per-tracker entries sorted by ``generalization_score`` desc.
        datasets: Dataset names in the order they appear in tables.
        metric: Which metric was used to rank; always ``"mean_iou"`` for now.
    """

    entries: List[GeneralizationEntry]
    datasets: List[str]
    metric: str = "mean_iou"

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def to_markdown_table(self, metric: str = "mean_iou") -> str:
        """Return a Markdown generalization matrix (trackers × datasets).

        Args:
            metric: Which metric to display: ``"mean_iou"``, ``"success_auc"``,
                ``"fps"``, or ``"ees"``.  Default: ``"mean_iou"``.

        Returns:
            Multi-line Markdown string with one column per dataset plus
            a ``Gen. Score`` summary column.
        """
        ds = self.datasets
        header = "| Tracker | " + " | ".join(ds) + " | Gen. Score | Consistency |"
        sep = "|---------|" + "|".join(["---------:"] * len(ds)) + "|-----------:|------------:|"

        lines = [f"## Cross-Dataset Generalization Matrix ({metric})\n", header, sep]
        for e in self.entries:
            cells = []
            for d in ds:
                bundle = e.metrics.get(d)
                if bundle is None:
                    cells.append("—")
                else:
                    val = getattr(bundle, metric, None)
                    cells.append(f"{val:.4f}" if val is not None else "—")
            row = (
                f"| {e.tracker_name} | "
                + " | ".join(cells)
                + f" | {e.generalization_score:.4f} | {e.consistency_penalty:.4f} |"
            )
            lines.append(row)
        return "\n".join(lines)

    def to_generalization_ranking(self) -> str:
        """Return a Markdown ranking table sorted by generalization score."""
        lines = [
            "## Tracker Generalization Ranking\n",
            "| Rank | Tracker | Gen. Score | Mean Norm. IoU | Consistency Penalty |",
            "|------|---------|----------:|---------------:|--------------------:|",
        ]
        for rank, e in enumerate(self.entries, start=1):
            lines.append(
                f"| {rank} | {e.tracker_name} "
                f"| {e.generalization_score:.4f} "
                f"| {e.mean_normalized_iou:.4f} "
                f"| {e.consistency_penalty:.4f} |"
            )
        return "\n".join(lines)

    def to_csv(self, metric: str = "mean_iou") -> str:
        """Return a CSV string of the generalization matrix.

        Args:
            metric: Which metric to export. Default: ``"mean_iou"``.

        Returns:
            CSV with header row and one data row per tracker.
        """
        ds = self.datasets
        header = "tracker," + ",".join(ds) + ",gen_score,consistency_penalty"
        rows = [header]
        for e in self.entries:
            vals = []
            for d in ds:
                bundle = e.metrics.get(d)
                val = getattr(bundle, metric, None) if bundle else None
                vals.append(f"{val:.6f}" if val is not None else "")
            rows.append(
                f"{e.tracker_name},"
                + ",".join(vals)
                + f",{e.generalization_score:.6f},{e.consistency_penalty:.6f}"
            )
        return "\n".join(rows)

    def to_latex_table(self, metric: str = "mean_iou", caption: str = "") -> str:
        """Return a LaTeX ``tabular`` for direct inclusion in a paper.

        Args:
            metric: Metric to display. Default: ``"mean_iou"``.
            caption: Optional table caption.

        Returns:
            LaTeX source string with ``\\begin{table}``…``\\end{table}`` wrapper.
        """
        ds = self.datasets
        n_cols = len(ds) + 3  # tracker + datasets + gen_score + consistency
        col_spec = "l" + "r" * (n_cols - 1)

        header_row = " & ".join(
            ["Tracker"] + ds + ["Gen.\\,Score", "Consistency"]
        ) + " \\\\"

        data_rows = []
        for e in self.entries:
            cells = [e.tracker_name.replace("_", r"\_")]
            for d in ds:
                bundle = e.metrics.get(d)
                val = getattr(bundle, metric, None) if bundle else None
                cells.append(f"{val:.4f}" if val is not None else "---")
            cells += [f"{e.generalization_score:.4f}", f"{e.consistency_penalty:.4f}"]
            data_rows.append(" & ".join(cells) + " \\\\")

        caption_line = f"  \\caption{{{caption}}}\n" if caption else ""
        return (
            "\\begin{table}[t]\n"
            "  \\centering\n"
            f"  \\begin{{tabular}}{{{col_spec}}}\n"
            "    \\toprule\n"
            f"    {header_row}\n"
            "    \\midrule\n"
            + "\n".join(f"    {r}" for r in data_rows)
            + "\n"
            "    \\bottomrule\n"
            "  \\end{tabular}\n"
            + caption_line
            + "\\end{table}"
        )


# ---------------------------------------------------------------------------
# Core analyzer
# ---------------------------------------------------------------------------

class CrossDatasetAnalyzer:
    """Compute generalization scores from multi-dataset benchmark results.

    Args:
        memory_budget_mb: Memory budget used when computing EES inside each
            ``MetricBundle``.  Default: ``512.0`` MB.
    """

    def __init__(self, memory_budget_mb: float = 512.0) -> None:
        self.memory_budget_mb = memory_budget_mb

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        results_by_dataset: Dict[str, List["BenchmarkResult"]],
    ) -> GeneralizationReport:
        """Build a full generalization report from per-dataset results.

        Args:
            results_by_dataset: Mapping ``{dataset_name: [BenchmarkResult, ...]}``.
                Each list may contain results for different trackers on the
                same dataset; the tracker name is taken from
                ``BenchmarkResult.tracker_name``.

        Returns:
            :class:`GeneralizationReport` with entries sorted by
            generalization score (highest first).

        Raises:
            ValueError: If no common tracker appears in more than one dataset.
        """
        datasets = sorted(results_by_dataset)

        # Build raw metric matrix: {tracker: {dataset: MetricBundle}}
        matrix: Dict[str, Dict[str, MetricBundle]] = {}
        for ds_name, results in results_by_dataset.items():
            for res in results:
                name = res.tracker_name
                if name not in matrix:
                    matrix[name] = {}
                matrix[name][ds_name] = self._extract_bundle(res)

        trackers = sorted(matrix)

        # Normalize mIoU per dataset to [0, 1] across trackers present on that
        # dataset.  This lets us compare trackers even when some are missing
        # from certain datasets.
        norm_iou: Dict[str, Dict[str, float]] = {t: {} for t in trackers}
        for ds in datasets:
            present = [(t, matrix[t][ds].mean_iou) for t in trackers if ds in matrix[t]]
            if not present:
                continue
            vals = [v for _, v in present]
            lo, hi = min(vals), max(vals)
            span = hi - lo if hi > lo else 1.0
            for t, v in present:
                norm_iou[t][ds] = (v - lo) / span

        entries: List[GeneralizationEntry] = []
        for t in trackers:
            norm_vals = list(norm_iou[t].values())
            if not norm_vals:
                continue
            mean_norm = float(np.mean(norm_vals))
            # Consistency penalty: std dev of normalized IoU across datasets.
            # Use population std (ddof=0) — even with 2 datasets, std is
            # meaningful as a spread measure.
            consistency_penalty = float(np.std(norm_vals, ddof=0))
            gen_score = max(0.0, mean_norm - consistency_penalty)

            entries.append(
                GeneralizationEntry(
                    tracker_name=t,
                    datasets=datasets,
                    metrics=matrix[t],
                    mean_normalized_iou=mean_norm,
                    consistency_penalty=consistency_penalty,
                    generalization_score=gen_score,
                )
            )

        entries.sort(key=lambda e: e.generalization_score, reverse=True)
        return GeneralizationReport(entries=entries, datasets=datasets)

    def compare_pair(
        self,
        results_a: List["BenchmarkResult"],
        results_b: List["BenchmarkResult"],
        name_a: str,
        name_b: str,
    ) -> Dict:
        """Convenience wrapper: compare two trackers across several datasets.

        Args:
            results_a: One ``BenchmarkResult`` per dataset for tracker A.
            results_b: One ``BenchmarkResult`` per dataset for tracker B.
            name_a: Human-readable name for tracker A (used as dict key if
                tracker_name differs across results).
            name_b: Human-readable name for tracker B.

        Returns:
            Dict with ``"tracker_a"``, ``"tracker_b"``, and ``"advantage_a"``
            (datasets where A outperforms B on mIoU) and ``"advantage_b"``.
        """
        by_ds: Dict[str, List["BenchmarkResult"]] = {}
        for r in results_a + results_b:
            by_ds.setdefault(r.dataset_name, []).append(r)
        report = self.analyze(by_ds)

        entry_a = next((e for e in report.entries if e.tracker_name == results_a[0].tracker_name), None)
        entry_b = next((e for e in report.entries if e.tracker_name == results_b[0].tracker_name), None)

        advantage_a, advantage_b = [], []
        for ds in report.datasets:
            m_a = entry_a.metrics.get(ds) if entry_a else None
            m_b = entry_b.metrics.get(ds) if entry_b else None
            if m_a is not None and m_b is not None:
                if m_a.mean_iou > m_b.mean_iou:
                    advantage_a.append(ds)
                elif m_b.mean_iou > m_a.mean_iou:
                    advantage_b.append(ds)

        return {
            "tracker_a": name_a,
            "tracker_b": name_b,
            "advantage_a": advantage_a,
            "advantage_b": advantage_b,
            "gen_score_a": entry_a.generalization_score if entry_a else None,
            "gen_score_b": entry_b.generalization_score if entry_b else None,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_bundle(self, res: "BenchmarkResult") -> MetricBundle:
        """Extract a :class:`MetricBundle` from a ``BenchmarkResult``."""
        import math

        mean_iou = res.mean_iou
        fps = res.mean_fps
        success_auc = res.mean_success_auc if res.mean_success_auc is not None else mean_iou
        peak_mem = res.peak_memory_mb
        ees = (
            mean_iou * math.log1p(fps) / (1.0 + peak_mem / self.memory_budget_mb)
            if fps > 0 else 0.0
        )
        return MetricBundle(
            mean_iou=mean_iou,
            success_auc=success_auc,
            fps=fps,
            ees=ees,
        )

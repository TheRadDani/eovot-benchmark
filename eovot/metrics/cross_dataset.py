"""Multi-dataset leaderboard aggregator for cross-benchmark tracker evaluation.

Assembles per-tracker performance summaries from multiple dataset benchmarks
into a unified ranked leaderboard.  This is the standard evaluation mode for
papers that compare trackers on OTB100, GOT-10k, and LaSOT simultaneously.

Key capabilities:

- **Dataset-weighted aggregate IoU** — configurable per-dataset importance
  weights so that larger or harder benchmarks can count for more.
- **Rank consistency** — Spearman's rank-correlation coefficient averaged
  across all dataset pairs, quantifying how consistently trackers rank
  across benchmarks.  A value near 1 means the leaderboard is
  dataset-agnostic; near 0 means rankings are noisy or dataset-specific.
- **Markdown leaderboard** — publication-ready table with per-dataset
  columns and a final weighted aggregate column.
- **JSON export** — machine-readable dict for downstream tooling.

Typical usage::

    from eovot.metrics.cross_dataset import CrossDatasetEvaluator

    # results maps dataset_name -> list[BenchmarkResult], one entry per tracker
    results = {
        "OTB100": [mosse_otb,  kcf_otb,  csrt_otb],
        "GOT10k": [mosse_got,  kcf_got,  csrt_got],
        "LaSOT":  [mosse_lasot, kcf_lasot, csrt_lasot],
    }

    evaluator = CrossDatasetEvaluator(
        dataset_weights={"OTB100": 1.0, "GOT10k": 1.5, "LaSOT": 2.0}
    )
    leaderboard = evaluator.build_leaderboard(results)
    print(evaluator.to_markdown(leaderboard))
    print(f"Rank consistency ρ = {evaluator.rank_consistency(leaderboard):.3f}")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DatasetEntry:
    """Per-dataset performance record for one tracker."""

    dataset_name: str
    mean_iou: float
    mean_fps: float
    success_auc: Optional[float] = None
    precision_auc: Optional[float] = None

    def to_dict(self) -> dict:
        d: dict = {
            "mean_iou": round(self.mean_iou, 4),
            "mean_fps": round(self.mean_fps, 2),
        }
        if self.success_auc is not None:
            d["success_auc"] = round(self.success_auc, 4)
        if self.precision_auc is not None:
            d["precision_auc"] = round(self.precision_auc, 4)
        return d


@dataclass
class AggregateEntry:
    """Cross-dataset aggregate performance record for one tracker."""

    tracker_name: str
    dataset_entries: Dict[str, DatasetEntry] = field(default_factory=dict)
    weighted_iou: float = 0.0
    rank: int = 0

    def __str__(self) -> str:
        datasets = ", ".join(
            f"{d}={e.mean_iou:.3f}" for d, e in sorted(self.dataset_entries.items())
        )
        return (
            f"AggregateEntry[#{self.rank} {self.tracker_name}] "
            f"weighted_iou={self.weighted_iou:.4f}  ({datasets})"
        )


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class CrossDatasetEvaluator:
    """Aggregate tracker performance across multiple dataset benchmarks.

    Args:
        dataset_weights: Optional mapping from dataset name to a positive
            real-valued weight.  Datasets absent from this dict receive
            weight ``1.0`` (equal contribution).  Weights are normalised
            internally so only their relative magnitudes matter.

    Example::

        evaluator = CrossDatasetEvaluator(
            dataset_weights={"OTB100": 1.0, "GOT10k": 1.5, "LaSOT": 2.0}
        )
        leaderboard = evaluator.build_leaderboard(results)
        md = evaluator.to_markdown(leaderboard)
        consistency = evaluator.rank_consistency(leaderboard)
    """

    def __init__(
        self,
        dataset_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        if dataset_weights is not None:
            for k, v in dataset_weights.items():
                if v <= 0:
                    raise ValueError(
                        f"dataset_weights['{k}'] = {v} must be positive."
                    )
        self.dataset_weights: Dict[str, float] = dataset_weights or {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def build_leaderboard(
        self,
        results: Dict[str, List[BenchmarkResult]],
    ) -> List[AggregateEntry]:
        """Build a cross-dataset leaderboard from per-dataset benchmark results.

        Args:
            results: Mapping ``{dataset_name: [BenchmarkResult, ...]}``.
                Each list should contain one :class:`BenchmarkResult` per
                tracker.  The same tracker names should appear in every
                dataset; a tracker missing from a dataset is omitted from
                that dataset's column in the table but still receives a
                weighted IoU calculated over the datasets it did appear in.

        Returns:
            List of :class:`AggregateEntry` sorted by ``weighted_iou``
            descending (rank 1 = best).  Returns an empty list when
            *results* is empty or contains no BenchmarkResult objects.
        """
        if not results:
            return []

        # Collect ordered unique tracker names (preserve insertion order)
        tracker_names: List[str] = []
        seen: set = set()
        for dataset_results in results.values():
            for r in dataset_results:
                if r.tracker_name not in seen:
                    tracker_names.append(r.tracker_name)
                    seen.add(r.tracker_name)

        if not tracker_names:
            return []

        # Build per-tracker aggregate entries
        entries: Dict[str, AggregateEntry] = {
            name: AggregateEntry(tracker_name=name) for name in tracker_names
        }

        for dataset_name, dataset_results in results.items():
            for r in dataset_results:
                de = DatasetEntry(
                    dataset_name=dataset_name,
                    mean_iou=r.mean_iou,
                    mean_fps=r.mean_fps,
                    success_auc=r.mean_success_auc,
                    precision_auc=r.mean_precision_auc,
                )
                entries[r.tracker_name].dataset_entries[dataset_name] = de

        # Compute weighted IoU for each tracker
        for agg in entries.values():
            total_w = 0.0
            weighted_sum = 0.0
            for ds, de in agg.dataset_entries.items():
                w = self.dataset_weights.get(ds, 1.0)
                weighted_sum += w * de.mean_iou
                total_w += w
            agg.weighted_iou = weighted_sum / total_w if total_w > 0 else 0.0

        # Sort by weighted IoU descending and assign 1-indexed ranks
        ranked = sorted(entries.values(), key=lambda e: e.weighted_iou, reverse=True)
        for i, entry in enumerate(ranked):
            entry.rank = i + 1

        return ranked

    def rank_consistency(self, leaderboard: List[AggregateEntry]) -> float:
        """Mean Spearman rank-correlation across all dataset pairs.

        Measures whether tracker rankings are consistent across different
        benchmarks.  A value of 1.0 means identical ranking on every
        dataset; 0.0 means uncorrelated rankings; −1.0 means inverted.

        For papers: a high rank consistency (ρ > 0.8) supports the claim
        that the leaderboard generalises across datasets.

        Args:
            leaderboard: Output of :meth:`build_leaderboard`.

        Returns:
            Mean Spearman ρ over all dataset pairs, or ``0.0`` when fewer
            than two datasets or fewer than two trackers are present.
        """
        if len(leaderboard) < 2:
            return 0.0

        all_datasets = sorted(
            {ds for entry in leaderboard for ds in entry.dataset_entries}
        )
        if len(all_datasets) < 2:
            return 0.0

        n = len(leaderboard)
        iou_matrix = np.full((n, len(all_datasets)), np.nan, dtype=np.float64)
        for i, entry in enumerate(leaderboard):
            for j, ds in enumerate(all_datasets):
                if ds in entry.dataset_entries:
                    iou_matrix[i, j] = entry.dataset_entries[ds].mean_iou

        correlations = []
        n_ds = len(all_datasets)
        for a in range(n_ds):
            for b in range(a + 1, n_ds):
                col_a = iou_matrix[:, a]
                col_b = iou_matrix[:, b]
                mask = ~(np.isnan(col_a) | np.isnan(col_b))
                if mask.sum() < 2:
                    continue
                rho = _spearman(col_a[mask], col_b[mask])
                correlations.append(rho)

        return float(np.mean(correlations)) if correlations else 0.0

    # ------------------------------------------------------------------
    # Output formats
    # ------------------------------------------------------------------

    def to_markdown(self, leaderboard: List[AggregateEntry]) -> str:
        """Render the leaderboard as a Markdown table.

        Columns: Rank, Tracker, one mIoU column per dataset, Weighted IoU.
        Missing values for a tracker/dataset pair are shown as ``—``.

        Args:
            leaderboard: Output of :meth:`build_leaderboard`.

        Returns:
            Multi-line Markdown string, ready for a README or paper appendix.
        """
        if not leaderboard:
            return "_No results._"

        all_datasets = sorted(
            {ds for e in leaderboard for ds in e.dataset_entries}
        )

        header = ["Rank", "Tracker"] + [f"{d} mIoU" for d in all_datasets] + ["Weighted IoU"]
        sep = [":---:", ":---"] + [":---:"] * len(all_datasets) + [":---:"]

        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(sep) + " |",
        ]
        for entry in leaderboard:
            cells = [f"#{entry.rank}", entry.tracker_name]
            for ds in all_datasets:
                if ds in entry.dataset_entries:
                    cells.append(f"{entry.dataset_entries[ds].mean_iou:.4f}")
                else:
                    cells.append("—")
            cells.append(f"**{entry.weighted_iou:.4f}**")
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    def to_dict(self, leaderboard: List[AggregateEntry]) -> dict:
        """Serialise the leaderboard to a JSON-compatible plain dict.

        Args:
            leaderboard: Output of :meth:`build_leaderboard`.

        Returns:
            Dict with keys ``"dataset_names"`` (list) and ``"leaderboard"``
            (list of per-tracker dicts with rank, tracker name, weighted IoU,
            and per-dataset metrics).
        """
        all_datasets = sorted(
            {ds for e in leaderboard for ds in e.dataset_entries}
        )
        entries = []
        for entry in leaderboard:
            per_ds: dict = {}
            for ds in all_datasets:
                if ds in entry.dataset_entries:
                    per_ds[ds] = entry.dataset_entries[ds].to_dict()
            entries.append({
                "rank": entry.rank,
                "tracker": entry.tracker_name,
                "weighted_iou": round(entry.weighted_iou, 4),
                "datasets": per_ds,
            })
        return {"dataset_names": all_datasets, "leaderboard": entries}


# ---------------------------------------------------------------------------
# Internal helpers (no scipy dependency)
# ---------------------------------------------------------------------------

def _rank_vector(arr: np.ndarray) -> np.ndarray:
    """Convert a 1-D float array to fractional ranks (average-tie method)."""
    n = len(arr)
    order = np.argsort(arr)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    # Average ranks for tied values
    uniq_vals, inv, counts = np.unique(arr, return_inverse=True, return_counts=True)
    for g in range(len(uniq_vals)):
        if counts[g] > 1:
            mask = inv == g
            ranks[mask] = ranks[mask].mean()
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman's rank correlation coefficient (pure NumPy, no scipy)."""
    n = len(a)
    if n < 2:
        return 0.0
    ra = _rank_vector(a)
    rb = _rank_vector(b)
    d2 = float(np.sum((ra - rb) ** 2))
    denom = n * (n * n - 1)
    if denom == 0:
        return 0.0
    return 1.0 - 6.0 * d2 / denom

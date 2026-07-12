"""Cross-dataset result aggregator for comparing tracker performance across sessions."""
from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence


@dataclass
class TrackerSummary:
    """Per-tracker-per-dataset performance summary loaded from a JSON result file."""

    tracker: str
    dataset: str
    num_sequences: int

    # Accuracy
    mean_iou: float
    success_auc: float
    precision_auc: float
    mean_center_distance_px: float

    # Efficiency
    mean_fps: float
    peak_memory_mb: float
    total_energy_j: float
    mean_energy_per_frame_mj: float

    # Optional metadata
    extra: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def from_dict(cls, d: dict) -> "TrackerSummary":
        known = {
            "tracker", "dataset", "num_sequences",
            "mean_iou", "success_auc", "precision_auc", "mean_center_distance_px",
            "mean_fps", "peak_memory_mb", "total_energy_j", "mean_energy_per_frame_mj",
        }
        core = {k: d[k] for k in known if k in d}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(**core, extra=extra)  # type: ignore[arg-type]

    def to_dict(self) -> dict:
        base = asdict(self)
        extra = base.pop("extra", {})
        base.update(extra)
        return base


# --------------------------------------------------------------------------- #
# Composite edge-deployment score
# --------------------------------------------------------------------------- #

def _composite_score(
    summary: TrackerSummary,
    *,
    accuracy_weight: float = 0.5,
    speed_weight: float = 0.3,
    memory_weight: float = 0.2,
    fps_scale: float = 30.0,
    mem_scale: float = 512.0,
) -> float:
    """Weighted composite score in [0, 1] suitable for edge deployment ranking.

    Higher is better.  Weights must sum to 1; the caller is responsible for that.
    """
    accuracy = (summary.mean_iou + summary.success_auc) / 2.0
    speed = min(summary.mean_fps / fps_scale, 1.0)
    memory = max(0.0, 1.0 - summary.peak_memory_mb / mem_scale)
    return accuracy_weight * accuracy + speed_weight * speed + memory_weight * memory


# --------------------------------------------------------------------------- #
# ResultAggregator
# --------------------------------------------------------------------------- #

class ResultAggregator:
    """Load, merge, and rank tracker results across datasets and benchmark sessions.

    Usage::

        agg = ResultAggregator()
        agg.load_many(["results/run1.json", "results/run2.json"])
        print(agg.leaderboard())
    """

    def __init__(
        self,
        accuracy_weight: float = 0.5,
        speed_weight: float = 0.3,
        memory_weight: float = 0.2,
        fps_scale: float = 30.0,
        mem_scale: float = 512.0,
    ) -> None:
        self._entries: List[TrackerSummary] = []
        self._acc_w = accuracy_weight
        self._spd_w = speed_weight
        self._mem_w = memory_weight
        self._fps_scale = fps_scale
        self._mem_scale = mem_scale

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def load(self, path: str | Path) -> "ResultAggregator":
        """Load one JSON result file and append its entries."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Result file not found: {path}")
        with path.open() as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")
        for item in data:
            self._entries.append(TrackerSummary.from_dict(item))
        return self

    def load_many(self, paths: Sequence[str | Path]) -> "ResultAggregator":
        """Load multiple JSON result files."""
        for p in paths:
            self.load(p)
        return self

    def add(self, summary: TrackerSummary) -> "ResultAggregator":
        """Manually append a TrackerSummary entry."""
        self._entries.append(summary)
        return self

    # ------------------------------------------------------------------ #
    # Querying
    # ------------------------------------------------------------------ #

    def entries(
        self,
        *,
        dataset: Optional[str] = None,
        tracker: Optional[str] = None,
    ) -> List[TrackerSummary]:
        """Return entries optionally filtered by dataset and/or tracker name."""
        result = self._entries
        if dataset is not None:
            result = [e for e in result if e.dataset == dataset]
        if tracker is not None:
            result = [e for e in result if e.tracker == tracker]
        return list(result)

    def ranked_entries(
        self,
        *,
        dataset: Optional[str] = None,
        sort_by: str = "composite",
    ) -> List[Dict]:
        """Return entries sorted by *sort_by* (descending), each augmented with a rank.

        *sort_by* options: ``composite``, ``mean_iou``, ``success_auc``,
        ``precision_auc``, ``mean_fps``, ``peak_memory_mb`` (ascending for memory).
        """
        subset = self.entries(dataset=dataset)
        ascending = sort_by == "peak_memory_mb"

        def key(s: TrackerSummary) -> float:
            if sort_by == "composite":
                return self._score(s)
            val = getattr(s, sort_by, None)
            if val is None:
                raise ValueError(f"Unknown sort_by field: {sort_by!r}")
            return float(val)

        ranked = sorted(subset, key=key, reverse=not ascending)
        out = []
        for rank, entry in enumerate(ranked, start=1):
            row = entry.to_dict()
            row["composite_score"] = round(self._score(entry), 4)
            row["rank"] = rank
            out.append(row)
        return out

    def aggregate_by_tracker(
        self,
        *,
        dataset: Optional[str] = None,
    ) -> Dict[str, Dict]:
        """Average metrics across datasets for each tracker.

        Returns a dict keyed by tracker name.
        """
        subset = self.entries(dataset=dataset)
        groups: Dict[str, List[TrackerSummary]] = {}
        for e in subset:
            groups.setdefault(e.tracker, []).append(e)

        agg = {}
        for tracker, items in groups.items():
            n = len(items)
            agg[tracker] = {
                "tracker": tracker,
                "num_datasets": n,
                "mean_iou": round(sum(i.mean_iou for i in items) / n, 4),
                "success_auc": round(sum(i.success_auc for i in items) / n, 4),
                "precision_auc": round(sum(i.precision_auc for i in items) / n, 4),
                "mean_fps": round(sum(i.mean_fps for i in items) / n, 2),
                "peak_memory_mb": round(sum(i.peak_memory_mb for i in items) / n, 2),
                "total_energy_j": round(sum(i.total_energy_j for i in items), 4),
                "mean_energy_per_frame_mj": round(
                    sum(i.mean_energy_per_frame_mj for i in items) / n, 4
                ),
                "mean_center_distance_px": round(
                    sum(i.mean_center_distance_px for i in items) / n, 2
                ),
                "composite_score": round(
                    sum(self._score(i) for i in items) / n, 4
                ),
            }
        return agg

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def leaderboard(
        self,
        *,
        dataset: Optional[str] = None,
        sort_by: str = "composite",
        top_n: Optional[int] = None,
    ) -> str:
        """Return a Markdown leaderboard table."""
        rows = self.ranked_entries(dataset=dataset, sort_by=sort_by)
        if top_n is not None:
            rows = rows[:top_n]

        header_label = f"Dataset: {dataset}" if dataset else "All Datasets"
        lines = [
            f"## EOVOT Leaderboard — {header_label}",
            "",
            "| Rank | Tracker | Dataset | IoU | Success AUC | Precision AUC | FPS | Mem (MB) | Score |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in rows:
            lines.append(
                f"| {r['rank']} "
                f"| {r['tracker']} "
                f"| {r['dataset']} "
                f"| {r['mean_iou']:.3f} "
                f"| {r['success_auc']:.3f} "
                f"| {r['precision_auc']:.3f} "
                f"| {r['mean_fps']:.1f} "
                f"| {r['peak_memory_mb']:.1f} "
                f"| {r['composite_score']:.4f} |"
            )
        return "\n".join(lines)

    def cross_dataset_summary(
        self,
        *,
        sort_by: str = "composite_score",
    ) -> str:
        """Return a Markdown table aggregated by tracker across all loaded datasets."""
        agg = self.aggregate_by_tracker()
        rows = sorted(
            agg.values(),
            key=lambda r: r.get(sort_by, 0.0),
            reverse=True,
        )
        lines = [
            "## EOVOT Cross-Dataset Summary",
            "",
            "| Rank | Tracker | Datasets | Avg IoU | Avg Success AUC | Avg FPS | Avg Mem (MB) | Score |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for rank, r in enumerate(rows, start=1):
            lines.append(
                f"| {rank} "
                f"| {r['tracker']} "
                f"| {r['num_datasets']} "
                f"| {r['mean_iou']:.3f} "
                f"| {r['success_auc']:.3f} "
                f"| {r['mean_fps']:.1f} "
                f"| {r['peak_memory_mb']:.1f} "
                f"| {r['composite_score']:.4f} |"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #

    def to_csv(self, path: str | Path) -> None:
        """Write all entries to a CSV file."""
        rows = self.ranked_entries()
        if not rows:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def to_json(self, path: str | Path) -> None:
        """Write all entries to a JSON file."""
        rows = [e.to_dict() for e in self._entries]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(rows, fh, indent=2)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _score(self, s: TrackerSummary) -> float:
        return _composite_score(
            s,
            accuracy_weight=self._acc_w,
            speed_weight=self._spd_w,
            memory_weight=self._mem_w,
            fps_scale=self._fps_scale,
            mem_scale=self._mem_scale,
        )

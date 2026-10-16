"""Cross-dataset aggregation and weighted ranking for EOVOT benchmarks.

The core research question in EOVOT is not "which tracker performs best on
one dataset?" but **"which tracker performs best across the entire evaluation
suite?"**  A tracker that dominates GOT-10k but fails on LaSOT is not a
reliable edge deployment choice.

:class:`MultiDatasetAggregator` answers this question by:

1. **Collecting** per-tracker, per-dataset metric summaries from a set of
   :class:`~eovot.benchmark.engine.BenchmarkResult` objects.
2. **Normalizing** each metric within each dataset using z-score
   normalization — so absolute IoU values on short GOT-10k sequences
   (typically higher) are comparable to those on long LaSOT sequences
   (typically lower) without arithmetic distortion.
3. **Aggregating** normalized scores into a single **composite score** per
   tracker, optionally weighting datasets and metrics by research relevance.
4. **Exporting** the ranked table as Markdown, CSV, or JSON.

Normalization
~~~~~~~~~~~~~
For each (dataset, metric) pair, the normalized score of tracker *t* is::

    z_{t,d,m} = (raw_{t,d,m} − μ_{d,m}) / (σ_{d,m} + ε)

where μ and σ are computed across all trackers evaluated on dataset *d*,
and ε = 1e-8 prevents division by zero when all trackers have identical
scores.  Lower-is-better metrics (FPS rank inverted, latency, memory) are
negated before aggregation so that higher z-score always means *better*.

Composite Score
~~~~~~~~~~~~~~~
::

    score_t = Σ_d  w_d × Σ_m  w_m × z_{t,d,m}

where ``w_d`` is the dataset weight (default: uniform) and ``w_m`` is the
metric weight (default: uniform).  The composite score is mean-centred to
zero so positive values indicate above-average trackers and negative values
indicate below-average ones.

Typical usage::

    from eovot.analysis.multi_dataset import MultiDatasetAggregator

    aggregator = MultiDatasetAggregator(
        dataset_weights={"GOT-10k": 2.0, "LaSOT": 2.0, "Synthetic": 0.5},
        metric_weights={"mean_iou": 2.0, "mean_fps": 1.0, "peak_memory_mb": 0.5},
    )
    aggregator.add_results(got10k_results)   # list of BenchmarkResult
    aggregator.add_results(lasot_results)
    aggregator.add_results(synthetic_results)

    print(aggregator.to_markdown())
    aggregator.to_csv("multi_dataset_ranking.csv")
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Metrics to aggregate (key, display_name, higher_is_better)
# ---------------------------------------------------------------------------

#: Metrics extracted from BenchmarkResult.summary() for cross-dataset ranking.
#: Each entry: (metric_key, display_name, higher_is_better)
AGGREGATION_METRICS: List[Tuple[str, str, bool]] = [
    ("mean_iou",              "mIoU",        True),
    ("success_auc",           "Succ. AUC",   True),
    ("precision_auc",         "Prec. AUC",   True),
    ("mean_fps",              "FPS",         True),
    ("peak_memory_mb",        "Mem (MiB)",   False),
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AggregatedEntry:
    """Cross-dataset ranking entry for one tracker.

    Attributes:
        tracker_name: Human-readable tracker identifier.
        composite_score: Weighted, z-score-normalized aggregate across all
            datasets and metrics.  Higher is better.  Mean-centred to zero
            across all trackers in the comparison set.
        per_dataset: Mapping ``{dataset_name: {metric_key: raw_value}}``.
        per_dataset_score: Mapping ``{dataset_name: weighted_z_score_sum}``.
        num_datasets: Number of datasets this tracker was evaluated on.
        rank: Rank by composite_score (set by the aggregator).
    """

    tracker_name: str
    composite_score: float
    per_dataset: Dict[str, Dict[str, Optional[float]]]
    per_dataset_score: Dict[str, float]
    num_datasets: int
    rank: int = 0

    def __str__(self) -> str:
        return (
            f"AggregatedEntry({self.tracker_name}  "
            f"score={self.composite_score:+.4f}  "
            f"rank={self.rank}  datasets={self.num_datasets})"
        )


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class MultiDatasetAggregator:
    """Rank trackers by composite performance across multiple datasets.

    Args:
        metrics: Sequence of ``(key, display_name, higher_is_better)``
            triples specifying which metrics to include.  Defaults to
            :data:`AGGREGATION_METRICS`.
        dataset_weights: Mapping ``{dataset_name: weight}`` controlling
            the relative importance of each dataset.  Datasets not
            present default to weight ``1.0``.  Weights are
            *relative* (not required to sum to 1.0).
        metric_weights: Mapping ``{metric_key: weight}`` controlling the
            importance of individual metrics.  Metrics not present default
            to weight ``1.0``.
        min_datasets: Minimum number of datasets a tracker must appear on
            to be included in the final ranking.  Default: ``1``.

    Example::

        agg = MultiDatasetAggregator(
            dataset_weights={"GOT-10k": 2.0, "LaSOT": 2.0},
            metric_weights={"mean_iou": 3.0, "mean_fps": 1.0},
        )
        agg.add_results(got10k_results)
        agg.add_results(lasot_results)
        ranking = agg.rank()
        print(agg.to_markdown())
    """

    def __init__(
        self,
        metrics: Optional[Sequence[Tuple[str, str, bool]]] = None,
        dataset_weights: Optional[Dict[str, float]] = None,
        metric_weights: Optional[Dict[str, float]] = None,
        min_datasets: int = 1,
    ) -> None:
        self._metrics = list(metrics) if metrics is not None else list(AGGREGATION_METRICS)
        self._dataset_weights: Dict[str, float] = dataset_weights or {}
        self._metric_weights: Dict[str, float] = metric_weights or {}
        self._min_datasets = max(1, int(min_datasets))

        # raw_data[tracker_name][dataset_name][metric_key] = float_value
        self._raw: Dict[str, Dict[str, Dict[str, Optional[float]]]] = defaultdict(
            lambda: defaultdict(dict)
        )

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def add_result(self, result: Any) -> None:
        """Register one :class:`~eovot.benchmark.engine.BenchmarkResult`.

        Args:
            result: A :class:`~eovot.benchmark.engine.BenchmarkResult`
                object or its ``to_dict()`` output.

        Raises:
            TypeError: If the type is not recognized.
        """
        summary = self._extract_summary(result)
        tracker = summary.get("tracker") or summary.get("tracker_name", "unknown")
        dataset = summary.get("dataset") or summary.get("dataset_name", "unknown")
        for key, _name, _hib in self._metrics:
            val = summary.get(key)
            self._raw[tracker][dataset][key] = float(val) if val is not None else None

    def add_results(self, results: Sequence[Any]) -> None:
        """Register multiple results at once.

        Args:
            results: Sequence of :class:`~eovot.benchmark.engine.BenchmarkResult`
                objects (or their dicts).  Multiple trackers or datasets in one call
                are all accepted.
        """
        for r in results:
            self.add_result(r)

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank(self) -> List[AggregatedEntry]:
        """Compute the cross-dataset composite ranking.

        Returns:
            List of :class:`AggregatedEntry` sorted by composite score
            (highest first).  Trackers that appear on fewer datasets than
            ``min_datasets`` are excluded.

        Raises:
            ValueError: If no results have been added.
        """
        if not self._raw:
            raise ValueError("MultiDatasetAggregator: no results added. Call add_result() first.")

        # ----------------------------------------------------------------
        # Step 1: collect all datasets and per-dataset z-scores
        # ----------------------------------------------------------------
        all_datasets = sorted({ds for tr_data in self._raw.values() for ds in tr_data})
        all_trackers = sorted(self._raw.keys())

        # z_scores[tracker][dataset][metric] = z_score
        z_scores: Dict[str, Dict[str, Dict[str, float]]] = {
            t: {d: {} for d in all_datasets} for t in all_trackers
        }

        for dataset in all_datasets:
            for metric_key, _name, higher_is_better in self._metrics:
                # Collect raw values for this (dataset, metric) pair
                vals = {
                    t: self._raw[t][dataset].get(metric_key)
                    for t in all_trackers
                    if dataset in self._raw[t]
                }
                nums = [v for v in vals.values() if v is not None]
                if not nums:
                    continue

                mu = float(np.mean(nums))
                sigma = float(np.std(nums))
                eps = 1e-8

                for tracker, raw_val in vals.items():
                    if raw_val is None:
                        z_scores[tracker][dataset][metric_key] = 0.0
                        continue
                    z = (raw_val - mu) / (sigma + eps)
                    # Negate for lower-is-better metrics so higher z → better
                    if not higher_is_better:
                        z = -z
                    z_scores[tracker][dataset][metric_key] = float(z)

        # ----------------------------------------------------------------
        # Step 2: compute composite scores
        # ----------------------------------------------------------------
        entries: List[AggregatedEntry] = []

        for tracker in all_trackers:
            datasets_present = [d for d in all_datasets if d in self._raw[tracker]]
            if len(datasets_present) < self._min_datasets:
                continue

            per_ds_score: Dict[str, float] = {}
            total_weighted_score = 0.0
            total_weight = 0.0

            for dataset in datasets_present:
                dw = self._dataset_weights.get(dataset, 1.0)
                metric_sum = 0.0
                metric_weight_sum = 0.0

                for metric_key, _name, _hib in self._metrics:
                    if metric_key not in z_scores[tracker][dataset]:
                        continue
                    mw = self._metric_weights.get(metric_key, 1.0)
                    metric_sum += mw * z_scores[tracker][dataset][metric_key]
                    metric_weight_sum += mw

                ds_score = (metric_sum / metric_weight_sum) if metric_weight_sum > 0 else 0.0
                per_ds_score[dataset] = float(ds_score)
                total_weighted_score += dw * ds_score
                total_weight += dw

            composite = (total_weighted_score / total_weight) if total_weight > 0 else 0.0

            entries.append(AggregatedEntry(
                tracker_name=tracker,
                composite_score=composite,
                per_dataset={d: dict(self._raw[tracker].get(d, {})) for d in datasets_present},
                per_dataset_score=per_ds_score,
                num_datasets=len(datasets_present),
            ))

        # ----------------------------------------------------------------
        # Step 3: sort and assign ranks
        # ----------------------------------------------------------------
        entries.sort(key=lambda e: e.composite_score, reverse=True)
        for i, e in enumerate(entries, start=1):
            e.rank = i

        return entries

    # ------------------------------------------------------------------
    # Export methods
    # ------------------------------------------------------------------

    def to_markdown(
        self,
        title: Optional[str] = "Cross-Dataset Tracker Ranking",
        show_per_dataset_scores: bool = True,
    ) -> str:
        """Render the ranking as a GitHub-flavoured Markdown table.

        Args:
            title: Optional heading above the table.
            show_per_dataset_scores: Append a per-dataset score column for
                each dataset.  Useful for diagnosing dataset-specific strengths.

        Returns:
            Multi-line string ready to paste into a README or GitHub comment.
        """
        entries = self.rank()
        all_datasets = sorted({d for e in entries for d in e.per_dataset_score})
        mkeys = [k for k, _, _ in self._metrics]

        # Header
        header = ["Rank", "Tracker", "Score", "# DS"]
        if show_per_dataset_scores:
            header += [f"Score({d[:8]})" for d in all_datasets]
        for key, display, _ in self._metrics:
            if any(e.per_dataset.get(d, {}).get(key) is not None
                   for e in entries for d in e.per_dataset):
                header.append(display)

        sep = [":---:"] * len(header)
        sep[1] = ":---"

        lines = []
        if title:
            lines.append(f"## {title}")
            lines.append("")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(sep) + " |")

        best_score = entries[0].composite_score if entries else None

        for e in entries:
            cells = [
                str(e.rank),
                e.tracker_name,
                f"**{e.composite_score:+.4f}**" if e.rank == 1 else f"{e.composite_score:+.4f}",
                str(e.num_datasets),
            ]
            if show_per_dataset_scores:
                for d in all_datasets:
                    sc = e.per_dataset_score.get(d)
                    cells.append(f"{sc:+.3f}" if sc is not None else "—")
            # Best raw value per metric (mean across datasets)
            for key, _, hib in self._metrics:
                vals = [
                    v for d in e.per_dataset
                    for v in [e.per_dataset[d].get(key)]
                    if v is not None
                ]
                if not vals:
                    cells.append("—")
                else:
                    mean_val = float(np.mean(vals))
                    cells.append(f"{mean_val:.4f}" if mean_val < 1000 else f"{mean_val:.1f}")
            lines.append("| " + " | ".join(cells) + " |")

        lines.append("")
        lines.append(
            f"*Score = weighted z-score composite across {len(all_datasets)} dataset(s). "
            "Higher is better, mean-centred to 0.*"
        )
        return "\n".join(lines)

    def to_csv(self, path: Union[str, Path]) -> Path:
        """Write the ranking to a CSV file.

        Args:
            path: Destination file path.  ``.csv`` extension appended if absent.

        Returns:
            Resolved :class:`pathlib.Path` of the written file.
        """
        entries = self.rank()
        all_datasets = sorted({d for e in entries for d in e.per_dataset_score})

        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".csv")
        p.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["rank", "tracker", "composite_score", "num_datasets"]
        for d in all_datasets:
            fieldnames.append(f"score_{d}")
        for key, _, _ in self._metrics:
            fieldnames.append(f"mean_{key}")

        with open(p, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for e in entries:
                row: Dict[str, Any] = {
                    "rank": e.rank,
                    "tracker": e.tracker_name,
                    "composite_score": round(e.composite_score, 6),
                    "num_datasets": e.num_datasets,
                }
                for d in all_datasets:
                    sc = e.per_dataset_score.get(d)
                    row[f"score_{d}"] = round(sc, 6) if sc is not None else ""
                for key, _, _ in self._metrics:
                    vals = [
                        v for d in e.per_dataset
                        for v in [e.per_dataset[d].get(key)]
                        if v is not None
                    ]
                    row[f"mean_{key}"] = round(float(np.mean(vals)), 6) if vals else ""
                writer.writerow(row)

        return p

    def to_json(self, path: Union[str, Path]) -> Path:
        """Serialise the full ranking to a JSON file.

        Args:
            path: Destination file path.  ``.json`` extension appended if absent.

        Returns:
            Resolved :class:`pathlib.Path` of the written file.
        """
        entries = self.rank()

        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".json")
        p.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "config": {
                "metrics": [(k, n, h) for k, n, h in self._metrics],
                "dataset_weights": self._dataset_weights,
                "metric_weights": self._metric_weights,
                "min_datasets": self._min_datasets,
            },
            "ranking": [
                {
                    "rank": e.rank,
                    "tracker": e.tracker_name,
                    "composite_score": round(e.composite_score, 6),
                    "num_datasets": e.num_datasets,
                    "per_dataset_score": {
                        d: round(s, 6) for d, s in e.per_dataset_score.items()
                    },
                    "per_dataset": {
                        d: {k: (round(v, 6) if v is not None else None) for k, v in m.items()}
                        for d, m in e.per_dataset.items()
                    },
                }
                for e in entries
            ],
        }

        with open(p, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        return p

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def summary_dict(self) -> Dict[str, Any]:
        """Return the ranking as a plain dict (for embedding in larger reports).

        Returns:
            Dict with keys ``"num_trackers"``, ``"num_datasets"``, and
            ``"ranking"`` (list of per-tracker dicts).
        """
        entries = self.rank()
        all_datasets = sorted({d for e in entries for d in e.per_dataset_score})
        return {
            "num_trackers": len(entries),
            "num_datasets": len(all_datasets),
            "datasets": all_datasets,
            "ranking": [
                {
                    "rank": e.rank,
                    "tracker": e.tracker_name,
                    "composite_score": round(e.composite_score, 6),
                    "num_datasets": e.num_datasets,
                }
                for e in entries
            ],
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_summary(result: Any) -> Dict[str, Any]:
        if isinstance(result, dict):
            return result.get("summary", result)
        if hasattr(result, "summary"):
            return result.summary()
        raise TypeError(
            f"MultiDatasetAggregator: unsupported result type {type(result).__name__}. "
            "Pass a BenchmarkResult or its to_dict()/summary() dict."
        )

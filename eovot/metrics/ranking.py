"""Configurable multi-criteria tracker ranking for EOVOT.

Extends the fixed Edge Efficiency Score with a weighted composite ranking
that simultaneously incorporates accuracy, throughput, memory, energy, and
robustness metrics.  Preset weight configurations model common deployment
scenarios so researchers can reproduce apples-to-apples comparisons.

Composite Score
~~~~~~~~~~~~~~~
Each metric is independently min-max normalised across the tracker set so
that all dimensions live in ``[0, 1]``.  The composite score is the
weighted sum of normalised values::

    score = w_accuracy    * norm(success_auc)
          + w_fps         * norm(fps)
          + w_memory      * (1 - norm(peak_memory_mb))   # lower is better
          + w_energy      * (1 - norm(energy_mj_frame))  # lower is better
          + w_robustness  * norm(mean_iou)

Weights must be non-negative but need not sum to 1 — they are re-normalised
internally, so only the *relative* magnitudes matter.

Preset configurations
~~~~~~~~~~~~~~~~~~~~~
Four named presets are provided as class methods on :class:`RankingWeights`
so researchers can report results under standardised scenarios:

- ``accuracy_first()`` — maximise tracking quality, ignore efficiency
- ``edge_balanced()``  — equal weight to accuracy, speed, and memory
- ``battery_saver()``  — heavy penalty on energy and memory
- ``throughput_max()`` — optimise for raw FPS on constrained devices

Example::

    from eovot.metrics.ranking import RankingEngine, RankingWeights

    weights = RankingWeights.edge_balanced()
    engine  = RankingEngine(weights)
    ranking = engine.rank(benchmark_results)

    print(engine.to_markdown_table(ranking))
    # Or as JSON-serialisable dicts:
    print(engine.to_summary_dict(ranking))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult


# ---------------------------------------------------------------------------
# Weight configuration
# ---------------------------------------------------------------------------

@dataclass
class RankingWeights:
    """Relative importance weights for the five ranking dimensions.

    All weights must be non-negative.  They are normalised internally so only
    their relative magnitudes matter — ``(0.5, 0.5)`` is identical to
    ``(1.0, 1.0)`` for a two-metric run.

    Attributes:
        accuracy:   Weight for tracking accuracy (success AUC, falling back to
                    mean IoU when AUC is unavailable).
        fps:        Weight for throughput (frames per second). Higher is better.
        memory:     Weight for memory efficiency. Lower peak memory is better.
        energy:     Weight for energy efficiency. Lower mJ/frame is better.
        robustness: Weight for mean IoU as a robustness proxy when both
                    success AUC *and* mean IoU are available together, this
                    captures a slightly different quality dimension.
        label:      Human-readable name for the weight configuration, used
                    in reports and table headings.
    """

    accuracy: float = 1.0
    fps: float = 1.0
    memory: float = 1.0
    energy: float = 0.0
    robustness: float = 0.0
    label: str = "custom"

    def __post_init__(self) -> None:
        for name, val in self._weight_values().items():
            if val < 0:
                raise ValueError(f"Weight '{name}' must be non-negative, got {val}.")

    def _weight_values(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "fps": self.fps,
            "memory": self.memory,
            "energy": self.energy,
            "robustness": self.robustness,
        }

    def total(self) -> float:
        return sum(self._weight_values().values())

    # ------------------------------------------------------------------
    # Named presets
    # ------------------------------------------------------------------

    @classmethod
    def accuracy_first(cls) -> "RankingWeights":
        """Maximise tracking quality; treat efficiency as secondary."""
        return cls(accuracy=5.0, fps=0.5, memory=0.5, energy=0.0, robustness=2.0,
                   label="accuracy_first")

    @classmethod
    def edge_balanced(cls) -> "RankingWeights":
        """Equal weight to accuracy, speed, and memory — typical edge scenario."""
        return cls(accuracy=1.0, fps=1.0, memory=1.0, energy=0.5, robustness=0.5,
                   label="edge_balanced")

    @classmethod
    def battery_saver(cls) -> "RankingWeights":
        """Penalise energy and memory; optimise for battery-constrained IoT."""
        return cls(accuracy=1.0, fps=0.5, memory=2.0, energy=3.0, robustness=0.0,
                   label="battery_saver")

    @classmethod
    def throughput_max(cls) -> "RankingWeights":
        """Optimise raw FPS for hard real-time edge applications."""
        return cls(accuracy=0.5, fps=4.0, memory=1.0, energy=0.5, robustness=0.0,
                   label="throughput_max")


# ---------------------------------------------------------------------------
# Per-tracker ranking result
# ---------------------------------------------------------------------------

@dataclass
class TrackerRank:
    """Ranking result for one tracker.

    Attributes:
        rank:              1-based position in the ranked list (1 = best).
        tracker_name:      Tracker identifier.
        dataset_name:      Dataset on which the tracker was evaluated.
        composite_score:   Weighted composite score in ``[0, 1]``.
        norm_accuracy:     Normalised accuracy score.
        norm_fps:          Normalised FPS score (higher is better).
        norm_memory:       Normalised memory score (higher = less memory used).
        norm_energy:       Normalised energy score (higher = less energy used).
        norm_robustness:   Normalised robustness score.
        raw_accuracy:      Raw success AUC (or mean IoU if AUC unavailable).
        raw_fps:           Raw mean FPS.
        raw_memory_mb:     Raw peak memory in MB.
        raw_energy_mj:     Raw energy per frame in mJ, or ``None``.
        raw_mean_iou:      Raw mean IoU.
        weights_label:     Label of the :class:`RankingWeights` preset used.
    """

    rank: int
    tracker_name: str
    dataset_name: str
    composite_score: float
    norm_accuracy: float
    norm_fps: float
    norm_memory: float
    norm_energy: float
    norm_robustness: float
    raw_accuracy: float
    raw_fps: float
    raw_memory_mb: float
    raw_energy_mj: Optional[float]
    raw_mean_iou: float
    weights_label: str = "custom"

    def __str__(self) -> str:
        return (
            f"[#{self.rank}] {self.tracker_name:<18s}  "
            f"score={self.composite_score:.4f}  "
            f"acc={self.raw_accuracy:.4f}  "
            f"fps={self.raw_fps:.1f}  "
            f"mem={self.raw_memory_mb:.1f} MB"
        )

    def to_dict(self) -> dict:
        d = {
            "rank": self.rank,
            "tracker": self.tracker_name,
            "dataset": self.dataset_name,
            "composite_score": round(self.composite_score, 6),
            "norm_accuracy": round(self.norm_accuracy, 4),
            "norm_fps": round(self.norm_fps, 4),
            "norm_memory": round(self.norm_memory, 4),
            "norm_energy": round(self.norm_energy, 4),
            "norm_robustness": round(self.norm_robustness, 4),
            "raw_accuracy": round(self.raw_accuracy, 4),
            "raw_fps": round(self.raw_fps, 2),
            "raw_memory_mb": round(self.raw_memory_mb, 2),
            "raw_mean_iou": round(self.raw_mean_iou, 4),
            "weights_label": self.weights_label,
        }
        if self.raw_energy_mj is not None:
            d["raw_energy_mj"] = round(self.raw_energy_mj, 4)
        return d


# ---------------------------------------------------------------------------
# Ranking engine
# ---------------------------------------------------------------------------

class RankingEngine:
    """Rank trackers across multiple evaluation dimensions with configurable weights.

    Args:
        weights: :class:`RankingWeights` controlling the importance of each
                 dimension. Defaults to the ``edge_balanced`` preset.

    Example::

        from eovot.metrics.ranking import RankingEngine, RankingWeights

        engine = RankingEngine(RankingWeights.battery_saver())
        ranking = engine.rank(benchmark_results)
        print(engine.to_markdown_table(ranking))
    """

    def __init__(self, weights: Optional[RankingWeights] = None) -> None:
        self.weights = weights if weights is not None else RankingWeights.edge_balanced()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(self, results: List["BenchmarkResult"]) -> List[TrackerRank]:
        """Rank a collection of :class:`~eovot.benchmark.engine.BenchmarkResult` objects.

        Args:
            results: One result per tracker / dataset combination.  At minimum
                each result must contain at least one :class:`~eovot.benchmark.engine.SequenceResult`.

        Returns:
            List of :class:`TrackerRank`, sorted by composite score (highest first),
            with ``rank`` fields assigned starting from 1.

        Raises:
            ValueError: If *results* is empty.
        """
        if not results:
            raise ValueError("Cannot rank an empty list of BenchmarkResults.")

        raw = self._extract_raw(results)
        norms = self._normalise(raw)
        scores = self._score(norms)

        ranked: List[TrackerRank] = []
        for i, result in enumerate(results):
            ranked.append(TrackerRank(
                rank=0,  # assigned below
                tracker_name=result.tracker_name,
                dataset_name=result.dataset_name,
                composite_score=scores[i],
                norm_accuracy=norms["accuracy"][i],
                norm_fps=norms["fps"][i],
                norm_memory=norms["memory"][i],
                norm_energy=norms["energy"][i],
                norm_robustness=norms["robustness"][i],
                raw_accuracy=raw["accuracy"][i],
                raw_fps=raw["fps"][i],
                raw_memory_mb=raw["memory"][i],
                raw_energy_mj=raw["energy"][i] if raw["energy"][i] >= 0 else None,
                raw_mean_iou=raw["iou"][i],
                weights_label=self.weights.label,
            ))

        ranked.sort(key=lambda r: r.composite_score, reverse=True)
        for pos, entry in enumerate(ranked, start=1):
            entry.rank = pos
        return ranked

    def to_markdown_table(self, ranking: List[TrackerRank]) -> str:
        """Format the ranking as a Markdown table.

        Args:
            ranking: Output of :meth:`rank`.

        Returns:
            Multi-line Markdown string ready for README or paper appendix.
        """
        label = ranking[0].weights_label if ranking else "custom"
        header = (
            f"**Tracker Ranking — weights: `{label}`**\n\n"
            "| Rank | Tracker | Dataset | Score | Acc | FPS | Mem (MB) | Energy (mJ) |\n"
            "|------|---------|---------|------:|----:|----:|---------:|------------:|"
        )
        rows = []
        for r in ranking:
            energy_str = f"{r.raw_energy_mj:.3f}" if r.raw_energy_mj is not None else "—"
            rows.append(
                f"| {r.rank} | {r.tracker_name} | {r.dataset_name} "
                f"| {r.composite_score:.4f} "
                f"| {r.raw_accuracy:.4f} "
                f"| {r.raw_fps:.1f} "
                f"| {r.raw_memory_mb:.1f} "
                f"| {energy_str} |"
            )
        return header + "\n" + "\n".join(rows)

    def to_summary_dict(self, ranking: List[TrackerRank]) -> List[dict]:
        """Return ranking as a list of plain dicts suitable for JSON serialisation."""
        return [r.to_dict() for r in ranking]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_raw(self, results: List["BenchmarkResult"]) -> dict:
        """Pull raw metric values from each BenchmarkResult."""
        accuracy, fps, memory, energy, iou = [], [], [], [], []
        for r in results:
            # Prefer success AUC as the primary accuracy signal.
            acc = r.mean_success_auc if r.mean_success_auc is not None else r.mean_iou
            accuracy.append(acc)
            fps.append(r.mean_fps)
            memory.append(r.peak_memory_mb)
            iou.append(r.mean_iou)
            epf = r.mean_energy_per_frame_mj
            energy.append(epf if epf is not None else -1.0)  # sentinel for missing
        return {"accuracy": accuracy, "fps": fps, "memory": memory,
                "energy": energy, "iou": iou}

    @staticmethod
    def _minmax(values: list, invert: bool = False) -> list:
        """Min-max normalise a list to [0, 1]; optionally invert (lower=better)."""
        arr = np.array(values, dtype=float)
        # Replace sentinels (-1) with NaN so they don't distort the range.
        arr_clean = np.where(arr < 0, np.nan, arr)
        valid = arr_clean[~np.isnan(arr_clean)]
        if valid.size == 0:
            # All values are sentinels (missing); assign neutral score.
            return [0.5] * len(values)
        lo, hi = float(valid.min()), float(valid.max())
        if hi == lo:
            normed = np.where(np.isnan(arr_clean), 0.5, 0.5)
        else:
            normed = (arr_clean - lo) / (hi - lo)
            normed = np.where(np.isnan(normed), 0.5, normed)
        if invert:
            normed = 1.0 - normed
        return normed.tolist()

    def _normalise(self, raw: dict) -> dict:
        return {
            "accuracy":   self._minmax(raw["accuracy"]),
            "fps":        self._minmax(raw["fps"]),
            "memory":     self._minmax(raw["memory"], invert=True),   # less = better
            "energy":     self._minmax(raw["energy"], invert=True),   # less = better
            "robustness": self._minmax(raw["iou"]),
        }

    def _score(self, norms: dict) -> list:
        w = self.weights
        total_w = w.total()
        if total_w == 0:
            raise ValueError("All weights are zero — cannot compute a composite score.")
        scores = []
        n = len(norms["accuracy"])
        for i in range(n):
            s = (
                w.accuracy   * norms["accuracy"][i]
                + w.fps      * norms["fps"][i]
                + w.memory   * norms["memory"][i]
                + w.energy   * norms["energy"][i]
                + w.robustness * norms["robustness"][i]
            ) / total_w
            scores.append(s)
        return scores

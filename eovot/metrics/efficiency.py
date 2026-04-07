"""Edge Efficiency Score — hardware-aware composite metric for tracker ranking.

In edge deployment, raw accuracy (IoU) is only part of the story.  A tracker
that runs at 5 FPS and consumes 800 MB of memory is not deployable on a
Raspberry Pi 4 or Jetson Nano, regardless of its VOT score.

This module defines an **EdgeEfficiencyScore**: a single scalar that combines
accuracy, throughput, and memory footprint into a deployability index.  It
allows trackers to be ranked not only by how well they track, but by how
efficiently they do so on constrained hardware.

Score formula
-------------
The score is a weighted harmonic mean in the spirit of the F1-score, but
adapted for three orthogonal dimensions:

    accuracy_score  = mean_iou                            ∈ [0, 1]
    speed_score     = min(fps / fps_ref, 1.0)             ∈ [0, 1]
    memory_score    = 1 - min(memory_mb / memory_ref, 1)  ∈ [0, 1]

    edge_score = w_a * accuracy + w_s * speed + w_m * memory
                 ─────────────────────────────────────────────
                            w_a + w_s + w_m

Default reference values are calibrated for a Raspberry Pi 4 / Jetson Nano
class device (30 FPS target, 512 MB memory budget).

Usage
-----
::

    from eovot.metrics.efficiency import EdgeEfficiencyScorer, EfficiencyResult

    scorer = EdgeEfficiencyScorer(fps_ref=30.0, memory_ref_mb=512.0)
    result = scorer.score(mean_iou=0.65, fps=45.0, memory_mb=120.0)
    print(result)
    # EfficiencyResult[MOSSE] score=0.820 (acc=0.650 spd=1.000 mem=0.766)

    # Rank multiple trackers
    ranking = scorer.rank(results_dict)
    for r in ranking:
        print(r)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class EfficiencyResult:
    """Efficiency decomposition for a single tracker configuration.

    Attributes:
        tracker_name: Tracker identifier.
        edge_score: Composite deployability score in ``[0, 1]``.
        accuracy_score: Raw mean IoU normalised to ``[0, 1]``.
        speed_score: Throughput score in ``[0, 1]``;
            ``1.0`` means the tracker meets or exceeds the FPS target.
        memory_score: Memory efficiency score in ``[0, 1]``;
            ``1.0`` means zero memory use; ``0.0`` means at or above the budget.
        mean_iou: Raw mean IoU value.
        fps: Raw throughput (frames per second).
        memory_mb: Raw peak memory usage (MiB).
    """

    tracker_name: str
    edge_score: float
    accuracy_score: float
    speed_score: float
    memory_score: float
    mean_iou: float
    fps: float
    memory_mb: float

    def to_dict(self) -> Dict:
        """Serialise to a plain dict suitable for JSON export or DataFrame construction."""
        return {
            "tracker": self.tracker_name,
            "edge_score": round(self.edge_score, 4),
            "accuracy_score": round(self.accuracy_score, 4),
            "speed_score": round(self.speed_score, 4),
            "memory_score": round(self.memory_score, 4),
            "mean_iou": round(self.mean_iou, 4),
            "fps": round(self.fps, 2),
            "memory_mb": round(self.memory_mb, 2),
        }

    def __str__(self) -> str:
        return (
            f"EfficiencyResult[{self.tracker_name}] "
            f"score={self.edge_score:.3f} "
            f"(acc={self.accuracy_score:.3f} "
            f"spd={self.speed_score:.3f} "
            f"mem={self.memory_score:.3f})"
        )


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class EdgeEfficiencyScorer:
    """Compute hardware-aware Edge Efficiency Scores for trackers.

    The score combines accuracy, throughput, and memory efficiency into a
    single deployability index that is meaningful for resource-constrained
    edge devices.

    Args:
        fps_ref: Target FPS for the deployment device.  A tracker achieving
            this FPS (or more) scores 1.0 on speed.  Default: ``30.0``
            (suitable for Raspberry Pi 4 / Jetson Nano real-time threshold).
        memory_ref_mb: Memory budget in MiB.  A tracker using this much
            memory scores 0.0 on memory efficiency.  Default: ``512.0``.
        weight_accuracy: Weight of the accuracy component.  Default: ``0.5``.
        weight_speed: Weight of the throughput component.  Default: ``0.3``.
        weight_memory: Weight of the memory-efficiency component.
            Default: ``0.2``.

    The weights must be positive; they are normalised internally so they do
    not need to sum to 1.

    Example::

        scorer = EdgeEfficiencyScorer(fps_ref=30.0, memory_ref_mb=512.0)
        result = scorer.score("MOSSE", mean_iou=0.55, fps=350.0, memory_mb=45.0)
        print(result)
    """

    def __init__(
        self,
        fps_ref: float = 30.0,
        memory_ref_mb: float = 512.0,
        weight_accuracy: float = 0.5,
        weight_speed: float = 0.3,
        weight_memory: float = 0.2,
    ) -> None:
        if fps_ref <= 0:
            raise ValueError(f"fps_ref must be positive, got {fps_ref}")
        if memory_ref_mb <= 0:
            raise ValueError(f"memory_ref_mb must be positive, got {memory_ref_mb}")
        if any(w <= 0 for w in (weight_accuracy, weight_speed, weight_memory)):
            raise ValueError("All weights must be positive.")

        self.fps_ref = fps_ref
        self.memory_ref_mb = memory_ref_mb

        total = weight_accuracy + weight_speed + weight_memory
        self._w_acc = weight_accuracy / total
        self._w_spd = weight_speed / total
        self._w_mem = weight_memory / total

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _speed_score(self, fps: float) -> float:
        """Normalise FPS to [0, 1] relative to the reference target."""
        return float(min(fps / self.fps_ref, 1.0))

    def _memory_score(self, memory_mb: float) -> float:
        """Compute memory-efficiency score: lower usage → higher score."""
        return float(max(0.0, 1.0 - memory_mb / self.memory_ref_mb))

    def score(
        self,
        tracker_name: str,
        mean_iou: float,
        fps: float,
        memory_mb: float,
    ) -> EfficiencyResult:
        """Compute the Edge Efficiency Score for one tracker configuration.

        Args:
            tracker_name: Human-readable tracker identifier.
            mean_iou: Mean Intersection-over-Union across evaluation sequences.
            fps: Measured throughput in frames per second.
            memory_mb: Peak memory usage in MiB.

        Returns:
            :class:`EfficiencyResult` with the composite score and its
            decomposition into accuracy, speed, and memory components.

        Raises:
            ValueError: If any input is outside a plausible range.
        """
        if not (0.0 <= mean_iou <= 1.0):
            raise ValueError(f"mean_iou must be in [0, 1], got {mean_iou}")
        if fps < 0:
            raise ValueError(f"fps must be non-negative, got {fps}")
        if memory_mb < 0:
            raise ValueError(f"memory_mb must be non-negative, got {memory_mb}")

        acc = float(np.clip(mean_iou, 0.0, 1.0))
        spd = self._speed_score(fps)
        mem = self._memory_score(memory_mb)
        edge = self._w_acc * acc + self._w_spd * spd + self._w_mem * mem

        return EfficiencyResult(
            tracker_name=tracker_name,
            edge_score=round(edge, 6),
            accuracy_score=round(acc, 6),
            speed_score=round(spd, 6),
            memory_score=round(mem, 6),
            mean_iou=round(mean_iou, 6),
            fps=round(fps, 4),
            memory_mb=round(memory_mb, 4),
        )

    def rank(
        self,
        tracker_metrics: Dict[str, Dict],
    ) -> List[EfficiencyResult]:
        """Rank a set of trackers by Edge Efficiency Score (descending).

        Args:
            tracker_metrics: Mapping of tracker name →
                ``{"mean_iou": float, "fps": float, "memory_mb": float}``.

        Returns:
            List of :class:`EfficiencyResult` objects sorted by
            ``edge_score`` (highest first).

        Raises:
            ValueError: If any required key is missing from a tracker entry.
        """
        required = {"mean_iou", "fps", "memory_mb"}
        results: List[EfficiencyResult] = []
        for name, metrics in tracker_metrics.items():
            missing = required - set(metrics.keys())
            if missing:
                raise ValueError(
                    f"Tracker '{name}' is missing required keys: {missing}"
                )
            results.append(
                self.score(
                    tracker_name=name,
                    mean_iou=metrics["mean_iou"],
                    fps=metrics["fps"],
                    memory_mb=metrics["memory_mb"],
                )
            )
        results.sort(key=lambda r: r.edge_score, reverse=True)
        return results

    def summary_table(
        self,
        tracker_metrics: Dict[str, Dict],
        title: str = "Edge Efficiency Ranking",
    ) -> str:
        """Build a Markdown-formatted ranking table.

        Args:
            tracker_metrics: Same format as :meth:`rank`.
            title: Table heading. Default: ``"Edge Efficiency Ranking"``.

        Returns:
            Multi-line Markdown string ready to paste into a README or report.
        """
        ranked = self.rank(tracker_metrics)
        lines = [
            f"## {title}\n",
            f"*Device profile: {self.fps_ref:.0f} FPS target, "
            f"{self.memory_ref_mb:.0f} MB memory budget*\n",
            "| Rank | Tracker | Edge Score | mIoU | FPS | Memory (MB) | "
            "Acc | Spd | Mem |",
            "|-----:|---------|:----------:|-----:|----:|------------:|"
            "----:|----:|----:|",
        ]
        for i, r in enumerate(ranked, start=1):
            lines.append(
                f"| {i} | {r.tracker_name} | **{r.edge_score:.4f}** "
                f"| {r.mean_iou:.4f} | {r.fps:.1f} | {r.memory_mb:.1f} "
                f"| {r.accuracy_score:.3f} | {r.speed_score:.3f} | {r.memory_score:.3f} |"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience: score from a BenchmarkResult dict
# ---------------------------------------------------------------------------


def score_from_summary(
    summary: Dict,
    scorer: Optional[EdgeEfficiencyScorer] = None,
) -> EfficiencyResult:
    """Compute an :class:`EfficiencyResult` directly from a benchmark summary dict.

    This is a convenience wrapper that accepts the ``summary`` sub-dict
    produced by :meth:`~eovot.benchmark.engine.BenchmarkResult.summary`.

    Args:
        summary: Dict with at minimum keys ``"tracker"``, ``"mean_iou"``,
            ``"mean_fps"``, and ``"peak_memory_mb"``.
        scorer: Optional pre-configured scorer.  A default scorer is used
            if not provided.

    Returns:
        :class:`EfficiencyResult` for the given tracker.
    """
    if scorer is None:
        scorer = EdgeEfficiencyScorer()
    return scorer.score(
        tracker_name=summary.get("tracker", "unknown"),
        mean_iou=float(summary.get("mean_iou", 0.0)),
        fps=float(summary.get("mean_fps", 0.0)),
        memory_mb=float(summary.get("peak_memory_mb", 0.0)),
    )

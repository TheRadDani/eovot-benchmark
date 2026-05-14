"""Edge deployment scoring for EOVOT.

Computes a composite *Edge Deployment Score* (EDS) that combines tracking
accuracy, throughput, and energy efficiency into a single deployability
number suited for comparing trackers on resource-constrained devices.

Motivation
----------
Academic benchmarks rank trackers solely by accuracy (mIoU / AUC).
On edge devices (Raspberry Pi, Jetson Nano, MCUs) a tracker that achieves
high accuracy but runs at 2 FPS or drains the battery quickly is useless
in practice.  EDS makes the accuracy–efficiency trade-off explicit and
tunable via weight parameters.

Score formula
-------------
EDS is a weighted geometric mean of three normalised sub-scores, each
in ``[0, 1]``:

.. code-block:: text

    throughput_score = min(fps / target_fps, 1.0)
    efficiency_score = 1 / (1 + energy_per_frame_mj / reference_energy_mj)
    accuracy_score   = mean_iou                          (already in [0,1])

    EDS = accuracy_score^w_acc * throughput_score^w_fps * efficiency_score^w_eff

A geometric mean is used so that a tracker that completely fails on any
single dimension scores near zero overall (no compensation between
dimensions).  When energy data is unavailable, the efficiency score is
excluded and the remaining two scores are re-weighted proportionally.

Default weights target a balanced edge deployment scenario:
- ``w_acc = 0.4`` — accuracy matters most
- ``w_fps = 0.4`` — real-time throughput matters equally
- ``w_eff = 0.2`` — energy efficiency is a secondary concern

Usage::

    from eovot.metrics.edge_score import EdgeScoreCalculator

    calc = EdgeScoreCalculator(target_fps=30.0, reference_energy_mj=5.0)

    score = calc.compute(
        mean_iou=0.62,
        fps=45.0,
        energy_per_frame_mj=3.2,   # omit or pass None to skip energy
    )
    print(score)
    # EdgeDeploymentScore(EDS=0.694, accuracy=0.620, throughput=1.000, efficiency=0.610)

    # Rank multiple trackers
    tracker_summaries = [
        {"tracker": "MOSSE", "mean_iou": 0.55, "mean_fps": 312, "mean_energy_per_frame_mj": None},
        {"tracker": "KCF",   "mean_iou": 0.61, "mean_fps": 120, "mean_energy_per_frame_mj": 4.1},
        {"tracker": "CSRT",  "mean_iou": 0.70, "mean_fps": 25,  "mean_energy_per_frame_mj": 8.5},
    ]
    ranked = calc.rank(tracker_summaries)
    for r in ranked:
        print(r)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class EdgeDeploymentScore:
    """Composite edge deployment score for a single tracker.

    Attributes:
        tracker_name:      Identifier of the tracker.
        eds:               Overall Edge Deployment Score in ``[0, 1]``.
        accuracy_score:    Normalised accuracy sub-score (= mean_iou).
        throughput_score:  Normalised throughput sub-score in ``[0, 1]``.
        efficiency_score:  Normalised energy efficiency sub-score in ``[0, 1]``,
                           or ``None`` if energy data was unavailable.
        mean_iou:          Raw mean IoU value used.
        fps:               Raw FPS value used.
        energy_per_frame_mj: Raw energy value used, or ``None``.
        target_fps:        The real-time FPS target used in the calculation.
    """

    tracker_name: str
    eds: float
    accuracy_score: float
    throughput_score: float
    efficiency_score: Optional[float]
    mean_iou: float
    fps: float
    energy_per_frame_mj: Optional[float]
    target_fps: float

    def __str__(self) -> str:
        eff = f"{self.efficiency_score:.3f}" if self.efficiency_score is not None else "N/A"
        return (
            f"EdgeDeploymentScore[{self.tracker_name}] "
            f"EDS={self.eds:.4f}  "
            f"accuracy={self.accuracy_score:.3f}  "
            f"throughput={self.throughput_score:.3f}  "
            f"efficiency={eff}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tracker_name": self.tracker_name,
            "eds": round(self.eds, 4),
            "accuracy_score": round(self.accuracy_score, 4),
            "throughput_score": round(self.throughput_score, 4),
            "efficiency_score": (
                round(self.efficiency_score, 4)
                if self.efficiency_score is not None else None
            ),
            "mean_iou": round(self.mean_iou, 4),
            "fps": round(self.fps, 2),
            "energy_per_frame_mj": (
                round(self.energy_per_frame_mj, 4)
                if self.energy_per_frame_mj is not None else None
            ),
            "target_fps": self.target_fps,
        }


class EdgeScoreCalculator:
    """Compute and rank Edge Deployment Scores for one or more trackers.

    Args:
        target_fps:          FPS threshold for full throughput credit.
                             A tracker at or above this speed gets
                             ``throughput_score = 1.0``.  Default: ``30.0``.
        reference_energy_mj: Energy reference point (mJ/frame).  A tracker
                             consuming this much energy gets
                             ``efficiency_score ≈ 0.5``.  Default: ``10.0``.
        w_acc:               Weight for the accuracy sub-score.  Default: ``0.4``.
        w_fps:               Weight for the throughput sub-score.  Default: ``0.4``.
        w_eff:               Weight for the energy efficiency sub-score.
                             Automatically re-weighted when energy data is
                             absent.  Default: ``0.2``.

    Example::

        calc = EdgeScoreCalculator(target_fps=25.0, reference_energy_mj=8.0)
        score = calc.compute("MOSSE", mean_iou=0.55, fps=312.0)
        print(score.eds)
    """

    def __init__(
        self,
        target_fps: float = 30.0,
        reference_energy_mj: float = 10.0,
        w_acc: float = 0.4,
        w_fps: float = 0.4,
        w_eff: float = 0.2,
    ) -> None:
        if target_fps <= 0:
            raise ValueError("target_fps must be positive")
        if reference_energy_mj <= 0:
            raise ValueError("reference_energy_mj must be positive")
        if not (w_acc > 0 and w_fps > 0 and w_eff >= 0):
            raise ValueError("weights must be positive (w_eff may be 0)")

        self.target_fps = target_fps
        self.reference_energy_mj = reference_energy_mj
        self.w_acc = w_acc
        self.w_fps = w_fps
        self.w_eff = w_eff

    def compute(
        self,
        tracker_name: str = "unknown",
        *,
        mean_iou: float,
        fps: float,
        energy_per_frame_mj: Optional[float] = None,
    ) -> EdgeDeploymentScore:
        """Compute the Edge Deployment Score for a single tracker result.

        Args:
            tracker_name:        Human-readable tracker identifier.
            mean_iou:            Mean IoU (accuracy), in ``[0, 1]``.
            fps:                 Mean throughput in frames per second.
            energy_per_frame_mj: Mean energy per frame (mJ).  Pass ``None``
                                 to exclude energy from the score.

        Returns:
            :class:`EdgeDeploymentScore` populated with all sub-scores and EDS.
        """
        accuracy_score = float(max(0.0, min(1.0, mean_iou)))
        throughput_score = float(min(fps / self.target_fps, 1.0)) if fps >= 0 else 0.0

        if energy_per_frame_mj is not None and energy_per_frame_mj >= 0:
            efficiency_score: Optional[float] = float(
                1.0 / (1.0 + energy_per_frame_mj / self.reference_energy_mj)
            )
        else:
            efficiency_score = None

        eds = self._geometric_mean(accuracy_score, throughput_score, efficiency_score)

        return EdgeDeploymentScore(
            tracker_name=tracker_name,
            eds=eds,
            accuracy_score=accuracy_score,
            throughput_score=throughput_score,
            efficiency_score=efficiency_score,
            mean_iou=mean_iou,
            fps=fps,
            energy_per_frame_mj=energy_per_frame_mj,
            target_fps=self.target_fps,
        )

    def from_benchmark_summary(
        self, summary: Dict[str, Any]
    ) -> EdgeDeploymentScore:
        """Convenience wrapper: compute EDS directly from a benchmark summary dict.

        The summary dict is the output of
        :meth:`~eovot.benchmark.engine.BenchmarkResult.summary` and
        contains keys ``"tracker"``, ``"mean_iou"``, ``"mean_fps"``,
        and optionally ``"mean_energy_per_frame_mj"``.

        Args:
            summary: Dict from :meth:`~eovot.benchmark.engine.BenchmarkResult.summary`.

        Returns:
            :class:`EdgeDeploymentScore`.
        """
        return self.compute(
            tracker_name=summary.get("tracker", "unknown"),
            mean_iou=float(summary.get("mean_iou", 0.0)),
            fps=float(summary.get("mean_fps", 0.0)),
            energy_per_frame_mj=summary.get("mean_energy_per_frame_mj"),
        )

    def rank(
        self, summaries: List[Dict[str, Any]]
    ) -> List[EdgeDeploymentScore]:
        """Compute and rank EDS for a list of benchmark summary dicts.

        Args:
            summaries: List of dicts from
                :meth:`~eovot.benchmark.engine.BenchmarkResult.summary`,
                one per tracker.

        Returns:
            List of :class:`EdgeDeploymentScore` objects sorted by EDS
            (highest first).
        """
        scores = [self.from_benchmark_summary(s) for s in summaries]
        scores.sort(key=lambda s: s.eds, reverse=True)
        return scores

    def leaderboard_markdown(
        self,
        scores: List[EdgeDeploymentScore],
        title: str = "Edge Deployment Leaderboard",
    ) -> str:
        """Render a ranked Markdown table from a list of EDS scores.

        Args:
            scores: Pre-sorted list of :class:`EdgeDeploymentScore` objects.
            title:  Table heading.

        Returns:
            Multi-line Markdown string.
        """
        lines = [
            f"# {title}\n",
            f"Target FPS: **{self.target_fps}**  |  "
            f"Reference energy: **{self.reference_energy_mj} mJ/frame**\n",
            "| Rank | Tracker | EDS | mIoU | FPS | Throughput | Efficiency |",
            "|------|---------|----:|-----:|----:|-----------:|-----------:|",
        ]
        for rank, s in enumerate(scores, start=1):
            eff = f"{s.efficiency_score:.3f}" if s.efficiency_score is not None else "—"
            lines.append(
                f"| {rank} | {s.tracker_name} | {s.eds:.4f} "
                f"| {s.mean_iou:.4f} | {s.fps:.1f} "
                f"| {s.throughput_score:.3f} | {eff} |"
            )
        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _geometric_mean(
        self,
        accuracy: float,
        throughput: float,
        efficiency: Optional[float],
    ) -> float:
        """Weighted geometric mean of two or three sub-scores.

        When efficiency is None the energy weight is redistributed
        proportionally to accuracy and throughput.
        """
        import math

        if efficiency is not None:
            total_w = self.w_acc + self.w_fps + self.w_eff
            wa = self.w_acc / total_w
            wf = self.w_fps / total_w
            we = self.w_eff / total_w
            # Guard against log(0)
            acc_c = max(accuracy, 1e-9)
            fps_c = max(throughput, 1e-9)
            eff_c = max(efficiency, 1e-9)
            log_eds = wa * math.log(acc_c) + wf * math.log(fps_c) + we * math.log(eff_c)
        else:
            total_w = self.w_acc + self.w_fps
            wa = self.w_acc / total_w
            wf = self.w_fps / total_w
            acc_c = max(accuracy, 1e-9)
            fps_c = max(throughput, 1e-9)
            log_eds = wa * math.log(acc_c) + wf * math.log(fps_c)

        return float(math.exp(log_eds))

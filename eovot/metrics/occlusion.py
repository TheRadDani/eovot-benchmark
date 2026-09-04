"""Occlusion robustness analysis for EOVOT benchmark results.

Standard VOT benchmarks measure what a tracker predicts (accuracy via IoU)
and how fast it runs (FPS/latency).  They do not separate tracking performance
into **pre-occlusion**, **during-occlusion**, and **post-occlusion** phases.
This distinction is critical for edge deployment: a tracker may have good
aggregate IoU yet catastrophically fail to recover after every gap.

This module provides structured analysis of tracker behavior across occlusion
events using the ``occlusion_mask`` produced by
:class:`~eovot.datasets.occlusion.OcclusionSyntheticDataset` (or any other
dataset that provides per-frame occlusion labels).

Metrics
~~~~~~~

Pre-occlusion IoU
    Mean IoU in the ``context_frames`` immediately before each occlusion onset.
    Establishes the tracker's baseline performance when the target is fully
    visible.  Low values indicate the tracker was already struggling before
    the occlusion began.

During-occlusion IoU
    Mean IoU across all frames where the target is occluded.  Classical trackers
    typically coast on momentum; correlation-filter trackers may snap to the
    occluder.  Lower values indicate the tracker cannot maintain proximity to
    the true location during gaps.

Post-occlusion IoU
    Mean IoU in the first ``recovery_window`` frames after each occlusion ends.
    The most deployment-critical metric: slow recovery here means the tracker
    needs manual reinitialization after every gap, which defeats the purpose
    of autonomous tracking.

Recovery AUC
    Area under the per-frame IoU recovery curve (first ``recovery_window``
    frames after each occlusion end), averaged over all events.  Captures
    *how fast* recovery happens, not just the final level.  A tracker that
    recovers instantly scores 1.0; one that never recovers scores 0.0.

IoU Drop Ratio
    ``(pre_iou - during_iou) / (pre_iou + eps)``.  Fraction of pre-occlusion
    accuracy lost during the gap.  Allows comparison across trackers with
    different absolute accuracy levels — a low-accuracy tracker that loses
    50 % during occlusion is equally impaired as a high-accuracy tracker that
    loses 50 %.

Typical usage::

    from eovot.datasets.occlusion import OcclusionSyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker
    from eovot.metrics.occlusion import OcclusionRobustnessAnalyzer

    ds = OcclusionSyntheticDataset(num_sequences=5, num_frames=100)
    engine = BenchmarkEngine(verbose=False)
    bench_result = engine.run(MOSSETracker(), ds, dataset_name="OcclusionSynth")

    analyzer = OcclusionRobustnessAnalyzer(context_frames=10, recovery_window=15)

    for seq_result, seq in zip(bench_result.sequence_results, ds):
        r = analyzer.analyze(
            ious=seq_result.ious,
            occlusion_mask=seq.occlusion_mask,
            tracker_name="MOSSE",
            sequence_name=seq.name,
        )
        print(r)

    # Aggregate across all sequences:
    agg = analyzer.analyze_benchmark(bench_result, ds, tracker_name="MOSSE")
    print(agg["aggregate"])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult
    from ..datasets.occlusion import OcclusionSyntheticDataset


@dataclass
class OcclusionRobustnessResult:
    """Per-sequence occlusion robustness summary.

    Attributes:
        tracker_name:          Human-readable tracker identifier.
        sequence_name:         Sequence identifier.
        pre_occlusion_iou:     Mean IoU in the ``context_frames`` before each event.
        during_occlusion_iou:  Mean IoU across all occluded frames.
        post_occlusion_iou:    Mean IoU in the ``recovery_window`` after each event.
        recovery_auc:          Average AUC of the per-event IoU recovery curve.
        iou_drop_ratio:        ``(pre - during) / (pre + eps)`` — fractional IoU loss.
        num_occlusion_events:  Number of distinct occlusion windows in the sequence.
    """

    tracker_name: str
    sequence_name: str
    pre_occlusion_iou: float
    during_occlusion_iou: float
    post_occlusion_iou: float
    recovery_auc: float
    iou_drop_ratio: float
    num_occlusion_events: int

    def __str__(self) -> str:
        return (
            f"OcclusionRobustnessResult[{self.tracker_name} on {self.sequence_name}] "
            f"pre={self.pre_occlusion_iou:.4f}  "
            f"during={self.during_occlusion_iou:.4f}  "
            f"post={self.post_occlusion_iou:.4f}  "
            f"recovery_AUC={self.recovery_auc:.4f}  "
            f"drop_ratio={self.iou_drop_ratio:.4f}  "
            f"events={self.num_occlusion_events}"
        )


class OcclusionRobustnessAnalyzer:
    """Decompose tracker performance into pre/during/post-occlusion phases.

    Args:
        context_frames:  Number of frames before each occlusion onset to
                         include in the pre-occlusion IoU average.
                         Must be ≥ 1.  Default: ``10``.
        recovery_window: Number of frames after each occlusion end to use
                         for the post-occlusion IoU and recovery AUC.
                         Must be ≥ 1.  Default: ``15``.

    Raises:
        ValueError: If ``context_frames`` or ``recovery_window`` is < 1.
    """

    def __init__(
        self,
        context_frames: int = 10,
        recovery_window: int = 15,
    ) -> None:
        if context_frames < 1:
            raise ValueError(f"context_frames must be >= 1, got {context_frames!r}")
        if recovery_window < 1:
            raise ValueError(f"recovery_window must be >= 1, got {recovery_window!r}")
        self.context_frames = context_frames
        self.recovery_window = recovery_window

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_events(mask: np.ndarray) -> List[Tuple[int, int]]:
        """Return ``(start, end_exclusive)`` pairs for each occlusion window.

        Args:
            mask: Boolean array, ``True`` where target is occluded.

        Returns:
            Sorted list of ``(start, end)`` index tuples.
        """
        events: List[Tuple[int, int]] = []
        in_occ = False
        start = 0
        for i, occ in enumerate(mask):
            if occ and not in_occ:
                start = i
                in_occ = True
            elif not occ and in_occ:
                events.append((start, i))
                in_occ = False
        if in_occ:
            events.append((start, len(mask)))
        return events

    @staticmethod
    def _trapz(y: np.ndarray, x: np.ndarray) -> float:
        """Numpy-version-agnostic trapezoidal integration."""
        try:
            return float(np.trapezoid(y, x))  # numpy >= 2.0
        except AttributeError:
            return float(np.trapz(y, x))      # numpy < 2.0

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        ious: np.ndarray,
        occlusion_mask: np.ndarray,
        tracker_name: str = "",
        sequence_name: str = "",
    ) -> OcclusionRobustnessResult:
        """Run occlusion robustness analysis on a single sequence.

        Args:
            ious:           Per-frame IoU array, shape ``(N,)``.
            occlusion_mask: Boolean array ``(N,)`` — ``True`` where occluded.
                            Produced by
                            :class:`~eovot.datasets.occlusion.OcclusionSequence`.
            tracker_name:   Stored in the result for identification.
            sequence_name:  Stored in the result for identification.

        Returns:
            :class:`OcclusionRobustnessResult` with all phase statistics.
        """
        ious = np.asarray(ious, dtype=np.float64)
        mask = np.asarray(occlusion_mask, dtype=bool)
        n = min(len(ious), len(mask))
        ious = ious[:n]
        mask = mask[:n]

        events = self._detect_events(mask)

        # During-occlusion IoU
        occ_idx = np.where(mask)[0]
        during_iou = float(ious[occ_idx].mean()) if len(occ_idx) > 0 else 0.0

        # Pre / post-occlusion IoU and recovery AUC per event
        pre_ious_all: List[float] = []
        post_ious_all: List[float] = []
        recovery_aucs: List[float] = []

        for start, end in events:
            # Pre-occlusion context window
            pre_start = max(0, start - self.context_frames)
            if pre_start < start:
                pre_ious_all.extend(ious[pre_start:start].tolist())

            # Post-occlusion recovery window
            rec_end = min(n, end + self.recovery_window)
            if end < rec_end:
                rec_ious = ious[end:rec_end]
                post_ious_all.extend(rec_ious.tolist())
                # Normalise time axis to [0, 1] for comparability across events
                t = np.linspace(0.0, 1.0, len(rec_ious))
                recovery_aucs.append(self._trapz(rec_ious, t))

        pre_iou = float(np.mean(pre_ious_all)) if pre_ious_all else 0.0
        post_iou = float(np.mean(post_ious_all)) if post_ious_all else 0.0
        rec_auc = float(np.mean(recovery_aucs)) if recovery_aucs else 0.0

        # IoU drop ratio: fraction of pre-occlusion accuracy lost during gap
        iou_drop = (pre_iou - during_iou) / (pre_iou + 1e-9) if pre_iou > 0 else 0.0
        iou_drop = float(np.clip(iou_drop, 0.0, 1.0))

        return OcclusionRobustnessResult(
            tracker_name=tracker_name,
            sequence_name=sequence_name,
            pre_occlusion_iou=pre_iou,
            during_occlusion_iou=during_iou,
            post_occlusion_iou=post_iou,
            recovery_auc=rec_auc,
            iou_drop_ratio=iou_drop,
            num_occlusion_events=len(events),
        )

    def analyze_benchmark(
        self,
        benchmark_result: "BenchmarkResult",
        dataset: "OcclusionSyntheticDataset",
        tracker_name: str = "",
    ) -> Dict:
        """Aggregate occlusion analysis across all sequences in a benchmark run.

        Args:
            benchmark_result: Output of
                :class:`~eovot.benchmark.engine.BenchmarkEngine`.
            dataset: The :class:`~eovot.datasets.occlusion.OcclusionSyntheticDataset`
                used during benchmarking — needed to retrieve ``occlusion_mask``.
            tracker_name: Human-readable tracker identifier.

        Returns:
            Dict with two keys:

            * ``"per_sequence"`` — ``{seq_name: OcclusionRobustnessResult}``
            * ``"aggregate"`` — cross-sequence mean scalars
        """
        per_seq: Dict[str, OcclusionRobustnessResult] = {}

        for seq_result in benchmark_result.sequence_results:
            # Find matching sequence in dataset by name
            matched_seq = None
            for seq in dataset:
                if seq.name == seq_result.sequence_name:
                    matched_seq = seq
                    break
            if matched_seq is None:
                continue

            r = self.analyze(
                ious=seq_result.ious,
                occlusion_mask=matched_seq.occlusion_mask,
                tracker_name=tracker_name,
                sequence_name=seq_result.sequence_name,
            )
            per_seq[seq_result.sequence_name] = r

        if not per_seq:
            return {"per_sequence": {}, "aggregate": {}}

        vals = list(per_seq.values())
        n = len(vals)

        return {
            "per_sequence": per_seq,
            "aggregate": {
                "tracker_name": tracker_name,
                "num_sequences": n,
                "total_occlusion_events": sum(r.num_occlusion_events for r in vals),
                "mean_pre_occlusion_iou": round(float(np.mean([r.pre_occlusion_iou for r in vals])), 4),
                "mean_during_occlusion_iou": round(float(np.mean([r.during_occlusion_iou for r in vals])), 4),
                "mean_post_occlusion_iou": round(float(np.mean([r.post_occlusion_iou for r in vals])), 4),
                "mean_recovery_auc": round(float(np.mean([r.recovery_auc for r in vals])), 4),
                "mean_iou_drop_ratio": round(float(np.mean([r.iou_drop_ratio for r in vals])), 4),
            },
        }

    def to_markdown_table(self, results: List[OcclusionRobustnessResult]) -> str:
        """Format a list of per-sequence results as a Markdown table.

        Args:
            results: Per-sequence :class:`OcclusionRobustnessResult` objects.

        Returns:
            Multi-line Markdown string suitable for embedding in reports.
        """
        lines = [
            "| Sequence | Pre-IoU | During-IoU | Post-IoU | Recovery AUC | Drop Ratio | Events |",
            "|----------|--------:|-----------:|---------:|-------------:|-----------:|-------:|",
        ]
        for r in results:
            lines.append(
                f"| {r.sequence_name} "
                f"| {r.pre_occlusion_iou:.4f} "
                f"| {r.during_occlusion_iou:.4f} "
                f"| {r.post_occlusion_iou:.4f} "
                f"| {r.recovery_auc:.4f} "
                f"| {r.iou_drop_ratio:.4f} "
                f"| {r.num_occlusion_events} |"
            )
        return "\n".join(lines)

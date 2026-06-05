"""Temporal frame-skip analysis for edge deployment benchmarking.

On resource-constrained edge hardware a tracker may not be able to process
every incoming video frame.  This module quantifies the accuracy/throughput
tradeoff when only every k-th frame is processed — the central question being:

    "What is the minimum processing rate that preserves acceptable tracking
    accuracy, and what throughput gain does that afford?"

Overview
--------
Given a tracker, a dataset, and a list of *skip rates* k ∈ {1, 2, 4, 8, …},
:class:`FrameSkipEvaluator` runs the tracker on every k-th frame and fills
the remaining frames with either:

* **hold_last** — repeat the last tracked bounding box (conservative, safe).
* **linear** — linearly interpolate position/size between tracked keyframes
  (optimistic, models smooth motion well).

Accuracy (mIoU, success AUC, failure rate) is measured against ground truth
on *all* frames, so the gap between skip_rate=1 and skip_rate=k directly
measures the cost of temporal subsampling.

The :meth:`FrameSkipAnalysis.optimal_skip_rate` property returns the highest
skip rate where mean IoU has not degraded by more than 10 % relative to the
baseline — a practical rule-of-thumb for edge deployment decisions.

Example::

    from eovot.analysis.frame_skip import FrameSkipEvaluator
    from eovot.trackers.mosse import MOSSETracker
    from eovot.datasets.synthetic import SyntheticDataset

    dataset  = SyntheticDataset(num_sequences=4, frames_per_sequence=80)
    tracker  = MOSSETracker()
    evaluator = FrameSkipEvaluator(skip_rates=[1, 2, 4, 8])
    analysis  = evaluator.evaluate(tracker, dataset, "Synthetic")

    print(analysis.summary_table())
    print(f"Optimal skip rate: {analysis.optimal_skip_rate}×")
    print(f"Throughput gain at optimal: {analysis.optimal_skip_rate}×")

    # Persist for downstream plotting
    import json, pathlib
    pathlib.Path("frame_skip_results.json").write_text(
        json.dumps(analysis.to_dict(), indent=2)
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np

from ..datasets.base import BaseDataset, Sequence
from ..metrics.accuracy import MetricsEngine
from ..trackers.base import BaseTracker, BBox


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SkipRateResult:
    """Accuracy and throughput statistics at one temporal skip rate.

    Attributes:
        skip_rate: Number of frames between tracker updates.
            ``1`` means every frame is processed (no skip).
        mean_iou: Mean IoU over all frames and sequences at this skip rate.
        success_auc: Area under the success curve (IoU thresholds 0 → 1).
        failure_rate: Fraction of frames with IoU < ``failure_iou_threshold``.
        effective_fps_multiplier: Throughput multiplier relative to skip_rate=1.
            Equal to ``skip_rate`` — processing k× fewer frames gives k× more
            throughput headroom for the same hardware budget.
    """

    skip_rate: int
    mean_iou: float
    success_auc: float
    failure_rate: float
    effective_fps_multiplier: float


@dataclass
class FrameSkipAnalysis:
    """Complete temporal-sampling analysis for one tracker on one dataset.

    Attributes:
        tracker_name: Name of the evaluated tracker.
        dataset_name: Name of the dataset.
        baseline_fps: Measured FPS at skip_rate=1 on the host machine.
        interpolation: Strategy used for unprocessed frames.
        skip_results: Per-skip-rate accuracy/throughput results, sorted by
            skip_rate ascending.
    """

    tracker_name: str
    dataset_name: str
    baseline_fps: float
    interpolation: str
    skip_results: List[SkipRateResult] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def optimal_skip_rate(self, tolerance: float = 0.10) -> int:
        """Highest skip rate within ``tolerance`` of baseline accuracy.

        Searches for the largest k such that the relative IoU drop from
        skip_rate=1 is ≤ ``tolerance``.  Returns 1 when no improvement is
        safe or when results are unavailable.

        Args:
            tolerance: Maximum allowed relative IoU degradation. Default 0.10
                (10 % drop from baseline).

        Returns:
            Integer skip rate (e.g. ``4`` means processing every 4th frame).
        """
        return self._find_optimal(tolerance)

    def _find_optimal(self, tolerance: float = 0.10) -> int:
        if not self.skip_results:
            return 1
        baseline = self.skip_results[0].mean_iou  # skip_rate=1 is always first
        best = 1
        for r in self.skip_results:
            if baseline > 0:
                relative_drop = (baseline - r.mean_iou) / baseline
                if relative_drop <= tolerance:
                    best = r.skip_rate
            else:
                # baseline IoU is 0 — no useful signal
                if r.mean_iou == 0.0:
                    best = r.skip_rate
        return best

    def accuracy_at(self, skip_rate: int) -> Optional[float]:
        """Return mean IoU for a specific skip rate, or ``None`` if not evaluated."""
        for r in self.skip_results:
            if r.skip_rate == skip_rate:
                return r.mean_iou
        return None

    def degradation_by_skip_rate(self) -> Dict[int, float]:
        """Map skip_rate → relative IoU drop from skip_rate=1 (fraction in [0, 1]).

        A value of ``0.0`` means no degradation; ``1.0`` means complete failure.
        """
        if not self.skip_results:
            return {}
        baseline = self.skip_results[0].mean_iou
        return {
            r.skip_rate: max(0.0, (baseline - r.mean_iou) / baseline) if baseline > 0 else 0.0
            for r in self.skip_results
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary_table(self) -> str:
        """Markdown table of skip rate vs accuracy metrics.

        Returns:
            Multi-line Markdown string suitable for embedding in reports.
        """
        header = (
            "| Skip Rate | FPS Mult | mIoU   | Success AUC | Fail Rate | Opt? |"
        )
        sep = "|----------:|---------:|-------:|------------:|----------:|:----:|"
        lines = [header, sep]
        opt = self.optimal_skip_rate
        baseline = self.skip_results[0].mean_iou if self.skip_results else 0.0
        for r in self.skip_results:
            drop = (
                f"-{(baseline - r.mean_iou) / baseline * 100:.1f}%"
                if baseline > 0 and r.skip_rate > 1
                else "baseline"
            )
            mark = "✓" if r.skip_rate == opt else ""
            lines.append(
                f"| {r.skip_rate:>9} "
                f"| {r.effective_fps_multiplier:>7.1f}× "
                f"| {r.mean_iou:.4f} "
                f"| {r.success_auc:>11.4f} "
                f"| {r.failure_rate:>9.4f} "
                f"| {mark:>4} |"
                f"  ({drop})"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON export."""
        return {
            "tracker_name": self.tracker_name,
            "dataset_name": self.dataset_name,
            "baseline_fps": round(self.baseline_fps, 2),
            "interpolation": self.interpolation,
            "optimal_skip_rate": self.optimal_skip_rate,
            "skip_results": [
                {
                    "skip_rate": r.skip_rate,
                    "mean_iou": round(r.mean_iou, 4),
                    "success_auc": round(r.success_auc, 4),
                    "failure_rate": round(r.failure_rate, 4),
                    "effective_fps_multiplier": round(r.effective_fps_multiplier, 2),
                }
                for r in self.skip_results
            ],
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class FrameSkipEvaluator:
    """Evaluate how tracker accuracy degrades under temporal subsampling.

    Simulates processing only every k-th frame, with remaining frames handled
    via hold-last or linear interpolation.  This models edge deployment
    scenarios where the device cannot keep pace with the full video frame rate.

    Args:
        skip_rates: Frame skip rates to evaluate.  ``1`` (no skip) is always
            inserted automatically.  Default: ``[1, 2, 4, 8]``.
        interpolation: Strategy for unprocessed frames.

            * ``"hold_last"`` — repeat the last tracked bbox. Conservative;
              models zero-motion assumption between keyframes.
            * ``"linear"`` — linearly interpolate position and size between
              the two surrounding keyframes. Optimistic; assumes smooth motion.

        failure_iou_threshold: IoU threshold below which a frame is counted
            as a tracking failure.  Default: ``0.5``.
        verbose: Print per-skip-rate progress.  Default: ``True``.

    Example::

        evaluator = FrameSkipEvaluator(skip_rates=[1, 2, 4, 8])
        analysis = evaluator.evaluate(tracker, dataset, dataset_name="OTB100",
                                      max_sequences=10)
        print(analysis.summary_table())
        print(f"Optimal skip rate: {analysis.optimal_skip_rate}×")
    """

    def __init__(
        self,
        skip_rates: Optional[List[int]] = None,
        interpolation: Literal["hold_last", "linear"] = "hold_last",
        failure_iou_threshold: float = 0.5,
        verbose: bool = True,
    ) -> None:
        raw = list(skip_rates) if skip_rates is not None else [1, 2, 4, 8]
        if 1 not in raw:
            raw.insert(0, 1)
        self.skip_rates: List[int] = sorted(set(raw))
        if any(k < 1 for k in self.skip_rates):
            raise ValueError("All skip_rates must be ≥ 1.")
        self.interpolation: str = interpolation
        self.failure_iou_threshold = failure_iou_threshold
        self.verbose = verbose
        self._metrics = MetricsEngine()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        tracker: BaseTracker,
        dataset: BaseDataset,
        dataset_name: str = "unknown",
        max_sequences: Optional[int] = None,
    ) -> FrameSkipAnalysis:
        """Run frame-skip analysis across a dataset.

        The tracker is **re-initialised for every (sequence, skip_rate) pair**
        so that skip-rate runs are independent and results are comparable.

        Args:
            tracker: Tracker implementing :class:`~eovot.trackers.base.BaseTracker`.
            dataset: Dataset to evaluate on.
            dataset_name: Label used in result summaries.
            max_sequences: Cap on number of sequences.  ``None`` uses all.

        Returns:
            :class:`FrameSkipAnalysis` with one :class:`SkipRateResult` per
            configured skip rate, sorted by skip_rate ascending.
        """
        n = min(len(dataset), max_sequences) if max_sequences is not None else len(dataset)
        sequences = [dataset[i] for i in range(n)]

        if self.verbose:
            print(
                f"\n[FrameSkipEvaluator] {tracker.name} on {dataset_name} "
                f"({n} seq, skip_rates={self.skip_rates}, "
                f"interp={self.interpolation})"
            )
            print("-" * 60)

        baseline_fps = self._measure_baseline_fps(tracker, sequences)

        skip_results: List[SkipRateResult] = []
        for k in self.skip_rates:
            result = self._evaluate_skip_rate(tracker, sequences, k)
            skip_results.append(result)
            if self.verbose:
                print(
                    f"  skip={k:>2}×  FPS_mult={result.effective_fps_multiplier:.1f}×  "
                    f"mIoU={result.mean_iou:.4f}  AUC={result.success_auc:.4f}  "
                    f"fail={result.failure_rate:.4f}"
                )

        analysis = FrameSkipAnalysis(
            tracker_name=tracker.name,
            dataset_name=dataset_name,
            baseline_fps=baseline_fps,
            interpolation=self.interpolation,
            skip_results=skip_results,
        )

        if self.verbose:
            print("-" * 60)
            print(f"  Optimal skip rate: {analysis.optimal_skip_rate}×  "
                  f"(baseline FPS: {baseline_fps:.1f})")

        return analysis

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _measure_baseline_fps(
        self, tracker: BaseTracker, sequences: List[Sequence]
    ) -> float:
        """Wall-clock FPS measurement at skip_rate=1 (process every frame)."""
        total_frames = 0
        total_elapsed = 0.0

        for seq in sequences:
            frames = list(seq)
            gt = seq.ground_truth
            n = min(len(frames), len(gt))
            frames = frames[:n]

            tracker.initialize(frames[0], tuple(gt[0]))  # type: ignore[arg-type]
            for frame in frames[1:]:
                t0 = time.perf_counter()
                tracker.update(frame)
                total_elapsed += time.perf_counter() - t0
                total_frames += 1

        return float(total_frames / total_elapsed) if total_elapsed > 0 else 0.0

    def _evaluate_skip_rate(
        self,
        tracker: BaseTracker,
        sequences: List[Sequence],
        skip_rate: int,
    ) -> SkipRateResult:
        """Compute accuracy metrics at *skip_rate* across all sequences."""
        all_ious: List[np.ndarray] = []

        for seq in sequences:
            frames = list(seq)
            gt = seq.ground_truth
            n = min(len(frames), len(gt))
            frames = frames[:n]
            gt_arr = np.asarray(gt[:n], dtype=np.float64)

            seq_ious = self._run_sequence_with_skip(tracker, frames, gt_arr, skip_rate)
            if len(seq_ious):
                all_ious.append(seq_ious)

        if not all_ious:
            return SkipRateResult(
                skip_rate=skip_rate,
                mean_iou=0.0,
                success_auc=0.0,
                failure_rate=1.0,
                effective_fps_multiplier=float(skip_rate),
            )

        ious = np.concatenate(all_ious)
        thresholds = np.linspace(0.0, 1.0, 101)
        _, success_rates = self._metrics.success_curve(ious, thresholds)

        try:
            _trapz = np.trapezoid  # NumPy ≥ 2.0
        except AttributeError:
            _trapz = np.trapz  # NumPy < 2.0

        success_auc = float(_trapz(success_rates, thresholds))
        failure_rate = float((ious < self.failure_iou_threshold).mean())

        return SkipRateResult(
            skip_rate=skip_rate,
            mean_iou=float(ious.mean()),
            success_auc=success_auc,
            failure_rate=failure_rate,
            effective_fps_multiplier=float(skip_rate),
        )

    def _run_sequence_with_skip(
        self,
        tracker: BaseTracker,
        frames: List[np.ndarray],
        gt: np.ndarray,
        skip_rate: int,
    ) -> np.ndarray:
        """Run tracker on keyframes; fill skipped frames via chosen strategy.

        Args:
            tracker: Tracker to initialise and update.
            frames: All frames in the sequence (length N).
            gt: Ground-truth boxes ``(N, 4)`` — used only for init bbox.
            skip_rate: Number of frames between tracker calls.

        Returns:
            Per-frame IoU array of shape ``(N,)``.
        """
        n = len(frames)
        if n == 0:
            return np.empty(0, dtype=np.float64)

        init_bbox: BBox = tuple(gt[0].tolist())  # type: ignore[arg-type]
        tracker.initialize(frames[0], init_bbox)

        # predictions[i] will hold the predicted bbox for frame i
        predictions: List[Optional[BBox]] = [None] * n
        predictions[0] = init_bbox

        prev_pred: BBox = init_bbox
        prev_keyframe_idx: int = 0

        for i in range(1, n):
            if i % skip_rate == 0:
                # Keyframe: run the tracker
                bbox = tracker.update(frames[i])
                predictions[i] = bbox

                if self.interpolation == "linear" and prev_keyframe_idx < i - 1:
                    # Back-fill frames between the last keyframe and this one
                    self._fill_linear(
                        predictions, prev_keyframe_idx, i, prev_pred, bbox
                    )
                elif self.interpolation == "hold_last":
                    # Hold-last for the gap (already set below in the else branch)
                    pass

                prev_pred = bbox
                prev_keyframe_idx = i
            else:
                # Non-keyframe: always hold_last as default (linear fills later)
                predictions[i] = prev_pred

        # Ensure no None entries remain (can happen at end of sequence)
        for i in range(n):
            if predictions[i] is None:
                predictions[i] = prev_pred

        preds_arr = np.array(predictions, dtype=np.float64)
        return self._metrics.batch_iou(preds_arr, gt)

    @staticmethod
    def _fill_linear(
        predictions: List[Optional[BBox]],
        start_idx: int,
        end_idx: int,
        start_bbox: BBox,
        end_bbox: BBox,
    ) -> None:
        """Fill predictions[start_idx+1 : end_idx] with linear interpolation.

        Args:
            predictions: In-place target list.
            start_idx: Index of the first known keyframe (not filled).
            end_idx: Index of the second known keyframe (not filled).
            start_bbox: Bbox at start_idx ``(x, y, w, h)``.
            end_bbox: Bbox at end_idx ``(x, y, w, h)``.
        """
        gap = end_idx - start_idx
        s = np.asarray(start_bbox, dtype=np.float64)
        e = np.asarray(end_bbox, dtype=np.float64)
        for offset in range(1, gap):
            alpha = offset / gap
            interp_arr = s + alpha * (e - s)
            predictions[start_idx + offset] = tuple(interp_arr.tolist())  # type: ignore[assignment]

"""Core benchmark engine for EOVOT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..datasets.base import BaseDataset, Sequence
from ..metrics.accuracy import AccuracyMetrics, MetricsEngine, center_distance
from ..profiling.energy import EnergyProfiler, EnergyResult
from ..profiling.profiler import Profiler, ProfilingResult
from ..trackers.base import BaseTracker


@dataclass
class SequenceResult:
    """Per-sequence evaluation output from the benchmark engine.

    Attributes:
        sequence_name: Name of the evaluated sequence.
        ious: Per-frame IoU values, shape ``(N,)``.
        profiling: Timing and memory profiling summary.
        predictions: Predicted bounding boxes, shape ``(N, 4)``.
        ground_truths: Ground-truth boxes aligned to predictions, shape ``(N, 4)``.
        center_distances: Per-frame centre-distance in pixels, shape ``(N,)``.
        energy: CPU energy profiling result (``None`` if energy profiling disabled).
        accuracy: AUC-based accuracy metrics (``None`` if sequence has < 2 frames).
    """

    sequence_name: str
    ious: np.ndarray
    profiling: ProfilingResult
    predictions: Optional[np.ndarray] = None        # shape (N, 4) — predicted boxes
    ground_truths: Optional[np.ndarray] = None      # shape (N, 4) — GT boxes aligned
    center_distances: Optional[np.ndarray] = None   # shape (N,)  — per-frame centre-dist
    energy: Optional[EnergyResult] = None           # energy estimate, if TDP provided
    accuracy: Optional[AccuracyMetrics] = None      # success/precision AUC metrics

    @property
    def mean_iou(self) -> float:
        return float(self.ious.mean()) if len(self.ious) else 0.0

    @property
    def mean_center_distance(self) -> Optional[float]:
        """Mean centre-distance in pixels, or ``None`` if not stored."""
        if self.center_distances is None or len(self.center_distances) == 0:
            return None
        return float(self.center_distances.mean())


@dataclass
class BenchmarkResult:
    """Aggregate benchmark output for one tracker evaluated on one dataset.

    Attributes:
        tracker_name: Name of the evaluated tracker.
        dataset_name: Name of the dataset used.
        sequence_results: One :class:`SequenceResult` per evaluated sequence.
    """

    tracker_name: str
    dataset_name: str
    sequence_results: List[SequenceResult] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Aggregate accuracy properties
    # ------------------------------------------------------------------

    @property
    def mean_iou(self) -> float:
        """Mean IoU across all frames and sequences."""
        all_ious = np.concatenate([r.ious for r in self.sequence_results])
        return float(all_ious.mean()) if len(all_ious) else 0.0

    @property
    def success_auc(self) -> Optional[float]:
        """Mean success-curve AUC across sequences, or ``None`` if unavailable."""
        aucs = [
            r.accuracy.success_auc
            for r in self.sequence_results
            if r.accuracy is not None
        ]
        return float(np.mean(aucs)) if aucs else None

    @property
    def precision_auc(self) -> Optional[float]:
        """Mean precision-curve AUC across sequences, or ``None`` if unavailable."""
        aucs = [
            r.accuracy.precision_auc
            for r in self.sequence_results
            if r.accuracy is not None
        ]
        return float(np.mean(aucs)) if aucs else None

    @property
    def mean_center_distance(self) -> Optional[float]:
        """Mean centre-distance across all sequences in pixels, or ``None``."""
        dists = [
            r.center_distances
            for r in self.sequence_results
            if r.center_distances is not None
        ]
        if not dists:
            return None
        return float(np.concatenate(dists).mean())

    # ------------------------------------------------------------------
    # Profiling aggregates
    # ------------------------------------------------------------------

    @property
    def mean_fps(self) -> float:
        """Mean FPS across all sequences."""
        return float(np.mean([r.profiling.fps for r in self.sequence_results]))

    @property
    def peak_memory_mb(self) -> float:
        """Peak RSS memory across all sequences in megabytes."""
        return float(np.max([r.profiling.peak_memory_mb for r in self.sequence_results]))

    @property
    def total_energy_j(self) -> Optional[float]:
        """Total estimated energy across all sequences (Joules), or ``None``."""
        with_energy = [r for r in self.sequence_results if r.energy is not None]
        if not with_energy:
            return None
        return sum(r.energy.total_energy_j for r in with_energy)  # type: ignore[union-attr]

    @property
    def mean_energy_per_frame_mj(self) -> Optional[float]:
        """Mean per-frame energy across all sequences (milli-Joules), or ``None``."""
        with_energy = [r for r in self.sequence_results if r.energy is not None]
        if not with_energy:
            return None
        return float(np.mean([r.energy.energy_per_frame_mj for r in with_energy]))  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        """Return a flat dict of aggregate metrics suitable for reporting.

        AUC fields (``success_auc``, ``precision_auc``) and energy fields
        (``total_energy_j``, ``mean_energy_per_frame_mj``) are included only
        when the underlying data is available.
        """
        d: Dict = {
            "tracker": self.tracker_name,
            "dataset": self.dataset_name,
            "num_sequences": len(self.sequence_results),
            "mean_iou": round(self.mean_iou, 4),
            "mean_fps": round(self.mean_fps, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
        }
        s_auc = self.success_auc
        if s_auc is not None:
            d["success_auc"] = round(s_auc, 4)
        p_auc = self.precision_auc
        if p_auc is not None:
            d["precision_auc"] = round(p_auc, 4)
        mcd = self.mean_center_distance
        if mcd is not None:
            d["mean_center_distance_px"] = round(mcd, 3)
        e_total = self.total_energy_j
        if e_total is not None:
            d["total_energy_j"] = round(e_total, 6)
        e_frame = self.mean_energy_per_frame_mj
        if e_frame is not None:
            d["mean_energy_per_frame_mj"] = round(e_frame, 4)
        return d

    def to_dict(self) -> Dict:
        """Serialise to the dict format consumed by :class:`~eovot.reporting.reporter.BenchmarkReporter`.

        Returns a nested dict with two keys:

        * ``"summary"`` — aggregate scalar metrics (same as :meth:`summary`).
        * ``"sequences"`` — list of per-sequence metric dicts, each including
          ``success_auc`` / ``precision_auc`` when accuracy data is present,
          and ``energy_j`` / ``energy_per_frame_mj`` when energy was profiled.
        """
        sequences = []
        for r in self.sequence_results:
            entry: Dict = {
                "sequence_name": r.sequence_name,
                "mean_iou": round(r.mean_iou, 4),
                "fps": round(r.profiling.fps, 2),
                "mean_latency_ms": round(r.profiling.latency_mean_ms, 3),
                "peak_memory_mb": round(r.profiling.peak_memory_mb, 2),
            }
            if r.accuracy is not None:
                entry["success_auc"] = round(r.accuracy.success_auc, 4)
                entry["precision_auc"] = round(r.accuracy.precision_auc, 4)
            if r.energy is not None:
                entry["energy_j"] = round(r.energy.total_energy_j, 6)
                entry["energy_per_frame_mj"] = round(r.energy.energy_per_frame_mj, 4)
            sequences.append(entry)
        return {"summary": self.summary(), "sequences": sequences}

    def __str__(self) -> str:
        s = self.summary()
        parts = [
            f"BenchmarkResult[{s['tracker']} on {s['dataset']}]",
            f"mIoU={s['mean_iou']}",
            f"FPS={s['mean_fps']}",
            f"mem={s['peak_memory_mb']} MiB",
            f"({s['num_sequences']} sequences)",
        ]
        if "success_auc" in s:
            parts.append(f"AUC={s['success_auc']}")
        if "total_energy_j" in s:
            parts.append(f"energy={s['total_energy_j']} J")
        return "  ".join(parts)


class BenchmarkEngine:
    """Run a tracker against a dataset and collect accuracy + profiling data.

    Args:
        verbose: Print per-sequence progress to stdout. Default: ``True``.
        tdp_watts: If provided, enables CPU energy estimation using
            :class:`~eovot.profiling.energy.EnergyProfiler` with this TDP
            value (Watts).  Set to the device's CPU TDP for meaningful
            estimates (e.g. ``6.0`` for Raspberry Pi 4, ``15.0`` for a
            laptop).  Default: ``None`` (energy profiling disabled).

    Example::

        engine = BenchmarkEngine(tdp_watts=10.0)
        result = engine.run(tracker, dataset, dataset_name="OTB100")
        print(result)           # includes mIoU, FPS, AUC, energy
        print(result.summary()) # flat dict for reporting
    """

    def __init__(self, verbose: bool = True, tdp_watts: Optional[float] = None) -> None:
        self.verbose = verbose
        self._metrics = MetricsEngine()
        self._profiler = Profiler()
        self._energy_profiler: Optional[EnergyProfiler] = (
            EnergyProfiler(tdp_watts=tdp_watts) if tdp_watts is not None else None
        )

    def run(
        self,
        tracker: BaseTracker,
        dataset: BaseDataset,
        dataset_name: str = "unknown",
        max_sequences: Optional[int] = None,
    ) -> BenchmarkResult:
        """Evaluate *tracker* on every sequence in *dataset*.

        Args:
            tracker: Tracker implementing :class:`~eovot.trackers.base.BaseTracker`.
            dataset: Dataset implementing :class:`~eovot.datasets.base.BaseDataset`.
            dataset_name: Label embedded in the result (default: ``"unknown"``)
            max_sequences: If set, evaluate at most this many sequences.

        Returns:
            :class:`BenchmarkResult` with per-sequence and aggregate metrics.
        """
        result = BenchmarkResult(tracker_name=tracker.name, dataset_name=dataset_name)
        n = min(len(dataset), max_sequences) if max_sequences is not None else len(dataset)

        if self.verbose:
            energy_tag = (
                f"  [energy TDP={self._energy_profiler.tdp_watts}W]"
                if self._energy_profiler else ""
            )
            print(f"\nEvaluating {tracker.name} on {dataset_name} ({n} sequences){energy_tag}")
            print("-" * 60)

        for idx in range(n):
            seq = dataset[idx]
            seq_result = self._run_sequence(tracker, seq)
            result.sequence_results.append(seq_result)
            if self.verbose:
                energy_str = ""
                if seq_result.energy is not None:
                    energy_str = f"  E={seq_result.energy.energy_per_frame_mj:.2f}mJ/fr"
                auc_str = ""
                if seq_result.accuracy is not None:
                    auc_str = f"  AUC={seq_result.accuracy.success_auc:.3f}"
                print(
                    f"  [{idx + 1:>3}/{n}] {seq_result.sequence_name:<30s} "
                    f"mIoU={seq_result.mean_iou:.3f}  "
                    f"FPS={seq_result.profiling.fps:.1f}"
                    f"{auc_str}{energy_str}"
                )

        if self.verbose:
            print("-" * 60)
            print(result)

        return result

    def _run_sequence(self, tracker: BaseTracker, seq: Sequence) -> SequenceResult:
        """Run a single sequence and return its result."""
        self._profiler.reset()
        if self._energy_profiler is not None:
            self._energy_profiler.reset()

        frames = list(seq)
        gt = seq.ground_truth
        preds: List = []

        for i, frame in enumerate(frames):
            if i == 0:
                tracker.initialize(frame, seq.init_bbox)
                preds.append(seq.init_bbox)
            else:
                self._profiler.start_frame()
                if self._energy_profiler is not None:
                    self._energy_profiler.start_frame()
                bbox = tracker.update(frame)
                self._profiler.end_frame()
                if self._energy_profiler is not None:
                    self._energy_profiler.end_frame()
                preds.append(bbox)

        preds_arr = np.array(preds, dtype=np.float64)
        n_eval = min(len(preds_arr), len(gt))
        preds_eval = preds_arr[:n_eval]
        gt_eval = gt[:n_eval]

        ious = self._metrics.batch_iou(preds_eval, gt_eval)

        # Per-frame centre-distances for precision curve computation.
        dists = np.array(
            [
                center_distance(tuple(preds_eval[i]), tuple(gt_eval[i]))  # type: ignore[arg-type]
                for i in range(n_eval)
            ],
            dtype=np.float64,
        )

        # Success and precision AUC — requires at least 2 evaluated frames.
        accuracy: Optional[AccuracyMetrics] = None
        if n_eval > 1:
            accuracy = self._metrics.compute_all(preds_eval, gt_eval)

        # Energy summary (skipped silently if no frames were measured).
        energy: Optional[EnergyResult] = None
        if self._energy_profiler is not None:
            try:
                energy = self._energy_profiler.summary(tracker.name)
            except ValueError:
                pass  # sequence too short (0 update frames)

        return SequenceResult(
            sequence_name=seq.name,
            ious=ious,
            profiling=self._profiler.summary(tracker.name),
            predictions=preds_eval,
            ground_truths=gt_eval,
            center_distances=dists,
            energy=energy,
            accuracy=accuracy,
        )

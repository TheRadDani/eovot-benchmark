"""Core benchmark engine for EOVOT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..datasets.base import BaseDataset, Sequence
from ..metrics.accuracy import MetricsEngine, center_distance
from ..metrics.vot_metrics import VOTEvaluator, VOTMetrics
from ..profiling.energy import EnergyProfiler, EnergyResult
from ..profiling.profiler import Profiler, ProfilingResult
from ..trackers.base import BaseTracker


@dataclass
class SequenceResult:
    sequence_name: str
    ious: np.ndarray
    profiling: ProfilingResult
    predictions: Optional[np.ndarray] = None       # shape (N, 4) — predicted boxes
    ground_truths: Optional[np.ndarray] = None     # shape (N, 4) — GT boxes aligned to predictions
    center_distances: Optional[np.ndarray] = None  # shape (N,)  — per-frame centre-distance (px)
    energy: Optional[EnergyResult] = None          # CPU energy estimate, if TDP profiling enabled

    @property
    def mean_iou(self) -> float:
        return float(self.ious.mean()) if len(self.ious) else 0.0

    @property
    def mean_center_distance(self) -> Optional[float]:
        """Mean centre-distance in pixels, or None if not stored."""
        if self.center_distances is None or len(self.center_distances) == 0:
            return None
        return float(self.center_distances.mean())


@dataclass
class BenchmarkResult:
    tracker_name: str
    dataset_name: str
    sequence_results: List[SequenceResult] = field(default_factory=list)
    vot_metrics: Optional[VOTMetrics] = None  # Populated when vot_protocol=True

    @property
    def mean_iou(self) -> float:
        all_ious = np.concatenate([r.ious for r in self.sequence_results])
        return float(all_ious.mean()) if len(all_ious) else 0.0

    @property
    def mean_center_distance(self) -> Optional[float]:
        """Mean centre-distance across all sequences in pixels, or None if not stored."""
        dists = [
            r.center_distances for r in self.sequence_results
            if r.center_distances is not None
        ]
        if not dists:
            return None
        return float(np.concatenate(dists).mean())

    @property
    def mean_fps(self) -> float:
        return float(np.mean([r.profiling.fps for r in self.sequence_results]))

    @property
    def peak_memory_mb(self) -> float:
        return float(np.max([r.profiling.peak_memory_mb for r in self.sequence_results]))

    @property
    def total_energy_j(self) -> Optional[float]:
        """Sum of per-sequence energy estimates (Joules), or ``None`` if not profiled."""
        with_energy = [r for r in self.sequence_results if r.energy is not None]
        if not with_energy:
            return None
        return sum(r.energy.total_energy_j for r in with_energy)

    @property
    def mean_energy_per_frame_mj(self) -> Optional[float]:
        """Mean per-frame energy across all sequences (milli-Joules), or ``None``."""
        with_energy = [r for r in self.sequence_results if r.energy is not None]
        if not with_energy:
            return None
        return float(np.mean([r.energy.energy_per_frame_mj for r in with_energy]))

    def summary(self) -> Dict:
        d: Dict = {
            "tracker": self.tracker_name,
            "dataset": self.dataset_name,
            "num_sequences": len(self.sequence_results),
            "mean_iou": round(self.mean_iou, 4),
            "mean_fps": round(self.mean_fps, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
        }
        mcd = self.mean_center_distance
        if mcd is not None:
            d["mean_center_distance_px"] = round(mcd, 3)
        energy = self.total_energy_j
        if energy is not None:
            d["total_energy_j"] = round(energy, 6)
        mepf = self.mean_energy_per_frame_mj
        if mepf is not None:
            d["mean_energy_per_frame_mj"] = round(mepf, 4)
        if self.vot_metrics is not None:
            d["vot_eao"] = round(self.vot_metrics.eao, 4)
            d["vot_accuracy"] = round(self.vot_metrics.accuracy, 4)
            d["vot_robustness"] = round(self.vot_metrics.robustness, 6)
        return d

    def to_dict(self) -> Dict:
        """Serialize the full result to a dict compatible with
        :class:`~eovot.reporting.reporter.BenchmarkReporter`.

        Returns a nested dict with keys ``"summary"`` (aggregate metrics)
        and ``"sequences"`` (per-sequence breakdown), suitable for JSON
        export and Markdown table generation.
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
            if r.energy is not None:
                entry["energy_j"] = round(r.energy.total_energy_j, 6)
                entry["energy_per_frame_mj"] = round(r.energy.energy_per_frame_mj, 4)
            sequences.append(entry)

        result: Dict = {"summary": self.summary(), "sequences": sequences}
        if self.vot_metrics is not None:
            result["vot"] = self.vot_metrics.summary()
        return result

    def __str__(self) -> str:
        s = self.summary()
        base = (
            f"BenchmarkResult[{s['tracker']} on {s['dataset']}] "
            f"mIoU={s['mean_iou']}  FPS={s['mean_fps']}  "
            f"mem={s['peak_memory_mb']} MiB  ({s['num_sequences']} sequences)"
        )
        if "total_energy_j" in s:
            base += f"  energy={s['total_energy_j']} J"
        if "vot_eao" in s:
            base += f"  EAO={s['vot_eao']}"
        return base


class BenchmarkEngine:
    """Run a tracker against a dataset and collect accuracy + profiling data.

    Args:
        verbose: Print per-sequence progress to stdout. Default: ``True``.
        tdp_watts: If provided, enables CPU energy estimation using
            :class:`~eovot.profiling.energy.EnergyProfiler` with this TDP
            value (Watts).  Set to the device's CPU TDP for meaningful
            estimates (e.g. ``6.0`` for Raspberry Pi 4, ``15.0`` for a
            laptop).  Default: ``None`` (energy profiling disabled).
        vot_protocol: If ``True``, compute VOT EAO / Robustness / Accuracy
            metrics after the run and attach them to the returned
            :class:`BenchmarkResult`.  Default: ``False``.
    """

    def __init__(
        self,
        verbose: bool = True,
        tdp_watts: Optional[float] = None,
        vot_protocol: bool = False,
    ) -> None:
        self.verbose = verbose
        self.vot_protocol = vot_protocol
        self._metrics = MetricsEngine()
        self._profiler = Profiler()
        self._energy_profiler: Optional[EnergyProfiler] = (
            EnergyProfiler(tdp_watts=tdp_watts) if tdp_watts is not None else None
        )
        self._vot_evaluator = VOTEvaluator() if vot_protocol else None

    def run(
        self,
        tracker: BaseTracker,
        dataset: BaseDataset,
        dataset_name: str = "unknown",
        max_sequences: Optional[int] = None,
    ) -> BenchmarkResult:
        """Evaluate *tracker* on every sequence in *dataset*.

        Args:
            tracker: Tracker instance implementing :class:`~eovot.trackers.base.BaseTracker`.
            dataset: Dataset providing sequences via ``__getitem__``.
            dataset_name: Human-readable dataset label embedded in the result.
            max_sequences: Cap evaluation at this many sequences.  ``None``
                evaluates the full dataset.

        Returns:
            :class:`BenchmarkResult` with per-sequence IoU, profiling data,
            and (optionally) VOT metrics.
        """
        result = BenchmarkResult(tracker_name=tracker.name, dataset_name=dataset_name)
        n = min(len(dataset), max_sequences) if max_sequences is not None else len(dataset)

        if self.verbose:
            energy_tag = (
                f"  [energy TDP={self._energy_profiler.tdp_watts}W]"
                if self._energy_profiler else ""
            )
            vot_tag = "  [VOT protocol]" if self.vot_protocol else ""
            print(
                f"\nEvaluating {tracker.name} on {dataset_name} "
                f"({n} sequences){energy_tag}{vot_tag}"
            )
            print("-" * 60)

        for idx in range(n):
            seq = dataset[idx]
            seq_result = self._run_sequence(tracker, seq)
            result.sequence_results.append(seq_result)
            if self.verbose:
                energy_str = ""
                if seq_result.energy is not None:
                    energy_str = f"  E={seq_result.energy.energy_per_frame_mj:.2f}mJ/fr"
                print(
                    f"  [{idx + 1:>3}/{n}] {seq_result.sequence_name:<30s} "
                    f"mIoU={seq_result.mean_iou:.3f}  "
                    f"FPS={seq_result.profiling.fps:.1f}"
                    f"{energy_str}"
                )

        if self.verbose:
            print("-" * 60)

        # Compute VOT metrics if requested
        if self._vot_evaluator is not None and result.sequence_results:
            seq_names = [r.sequence_name for r in result.sequence_results]
            seq_overlaps = [r.ious.tolist() for r in result.sequence_results]
            result.vot_metrics = self._vot_evaluator.evaluate(seq_names, seq_overlaps)

        if self.verbose:
            print(result)

        return result

    def _run_sequence(self, tracker: BaseTracker, seq: Sequence) -> SequenceResult:
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

        # Per-frame centre-distances for precision curve computation
        dists = np.array(
            [
                center_distance(
                    tuple(preds_eval[i]),  # type: ignore[arg-type]
                    tuple(gt_eval[i]),     # type: ignore[arg-type]
                )
                for i in range(n_eval)
            ],
            dtype=np.float64,
        )

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
        )

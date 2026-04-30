"""Core benchmark engine for EOVOT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..datasets.base import BaseDataset, Sequence
from ..metrics.accuracy import MetricsEngine, center_distance
from ..metrics.robustness import RobustnessAnalyzer, RobustnessResult
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
    energy: Optional[EnergyResult] = None          # energy estimate; None when TDP not configured
    robustness: Optional[RobustnessResult] = None  # VOT-style robustness analysis

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

    @property
    def mean_eao(self) -> Optional[float]:
        """Mean Expected Average Overlap across sequences, or None if not computed."""
        rob = [r.robustness for r in self.sequence_results if r.robustness is not None]
        if not rob:
            return None
        return float(np.mean([r.eao for r in rob]))

    @property
    def total_failures(self) -> Optional[int]:
        """Total tracking failures across all sequences, or None if not computed."""
        rob = [r.robustness for r in self.sequence_results if r.robustness is not None]
        if not rob:
            return None
        return sum(r.num_failures for r in rob)

    @property
    def mean_survival_rate(self) -> Optional[float]:
        """Mean survival rate (fraction of frames above IoU threshold), or None."""
        rob = [r.robustness for r in self.sequence_results if r.robustness is not None]
        if not rob:
            return None
        return float(np.mean([r.survival_rate for r in rob]))

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
        eao = self.mean_eao
        if eao is not None:
            d["mean_eao"] = round(eao, 4)
        failures = self.total_failures
        if failures is not None:
            d["total_failures"] = failures
        survival = self.mean_survival_rate
        if survival is not None:
            d["mean_survival_rate"] = round(survival, 4)
        e_total = self.total_energy_j
        if e_total is not None:
            d["total_energy_j"] = round(e_total, 4)
        e_frame = self.mean_energy_per_frame_mj
        if e_frame is not None:
            d["mean_energy_per_frame_mj"] = round(e_frame, 4)
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
            if r.robustness is not None:
                entry["eao"] = round(r.robustness.eao, 4)
                entry["num_failures"] = r.robustness.num_failures
                entry["survival_rate"] = round(r.robustness.survival_rate, 4)
                entry["mean_recovery_lag"] = round(r.robustness.mean_recovery_lag, 2)
            sequences.append(entry)
        return {"summary": self.summary(), "sequences": sequences}

    def __str__(self) -> str:
        s = self.summary()
        base = (
            f"BenchmarkResult[{s['tracker']} on {s['dataset']}] "
            f"mIoU={s['mean_iou']}  FPS={s['mean_fps']}  "
            f"mem={s['peak_memory_mb']} MiB  ({s['num_sequences']} sequences)"
        )
        if "mean_eao" in s:
            base += f"  EAO={s['mean_eao']}"
        if "total_energy_j" in s:
            base += f"  energy={s['total_energy_j']} J"
        return base


class BenchmarkEngine:
    """Run a tracker against a dataset and collect accuracy, robustness, and profiling data.

    Args:
        verbose: Print per-sequence progress to stdout. Default: ``True``.
        tdp_watts: If provided, enables CPU energy estimation using
            :class:`~eovot.profiling.energy.EnergyProfiler` with this TDP
            value (Watts).  Set to the device's CPU TDP for meaningful
            estimates (e.g. ``6.0`` for Raspberry Pi 4, ``15.0`` for a
            laptop).  Default: ``None`` (energy profiling disabled).
        failure_threshold: IoU below which a frame is counted as a tracking
            failure for robustness analysis. Default: ``0.1``.
        burn_in_frames: Frames to skip at sequence start before counting
            failures. Default: ``5``.
    """

    def __init__(
        self,
        verbose: bool = True,
        tdp_watts: Optional[float] = None,
        failure_threshold: float = 0.1,
        burn_in_frames: int = 5,
    ) -> None:
        self.verbose = verbose
        self._metrics = MetricsEngine()
        self._profiler = Profiler()
        self._robustness = RobustnessAnalyzer(
            failure_threshold=failure_threshold,
            burn_in_frames=burn_in_frames,
        )
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

        Returns a :class:`BenchmarkResult` containing per-sequence and
        aggregate accuracy, robustness, profiling, and (optionally) energy metrics.
        """
        result = BenchmarkResult(tracker_name=tracker.name, dataset_name=dataset_name)
        n = min(len(dataset), max_sequences) if max_sequences is not None else len(dataset)

        if self.verbose:
            energy_tag = f"  [energy TDP={self._energy_profiler.tdp_watts}W]" if self._energy_profiler else ""
            print(f"\nEvaluating {tracker.name} on {dataset_name} ({n} sequences){energy_tag}")
            print("-" * 60)

        for idx in range(n):
            seq = dataset[idx]
            seq_result = self._run_sequence(tracker, seq)
            result.sequence_results.append(seq_result)
            if self.verbose:
                rob_str = ""
                if seq_result.robustness is not None:
                    rob_str = (
                        f"  EAO={seq_result.robustness.eao:.3f}"
                        f"  fails={seq_result.robustness.num_failures}"
                    )
                energy_str = ""
                if seq_result.energy is not None:
                    energy_str = f"  E={seq_result.energy.energy_per_frame_mj:.2f}mJ/fr"
                print(
                    f"  [{idx + 1:>3}/{n}] {seq_result.sequence_name:<28s} "
                    f"mIoU={seq_result.mean_iou:.3f}  "
                    f"FPS={seq_result.profiling.fps:.1f}"
                    f"{rob_str}{energy_str}"
                )

        if self.verbose:
            print("-" * 60)
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

        dists = np.array(
            [center_distance(tuple(preds_eval[i]), tuple(gt_eval[i]))  # type: ignore[arg-type]
             for i in range(n_eval)],
            dtype=np.float64,
        )

        robustness = self._robustness.analyze_sequence(
            ious,
            tracker_name=tracker.name,
            sequence_name=seq.name,
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
            robustness=robustness,
        )

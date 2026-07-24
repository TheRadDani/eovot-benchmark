"""Core benchmark engine for EOVOT."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from ..datasets.base import BaseDataset, Sequence
from ..metrics.accuracy import AccuracyMetrics, MetricsEngine
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
    accuracy_metrics: Optional[AccuracyMetrics] = None  # success AUC, precision AUC

    @property
    def mean_iou(self) -> float:
        return float(self.ious.mean()) if len(self.ious) else 0.0

    @property
    def mean_center_distance(self) -> Optional[float]:
        """Mean centre-distance in pixels, or None if not stored."""
        if self.center_distances is None or len(self.center_distances) == 0:
            return None
        return float(self.center_distances.mean())

    @property
    def success_auc(self) -> Optional[float]:
        """Area under the success curve, or None if not computed."""
        return self.accuracy_metrics.success_auc if self.accuracy_metrics else None

    @property
    def precision_auc(self) -> Optional[float]:
        """Normalised AUC of the precision curve, or None if not computed."""
        return self.accuracy_metrics.precision_auc if self.accuracy_metrics else None


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
    def mean_success_auc(self) -> Optional[float]:
        """Mean success-curve AUC across all sequences, or ``None`` if not computed."""
        aucs = [r.accuracy_metrics.success_auc for r in self.sequence_results
                if r.accuracy_metrics is not None]
        return float(np.mean(aucs)) if aucs else None

    @property
    def mean_precision_auc(self) -> Optional[float]:
        """Mean precision-curve AUC across all sequences, or ``None`` if not computed."""
        aucs = [r.accuracy_metrics.precision_auc for r in self.sequence_results
                if r.accuracy_metrics is not None]
        return float(np.mean(aucs)) if aucs else None

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
        sauc = self.mean_success_auc
        if sauc is not None:
            d["success_auc"] = round(sauc, 4)
        pauc = self.mean_precision_auc
        if pauc is not None:
            d["precision_auc"] = round(pauc, 4)
        e_total = self.total_energy_j
        if e_total is not None:
            d["total_energy_j"] = round(e_total, 4)
        e_frame = self.mean_energy_per_frame_mj
        if e_frame is not None:
            d["mean_energy_per_frame_mj"] = round(e_frame, 4)
        return d

    def to_dict(self) -> Dict:
        """Serialize the full result to a dict compatible with :class:`~eovot.reporting.reporter.BenchmarkReporter`.

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
            if r.accuracy_metrics is not None:
                entry["success_auc"] = round(r.accuracy_metrics.success_auc, 4)
                entry["precision_auc"] = round(r.accuracy_metrics.precision_auc, 4)
            if r.energy is not None:
                entry["energy_j"] = round(r.energy.total_energy_j, 6)
                entry["energy_per_frame_mj"] = round(r.energy.energy_per_frame_mj, 4)
            # Per-frame arrays enable full success/precision curve plots from JSON.
            if r.ious is not None and len(r.ious) > 0:
                entry["ious"] = [round(float(v), 6) for v in r.ious]
            if r.center_distances is not None and len(r.center_distances) > 0:
                entry["center_distances"] = [round(float(v), 3) for v in r.center_distances]
            sequences.append(entry)
        return {"summary": self.summary(), "sequences": sequences}

    def __str__(self) -> str:
        s = self.summary()
        base = (
            f"BenchmarkResult[{s['tracker']} on {s['dataset']}] "
            f"mIoU={s['mean_iou']}  FPS={s['mean_fps']}  "
            f"mem={s['peak_memory_mb']} MiB  ({s['num_sequences']} sequences)"
        )
        if "total_energy_j" in s:
            base += f"  energy={s['total_energy_j']} J"
        return base

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> Path:
        """Serialise this result to a JSON file at *path*.

        The file can be loaded back with :meth:`load`, fully reconstructing
        per-sequence metrics and profiling summaries.

        Args:
            path: Destination file path.  Parent directories are created
                automatically.  A ``.json`` extension is appended when the
                path has none.

        Returns:
            The resolved :class:`pathlib.Path` that was written.
        """
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".json")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=_json_default)
        return p

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BenchmarkResult":
        """Deserialise a :class:`BenchmarkResult` from a JSON file.

        Reconstructs all per-sequence IoU arrays, profiling summaries,
        accuracy metrics, and (when present) energy estimates from the file
        written by :meth:`save` or :meth:`to_dict`.

        Args:
            path: Path to the JSON file produced by :meth:`save`.

        Returns:
            A fully populated :class:`BenchmarkResult` object.

        Raises:
            FileNotFoundError: If *path* does not exist.
            KeyError: If required keys are absent from the JSON.
        """
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: Dict) -> "BenchmarkResult":
        """Reconstruct a :class:`BenchmarkResult` from its :meth:`to_dict` output.

        Fields that are not stored in the serialised format (e.g. raw
        per-frame latency distributions, ground-truth boxes) are approximated
        from the available scalar summaries.

        Args:
            d: Dict produced by :meth:`to_dict`.

        Returns:
            A :class:`BenchmarkResult` with all stored metrics populated.
        """
        from ..profiling.profiler import ProfilingResult
        from ..profiling.energy import EnergyResult
        from ..metrics.accuracy import AccuracyMetrics

        summary = d["summary"]
        tracker_name: str = summary.get("tracker") or summary.get("tracker_name", "unknown")
        dataset_name: str = summary.get("dataset") or summary.get("dataset_name", "unknown")

        seq_results: List[SequenceResult] = []
        for seq in d.get("sequences", []):
            seq_name: str = seq["sequence_name"]
            ious = np.array(seq.get("ious", [seq.get("mean_iou", 0.0)]), dtype=np.float64)
            dists = (
                np.array(seq["center_distances"], dtype=np.float64)
                if "center_distances" in seq else None
            )
            fps: float = float(seq.get("fps", 0.0))
            lat_ms: float = float(seq.get("mean_latency_ms", 1000.0 / fps if fps > 0 else 0.0))
            mem_mb: float = float(seq.get("peak_memory_mb", 0.0))

            profiling = ProfilingResult(
                tracker_name=tracker_name,
                frame_count=len(ious),
                fps=fps,
                latency_mean_ms=lat_ms,
                latency_std_ms=0.0,
                latency_p95_ms=lat_ms,
                peak_memory_mb=mem_mb,
            )

            accuracy: Optional[AccuracyMetrics] = None
            if "success_auc" in seq:
                accuracy = AccuracyMetrics(
                    mean_iou=float(seq.get("mean_iou", float(ious.mean()) if len(ious) else 0.0)),
                    success_auc=float(seq["success_auc"]),
                    precision_auc=float(seq.get("precision_auc", 0.0)),
                )

            energy: Optional[EnergyResult] = None
            if "energy_j" in seq:
                energy = EnergyResult(
                    tracker_name=tracker_name,
                    frame_count=len(ious),
                    tdp_watts=0.0,
                    total_energy_j=float(seq["energy_j"]),
                    mean_power_w=0.0,
                    energy_per_frame_mj=float(seq.get("energy_per_frame_mj", 0.0)),
                    peak_cpu_pct=0.0,
                    mean_cpu_pct=0.0,
                )

            seq_results.append(
                SequenceResult(
                    sequence_name=seq_name,
                    ious=ious,
                    profiling=profiling,
                    center_distances=dists,
                    energy=energy,
                    accuracy_metrics=accuracy,
                )
            )

        result = cls(tracker_name=tracker_name, dataset_name=dataset_name)
        result.sequence_results = seq_results
        return result


def _json_default(obj):
    """Fallback serialiser for non-standard types inside :meth:`BenchmarkResult.save`."""
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


class BenchmarkEngine:
    """Run a tracker against a dataset and collect accuracy + profiling data.

    Args:
        verbose: Print per-sequence progress to stdout. Default: ``True``.
        tdp_watts: If provided, enables CPU energy estimation using
            :class:`~eovot.profiling.energy.EnergyProfiler` with this TDP
            value (Watts).  Set to the device's CPU TDP for meaningful
            estimates (e.g. ``6.0`` for Raspberry Pi 4, ``15.0`` for a
            laptop).  Default: ``None`` (energy profiling disabled).
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
        """Evaluate *tracker* on every sequence in *dataset*."""
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
                energy_str = ""
                if seq_result.energy is not None:
                    energy_str = f"  E={seq_result.energy.energy_per_frame_mj:.2f}mJ/fr"
                sauc_str = ""
                if seq_result.accuracy_metrics is not None:
                    sauc_str = f"  AUC={seq_result.accuracy_metrics.success_auc:.3f}"
                print(
                    f"  [{idx + 1:>3}/{n}] {seq_result.sequence_name:<30s} "
                    f"mIoU={seq_result.mean_iou:.3f}  "
                    f"FPS={seq_result.profiling.fps:.1f}"
                    f"{sauc_str}"
                    f"{energy_str}"
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

        # Vectorised centre-distance computation (replaces element-wise Python loop).
        dists = self._metrics.batch_center_distance(preds_eval, gt_eval)

        # Full VOT accuracy metrics: success AUC, precision AUC.
        accuracy = self._metrics.compute_all(preds_eval, gt_eval)

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
            accuracy_metrics=accuracy,
        )

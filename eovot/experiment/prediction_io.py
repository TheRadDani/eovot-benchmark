"""VOT-standard prediction export and import for reproducible benchmarking.

Persists and reloads per-frame tracker bounding boxes using the standard VOT
text format: one ``x,y,w,h`` line per frame, ``nan,nan,nan,nan`` for lost
or unavailable frames.  Each sequence is stored as a separate ``.txt`` file
under ``<output_dir>/<tracker_name>/<sequence_name>.txt``.

This supports three core reproducibility workflows:

1. **Share predictions without data** — export bboxes and send them to
   collaborators who can recompute any metric without access to raw video.

2. **Offline metric recomputation** — add a new metric after a long run
   finishes; reload the saved predictions instead of re-running inference.

3. **Cross-tracker spatial comparison** — compare predictions from two
   trackers frame-by-frame without needing ground-truth annotations
   (useful for measuring ensemble agreement or knowledge distillation).

Typical usage::

    from eovot.experiment.prediction_io import PredictionExporter, PredictionLoader

    # --- export after a benchmark run ---
    exporter = PredictionExporter("results/predictions/")
    saved_paths = exporter.save(benchmark_result)      # BenchmarkResult
    # or per-sequence:
    exporter.save_sequence("MOSSE", "Basketball", predictions_array)

    # --- reload for offline analysis ---
    loader = PredictionLoader("results/predictions/MOSSE/")
    preds = loader.load_sequence("Basketball")         # (N, 4) float64
    all_preds = loader.load_all()                      # {name: array}

    # --- compare two trackers without GT ---
    from eovot.experiment.prediction_io import PredictionComparator
    cmp = PredictionComparator(
        "results/predictions/MOSSE/",
        "results/predictions/KCF/",
    )
    shared = cmp.common_sequences()
    agreement_ious = cmp.iou_delta(shared[0])          # spatial agreement
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class PredictionExporter:
    """Export tracker predictions to VOT-compatible per-sequence text files.

    File layout::

        <output_dir>/
            <tracker_name>/
                <sequence_name>.txt   # one 'x,y,w,h' per line

    Lost / invalid frames are written as ``nan,nan,nan,nan``.

    Args:
        output_dir: Root directory.  Sub-directories are created as needed.
    """

    def __init__(self, output_dir: str = "results/predictions/") -> None:
        self.output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, result: "BenchmarkResult") -> Dict[str, Path]:  # type: ignore[name-defined]
        """Persist all per-sequence predictions from a :class:`BenchmarkResult`.

        Args:
            result: Output of :meth:`~eovot.benchmark.engine.BenchmarkEngine.run`.
                Only sequences whose :attr:`~eovot.benchmark.engine.SequenceResult.predictions`
                attribute is not ``None`` are written.

        Returns:
            Mapping of sequence name → :class:`pathlib.Path` of the written file.
            Sequences with no stored predictions are omitted from the mapping.
        """
        tracker_dir = self.output_dir / result.tracker_name
        tracker_dir.mkdir(parents=True, exist_ok=True)

        written: Dict[str, Path] = {}
        for seq_result in result.sequence_results:
            if seq_result.predictions is None:
                continue
            path = _write_sequence(
                tracker_dir,
                seq_result.sequence_name,
                seq_result.predictions,
            )
            written[seq_result.sequence_name] = path
        return written

    def save_sequence(
        self,
        tracker_name: str,
        sequence_name: str,
        predictions: np.ndarray,
    ) -> Path:
        """Write predictions for a single sequence.

        Args:
            tracker_name: Used as the sub-directory name under :attr:`output_dir`.
            sequence_name: Used as the file stem (``<sequence_name>.txt``).
            predictions: ``(N, 4)`` float array of ``(x, y, w, h)`` boxes.
                Rows containing ``nan`` are written as ``nan,nan,nan,nan``.

        Returns:
            :class:`pathlib.Path` of the written file.
        """
        tracker_dir = self.output_dir / tracker_name
        tracker_dir.mkdir(parents=True, exist_ok=True)
        return _write_sequence(tracker_dir, sequence_name, predictions)


class PredictionLoader:
    """Load tracker predictions from VOT-compatible per-sequence text files.

    Args:
        tracker_dir: Directory containing ``*.txt`` prediction files
            (the ``<tracker_name>/`` sub-directory written by
            :class:`PredictionExporter`).
    """

    def __init__(self, tracker_dir: str) -> None:
        self.tracker_dir = Path(tracker_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_sequence(self, sequence_name: str) -> np.ndarray:
        """Load predictions for one sequence.

        Args:
            sequence_name: Sequence identifier (file stem, without ``.txt``).

        Returns:
            ``(N, 4)`` float64 array of ``(x, y, w, h)`` boxes.
            Lost frames are represented as rows of ``nan``.

        Raises:
            FileNotFoundError: If the expected file does not exist.
        """
        path = self.tracker_dir / f"{sequence_name}.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"No predictions found for '{sequence_name}' at {path}"
            )
        return _read_sequence(path)

    def load_all(self) -> Dict[str, np.ndarray]:
        """Load all sequences found in :attr:`tracker_dir`.

        Returns:
            Dict mapping sequence name → ``(N, 4)`` prediction array.
            Returns an empty dict if no ``.txt`` files are present.
        """
        return {
            p.stem: _read_sequence(p)
            for p in sorted(self.tracker_dir.glob("*.txt"))
        }

    def list_sequences(self) -> List[str]:
        """Sorted list of sequence names available in :attr:`tracker_dir`."""
        return sorted(p.stem for p in self.tracker_dir.glob("*.txt"))


class PredictionComparator:
    """Compare predictions from two trackers on the same sequences.

    Designed for offline ablation studies: load predictions from both
    trackers and measure their *spatial agreement* without needing
    ground-truth annotations.  For accuracy deltas (relative to GT),
    pair each prediction file with the dataset annotations externally.

    Args:
        tracker_a_dir: Directory for the first tracker's prediction files.
        tracker_b_dir: Directory for the second tracker's prediction files.
    """

    def __init__(self, tracker_a_dir: str, tracker_b_dir: str) -> None:
        self.loader_a = PredictionLoader(tracker_a_dir)
        self.loader_b = PredictionLoader(tracker_b_dir)

    def common_sequences(self) -> List[str]:
        """Sorted list of sequences available for **both** trackers."""
        seqs_a = set(self.loader_a.list_sequences())
        seqs_b = set(self.loader_b.list_sequences())
        return sorted(seqs_a & seqs_b)

    def spatial_iou(self, sequence_name: str) -> np.ndarray:
        """Per-frame IoU between tracker A and tracker B on one sequence.

        Measures spatial *agreement* between the two prediction sets, not
        accuracy relative to ground truth.  A value of 1.0 means identical
        boxes; 0.0 means the predicted regions do not overlap at all.

        Args:
            sequence_name: A sequence returned by :meth:`common_sequences`.

        Returns:
            ``(N,)`` float64 array of IoU values in ``[0, 1]``.  Length N
            is the minimum of the two sequences' prediction counts.
        """
        preds_a = self.loader_a.load_sequence(sequence_name)
        preds_b = self.loader_b.load_sequence(sequence_name)
        from ..metrics.accuracy import MetricsEngine
        engine = MetricsEngine()
        n = min(len(preds_a), len(preds_b))
        return engine.batch_iou(preds_a[:n], preds_b[:n])

    def mean_spatial_iou(self, sequence_name: str) -> float:
        """Mean per-frame spatial IoU between A and B on one sequence.

        Args:
            sequence_name: A sequence returned by :meth:`common_sequences`.

        Returns:
            Scalar in ``[0, 1]``; higher means the two trackers agree more
            closely in their spatial predictions.
        """
        ious = self.spatial_iou(sequence_name)
        return float(np.nanmean(ious)) if len(ious) else 0.0

    def agreement_summary(self) -> Dict[str, float]:
        """Mean spatial IoU for every sequence common to both trackers.

        Returns:
            Dict mapping sequence name → mean spatial IoU, sorted by name.
        """
        return {
            seq: self.mean_spatial_iou(seq)
            for seq in self.common_sequences()
        }


# ---------------------------------------------------------------------------
# Internal helpers (module-level to allow reuse and testing)
# ---------------------------------------------------------------------------

def _write_sequence(
    directory: Path,
    sequence_name: str,
    predictions: np.ndarray,
) -> Path:
    """Write an ``(N, 4)`` prediction array to ``directory/sequence_name.txt``."""
    path = directory / f"{sequence_name}.txt"
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        for box in predictions:
            if any(_is_nan(v) for v in box):
                writer.writerow(["nan", "nan", "nan", "nan"])
            else:
                writer.writerow([f"{float(v):.6f}" for v in box])
    return path


def _read_sequence(path: Path) -> np.ndarray:
    """Read a VOT-format prediction file into an ``(N, 4)`` float64 array."""
    rows: List[List[float]] = []
    with open(path, newline="") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 4:
                continue
            try:
                vals: List[float] = [float(v) for v in parts]
            except ValueError:
                vals = [float("nan")] * 4
            rows.append(vals)
    if not rows:
        return np.empty((0, 4), dtype=np.float64)
    return np.array(rows, dtype=np.float64)


def _is_nan(value: float) -> bool:
    """Return True for NaN floats (handles both Python float and numpy scalars)."""
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False

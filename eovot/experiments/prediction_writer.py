"""Prediction file writer for EOVOT benchmark results.

Saves per-sequence tracker predictions in formats compatible with the major
Visual Object Tracking evaluation toolkits, enabling offline re-evaluation and
submission to benchmark servers such as GOT-10k and OTB.

Supported formats
-----------------
- **otb** (default) — one bounding box per line, space-delimited: ``x y w h``
- **got10k**        — comma-delimited: ``x,y,w,h`` (GOT-10k server format)
- **vot**           — comma-delimited, 1-indexed, compatible with VOT toolkit

Usage::

    from eovot.experiments.prediction_writer import PredictionWriter

    writer = PredictionWriter(output_dir="results/predictions", fmt="got10k")
    writer.write_result(benchmark_result)        # all sequences at once
    writer.write_sequence("Basketball", preds)   # one sequence manually

Offline re-evaluation::

    # After saving predictions, re-compute metrics without re-running the tracker:
    preds = PredictionWriter.load_sequence("results/predictions/MOSSE/Basketball.txt")
    ious  = MetricsEngine().batch_iou(preds, ground_truth)
"""

from __future__ import annotations

import os
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult


class PredictionFormat(str, Enum):
    """Output format for saved prediction files."""

    OTB = "otb"
    GOT10K = "got10k"
    VOT = "vot"


_FMT_DELIMITER: Dict[PredictionFormat, str] = {
    PredictionFormat.OTB: " ",
    PredictionFormat.GOT10K: ",",
    PredictionFormat.VOT: ",",
}

_FMT_DESCRIPTION: Dict[PredictionFormat, str] = {
    PredictionFormat.OTB: "OTB space-delimited (x y w h)",
    PredictionFormat.GOT10K: "GOT-10k comma-delimited (x,y,w,h)",
    PredictionFormat.VOT: "VOT comma-delimited (x,y,w,h, 1-indexed frames)",
}


class PredictionWriter:
    """Write tracker predictions to disk in a standard format.

    Directory layout::

        <output_dir>/
          <tracker_name>/
            <sequence_name>.txt   ← one bounding box per line

    Args:
        output_dir: Root directory for prediction files.
        fmt: Output format — ``"otb"`` (default), ``"got10k"``, or ``"vot"``.

    Example::

        writer = PredictionWriter("results/preds", fmt="got10k")
        writer.write_result(result)   # BenchmarkResult from engine.run()

        # Or write a single sequence manually:
        writer.write_sequence("CarScale", predictions_array, tracker_name="MOSSE")
    """

    def __init__(
        self,
        output_dir: str,
        fmt: str = "otb",
    ) -> None:
        self.output_dir = output_dir
        try:
            self.fmt = PredictionFormat(fmt.lower())
        except ValueError:
            valid = [f.value for f in PredictionFormat]
            raise ValueError(
                f"Unknown prediction format {fmt!r}. Valid options: {valid}"
            )

    @property
    def delimiter(self) -> str:
        return _FMT_DELIMITER[self.fmt]

    def write_result(self, result: "BenchmarkResult") -> List[str]:
        """Save all per-sequence predictions from *result* to disk.

        Args:
            result: A :class:`~eovot.benchmark.engine.BenchmarkResult` returned
                by :meth:`~eovot.benchmark.engine.BenchmarkEngine.run`.

        Returns:
            List of absolute file paths written (one per sequence that had
            prediction data stored).

        Raises:
            ValueError: If no sequence results contain prediction arrays.
        """
        written: List[str] = []
        for sr in result.sequence_results:
            if sr.predictions is None:
                continue
            path = self.write_sequence(
                sequence_name=sr.sequence_name,
                predictions=sr.predictions,
                tracker_name=result.tracker_name,
            )
            written.append(path)

        if not written:
            raise ValueError(
                "No prediction arrays found in result. "
                "Ensure the BenchmarkEngine stores predictions (default behaviour)."
            )
        return written

    def write_sequence(
        self,
        sequence_name: str,
        predictions: np.ndarray,
        tracker_name: str = "tracker",
    ) -> str:
        """Save predictions for a single sequence.

        Args:
            sequence_name: Used as the file stem (e.g. ``"Basketball"``).
            predictions: Array of shape ``(N, 4)`` — bounding boxes
                ``(x, y, w, h)`` for each of *N* frames.
            tracker_name: Top-level sub-directory name.

        Returns:
            Absolute path to the written prediction file.

        Raises:
            ValueError: If *predictions* is not a 2-D array with 4 columns.
        """
        if predictions.ndim != 2 or predictions.shape[1] != 4:
            raise ValueError(
                f"predictions must be shape (N, 4), got {predictions.shape}"
            )

        out_dir = os.path.join(self.output_dir, tracker_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{sequence_name}.txt")

        with open(out_path, "w", encoding="utf-8") as fh:
            for bbox in predictions:
                line = self.delimiter.join(f"{v:.4f}" for v in bbox)
                fh.write(line + "\n")

        return os.path.abspath(out_path)

    @staticmethod
    def load_sequence(path: str, delimiter: Optional[str] = None) -> np.ndarray:
        """Load a prediction file written by :meth:`write_sequence`.

        Auto-detects the delimiter (comma or space) if *delimiter* is ``None``.

        Args:
            path: Path to a prediction text file.
            delimiter: Override delimiter. ``None`` = auto-detect.

        Returns:
            Array of shape ``(N, 4)`` with ``float64`` bounding boxes.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the file cannot be parsed as an ``(N, 4)`` array.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Prediction file not found: {path}")

        if delimiter is None:
            with open(path, "r", encoding="utf-8") as fh:
                first_line = fh.readline().strip()
            delimiter = "," if "," in first_line else " "

        try:
            arr = np.loadtxt(path, delimiter=delimiter)
        except Exception as exc:
            raise ValueError(f"Cannot parse prediction file {path!r}: {exc}") from exc

        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError(
                f"Expected (N, 4) array from {path!r}, got shape {arr.shape}"
            )
        return arr.astype(np.float64)

    def summary(self, tracker_name: str, sequence_names: Optional[List[str]] = None) -> str:
        """Return a short human-readable summary of saved files.

        Args:
            tracker_name: Tracker sub-directory to inspect.
            sequence_names: If given, check only these sequences; otherwise
                list all ``.txt`` files in the tracker directory.

        Returns:
            Multi-line string suitable for printing to the console.
        """
        tracker_dir = os.path.join(self.output_dir, tracker_name)
        if not os.path.isdir(tracker_dir):
            return f"[PredictionWriter] No files found for tracker '{tracker_name}' in {self.output_dir}"

        if sequence_names is not None:
            files = [os.path.join(tracker_dir, f"{n}.txt") for n in sequence_names]
        else:
            files = sorted(
                os.path.join(tracker_dir, f)
                for f in os.listdir(tracker_dir)
                if f.endswith(".txt")
            )

        lines = [
            f"PredictionWriter — format: {_FMT_DESCRIPTION[self.fmt]}",
            f"Output directory : {os.path.abspath(self.output_dir)}",
            f"Tracker          : {tracker_name}",
            f"Sequences saved  : {len(files)}",
        ]
        for fp in files[:10]:
            exists = os.path.isfile(fp)
            tag = "✓" if exists else "✗ MISSING"
            lines.append(f"  {tag} {os.path.basename(fp)}")
        if len(files) > 10:
            lines.append(f"  … and {len(files) - 10} more")
        return "\n".join(lines)

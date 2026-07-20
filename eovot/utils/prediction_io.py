"""Prediction I/O for standard VOT benchmark formats.

Supports reading and writing tracker bounding-box predictions in the
formats used by the major visual object tracking benchmark suites:

- **OTB** (``"otb"``) — one ``.txt`` file per sequence, one row per frame
  with comma-or-whitespace-separated ``x,y,w,h``.
- **GOT-10k** (``"got10k"``) — same layout as OTB; the GOT-10k toolkit
  expects files named ``<tracker>/<sequence>/00000001.txt`` etc., so this
  writer produces ``<sequence>/prediction.txt`` to match.
- **VOT** (``"vot"``) — one ``.txt`` per sequence, optionally with a
  confidence score column: ``x,y,w,h`` or ``x,y,w,h,conf``.
- **EOVOT-JSON** (``"json"``) — the native EOVOT format produced by
  :meth:`~eovot.benchmark.engine.BenchmarkResult.to_dict`, carrying full
  metadata (tracker name, dataset, per-frame IoU, centre distances, energy).

Motivation
----------
Researchers often want to:

1. **Submit** results to an official benchmark leaderboard (OTB / GOT-10k)
   without sharing the codebase — saving predictions as plain text files
   is the standard submission pathway.
2. **Import** predictions from an external system (another lab's tracker,
   a MATLAB baseline) into the EOVOT metrics engine to produce comparable
   numbers under the same evaluation protocol.
3. **Cache** prediction files so expensive tracker runs do not need to be
   repeated when only the metrics or the report format changes.

Typical usage
-------------
Export predictions from a benchmark run::

    from eovot.utils.prediction_io import PredictionWriter, PredictionFormat
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.mosse import MOSSETracker

    dataset = SyntheticDataset(num_sequences=3, num_frames=50)
    engine  = BenchmarkEngine(verbose=False)
    result  = engine.run(MOSSETracker(), dataset, dataset_name="Synthetic")

    writer = PredictionWriter(output_dir="predictions/", fmt=PredictionFormat.OTB)
    writer.write_benchmark_result(result)

Load predictions back and run the metrics engine::

    from eovot.utils.prediction_io import PredictionReader

    reader = PredictionReader(input_dir="predictions/", fmt=PredictionFormat.OTB)
    loaded = reader.read_all()
    # loaded: Dict[str, np.ndarray]  — {sequence_name: (N, 4) array}

    from eovot.metrics.accuracy import MetricsEngine
    from eovot.datasets.synthetic import SyntheticDataset

    dataset  = SyntheticDataset(num_sequences=3, num_frames=50)
    metrics  = MetricsEngine()
    for seq in dataset:
        preds = loaded[seq.name]
        result = metrics.compute_all(preds, seq.ground_truth)
        print(seq.name, result)
"""

from __future__ import annotations

import csv
import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]


class PredictionFormat(str, Enum):
    """Supported prediction file formats."""

    OTB = "otb"
    """OTB-style: one sequence directory, ``predictions.txt`` inside."""

    GOT10K = "got10k"
    """GOT-10k-style: one sequence directory, ``prediction.txt`` inside."""

    VOT = "vot"
    """VOT-style: same as OTB/GOT-10k; optionally includes a confidence column."""

    JSON = "json"
    """EOVOT native JSON — preserves all metadata from BenchmarkResult."""


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class PredictionWriter:
    """Writes tracker predictions to disk in a chosen format.

    Each sequence's predictions are saved as a separate file inside a
    per-tracker sub-directory of ``output_dir``.

    Directory layout produced by :meth:`write_benchmark_result`:

    - OTB / GOT-10k / VOT::

        <output_dir>/<tracker_name>/<sequence_name>/predictions.txt

    - JSON::

        <output_dir>/<tracker_name>/<sequence_name>.json

    The JSON format also writes a combined ``<tracker_name>.json`` file
    with the full benchmark result dict.

    Args:
        output_dir: Root directory to write into.  Created if absent.
        fmt:        Output format.  Default: :attr:`PredictionFormat.OTB`.
        delimiter:  Column separator for text formats.  Default: ``","`` (CSV).
        include_confidence: VOT format only — append a ``1.0`` confidence
            column so the file is compatible with VOT toolkit parsers that
            expect it.  Default: ``False``.
    """

    def __init__(
        self,
        output_dir: str,
        fmt: PredictionFormat = PredictionFormat.OTB,
        delimiter: str = ",",
        include_confidence: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.fmt = PredictionFormat(fmt)
        self.delimiter = delimiter
        self.include_confidence = include_confidence
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # High-level write from BenchmarkResult dict
    # ------------------------------------------------------------------

    def write_benchmark_result(self, result: Any) -> Dict[str, Path]:
        """Write all sequence predictions from a :class:`~eovot.benchmark.engine.BenchmarkResult`.

        Accepts either a live ``BenchmarkResult`` object (preferred — has
        access to raw ``predictions`` arrays on each ``SequenceResult``) or
        its ``to_dict()`` serialisation (only sequences whose dict entry
        contains a ``"predictions"`` key will be exported; IoU-only entries
        are skipped because bounding boxes cannot be reconstructed from IoU).

        Args:
            result: A ``BenchmarkResult`` instance or its dict equivalent.

        Returns:
            ``{sequence_name: path}`` mapping of written files.
        """
        if hasattr(result, "sequence_results"):
            return self._write_from_live_result(result)

        # Dict path — only works when predictions were explicitly included.
        result_dict = result
        tracker_name = result_dict.get("summary", {}).get("tracker", "unknown")
        paths: Dict[str, Path] = {}
        for seq in result_dict.get("sequences", []):
            seq_name = seq.get("sequence_name", "unknown")
            if "predictions" not in seq or not seq["predictions"]:
                continue
            bbox_array = np.array(seq["predictions"], dtype=np.float64)
            if self.fmt == PredictionFormat.JSON:
                p = self._write_sequence_json(tracker_name, seq_name, seq)
            else:
                p = self._write_sequence_text(tracker_name, seq_name, bbox_array)
            paths[seq_name] = p

        if self.fmt == PredictionFormat.JSON:
            tracker_dir = self._tracker_dir(tracker_name)
            tracker_dir.mkdir(parents=True, exist_ok=True)
            full_path = tracker_dir / "_benchmark_result.json"
            with open(full_path, "w", encoding="utf-8") as fh:
                json.dump(result_dict, fh, indent=2, default=_json_default)

        return paths

    def _write_from_live_result(self, result: Any) -> Dict[str, Path]:
        """Write predictions from a live ``BenchmarkResult`` object."""
        tracker_name = result.tracker_name
        result_dict = result.to_dict()
        paths: Dict[str, Path] = {}

        for sr in result.sequence_results:
            if sr.predictions is None or len(sr.predictions) == 0:
                continue
            bbox_array = np.asarray(sr.predictions, dtype=np.float64)
            seq_name = sr.sequence_name

            # Build a per-sequence dict that includes predictions for JSON export.
            seq_dict = next(
                (s for s in result_dict.get("sequences", [])
                 if s.get("sequence_name") == seq_name),
                {"sequence_name": seq_name},
            )
            seq_dict = dict(seq_dict)
            seq_dict["predictions"] = bbox_array.tolist()

            if self.fmt == PredictionFormat.JSON:
                p = self._write_sequence_json(tracker_name, seq_name, seq_dict)
            else:
                p = self._write_sequence_text(tracker_name, seq_name, bbox_array)
            paths[seq_name] = p

        if self.fmt == PredictionFormat.JSON:
            tracker_dir = self._tracker_dir(tracker_name)
            tracker_dir.mkdir(parents=True, exist_ok=True)
            full_path = tracker_dir / "_benchmark_result.json"
            with open(full_path, "w", encoding="utf-8") as fh:
                json.dump(result_dict, fh, indent=2, default=_json_default)

        return paths

    def write_sequence(
        self,
        tracker_name: str,
        sequence_name: str,
        predictions: np.ndarray,
        confidence: Optional[np.ndarray] = None,
    ) -> Path:
        """Write predictions for a single sequence.

        Args:
            tracker_name:  Identifier for the tracker (used as sub-directory).
            sequence_name: Sequence name.
            predictions:   ``(N, 4)`` array of ``(x, y, w, h)`` boxes.
            confidence:    Optional ``(N,)`` confidence scores (VOT format).

        Returns:
            Path of the written file.
        """
        if self.fmt == PredictionFormat.JSON:
            return self._write_sequence_json(tracker_name, sequence_name, {
                "sequence_name": sequence_name,
                "predictions": predictions.tolist(),
                "confidence": confidence.tolist() if confidence is not None else None,
            })
        return self._write_sequence_text(
            tracker_name, sequence_name, predictions, confidence
        )

    # ------------------------------------------------------------------
    # Format-specific write helpers
    # ------------------------------------------------------------------

    def _write_sequence_text(
        self,
        tracker_name: str,
        sequence_name: str,
        predictions: np.ndarray,
        confidence: Optional[np.ndarray] = None,
    ) -> Path:
        seq_dir = self._seq_dir(tracker_name, sequence_name)
        seq_dir.mkdir(parents=True, exist_ok=True)
        filename = "prediction.txt" if self.fmt == PredictionFormat.GOT10K else "predictions.txt"
        path = seq_dir / filename

        with open(path, "w", newline="") as fh:
            for i, row in enumerate(predictions):
                x, y, w, h = [float(v) for v in row]
                line_parts = [f"{x:.4f}", f"{y:.4f}", f"{w:.4f}", f"{h:.4f}"]
                if self.fmt == PredictionFormat.VOT and self.include_confidence:
                    conf = float(confidence[i]) if confidence is not None else 1.0
                    line_parts.append(f"{conf:.6f}")
                fh.write(self.delimiter.join(line_parts) + "\n")

        return path

    def _write_sequence_json(
        self,
        tracker_name: str,
        sequence_name: str,
        seq_data: Dict[str, Any],
    ) -> Path:
        tracker_dir = self._tracker_dir(tracker_name)
        tracker_dir.mkdir(parents=True, exist_ok=True)
        path = tracker_dir / f"{sequence_name}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(seq_data, fh, indent=2, default=_json_default)
        return path

    def _tracker_dir(self, tracker_name: str) -> Path:
        return self.output_dir / _sanitize(tracker_name)

    def _seq_dir(self, tracker_name: str, sequence_name: str) -> Path:
        return self._tracker_dir(tracker_name) / _sanitize(sequence_name)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class PredictionReader:
    """Reads saved tracker predictions from disk.

    Mirrors the layout produced by :class:`PredictionWriter` so that saved
    predictions can be loaded back for offline metric computation without
    re-running the tracker.

    Args:
        input_dir:  Directory to read from.
        fmt:        Format to parse.  Default: :attr:`PredictionFormat.OTB`.
        delimiter:  Column separator for text formats.  ``None`` means
                    whitespace-split (handles both tabs and spaces).
    """

    def __init__(
        self,
        input_dir: str,
        fmt: PredictionFormat = PredictionFormat.OTB,
        delimiter: Optional[str] = None,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.fmt = PredictionFormat(fmt)
        self.delimiter = delimiter

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def read_tracker(
        self, tracker_name: str
    ) -> Dict[str, np.ndarray]:
        """Load all sequences for one tracker.

        Args:
            tracker_name: Tracker name (matches the sub-directory written by
                :class:`PredictionWriter`).

        Returns:
            ``{sequence_name: (N, 4) float64 array}`` mapping.
        """
        if self.fmt == PredictionFormat.JSON:
            return self._read_tracker_json(tracker_name)

        tracker_dir = self.input_dir / _sanitize(tracker_name)
        if not tracker_dir.is_dir():
            raise FileNotFoundError(
                f"Tracker directory not found: {tracker_dir}"
            )

        result: Dict[str, np.ndarray] = {}
        for seq_dir in sorted(tracker_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            preds = self._read_sequence_text(seq_dir)
            if preds is not None:
                result[seq_dir.name] = preds
        return result

    def read_sequence(
        self, tracker_name: str, sequence_name: str
    ) -> np.ndarray:
        """Load predictions for a single sequence.

        Args:
            tracker_name:  Tracker sub-directory name.
            sequence_name: Sequence sub-sub-directory name.

        Returns:
            ``(N, 4)`` float64 array of ``(x, y, w, h)`` boxes.

        Raises:
            FileNotFoundError: If the prediction file does not exist.
        """
        if self.fmt == PredictionFormat.JSON:
            path = (
                self.input_dir
                / _sanitize(tracker_name)
                / f"{_sanitize(sequence_name)}.json"
            )
            if not path.exists():
                raise FileNotFoundError(f"JSON prediction file not found: {path}")
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            return np.array(data.get("predictions", []), dtype=np.float64)

        seq_dir = (
            self.input_dir / _sanitize(tracker_name) / _sanitize(sequence_name)
        )
        result = self._read_sequence_text(seq_dir)
        if result is None:
            raise FileNotFoundError(
                f"No prediction file found in: {seq_dir}"
            )
        return result

    def read_all(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Read all trackers from ``input_dir``.

        Returns:
            ``{tracker_name: {sequence_name: (N, 4) array}}`` nested dict.
        """
        result: Dict[str, Dict[str, np.ndarray]] = {}
        for entry in sorted(self.input_dir.iterdir()):
            if not entry.is_dir():
                continue
            try:
                seqs = self.read_tracker(entry.name)
                if seqs:
                    result[entry.name] = seqs
            except (FileNotFoundError, ValueError):
                continue
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_sequence_text(self, seq_dir: Path) -> Optional[np.ndarray]:
        """Try to find and parse the text prediction file in *seq_dir*."""
        for fname in ("prediction.txt", "predictions.txt"):
            path = seq_dir / fname
            if path.exists():
                return self._parse_text_file(path)
        return None

    def _parse_text_file(self, path: Path) -> np.ndarray:
        """Parse an OTB/GOT-10k/VOT text prediction file.

        Handles comma, tab, and space delimiters automatically.
        Keeps only the first four columns (x, y, w, h), ignoring any
        confidence or additional fields.
        """
        rows: List[List[float]] = []
        with open(path, newline="") as fh:
            content = fh.read().strip()
        if not content:
            return np.empty((0, 4), dtype=np.float64)

        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            if self.delimiter is not None:
                parts = line.split(self.delimiter)
            else:
                # Auto-detect: try comma first, then whitespace.
                parts = line.split(",") if "," in line else line.split()
            try:
                xywh = [float(p.strip()) for p in parts[:4]]
                if len(xywh) == 4:
                    rows.append(xywh)
            except ValueError:
                continue  # skip header or malformed rows

        return np.array(rows, dtype=np.float64) if rows else np.empty((0, 4), dtype=np.float64)

    def _read_tracker_json(self, tracker_name: str) -> Dict[str, np.ndarray]:
        tracker_dir = self.input_dir / _sanitize(tracker_name)
        if not tracker_dir.is_dir():
            raise FileNotFoundError(f"Tracker directory not found: {tracker_dir}")

        result: Dict[str, np.ndarray] = {}

        # Individual per-sequence JSON files take priority because they carry
        # the full predictions array even when _benchmark_result.json does not.
        for path in sorted(tracker_dir.glob("*.json")):
            if path.name.startswith("_"):
                continue
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            preds = data.get("predictions")
            if preds:
                result[path.stem] = np.array(preds, dtype=np.float64)

        if result:
            return result

        # Fall back to the combined benchmark result JSON (may lack predictions).
        full_path = tracker_dir / "_benchmark_result.json"
        if full_path.exists():
            with open(full_path, encoding="utf-8") as fh:
                data = json.load(fh)
            for seq in data.get("sequences", []):
                name = seq.get("sequence_name", "unknown")
                preds = seq.get("predictions")
                if preds:
                    result[name] = np.array(preds, dtype=np.float64)

        return result


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def load_predictions_from_benchmark_result(
    result: Any,
) -> Dict[str, np.ndarray]:
    """Extract per-sequence prediction arrays from a :class:`~eovot.benchmark.engine.BenchmarkResult`.

    Accepts both a live ``BenchmarkResult`` object (with
    :attr:`sequence_results` populated) and a dict from
    :meth:`~eovot.benchmark.engine.BenchmarkResult.to_dict`.

    Args:
        result: ``BenchmarkResult`` or its dict equivalent.

    Returns:
        ``{sequence_name: (N, 4) float64 array}`` mapping.  Sequences
        without stored prediction arrays are silently omitted.
    """
    if hasattr(result, "sequence_results"):
        out: Dict[str, np.ndarray] = {}
        for sr in result.sequence_results:
            if sr.predictions is not None and len(sr.predictions) > 0:
                out[sr.sequence_name] = np.asarray(sr.predictions, dtype=np.float64)
        return out

    # Dict form
    out = {}
    for seq in result.get("sequences", []):
        preds = seq.get("predictions")
        if preds:
            out[seq["sequence_name"]] = np.array(preds, dtype=np.float64)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize(name: str) -> str:
    """Replace filesystem-unsafe characters in a tracker or sequence name."""
    for ch in r'\/:*?"<>|()':
        name = name.replace(ch, "_")
    return name.strip()


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)

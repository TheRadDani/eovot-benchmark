"""ResultsBank — persistent storage and comparison of benchmark results.

A :class:`ResultsBank` manages a directory of serialised
:class:`~eovot.benchmark.engine.BenchmarkResult` files, enabling an
offline analysis workflow that separates data collection from analysis:

1. Run benchmarks on any machine and call ``bank.save(result)`` after each.
2. Later (different session, different machine) call ``bank.load_all()`` or
   ``bank.compare()`` to generate comparison tables without re-running.

Directory layout::

    <bank_dir>/
        MOSSE_OTB100_20240115T143022.json
        KCF_OTB100_20240115T143510.json
        CSRT_OTB100_20240115T144001.json
        index.json                          ← summary index, updated on every save

The index is a lightweight catalogue of the saved files, enabling fast
listing and querying without loading every full result file.

Example::

    from eovot.results.bank import ResultsBank
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kcf import KCFTracker
    from eovot.datasets.synthetic import SyntheticDataset

    bank   = ResultsBank("results/bank")
    engine = BenchmarkEngine(verbose=False)
    dataset = SyntheticDataset(num_sequences=3)

    for tracker in [MOSSETracker(), KCFTracker()]:
        result = engine.run(tracker, dataset, dataset_name="Synthetic")
        bank.save(result)

    # Later — compare without re-running
    print(bank.compare())

    # Load one result back for deeper analysis
    result = bank.load("MOSSE_Synthetic_*")
"""

from __future__ import annotations

import fnmatch
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult


# ---------------------------------------------------------------------------
# Index entry
# ---------------------------------------------------------------------------

class _IndexEntry:
    """Lightweight summary of one saved result (no per-frame arrays)."""

    def __init__(
        self,
        filename: str,
        tracker: str,
        dataset: str,
        mean_iou: float,
        mean_fps: float,
        peak_memory_mb: float,
        num_sequences: int,
        timestamp: str,
        success_auc: Optional[float] = None,
        precision_auc: Optional[float] = None,
    ) -> None:
        self.filename = filename
        self.tracker = tracker
        self.dataset = dataset
        self.mean_iou = mean_iou
        self.mean_fps = mean_fps
        self.peak_memory_mb = peak_memory_mb
        self.num_sequences = num_sequences
        self.timestamp = timestamp
        self.success_auc = success_auc
        self.precision_auc = precision_auc

    def to_dict(self) -> Dict:
        d = {
            "filename": self.filename,
            "tracker": self.tracker,
            "dataset": self.dataset,
            "mean_iou": self.mean_iou,
            "mean_fps": self.mean_fps,
            "peak_memory_mb": self.peak_memory_mb,
            "num_sequences": self.num_sequences,
            "timestamp": self.timestamp,
        }
        if self.success_auc is not None:
            d["success_auc"] = self.success_auc
        if self.precision_auc is not None:
            d["precision_auc"] = self.precision_auc
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "_IndexEntry":
        return cls(
            filename=d["filename"],
            tracker=d["tracker"],
            dataset=d["dataset"],
            mean_iou=d["mean_iou"],
            mean_fps=d["mean_fps"],
            peak_memory_mb=d["peak_memory_mb"],
            num_sequences=d["num_sequences"],
            timestamp=d["timestamp"],
            success_auc=d.get("success_auc"),
            precision_auc=d.get("precision_auc"),
        )

    @classmethod
    def from_summary(cls, summary: Dict, filename: str, timestamp: str) -> "_IndexEntry":
        return cls(
            filename=filename,
            tracker=summary.get("tracker") or summary.get("tracker_name", "unknown"),
            dataset=summary.get("dataset") or summary.get("dataset_name", "unknown"),
            mean_iou=float(summary.get("mean_iou", 0.0)),
            mean_fps=float(summary.get("mean_fps", 0.0)),
            peak_memory_mb=float(summary.get("peak_memory_mb", 0.0)),
            num_sequences=int(summary.get("num_sequences", 0)),
            timestamp=timestamp,
            success_auc=float(summary["success_auc"]) if "success_auc" in summary else None,
            precision_auc=float(summary["precision_auc"]) if "precision_auc" in summary else None,
        )


# ---------------------------------------------------------------------------
# ResultsBank
# ---------------------------------------------------------------------------

class ResultsBank:
    """Manage a directory of persisted :class:`~eovot.benchmark.engine.BenchmarkResult` files.

    Args:
        directory: Path to the bank directory.  Created automatically if it
            does not exist.

    Example::

        bank = ResultsBank("results/bank")
        bank.save(benchmark_result)
        print(bank.list_results())
        print(bank.compare())
    """

    _INDEX_FILE = "index.json"

    def __init__(self, directory: Union[str, Path]) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        result: "BenchmarkResult",
        name: Optional[str] = None,
    ) -> Path:
        """Persist *result* to the bank and update the index.

        Args:
            result: :class:`~eovot.benchmark.engine.BenchmarkResult` to save.
            name: Optional base filename (without extension).  If ``None``,
                a name is generated from the tracker name, dataset name, and
                a UTC timestamp: ``{tracker}_{dataset}_{timestamp}.json``.

        Returns:
            Path to the written JSON file.
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        if name is None:
            safe_tracker = _safe_name(result.tracker_name)
            safe_dataset = _safe_name(result.dataset_name)
            name = f"{safe_tracker}_{safe_dataset}_{ts}"
        path = result.save(self.directory / name)
        self._update_index(result.summary(), filename=path.name, timestamp=ts)
        return path

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, name_or_pattern: str) -> "BenchmarkResult":
        """Load one result by filename (stem) or glob pattern.

        Args:
            name_or_pattern: Exact filename stem (without ``.json``) or a
                ``fnmatch``-style glob (e.g. ``"MOSSE_Synthetic_*"``).  The
                first match is returned when multiple files match.

        Returns:
            The deserialised :class:`~eovot.benchmark.engine.BenchmarkResult`.

        Raises:
            FileNotFoundError: If no matching result is found.
        """
        from ..benchmark.engine import BenchmarkResult

        target = self._resolve(name_or_pattern)
        return BenchmarkResult.load(target)

    def load_all(self, dataset: Optional[str] = None) -> List["BenchmarkResult"]:
        """Load all saved results, optionally filtered by dataset name.

        Args:
            dataset: If given, only return results whose dataset name matches
                (case-insensitive substring match).

        Returns:
            List of :class:`~eovot.benchmark.engine.BenchmarkResult` objects,
            sorted by mean IoU (highest first).
        """
        from ..benchmark.engine import BenchmarkResult

        entries = self._read_index()
        if dataset is not None:
            entries = [e for e in entries if dataset.lower() in e.dataset.lower()]

        results = []
        for entry in entries:
            path = self.directory / entry.filename
            if path.exists():
                results.append(BenchmarkResult.load(path))
        results.sort(key=lambda r: r.mean_iou, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Listing and comparison
    # ------------------------------------------------------------------

    def list_results(self, dataset: Optional[str] = None) -> List[Dict]:
        """Return a list of index entries (lightweight summaries) as plain dicts.

        Args:
            dataset: Optional case-insensitive substring filter on dataset name.

        Returns:
            List of summary dicts, sorted by mean IoU (highest first).
        """
        entries = self._read_index()
        if dataset is not None:
            entries = [e for e in entries if dataset.lower() in e.dataset.lower()]
        entries.sort(key=lambda e: e.mean_iou, reverse=True)
        return [e.to_dict() for e in entries]

    def compare(
        self,
        dataset: Optional[str] = None,
        sort_by: str = "mean_iou",
    ) -> str:
        """Generate a Markdown comparison table from all saved results.

        Args:
            dataset: Optional dataset filter (case-insensitive substring).
            sort_by: Column to sort by.  One of ``"mean_iou"``,
                ``"mean_fps"``, ``"success_auc"``, ``"precision_auc"``.
                Default: ``"mean_iou"``.

        Returns:
            Multi-line Markdown table string, or a message when the bank is empty.
        """
        entries = self._read_index()
        if dataset is not None:
            entries = [e for e in entries if dataset.lower() in e.dataset.lower()]

        if not entries:
            return "_No results in bank" + (f" for dataset '{dataset}'" if dataset else "") + "._"

        _valid_sort = {"mean_iou", "mean_fps", "success_auc", "precision_auc"}
        if sort_by not in _valid_sort:
            sort_by = "mean_iou"

        def _sort_key(e: _IndexEntry) -> float:
            return float(getattr(e, sort_by) or 0.0)

        entries.sort(key=_sort_key, reverse=True)

        has_auc = any(e.success_auc is not None for e in entries)

        if has_auc:
            header = (
                "| Rank | Tracker | Dataset | mIoU | Success AUC | Precision AUC "
                "| FPS | Mem (MB) | Saved At |\n"
                "|------|---------|---------|-----:|------------:|--------------:"
                "|----:|---------:|----------|\n"
            )
            rows = []
            for rank, e in enumerate(entries, 1):
                sauc = f"{e.success_auc:.4f}" if e.success_auc is not None else "—"
                pauc = f"{e.precision_auc:.4f}" if e.precision_auc is not None else "—"
                rows.append(
                    f"| {rank} | {e.tracker} | {e.dataset} | {e.mean_iou:.4f} "
                    f"| {sauc} | {pauc} "
                    f"| {e.mean_fps:.1f} | {e.peak_memory_mb:.1f} | {e.timestamp} |"
                )
        else:
            header = (
                "| Rank | Tracker | Dataset | mIoU | FPS | Mem (MB) | Saved At |\n"
                "|------|---------|---------|-----:|----:|---------:|----------|\n"
            )
            rows = [
                f"| {rank} | {e.tracker} | {e.dataset} | {e.mean_iou:.4f} "
                f"| {e.mean_fps:.1f} | {e.peak_memory_mb:.1f} | {e.timestamp} |"
                for rank, e in enumerate(entries, 1)
            ]

        return header + "\n".join(rows)

    def delete(self, name_or_pattern: str) -> int:
        """Delete one or more results matching *name_or_pattern*.

        Args:
            name_or_pattern: Exact filename stem or glob pattern.

        Returns:
            Number of files deleted.
        """
        matches = self._glob(name_or_pattern)
        deleted = 0
        for path in matches:
            path.unlink(missing_ok=True)
            deleted += 1
        if deleted:
            self._rebuild_index()
        return deleted

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _index_path(self) -> Path:
        return self.directory / self._INDEX_FILE

    def _read_index(self) -> List[_IndexEntry]:
        p = self._index_path()
        if not p.exists():
            self._rebuild_index()
        try:
            with open(p, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            return [_IndexEntry.from_dict(e) for e in raw]
        except (json.JSONDecodeError, KeyError):
            self._rebuild_index()
            with open(p, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            return [_IndexEntry.from_dict(e) for e in raw]

    def _write_index(self, entries: List[_IndexEntry]) -> None:
        with open(self._index_path(), "w", encoding="utf-8") as fh:
            json.dump([e.to_dict() for e in entries], fh, indent=2)

    def _update_index(self, summary: Dict, filename: str, timestamp: str) -> None:
        entries = self._read_index()
        # Remove any existing entry for the same file.
        entries = [e for e in entries if e.filename != filename]
        entries.append(_IndexEntry.from_summary(summary, filename, timestamp))
        self._write_index(entries)

    def _rebuild_index(self) -> None:
        """Scan the bank directory and rebuild the index from JSON files."""
        entries = []
        for p in sorted(self.directory.glob("*.json")):
            if p.name == self._INDEX_FILE:
                continue
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    d = json.load(fh)
                ts = p.stem.rsplit("_", 1)[-1] if "_" in p.stem else "unknown"
                entries.append(_IndexEntry.from_summary(d.get("summary", {}), p.name, ts))
            except Exception:
                pass
        self._write_index(entries)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _glob(self, pattern: str) -> List[Path]:
        stem_pattern = pattern if "*" in pattern or "?" in pattern else pattern
        matches = [
            p for p in self.directory.glob("*.json")
            if p.name != self._INDEX_FILE
            and fnmatch.fnmatch(p.stem, stem_pattern)
        ]
        if not matches:
            # Try exact stem match without wildcard
            exact = self.directory / f"{pattern}.json"
            if exact.exists():
                matches = [exact]
        return matches

    def _resolve(self, name_or_pattern: str) -> Path:
        matches = self._glob(name_or_pattern)
        if not matches:
            raise FileNotFoundError(
                f"No result matching '{name_or_pattern}' found in {self.directory}"
            )
        # Return the most recently modified match
        return max(matches, key=lambda p: p.stat().st_mtime)


def _safe_name(s: str) -> str:
    """Convert an arbitrary string to a filesystem-safe identifier."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)

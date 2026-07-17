"""Integration tests for attribute-aware benchmarking.

Verifies that AttributeAnalyzer integrates correctly with BenchmarkResult
and that BenchmarkReporter.save_attribute_markdown produces valid output.

No real dataset files are needed — sequences are generated in memory.
"""

from __future__ import annotations

import tempfile
from typing import Iterator

import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkEngine, BenchmarkResult
from eovot.datasets.base import BaseDataset, Sequence
from eovot.metrics.attributes import ALL_ATTRIBUTES, AttributeAnalyzer, AttributeDetector
from eovot.reporting.reporter import BenchmarkReporter
from eovot.trackers.base import BaseTracker

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NUM_FRAMES = 30
FIXED_BOX = (10.0, 10.0, 50.0, 50.0)


class ConstantTracker(BaseTracker):
    def __init__(self) -> None:
        super().__init__(name="ConstantTracker")

    def initialize(self, frame: np.ndarray, bbox) -> None:
        pass

    def update(self, frame: np.ndarray):
        return FIXED_BOX


class SyntheticSeq(Sequence):
    def __init__(self, name: str, n_frames: int, gt: np.ndarray) -> None:
        super().__init__(
            name=name,
            frame_paths=[f"f{i}.jpg" for i in range(n_frames)],
            ground_truth=gt,
        )
        self._n = n_frames

    def __iter__(self) -> Iterator[np.ndarray]:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for _ in range(self._n):
            yield frame


class SyntheticDS(BaseDataset):
    def __init__(self, n: int = 3) -> None:
        gt = np.tile(np.array(FIXED_BOX), (NUM_FRAMES, 1)).astype(np.float64)
        self._seqs = [SyntheticSeq(f"seq_{i:02d}", NUM_FRAMES, gt.copy()) for i in range(n)]

    def __len__(self) -> int:
        return len(self._seqs)

    def __getitem__(self, idx: int) -> Sequence:
        return self._seqs[idx]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_attribute_breakdown_returns_table():
    """BenchmarkResult.attribute_breakdown() returns an AttributePerformanceTable."""
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(ConstantTracker(), SyntheticDS(3), dataset_name="Synthetic")

    table = result.attribute_breakdown()

    assert table.tracker_name == "ConstantTracker"
    assert table.dataset_name == "Synthetic"
    # entries may be empty for trivial synthetic sequences (no attributes fire),
    # but the method must complete without error
    assert isinstance(table.entries, dict)


def test_attribute_breakdown_markdown_renders():
    """to_markdown() produces a non-empty string regardless of attribute coverage."""
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(ConstantTracker(), SyntheticDS(3), dataset_name="Synthetic")
    table = result.attribute_breakdown()
    md = table.to_markdown()
    assert isinstance(md, str)
    assert len(md) > 0


def test_attribute_breakdown_consistency():
    """attribute_breakdown() and AttributeAnalyzer().breakdown() return equal results."""
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(ConstantTracker(), SyntheticDS(4), dataset_name="Synthetic")

    via_method = result.attribute_breakdown()
    via_analyzer = AttributeAnalyzer().breakdown(result)

    assert via_method.tracker_name == via_analyzer.tracker_name
    assert via_method.entries == via_analyzer.entries


def test_attribute_breakdown_fast_motion_detected():
    """Fast-motion attribute is detected when GT boxes move rapidly."""
    # Create a sequence with large frame-to-frame displacements.
    n = 20
    gt = np.zeros((n, 4), dtype=np.float64)
    gt[:, 2] = 50.0  # width
    gt[:, 3] = 50.0  # height
    # Move the box 200px per frame — far beyond the 20% diagonal threshold.
    for i in range(n):
        gt[i, 0] = float(i * 200)
        gt[i, 1] = float(i * 200)

    detector = AttributeDetector()
    attrs = detector.detect(gt, "fast_seq")
    assert attrs.has("fast_motion"), "fast_motion should be detected for large displacements"


def test_attribute_breakdown_scale_variation_detected():
    """Scale-variation attribute is detected when GT box area changes significantly."""
    n = 20
    gt = np.zeros((n, 4), dtype=np.float64)
    # Start small, grow large: area ratio >> 4
    for i in range(n):
        size = 5.0 + i * 20.0
        gt[i, 2] = size
        gt[i, 3] = size

    detector = AttributeDetector()
    attrs = detector.detect(gt, "scale_seq")
    assert attrs.has("scale_variation"), "scale_variation should be detected for large area changes"


def test_save_attribute_markdown_writes_file():
    """BenchmarkReporter.save_attribute_markdown() writes a .md file to disk."""
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(ConstantTracker(), SyntheticDS(3), dataset_name="Synthetic")

    with tempfile.TemporaryDirectory() as tmpdir:
        reporter = BenchmarkReporter(output_dir=tmpdir)
        path = reporter.save_attribute_markdown(result, name="test-result")

        assert path.exists(), "Attribute markdown file was not written"
        content = path.read_text()
        assert len(content) > 0, "Attribute markdown file is empty"


def test_attribute_table_entries_are_valid():
    """All entries in the table have required keys with valid numeric values."""
    n = 10
    gt = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        size = 5.0 + i * 30.0
        gt[i, 2] = size
        gt[i, 3] = size

    seq = SyntheticSeq("scale_seq", n, gt)

    class DS(BaseDataset):
        def __len__(self):
            return 1
        def __getitem__(self, idx):
            return seq

    engine = BenchmarkEngine(verbose=False)
    result = engine.run(ConstantTracker(), DS(), dataset_name="DS")
    table = result.attribute_breakdown()

    for attr, entry in table.entries.items():
        assert attr in ALL_ATTRIBUTES
        assert "n_sequences" in entry
        assert "mean_iou" in entry
        assert 0 <= entry["mean_iou"] <= 1.0
        assert entry["n_sequences"] >= 1

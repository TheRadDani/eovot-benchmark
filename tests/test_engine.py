"""Integration tests for BenchmarkEngine using a synthetic dataset.

No real dataset files are needed — sequences are generated entirely in memory
using a dummy tracker and a mock dataset.
"""

from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkEngine, BenchmarkResult, SequenceResult
from eovot.datasets.base import BaseDataset, Sequence
from eovot.trackers.base import BaseTracker

BBox = Tuple[float, float, float, float]

# ---------------------------------------------------------------------------
# Helpers — synthetic tracker and dataset
# ---------------------------------------------------------------------------

NUM_FRAMES = 20
FIXED_BOX: BBox = (10.0, 10.0, 50.0, 50.0)


class ConstantTracker(BaseTracker):
    """Tracker that always returns a fixed bounding box regardless of input."""

    def __init__(self, box: BBox = FIXED_BOX) -> None:
        self._box = box

    @property
    def name(self) -> str:
        return "ConstantTracker"

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        pass  # nothing to initialise

    def update(self, frame: np.ndarray) -> BBox:
        return self._box


class SyntheticSequence(Sequence):
    """An in-memory sequence that yields black frames of a fixed size."""

    def __init__(self, name: str, n_frames: int, gt_box: BBox) -> None:
        gt = np.tile(np.array(gt_box), (n_frames, 1))
        # Provide dummy frame_paths; we override __iter__ to avoid file I/O.
        super().__init__(
            name=name,
            frame_paths=[f"frame_{i:04d}.jpg" for i in range(n_frames)],
            ground_truth=gt,
        )
        self._n_frames = n_frames

    def __iter__(self) -> Iterator[np.ndarray]:  # type: ignore[override]
        """Yield black BGR frames (no disk I/O)."""
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for _ in range(self._n_frames):
            yield frame


class SyntheticDataset(BaseDataset):
    """Small synthetic dataset of identical sequences."""

    def __init__(self, n_sequences: int = 3) -> None:
        self._seqs = [
            SyntheticSequence(f"seq_{i:02d}", NUM_FRAMES, FIXED_BOX)
            for i in range(n_sequences)
        ]

    def __len__(self) -> int:
        return len(self._seqs)

    def __getitem__(self, idx: int) -> Sequence:
        return self._seqs[idx]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBenchmarkEngine:
    def setup_method(self):
        self.engine = BenchmarkEngine(verbose=False)
        self.tracker = ConstantTracker(FIXED_BOX)
        self.dataset = SyntheticDataset(n_sequences=3)

    def test_run_returns_benchmark_result(self):
        result = self.engine.run(self.tracker, self.dataset, dataset_name="Synthetic")
        assert isinstance(result, BenchmarkResult)

    def test_correct_number_of_sequences(self):
        result = self.engine.run(self.tracker, self.dataset, dataset_name="Synthetic")
        assert len(result.sequence_results) == 3

    def test_max_sequences_cap(self):
        result = self.engine.run(
            self.tracker, self.dataset,
            dataset_name="Synthetic",
            max_sequences=2,
        )
        assert len(result.sequence_results) == 2

    def test_perfect_tracker_iou(self):
        """A tracker that predicts the exact GT box should achieve mIoU=1.0."""
        result = self.engine.run(self.tracker, self.dataset, dataset_name="Synthetic")
        assert result.mean_iou == pytest.approx(1.0)

    def test_mean_fps_is_positive(self):
        result = self.engine.run(self.tracker, self.dataset, dataset_name="Synthetic")
        assert result.mean_fps > 0.0

    def test_peak_memory_is_positive(self):
        result = self.engine.run(self.tracker, self.dataset, dataset_name="Synthetic")
        assert result.peak_memory_mb > 0.0

    def test_sequence_result_structure(self):
        result = self.engine.run(self.tracker, self.dataset, dataset_name="Synthetic")
        for sr in result.sequence_results:
            assert isinstance(sr, SequenceResult)
            # Engine appends init prediction for frame 0 (GT box) + predictions for
            # frames 1..N-1, giving N total predictions aligned against N GT boxes.
            assert len(sr.ious) == NUM_FRAMES
            assert sr.mean_iou == pytest.approx(1.0)

    def test_tracker_name_propagated(self):
        result = self.engine.run(self.tracker, self.dataset, dataset_name="Synthetic")
        assert result.tracker_name == "ConstantTracker"

    def test_dataset_name_propagated(self):
        result = self.engine.run(self.tracker, self.dataset, dataset_name="MyDataset")
        assert result.dataset_name == "MyDataset"

    def test_summary_dict_keys(self):
        result = self.engine.run(self.tracker, self.dataset, dataset_name="Synthetic")
        s = result.summary()
        for key in ("tracker", "dataset", "num_sequences", "mean_iou", "mean_fps", "peak_memory_mb"):
            assert key in s

    def test_to_dict_structure(self):
        result = self.engine.run(self.tracker, self.dataset, dataset_name="Synthetic")
        d = result.to_dict()
        assert "summary" in d
        assert "sequences" in d
        assert len(d["sequences"]) == len(result.sequence_results)
        for seq_entry in d["sequences"]:
            for key in ("sequence_name", "mean_iou", "fps", "mean_latency_ms", "peak_memory_mb"):
                assert key in seq_entry

    def test_imperfect_tracker_iou(self):
        """A tracker predicting a shifted box should have mIoU < 1."""
        shifted_box = (30.0, 30.0, 50.0, 50.0)  # offset from GT (10,10,50,50)
        tracker = ConstantTracker(shifted_box)
        result = self.engine.run(tracker, self.dataset, dataset_name="Synthetic")
        assert result.mean_iou < 1.0
        assert result.mean_iou > 0.0  # still some overlap

    def test_sequence_result_stores_predictions(self):
        """Engine must store per-frame predictions on each SequenceResult."""
        result = self.engine.run(self.tracker, self.dataset, dataset_name="Synthetic")
        for sr in result.sequence_results:
            assert sr.predictions is not None, "predictions should not be None"
            assert sr.predictions.shape == (NUM_FRAMES, 4)

    def test_sequence_result_stores_ground_truths(self):
        """Engine must store GT boxes aligned to predictions on each SequenceResult."""
        result = self.engine.run(self.tracker, self.dataset, dataset_name="Synthetic")
        for sr in result.sequence_results:
            assert sr.ground_truths is not None, "ground_truths should not be None"
            assert sr.ground_truths.shape == (NUM_FRAMES, 4)

    def test_sequence_result_stores_center_distances(self):
        """Engine must store per-frame centre-distances on each SequenceResult."""
        result = self.engine.run(self.tracker, self.dataset, dataset_name="Synthetic")
        for sr in result.sequence_results:
            assert sr.center_distances is not None, "center_distances should not be None"
            assert sr.center_distances.shape == (NUM_FRAMES,)
            assert (sr.center_distances >= 0.0).all(), "distances must be non-negative"

    def test_perfect_tracker_zero_center_distance(self):
        """A tracker predicting exact GT boxes should have centre-distance = 0 everywhere."""
        result = self.engine.run(self.tracker, self.dataset, dataset_name="Synthetic")
        for sr in result.sequence_results:
            np.testing.assert_allclose(
                sr.center_distances, 0.0, atol=1e-9,
                err_msg="Perfect tracker should have zero centre-distances",
            )

    def test_mean_center_distance_on_result(self):
        """BenchmarkResult.mean_center_distance aggregates across sequences."""
        result = self.engine.run(self.tracker, self.dataset, dataset_name="Synthetic")
        mcd = result.mean_center_distance
        assert mcd is not None
        assert mcd == pytest.approx(0.0)

    def test_summary_includes_center_distance(self):
        """summary() should include mean_center_distance_px when data is stored."""
        result = self.engine.run(self.tracker, self.dataset, dataset_name="Synthetic")
        s = result.summary()
        assert "mean_center_distance_px" in s

    def test_imperfect_tracker_nonzero_center_distance(self):
        """A shifted-box tracker must have non-zero mean centre-distance."""
        shifted_box = (30.0, 30.0, 50.0, 50.0)
        tracker = ConstantTracker(shifted_box)
        result = self.engine.run(tracker, self.dataset, dataset_name="Synthetic")
        assert result.mean_center_distance is not None
        assert result.mean_center_distance > 0.0

    def test_accuracy_metrics_stored_on_sequence_result(self):
        """Each SequenceResult must carry an AccuracyMetrics object."""
        result = self.engine.run(self.tracker, self.dataset, dataset_name="Synthetic")
        for sr in result.sequence_results:
            assert sr.accuracy_metrics is not None

    def test_perfect_tracker_success_auc_is_one(self):
        """A perfect constant tracker should achieve success_auc = 1.0."""
        result = self.engine.run(self.tracker, self.dataset, dataset_name="Synthetic")
        assert result.mean_success_auc is not None
        assert result.mean_success_auc == pytest.approx(1.0, abs=0.01)

    def test_perfect_tracker_np_auc_is_one(self):
        """A perfect constant tracker should achieve np_auc close to 1.0.

        The strict ``<`` check at threshold=0.0 gives a rate of 0.0 for
        exact predictions, so the AUC is marginally below 1.0 by design.
        """
        result = self.engine.run(self.tracker, self.dataset, dataset_name="Synthetic")
        assert result.mean_np_auc is not None
        assert result.mean_np_auc == pytest.approx(1.0, abs=0.02)

    def test_summary_includes_accuracy_aucs(self):
        """summary() must include mean_success_auc, mean_precision_auc, mean_np_auc."""
        result = self.engine.run(self.tracker, self.dataset, dataset_name="Synthetic")
        s = result.summary()
        assert "mean_success_auc" in s
        assert "mean_precision_auc" in s
        assert "mean_np_auc" in s

    def test_to_dict_sequence_includes_accuracy_aucs(self):
        """to_dict() must include success_auc, precision_auc, np_auc per sequence."""
        result = self.engine.run(self.tracker, self.dataset, dataset_name="Synthetic")
        d = result.to_dict()
        for seq in d["sequences"]:
            assert "success_auc" in seq
            assert "precision_auc" in seq
            assert "np_auc" in seq

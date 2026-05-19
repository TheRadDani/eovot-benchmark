"""Integration tests for VOT accuracy metrics (success AUC, precision AUC) in BenchmarkEngine.

Verifies that BenchmarkEngine now stores AccuracyMetrics on every SequenceResult
and that BenchmarkResult aggregates mean_success_auc / mean_precision_auc correctly.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkEngine, BenchmarkResult, SequenceResult
from eovot.datasets.base import BaseDataset, Sequence
from eovot.metrics.accuracy import AccuracyMetrics, MetricsEngine
from eovot.trackers.base import BaseTracker

# ---------------------------------------------------------------------------
# Minimal test stubs
# ---------------------------------------------------------------------------

_BBOX = (10.0, 10.0, 50.0, 50.0)
_SHIFTED = (30.0, 30.0, 50.0, 50.0)
_NUM_FRAMES = 30


class ConstantTracker(BaseTracker):
    def __init__(self, box=_BBOX):
        self._box = box

    @property
    def name(self):
        return "ConstantTracker"

    def initialize(self, frame, bbox):
        pass

    def update(self, frame):
        return self._box


class TinySequence(Sequence):
    def __init__(self, name="s0", box=_BBOX, n=_NUM_FRAMES):
        gt = np.tile(np.array(box), (n, 1))
        super().__init__(name, [f"f{i}" for i in range(n)], gt)
        self._n = n

    def __iter__(self) -> Iterator[np.ndarray]:
        for _ in range(self._n):
            yield np.zeros((120, 160, 3), dtype=np.uint8)


class TinyDataset(BaseDataset):
    def __init__(self, n_seqs=2, box=_BBOX, frames=_NUM_FRAMES):
        self._seqs = [TinySequence(f"s{i}", box, frames) for i in range(n_seqs)]

    def __len__(self):
        return len(self._seqs)

    def __getitem__(self, idx):
        return self._seqs[idx]


# ---------------------------------------------------------------------------
# accuracy_metrics field on SequenceResult
# ---------------------------------------------------------------------------

class TestSequenceResultAccuracyMetrics:
    def setup_method(self):
        self.engine = BenchmarkEngine(verbose=False)

    def test_accuracy_metrics_populated(self):
        result = self.engine.run(ConstantTracker(), TinyDataset(), "T")
        for sr in result.sequence_results:
            assert sr.accuracy_metrics is not None
            assert isinstance(sr.accuracy_metrics, AccuracyMetrics)

    def test_perfect_tracker_success_auc_near_one(self):
        result = self.engine.run(ConstantTracker(_BBOX), TinyDataset(box=_BBOX), "T")
        for sr in result.sequence_results:
            assert sr.accuracy_metrics is not None
            # Perfect overlap → success at all thresholds ≤ 1 → AUC ≈ 1
            assert sr.accuracy_metrics.success_auc > 0.95

    def test_shifted_tracker_lower_success_auc(self):
        perfect = self.engine.run(ConstantTracker(_BBOX), TinyDataset(box=_BBOX), "T")
        shifted = self.engine.run(ConstantTracker(_SHIFTED), TinyDataset(box=_BBOX), "T")
        perfect_auc = perfect.sequence_results[0].accuracy_metrics.success_auc
        shifted_auc = shifted.sequence_results[0].accuracy_metrics.success_auc
        assert perfect_auc > shifted_auc

    def test_success_auc_property_on_sequence_result(self):
        result = self.engine.run(ConstantTracker(), TinyDataset(), "T")
        for sr in result.sequence_results:
            assert sr.success_auc is not None
            assert 0.0 <= sr.success_auc <= 1.0

    def test_precision_auc_property_on_sequence_result(self):
        result = self.engine.run(ConstantTracker(), TinyDataset(), "T")
        for sr in result.sequence_results:
            assert sr.precision_auc is not None
            assert 0.0 <= sr.precision_auc <= 1.0

    def test_accuracy_metrics_mean_iou_matches_sr_mean_iou(self):
        result = self.engine.run(ConstantTracker(_BBOX), TinyDataset(box=_BBOX), "T")
        for sr in result.sequence_results:
            assert sr.accuracy_metrics.mean_iou == pytest.approx(sr.mean_iou, rel=1e-5)


# ---------------------------------------------------------------------------
# mean_success_auc / mean_precision_auc on BenchmarkResult
# ---------------------------------------------------------------------------

class TestBenchmarkResultAUCProperties:
    def setup_method(self):
        self.engine = BenchmarkEngine(verbose=False)

    def test_mean_success_auc_not_none(self):
        result = self.engine.run(ConstantTracker(), TinyDataset(), "T")
        assert result.mean_success_auc is not None

    def test_mean_precision_auc_not_none(self):
        result = self.engine.run(ConstantTracker(), TinyDataset(), "T")
        assert result.mean_precision_auc is not None

    def test_mean_success_auc_in_range(self):
        result = self.engine.run(ConstantTracker(), TinyDataset(), "T")
        assert 0.0 <= result.mean_success_auc <= 1.0

    def test_mean_precision_auc_in_range(self):
        result = self.engine.run(ConstantTracker(), TinyDataset(), "T")
        assert 0.0 <= result.mean_precision_auc <= 1.0

    def test_perfect_tracker_high_mean_success_auc(self):
        result = self.engine.run(ConstantTracker(_BBOX), TinyDataset(box=_BBOX), "T")
        assert result.mean_success_auc > 0.95

    def test_shifted_tracker_lower_mean_success_auc(self):
        perfect = self.engine.run(ConstantTracker(_BBOX), TinyDataset(box=_BBOX), "T")
        shifted = self.engine.run(ConstantTracker(_SHIFTED), TinyDataset(box=_BBOX), "T")
        assert perfect.mean_success_auc > shifted.mean_success_auc

    def test_mean_success_auc_is_average_of_sequence_aucs(self):
        result = self.engine.run(ConstantTracker(), TinyDataset(n_seqs=4), "T")
        seq_aucs = [sr.accuracy_metrics.success_auc for sr in result.sequence_results]
        expected = float(np.mean(seq_aucs))
        assert result.mean_success_auc == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# summary() now includes success_auc / precision_auc
# ---------------------------------------------------------------------------

class TestSummaryIncludesAUC:
    def setup_method(self):
        self.engine = BenchmarkEngine(verbose=False)

    def test_success_auc_in_summary(self):
        result = self.engine.run(ConstantTracker(), TinyDataset(), "T")
        s = result.summary()
        assert "success_auc" in s

    def test_precision_auc_in_summary(self):
        result = self.engine.run(ConstantTracker(), TinyDataset(), "T")
        s = result.summary()
        assert "precision_auc" in s

    def test_summary_auc_values_are_floats_in_range(self):
        result = self.engine.run(ConstantTracker(), TinyDataset(), "T")
        s = result.summary()
        assert isinstance(s["success_auc"], float)
        assert 0.0 <= s["success_auc"] <= 1.0
        assert isinstance(s["precision_auc"], float)
        assert 0.0 <= s["precision_auc"] <= 1.0


# ---------------------------------------------------------------------------
# to_dict() includes per-sequence success_auc / precision_auc
# ---------------------------------------------------------------------------

class TestToDictIncludesAUC:
    def setup_method(self):
        self.engine = BenchmarkEngine(verbose=False)

    def test_per_sequence_entry_has_success_auc(self):
        result = self.engine.run(ConstantTracker(), TinyDataset(), "T")
        d = result.to_dict()
        for seq_entry in d["sequences"]:
            assert "success_auc" in seq_entry

    def test_per_sequence_entry_has_precision_auc(self):
        result = self.engine.run(ConstantTracker(), TinyDataset(), "T")
        d = result.to_dict()
        for seq_entry in d["sequences"]:
            assert "precision_auc" in seq_entry

    def test_per_sequence_auc_matches_sequence_result(self):
        result = self.engine.run(ConstantTracker(), TinyDataset(n_seqs=3), "T")
        d = result.to_dict()
        for i, seq_entry in enumerate(d["sequences"]):
            sr = result.sequence_results[i]
            assert seq_entry["success_auc"] == pytest.approx(
                sr.accuracy_metrics.success_auc, rel=1e-4
            )


# ---------------------------------------------------------------------------
# Vectorized batch_iou consistency check
# ---------------------------------------------------------------------------

class TestVectorizedBatchIoU:
    """Confirm the vectorised batch_iou gives identical results to the scalar iou()."""

    def setup_method(self):
        self.engine_m = MetricsEngine()

    def test_matches_scalar_iou_random_boxes(self):
        from eovot.metrics.accuracy import iou
        rng = np.random.default_rng(99)
        for _ in range(20):
            n = rng.integers(5, 50)
            preds = rng.uniform(0, 100, (n, 4))
            gts = rng.uniform(0, 100, (n, 4))
            preds[:, 2:] = np.abs(preds[:, 2:]) + 1
            gts[:, 2:] = np.abs(gts[:, 2:]) + 1

            vec = self.engine_m.batch_iou(preds, gts)
            scalar = np.array([iou(tuple(preds[i]), tuple(gts[i])) for i in range(n)])
            np.testing.assert_allclose(vec, scalar, rtol=1e-9,
                err_msg="Vectorised batch_iou must match scalar iou() element-wise")

    def test_empty_input_returns_empty(self):
        result = self.engine_m.batch_iou(np.empty((0, 4)), np.empty((0, 4)))
        assert len(result) == 0

    def test_batch_center_distance_vectorised(self):
        from eovot.metrics.accuracy import center_distance
        rng = np.random.default_rng(7)
        n = 40
        preds = rng.uniform(0, 200, (n, 4))
        gts = rng.uniform(0, 200, (n, 4))
        preds[:, 2:] += 5
        gts[:, 2:] += 5

        vec = self.engine_m.batch_center_distance(preds, gts)
        scalar = np.array([center_distance(tuple(preds[i]), tuple(gts[i])) for i in range(n)])
        np.testing.assert_allclose(vec, scalar, rtol=1e-9)

    def test_batch_center_distance_empty(self):
        result = self.engine_m.batch_center_distance(np.empty((0, 4)), np.empty((0, 4)))
        assert len(result) == 0

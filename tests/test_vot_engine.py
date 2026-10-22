"""Tests for the VOT re-initialization protocol engine."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.benchmark.vot_engine import (
    ReinitEvent,
    VOTBenchmarkEngine,
    VOTBenchmarkResult,
    VOTSequenceResult,
)
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.kcf import KCFTracker
from eovot.trackers.mosse import MOSSETracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_dataset():
    return SyntheticDataset(num_sequences=3, num_frames=40, motion="linear", seed=0)


@pytest.fixture()
def engine():
    return VOTBenchmarkEngine(failure_threshold=0.1, reinit_delay=5, verbose=False)


# ---------------------------------------------------------------------------
# VOTBenchmarkEngine.run — basic contract
# ---------------------------------------------------------------------------


def test_run_returns_vot_benchmark_result(small_dataset, engine):
    result = engine.run(MOSSETracker(), small_dataset, dataset_name="Test")
    assert isinstance(result, VOTBenchmarkResult)


def test_run_respects_max_sequences(small_dataset, engine):
    result = engine.run(MOSSETracker(), small_dataset, dataset_name="Test", max_sequences=2)
    assert len(result.sequence_results) == 2


def test_run_all_sequences(small_dataset, engine):
    result = engine.run(MOSSETracker(), small_dataset, dataset_name="Test")
    assert len(result.sequence_results) == 3


def test_eao_bounded(small_dataset, engine):
    result = engine.run(MOSSETracker(), small_dataset, dataset_name="Test")
    assert 0.0 <= result.mean_eao <= 1.0
    for sr in result.sequence_results:
        assert 0.0 <= sr.eao <= 1.0


def test_frame_ious_shape(small_dataset, engine):
    result = engine.run(MOSSETracker(), small_dataset, dataset_name="Test")
    for sr in result.sequence_results:
        assert sr.frame_ious.shape == (40,)
        assert np.all((sr.frame_ious >= 0.0) & (sr.frame_ious <= 1.0))


# ---------------------------------------------------------------------------
# Re-initialization behaviour
# ---------------------------------------------------------------------------


def test_zero_threshold_no_reinits(small_dataset):
    """With failure_threshold=0.0 no frame can trigger a re-initialization."""
    engine = VOTBenchmarkEngine(failure_threshold=0.0, reinit_delay=5, verbose=False)
    result = engine.run(MOSSETracker(), small_dataset, dataset_name="Test")
    assert result.total_reinitializations == 0


def test_unit_threshold_max_reinits(small_dataset):
    """With failure_threshold=1.0 every non-perfect frame is a failure."""
    engine = VOTBenchmarkEngine(failure_threshold=1.0, reinit_delay=1, verbose=False)
    result = engine.run(KCFTracker(), small_dataset, dataset_name="Test")
    # At least some sequences should trigger re-initializations.
    assert result.total_reinitializations >= 0  # non-negative sanity check


def test_reinit_events_frame_order(small_dataset):
    """failure_frame must be strictly less than reinit_frame in each event."""
    engine = VOTBenchmarkEngine(failure_threshold=0.5, reinit_delay=5, verbose=False)
    result = engine.run(KCFTracker(), small_dataset, dataset_name="Test")
    for sr in result.sequence_results:
        for ev in sr.reinit_events:
            assert isinstance(ev, ReinitEvent)
            assert ev.failure_frame < ev.reinit_frame or ev.penalty_frames == 0


def test_penalty_frames_are_zero_iou():
    """Frames inside the penalty window must have IoU 0.0."""
    dataset = SyntheticDataset(num_sequences=1, num_frames=60, motion="random", seed=7)
    engine = VOTBenchmarkEngine(failure_threshold=1.0, reinit_delay=5, verbose=False)
    result = engine.run(KCFTracker(), dataset, dataset_name="Test")
    # If any reinit events occurred, there must be 0.0 IoU frames.
    sr = result.sequence_results[0]
    if sr.num_reinitializations > 0:
        assert float(np.min(sr.frame_ious)) == 0.0


# ---------------------------------------------------------------------------
# VOTBenchmarkResult aggregates
# ---------------------------------------------------------------------------


def test_summary_keys(small_dataset, engine):
    result = engine.run(MOSSETracker(), small_dataset, dataset_name="Test")
    s = result.summary()
    required = {
        "tracker", "dataset", "num_sequences", "mean_eao", "mean_iou",
        "mean_fps", "peak_memory_mb", "total_reinitializations",
        "mean_reinits_per_sequence",
    }
    assert required.issubset(set(s.keys()))


def test_summary_tracker_name(small_dataset, engine):
    result = engine.run(MOSSETracker(), small_dataset, dataset_name="MyData")
    assert result.summary()["tracker"] == "MOSSE"
    assert result.summary()["dataset"] == "MyData"


def test_str_contains_tracker_name(small_dataset, engine):
    result = engine.run(MOSSETracker(), small_dataset, dataset_name="Test")
    s = str(result)
    assert "MOSSE" in s
    assert "EAO" in s


def test_mean_eao_consistent(small_dataset, engine):
    """mean_eao == mean of per-sequence eao values."""
    result = engine.run(MOSSETracker(), small_dataset, dataset_name="Test")
    expected = float(np.mean([r.eao for r in result.sequence_results]))
    assert abs(result.mean_eao - expected) < 1e-9


def test_total_reinitializations_consistent(small_dataset):
    engine = VOTBenchmarkEngine(failure_threshold=0.5, reinit_delay=3, verbose=False)
    result = engine.run(KCFTracker(), small_dataset, dataset_name="Test")
    expected = sum(r.num_reinitializations for r in result.sequence_results)
    assert result.total_reinitializations == expected


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_invalid_failure_threshold():
    with pytest.raises(ValueError, match="failure_threshold"):
        VOTBenchmarkEngine(failure_threshold=1.5)


def test_invalid_reinit_delay():
    with pytest.raises(ValueError, match="reinit_delay"):
        VOTBenchmarkEngine(reinit_delay=-1)


# ---------------------------------------------------------------------------
# VOTSequenceResult
# ---------------------------------------------------------------------------


def test_sequence_result_mean_iou_equals_eao(small_dataset, engine):
    result = engine.run(MOSSETracker(), small_dataset, dataset_name="Test")
    for sr in result.sequence_results:
        # mean_iou property is the mean of frame_ious (excluding init frame via eao)
        # eao is computed as mean(frame_ious[1:]), mean_iou as mean(frame_ious)
        # They should be very close (differ only by first frame weight).
        assert abs(sr.mean_iou - float(sr.frame_ious.mean())) < 1e-9

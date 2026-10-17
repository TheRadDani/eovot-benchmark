"""Tests for the VOT reset-based evaluation protocol engine."""

from __future__ import annotations

from typing import List, Optional
import numpy as np
import pytest

from eovot.benchmark.vot_engine import VOTEngine, VOTSequenceResult, VOTDatasetResult
from eovot.datasets.synthetic import SyntheticDataset


# ---------------------------------------------------------------------------
# Stub trackers — deterministic fixtures for unit testing
# ---------------------------------------------------------------------------

class _StaticTracker:
    """Returns the initialization box on every update (tracks static targets)."""

    name = "static"

    def __init__(self):
        self._bbox = None

    def initialize(self, frame, bbox):
        self._bbox = tuple(bbox)

    def update(self, frame):
        return self._bbox


class _AlwaysFailTracker:
    """Returns a box completely outside the frame — always fails."""

    name = "always_fail"

    def initialize(self, frame, bbox):
        pass

    def update(self, frame):
        return (-9999.0, -9999.0, 1.0, 1.0)


class _FailOnceTracker:
    """Fails exactly once at a given frame index after each initialization."""

    name = "fail_once"

    def __init__(self, fail_at: int = 5):
        self.fail_at = fail_at
        self._update_count = 0
        self._bbox = None

    def initialize(self, frame, bbox):
        self._bbox = tuple(bbox)
        self._update_count = 0

    def update(self, frame):
        self._update_count += 1
        if self._update_count == self.fail_at:
            return (-9999.0, -9999.0, 1.0, 1.0)
        return self._bbox


class _OracleTracker:
    """Replays exact ground-truth boxes from a sequence — never fails.

    Usage: call ``set_ground_truth(seq.ground_truth)`` before passing this
    tracker to ``VOTEngine.run_sequence``.  The tracker replays the correct
    box for every update call, including post-re-initialization frames.
    """

    name = "oracle"

    def __init__(self):
        self._gts: Optional[np.ndarray] = None
        self._frame_idx: int = 0
        self._reinit_offset: int = 0

    def set_ground_truth(self, gts: np.ndarray) -> None:
        self._gts = gts

    def initialize(self, frame, bbox) -> None:
        # Identify current frame by matching bbox to stored GTs
        if self._gts is not None:
            for i, gt in enumerate(self._gts):
                if np.allclose(bbox, gt, atol=2.0):
                    self._frame_idx = i
                    return
        self._frame_idx = 0

    def update(self, frame):
        self._frame_idx += 1
        if self._gts is not None and self._frame_idx < len(self._gts):
            return tuple(self._gts[self._frame_idx])
        if self._gts is not None:
            return tuple(self._gts[-1])
        return (0.0, 0.0, 10.0, 10.0)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _make_dataset(n_seq=3, n_frames=30, motion="linear", seed=0):
    return SyntheticDataset(
        num_sequences=n_seq,
        num_frames=n_frames,
        frame_size=(160, 120),
        bbox_size=(30, 30),
        motion=motion,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# VOTEngine construction
# ---------------------------------------------------------------------------

def test_vot_engine_default_params():
    engine = VOTEngine()
    assert engine.gap == 5
    assert engine.failure_threshold == 0.0
    assert engine.verbose is True


def test_vot_engine_rejects_negative_gap():
    with pytest.raises(ValueError, match="gap must be >= 0"):
        VOTEngine(gap=-1)


# ---------------------------------------------------------------------------
# run_sequence — oracle tracker (never fails)
# ---------------------------------------------------------------------------

def test_oracle_tracker_no_failures():
    dataset = _make_dataset(n_seq=1, n_frames=40)
    engine = VOTEngine(gap=5, verbose=False)
    seq = dataset[0]

    oracle = _OracleTracker()
    oracle.set_ground_truth(seq.ground_truth)

    result = engine.run_sequence(oracle, seq)

    assert isinstance(result, VOTSequenceResult)
    assert result.num_failures == 0
    assert len(result.failure_frames) == 0
    assert len(result.reinit_frames) == 0


def test_oracle_tracker_high_eao():
    dataset = _make_dataset(n_seq=1, n_frames=40)
    engine = VOTEngine(gap=5, verbose=False)
    seq = dataset[0]

    oracle = _OracleTracker()
    oracle.set_ground_truth(seq.ground_truth)

    result = engine.run_sequence(oracle, seq)

    # Oracle should yield high EAO (close to 1.0)
    assert result.eao > 0.8


def test_oracle_tracker_overlap_length():
    dataset = _make_dataset(n_seq=1, n_frames=20)
    engine = VOTEngine(gap=5, verbose=False)
    seq = dataset[0]

    oracle = _OracleTracker()
    oracle.set_ground_truth(seq.ground_truth)

    result = engine.run_sequence(oracle, seq)

    n = min(len(list(seq)), len(seq.ground_truth))
    assert len(result.overlaps) == n


# ---------------------------------------------------------------------------
# run_sequence — always-fail tracker
# ---------------------------------------------------------------------------

def test_always_fail_tracker_eao_near_zero():
    dataset = _make_dataset(n_seq=1, n_frames=50)
    engine = VOTEngine(gap=5, failure_threshold=0.0, verbose=False)
    result = engine.run_sequence(_AlwaysFailTracker(), dataset[0])

    # EAO should be very low — mostly zero from gap windows
    assert result.eao < 0.2
    assert result.num_failures > 0


def test_always_fail_tracker_reinit_count():
    dataset = _make_dataset(n_seq=1, n_frames=60)
    engine = VOTEngine(gap=5, verbose=False)
    result = engine.run_sequence(_AlwaysFailTracker(), dataset[0])

    # Each failure triggers a re-init unless the gap extends past the sequence end
    assert len(result.reinit_frames) <= len(result.failure_frames)
    # With 60 frames and gap=5 the last failure may not have room for re-init
    assert len(result.reinit_frames) >= len(result.failure_frames) - 1


# ---------------------------------------------------------------------------
# run_sequence — fail-once tracker
# ---------------------------------------------------------------------------

def test_fail_once_tracker_single_failure():
    """_FailOnceTracker fails at update 5 after EACH initialization.

    After a failure at frame F, the engine re-initializes at frame F+gap+1
    which resets the tracker's counter, so it fails again at update 5
    post-re-init.  We validate that the first failure is recorded and the
    re-initialization machinery activates at least once.
    """
    dataset = _make_dataset(n_seq=1, n_frames=40)
    engine = VOTEngine(gap=5, verbose=False)
    result = engine.run_sequence(_FailOnceTracker(fail_at=5), dataset[0])

    # At least one failure must be recorded
    assert result.num_failures >= 1
    # At least one re-initialization must follow the first failure
    assert len(result.reinit_frames) >= 1
    # First failure is at the expected frame (1-indexed update_count == fail_at)
    assert 5 in result.failure_frames


def test_fail_once_tracker_gap_zeros():
    """Gap frames (5 frames after a failure) must carry zero overlap."""
    dataset = _make_dataset(n_seq=1, n_frames=40)
    engine = VOTEngine(gap=5, verbose=False)
    fail_at = 5
    result = engine.run_sequence(_FailOnceTracker(fail_at=fail_at), dataset[0])

    # Failure frame itself → overlap == 0
    assert result.overlaps[fail_at] == 0.0

    # Next `gap` frames also == 0
    gap_end = fail_at + engine.gap
    for t in range(fail_at + 1, min(gap_end + 1, len(result.overlaps))):
        assert result.overlaps[t] == 0.0, f"Expected 0 at gap frame {t}"


# ---------------------------------------------------------------------------
# run_dataset aggregation — use stub trackers only
# ---------------------------------------------------------------------------

def test_run_dataset_returns_vot_dataset_result():
    dataset = _make_dataset(n_seq=3, n_frames=30)
    engine = VOTEngine(gap=5, verbose=False)

    result = engine.run_dataset(_StaticTracker(), dataset, dataset_name="Synthetic")

    assert isinstance(result, VOTDatasetResult)
    assert result.dataset_name == "Synthetic"
    assert len(result.sequence_results) == 3


def test_run_dataset_max_sequences():
    dataset = _make_dataset(n_seq=5, n_frames=30)
    engine = VOTEngine(gap=5, verbose=False)

    result = engine.run_dataset(
        _StaticTracker(), dataset, max_sequences=2, dataset_name="Synthetic"
    )

    assert len(result.sequence_results) == 2


def test_run_dataset_eao_in_valid_range():
    dataset = _make_dataset(n_seq=3, n_frames=30)
    engine = VOTEngine(gap=5, verbose=False)

    result = engine.run_dataset(_StaticTracker(), dataset, dataset_name="Synthetic")

    assert 0.0 <= result.eao <= 1.0
    assert 0.0 <= result.accuracy <= 1.0


def test_run_dataset_summary_keys():
    dataset = _make_dataset(n_seq=2, n_frames=30)
    engine = VOTEngine(gap=5, verbose=False)

    result = engine.run_dataset(_StaticTracker(), dataset, dataset_name="Synthetic")
    summary = result.summary()

    required_keys = {
        "tracker", "dataset", "num_sequences", "eao", "accuracy",
        "total_failures", "mean_failures_per_sequence", "mean_fps",
        "peak_memory_mb",
    }
    assert required_keys.issubset(summary.keys())


# ---------------------------------------------------------------------------
# EAO ordering: oracle > static > always-fail
# ---------------------------------------------------------------------------

def test_eao_ordering():
    n_seq, n_frames = 2, 40
    engine = VOTEngine(gap=5, verbose=False)

    oracle_results = []
    static_results = []
    fail_results = []

    for i in range(n_seq):
        dataset = _make_dataset(n_seq=1, n_frames=n_frames, seed=i * 7)
        seq = dataset[0]

        oracle = _OracleTracker()
        oracle.set_ground_truth(seq.ground_truth)
        oracle_results.append(engine.run_sequence(oracle, seq).eao)
        static_results.append(engine.run_sequence(_StaticTracker(), seq).eao)
        fail_results.append(engine.run_sequence(_AlwaysFailTracker(), seq).eao)

    assert np.mean(oracle_results) >= np.mean(static_results)
    assert np.mean(static_results) >= np.mean(fail_results)


# ---------------------------------------------------------------------------
# Strict failure threshold → at least as many failures as loose threshold
# ---------------------------------------------------------------------------

def test_strict_failure_threshold():
    dataset = _make_dataset(n_seq=1, n_frames=50)
    seq = dataset[0]

    engine_loose = VOTEngine(gap=5, failure_threshold=0.0, verbose=False)
    engine_strict = VOTEngine(gap=5, failure_threshold=0.5, verbose=False)

    # Static tracker returns init box; if target moves, IoU eventually < 0.5
    r_loose = engine_loose.run_sequence(_StaticTracker(), seq)
    r_strict = engine_strict.run_sequence(_StaticTracker(), seq)

    assert r_strict.num_failures >= r_loose.num_failures


# ---------------------------------------------------------------------------
# Gap=0 means immediate re-initialization with no blank frames
# ---------------------------------------------------------------------------

def test_gap_zero_no_blank_frames():
    dataset = _make_dataset(n_seq=1, n_frames=30)
    engine = VOTEngine(gap=0, verbose=False)
    result = engine.run_sequence(_FailOnceTracker(fail_at=5), dataset[0])

    # Failure at frame 5; with gap=0, frame 6 is already re-init (overlap=1)
    assert result.overlaps[5] == 0.0
    if len(result.overlaps) > 6:
        # Frame 6 is a re-init frame → assigned overlap 1.0
        assert result.overlaps[6] == 1.0


# ---------------------------------------------------------------------------
# Predictions and ground truths are stored with correct shape
# ---------------------------------------------------------------------------

def test_predictions_and_ground_truths_stored():
    dataset = _make_dataset(n_seq=1, n_frames=20)
    engine = VOTEngine(gap=5, verbose=False)
    result = engine.run_sequence(_StaticTracker(), dataset[0])

    assert result.predictions is not None
    assert result.ground_truths is not None
    assert result.predictions.shape[1] == 4
    assert result.ground_truths.shape[1] == 4
    assert len(result.predictions) == len(result.overlaps)


# ---------------------------------------------------------------------------
# total_failures aggregation on VOTDatasetResult
# ---------------------------------------------------------------------------

def test_total_failures_aggregation():
    dataset = _make_dataset(n_seq=3, n_frames=30)
    engine = VOTEngine(gap=5, verbose=False)
    result = engine.run_dataset(_AlwaysFailTracker(), dataset, dataset_name="S")

    expected = sum(r.num_failures for r in result.sequence_results)
    assert result.total_failures == expected

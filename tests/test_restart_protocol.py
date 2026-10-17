"""Tests for the VOT Restart evaluation protocol."""

import numpy as np
import pytest

from eovot.benchmark.restart import (
    RestartBenchmarkResult,
    RestartEvent,
    RestartProtocolEngine,
    RestartSequenceResult,
)
from eovot.datasets.base import BaseDataset, Sequence
from eovot.trackers.base import BaseTracker


# ---------------------------------------------------------------------------
# In-memory dataset helpers for deterministic protocol testing
# ---------------------------------------------------------------------------


class _InMemorySequence(Sequence):
    """Sequence backed by pre-built numpy frames; no disk I/O."""

    def __init__(self, name: str, frames: list, ground_truth: np.ndarray) -> None:
        super().__init__(
            name=name,
            frame_paths=["<mem>"] * len(frames),
            ground_truth=ground_truth,
        )
        self._mem_frames = frames

    def __iter__(self):
        yield from self._mem_frames


class StaticDataset(BaseDataset):
    """Dataset where all sequences have a constant (non-moving) GT bbox.

    Useful for pure protocol testing: a tracker that returns the init bbox
    will always have IoU=1.0, giving zero restarts.
    """

    def __init__(
        self,
        n_sequences: int = 2,
        n_frames: int = 30,
        bbox: tuple = (20, 20, 30, 30),
        frame_size: tuple = (100, 100),
    ) -> None:
        h, w = frame_size
        gt = np.tile(np.array(bbox, dtype=np.float64), (n_frames, 1))
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        frames = [blank.copy() for _ in range(n_frames)]
        self._seqs = [
            _InMemorySequence(name=f"static_{i:03d}", frames=frames, ground_truth=gt)
            for i in range(n_sequences)
        ]

    def __len__(self) -> int:
        return len(self._seqs)

    def __getitem__(self, idx: int) -> Sequence:
        return self._seqs[idx]


# ---------------------------------------------------------------------------
# Stub trackers
# ---------------------------------------------------------------------------


class ConstantBboxTracker(BaseTracker):
    """Always returns the initialization bbox — perfect on a static target."""

    def __init__(self):
        super().__init__(name="constant_bbox")
        self._bbox = None

    def initialize(self, frame, bbox):
        self._bbox = list(bbox)

    def update(self, frame):
        return tuple(self._bbox)


class AlwaysFailTracker(BaseTracker):
    """Always returns (0, 0, 0, 0)."""

    def __init__(self):
        super().__init__(name="always_fail")

    def initialize(self, frame, bbox):
        self._bbox = list(bbox)

    def update(self, frame):
        return (0.0, 0.0, 0.0, 0.0)


class SingleFailTracker(BaseTracker):
    """Fails exactly once on the N-th global update() call, then returns GT.

    The failure flag is NOT reset on initialize(), so the failure happens
    exactly once across the entire sequence even if the tracker is restarted.
    """

    def __init__(self, fail_on: int):
        super().__init__(name=f"single_fail_{fail_on}")
        self._fail_on = fail_on
        self._update_count = 0
        self._has_failed = False
        self._bbox = None

    def initialize(self, frame, bbox):
        self._bbox = list(bbox)

    def update(self, frame):
        self._update_count += 1
        if self._update_count == self._fail_on and not self._has_failed:
            self._has_failed = True
            return (0.0, 0.0, 0.0, 0.0)
        return tuple(self._bbox)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestRestartProtocolEngineInit:
    def test_default_params(self):
        engine = RestartProtocolEngine()
        assert engine.failure_threshold == 0.0
        assert engine.grace_period == 5

    def test_custom_params(self):
        engine = RestartProtocolEngine(failure_threshold=0.1, grace_period=3)
        assert engine.failure_threshold == 0.1
        assert engine.grace_period == 3

    def test_invalid_threshold_negative(self):
        with pytest.raises(ValueError):
            RestartProtocolEngine(failure_threshold=-0.1)

    def test_invalid_threshold_one(self):
        with pytest.raises(ValueError):
            RestartProtocolEngine(failure_threshold=1.0)

    def test_invalid_grace_period(self):
        with pytest.raises(ValueError):
            RestartProtocolEngine(grace_period=-1)

    def test_zero_grace_period_allowed(self):
        engine = RestartProtocolEngine(grace_period=0)
        assert engine.grace_period == 0


# ---------------------------------------------------------------------------
# Perfect tracker on static dataset — EAO=1.0, zero restarts
# ---------------------------------------------------------------------------


class TestPerfectTracker:
    def setup_method(self):
        self.dataset = StaticDataset(n_sequences=2, n_frames=20)
        self.engine = RestartProtocolEngine(verbose=False)

    def test_zero_restarts(self):
        tracker = ConstantBboxTracker()
        result = self.engine.run(tracker, self.dataset, dataset_name="test")
        assert result.total_restarts == 0

    def test_eao_equals_one(self):
        tracker = ConstantBboxTracker()
        result = self.engine.run(tracker, self.dataset, dataset_name="test")
        assert result.mean_eao == pytest.approx(1.0, abs=1e-6)

    def test_sequence_count(self):
        tracker = ConstantBboxTracker()
        result = self.engine.run(tracker, self.dataset, dataset_name="test")
        assert len(result.sequence_results) == 2

    def test_no_restart_events(self):
        tracker = ConstantBboxTracker()
        result = self.engine.run(tracker, self.dataset, dataset_name="test")
        for seq_result in result.sequence_results:
            assert seq_result.restarts == []


# ---------------------------------------------------------------------------
# Always-failing tracker — many restarts, low EAO
# ---------------------------------------------------------------------------


class TestAlwaysFailTracker:
    def setup_method(self):
        self.dataset = StaticDataset(n_sequences=1, n_frames=30)
        self.engine = RestartProtocolEngine(grace_period=5, verbose=False)

    def test_many_restarts(self):
        tracker = AlwaysFailTracker()
        result = self.engine.run(tracker, self.dataset, dataset_name="test")
        assert result.total_restarts > 0

    def test_eao_below_perfect(self):
        tracker = AlwaysFailTracker()
        result = self.engine.run(tracker, self.dataset, dataset_name="test")
        assert result.mean_eao < 1.0

    def test_restart_frames_recorded(self):
        tracker = AlwaysFailTracker()
        result = self.engine.run(tracker, self.dataset, dataset_name="test")
        seq = result.sequence_results[0]
        for event in seq.restarts:
            assert isinstance(event.failure_frame, int)
            assert isinstance(event.restart_frame, int)
            assert event.restart_frame > event.failure_frame


# ---------------------------------------------------------------------------
# Single-failure tracker — exactly one restart, grace period correctly placed
# ---------------------------------------------------------------------------


class TestSingleFailure:
    GRACE = 5
    FAIL_ON = 5  # fail on the 5th update() call → frame index 5

    def setup_method(self):
        self.dataset = StaticDataset(n_sequences=1, n_frames=40)
        self.engine = RestartProtocolEngine(
            failure_threshold=0.0, grace_period=self.GRACE, verbose=False
        )

    def test_exactly_one_restart(self):
        tracker = SingleFailTracker(fail_on=self.FAIL_ON)
        result = self.engine.run(tracker, self.dataset, dataset_name="test")
        assert result.total_restarts == 1

    def test_failure_frame_index(self):
        tracker = SingleFailTracker(fail_on=self.FAIL_ON)
        result = self.engine.run(tracker, self.dataset, dataset_name="test")
        seq = result.sequence_results[0]
        assert seq.restarts[0].failure_frame == self.FAIL_ON

    def test_restart_frame_index(self):
        tracker = SingleFailTracker(fail_on=self.FAIL_ON)
        result = self.engine.run(tracker, self.dataset, dataset_name="test")
        seq = result.sequence_results[0]
        expected_reinit = self.FAIL_ON + self.GRACE + 1
        assert seq.restarts[0].restart_frame == expected_reinit

    def test_grace_period_frames_are_zero(self):
        tracker = SingleFailTracker(fail_on=self.FAIL_ON)
        result = self.engine.run(tracker, self.dataset, dataset_name="test")
        seq = result.sequence_results[0]
        for gf in range(self.FAIL_ON + 1, self.FAIL_ON + self.GRACE + 1):
            assert seq.ious[gf] == pytest.approx(0.0), f"frame {gf} should be 0"

    def test_reinit_frame_is_one(self):
        tracker = SingleFailTracker(fail_on=self.FAIL_ON)
        result = self.engine.run(tracker, self.dataset, dataset_name="test")
        seq = result.sequence_results[0]
        reinit_frame = self.FAIL_ON + self.GRACE + 1
        assert seq.ious[reinit_frame] == pytest.approx(1.0)

    def test_frames_after_reinit_are_one(self):
        tracker = SingleFailTracker(fail_on=self.FAIL_ON)
        result = self.engine.run(tracker, self.dataset, dataset_name="test")
        seq = result.sequence_results[0]
        reinit_frame = self.FAIL_ON + self.GRACE + 1
        for f in range(reinit_frame + 1, len(seq.ious)):
            assert seq.ious[f] == pytest.approx(1.0), f"frame {f} should be 1.0"

    def test_iou_at_failure_recorded(self):
        tracker = SingleFailTracker(fail_on=self.FAIL_ON)
        result = self.engine.run(tracker, self.dataset, dataset_name="test")
        seq = result.sequence_results[0]
        assert seq.restarts[0].iou_at_failure == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Zero grace period
# ---------------------------------------------------------------------------


class TestZeroGracePeriod:
    def test_zero_grace_immediate_reinit(self):
        """With grace_period=0, reinit happens on the very next frame."""
        dataset = StaticDataset(n_sequences=1, n_frames=20)
        engine = RestartProtocolEngine(failure_threshold=0.0, grace_period=0, verbose=False)
        tracker = SingleFailTracker(fail_on=3)
        result = engine.run(tracker, dataset, dataset_name="test")
        assert result.total_restarts == 1
        seq = result.sequence_results[0]
        # reinit_at = 3 + 0 + 1 = 4 → frame 4 is reinit
        assert seq.restarts[0].restart_frame == 4
        assert seq.ious[4] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# RestartSequenceResult dataclass
# ---------------------------------------------------------------------------


class TestRestartSequenceResult:
    def _make_result(self, ious_list, n_restarts=1):
        n = len(ious_list)
        restarts = [
            RestartEvent(failure_frame=2, restart_frame=8, iou_at_failure=0.0)
        ] * n_restarts
        return RestartSequenceResult(
            sequence_name="test_seq",
            tracker_name="test_tracker",
            ious=np.array(ious_list, dtype=np.float64),
            predictions=np.zeros((n, 4)),
            ground_truths=np.zeros((n, 4)),
            restarts=restarts,
            fps=100.0,
            mean_latency_ms=10.0,
        )

    def test_eao_is_mean_iou(self):
        ious = [1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.9]
        result = self._make_result(ious)
        assert result.eao == pytest.approx(np.mean(ious))

    def test_num_restarts(self):
        result = self._make_result([1.0, 0.5, 0.0], n_restarts=2)
        assert result.num_restarts == 2

    def test_restart_frames(self):
        result = self._make_result([1.0, 0.5, 0.0], n_restarts=1)
        assert result.restart_frames == [2]

    def test_eao_empty(self):
        result = RestartSequenceResult(
            sequence_name="s",
            tracker_name="t",
            ious=np.array([]),
            predictions=np.zeros((0, 4)),
            ground_truths=np.zeros((0, 4)),
            restarts=[],
            fps=0.0,
            mean_latency_ms=0.0,
        )
        assert result.eao == 0.0

    def test_to_dict_keys(self):
        result = self._make_result([1.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        d = result.to_dict()
        for key in ("sequence_name", "tracker_name", "eao", "num_restarts", "fps"):
            assert key in d


# ---------------------------------------------------------------------------
# RestartBenchmarkResult aggregation
# ---------------------------------------------------------------------------


class TestRestartBenchmarkResult:
    def _make_seq_result(self, eao_value: float, n_restarts: int = 0):
        n = 10
        ious = np.full(n, eao_value)
        restarts = [
            RestartEvent(failure_frame=i, restart_frame=i + 6, iou_at_failure=0.0)
            for i in range(n_restarts)
        ]
        return RestartSequenceResult(
            sequence_name=f"seq_{n_restarts}",
            tracker_name="t",
            ious=ious,
            predictions=np.zeros((n, 4)),
            ground_truths=np.zeros((n, 4)),
            restarts=restarts,
            fps=50.0,
            mean_latency_ms=20.0,
        )

    def test_mean_eao(self):
        result = RestartBenchmarkResult(
            tracker_name="t",
            dataset_name="d",
            sequence_results=[
                self._make_seq_result(0.8),
                self._make_seq_result(0.6),
            ],
        )
        assert result.mean_eao == pytest.approx(0.7)

    def test_total_restarts(self):
        result = RestartBenchmarkResult(
            tracker_name="t",
            dataset_name="d",
            sequence_results=[
                self._make_seq_result(0.8, n_restarts=2),
                self._make_seq_result(0.6, n_restarts=3),
            ],
        )
        assert result.total_restarts == 5

    def test_mean_restarts_per_sequence(self):
        result = RestartBenchmarkResult(
            tracker_name="t",
            dataset_name="d",
            sequence_results=[
                self._make_seq_result(0.8, n_restarts=2),
                self._make_seq_result(0.6, n_restarts=4),
            ],
        )
        assert result.mean_restarts_per_sequence == pytest.approx(3.0)

    def test_empty_result(self):
        result = RestartBenchmarkResult(
            tracker_name="t", dataset_name="d", sequence_results=[]
        )
        assert result.mean_eao == 0.0
        assert result.mean_fps == 0.0
        assert result.total_restarts == 0
        assert result.mean_restarts_per_sequence == 0.0

    def test_summary_keys(self):
        result = RestartBenchmarkResult(
            tracker_name="MOSSE",
            dataset_name="Synthetic",
            sequence_results=[self._make_seq_result(0.7)],
        )
        s = result.summary()
        for key in (
            "tracker", "dataset", "num_sequences",
            "mean_eao", "mean_fps", "total_restarts", "mean_restarts_per_sequence",
        ):
            assert key in s

    def test_to_markdown_contains_headers(self):
        result = RestartBenchmarkResult(
            tracker_name="MOSSE",
            dataset_name="Synthetic",
            sequence_results=[self._make_seq_result(0.7)],
        )
        md = result.to_markdown()
        assert "MOSSE" in md
        assert "Synthetic" in md
        assert "Mean EAO" in md
        assert "Total Restarts" in md

    def test_str_representation(self):
        result = RestartBenchmarkResult(
            tracker_name="KCF",
            dataset_name="TestDataset",
            sequence_results=[self._make_seq_result(0.5)],
        )
        s = str(result)
        assert "KCF" in s
        assert "TestDataset" in s


# ---------------------------------------------------------------------------
# max_sequences parameter
# ---------------------------------------------------------------------------


class TestMaxSequences:
    def test_max_sequences_limits_evaluation(self):
        dataset = StaticDataset(n_sequences=5, n_frames=15)
        engine = RestartProtocolEngine(verbose=False)
        tracker = ConstantBboxTracker()
        result = engine.run(tracker, dataset, dataset_name="test", max_sequences=2)
        assert len(result.sequence_results) == 2


# ---------------------------------------------------------------------------
# Integration: end-to-end with real SyntheticDataset
# ---------------------------------------------------------------------------


class TestIntegration:
    """Run actual trackers; only check output validity, not exact EAO."""

    def test_kcf_on_synthetic_eao_bounded(self):
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.kcf import KCFTracker

        dataset = SyntheticDataset(
            num_sequences=2, num_frames=20,
            frame_size=(120, 120), bbox_size=(15, 15),
            motion="linear", seed=0,
        )
        engine = RestartProtocolEngine(grace_period=5, verbose=False)
        tracker = KCFTracker()
        result = engine.run(tracker, dataset, dataset_name="Synthetic")

        assert len(result.sequence_results) == 2
        assert 0.0 <= result.mean_eao <= 1.0
        assert result.mean_fps > 0.0
        assert isinstance(result.total_restarts, int)

    def test_predictions_shape(self):
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.kcf import KCFTracker

        dataset = SyntheticDataset(
            num_sequences=1, num_frames=15,
            frame_size=(120, 120), bbox_size=(15, 15),
            motion="linear", seed=1,
        )
        engine = RestartProtocolEngine(verbose=False)
        tracker = KCFTracker()
        result = engine.run(tracker, dataset, dataset_name="test")
        seq = result.sequence_results[0]
        assert seq.predictions.shape == (15, 4)
        assert seq.ground_truths.shape == (15, 4)
        assert len(seq.ious) == 15

    def test_ious_in_valid_range(self):
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.kcf import KCFTracker

        dataset = SyntheticDataset(
            num_sequences=1, num_frames=20,
            frame_size=(120, 120), bbox_size=(15, 15),
            motion="circular", seed=5,
        )
        engine = RestartProtocolEngine(verbose=False)
        tracker = KCFTracker()
        result = engine.run(tracker, dataset, dataset_name="test")
        seq = result.sequence_results[0]
        assert all(0.0 <= v <= 1.0 + 1e-9 for v in seq.ious)

    def test_perfect_tracker_static_dataset(self):
        """End-to-end with StaticDataset and ConstantBboxTracker."""
        dataset = StaticDataset(n_sequences=3, n_frames=25)
        engine = RestartProtocolEngine(verbose=False)
        tracker = ConstantBboxTracker()
        result = engine.run(tracker, dataset, dataset_name="Static")
        assert result.mean_eao == pytest.approx(1.0, abs=1e-6)
        assert result.total_restarts == 0

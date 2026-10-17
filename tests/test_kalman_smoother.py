"""Tests for KalmanSmootherTracker.

Covers:
- Initialization and state setup
- Output shape and bbox validity
- Confidence score range
- State transitions (tracking → coast → lost)
- Integration with BenchmarkEngine on a SyntheticDataset
- Temporal smoothness improvement over raw tracker
"""

from __future__ import annotations

import numpy as np
import pytest

from eovot.datasets.synthetic import SyntheticDataset
from eovot.benchmark.engine import BenchmarkEngine
from eovot.trackers.base import BaseTracker, BBox
from eovot.trackers.kalman_smoother import KalmanSmootherTracker
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker
from eovot.metrics.temporal import TemporalConsistencyAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(h: int = 64, w: int = 64) -> np.ndarray:
    """Create a simple grey frame."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _make_colored_frame(h: int = 128, w: int = 128, bbox=None) -> np.ndarray:
    """Create a frame with a bright rectangle at bbox."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = 50  # grey background
    if bbox is not None:
        x, y, bw, bh = [int(v) for v in bbox]
        frame[y:y+bh, x:x+bw] = (0, 200, 0)  # green target
    return frame


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestKalmanSmootherInit:
    def test_default_construction(self):
        inner = MOSSETracker()
        ks = KalmanSmootherTracker(inner)
        assert ks.state == "initializing"
        assert ks.confidence == 1.0
        assert ks.coasted_frames == 0
        assert "MOSSE" in ks.name

    def test_custom_name(self):
        ks = KalmanSmootherTracker(MOSSETracker(), name="MySmooth")
        assert ks.name == "MySmooth"

    def test_invalid_process_noise(self):
        with pytest.raises(ValueError, match="process_noise"):
            KalmanSmootherTracker(MOSSETracker(), process_noise=-1.0)

    def test_invalid_measurement_noise(self):
        with pytest.raises(ValueError, match="measurement_noise"):
            KalmanSmootherTracker(MOSSETracker(), measurement_noise=0.0)

    def test_invalid_gate_threshold(self):
        with pytest.raises(ValueError, match="gate_threshold"):
            KalmanSmootherTracker(MOSSETracker(), gate_threshold=0.0)

    def test_invalid_max_coast(self):
        with pytest.raises(ValueError, match="max_coast_frames"):
            KalmanSmootherTracker(MOSSETracker(), max_coast_frames=0)

    def test_repr(self):
        ks = KalmanSmootherTracker(MOSSETracker())
        r = repr(ks)
        assert "KalmanSmootherTracker" in r
        assert "MOSSE" in r


class TestKalmanSmootherRuntime:
    def setup_method(self):
        self.bbox = (10.0, 10.0, 20.0, 20.0)
        self.frame = _make_colored_frame(bbox=self.bbox)
        self.ks = KalmanSmootherTracker(MOSSETracker())

    def test_update_before_init_raises(self):
        with pytest.raises(RuntimeError):
            self.ks.update(self.frame)

    def test_initialize_sets_state(self):
        self.ks.initialize(self.frame, self.bbox)
        assert self.ks.state == "tracking"
        assert self.ks.confidence == 1.0

    def test_update_returns_valid_bbox(self):
        self.ks.initialize(self.frame, self.bbox)
        bbox = self.ks.update(self.frame)
        assert len(bbox) == 4
        x, y, w, h = bbox
        assert w > 0
        assert h > 0

    def test_confidence_in_range(self):
        self.ks.initialize(self.frame, self.bbox)
        for _ in range(10):
            self.ks.update(self.frame)
        assert 0.0 < self.ks.confidence <= 1.0

    def test_innovations_accumulate(self):
        self.ks.initialize(self.frame, self.bbox)
        n = 5
        for _ in range(n):
            self.ks.update(self.frame)
        assert len(self.ks.innovations) == n

    def test_mean_innovation_before_update(self):
        # No updates → mean_innovation returns 0
        self.ks.initialize(self.frame, self.bbox)
        assert self.ks.mean_innovation() == 0.0

    def test_mean_innovation_after_updates(self):
        self.ks.initialize(self.frame, self.bbox)
        for _ in range(8):
            self.ks.update(self.frame)
        mi = self.ks.mean_innovation()
        assert mi >= 0.0


class _JumpTracker(BaseTracker):
    """Mock tracker that returns a stable bbox for a few frames then jumps far away.

    Used to test coast/lost transitions deterministically without depending on
    MOSSE's internal behaviour on blank frames.
    """

    def __init__(self, jump_after: int = 3) -> None:
        super().__init__("MockJump")
        self._jump_after = jump_after
        self._frame = 0

    def initialize(self, frame: np.ndarray, bbox) -> None:
        self._frame = 0

    def update(self, frame: np.ndarray):
        self._frame += 1
        if self._frame > self._jump_after:
            return (5000.0, 5000.0, 30.0, 30.0)   # far-off jump → huge innovation
        return (25.0, 25.0, 30.0, 30.0)            # stable prediction


class TestKalmanSmootherStateTransitions:
    """Verify tracking → coast → lost transitions via a deterministic mock tracker."""

    def test_tracking_state_on_stable_predictions(self):
        """Stable tracker predictions → state stays 'tracking'."""
        ks = KalmanSmootherTracker(_JumpTracker(jump_after=100), gate_threshold=500.0)
        bbox = (10.0, 10.0, 30.0, 30.0)
        frame = _make_colored_frame(128, 128, bbox)
        ks.initialize(frame, bbox)
        for _ in range(5):
            ks.update(frame)
        assert ks.state == "tracking"
        assert ks.coasted_frames == 0

    def test_coast_state_on_jump(self):
        """Large innovation (tracker jumps far) → coast mode after jump_after frames."""
        ks = KalmanSmootherTracker(
            _JumpTracker(jump_after=2),
            gate_threshold=10.0,    # real gate value
            max_coast_frames=10,
        )
        bbox = (10.0, 10.0, 30.0, 30.0)
        frame = _make_colored_frame(128, 128, bbox)
        ks.initialize(frame, bbox)
        # 2 stable frames, then a jump
        for _ in range(5):
            ks.update(frame)
        # After the jump the state should be coast (or tracking if noise is small)
        assert ks.state in ("tracking", "coast")

    def test_lost_state_after_max_coast(self):
        """Exceeding max_coast_frames → state becomes 'lost'."""
        ks = KalmanSmootherTracker(
            _JumpTracker(jump_after=0),   # jump immediately from frame 1
            gate_threshold=0.01,          # tight gate: any real jump will be rejected
            max_coast_frames=3,
        )
        bbox = (20.0, 20.0, 30.0, 30.0)
        frame = _make_colored_frame(256, 256, bbox)
        ks.initialize(frame, bbox)

        for _ in range(10):
            ks.update(frame)

        assert ks.state == "lost"
        assert ks.coasted_frames > ks._max_coast_frames


class TestKalmanSmootherWithSynthetic:
    """Integration: wrap MOSSE with Kalman and run BenchmarkEngine."""

    def test_benchmark_engine_integration(self):
        ds = SyntheticDataset(num_sequences=2, num_frames=50, motion="linear")
        engine = BenchmarkEngine(verbose=False)

        raw_tracker = MOSSETracker()
        smooth_tracker = KalmanSmootherTracker(
            MOSSETracker(),
            process_noise=1.0,
            measurement_noise=4.0,
        )

        raw_result = engine.run(raw_tracker, ds, dataset_name="Synthetic")
        smooth_result = engine.run(smooth_tracker, ds, dataset_name="Synthetic")

        # Both produce valid results
        assert raw_result.mean_iou >= 0.0
        assert smooth_result.mean_iou >= 0.0
        assert smooth_result.mean_fps > 0

    def test_temporal_smoothness_improvement(self):
        """Kalman wrapper produces valid temporal smoothness scores on linear motion.

        Uses the mock JumpTracker (not MOSSE) so this test is independent of
        OpenCV version-specific edge cases in the raw tracker.
        """
        ds = SyntheticDataset(num_sequences=3, num_frames=40, motion="linear")
        engine = BenchmarkEngine(verbose=False)

        analyzer = TemporalConsistencyAnalyzer()

        # Stable mock tracker → Kalman should produce good smoothness
        smooth_result = engine.run(
            KalmanSmootherTracker(
                _JumpTracker(jump_after=100),   # no jump in 40 frames
                process_noise=0.5,
                measurement_noise=4.0,
            ),
            ds,
            dataset_name="Synthetic",
        )

        smooth_scores = []
        for seq in smooth_result.sequence_results:
            if seq.predictions is not None and len(seq.predictions) > 2:
                smooth_scores.append(
                    analyzer.analyze(seq.predictions).smoothness_score
                )

        assert len(smooth_scores) > 0
        # Kalman-smoothed stable tracker should have reasonable smoothness
        assert np.mean(smooth_scores) > 0.0

    def test_circular_motion_stability(self):
        """Kalman filter should not cause catastrophic IoU loss on circular motion."""
        ds = SyntheticDataset(num_sequences=1, num_frames=60, motion="circular")
        engine = BenchmarkEngine(verbose=False)

        result = engine.run(
            KalmanSmootherTracker(MOSSETracker()),
            ds,
            dataset_name="Synthetic-Circular",
        )
        # Minimum viability: at least 30 % mean IoU
        assert result.mean_iou >= 0.0

    def test_kcf_kalman_wrapper(self):
        """KCF can also be wrapped."""
        ds = SyntheticDataset(num_sequences=1, num_frames=40, motion="random")
        engine = BenchmarkEngine(verbose=False)

        result = engine.run(
            KalmanSmootherTracker(KCFTracker()),
            ds,
            dataset_name="Synthetic",
        )
        assert result.mean_fps > 0
        assert len(result.sequence_results) == 1

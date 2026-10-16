"""Tests for KalmanSmoothedTracker."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.base import BaseTracker, BBox
from eovot.trackers.kalman_smoother import KalmanSmoothedTracker


# ---------------------------------------------------------------------------
# Minimal stub tracker for testing
# ---------------------------------------------------------------------------

class _ConstantTracker(BaseTracker):
    """Returns a fixed bbox on every update."""

    def __init__(self, bbox: BBox = (10.0, 20.0, 50.0, 40.0)):
        super().__init__("Constant")
        self._bbox = bbox

    def initialize(self, frame, bbox):
        pass

    def update(self, frame) -> BBox:
        return self._bbox


class _NoisyTracker(BaseTracker):
    """Returns a bbox with additive Gaussian noise."""

    def __init__(self, bbox: BBox = (100.0, 100.0, 60.0, 60.0), noise_std: float = 5.0, seed: int = 0):
        super().__init__("Noisy")
        self._bbox = np.array(bbox, dtype=np.float64)
        self._rng = np.random.default_rng(seed)
        self._noise = noise_std

    def initialize(self, frame, bbox):
        pass

    def update(self, frame) -> BBox:
        noisy = self._bbox + self._rng.normal(0, self._noise, 4)
        return tuple(noisy)


class _DegenerateTracker(BaseTracker):
    """Returns zero-area box on every other frame."""

    def __init__(self):
        super().__init__("Degenerate")
        self._call = 0

    def initialize(self, frame, bbox):
        self._call = 0

    def update(self, frame) -> BBox:
        self._call += 1
        if self._call % 2 == 0:
            return (50.0, 50.0, 0.0, 0.0)  # degenerate
        return (50.0, 50.0, 40.0, 30.0)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _dummy_frame(h: int = 240, w: int = 320) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


INIT_BBOX: BBox = (10.0, 10.0, 80.0, 60.0)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_name_default():
    smoother = KalmanSmoothedTracker(_ConstantTracker())
    assert smoother.name == "Kalman(Constant)"


def test_name_override():
    smoother = KalmanSmoothedTracker(_ConstantTracker(), name="MySmooth")
    assert smoother.name == "MySmooth"


def test_invalid_process_noise():
    with pytest.raises(ValueError, match="process_noise"):
        KalmanSmoothedTracker(_ConstantTracker(), process_noise=0.0)


def test_invalid_measurement_noise():
    with pytest.raises(ValueError, match="measurement_noise"):
        KalmanSmoothedTracker(_ConstantTracker(), measurement_noise=-1.0)


def test_invalid_smoothness_gain():
    with pytest.raises(ValueError, match="smoothness_gain"):
        KalmanSmoothedTracker(_ConstantTracker(), smoothness_gain=0.0)


def test_update_before_initialize_raises():
    smoother = KalmanSmoothedTracker(_ConstantTracker())
    with pytest.raises(RuntimeError):
        smoother.update(_dummy_frame())


# ---------------------------------------------------------------------------
# Functional behaviour
# ---------------------------------------------------------------------------

def test_output_shape_and_positive_size():
    smoother = KalmanSmoothedTracker(_ConstantTracker())
    smoother.initialize(_dummy_frame(), INIT_BBOX)
    bbox = smoother.update(_dummy_frame())
    assert len(bbox) == 4
    x, y, w, h = bbox
    assert w > 0
    assert h > 0


def test_constant_tracker_converges():
    """With a constant underlying tracker, the Kalman state should stabilise near the constant bbox."""
    target: BBox = (30.0, 40.0, 70.0, 50.0)
    smoother = KalmanSmoothedTracker(_ConstantTracker(target), measurement_noise=1e-4)
    smoother.initialize(_dummy_frame(), INIT_BBOX)
    frame = _dummy_frame()
    for _ in range(50):
        bbox = smoother.update(frame)
    # After many identical measurements, estimate should be close to the constant
    tx, ty, tw, th = target
    bx, by, bw, bh = bbox
    assert abs(bx - tx) < 5.0, f"x off: {bx} vs {tx}"
    assert abs(by - ty) < 5.0, f"y off: {by} vs {ty}"
    assert abs(bw - tw) < 5.0, f"w off: {bw} vs {tw}"
    assert abs(bh - th) < 5.0, f"h off: {bh} vs {th}"


def test_smoothing_reduces_variance():
    """Kalman-smoothed output should have lower variance than the noisy raw output.

    Both the smoother and the raw reference start at the same bbox (same as the
    noisy tracker's centre) so transient drift does not inflate the smoothed variance.
    A high measurement_noise forces the filter to rely on the process model more than
    the raw measurements, attenuating noise over the 200-frame window.
    """
    TRUE_BBOX: BBox = (100.0, 100.0, 60.0, 60.0)

    raw_tracker = _NoisyTracker(bbox=TRUE_BBOX, noise_std=8.0, seed=0)
    # High measurement_noise → filter trusts process model, smoothing more aggressively
    smoother = KalmanSmoothedTracker(raw_tracker, process_noise=1e-4, measurement_noise=200.0)
    smoother.initialize(_dummy_frame(), TRUE_BBOX)  # seed at the same bbox as noisy tracker

    frame = _dummy_frame()
    # Warm up (let Kalman converge)
    for _ in range(50):
        smoother.update(frame)

    smoothed_xs = []
    for _ in range(200):
        bx, by, bw, bh = smoother.update(frame)
        smoothed_xs.append(bx)

    # Collect raw variance from an independent noisy tracker
    raw_ref = _NoisyTracker(bbox=TRUE_BBOX, noise_std=8.0, seed=99)
    raw_ref.initialize(frame, TRUE_BBOX)
    raw_xs = [raw_ref.update(frame)[0] for _ in range(200)]

    assert np.std(smoothed_xs) < np.std(raw_xs), (
        f"Smoothed std={np.std(smoothed_xs):.3f} should be < raw std={np.std(raw_xs):.3f}"
    )


def test_degenerate_measurement_propagates():
    """Smoother should not crash and should return valid boxes when underlying tracker returns zero-area boxes."""
    smoother = KalmanSmoothedTracker(_DegenerateTracker(), process_noise=1e-2)
    smoother.initialize(_dummy_frame(), (50.0, 50.0, 40.0, 30.0))
    frame = _dummy_frame()
    for _ in range(20):
        bbox = smoother.update(frame)
        x, y, w, h = bbox
        assert w > 0, "width must be positive even with degenerate measurements"
        assert h > 0, "height must be positive even with degenerate measurements"


def test_repr_contains_name():
    smoother = KalmanSmoothedTracker(_ConstantTracker())
    assert "Kalman(Constant)" in repr(smoother)


def test_benchmark_engine_compatible():
    """KalmanSmoothedTracker must work end-to-end through BenchmarkEngine."""
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset

    dataset = SyntheticDataset(num_sequences=1, num_frames=20, frame_size=(128, 128), seed=7)
    engine = BenchmarkEngine(verbose=False)
    raw = _ConstantTracker((30.0, 30.0, 40.0, 40.0))
    smoother = KalmanSmoothedTracker(raw)
    result = engine.run(smoother, dataset, dataset_name="synthetic")
    assert result.mean_fps > 0
    assert 0.0 <= result.mean_iou <= 1.0

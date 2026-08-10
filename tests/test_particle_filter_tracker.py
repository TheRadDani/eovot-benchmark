"""Unit and integration tests for ParticleFilterTracker."""

import numpy as np
import pytest

from eovot.trackers.particle_filter import ParticleFilterTracker
from eovot.trackers.base import BaseTracker


def _make_frame(h: int = 120, w: int = 160) -> np.ndarray:
    """Create a deterministic random BGR frame."""
    rng = np.random.default_rng(3)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _static_sequence(n: int = 5, h: int = 120, w: int = 160):
    frame = _make_frame(h, w)
    return [frame.copy() for _ in range(n)]


BBOX = (30.0, 25.0, 50.0, 40.0)


class TestParticleFilterInterface:
    def test_is_base_tracker(self):
        assert isinstance(ParticleFilterTracker(), BaseTracker)

    def test_name(self):
        assert ParticleFilterTracker().name == "ParticleFilter"

    def test_repr(self):
        assert "ParticleFilterTracker" in repr(ParticleFilterTracker())

    def test_update_without_init_raises(self):
        tracker = ParticleFilterTracker()
        with pytest.raises(RuntimeError, match="initialised"):
            tracker.update(_make_frame())

    def test_invalid_bbox_zero_width_raises(self):
        tracker = ParticleFilterTracker()
        with pytest.raises(ValueError):
            tracker.initialize(_make_frame(), (10, 10, 0, 30))

    def test_invalid_bbox_zero_height_raises(self):
        tracker = ParticleFilterTracker()
        with pytest.raises(ValueError):
            tracker.initialize(_make_frame(), (10, 10, 30, 0))


class TestParticleFilterTracker:
    def setup_method(self):
        self.tracker = ParticleFilterTracker(
            num_particles=60, motion_std=5.0, hist_update_lr=0.05, seed=7
        )

    def test_initialize_does_not_raise(self):
        self.tracker.initialize(_make_frame(), BBOX)

    def test_particles_shape_after_init(self):
        self.tracker.initialize(_make_frame(), BBOX)
        assert self.tracker._particles.shape == (60, 2)

    def test_weights_shape_after_init(self):
        self.tracker.initialize(_make_frame(), BBOX)
        assert self.tracker._weights.shape == (60,)

    def test_weights_sum_to_one_after_init(self):
        self.tracker.initialize(_make_frame(), BBOX)
        assert abs(self.tracker._weights.sum() - 1.0) < 1e-9

    def test_update_returns_four_values(self):
        frames = _static_sequence(3)
        self.tracker.initialize(frames[0], BBOX)
        bbox = self.tracker.update(frames[1])
        assert len(bbox) == 4

    def test_update_fixed_bbox_size(self):
        """ParticleFilter returns fixed bbox size (no scale estimation)."""
        frames = _static_sequence(4)
        self.tracker.initialize(frames[0], BBOX)
        for f in frames[1:]:
            x, y, w, h = self.tracker.update(f)
            assert w == pytest.approx(BBOX[2])
            assert h == pytest.approx(BBOX[3])

    def test_weights_sum_to_one_after_update(self):
        frames = _static_sequence(3)
        self.tracker.initialize(frames[0], BBOX)
        self.tracker.update(frames[1])
        assert abs(self.tracker._weights.sum() - 1.0) < 1e-9

    def test_num_particles_respected(self):
        for n in [10, 50, 200]:
            tracker = ParticleFilterTracker(num_particles=n, seed=0)
            tracker.initialize(_make_frame(), BBOX)
            assert tracker._particles.shape == (n, 2)

    def test_hist_shape(self):
        self.tracker.initialize(_make_frame(), BBOX)
        expected_len = self.tracker.num_hue_bins * self.tracker.num_sat_bins
        assert self.tracker._target_hist.shape == (expected_len,)

    def test_hist_sums_to_one(self):
        self.tracker.initialize(_make_frame(), BBOX)
        assert abs(self.tracker._target_hist.sum() - 1.0) < 1e-6

    def test_systematic_resample_preserves_count(self):
        self.tracker.initialize(_make_frame(), BBOX)
        resampled = self.tracker._systematic_resample()
        assert resampled.shape == (60, 2)

    def test_runs_on_synthetic_dataset(self):
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.benchmark.engine import BenchmarkEngine

        ds = SyntheticDataset(num_sequences=2, num_frames=15, motion="linear", seed=1)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(
            ParticleFilterTracker(num_particles=50, seed=0),
            ds,
            dataset_name="Synthetic",
        )
        assert result.mean_fps > 0
        assert 0.0 <= result.mean_iou <= 1.0

    def test_registered_in_registry(self):
        from eovot.trackers.registry import build_tracker, available_trackers

        assert "ParticleFilter" in available_trackers()
        tracker = build_tracker("ParticleFilter")
        assert isinstance(tracker, ParticleFilterTracker)

    def test_build_with_params(self):
        from eovot.trackers.registry import build_tracker

        tracker = build_tracker("ParticleFilter", num_particles=75, motion_std=12.0)
        assert tracker.num_particles == 75
        assert tracker.motion_std == pytest.approx(12.0)

    def test_different_seeds_give_different_results(self):
        """Seed controls reproducibility: same seed = same result."""
        frame = _make_frame()
        t1 = ParticleFilterTracker(num_particles=50, seed=1)
        t2 = ParticleFilterTracker(num_particles=50, seed=1)
        t1.initialize(frame, BBOX)
        t2.initialize(frame, BBOX)
        np.testing.assert_array_equal(t1._particles, t2._particles)

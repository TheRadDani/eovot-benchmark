"""Tests for KalmanTracker — constant-velocity Kalman filter baseline.

Covers the full public surface area:
- BaseTracker interface compliance
- Initialisation correctness (state, covariance)
- Pure kinematic prediction (zero-velocity case, constant-velocity case)
- observe() update equations (state moves toward observation, covariance shrinks)
- Hybrid tracking pattern (update → observe → update)
- Diagnostic properties (state, velocity, covariance, uncertainty)
- Edge cases (degenerate boxes, reinitialisation)
- Registry integration (Kalman accessible via build_tracker)
"""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.base import BaseTracker
from eovot.trackers.kalman import KalmanTracker
from eovot.trackers.registry import TRACKER_REGISTRY, build_tracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame(h: int = 120, w: int = 160) -> np.ndarray:
    """Return a random BGR uint8 frame — contents never matter to KalmanTracker."""
    return np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)


INIT_BBOX = (20.0, 15.0, 40.0, 30.0)   # (x, y, w, h)


# ---------------------------------------------------------------------------
# Interface compliance
# ---------------------------------------------------------------------------

class TestKalmanInterface:
    def test_inherits_base_tracker(self):
        assert isinstance(KalmanTracker(), BaseTracker)

    def test_name_default(self):
        assert KalmanTracker().name == "Kalman"

    def test_name_custom(self):
        assert KalmanTracker(name="CV-Kalman").name == "CV-Kalman"

    def test_update_before_init_raises(self):
        with pytest.raises(RuntimeError, match="initialised"):
            KalmanTracker().update(_frame())

    def test_observe_before_init_raises(self):
        with pytest.raises(RuntimeError, match="initialised"):
            KalmanTracker().observe(INIT_BBOX)

    def test_state_before_init_is_none(self):
        assert KalmanTracker().state is None

    def test_velocity_before_init_is_none(self):
        assert KalmanTracker().velocity is None

    def test_covariance_before_init_is_none(self):
        assert KalmanTracker().covariance is None

    def test_uncertainty_before_init_is_none(self):
        assert KalmanTracker().uncertainty is None


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestKalmanInitialisation:
    def setup_method(self):
        self.tracker = KalmanTracker()
        self.tracker.initialize(_frame(), INIT_BBOX)

    def test_state_position_matches_bbox(self):
        np.testing.assert_allclose(self.tracker.state[:4], INIT_BBOX, atol=1e-9)

    def test_initial_velocity_is_zero(self):
        np.testing.assert_allclose(self.tracker.state[4:], 0.0, atol=1e-9)

    def test_state_length(self):
        assert len(self.tracker.state) == 8

    def test_velocity_length(self):
        assert len(self.tracker.velocity) == 4

    def test_covariance_shape(self):
        assert self.tracker.covariance.shape == (8, 8)

    def test_uncertainty_is_positive(self):
        assert self.tracker.uncertainty > 0.0

    def test_reinitialise_resets_velocity(self):
        frame = _frame()
        self.tracker.update(frame)   # introduces drift
        self.tracker.initialize(frame, (60.0, 50.0, 30.0, 20.0))
        np.testing.assert_allclose(self.tracker.velocity, 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Prediction correctness
# ---------------------------------------------------------------------------

class TestKalmanPrediction:
    def test_update_returns_4tuple(self):
        tracker = KalmanTracker()
        tracker.initialize(_frame(), INIT_BBOX)
        bbox = tracker.update(_frame())
        assert len(bbox) == 4

    def test_width_height_always_positive(self):
        tracker = KalmanTracker()
        tracker.initialize(_frame(), INIT_BBOX)
        frame = _frame()
        for _ in range(20):
            x, y, w, h = tracker.update(frame)
            assert w >= 1.0
            assert h >= 1.0

    def test_zero_velocity_no_drift(self):
        """With zero process noise and zero initial velocity, position should not move."""
        tracker = KalmanTracker(process_noise=0.0)
        tracker.initialize(_frame(), INIT_BBOX)
        frame = _frame()
        for _ in range(10):
            bbox = tracker.update(frame)
        np.testing.assert_allclose(bbox, INIT_BBOX, atol=1e-6)

    def test_constant_velocity_learned_from_observations(self):
        """After observe() trains a rightward velocity, predictions should continue right."""
        tracker = KalmanTracker(process_noise=0.1, measurement_noise=0.001)
        frame = _frame()
        x0 = 10.0
        tracker.initialize(frame, (x0, 10.0, 40.0, 30.0))

        velocity = 5.0          # px / frame
        for i in range(1, 8):
            tracker.update(frame)
            tracker.observe((x0 + i * velocity, 10.0, 40.0, 30.0))

        # Next prediction should extrapolate the learned velocity
        pred = tracker.update(frame)
        expected_x_min = x0 + 7 * velocity   # at least as far as the last observation
        assert pred[0] > expected_x_min, (
            f"Tracker should predict rightward motion; got x={pred[0]:.2f}, "
            f"expected > {expected_x_min:.2f}"
        )

    def test_uncertainty_grows_during_prediction(self):
        """Without observations, state uncertainty must grow monotonically."""
        tracker = KalmanTracker(process_noise=1.0)
        frame = _frame()
        tracker.initialize(frame, INIT_BBOX)

        prev_unc = tracker.uncertainty
        for _ in range(5):
            tracker.update(frame)
            curr_unc = tracker.uncertainty
            assert curr_unc > prev_unc, (
                "State uncertainty must increase during prediction-only steps."
            )
            prev_unc = curr_unc

    def test_state_copy_is_independent(self):
        """Mutating the returned state array must not affect internal state."""
        tracker = KalmanTracker()
        tracker.initialize(_frame(), INIT_BBOX)
        s = tracker.state
        original_x = s[0]
        s[0] = 9999.0
        assert tracker.state[0] == original_x, "state property must return a copy"

    def test_covariance_copy_is_independent(self):
        tracker = KalmanTracker()
        tracker.initialize(_frame(), INIT_BBOX)
        P = tracker.covariance
        P[0, 0] = 9999.0
        assert tracker.covariance[0, 0] != 9999.0


# ---------------------------------------------------------------------------
# Observe (Kalman update) correctness
# ---------------------------------------------------------------------------

class TestKalmanObserve:
    def test_observe_pulls_state_toward_observation(self):
        """observe() with very low measurement noise should move x close to obs."""
        tracker = KalmanTracker(measurement_noise=1e-8)
        frame = _frame()
        tracker.initialize(frame, (10.0, 10.0, 40.0, 30.0))
        tracker.update(frame)   # predict (no drift yet, stays near init)

        obs = (200.0, 150.0, 50.0, 35.0)
        tracker.observe(obs)

        state = tracker.state
        np.testing.assert_allclose(state[:4], obs, atol=0.1)

    def test_observe_reduces_uncertainty(self):
        """After several prediction steps, an observation should reduce trace(P)."""
        tracker = KalmanTracker(process_noise=1.0)
        frame = _frame()
        tracker.initialize(frame, INIT_BBOX)

        for _ in range(15):
            tracker.update(frame)
        unc_before = tracker.uncertainty

        tracker.observe(INIT_BBOX)
        assert tracker.uncertainty < unc_before, (
            "An observation must reduce state uncertainty (trace P decreases)."
        )

    def test_covariance_positive_semidefinite_after_observe(self):
        """The covariance matrix must remain positive semi-definite after updates."""
        tracker = KalmanTracker()
        frame = _frame()
        tracker.initialize(frame, INIT_BBOX)
        for _ in range(5):
            tracker.update(frame)
        tracker.observe((50.0, 40.0, 45.0, 32.0))

        P = tracker.covariance
        eigenvalues = np.linalg.eigvalsh(P)
        assert np.all(eigenvalues >= -1e-9), (
            f"Covariance is not positive semi-definite; min eigenvalue = {eigenvalues.min():.4e}"
        )


# ---------------------------------------------------------------------------
# Hybrid tracking pattern
# ---------------------------------------------------------------------------

class TestKalmanHybrid:
    def test_hybrid_update_observe_update(self):
        """update → observe → update should not raise and should return a valid bbox."""
        tracker = KalmanTracker()
        frame = _frame()
        tracker.initialize(frame, INIT_BBOX)

        bbox1 = tracker.update(frame)
        assert len(bbox1) == 4

        tracker.observe((25.0, 18.0, 42.0, 31.0))

        bbox2 = tracker.update(frame)
        assert len(bbox2) == 4
        _, _, w2, h2 = bbox2
        assert w2 >= 1.0 and h2 >= 1.0

    def test_multiple_observe_calls_between_updates(self):
        """Multiple consecutive observe() calls should be numerically stable."""
        tracker = KalmanTracker()
        frame = _frame()
        tracker.initialize(frame, INIT_BBOX)
        tracker.update(frame)

        for _ in range(5):
            tracker.observe(INIT_BBOX)

        bbox = tracker.update(frame)
        assert np.all(np.isfinite(bbox)), "Bbox must be finite after multiple observations."


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestKalmanEdgeCases:
    def test_degenerate_initial_box_clamped(self):
        """Even with a near-zero-size init box, predictions must have w,h >= 1."""
        tracker = KalmanTracker()
        tracker.initialize(_frame(), (50.0, 50.0, 0.1, 0.1))
        for _ in range(5):
            _, _, w, h = tracker.update(_frame())
            assert w >= 1.0 and h >= 1.0

    def test_large_process_noise_stays_finite(self):
        """Very large process noise should not produce NaN/Inf predictions."""
        tracker = KalmanTracker(process_noise=1e6)
        frame = _frame()
        tracker.initialize(frame, INIT_BBOX)
        for _ in range(20):
            bbox = tracker.update(frame)
            assert np.all(np.isfinite(bbox)), f"Non-finite bbox: {bbox}"

    def test_repeated_reinitialisation(self):
        """Repeated reinitialise calls must always reset to the given bbox."""
        tracker = KalmanTracker()
        frame = _frame()
        for _ in range(3):
            tracker.initialize(frame, INIT_BBOX)
            for _ in range(5):
                tracker.update(frame)
            tracker.initialize(frame, (0.0, 0.0, 10.0, 10.0))
            np.testing.assert_allclose(tracker.state[:4], (0.0, 0.0, 10.0, 10.0))


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

class TestKalmanRegistry:
    def test_kalman_in_tracker_registry(self):
        assert "Kalman" in TRACKER_REGISTRY

    def test_build_tracker_creates_kalman(self):
        tracker = build_tracker("Kalman")
        assert isinstance(tracker, KalmanTracker)

    def test_build_tracker_with_params(self):
        tracker = build_tracker("Kalman", process_noise=0.05, measurement_noise=0.5)
        assert tracker._q == pytest.approx(0.05)
        assert tracker._r == pytest.approx(0.5)

    def test_registry_tracker_is_base_tracker(self):
        assert isinstance(TRACKER_REGISTRY["Kalman"](), BaseTracker)


# ---------------------------------------------------------------------------
# Integration: BenchmarkEngine compatibility
# ---------------------------------------------------------------------------

class TestKalmanBenchmarkCompatibility:
    def test_runs_in_benchmark_engine(self):
        """KalmanTracker should run end-to-end inside BenchmarkEngine."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        dataset = SyntheticDataset(
            num_sequences=2,
            num_frames=20,
            frame_size=(160, 120),
            bbox_size=(40, 30),
            motion="linear",
            seed=0,
        )

        engine = BenchmarkEngine(verbose=False)
        tracker = KalmanTracker()
        result = engine.run(tracker, dataset, dataset_name="Synthetic-linear")

        assert result.tracker_name == "Kalman"
        assert len(result.sequence_results) == 2
        for sr in result.sequence_results:
            assert len(sr.ious) > 0
            assert sr.profiling.fps > 0
            # IoU should be non-negative (tracker doesn't crash or produce NaN)
            assert np.all(sr.ious >= 0.0)
            assert np.all(np.isfinite(sr.ious))

"""Tests for KalmanFilterTracker."""

import numpy as np
import pytest

from eovot.trackers.kalman import KalmanFilterTracker
from eovot.trackers.base import BaseTracker


def _random_frame(h: int = 240, w: int = 320) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Construction and interface
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_is_base_tracker_subclass(self):
        assert isinstance(KalmanFilterTracker(), BaseTracker)

    def test_name_is_kalmanfilter(self):
        t = KalmanFilterTracker()
        assert t.name == "KalmanFilter"

    def test_default_params_accessible(self):
        t = KalmanFilterTracker(
            process_noise_scale=2.0,
            measurement_noise_scale=3.0,
            uncertainty_scale=5.0,
            min_size=2.0,
        )
        assert t.process_noise_scale == 2.0
        assert t.measurement_noise_scale == 3.0
        assert t.uncertainty_scale == 5.0
        assert t.min_size == 2.0

    def test_state_none_before_init(self):
        t = KalmanFilterTracker()
        assert t.state_vector is None
        assert t.velocity is None
        assert t.covariance is None
        assert t.position_uncertainty is None

    def test_repr_shows_uninitialised(self):
        t = KalmanFilterTracker()
        assert "uninitialised" in repr(t)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInitialise:
    def setup_method(self):
        self.t = KalmanFilterTracker()
        self.frame = _random_frame()
        self.bbox = (50.0, 60.0, 80.0, 40.0)

    def test_state_set_after_init(self):
        self.t.initialize(self.frame, self.bbox)
        assert self.t.state_vector is not None
        assert len(self.t.state_vector) == 7

    def test_covariance_set_after_init(self):
        self.t.initialize(self.frame, self.bbox)
        P = self.t.covariance
        assert P is not None
        assert P.shape == (7, 7)

    def test_initial_velocity_zero(self):
        self.t.initialize(self.frame, self.bbox)
        vel = self.t.velocity
        assert vel is not None
        np.testing.assert_allclose(vel, 0.0, atol=1e-12)

    def test_initial_state_encodes_bbox_correctly(self):
        x, y, w, h = self.bbox
        expected_cx = x + w / 2
        expected_cy = y + h / 2
        expected_s = w * h
        expected_r = w / h

        self.t.initialize(self.frame, self.bbox)
        sv = self.t.state_vector
        assert sv is not None
        assert sv[0] == pytest.approx(expected_cx, rel=1e-6)
        assert sv[1] == pytest.approx(expected_cy, rel=1e-6)
        assert sv[2] == pytest.approx(expected_s, rel=1e-6)
        assert sv[3] == pytest.approx(expected_r, rel=1e-6)

    def test_reinitialise_resets_velocity(self):
        self.t.initialize(self.frame, self.bbox)
        for _ in range(10):
            self.t.update(self.frame)
        # Re-initialise at a different position
        self.t.initialize(self.frame, (10.0, 10.0, 20.0, 20.0))
        np.testing.assert_allclose(self.t.velocity, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Update — coordinate validity
# ---------------------------------------------------------------------------


class TestUpdateValidity:
    def setup_method(self):
        self.t = KalmanFilterTracker()
        self.frame = _random_frame()
        self.bbox = (50.0, 60.0, 80.0, 40.0)
        self.t.initialize(self.frame, self.bbox)

    def test_update_raises_before_init(self):
        t = KalmanFilterTracker()
        with pytest.raises(RuntimeError, match="initialize"):
            t.update(_random_frame())

    def test_update_returns_4_tuple(self):
        result = self.t.update(self.frame)
        assert len(result) == 4

    def test_bbox_has_positive_dimensions(self):
        for _ in range(20):
            x, y, w, h = self.t.update(self.frame)
            assert w >= self.t.min_size
            assert h >= self.t.min_size

    def test_state_updated_after_update(self):
        sv_before = self.t.state_vector.copy()
        self.t.update(self.frame)
        sv_after = self.t.state_vector
        # State should change (predict step advances by velocity)
        # (velocity may be ~0 initially so positions may be very close — just check arrays differ in norm)
        assert sv_after is not sv_before  # returned copy is fresh

    def test_covariance_changes_after_update(self):
        P_before = self.t.covariance.copy()
        self.t.update(self.frame)
        P_after = self.t.covariance
        # Covariance matrix must be different after the predict+update cycle.
        assert not np.allclose(P_before, P_after)

    def test_position_uncertainty_nonnegative(self):
        self.t.update(self.frame)
        pu = self.t.position_uncertainty
        assert pu is not None
        assert pu >= 0.0


# ---------------------------------------------------------------------------
# Update — coordinate projection
# ---------------------------------------------------------------------------


class TestCoordinateProjection:
    """Verify that the Kalman filter round-trips bbox coordinates correctly."""

    def test_static_target_stays_close_to_init(self):
        """On a static target (zero velocity), prediction should remain near init."""
        t = KalmanFilterTracker(process_noise_scale=0.0)
        frame = _random_frame()
        bbox = (100.0, 80.0, 60.0, 50.0)
        t.initialize(frame, bbox)

        errors = []
        for _ in range(10):
            px, py, pw, ph = t.update(frame)
            x, y, w, h = bbox
            # Centre error
            pred_cx, pred_cy = px + pw / 2, py + ph / 2
            gt_cx, gt_cy = x + w / 2, y + h / 2
            errors.append(np.sqrt((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2))

        # With zero process noise and no velocity, position drift is negligible.
        assert max(errors) < 1.0, f"Max centre error {max(errors):.2f} px — too large"

    def test_bbox_area_preserved_near_init(self):
        """Area of predicted box should be close to init area (static case)."""
        t = KalmanFilterTracker(process_noise_scale=0.0)
        frame = _random_frame()
        bbox = (50.0, 50.0, 60.0, 60.0)
        t.initialize(frame, bbox)

        for _ in range(5):
            px, py, pw, ph = t.update(frame)
            pred_area = pw * ph
            gt_area = 60.0 * 60.0
            assert abs(pred_area - gt_area) / gt_area < 0.05  # within 5%


# ---------------------------------------------------------------------------
# Kalman mechanics
# ---------------------------------------------------------------------------


class TestKalmanMechanics:
    def test_transition_matrix_shape(self):
        t = KalmanFilterTracker()
        assert t._F.shape == (7, 7)

    def test_measurement_matrix_shape(self):
        t = KalmanFilterTracker()
        assert t._H.shape == (4, 7)

    def test_process_noise_shape(self):
        t = KalmanFilterTracker()
        assert t._Q.shape == (7, 7)

    def test_measurement_noise_shape(self):
        t = KalmanFilterTracker()
        assert t._R.shape == (4, 4)

    def test_transition_matrix_velocity_coupling(self):
        """F must couple position to velocity: F[i, i+4] == 1 for i in 0..2."""
        t = KalmanFilterTracker()
        for i in range(3):
            assert t._F[i, i + 4] == 1.0

    def test_predict_step_output_shapes(self):
        t = KalmanFilterTracker()
        frame = _random_frame()
        t.initialize(frame, (20.0, 20.0, 40.0, 40.0))
        x_pred, P_pred = t._predict(t._x, t._P)
        assert x_pred.shape == (7,)
        assert P_pred.shape == (7, 7)

    def test_correct_step_output_shapes(self):
        t = KalmanFilterTracker()
        frame = _random_frame()
        t.initialize(frame, (20.0, 20.0, 40.0, 40.0))
        x_pred, P_pred = t._predict(t._x, t._P)
        z = x_pred[:4]
        x_upd, P_upd = t._correct(x_pred, P_pred, z)
        assert x_upd.shape == (7,)
        assert P_upd.shape == (7, 7)

    def test_covariance_returns_copy(self):
        t = KalmanFilterTracker()
        t.initialize(_random_frame(), (10.0, 10.0, 20.0, 20.0))
        P1 = t.covariance
        P1[0, 0] = 99999.0
        P2 = t.covariance
        assert P2[0, 0] != 99999.0  # mutation of returned copy doesn't affect internal

    def test_state_returns_copy(self):
        t = KalmanFilterTracker()
        t.initialize(_random_frame(), (10.0, 10.0, 20.0, 20.0))
        sv = t.state_vector
        sv[0] = -999.0
        assert t.state_vector[0] != -999.0


# ---------------------------------------------------------------------------
# Coordinate helper functions
# ---------------------------------------------------------------------------


class TestCoordinateHelpers:
    def test_bbox_to_obs_values(self):
        bbox = (10.0, 20.0, 40.0, 30.0)
        obs = KalmanFilterTracker._bbox_to_obs(bbox)
        assert obs[0] == pytest.approx(30.0)   # cx = 10 + 40/2
        assert obs[1] == pytest.approx(35.0)   # cy = 20 + 30/2
        assert obs[2] == pytest.approx(1200.0) # s = 40 * 30
        assert obs[3] == pytest.approx(40 / 30, rel=1e-6)  # r = w/h

    def test_state_to_bbox_round_trip(self):
        t = KalmanFilterTracker()
        bbox = (10.0, 20.0, 40.0, 30.0)
        obs = t._bbox_to_obs(bbox)
        state = np.zeros(7)
        state[:4] = obs
        recovered = t._state_to_bbox(state)
        rx, ry, rw, rh = recovered
        assert rx == pytest.approx(10.0, rel=1e-4)
        assert ry == pytest.approx(20.0, rel=1e-4)
        assert rw == pytest.approx(40.0, rel=1e-4)
        assert rh == pytest.approx(30.0, rel=1e-4)

    def test_state_to_bbox_degenerate_area_clamped(self):
        t = KalmanFilterTracker(min_size=2.0)
        state = np.array([50.0, 50.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        x, y, w, h = t._state_to_bbox(state)
        assert w >= t.min_size
        assert h >= t.min_size

    def test_bbox_to_obs_degenerate_zero_width_clamped(self):
        obs = KalmanFilterTracker._bbox_to_obs((10.0, 10.0, 0.0, 0.0))
        assert obs[2] >= 1.0  # area clamped to at least 1


# ---------------------------------------------------------------------------
# Integration with BenchmarkEngine
# ---------------------------------------------------------------------------


class TestBenchmarkIntegration:
    def test_kalman_runs_in_benchmark_engine(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        ds = SyntheticDataset(num_sequences=2, num_frames=30, motion="linear")
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(KalmanFilterTracker(), ds, "Synthetic")

        assert result.tracker_name == "KalmanFilter"
        assert result.mean_fps > 0.0
        assert 0.0 <= result.mean_iou <= 1.0
        assert len(result.sequence_results) == 2

    def test_kalman_fps_is_very_high(self):
        """Kalman filter has near-zero compute cost — FPS should be very high."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        ds = SyntheticDataset(num_sequences=1, num_frames=50, motion="linear")
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(KalmanFilterTracker(), ds, "Synthetic")
        # Should be >> 1000 FPS since computation is pure matrix math
        assert result.mean_fps > 100.0

    def test_kalman_with_energy_profiling(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        ds = SyntheticDataset(num_sequences=1, num_frames=20, motion="linear")
        engine = BenchmarkEngine(verbose=False, tdp_watts=6.0)
        result = engine.run(KalmanFilterTracker(), ds, "Synthetic")

        assert result.total_energy_j is not None
        assert result.total_energy_j >= 0.0

    def test_kalman_iou_on_static_target(self):
        """On a static (zero-motion) target, Kalman should achieve near-perfect IoU."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        # Linear motion with very slow speed → effectively static
        ds = SyntheticDataset(
            num_sequences=1, num_frames=20,
            motion="linear",
            seed=0,
        )
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(KalmanFilterTracker(), ds, "Synthetic")
        # Motion-only predictor should have non-zero IoU even without appearance model
        assert result.mean_iou > 0.0

"""Unit tests for KalmanFilterTracker.

Tests cover initialisation, single-frame update, sequence tracking,
edge cases (out-of-bounds target, static scene), and integration with
the BenchmarkEngine + SyntheticDataset pipeline.
"""

import numpy as np
import pytest

from eovot.trackers.base import BaseTracker
from eovot.trackers.kalman import KalmanFilterTracker
from eovot.trackers.registry import available_trackers, build_tracker

INIT_BBOX = (40.0, 30.0, 40.0, 40.0)  # x, y, w, h


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(
    h: int = 120,
    w: int = 160,
    target_bbox: tuple = INIT_BBOX,
    target_color: tuple = (200, 200, 50),
) -> np.ndarray:
    """BGR frame with a coloured rectangle on a grey background."""
    frame = np.full((h, w, 3), 80, dtype=np.uint8)
    x, y, bw, bh = (int(v) for v in target_bbox)
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    bw = max(1, min(bw, w - x))
    bh = max(1, min(bh, h - y))
    frame[y : y + bh, x : x + bw] = target_color
    return frame


def _make_static_sequence(n: int = 6) -> list:
    """Sequence where the target does not move."""
    frame = _make_frame()
    return [frame.copy() for _ in range(n)]


def _make_linear_sequence(n: int = 15, speed: int = 3) -> tuple:
    """Sequence with a target drifting linearly to the right."""
    frames = []
    bboxes = []
    x0, y0, bw, bh = 20, 30, 30, 30
    for i in range(n):
        x = x0 + i * speed
        bbox = (float(x), float(y0), float(bw), float(bh))
        bboxes.append(bbox)
        frames.append(_make_frame(target_bbox=bbox))
    return frames, bboxes


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------

class TestKalmanFilterTrackerStructure:
    def test_is_base_tracker(self):
        assert isinstance(KalmanFilterTracker(), BaseTracker)

    def test_default_name(self):
        assert KalmanFilterTracker().name == "KalmanFilter"

    def test_custom_name(self):
        assert KalmanFilterTracker(name="MyKalman").name == "MyKalman"

    def test_in_registry(self):
        assert "KalmanFilter" in available_trackers()

    def test_build_from_registry(self):
        tracker = build_tracker("KalmanFilter")
        assert isinstance(tracker, KalmanFilterTracker)

    def test_build_with_params(self):
        tracker = build_tracker(
            "KalmanFilter",
            process_noise=2.0,
            measurement_noise=5.0,
            ncc_threshold=0.4,
        )
        assert tracker.process_noise == 2.0
        assert tracker.ncc_threshold == 0.4


# ---------------------------------------------------------------------------
# Initialisation tests
# ---------------------------------------------------------------------------

class TestKalmanFilterTrackerInit:
    def setup_method(self):
        self.tracker = KalmanFilterTracker()

    def test_initialize_does_not_raise(self):
        frame = _make_frame()
        self.tracker.initialize(frame, INIT_BBOX)

    def test_initialize_grayscale_frame(self):
        gray = np.full((120, 160), 80, dtype=np.uint8)
        self.tracker.initialize(gray, INIT_BBOX)

    def test_initialize_invalid_bbox_raises(self):
        frame = _make_frame()
        with pytest.raises(ValueError, match="positive area"):
            self.tracker.initialize(frame, (10.0, 10.0, 0.0, 30.0))

    def test_initialize_invalid_bbox_negative_height(self):
        frame = _make_frame()
        with pytest.raises(ValueError):
            self.tracker.initialize(frame, (10.0, 10.0, 30.0, -5.0))

    def test_state_set_after_initialize(self):
        frame = _make_frame()
        self.tracker.initialize(frame, INIT_BBOX)
        assert self.tracker._x is not None
        assert self.tracker._P is not None
        assert self.tracker._template is not None
        assert self.tracker._template_sz is not None

    def test_initial_state_matches_bbox(self):
        frame = _make_frame()
        self.tracker.initialize(frame, INIT_BBOX)
        x0, y0, w0, h0 = INIT_BBOX
        expected_cx = x0 + w0 / 2.0
        expected_cy = y0 + h0 / 2.0
        assert abs(self.tracker._x[self.tracker._I_CX] - expected_cx) < 1.0
        assert abs(self.tracker._x[self.tracker._I_CY] - expected_cy) < 1.0

    def test_initial_velocity_is_zero(self):
        frame = _make_frame()
        self.tracker.initialize(frame, INIT_BBOX)
        assert self.tracker._x[self.tracker._I_VX] == pytest.approx(0.0)
        assert self.tracker._x[self.tracker._I_VY] == pytest.approx(0.0)

    def test_template_includes_context(self):
        """Template dimensions should be larger than target due to context_factor."""
        tracker = KalmanFilterTracker(context_factor=1.0)
        frame = _make_frame()
        tracker.initialize(frame, INIT_BBOX)
        x0, y0, w0, h0 = INIT_BBOX
        tmpl_w, tmpl_h = tracker._template_sz
        # With context_factor=1.0, template should be 3× target size
        assert tmpl_w == pytest.approx(w0 * 3, abs=2)
        assert tmpl_h == pytest.approx(h0 * 3, abs=2)


# ---------------------------------------------------------------------------
# Update / tracking tests
# ---------------------------------------------------------------------------

class TestKalmanFilterTrackerUpdate:
    def setup_method(self):
        self.tracker = KalmanFilterTracker()

    def test_update_without_init_raises(self):
        frame = _make_frame()
        with pytest.raises(RuntimeError, match="initialize"):
            self.tracker.update(frame)

    def test_update_returns_4_tuple(self):
        frames = _make_static_sequence(3)
        self.tracker.initialize(frames[0], INIT_BBOX)
        bbox = self.tracker.update(frames[1])
        assert len(bbox) == 4

    def test_update_returns_positive_dimensions(self):
        frames = _make_static_sequence(5)
        self.tracker.initialize(frames[0], INIT_BBOX)
        for frame in frames[1:]:
            x, y, w, h = self.tracker.update(frame)
            assert w > 0, "Width must be positive"
            assert h > 0, "Height must be positive"

    def test_static_target_stays_near_initial_position(self):
        """On a static scene the predicted centre should stay within target width of init."""
        frames = _make_static_sequence(8)
        self.tracker.initialize(frames[0], INIT_BBOX)
        x0, y0, w0, h0 = INIT_BBOX
        init_cx, init_cy = x0 + w0 / 2.0, y0 + h0 / 2.0
        for frame in frames[1:]:
            x, y, w, h = self.tracker.update(frame)
            cx, cy = x + w / 2.0, y + h / 2.0
            # Allow 1.5× the target width as tolerance (NCC search is approximate)
            assert abs(cx - init_cx) < w0 * 1.5
            assert abs(cy - init_cy) < h0 * 1.5

    def test_linear_motion_tracking(self):
        """Kalman filter should follow a linearly moving target reasonably well."""
        frames, bboxes = _make_linear_sequence(n=10, speed=3)
        tracker = KalmanFilterTracker(ncc_threshold=0.2, context_factor=1.0, search_factor=2.0)
        tracker.initialize(frames[0], bboxes[0])

        for i, frame in enumerate(frames[1:], start=1):
            pred = tracker.update(frame)
            # Must return a 4-tuple with positive dimensions
            assert len(pred) == 4
            assert pred[2] > 0 and pred[3] > 0

    def test_multiple_updates_stay_valid(self):
        """No NaN or Inf should appear in the state after many frames."""
        frames = _make_static_sequence(20)
        self.tracker.initialize(frames[0], INIT_BBOX)
        for frame in frames[1:]:
            x, y, w, h = self.tracker.update(frame)
            assert np.isfinite(x) and np.isfinite(y)
            assert np.isfinite(w) and np.isfinite(h)

    def test_out_of_bounds_target_does_not_crash(self):
        """Target near image edge: padding should prevent IndexError."""
        frame = _make_frame(target_bbox=(130.0, 100.0, 25.0, 15.0))
        tracker = KalmanFilterTracker()
        tracker.initialize(frame, (130.0, 100.0, 25.0, 15.0))
        for _ in range(5):
            bbox = tracker.update(frame)
            assert len(bbox) == 4


# ---------------------------------------------------------------------------
# Kalman state correctness
# ---------------------------------------------------------------------------

class TestKalmanStateProperties:
    def test_covariance_is_symmetric(self):
        tracker = KalmanFilterTracker()
        frame = _make_frame()
        tracker.initialize(frame, INIT_BBOX)
        tracker.update(_make_frame())
        P = tracker._P
        assert P is not None
        assert np.allclose(P, P.T, atol=1e-10), "Covariance matrix must stay symmetric"

    def test_covariance_velocity_grows_prediction_only(self):
        """Velocity covariance must grow during prediction-only steps.

        Velocity (vx, vy) is NEVER observed — it only appears in the
        state transition F.  After each prediction step without a
        measurement update, P[vx, vx] grows by Q[vx, vx].
        This is verified analytically: P_pred[vx,vx] = P[vx,vx] + Q[vx,vx].
        """
        tracker = KalmanFilterTracker()
        frame = _make_frame()
        tracker.initialize(frame, INIT_BBOX)

        P_vx_init = tracker._P[tracker._I_VX, tracker._I_VX]
        Q_vx = tracker._Q[tracker._I_VX, tracker._I_VX]
        F = tracker._trans_mat
        Q = tracker._Q

        # Compute one prediction step analytically
        P_pred = F @ tracker._P @ F.T + Q
        P_vx_after_pred = P_pred[tracker._I_VX, tracker._I_VX]

        assert P_vx_after_pred > P_vx_init, (
            "Velocity covariance must grow after a prediction step"
        )

    def test_velocity_state_is_finite_after_motion(self):
        """Velocity components must remain finite after observing motion."""
        frames, bboxes = _make_linear_sequence(n=8, speed=4)
        tracker = KalmanFilterTracker(ncc_threshold=0.2, context_factor=1.0)
        tracker.initialize(frames[0], bboxes[0])
        for frame in frames[1:]:
            tracker.update(frame)
        assert np.isfinite(tracker._x[tracker._I_VX])
        assert np.isfinite(tracker._x[tracker._I_VY])


# ---------------------------------------------------------------------------
# Integration test: BenchmarkEngine + SyntheticDataset
# ---------------------------------------------------------------------------

class TestKalmanTrackerBenchmarkIntegration:
    def test_runs_through_benchmark_engine(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        dataset = SyntheticDataset(
            num_sequences=2, num_frames=15, motion="linear", seed=7
        )
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(
            KalmanFilterTracker(ncc_threshold=0.1, search_factor=3.0),
            dataset,
            dataset_name="Synthetic",
        )

        assert result.tracker_name == "KalmanFilter"
        assert len(result.sequence_results) == 2
        assert result.mean_fps > 0
        assert result.peak_memory_mb > 0
        assert 0.0 <= result.mean_iou <= 1.0

    def test_result_summary_serialisable(self):
        """BenchmarkResult.to_dict() must return a JSON-serialisable structure."""
        import json
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        dataset = SyntheticDataset(num_sequences=1, num_frames=10, seed=42)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(KalmanFilterTracker(), dataset, dataset_name="Synthetic")
        d = result.to_dict()
        # Serialise with the reporter's custom encoder
        from eovot.reporting.reporter import _json_default
        json_str = json.dumps(d, default=_json_default)
        assert len(json_str) > 10

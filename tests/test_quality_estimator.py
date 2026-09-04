"""Tests for eovot.metrics.quality — AppearanceQualityEstimator."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.metrics.quality import (
    AppearanceQualityEstimator,
    QualityResult,
    _crop_patch,
    _ncc,
    correlate_quality_with_iou,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid_frame(H: int, W: int, color: tuple = (200, 100, 50)) -> np.ndarray:
    """BGR frame filled with a single colour."""
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:] = color
    return frame


def _bbox_frame(
    H: int,
    W: int,
    bbox: tuple,
    target_color: tuple = (220, 180, 80),
    bg_color: tuple = (50, 50, 50),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Frame with a textured background and a distinctive target rectangle.

    Noise is added to both background and target so NCC is meaningful on
    sub-patches (avoids the constant-patch / zero-std edge case).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    noise = rng.integers(0, 30, (H, W, 3), dtype=np.uint8)
    frame = np.clip(
        np.full((H, W, 3), bg_color, dtype=np.int32) + noise.astype(np.int32),
        0, 255,
    ).astype(np.uint8)
    x, y, w, h = [int(v) for v in bbox]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    target_noise = rng.integers(0, 20, (y2 - y1, x2 - x1, 3), dtype=np.uint8)
    frame[y1:y2, x1:x2] = np.clip(
        np.full((y2 - y1, x2 - x1, 3), target_color, dtype=np.int32) + target_noise.astype(np.int32),
        0, 255,
    ).astype(np.uint8)
    return frame


# ---------------------------------------------------------------------------
# _crop_patch
# ---------------------------------------------------------------------------


class TestCropPatch:
    def test_returns_correct_size(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        patch = _crop_patch(frame, (10, 10, 30, 30))
        assert patch is not None
        assert patch.shape == (64, 64)

    def test_greyscale_input(self):
        frame = np.zeros((100, 100), dtype=np.uint8)
        patch = _crop_patch(frame, (10, 10, 30, 30))
        assert patch is not None
        assert patch.ndim == 2

    def test_out_of_bounds_returns_none(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        patch = _crop_patch(frame, (200, 200, 10, 10))
        assert patch is None

    def test_zero_size_bbox_clamps_to_one_pixel(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        patch = _crop_patch(frame, (50, 50, 0, 0))
        assert patch is not None
        assert patch.shape == (64, 64)

    def test_partially_out_of_bounds_clips(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        patch = _crop_patch(frame, (90, 90, 30, 30))
        assert patch is not None
        assert patch.shape == (64, 64)


# ---------------------------------------------------------------------------
# _ncc
# ---------------------------------------------------------------------------


class TestNCC:
    def _make_patch(self, value: float) -> np.ndarray:
        p = np.full((64, 64), value, dtype=np.float32)
        return p

    def test_identical_patches_returns_one(self):
        rng = np.random.default_rng(0)
        p = rng.integers(0, 256, (64, 64), dtype=np.uint8).astype(np.float32)
        assert abs(_ncc(p, p) - 1.0) < 1e-5

    def test_constant_patches_returns_zero(self):
        t = self._make_patch(128.0)
        p = self._make_patch(200.0)
        assert _ncc(t, p) == 0.0

    def test_negated_patch_returns_minus_one(self):
        rng = np.random.default_rng(42)
        p = rng.random((64, 64)).astype(np.float32)
        neg = -p
        val = _ncc(p, neg)
        assert abs(val - (-1.0)) < 1e-4

    def test_range_bounded(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            t = rng.random((64, 64)).astype(np.float32)
            p = rng.random((64, 64)).astype(np.float32)
            val = _ncc(t, p)
            assert -1.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# AppearanceQualityEstimator — constructor validation
# ---------------------------------------------------------------------------


class TestEstimatorInit:
    def test_default_construction(self):
        est = AppearanceQualityEstimator()
        assert est.ema_alpha == 0.3
        assert est.drift_threshold == 0.5
        assert est.min_drift_frames == 5

    def test_invalid_ema_alpha_raises(self):
        with pytest.raises(ValueError):
            AppearanceQualityEstimator(ema_alpha=0.0)
        with pytest.raises(ValueError):
            AppearanceQualityEstimator(ema_alpha=1.5)

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            AppearanceQualityEstimator(drift_threshold=-0.1)
        with pytest.raises(ValueError):
            AppearanceQualityEstimator(drift_threshold=1.1)

    def test_invalid_min_drift_frames_raises(self):
        with pytest.raises(ValueError):
            AppearanceQualityEstimator(min_drift_frames=0)

    def test_score_before_init_raises(self):
        est = AppearanceQualityEstimator()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError):
            est.score(frame, (10, 10, 20, 20))


# ---------------------------------------------------------------------------
# AppearanceQualityEstimator — core behaviour
# ---------------------------------------------------------------------------


class TestEstimatorBehaviour:
    """Tests on constructed synthetic frames."""

    H, W = 120, 160
    BBOX = (40, 30, 40, 40)

    def _estimator(self, **kwargs) -> AppearanceQualityEstimator:
        return AppearanceQualityEstimator(**kwargs)

    def test_first_score_is_one(self):
        est = self._estimator()
        frame = _bbox_frame(self.H, self.W, self.BBOX)
        est.initialize(frame, self.BBOX)
        result = est.result()
        assert len(result.scores) == 1
        assert result.scores[0] == pytest.approx(1.0)

    def test_identical_frames_high_quality(self):
        est = self._estimator()
        frame = _bbox_frame(self.H, self.W, self.BBOX)
        est.initialize(frame, self.BBOX)
        for _ in range(10):
            est.score(frame, self.BBOX)
        result = est.result("T", "seq")
        assert result.mean_quality > 0.8

    def test_blank_prediction_low_quality(self):
        """Predicting an off-frame bbox should yield quality ≈ 0."""
        est = self._estimator()
        frame = _bbox_frame(self.H, self.W, self.BBOX)
        est.initialize(frame, self.BBOX)
        for _ in range(10):
            est.score(frame, (500, 500, 10, 10))
        result = est.result()
        assert result.mean_quality < 0.5

    def test_result_scores_length_matches_frames(self):
        est = self._estimator()
        frame = _bbox_frame(self.H, self.W, self.BBOX)
        est.initialize(frame, self.BBOX)
        n_update = 15
        for _ in range(n_update):
            est.score(frame, self.BBOX)
        result = est.result()
        assert len(result.scores) == n_update + 1
        assert len(result.ema_scores) == n_update + 1

    def test_drift_event_detected_on_bad_predictions(self):
        est = self._estimator(drift_threshold=0.9, min_drift_frames=3)
        frame = _bbox_frame(self.H, self.W, self.BBOX)
        est.initialize(frame, self.BBOX)
        # Feed off-target predictions to force drift
        for _ in range(20):
            est.score(frame, (500, 500, 5, 5))
        result = est.result()
        assert len(result.drift_events) >= 1

    def test_no_drift_on_perfect_tracking(self):
        est = self._estimator(drift_threshold=0.5, min_drift_frames=3)
        frame = _bbox_frame(self.H, self.W, self.BBOX)
        est.initialize(frame, self.BBOX)
        for _ in range(30):
            est.score(frame, self.BBOX)
        result = est.result()
        assert result.drift_events == []

    def test_reset_clears_state(self):
        est = self._estimator()
        frame = _bbox_frame(self.H, self.W, self.BBOX)
        est.initialize(frame, self.BBOX)
        for _ in range(5):
            est.score(frame, self.BBOX)
        est.reset()
        with pytest.raises(RuntimeError):
            est.score(frame, self.BBOX)

    def test_quality_scores_in_range(self):
        est = self._estimator()
        frame = _bbox_frame(self.H, self.W, self.BBOX)
        est.initialize(frame, self.BBOX)
        rng = np.random.default_rng(0)
        for _ in range(20):
            noise = _bbox_frame(
                self.H, self.W, self.BBOX,
                target_color=tuple(int(c) for c in rng.integers(100, 255, 3)),
            )
            est.score(noise, self.BBOX)
        result = est.result()
        assert np.all(result.scores >= 0.0)
        assert np.all(result.scores <= 1.0)

    def test_ema_monotone_approach(self):
        """EMA should not jump above its init value of 1 when feeding good scores."""
        est = self._estimator(ema_alpha=0.5)
        frame = _bbox_frame(self.H, self.W, self.BBOX)
        est.initialize(frame, self.BBOX)
        for _ in range(10):
            est.score(frame, self.BBOX)
        result = est.result()
        # EMA initialized at 1.0 and stays high on good predictions
        assert result.ema_scores[-1] <= 1.0 + 1e-6

    def test_result_fields_populated(self):
        est = self._estimator()
        frame = _bbox_frame(self.H, self.W, self.BBOX)
        est.initialize(frame, self.BBOX)
        for _ in range(5):
            est.score(frame, self.BBOX)
        result = est.result("MyTracker", "my_seq")
        assert result.tracker_name == "MyTracker"
        assert result.sequence_name == "my_seq"
        assert isinstance(result.scores, np.ndarray)
        assert isinstance(result.ema_scores, np.ndarray)
        assert isinstance(result.drift_events, list)

    def test_str_representation(self):
        est = self._estimator()
        frame = _bbox_frame(self.H, self.W, self.BBOX)
        est.initialize(frame, self.BBOX)
        est.score(frame, self.BBOX)
        result = est.result("MOSSE", "car1")
        s = str(result)
        assert "MOSSE" in s
        assert "car1" in s


# ---------------------------------------------------------------------------
# correlate_quality_with_iou
# ---------------------------------------------------------------------------


class TestCorrelateQualityWithIoU:
    def test_perfect_correlation(self):
        x = np.linspace(0, 1, 50)
        r = correlate_quality_with_iou(x, x)
        assert abs(r - 1.0) < 1e-6

    def test_anti_correlation(self):
        x = np.linspace(0, 1, 50)
        r = correlate_quality_with_iou(x, 1 - x)
        assert abs(r + 1.0) < 1e-4

    def test_constant_input_returns_zero(self):
        q = np.ones(30)
        iou = np.linspace(0, 1, 30)
        assert correlate_quality_with_iou(q, iou) == 0.0

    def test_mismatched_lengths_uses_shorter(self):
        q = np.linspace(0, 1, 10)
        iou = np.linspace(0, 1, 20)
        r = correlate_quality_with_iou(q, iou)
        assert -1.0 <= r <= 1.0

    def test_too_short_returns_zero(self):
        assert correlate_quality_with_iou(np.array([0.5]), np.array([0.5])) == 0.0


# ---------------------------------------------------------------------------
# Integration: BenchmarkEngine with quality_estimator
# ---------------------------------------------------------------------------


class TestBenchmarkEngineQualityIntegration:
    def test_quality_attached_to_sequence_result(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        ds = SyntheticDataset(num_sequences=2, num_frames=20, motion="linear", seed=0)
        tracker = MOSSETracker()
        engine = BenchmarkEngine(
            verbose=False,
            quality_estimator=AppearanceQualityEstimator(),
        )
        result = engine.run(tracker, ds, dataset_name="Synthetic")

        for seq_result in result.sequence_results:
            assert seq_result.quality is not None
            assert isinstance(seq_result.quality, QualityResult)
            assert len(seq_result.quality.scores) == len(seq_result.ious)

    def test_no_quality_when_not_configured(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        ds = SyntheticDataset(num_sequences=1, num_frames=15, motion="linear", seed=1)
        tracker = MOSSETracker()
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(tracker, ds, dataset_name="Synthetic")

        for seq_result in result.sequence_results:
            assert seq_result.quality is None

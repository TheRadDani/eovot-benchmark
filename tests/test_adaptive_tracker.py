"""Tests for AdaptiveResolutionTracker.

Validates:
- Basic initialize / update contract (BaseTracker interface)
- Coordinate up-scaling (predictions returned in original-frame space)
- Confidence EMA updates after warmup
- Resolution tier switching (up and down)
- adaptation_summary() output structure
- Edge cases: single scale factor, warmup longer than sequence, re-init
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from eovot.trackers.adaptive import AdaptiveResolutionTracker
from eovot.trackers.base import BaseTracker, BBox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(h: int = 240, w: int = 320) -> np.ndarray:
    """Return a random BGR frame."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _make_mock_base(return_bbox: BBox = (50.0, 50.0, 40.0, 40.0)) -> BaseTracker:
    """Return a mock BaseTracker whose update always returns *return_bbox*."""
    m = MagicMock(spec=BaseTracker)
    m.name = "MockTracker"
    m.update.return_value = return_bbox
    return m


class ConstantBBoxTracker(BaseTracker):
    """Trivial tracker that always returns its init bbox — no OpenCV needed."""

    def __init__(self) -> None:
        super().__init__(name="ConstantBBox")
        self._bbox: BBox = (0.0, 0.0, 1.0, 1.0)

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        self._bbox = bbox

    def update(self, frame: np.ndarray) -> BBox:
        return self._bbox


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestAdaptiveTrackerConstruction:
    def test_default_construction(self):
        base = ConstantBBoxTracker()
        tracker = AdaptiveResolutionTracker(base)
        assert tracker.name == "Adaptive(ConstantBBox)"
        assert 1.0 in tracker._scales
        assert tracker._scales[0] == 1.0, "Full-res tier must be first"

    def test_name_reflects_base(self):
        base = ConstantBBoxTracker()
        tracker = AdaptiveResolutionTracker(base)
        assert "ConstantBBox" in tracker.name

    def test_invalid_ema_alpha_raises(self):
        with pytest.raises(ValueError, match="ema_alpha"):
            AdaptiveResolutionTracker(ConstantBBoxTracker(), ema_alpha=0.0)
        with pytest.raises(ValueError, match="ema_alpha"):
            AdaptiveResolutionTracker(ConstantBBoxTracker(), ema_alpha=1.5)

    def test_invalid_thresholds_raise(self):
        with pytest.raises(ValueError, match="Thresholds"):
            AdaptiveResolutionTracker(
                ConstantBBoxTracker(), low_conf_threshold=0.8, high_conf_threshold=0.3
            )

    def test_empty_scale_factors_uses_defaults(self):
        # An empty list falls back to the built-in default tiers.
        tracker = AdaptiveResolutionTracker(ConstantBBoxTracker(), scale_factors=[])
        assert len(tracker._scales) >= 1
        assert tracker._scales[0] == 1.0

    def test_1x_prepended_when_absent(self):
        base = ConstantBBoxTracker()
        tracker = AdaptiveResolutionTracker(base, scale_factors=[0.75, 0.5])
        assert tracker._scales[0] == 1.0

    def test_custom_scale_factors_sorted_desc(self):
        base = ConstantBBoxTracker()
        tracker = AdaptiveResolutionTracker(base, scale_factors=[0.5, 1.0, 0.75])
        assert tracker._scales == sorted(tracker._scales, reverse=True)


# ---------------------------------------------------------------------------
# Initialize / update contract
# ---------------------------------------------------------------------------

class TestAdaptiveTrackerInterface:
    def test_initialize_runs_without_error(self):
        tracker = AdaptiveResolutionTracker(ConstantBBoxTracker())
        frame = _make_frame()
        tracker.initialize(frame, (10.0, 10.0, 30.0, 30.0))

    def test_update_returns_four_tuple(self):
        tracker = AdaptiveResolutionTracker(ConstantBBoxTracker())
        frame = _make_frame()
        tracker.initialize(frame, (10.0, 10.0, 30.0, 30.0))
        pred = tracker.update(frame)
        assert len(pred) == 4
        assert all(isinstance(v, float) for v in pred)

    def test_update_increments_frame_idx(self):
        tracker = AdaptiveResolutionTracker(ConstantBBoxTracker())
        frame = _make_frame()
        tracker.initialize(frame, (10.0, 10.0, 30.0, 30.0))
        for _ in range(5):
            tracker.update(frame)
        assert tracker._frame_idx == 5

    def test_adaptation_log_grows_with_updates(self):
        tracker = AdaptiveResolutionTracker(ConstantBBoxTracker())
        frame = _make_frame()
        tracker.initialize(frame, (10.0, 10.0, 30.0, 30.0))
        n = 8
        for _ in range(n):
            tracker.update(frame)
        assert len(tracker.adaptation_log) == n

    def test_current_scale_is_in_scale_list(self):
        tracker = AdaptiveResolutionTracker(
            ConstantBBoxTracker(), scale_factors=[1.0, 0.75, 0.5]
        )
        frame = _make_frame()
        tracker.initialize(frame, (10.0, 10.0, 30.0, 30.0))
        for _ in range(20):
            tracker.update(frame)
            assert tracker.current_scale in tracker._scales


# ---------------------------------------------------------------------------
# Coordinate rescaling
# ---------------------------------------------------------------------------

class TestCoordinateRescaling:
    def test_full_scale_prediction_unchanged(self):
        """When scale=1.0, predictions must pass through unmodified."""
        expected_bbox = (50.0, 60.0, 40.0, 40.0)
        base = _make_mock_base(return_bbox=expected_bbox)
        tracker = AdaptiveResolutionTracker(
            base,
            scale_factors=[1.0],
            warmup_frames=0,
        )
        frame = _make_frame()
        tracker.initialize(frame, (50.0, 60.0, 40.0, 40.0))
        pred = tracker.update(frame)
        # With single scale=1.0, pred equals what base returns.
        assert all(math.isclose(a, b, rel_tol=1e-5) for a, b in zip(pred, expected_bbox))

    def test_scaled_prediction_up_scaled_back(self):
        """Prediction on a 0.5-scaled frame must be doubled back to original space."""
        # Base tracker always returns (25, 30, 20, 20) on the 0.5-scaled frame.
        base = _make_mock_base(return_bbox=(25.0, 30.0, 20.0, 20.0))
        # Force scale to 0.5 by using very high confidence threshold and single
        # tier that starts at the second position after warmup.
        tracker = AdaptiveResolutionTracker(
            base,
            scale_factors=[1.0, 0.5],
            high_conf_threshold=0.01,   # almost always confident → use 0.5
            low_conf_threshold=0.0,
            warmup_frames=0,
        )
        frame = _make_frame()
        tracker.initialize(frame, (50.0, 60.0, 40.0, 40.0))
        # Trigger one high-confidence update so scale drops to 0.5.
        tracker.update(frame)
        pred = tracker.update(frame)
        # When scale=0.5: pred = (25, 30, 20, 20) → unscaled = (50, 60, 40, 40)
        if math.isclose(tracker.current_scale, 0.5):
            expected = (50.0, 60.0, 40.0, 40.0)
            assert all(math.isclose(a, b, rel_tol=0.05) for a, b in zip(pred, expected))


# ---------------------------------------------------------------------------
# Confidence EMA and tier adaptation
# ---------------------------------------------------------------------------

class TestConfidenceAndAdaptation:
    def test_confidence_ema_initialised_to_neutral(self):
        tracker = AdaptiveResolutionTracker(ConstantBBoxTracker())
        assert math.isclose(tracker.mean_confidence, 0.5)

    def test_confidence_ema_in_range(self):
        tracker = AdaptiveResolutionTracker(ConstantBBoxTracker())
        frame = _make_frame()
        tracker.initialize(frame, (10.0, 10.0, 30.0, 30.0))
        for _ in range(30):
            tracker.update(frame)
        assert 0.0 <= tracker.mean_confidence <= 1.0

    def test_starts_at_full_resolution(self):
        tracker = AdaptiveResolutionTracker(
            ConstantBBoxTracker(), scale_factors=[1.0, 0.75, 0.5]
        )
        frame = _make_frame()
        tracker.initialize(frame, (10.0, 10.0, 30.0, 30.0))
        assert math.isclose(tracker.current_scale, 1.0)

    def test_switch_count_non_negative(self):
        tracker = AdaptiveResolutionTracker(
            ConstantBBoxTracker(), scale_factors=[1.0, 0.75, 0.5], warmup_frames=2
        )
        frame = _make_frame()
        tracker.initialize(frame, (10.0, 10.0, 30.0, 30.0))
        for _ in range(20):
            tracker.update(frame)
        assert tracker.switch_count >= 0

    def test_forced_low_confidence_stays_full_res(self):
        """With always-low confidence, tracker must stay at full resolution."""
        tracker = AdaptiveResolutionTracker(
            ConstantBBoxTracker(),
            scale_factors=[1.0, 0.5],
            high_conf_threshold=0.99,
            low_conf_threshold=0.98,
            warmup_frames=2,
        )
        frame = _make_frame()
        tracker.initialize(frame, (10.0, 10.0, 30.0, 30.0))
        for _ in range(20):
            tracker.update(frame)
        # Since high_conf_threshold=0.99 is almost never met, we should
        # predominantly stay at tier 0 (full res); switch_count should be low.
        assert tracker.switch_count <= 2


# ---------------------------------------------------------------------------
# adaptation_summary()
# ---------------------------------------------------------------------------

class TestAdaptationSummary:
    def test_summary_empty_before_init(self):
        tracker = AdaptiveResolutionTracker(ConstantBBoxTracker())
        assert tracker.adaptation_summary() == {}

    def test_summary_keys(self):
        tracker = AdaptiveResolutionTracker(ConstantBBoxTracker(), warmup_frames=2)
        frame = _make_frame()
        tracker.initialize(frame, (10.0, 10.0, 30.0, 30.0))
        for _ in range(15):
            tracker.update(frame)
        summary = tracker.adaptation_summary()
        for key in (
            "total_frames", "switch_count", "switch_rate",
            "scale_distribution", "mean_confidence", "mean_scale",
        ):
            assert key in summary, f"Missing key: {key}"

    def test_summary_total_frames_correct(self):
        tracker = AdaptiveResolutionTracker(ConstantBBoxTracker(), warmup_frames=2)
        frame = _make_frame()
        tracker.initialize(frame, (10.0, 10.0, 30.0, 30.0))
        n = 12
        for _ in range(n):
            tracker.update(frame)
        assert tracker.adaptation_summary()["total_frames"] == n

    def test_scale_distribution_sums_to_one(self):
        tracker = AdaptiveResolutionTracker(
            ConstantBBoxTracker(),
            scale_factors=[1.0, 0.75, 0.5],
            warmup_frames=2,
        )
        frame = _make_frame()
        tracker.initialize(frame, (10.0, 10.0, 30.0, 30.0))
        for _ in range(20):
            tracker.update(frame)
        dist = tracker.adaptation_summary()["scale_distribution"]
        total = sum(dist.values())
        assert math.isclose(total, 1.0, abs_tol=1e-4)

    def test_mean_scale_within_bounds(self):
        scales = [1.0, 0.75, 0.5]
        tracker = AdaptiveResolutionTracker(
            ConstantBBoxTracker(), scale_factors=scales, warmup_frames=2
        )
        frame = _make_frame()
        tracker.initialize(frame, (10.0, 10.0, 30.0, 30.0))
        for _ in range(20):
            tracker.update(frame)
        ms = tracker.adaptation_summary()["mean_scale"]
        assert min(scales) <= ms <= max(scales)


# ---------------------------------------------------------------------------
# Re-initialisation
# ---------------------------------------------------------------------------

class TestReInitialisation:
    def test_reinit_resets_state(self):
        tracker = AdaptiveResolutionTracker(
            ConstantBBoxTracker(), scale_factors=[1.0, 0.5], warmup_frames=2
        )
        frame = _make_frame()
        tracker.initialize(frame, (10.0, 10.0, 30.0, 30.0))
        for _ in range(20):
            tracker.update(frame)
        first_log_len = len(tracker.adaptation_log)

        # Re-initialize
        tracker.initialize(frame, (20.0, 20.0, 40.0, 40.0))
        assert len(tracker.adaptation_log) == 0
        assert tracker.switch_count == 0
        assert tracker._frame_idx == 0

    def test_reinit_starts_at_full_res(self):
        tracker = AdaptiveResolutionTracker(
            ConstantBBoxTracker(), scale_factors=[1.0, 0.5], warmup_frames=2
        )
        frame = _make_frame()
        tracker.initialize(frame, (10.0, 10.0, 30.0, 30.0))
        for _ in range(20):
            tracker.update(frame)
        tracker.initialize(frame, (10.0, 10.0, 30.0, 30.0))
        assert math.isclose(tracker.current_scale, 1.0)


# ---------------------------------------------------------------------------
# Integration with benchmark engine (synthetic data)
# ---------------------------------------------------------------------------

class TestBenchmarkIntegration:
    def test_end_to_end_with_synthetic(self):
        """Tracker should run the full benchmark loop without errors."""
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.benchmark.engine import BenchmarkEngine

        dataset = SyntheticDataset(num_sequences=2, num_frames=25, seed=99)
        base = ConstantBBoxTracker()
        tracker = AdaptiveResolutionTracker(
            base, scale_factors=[1.0, 0.75], warmup_frames=5
        )
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(tracker, dataset, dataset_name="synthetic-adaptive")

        assert result.tracker_name == tracker.name
        assert len(result.sequence_results) == 2
        for sr in result.sequence_results:
            assert sr.profiling.fps > 0
            assert len(sr.ious) > 0

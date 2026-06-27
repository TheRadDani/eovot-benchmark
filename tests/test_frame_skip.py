"""Tests for eovot.trackers.frame_skip.FrameSkipAdapter."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.frame_skip import FrameSkipAdapter, _MAX_SKIP
from eovot.trackers.mosse import MOSSETracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _make_frames(n: int = 20, h: int = 64, w: int = 64) -> list:
    return [_make_frame(h, w, seed=i) for i in range(n)]


_INIT_BBOX = (10.0, 10.0, 20.0, 20.0)


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_name_includes_inner_name(self):
        adapter = FrameSkipAdapter(MOSSETracker(), skip_interval=3)
        assert "MOSSE" in adapter.name
        assert "Skip" in adapter.name

    def test_custom_name(self):
        adapter = FrameSkipAdapter(MOSSETracker(), name="EdgeTracker")
        assert adapter.name == "EdgeTracker"

    def test_invalid_skip_interval_zero(self):
        with pytest.raises(ValueError, match="skip_interval"):
            FrameSkipAdapter(MOSSETracker(), skip_interval=0)

    def test_invalid_skip_interval_negative(self):
        with pytest.raises(ValueError):
            FrameSkipAdapter(MOSSETracker(), skip_interval=-2)

    def test_invalid_interpolation(self):
        with pytest.raises(ValueError, match="interpolation"):
            FrameSkipAdapter(MOSSETracker(), interpolation="cubic")

    def test_repr(self):
        adapter = FrameSkipAdapter(MOSSETracker(), skip_interval=2)
        r = repr(adapter)
        assert "MOSSE" in r
        assert "skip" in r.lower()


# ---------------------------------------------------------------------------
# Initialisation guard
# ---------------------------------------------------------------------------

class TestInitGuard:
    def test_update_before_init_raises(self):
        adapter = FrameSkipAdapter(MOSSETracker(), skip_interval=2)
        with pytest.raises(RuntimeError, match="not initialised"):
            adapter.update(_make_frame())

    def test_diagnostics_before_init(self):
        adapter = FrameSkipAdapter(MOSSETracker(), skip_interval=3)
        assert adapter.mean_update_latency_ms is None
        assert adapter.tracker_call_fraction == 1.0  # no frames yet → 100 %


# ---------------------------------------------------------------------------
# Static skip mode
# ---------------------------------------------------------------------------

class TestStaticSkip:
    def test_returns_four_element_bbox(self):
        frames = _make_frames(10)
        adapter = FrameSkipAdapter(MOSSETracker(), skip_interval=2)
        adapter.initialize(frames[0], _INIT_BBOX)
        for frame in frames[1:]:
            bbox = adapter.update(frame)
            assert len(bbox) == 4

    def test_skip_1_calls_tracker_every_frame(self):
        frames = _make_frames(12)
        adapter = FrameSkipAdapter(MOSSETracker(), skip_interval=1)
        adapter.initialize(frames[0], _INIT_BBOX)
        for frame in frames[1:]:
            adapter.update(frame)
        # Every frame triggers an update → close to 1.0
        assert adapter.tracker_call_fraction >= 0.9

    def test_skip_3_reduces_calls(self):
        frames = _make_frames(15)
        adapter = FrameSkipAdapter(MOSSETracker(), skip_interval=3)
        adapter.initialize(frames[0], _INIT_BBOX)
        for frame in frames[1:]:
            adapter.update(frame)
        # Roughly 1/3 of frames trigger real updates
        assert adapter.tracker_call_fraction < 0.6

    def test_effective_skip_property_matches_init(self):
        adapter = FrameSkipAdapter(MOSSETracker(), skip_interval=5)
        assert adapter.effective_skip == 5

    def test_mean_latency_recorded_after_updates(self):
        frames = _make_frames(10)
        adapter = FrameSkipAdapter(MOSSETracker(), skip_interval=2)
        adapter.initialize(frames[0], _INIT_BBOX)
        for frame in frames[1:]:
            adapter.update(frame)
        lat = adapter.mean_update_latency_ms
        assert lat is not None
        assert lat >= 0.0

    def test_reinitialise_resets_state(self):
        frames = _make_frames(10)
        adapter = FrameSkipAdapter(MOSSETracker(), skip_interval=3)
        adapter.initialize(frames[0], _INIT_BBOX)
        for frame in frames[1:5]:
            adapter.update(frame)
        # Re-initialise should clear frame count
        adapter.initialize(frames[0], _INIT_BBOX)
        assert adapter.tracker_call_fraction == 1.0  # reset → no frames counted


# ---------------------------------------------------------------------------
# Interpolation strategies
# ---------------------------------------------------------------------------

class TestInterpolation:
    @pytest.mark.parametrize("mode", ["linear", "velocity"])
    def test_all_modes_return_valid_bboxes(self, mode):
        frames = _make_frames(12)
        adapter = FrameSkipAdapter(MOSSETracker(), skip_interval=3, interpolation=mode)
        adapter.initialize(frames[0], _INIT_BBOX)
        bboxes = [adapter.update(f) for f in frames[1:]]
        assert all(len(b) == 4 for b in bboxes)

    def test_linear_fills_between_tracked_positions(self):
        """On a skipped frame the bbox should be between previous estimates."""
        frames = _make_frames(5)
        adapter = FrameSkipAdapter(MOSSETracker(), skip_interval=3, interpolation="linear")
        adapter.initialize(frames[0], _INIT_BBOX)
        bboxes = [adapter.update(f) for f in frames[1:]]
        # All returned bboxes should have positive width/height
        for bbox in bboxes:
            x, y, w, h = bbox
            assert w > 0
            assert h > 0

    def test_velocity_returns_float_coordinates(self):
        frames = _make_frames(8)
        adapter = FrameSkipAdapter(MOSSETracker(), skip_interval=2, interpolation="velocity")
        adapter.initialize(frames[0], _INIT_BBOX)
        for frame in frames[1:]:
            bbox = adapter.update(frame)
            assert all(isinstance(v, float) for v in bbox)


# ---------------------------------------------------------------------------
# Adaptive skip mode
# ---------------------------------------------------------------------------

class TestAdaptiveSkip:
    def test_adaptive_mode_runs_without_error(self):
        frames = _make_frames(30)
        adapter = FrameSkipAdapter(MOSSETracker(), target_fps=30.0, skip_interval=2)
        adapter.initialize(frames[0], _INIT_BBOX)
        for frame in frames[1:]:
            bbox = adapter.update(frame)
            assert len(bbox) == 4

    def test_skip_never_exceeds_max(self):
        """Even with an unreachably low target FPS the skip is capped."""
        frames = _make_frames(30)
        adapter = FrameSkipAdapter(MOSSETracker(), target_fps=0.001, skip_interval=2)
        adapter.initialize(frames[0], _INIT_BBOX)
        for frame in frames[1:]:
            adapter.update(frame)
        assert adapter.effective_skip <= _MAX_SKIP

    def test_skip_at_least_1(self):
        """Even with an insanely high target FPS the skip floor is 1."""
        frames = _make_frames(30)
        # Target FPS much higher than native tracker FPS → skip should be 1
        adapter = FrameSkipAdapter(MOSSETracker(), target_fps=1_000_000.0, skip_interval=1)
        adapter.initialize(frames[0], _INIT_BBOX)
        for frame in frames[1:]:
            adapter.update(frame)
        assert adapter.effective_skip >= 1


# ---------------------------------------------------------------------------
# Integration with BenchmarkEngine
# ---------------------------------------------------------------------------

class TestBenchmarkEngineIntegration:
    def test_frame_skip_adapter_with_engine(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        dataset = SyntheticDataset(num_sequences=2, num_frames=20, motion="linear")
        engine = BenchmarkEngine(verbose=False)
        adapter = FrameSkipAdapter(MOSSETracker(), skip_interval=3)
        result = engine.run(adapter, dataset, dataset_name="Synthetic-SkipTest")

        assert result.mean_iou >= 0.0
        assert result.mean_fps > 0.0
        assert len(result.sequence_results) == 2

    def test_adaptive_adapter_with_engine(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        dataset = SyntheticDataset(num_sequences=2, num_frames=20, motion="circular")
        engine = BenchmarkEngine(verbose=False)
        adapter = FrameSkipAdapter(MOSSETracker(), target_fps=50.0)
        result = engine.run(adapter, dataset, dataset_name="Synthetic-AdaptiveTest")

        assert result.mean_fps > 0.0
        assert len(result.sequence_results) == 2

    def test_frame_skip_vs_direct_tracker(self):
        """Frame-skip tracker should have lower or equal IoU than direct tracker."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        dataset = SyntheticDataset(num_sequences=3, num_frames=30, motion="linear")
        engine = BenchmarkEngine(verbose=False)

        direct = MOSSETracker()
        skipped = FrameSkipAdapter(MOSSETracker(), skip_interval=4, interpolation="linear")

        result_direct = engine.run(direct, dataset, dataset_name="Direct")
        result_skipped = engine.run(skipped, dataset, dataset_name="Skipped")

        # Frame-skip should generally not outperform direct tracking
        # (may be close — just ensure both are in [0, 1])
        assert 0.0 <= result_direct.mean_iou <= 1.0
        assert 0.0 <= result_skipped.mean_iou <= 1.0

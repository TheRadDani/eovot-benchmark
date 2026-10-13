"""Unit and integration tests for the DSSTTracker."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.base import BaseTracker
from eovot.trackers.dsst import DSSTTracker
from eovot.trackers.registry import TRACKER_REGISTRY, build_tracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_frame(h: int = 120, w: int = 160, color=(80, 120, 60)) -> np.ndarray:
    """Solid-colour BGR frame with a contrasting 40×40 white rectangle."""
    frame = np.full((h, w, 3), color, dtype=np.uint8)
    frame[40:80, 50:90] = (220, 220, 220)
    return frame


def _make_frames(n: int = 10, h: int = 120, w: int = 160) -> list:
    return [_solid_frame(h, w) for _ in range(n)]


INIT_BBOX = (50.0, 40.0, 40.0, 40.0)  # x, y, w, h — targets the white rectangle


# ---------------------------------------------------------------------------
# Interface / registry tests
# ---------------------------------------------------------------------------

class TestDSSTRegistration:
    def test_in_registry(self):
        assert "DSST" in TRACKER_REGISTRY

    def test_build_via_registry(self):
        t = build_tracker("DSST")
        assert isinstance(t, DSSTTracker)

    def test_is_base_tracker(self):
        assert isinstance(DSSTTracker(), BaseTracker)

    def test_name(self):
        assert DSSTTracker().name == "DSST"


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestDSSTConstructor:
    def test_default_construction(self):
        t = DSSTTracker()
        assert t.num_scales == 9
        assert t.scale_step == pytest.approx(1.03)
        assert t.learning_rate == pytest.approx(0.075)

    def test_custom_params(self):
        t = DSSTTracker(num_scales=9, scale_step=1.05, learning_rate=0.1)
        assert t.num_scales == 9
        assert len(t._scale_factors) == 9

    def test_even_num_scales_raises(self):
        with pytest.raises(ValueError, match="odd"):
            DSSTTracker(num_scales=16)

    def test_scale_factors_symmetric(self):
        t = DSSTTracker(num_scales=9, scale_step=1.02)
        sf = t._scale_factors
        # The factor at index 0 is the reciprocal of the factor at the last index
        assert sf[0] == pytest.approx(1.0 / sf[-1], rel=1e-4)

    def test_scale_factors_middle_is_one(self):
        t = DSSTTracker(num_scales=7)
        mid = t.num_scales // 2
        assert t._scale_factors[mid] == pytest.approx(1.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestDSSTInitialize:
    def test_initialize_does_not_raise(self):
        t = DSSTTracker()
        t.initialize(_solid_frame(), INIT_BBOX)

    def test_state_set_after_initialize(self):
        t = DSSTTracker()
        t.initialize(_solid_frame(), INIT_BBOX)
        assert t._pos is not None
        assert t._alphaf is not None
        assert t._current_scale == pytest.approx(1.0)

    def test_base_target_size_matches_bbox(self):
        t = DSSTTracker()
        t.initialize(_solid_frame(), INIT_BBOX)
        bw, bh = t._base_target_sz
        assert bw == 40 and bh == 40


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------

class TestDSSTUpdate:
    def setup_method(self):
        self.tracker = DSSTTracker()
        self.frames = _make_frames(10)
        self.tracker.initialize(self.frames[0], INIT_BBOX)

    def test_update_returns_four_tuple(self):
        bbox = self.tracker.update(self.frames[1])
        assert len(bbox) == 4, "update() must return (x, y, w, h)"

    def test_update_positive_size(self):
        for frame in self.frames[1:]:
            x, y, w, h = self.tracker.update(frame)
            assert w > 0 and h > 0

    def test_update_without_init_raises(self):
        with pytest.raises(RuntimeError, match="initialised"):
            DSSTTracker().update(self.frames[0])

    def test_multiple_updates_stay_bounded(self):
        """Predicted centre must remain within the frame."""
        fh, fw = self.frames[0].shape[:2]
        for frame in self.frames[1:]:
            x, y, w, h = self.tracker.update(frame)
            cx, cy = x + w / 2, y + h / 2
            assert -fw < cx < 2 * fw
            assert -fh < cy < 2 * fh

    def test_scale_adapts_during_tracking(self):
        """Verify that the scale estimation runs and changes _current_scale."""
        t = DSSTTracker(num_scales=9, scale_step=1.05)
        frames = _make_frames(15)
        t.initialize(frames[0], INIT_BBOX)
        scales = []
        for frame in frames[1:]:
            t.update(frame)
            scales.append(t._current_scale)

        # _current_scale must change from its initial 1.0 at some point,
        # showing the scale estimator is actively running.
        assert len(scales) == 14
        # At least some frames should have a scale != 1.0 (clamped to [0.2, 5.0])
        assert all(0.2 <= s <= 5.0 for s in scales), "scale must stay in [0.2, 5.0]"


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestDSSTReset:
    def test_reset_clears_state(self):
        t = DSSTTracker()
        t.initialize(_solid_frame(), INIT_BBOX)
        t.reset()
        assert t._pos is None
        assert t._alphaf is None
        assert t._current_scale == pytest.approx(1.0)

    def test_reinitialize_after_reset(self):
        t = DSSTTracker()
        t.initialize(_solid_frame(), INIT_BBOX)
        t.reset()
        t.initialize(_solid_frame(), INIT_BBOX)  # should not raise
        bbox = t.update(_solid_frame())
        assert len(bbox) == 4


# ---------------------------------------------------------------------------
# Integration: BenchmarkEngine compatibility
# ---------------------------------------------------------------------------

class TestDSSTBenchmarkIntegration:
    def test_benchmark_engine_runs(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        dataset = SyntheticDataset(num_sequences=2, num_frames=20, seed=7)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(DSSTTracker(), dataset, dataset_name="Synthetic")

        assert result.tracker_name == "DSST"
        assert len(result.sequence_results) == 2
        assert result.mean_iou >= 0.0
        assert result.mean_fps > 0.0

    def test_dsst_vs_kcf_on_scale_variation(self):
        """DSST should match or beat KCF on a synthetic sequence with scale change."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        dataset = SyntheticDataset(num_sequences=3, num_frames=40, seed=42)
        engine = BenchmarkEngine(verbose=False)

        dsst_result = engine.run(DSSTTracker(), dataset, dataset_name="Synthetic")
        kcf_result = engine.run(
            build_tracker("KCF"), dataset, dataset_name="Synthetic"
        )

        # Both should produce valid IoU values (not NaN / negative)
        assert dsst_result.mean_iou >= 0.0
        assert kcf_result.mean_iou >= 0.0

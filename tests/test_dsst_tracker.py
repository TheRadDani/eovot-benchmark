"""Unit and integration tests for DSSTTracker."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.dsst import DSSTTracker
from eovot.datasets.synthetic import SyntheticDataset
from eovot.benchmark.engine import BenchmarkEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(h: int = 240, w: int = 320) -> np.ndarray:
    """Return a random BGR frame."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _bbox() -> tuple:
    return (50.0, 60.0, 40.0, 30.0)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestDSSTTrackerConstruction:
    def test_default_construction(self):
        t = DSSTTracker()
        assert t.name == "DSST"
        assert t.n_scales % 2 == 1, "n_scales must be odd"
        assert t.scale_step > 1.0

    def test_even_n_scales_forced_odd(self):
        t = DSSTTracker(n_scales=32)
        assert t.n_scales == 33

    def test_scale_factors_shape(self):
        t = DSSTTracker(n_scales=11)
        assert len(t._scale_factors) == 11

    def test_scale_factors_center_is_one(self):
        t = DSSTTracker(n_scales=11)
        center = t.n_scales // 2
        assert t._scale_factors[center] == pytest.approx(1.0)

    def test_scale_factors_monotone(self):
        t = DSSTTracker(n_scales=11)
        diffs = np.diff(t._scale_factors)
        assert np.all(diffs > 0)

    def test_repr(self):
        t = DSSTTracker()
        assert "DSST" in repr(t)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestDSSTTrackerInitialize:
    def test_state_set_after_initialize(self):
        t = DSSTTracker()
        frame = _make_frame()
        bbox = _bbox()
        t.initialize(frame, bbox)
        assert t._pos is not None
        assert t._base_target_sz is not None
        assert t._trans_alphaf is not None
        assert t._scale_num is not None
        assert t._scale_den is not None

    def test_pos_matches_bbox_center(self):
        t = DSSTTracker()
        frame = _make_frame()
        x, y, w, h = _bbox()
        t.initialize(frame, (x, y, w, h))
        cx, cy = t._pos
        assert cx == pytest.approx(x + w / 2)
        assert cy == pytest.approx(y + h / 2)

    def test_current_scale_is_one_after_init(self):
        t = DSSTTracker()
        t.initialize(_make_frame(), _bbox())
        assert t._current_scale == pytest.approx(1.0)

    def test_translation_filter_shape(self):
        t = DSSTTracker(padding=1.0)
        frame = _make_frame()
        x, y, w, h = _bbox()
        t.initialize(frame, (x, y, w, h))
        sw, sh = t._trans_sz
        assert t._trans_alphaf.shape == (sh, sw)

    def test_scale_filter_shape(self):
        t = DSSTTracker(n_scales=11)
        frame = _make_frame()
        t.initialize(frame, _bbox())
        sm_w, sm_h = t._scale_model_sz
        n_feat = sm_w * sm_h
        assert t._scale_num.shape == (n_feat, 11)
        assert t._scale_den.shape == (11,)


# ---------------------------------------------------------------------------
# Update (single frame)
# ---------------------------------------------------------------------------

class TestDSSTTrackerUpdate:
    def test_update_before_initialize_raises(self):
        t = DSSTTracker()
        with pytest.raises(RuntimeError, match="initialised"):
            t.update(_make_frame())

    def test_update_returns_tuple_of_four(self):
        t = DSSTTracker()
        frame = _make_frame()
        t.initialize(frame, _bbox())
        result = t.update(frame)
        assert len(result) == 4

    def test_update_returns_positive_size(self):
        t = DSSTTracker()
        frame = _make_frame()
        t.initialize(frame, _bbox())
        x, y, w, h = t.update(frame)
        assert w > 0
        assert h > 0

    def test_static_target_stays_close(self):
        """On a static target (same frame repeated) the tracker should not drift far."""
        t = DSSTTracker()
        frame = _make_frame()
        x0, y0, w0, h0 = 60.0, 70.0, 40.0, 30.0
        t.initialize(frame, (x0, y0, w0, h0))
        cx0 = x0 + w0 / 2.0
        cy0 = y0 + h0 / 2.0
        for _ in range(10):
            x, y, w, h = t.update(frame)
        cx = x + w / 2.0
        cy = y + h / 2.0
        assert abs(cx - cx0) < 10.0, f"x drift too large: {abs(cx - cx0):.1f} px"
        assert abs(cy - cy0) < 10.0, f"y drift too large: {abs(cy - cy0):.1f} px"


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestDSSTTrackerReset:
    def test_reset_clears_state(self):
        t = DSSTTracker()
        t.initialize(_make_frame(), _bbox())
        t.reset()
        assert t._pos is None
        assert t._trans_alphaf is None
        assert t._scale_num is None

    def test_update_after_reset_raises(self):
        t = DSSTTracker()
        t.initialize(_make_frame(), _bbox())
        t.reset()
        with pytest.raises(RuntimeError):
            t.update(_make_frame())

    def test_reinitialize_after_reset(self):
        t = DSSTTracker()
        t.initialize(_make_frame(), _bbox())
        t.reset()
        t.initialize(_make_frame(), _bbox())
        assert t._pos is not None


# ---------------------------------------------------------------------------
# Scale adaptation (key property of DSST vs KCF)
# ---------------------------------------------------------------------------

def _make_textured_frame(
    H: int, W: int, cx: float, cy: float, tw: int, th: int, rng: np.random.Generator
) -> np.ndarray:
    """Return a frame where the target is a vivid gradient patch (easy to discriminate)."""
    frame = rng.integers(30, 80, (H, W, 3), dtype=np.uint8)
    x1, y1 = max(0, int(cx - tw / 2)), max(0, int(cy - th / 2))
    x2, y2 = min(W, x1 + tw), min(H, y1 + th)
    # Gradient pattern: distinct colours varying along rows and columns
    for r in range(y1, y2):
        for c in range(x1, x2):
            frame[r, c, 0] = 150 + int(100 * (c - x1) / max(tw, 1))
            frame[r, c, 1] = 150 + int(100 * (r - y1) / max(th, 1))
            frame[r, c, 2] = 200
    return frame


class TestDSSTScaleAdaptation:
    def test_scale_changes_on_scale_variation_sequence(self):
        """The internal scale estimate must respond to a growing textured target."""
        rng = np.random.default_rng(7)
        H, W = 240, 320
        bw, bh = 50, 40
        cx, cy = 160.0, 120.0

        t = DSSTTracker(n_scales=17, scale_step=1.05, scale_lr=0.15)
        frame0 = _make_textured_frame(H, W, cx, cy, bw, bh, rng)
        x0, y0 = int(cx - bw / 2), int(cy - bh / 2)
        t.initialize(frame0, (x0, y0, bw, bh))
        initial_scale = t._current_scale

        # Grow target by 10 % per frame over 10 frames; this stays within the
        # pyramid range (scale_step^8 ≈ 1.48, so 1.1^10 ≈ 2.6 is reachable over
        # multiple frames as current_scale accumulates).
        scales_seen = []
        for k in range(10):
            s = 1.0 + 0.10 * (k + 1)
            bw_s = int(round(bw * s))
            bh_s = int(round(bh * s))
            frame_k = _make_textured_frame(H, W, cx, cy, bw_s, bh_s, rng)
            t.update(frame_k)
            scales_seen.append(t._current_scale)

        # Scale must have moved away from 1.0 in at least one frame.
        assert max(scales_seen) != initial_scale, (
            "Scale should change when the target appears larger; "
            f"initial={initial_scale:.3f}, seen={[round(s, 3) for s in scales_seen]}"
        )

    def test_returned_bbox_size_reflects_scale(self):
        """The predicted box w/h must enlarge after enough frames with a larger target."""
        rng = np.random.default_rng(11)
        H, W = 240, 320
        bw, bh = 50, 40
        cx, cy = 160.0, 120.0

        t = DSSTTracker(n_scales=17, scale_step=1.05, scale_lr=0.15)
        frame0 = _make_textured_frame(H, W, cx, cy, bw, bh, rng)
        x0, y0 = int(cx - bw / 2), int(cy - bh / 2)
        t.initialize(frame0, (x0, y0, bw, bh))

        # Target grows by 10 % per frame over 15 frames with a textured pattern
        for k in range(15):
            s = 1.0 + 0.10 * (k + 1)
            bw_s = int(round(bw * s))
            bh_s = int(round(bh * s))
            frame_k = _make_textured_frame(H, W, cx, cy, bw_s, bh_s, rng)
            t.update(frame_k)

        # After enough scale-expanding frames, the internal scale should be > 1.0,
        # which means the returned box is strictly larger than the initial size.
        assert t._current_scale > 1.0, (
            f"current_scale should be > 1.0 after seeing an expanding target; "
            f"got {t._current_scale:.4f}"
        )
        # Derived check: the returned box must be at least as large as initial.
        _, _, wn, hn = t.update(_make_textured_frame(H, W, cx, cy, bw * 2, bh * 2, rng))
        assert wn >= bw and hn >= bh, (
            f"Returned box ({wn:.1f}×{hn:.1f}) should be ≥ initial size ({bw}×{bh})"
        )


# ---------------------------------------------------------------------------
# Integration with BenchmarkEngine
# ---------------------------------------------------------------------------

class TestDSSTBenchmarkIntegration:
    def test_benchmark_engine_runs(self):
        """DSSTTracker integrates cleanly with the BenchmarkEngine pipeline."""
        ds = SyntheticDataset(num_sequences=2, num_frames=20, motion="linear", seed=42)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(DSSTTracker(), ds, dataset_name="Synthetic")
        assert result.tracker_name == "DSST"
        assert len(result.sequence_results) == 2
        assert result.mean_fps > 0

    def test_benchmark_mean_iou_positive(self):
        ds = SyntheticDataset(num_sequences=3, num_frames=15, motion="circular", seed=7)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(DSSTTracker(), ds, dataset_name="Synthetic-Circular")
        assert result.mean_iou >= 0.0

    def test_benchmark_result_has_accuracy_metrics(self):
        ds = SyntheticDataset(num_sequences=2, num_frames=20, motion="linear", seed=1)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(DSSTTracker(), ds, dataset_name="Synthetic")
        for sr in result.sequence_results:
            assert sr.accuracy_metrics is not None
            assert 0.0 <= sr.accuracy_metrics.success_auc <= 1.0

    def test_dsst_vs_kcf_benchmark(self):
        """Both trackers should produce valid benchmark results on the same dataset."""
        from eovot.trackers.kcf import KCFTracker

        ds = SyntheticDataset(num_sequences=2, num_frames=20, motion="linear", seed=99)
        engine = BenchmarkEngine(verbose=False)

        r_dsst = engine.run(DSSTTracker(), ds, dataset_name="Synthetic")
        r_kcf = engine.run(KCFTracker(), ds, dataset_name="Synthetic")

        assert r_dsst.tracker_name == "DSST"
        assert r_kcf.tracker_name == "KCF"
        assert r_dsst.mean_fps > 0
        assert r_kcf.mean_fps > 0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestDSSTRegistry:
    def test_dsst_in_registry(self):
        from eovot.trackers.registry import TRACKER_REGISTRY, build_tracker
        assert "DSST" in TRACKER_REGISTRY

    def test_build_tracker_dsst(self):
        from eovot.trackers.registry import build_tracker
        t = build_tracker("DSST")
        assert isinstance(t, DSSTTracker)

    def test_build_tracker_dsst_with_params(self):
        from eovot.trackers.registry import build_tracker
        t = build_tracker("DSST", n_scales=17, scale_step=1.05)
        assert t.n_scales == 17
        assert t.scale_step == pytest.approx(1.05)

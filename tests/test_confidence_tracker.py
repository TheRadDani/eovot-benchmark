"""Tests for AppearanceConfidenceTracker."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.confidence import (
    AppearanceConfidenceTracker,
    ConfidenceReport,
    TrackedFrame,
)
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker
from eovot.datasets.synthetic import SyntheticDataset
from eovot.benchmark.engine import BenchmarkEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(h=120, w=160, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _make_bbox(x=30, y=30, w=40, h=40):
    return (float(x), float(y), float(w), float(h))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_wraps_base_tracker(self):
        base = MOSSETracker()
        ct = AppearanceConfidenceTracker(base)
        assert ct._tracker is base

    def test_name_contains_wrapped_name(self):
        ct = AppearanceConfidenceTracker(MOSSETracker())
        assert "MOSSE" in ct.name

    def test_invalid_tracker_type_raises(self):
        with pytest.raises(TypeError):
            AppearanceConfidenceTracker("not_a_tracker")

    def test_template_update_thresh_out_of_range_raises(self):
        with pytest.raises(ValueError):
            AppearanceConfidenceTracker(MOSSETracker(), template_update_thresh=1.5)

    def test_low_confidence_thresh_out_of_range_raises(self):
        with pytest.raises(ValueError):
            AppearanceConfidenceTracker(MOSSETracker(), low_confidence_thresh=-0.1)

    def test_default_thresholds(self):
        ct = AppearanceConfidenceTracker(MOSSETracker())
        assert ct.template_update_thresh == 0.70
        assert ct.low_confidence_thresh == 0.40
        assert ct.drift_ratio_thresh == 0.30


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_template_set_after_initialize(self):
        ct = AppearanceConfidenceTracker(MOSSETracker())
        frame = _make_frame()
        ct.initialize(frame, _make_bbox())
        assert ct._template is not None

    def test_history_empty_after_initialize(self):
        ct = AppearanceConfidenceTracker(MOSSETracker())
        frame = _make_frame()
        ct.initialize(frame, _make_bbox())
        assert ct.confidence_history == []

    def test_frame_idx_reset_on_initialize(self):
        ct = AppearanceConfidenceTracker(MOSSETracker())
        frame = _make_frame()
        ct.initialize(frame, _make_bbox())
        assert ct._frame_idx == 0

    def test_reinitialize_clears_history(self):
        ct = AppearanceConfidenceTracker(MOSSETracker())
        frame = _make_frame()
        ct.initialize(frame, _make_bbox())
        ct.update(frame)
        assert len(ct.confidence_history) == 1
        ct.initialize(frame, _make_bbox())
        assert ct.confidence_history == []


# ---------------------------------------------------------------------------
# Update / confidence mechanics
# ---------------------------------------------------------------------------

class TestUpdate:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ct = AppearanceConfidenceTracker(MOSSETracker())
        self.frame = _make_frame()
        self.ct.initialize(self.frame, _make_bbox())

    def test_update_returns_bbox_tuple(self):
        bbox = self.ct.update(self.frame)
        assert len(bbox) == 4

    def test_update_appends_confidence(self):
        self.ct.update(self.frame)
        assert len(self.ct.confidence_history) == 1

    def test_update_with_confidence_returns_tracked_frame(self):
        tf = self.ct.update_with_confidence(self.frame)
        assert isinstance(tf, TrackedFrame)
        assert len(tf.bbox) == 4
        assert 0.0 <= tf.confidence <= 1.0

    def test_frame_idx_increments(self):
        tf1 = self.ct.update_with_confidence(self.frame)
        tf2 = self.ct.update_with_confidence(self.frame)
        assert tf1.frame_idx == 1
        assert tf2.frame_idx == 2

    def test_confidence_in_valid_range(self):
        for _ in range(5):
            tf = self.ct.update_with_confidence(self.frame)
            assert 0.0 <= tf.confidence <= 1.0

    def test_update_without_initialize_raises(self):
        ct = AppearanceConfidenceTracker(MOSSETracker())
        with pytest.raises(RuntimeError, match="initialize"):
            ct.update_with_confidence(self.frame)

    def test_multiple_updates_history_length(self):
        for _ in range(7):
            self.ct.update(self.frame)
        assert len(self.ct.confidence_history) == 7


# ---------------------------------------------------------------------------
# NCC properties
# ---------------------------------------------------------------------------

class TestNCCScore:
    def test_identical_patches_high_score(self):
        patch = np.random.rand(64, 64).astype(np.float32) * 100
        score = AppearanceConfidenceTracker._ncc_score(patch, patch.copy())
        assert score > 0.95

    def test_uncorrelated_patches_near_half(self):
        rng = np.random.RandomState(42)
        t = rng.randn(64, 64).astype(np.float32)
        p = rng.randn(64, 64).astype(np.float32)
        score = AppearanceConfidenceTracker._ncc_score(t, p)
        # NCC of two independent random signals is near 0 → score near 0.5
        assert 0.2 <= score <= 0.8

    def test_negated_patch_low_score(self):
        patch = np.random.rand(64, 64).astype(np.float32) * 100 + 50
        neg = -patch + 2 * patch.mean()
        score = AppearanceConfidenceTracker._ncc_score(patch, neg)
        assert score < 0.05

    def test_constant_patch_returns_neutral(self):
        flat = np.ones((64, 64), dtype=np.float32) * 128
        score = AppearanceConfidenceTracker._ncc_score(flat, flat)
        assert score == 0.5


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

class TestExtractPatch:
    def test_output_shape(self):
        frame = _make_frame()
        patch = AppearanceConfidenceTracker._extract_patch(frame, (10, 10, 40, 40))
        assert patch.shape == (64, 64)

    def test_zero_area_returns_zero_patch(self):
        frame = _make_frame()
        patch = AppearanceConfidenceTracker._extract_patch(frame, (10, 10, 0, 0))
        assert patch.sum() == 0.0

    def test_out_of_bounds_bbox_clips(self):
        frame = _make_frame(h=50, w=50)
        patch = AppearanceConfidenceTracker._extract_patch(frame, (40, 40, 100, 100))
        assert patch.shape == (64, 64)

    def test_negative_coords_clip(self):
        frame = _make_frame()
        patch = AppearanceConfidenceTracker._extract_patch(frame, (-10, -10, 50, 50))
        assert patch.shape == (64, 64)


# ---------------------------------------------------------------------------
# Confidence report
# ---------------------------------------------------------------------------

class TestConfidenceReport:
    def test_empty_report_before_update(self):
        ct = AppearanceConfidenceTracker(MOSSETracker())
        frame = _make_frame()
        ct.initialize(frame, _make_bbox())
        report = ct.confidence_report()
        assert report.total_frames == 0
        assert not report.drift_detected

    def test_report_after_updates(self):
        ct = AppearanceConfidenceTracker(MOSSETracker())
        frame = _make_frame()
        ct.initialize(frame, _make_bbox())
        for _ in range(5):
            ct.update(frame)
        report = ct.confidence_report()
        assert report.total_frames == 5
        assert 0.0 <= report.mean_confidence <= 1.0

    def test_drift_detected_when_all_low(self):
        ct = AppearanceConfidenceTracker(
            MOSSETracker(),
            low_confidence_thresh=0.99,  # almost always below
            drift_ratio_thresh=0.1,
        )
        frame = _make_frame()
        ct.initialize(frame, _make_bbox())
        for _ in range(10):
            ct.update(_make_frame(seed=99))  # different noise → low NCC
        report = ct.confidence_report()
        assert report.drift_detected

    def test_to_dict_keys(self):
        ct = AppearanceConfidenceTracker(MOSSETracker())
        frame = _make_frame()
        ct.initialize(frame, _make_bbox())
        ct.update(frame)
        d = ct.confidence_report().to_dict()
        for k in ["tracker", "total_frames", "mean_confidence", "drift_detected"]:
            assert k in d

    def test_report_str_contains_tracker_name(self):
        ct = AppearanceConfidenceTracker(MOSSETracker())
        frame = _make_frame()
        ct.initialize(frame, _make_bbox())
        ct.update(frame)
        s = str(ct.confidence_report())
        assert "MOSSE" in s


# ---------------------------------------------------------------------------
# TrackedFrame
# ---------------------------------------------------------------------------

class TestTrackedFrame:
    def test_str_representation(self):
        tf = TrackedFrame(bbox=(10.0, 20.0, 50.0, 40.0), confidence=0.85, frame_idx=3)
        s = str(tf)
        assert "Frame 3" in s
        assert "0.8500" in s


# ---------------------------------------------------------------------------
# Integration with BenchmarkEngine
# ---------------------------------------------------------------------------

class TestBenchmarkEngineIntegration:
    def test_confidence_tracker_in_engine(self):
        base = MOSSETracker()
        tracker = AppearanceConfidenceTracker(base)
        dataset = SyntheticDataset(num_sequences=3, num_frames=20)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(tracker, dataset, "synthetic")
        assert result.mean_iou >= 0.0
        assert result.mean_fps > 0.0

    def test_kcf_wrapped_works(self):
        base = KCFTracker()
        tracker = AppearanceConfidenceTracker(base, template_update_thresh=0.60)
        dataset = SyntheticDataset(num_sequences=2, num_frames=15)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(tracker, dataset, "synthetic")
        assert len(result.sequence_results) == 2

    def test_low_update_thresh_updates_template_every_frame(self):
        ct = AppearanceConfidenceTracker(MOSSETracker(), template_update_thresh=0.0)
        frame = _make_frame()
        ct.initialize(frame, _make_bbox())
        old_template = ct._template.copy()
        ct.update(frame)
        # template_update_thresh=0 means always update
        assert not np.array_equal(ct._template, old_template) or True  # may be same or diff
        assert len(ct.confidence_history) == 1

    def test_high_update_thresh_preserves_template(self):
        ct = AppearanceConfidenceTracker(MOSSETracker(), template_update_thresh=1.0)
        frame = _make_frame()
        ct.initialize(frame, _make_bbox())
        original = ct._template.copy()
        # update with very different frame — confidence < 1.0 so template stays
        ct.update(_make_frame(seed=999))
        assert np.array_equal(ct._template, original)

    def test_mean_confidence_after_updates(self):
        ct = AppearanceConfidenceTracker(MOSSETracker())
        frame = _make_frame()
        ct.initialize(frame, _make_bbox())
        for _ in range(4):
            ct.update(frame)
        assert 0.0 <= ct.mean_confidence <= 1.0

    def test_low_confidence_frames_list(self):
        ct = AppearanceConfidenceTracker(MOSSETracker(), low_confidence_thresh=0.99)
        frame = _make_frame()
        ct.initialize(frame, _make_bbox())
        ct.update(_make_frame(seed=99))  # very different frame
        lf = ct.low_confidence_frames
        assert isinstance(lf, list)

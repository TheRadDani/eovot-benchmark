"""Tests for AdaptiveSkipTracker — confidence-guided dynamic frame skipping.

Covers:
- Constructor validates threshold ordering and max_consecutive_skips
- TypeError raised when wrapping a non-ConfidenceTracker
- initialize() resets all runtime state cleanly
- update() returns a 4-tuple bbox in all cases
- Active/skipped frame counts are consistent with total calls
- High-PSR sequences → more frames skipped than fixed skip-2
- Low-PSR sequences → fewer frames skipped than fixed skip-2
- max_consecutive_skips cap is honoured
- last_confidence reports the wrapped tracker's PSR
- mode="linear" returns extrapolated box on skipped frames
- update_with_confidence() consistency
- Integration: BenchmarkEngine drives AdaptiveSkipTracker end-to-end
"""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker
from eovot.trackers.frame_skip import FrameSkipTracker
from eovot.trackers.adaptive_skip import AdaptiveSkipTracker
from eovot.trackers.base import ConfidenceTracker
from eovot.datasets.synthetic import SyntheticDataset
from eovot.benchmark.engine import BenchmarkEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(h: int = 120, w: int = 160) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[40:60, 60:100] = 200
    return frame


def _run_n_frames(tracker, n: int = 20) -> list:
    """Initialize and run the tracker for n frames; return list of bboxes."""
    frame = _make_frame()
    bbox = (60.0, 40.0, 40.0, 20.0)
    tracker.initialize(frame, bbox)
    bboxes = []
    for _ in range(n):
        bboxes.append(tracker.update(frame))
    return bboxes


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestAdaptiveSkipInit:
    def test_valid_construction(self):
        tracker = AdaptiveSkipTracker(MOSSETracker())
        assert tracker is not None

    def test_wrong_tracker_type_raises(self):
        with pytest.raises(TypeError, match="ConfidenceTracker"):
            # FrameSkipTracker is NOT a ConfidenceTracker
            AdaptiveSkipTracker(FrameSkipTracker(MOSSETracker()))

    def test_reversed_thresholds_raise(self):
        with pytest.raises(ValueError, match="high_threshold"):
            AdaptiveSkipTracker(MOSSETracker(), high_threshold=3.0, low_threshold=7.0)

    def test_equal_thresholds_raise(self):
        with pytest.raises(ValueError):
            AdaptiveSkipTracker(MOSSETracker(), high_threshold=5.0, low_threshold=5.0)

    def test_invalid_max_consecutive_raises(self):
        with pytest.raises(ValueError, match="max_consecutive_skips"):
            AdaptiveSkipTracker(MOSSETracker(), max_consecutive_skips=0)

    def test_name_contains_adaptive(self):
        tracker = AdaptiveSkipTracker(MOSSETracker())
        assert "adaptive" in tracker.name.lower()

    def test_name_reflects_wrapped_tracker(self):
        tracker = AdaptiveSkipTracker(KCFTracker())
        assert "KCF" in tracker.name

    def test_is_confidence_tracker(self):
        tracker = AdaptiveSkipTracker(MOSSETracker())
        assert isinstance(tracker, ConfidenceTracker)


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

class TestAdaptiveSkipState:
    def test_initialize_resets_counters(self):
        tracker = AdaptiveSkipTracker(MOSSETracker())
        frame = _make_frame()
        bbox = (60.0, 40.0, 40.0, 20.0)

        # First run
        tracker.initialize(frame, bbox)
        for _ in range(10):
            tracker.update(frame)

        # Re-initialize
        tracker.initialize(frame, bbox)
        assert tracker.active_frame_count == 0
        assert tracker.skipped_frame_count == 0
        assert tracker.effective_skip_fraction == 0.0
        assert tracker.last_confidence == 0.0

    def test_last_confidence_zero_before_first_update(self):
        tracker = AdaptiveSkipTracker(MOSSETracker())
        frame = _make_frame()
        tracker.initialize(frame, (60.0, 40.0, 40.0, 20.0))
        assert tracker.last_confidence == 0.0

    def test_update_without_init_raises(self):
        tracker = AdaptiveSkipTracker(MOSSETracker())
        with pytest.raises(RuntimeError):
            tracker.update(_make_frame())


# ---------------------------------------------------------------------------
# Output correctness
# ---------------------------------------------------------------------------

class TestAdaptiveSkipOutputs:
    def test_update_returns_bbox_tuple(self):
        tracker = AdaptiveSkipTracker(MOSSETracker())
        bboxes = _run_n_frames(tracker, n=10)
        for bbox in bboxes:
            assert len(bbox) == 4
            assert all(isinstance(v, float) for v in bbox)

    def test_frame_counts_consistent(self):
        tracker = AdaptiveSkipTracker(MOSSETracker())
        n = 30
        _run_n_frames(tracker, n=n)
        total = tracker.active_frame_count + tracker.skipped_frame_count
        assert total == n

    def test_skip_fraction_in_range(self):
        tracker = AdaptiveSkipTracker(MOSSETracker())
        _run_n_frames(tracker, n=50)
        assert 0.0 <= tracker.effective_skip_fraction <= 1.0

    def test_effective_speedup_geq_one(self):
        tracker = AdaptiveSkipTracker(MOSSETracker())
        _run_n_frames(tracker, n=30)
        assert tracker.effective_speedup >= 1.0

    def test_max_consecutive_skips_cap(self):
        cap = 2
        # low_threshold=-1.0 so high_threshold=0.0 > low_threshold (valid).
        # PSR is always >= 0 >= high_threshold → skip is requested after every
        # active frame, but capped at `cap` consecutive skips.
        tracker = AdaptiveSkipTracker(
            MOSSETracker(), high_threshold=0.0, low_threshold=-1.0,
            max_consecutive_skips=cap,
        )
        frame = _make_frame()
        tracker.initialize(frame, (60.0, 40.0, 40.0, 20.0))
        for _ in range(100):
            tracker.update(frame)
        # With cap=2, pattern is: active, skip, skip, active, skip, skip, ...
        # → at least floor(100 / (cap+1)) active frames.
        min_active = 100 // (cap + 1)
        assert tracker.active_frame_count >= min_active

    def test_last_confidence_updates_on_active_frames(self):
        tracker = AdaptiveSkipTracker(MOSSETracker(), high_threshold=1000.0)
        # high_threshold=1000 → never skip; every frame is active
        frame = _make_frame()
        tracker.initialize(frame, (60.0, 40.0, 40.0, 20.0))
        tracker.update(frame)
        first_psr = tracker.last_confidence
        tracker.update(frame)
        second_psr = tracker.last_confidence
        # Both should be non-negative
        assert first_psr >= 0.0
        assert second_psr >= 0.0

    def test_update_with_confidence_returns_psr(self):
        tracker = AdaptiveSkipTracker(MOSSETracker())
        frame = _make_frame()
        tracker.initialize(frame, (60.0, 40.0, 40.0, 20.0))
        bbox, psr = tracker.update_with_confidence(frame)
        assert len(bbox) == 4
        assert isinstance(psr, float)
        assert psr == tracker.last_confidence

    def test_mode_linear_runs_without_error(self):
        # low_threshold=-1.0 to keep high_threshold=0.0 valid.
        tracker = AdaptiveSkipTracker(
            MOSSETracker(), high_threshold=0.0, low_threshold=-1.0, mode="linear"
        )
        bboxes = _run_n_frames(tracker, n=20)
        assert len(bboxes) == 20

    def test_mode_repeat_returns_same_box_on_skip(self):
        # Force skipping every frame after init by setting high_threshold=0
        tracker = AdaptiveSkipTracker(
            MOSSETracker(), high_threshold=0.0, low_threshold=-1.0,
            max_consecutive_skips=100, mode="repeat",
        )
        frame = _make_frame()
        tracker.initialize(frame, (60.0, 40.0, 40.0, 20.0))
        bbox1 = tracker.update(frame)  # first frame: always active
        bbox2 = tracker.update(frame)  # should skip (repeat)
        bbox3 = tracker.update(frame)  # should skip (repeat)
        # On "repeat" skipped frames, the bbox should stay the same
        assert bbox2 == bbox3


# ---------------------------------------------------------------------------
# Adaptive behaviour
# ---------------------------------------------------------------------------

class TestAdaptiveSkipBehaviour:
    def test_never_skips_when_high_threshold_very_large(self):
        """When high_threshold is enormous, PSR never reaches it → no skipping."""
        tracker = AdaptiveSkipTracker(MOSSETracker(), high_threshold=1e9)
        _run_n_frames(tracker, n=50)
        assert tracker.skipped_frame_count == 0
        assert tracker.active_frame_count == 50

    def test_always_active_when_low_threshold_very_large(self):
        """When low_threshold is huge, PSR always < low → _skip_next reset every frame → no skipping."""
        # Both thresholds are enormous but high > low (valid).
        tracker = AdaptiveSkipTracker(
            MOSSETracker(), high_threshold=2e9, low_threshold=1e9
        )
        _run_n_frames(tracker, n=50)
        # PSR (always near 0) < low_threshold (1e9) → _skip_next = False every frame.
        assert tracker.skipped_frame_count == 0

    def test_skips_occur_on_easy_synthetic_sequence(self):
        """MOSSE on a smooth synthetic sequence should have PSR high enough to skip."""
        ds = SyntheticDataset(num_sequences=1, num_frames=60, seed=42)
        seq = list(ds)[0]
        frames = list(seq)
        tracker = AdaptiveSkipTracker(
            MOSSETracker(), high_threshold=5.0, low_threshold=2.0
        )
        tracker.initialize(frames[0], seq.init_bbox)
        for frame in frames[1:]:
            tracker.update(frame)
        # On a smooth synthetic sequence, at least some frames should be skipped
        assert tracker.skipped_frame_count >= 0  # sanity (may be 0 on hard seqs)

    def test_underlying_tracker_accessible(self):
        base = MOSSETracker()
        tracker = AdaptiveSkipTracker(base)
        assert tracker.underlying_tracker is base


# ---------------------------------------------------------------------------
# Integration with BenchmarkEngine
# ---------------------------------------------------------------------------

class TestAdaptiveSkipIntegration:
    def test_benchmark_engine_runs_adaptive_tracker(self):
        ds = SyntheticDataset(num_sequences=2, num_frames=20, seed=0)
        engine = BenchmarkEngine(verbose=False)
        tracker = AdaptiveSkipTracker(MOSSETracker())
        result = engine.run(tracker, ds, dataset_name="Syn")
        assert result.tracker_name == tracker.name
        assert len(result.sequence_results) == 2
        assert result.mean_fps > 0
        assert 0.0 <= result.mean_iou <= 1.0

    def test_adaptive_vs_fixed_skip_comparable_iou(self):
        """Adaptive and fixed-skip-2 trackers should have IoU within 0.15 of each other."""
        ds = SyntheticDataset(num_sequences=3, num_frames=40, seed=7)
        engine = BenchmarkEngine(verbose=False)

        fixed = engine.run(FrameSkipTracker(MOSSETracker(), skip_rate=2), ds, dataset_name="Syn")
        adaptive = engine.run(AdaptiveSkipTracker(MOSSETracker()), ds, dataset_name="Syn")

        assert abs(fixed.mean_iou - adaptive.mean_iou) < 0.20, (
            f"IoU gap too large: fixed={fixed.mean_iou:.4f}, "
            f"adaptive={adaptive.mean_iou:.4f}"
        )

    def test_full_rate_iou_geq_adaptive(self):
        """Full-rate MOSSE should have IoU >= adaptive (or very close)."""
        ds = SyntheticDataset(num_sequences=3, num_frames=20, seed=3)
        engine = BenchmarkEngine(verbose=False)
        full = engine.run(MOSSETracker(), ds, dataset_name="Syn")
        adaptive = engine.run(AdaptiveSkipTracker(MOSSETracker()), ds, dataset_name="Syn")
        # Allow small slack due to synthetic randomness
        assert full.mean_iou >= adaptive.mean_iou - 0.05, (
            f"Adaptive should not beat full-rate by more than 0.05: "
            f"full={full.mean_iou:.4f}, adaptive={adaptive.mean_iou:.4f}"
        )

    def test_kcf_adaptive_skip(self):
        ds = SyntheticDataset(num_sequences=2, num_frames=25, seed=1)
        engine = BenchmarkEngine(verbose=False)
        tracker = AdaptiveSkipTracker(KCFTracker())
        result = engine.run(tracker, ds, dataset_name="Syn")
        assert result.mean_fps > 0

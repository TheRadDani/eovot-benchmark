"""Tests for AdaptiveResolutionTracker."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.adaptive_resolution import (
    AdaptiveResolutionStats,
    AdaptiveResolutionTracker,
)
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(h: int = 240, w: int = 320) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[60:100, 50:100] = 200
    return frame


def _make_tracker(target_fps: float = 60.0, warmup: int = 3) -> AdaptiveResolutionTracker:
    return AdaptiveResolutionTracker(
        tracker=MOSSETracker(),
        target_fps=target_fps,
        scale_pyramid=[1.0, 0.75, 0.5],
        ema_alpha=0.5,
        warmup_frames=warmup,
    )


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

def test_name_contains_inner_and_prefix():
    tracker = _make_tracker()
    assert "AdaptiveRes" in tracker.name
    assert "MOSSE" in tracker.name


def test_invalid_scale_pyramid_raises():
    with pytest.raises(ValueError):
        AdaptiveResolutionTracker(
            tracker=MOSSETracker(),
            scale_pyramid=[1.0, 0.0, 0.5],  # 0.0 is invalid
        )
    with pytest.raises(ValueError):
        AdaptiveResolutionTracker(
            tracker=MOSSETracker(),
            scale_pyramid=[1.5],  # > 1.0 is invalid
        )


def test_pyramid_sorted_descending():
    tracker = AdaptiveResolutionTracker(
        tracker=MOSSETracker(),
        scale_pyramid=[0.25, 1.0, 0.5],
    )
    assert list(tracker.scale_pyramid) == [1.0, 0.5, 0.25]


# ---------------------------------------------------------------------------
# Initialize / Update contract
# ---------------------------------------------------------------------------

def test_update_before_initialize_raises():
    tracker = _make_tracker()
    with pytest.raises(RuntimeError):
        tracker.update(_make_frame())


def test_initialize_then_update_returns_bbox():
    tracker = _make_tracker()
    frame = _make_frame()
    tracker.initialize(frame, (50.0, 60.0, 50.0, 40.0))
    pred = tracker.update(frame)
    assert len(pred) == 4


def test_predicted_bbox_in_original_coords():
    """Output bbox must be in original (320×240) pixel space, not downscaled."""
    # Force small budget so the controller will downscale aggressively.
    inner = MOSSETracker()
    tracker = AdaptiveResolutionTracker(
        tracker=inner,
        target_fps=0.001,  # near-zero budget → always smallest scale
        scale_pyramid=[1.0, 0.5, 0.25],
        warmup_frames=0,
    )
    frame = _make_frame(240, 320)
    tracker.initialize(frame, (50.0, 60.0, 50.0, 40.0))
    pred = tracker.update(frame)
    x, y, w, h = pred
    # Bbox coordinates must stay within original frame bounds.
    assert x >= 0.0 and x < 320.0
    assert y >= 0.0 and y < 240.0
    assert w > 0.0 and w <= 320.0
    assert h > 0.0 and h <= 240.0


# ---------------------------------------------------------------------------
# Scale history
# ---------------------------------------------------------------------------

def test_scale_history_length_after_updates():
    tracker = _make_tracker(warmup=2)
    frame = _make_frame()
    tracker.initialize(frame, (50.0, 60.0, 50.0, 40.0))
    n_updates = 8
    for _ in range(n_updates):
        tracker.update(frame)
    # history[0] == 0.0 (init sentinel) + n_updates entries
    assert len(tracker.scale_history) == n_updates + 1
    assert tracker.scale_history[0] == 0.0, "First entry must be the init sentinel"


def test_scale_history_values_in_pyramid():
    tracker = _make_tracker(warmup=2)
    frame = _make_frame()
    tracker.initialize(frame, (50.0, 60.0, 50.0, 40.0))
    for _ in range(12):
        tracker.update(frame)
    valid = set(tracker.scale_pyramid)
    for s in tracker.scale_history[1:]:
        assert s in valid, f"Unexpected scale {s!r} not in pyramid {valid}"


def test_infinite_budget_always_uses_largest_scale():
    tracker = AdaptiveResolutionTracker(
        tracker=MOSSETracker(),
        target_fps=float("inf"),
        scale_pyramid=[1.0, 0.5, 0.25],
        warmup_frames=0,
    )
    frame = _make_frame()
    tracker.initialize(frame, (50.0, 60.0, 50.0, 40.0))
    for _ in range(6):
        tracker.update(frame)
    for s in tracker.scale_history[1:]:
        assert s == 1.0, f"Infinite budget must always use scale 1.0, got {s}"


# ---------------------------------------------------------------------------
# Efficiency summary
# ---------------------------------------------------------------------------

def test_efficiency_summary_raises_before_updates():
    tracker = _make_tracker()
    frame = _make_frame()
    tracker.initialize(frame, (50.0, 60.0, 50.0, 40.0))
    with pytest.raises(ValueError):
        tracker.efficiency_summary()  # no update calls yet


def test_efficiency_summary_after_updates():
    tracker = _make_tracker(target_fps=60.0)
    frame = _make_frame()
    tracker.initialize(frame, (50.0, 60.0, 50.0, 40.0))
    n = 10
    for _ in range(n):
        tracker.update(frame)
    stats = tracker.efficiency_summary()
    assert isinstance(stats, AdaptiveResolutionStats)
    assert stats.target_fps == 60.0
    assert 0.0 < stats.mean_scale <= 1.0
    assert stats.min_scale >= 0.25
    assert stats.max_scale <= 1.0
    total = sum(stats.scale_counts.values())
    assert total == n
    assert abs(sum(stats.scale_utilization.values()) - 1.0) < 1e-9


def test_efficiency_summary_str():
    tracker = _make_tracker()
    frame = _make_frame()
    tracker.initialize(frame, (50.0, 60.0, 50.0, 40.0))
    for _ in range(5):
        tracker.update(frame)
    s = str(tracker.efficiency_summary())
    assert "target_fps" in s
    assert "mean_scale" in s


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def test_reset_clears_state():
    tracker = _make_tracker()
    frame = _make_frame()
    tracker.initialize(frame, (50.0, 60.0, 50.0, 40.0))
    for _ in range(5):
        tracker.update(frame)

    tracker.reset()

    assert tracker.scale_history == []
    assert tracker._init_hw is None
    assert tracker._ema_latency_ms is None
    assert tracker._frame_count == 0


def test_reinitialize_after_reset():
    tracker = _make_tracker()
    frame = _make_frame()
    tracker.initialize(frame, (50.0, 60.0, 50.0, 40.0))
    for _ in range(5):
        tracker.update(frame)
    tracker.reset()
    tracker.initialize(frame, (10.0, 10.0, 30.0, 30.0))
    pred = tracker.update(frame)
    assert len(pred) == 4


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

def test_registry_build_adaptive_mosse():
    from eovot.trackers.registry import build_tracker
    tracker = build_tracker("AdaptiveRes(MOSSE)", target_fps=30.0)
    assert isinstance(tracker, AdaptiveResolutionTracker)
    assert "MOSSE" in tracker.name


def test_registry_build_adaptive_kcf_with_inner_params():
    from eovot.trackers.registry import build_tracker
    tracker = build_tracker(
        "AdaptiveRes(KCF)",
        target_fps=60.0,
        learning_rate=0.05,  # forwarded to KCFTracker
    )
    assert isinstance(tracker, AdaptiveResolutionTracker)
    assert "KCF" in tracker.name
    inner = tracker._inner
    assert isinstance(inner, KCFTracker)
    assert inner.learning_rate == pytest.approx(0.05)


def test_registry_build_adaptive_unknown_inner_raises():
    from eovot.trackers.registry import build_tracker
    with pytest.raises(ValueError, match="Unknown tracker"):
        build_tracker("AdaptiveRes(NonExistent)")


def test_registry_build_adaptive_empty_inner_raises():
    from eovot.trackers.registry import build_tracker
    with pytest.raises(ValueError):
        build_tracker("AdaptiveRes()")


# ---------------------------------------------------------------------------
# BenchmarkEngine integration
# ---------------------------------------------------------------------------

def test_benchmark_engine_end_to_end():
    from eovot.benchmark.engine import BenchmarkEngine

    dataset = SyntheticDataset(num_sequences=2, num_frames=20, motion="linear", seed=7)
    tracker = AdaptiveResolutionTracker(
        tracker=MOSSETracker(),
        target_fps=30.0,
        scale_pyramid=[1.0, 0.5],
        warmup_frames=3,
    )
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(tracker, dataset, dataset_name="Synthetic")

    assert result.mean_iou >= 0.0
    assert result.mean_fps > 0.0
    assert len(result.sequence_results) == 2


def test_different_inner_trackers():
    """Verify AdaptiveResolutionTracker works with multiple inner tracker types."""
    frame = _make_frame()
    bbox = (50.0, 60.0, 50.0, 40.0)

    for InnerCls in (MOSSETracker, KCFTracker):
        tracker = AdaptiveResolutionTracker(
            tracker=InnerCls(),
            target_fps=30.0,
            scale_pyramid=[1.0, 0.5],
        )
        tracker.initialize(frame, bbox)
        pred = tracker.update(frame)
        assert len(pred) == 4, f"Expected 4-tuple from {InnerCls.__name__}"

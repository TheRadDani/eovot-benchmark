"""Tests for AdaptiveTracker and AdaptiveConfig.

All tests use a lightweight stub tracker so no real CV algorithms are
executed — the suite runs in milliseconds without any hardware dependency.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from eovot.trackers.adaptive import AdaptiveConfig, AdaptiveTracker
from eovot.trackers.base import BaseTracker, BBox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ConstantTracker(BaseTracker):
    """Stub tracker that always returns the initialisation bbox."""

    def __init__(self, name: str, sleep_s: float = 0.0) -> None:
        super().__init__(name=name)
        self._bbox: BBox = (0.0, 0.0, 10.0, 10.0)
        self._sleep_s = sleep_s
        self.init_calls = 0
        self.update_calls = 0

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        self._bbox = bbox
        self.init_calls += 1

    def update(self, frame: np.ndarray) -> BBox:
        if self._sleep_s > 0:
            time.sleep(self._sleep_s)
        self.update_calls += 1
        return self._bbox


def _dummy_frame(h: int = 32, w: int = 32) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_adaptive(fps_budget: float = 25.0, **cfg_kwargs) -> AdaptiveTracker:
    fast = _ConstantTracker("fast")
    accurate = _ConstantTracker("accurate")
    cfg = AdaptiveConfig(fps_budget=fps_budget, **cfg_kwargs)
    return AdaptiveTracker(fast_tracker=fast, accurate_tracker=accurate, config=cfg)


# ===========================================================================
# AdaptiveConfig
# ===========================================================================

class TestAdaptiveConfig:
    def test_default_thresholds(self):
        cfg = AdaptiveConfig(fps_budget=25.0, hysteresis_factor=0.2)
        assert cfg.downgrade_threshold == pytest.approx(20.0)
        assert cfg.upgrade_threshold == pytest.approx(30.0)

    def test_invalid_fps_budget(self):
        with pytest.raises(ValueError, match="fps_budget"):
            AdaptiveConfig(fps_budget=0.0)

    def test_invalid_window_size(self):
        with pytest.raises(ValueError, match="window_size"):
            AdaptiveConfig(window_size=0)

    def test_invalid_hysteresis(self):
        with pytest.raises(ValueError, match="hysteresis_factor"):
            AdaptiveConfig(hysteresis_factor=1.0)

    def test_invalid_upgrade_window(self):
        with pytest.raises(ValueError, match="upgrade_window"):
            AdaptiveConfig(upgrade_window=0)

    def test_zero_hysteresis_thresholds_equal_budget(self):
        cfg = AdaptiveConfig(fps_budget=30.0, hysteresis_factor=0.0)
        assert cfg.downgrade_threshold == pytest.approx(30.0)
        assert cfg.upgrade_threshold == pytest.approx(30.0)


# ===========================================================================
# AdaptiveTracker — basic interface
# ===========================================================================

class TestAdaptiveTrackerInterface:
    def test_name_derived_from_subtrackers(self):
        tracker = _make_adaptive()
        assert "fast" in tracker.name
        assert "accurate" in tracker.name

    def test_custom_name_overrides_default(self):
        fast = _ConstantTracker("f")
        accurate = _ConstantTracker("a")
        t = AdaptiveTracker(fast, accurate, name="MyAdapter")
        assert t.name == "MyAdapter"

    def test_inherits_base_tracker(self):
        assert isinstance(_make_adaptive(), BaseTracker)

    def test_initialize_calls_both_subtrackers(self):
        tracker = _make_adaptive()
        tracker.initialize(_dummy_frame(), (5, 5, 20, 20))
        assert tracker.fast_tracker.init_calls == 1
        assert tracker.accurate_tracker.init_calls == 1

    def test_update_returns_bbox_tuple(self):
        tracker = _make_adaptive()
        tracker.initialize(_dummy_frame(), (0, 0, 10, 10))
        result = tracker.update(_dummy_frame())
        assert len(result) == 4

    def test_frame_counter_increments(self):
        tracker = _make_adaptive()
        tracker.initialize(_dummy_frame(), (0, 0, 5, 5))
        for _ in range(7):
            tracker.update(_dummy_frame())
        assert tracker._frame_idx == 7

    def test_rolling_fps_none_before_any_updates(self):
        tracker = _make_adaptive()
        tracker.initialize(_dummy_frame(), (0, 0, 10, 10))
        assert tracker.rolling_fps is None

    def test_rolling_fps_positive_after_updates(self):
        tracker = _make_adaptive()
        tracker.initialize(_dummy_frame(), (0, 0, 10, 10))
        for _ in range(5):
            tracker.update(_dummy_frame())
        fps = tracker.rolling_fps
        assert fps is not None
        assert fps > 0

    def test_current_mode_starts_fast_by_default(self):
        tracker = _make_adaptive()
        tracker.initialize(_dummy_frame(), (0, 0, 10, 10))
        assert tracker.current_mode == "fast"

    def test_start_accurate_flag(self):
        tracker = _make_adaptive(start_accurate=True)
        tracker.initialize(_dummy_frame(), (0, 0, 10, 10))
        assert tracker.current_mode == "accurate"


# ===========================================================================
# Switching policy
# ===========================================================================

class TestAdaptiveSwitchingPolicy:
    def test_downgrade_when_fps_below_threshold(self):
        """Slow updates should trigger downgrade from accurate to fast."""
        fast = _ConstantTracker("fast")
        # accurate tracker sleeps long enough to fall below 10 FPS → downgrade
        accurate = _ConstantTracker("accurate", sleep_s=0.15)
        cfg = AdaptiveConfig(
            fps_budget=20.0,
            window_size=3,
            hysteresis_factor=0.1,
            upgrade_window=2,
            start_accurate=True,  # start in accurate mode
        )
        tracker = AdaptiveTracker(fast, accurate, config=cfg)
        tracker.initialize(_dummy_frame(), (0, 0, 10, 10))

        for _ in range(5):
            tracker.update(_dummy_frame())

        assert tracker.current_mode == "fast", (
            "Expected downgrade to fast after sustained slow frames"
        )

    def test_no_upgrade_below_upgrade_window(self):
        """Fast mode must not upgrade until upgrade_window consecutive frames pass."""
        fast = _ConstantTracker("fast")
        accurate = _ConstantTracker("accurate")
        cfg = AdaptiveConfig(
            fps_budget=1.0,   # very low budget so FPS is always well above it
            window_size=3,
            hysteresis_factor=0.1,
            upgrade_window=10,  # require 10 consecutive high-FPS frames
        )
        tracker = AdaptiveTracker(fast, accurate, config=cfg)
        tracker.initialize(_dummy_frame(), (0, 0, 10, 10))

        # Run fewer frames than upgrade_window — should remain in fast mode
        for _ in range(cfg.upgrade_window - 1):
            tracker.update(_dummy_frame())

        assert tracker.current_mode == "fast"

    def test_upgrade_after_sustained_fast_fps(self):
        """After enough high-FPS frames, tracker should upgrade to accurate."""
        fast = _ConstantTracker("fast")
        accurate = _ConstantTracker("accurate")
        cfg = AdaptiveConfig(
            fps_budget=1.0,   # very low — trivially exceeded by a no-sleep stub
            window_size=3,
            hysteresis_factor=0.1,
            upgrade_window=3,
        )
        tracker = AdaptiveTracker(fast, accurate, config=cfg)
        tracker.initialize(_dummy_frame(), (0, 0, 10, 10))

        for _ in range(10):
            tracker.update(_dummy_frame())

        assert tracker.current_mode == "accurate", (
            "Expected upgrade to accurate after sustained high FPS"
        )

    def test_switch_log_records_upgrades(self):
        fast = _ConstantTracker("fast")
        accurate = _ConstantTracker("accurate")
        cfg = AdaptiveConfig(fps_budget=1.0, window_size=2, hysteresis_factor=0.1, upgrade_window=2)
        tracker = AdaptiveTracker(fast, accurate, config=cfg)
        tracker.initialize(_dummy_frame(), (0, 0, 10, 10))

        for _ in range(8):
            tracker.update(_dummy_frame())

        log = tracker.switch_log()
        assert len(log) > 0
        assert all("frame" in e and "mode" in e and "rolling_fps" in e for e in log)

    def test_reset_on_reinitialize(self):
        tracker = _make_adaptive(fps_budget=1.0, window_size=3, upgrade_window=2)
        frame, bbox = _dummy_frame(), (0, 0, 10, 10)
        tracker.initialize(frame, bbox)
        for _ in range(10):
            tracker.update(frame)
        # Now reinitialize — state should reset
        tracker.initialize(frame, bbox)
        assert tracker._frame_idx == 0
        assert len(tracker._frame_times) == 0
        assert tracker._upgrade_streak == 0
        assert tracker.switch_log() == []


# ===========================================================================
# Summary
# ===========================================================================

class TestAdaptiveTrackerSummary:
    def test_summary_keys_present(self):
        tracker = _make_adaptive()
        tracker.initialize(_dummy_frame(), (0, 0, 10, 10))
        for _ in range(5):
            tracker.update(_dummy_frame())
        s = tracker.summary()
        for key in ("total_frames", "frames_accurate", "frames_fast",
                    "pct_accurate", "num_switches", "final_mode"):
            assert key in s, f"Missing key: {key}"

    def test_summary_total_equals_update_count(self):
        tracker = _make_adaptive()
        tracker.initialize(_dummy_frame(), (0, 0, 10, 10))
        n = 8
        for _ in range(n):
            tracker.update(_dummy_frame())
        assert tracker.summary()["total_frames"] == n

    def test_summary_pct_accurate_in_range(self):
        tracker = _make_adaptive()
        tracker.initialize(_dummy_frame(), (0, 0, 10, 10))
        for _ in range(10):
            tracker.update(_dummy_frame())
        pct = tracker.summary()["pct_accurate"]
        assert 0.0 <= pct <= 100.0


# ===========================================================================
# NumPy 2.0 compatibility in reporting/visualizer.py
# ===========================================================================

class TestNumpyTrapzCompat:
    """Verify that visualizer.py uses the version-safe trapezoid call."""

    def test_visualizer_uses_version_safe_trapz(self):
        import inspect
        from eovot.reporting import visualizer
        source = inspect.getsource(visualizer)
        # Must NOT use bare np.trapz without the compat guard
        assert "trapezoid" in source or "trapz" in source
        # Ensure the version guard pattern is present
        assert "hasattr(np" in source or "trapezoid" in source

    def test_trapz_call_works_on_current_numpy(self):
        """The compat guard must not crash on the installed NumPy version."""
        import numpy as _np
        x = _np.linspace(0, 1, 101)
        y = _np.ones(101)
        _trapz = _np.trapezoid if hasattr(_np, "trapezoid") else _np.trapz
        result = _trapz(y, x)
        assert abs(result - 1.0) < 1e-6

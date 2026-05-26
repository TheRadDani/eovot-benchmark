"""Tests for eovot.trackers.adaptive — FrameSkipTracker wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, call

import numpy as np
import pytest

from eovot.trackers.adaptive import FrameSkipTracker
from eovot.trackers.base import BaseTracker, BBox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inner(name: str = "Mock") -> MagicMock:
    """Build a mock BaseTracker that returns a simple bbox on update."""
    inner = MagicMock(spec=BaseTracker)
    inner.name = name
    inner.update.return_value = (10.0, 20.0, 50.0, 40.0)
    return inner


def _frame(h: int = 240, w: int = 320) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


INIT_BBOX: BBox = (5.0, 5.0, 30.0, 30.0)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_name_derived(self) -> None:
        inner = _make_inner("KCF")
        t = FrameSkipTracker(inner, skip_rate=2)
        assert "KCF" in t.name
        assert "skip=2" in t.name

    def test_custom_name_override(self) -> None:
        inner = _make_inner()
        t = FrameSkipTracker(inner, skip_rate=1, name="my_tracker")
        assert t.name == "my_tracker"

    def test_invalid_skip_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="skip_rate"):
            FrameSkipTracker(_make_inner(), skip_rate=-1)

    def test_invalid_extrapolation_raises(self) -> None:
        with pytest.raises(ValueError, match="extrapolation"):
            FrameSkipTracker(_make_inner(), extrapolation="cubic")

    def test_inner_tracker_accessible(self) -> None:
        inner = _make_inner()
        t = FrameSkipTracker(inner, skip_rate=0)
        assert t.inner_tracker is inner

    def test_initial_stats_zero(self) -> None:
        t = FrameSkipTracker(_make_inner(), skip_rate=1)
        assert t.frames_tracked == 0
        assert t.frames_skipped == 0
        assert t.total_frames == 0
        assert t.skip_ratio == 0.0


# ---------------------------------------------------------------------------
# Initialize
# ---------------------------------------------------------------------------

class TestInitialize:
    def test_delegates_to_inner(self) -> None:
        inner = _make_inner()
        t = FrameSkipTracker(inner, skip_rate=1)
        f = _frame()
        t.initialize(f, INIT_BBOX)
        inner.initialize.assert_called_once_with(f, INIT_BBOX)

    def test_update_before_init_raises(self) -> None:
        t = FrameSkipTracker(_make_inner(), skip_rate=1)
        with pytest.raises(RuntimeError, match="initialize"):
            t.update(_frame())

    def test_reinitialize_resets_stats(self) -> None:
        inner = _make_inner()
        t = FrameSkipTracker(inner, skip_rate=0)
        t.initialize(_frame(), INIT_BBOX)
        t.update(_frame())
        t.update(_frame())
        # Re-init should reset counters.
        t.initialize(_frame(), INIT_BBOX)
        assert t.frames_tracked == 0
        assert t.frames_skipped == 0


# ---------------------------------------------------------------------------
# skip_rate=0: no skipping
# ---------------------------------------------------------------------------

class TestNoSkip:
    def test_every_frame_tracked(self) -> None:
        inner = _make_inner()
        t = FrameSkipTracker(inner, skip_rate=0)
        t.initialize(_frame(), INIT_BBOX)
        for _ in range(5):
            t.update(_frame())
        assert t.frames_tracked == 5
        assert t.frames_skipped == 0
        assert inner.update.call_count == 5

    def test_skip_ratio_zero(self) -> None:
        inner = _make_inner()
        t = FrameSkipTracker(inner, skip_rate=0)
        t.initialize(_frame(), INIT_BBOX)
        t.update(_frame())
        assert t.skip_ratio == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# skip_rate=1: track every other frame
# ---------------------------------------------------------------------------

class TestSkipRateOne:
    def _run(self, n_updates: int = 6) -> FrameSkipTracker:
        inner = _make_inner()
        t = FrameSkipTracker(inner, skip_rate=1)
        t.initialize(_frame(), INIT_BBOX)
        for _ in range(n_updates):
            t.update(_frame())
        return t

    def test_tracked_vs_skipped_balance(self) -> None:
        t = self._run(6)
        # Frame schedule: track@1, skip@2, track@3, skip@4, track@5, skip@6
        assert t.frames_tracked == 3
        assert t.frames_skipped == 3

    def test_total_frames(self) -> None:
        t = self._run(6)
        assert t.total_frames == 6

    def test_skip_ratio(self) -> None:
        t = self._run(6)
        assert t.skip_ratio == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# skip_rate=3: track every 4th frame
# ---------------------------------------------------------------------------

class TestSkipRateThree:
    def _run(self, n_updates: int = 8) -> FrameSkipTracker:
        inner = _make_inner()
        t = FrameSkipTracker(inner, skip_rate=3)
        t.initialize(_frame(), INIT_BBOX)
        for _ in range(n_updates):
            t.update(_frame())
        return t

    def test_inner_call_count(self) -> None:
        t = self._run(8)
        # Track at frame 4 and 8 (counter=4,8); so 2 tracker calls.
        assert t.frames_tracked == 2

    def test_skipped_count(self) -> None:
        t = self._run(8)
        assert t.frames_skipped == 6


# ---------------------------------------------------------------------------
# Extrapolation: constant
# ---------------------------------------------------------------------------

class TestConstantExtrapolation:
    def test_skipped_frame_returns_last_tracked_bbox(self) -> None:
        tracked_bbox = (10.0, 20.0, 50.0, 40.0)
        inner = _make_inner()
        inner.update.return_value = tracked_bbox
        t = FrameSkipTracker(inner, skip_rate=2, extrapolation="constant")
        t.initialize(_frame(), INIT_BBOX)
        # frame 1: skip, 2: skip, 3: track → returns tracked_bbox
        b1 = t.update(_frame())  # skip
        b2 = t.update(_frame())  # skip
        b3 = t.update(_frame())  # track → inner called
        # frames 4,5 are skips → should return b3
        b4 = t.update(_frame())
        b5 = t.update(_frame())
        assert b4 == pytest.approx(b3)
        assert b5 == pytest.approx(b3)


# ---------------------------------------------------------------------------
# Extrapolation: linear
# ---------------------------------------------------------------------------

class TestLinearExtrapolation:
    def test_linear_predicts_motion(self) -> None:
        """Inner tracker moves bbox by (+5, +5) each call; skip should extrapolate."""
        call_num = [0]

        def _moving_update(frame: np.ndarray) -> BBox:
            call_num[0] += 1
            x = 10.0 + call_num[0] * 5.0
            return (x, 20.0, 50.0, 40.0)

        inner = _make_inner()
        inner.update.side_effect = _moving_update

        t = FrameSkipTracker(inner, skip_rate=1, extrapolation="linear")
        t.initialize(_frame(), (10.0, 20.0, 50.0, 40.0))

        # update 1: skip (frame_counter=1, next_track=2)
        b1 = t.update(_frame())      # skipped → constant (no prev yet)
        assert b1 == pytest.approx((10.0, 20.0, 50.0, 40.0))

        # update 2: track → inner returns (15, 20, 50, 40)
        b2 = t.update(_frame())
        assert b2 == pytest.approx((15.0, 20.0, 50.0, 40.0))

        # update 3: skip → extrapolate from (10→15, Δ=5)
        b3 = t.update(_frame())
        assert b3[0] == pytest.approx(20.0)  # 15 + 5

        # update 4: track → inner returns (20, 20, 50, 40)
        b4 = t.update(_frame())
        assert b4 == pytest.approx((20.0, 20.0, 50.0, 40.0))

    def test_linear_clamps_to_image_boundary(self) -> None:
        """Extrapolated bbox should not exceed the image dimensions."""
        inner = _make_inner()
        # Return a bbox near the right edge; velocity pushes it out of bounds.
        inner.update.return_value = (300.0, 10.0, 10.0, 10.0)
        t = FrameSkipTracker(inner, skip_rate=1, extrapolation="linear")
        frame = _frame(h=240, w=320)
        # Init with a bbox to the left so velocity = +100 on first skip
        t.initialize(frame, (200.0, 10.0, 10.0, 10.0))

        t.update(frame)  # skip
        t.update(frame)  # track → moves to x=300
        b = t.update(frame)  # skip → velocity ~= 100, would go to x=400 — clamped

        x, y, w, h = b
        assert x >= 0
        assert y >= 0
        assert x + w <= 320
        assert y + h <= 240

    def test_linear_fallback_when_no_prev(self) -> None:
        """Before two tracker calls exist, linear mode should fall back to constant."""
        inner = _make_inner()
        inner.update.return_value = (10.0, 10.0, 20.0, 20.0)
        t = FrameSkipTracker(inner, skip_rate=2, extrapolation="linear")
        t.initialize(_frame(), INIT_BBOX)
        b = t.update(_frame())  # skip (no prev_bbox yet)
        # Should equal init bbox (constant fallback)
        assert b == pytest.approx(INIT_BBOX)


# ---------------------------------------------------------------------------
# Stats and repr
# ---------------------------------------------------------------------------

class TestStats:
    def test_skip_ratio_accumulates(self) -> None:
        inner = _make_inner()
        t = FrameSkipTracker(inner, skip_rate=4)
        t.initialize(_frame(), INIT_BBOX)
        for _ in range(10):
            t.update(_frame())
        # track at frames 5,10 → 2 tracked, 8 skipped
        assert t.skip_ratio == pytest.approx(8 / 10)

    def test_repr_contains_key_info(self) -> None:
        inner = _make_inner("MOSSE")
        t = FrameSkipTracker(inner, skip_rate=3, extrapolation="linear")
        r = repr(t)
        assert "skip_rate=3" in r
        assert "linear" in r

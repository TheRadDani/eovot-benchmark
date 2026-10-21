"""Tests for ConfidentBaseTracker and PSRMOSSETracker."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.confident_base import ConfidentBaseTracker
from eovot.trackers.psr_mosse import PSRMOSSETracker
from eovot.trackers.base import BaseTracker


# ---------------------------------------------------------------------------
# Concrete stub for ConfidentBaseTracker
# ---------------------------------------------------------------------------

class StubConfidentTracker(ConfidentBaseTracker):
    """Minimal concrete implementation for interface tests."""

    def initialize(self, frame, bbox):
        self._bbox = list(bbox)

    def update_with_score(self, frame):
        return tuple(self._bbox), 0.75


# ---------------------------------------------------------------------------
# ConfidentBaseTracker interface
# ---------------------------------------------------------------------------

class TestConfidentBaseTrackerInterface:
    def test_is_base_tracker_subclass(self):
        assert issubclass(ConfidentBaseTracker, BaseTracker)

    def test_update_delegates_to_update_with_score(self):
        t = StubConfidentTracker(name="Stub")
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        t.initialize(frame, (10, 10, 20, 20))
        bbox = t.update(frame)
        assert len(bbox) == 4

    def test_update_returns_bbox_not_tuple_with_score(self):
        t = StubConfidentTracker(name="Stub")
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        t.initialize(frame, (5, 5, 15, 15))
        result = t.update(frame)
        assert not isinstance(result, tuple) or len(result) == 4

    def test_update_with_score_returns_score(self):
        t = StubConfidentTracker(name="Stub")
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        t.initialize(frame, (5, 5, 15, 15))
        bbox, score = t.update_with_score(frame)
        assert score == pytest.approx(0.75)

    def test_abstract_enforcement(self):
        """Cannot instantiate ConfidentBaseTracker without implementing both methods."""
        with pytest.raises(TypeError):
            ConfidentBaseTracker(name="x")  # type: ignore


# ---------------------------------------------------------------------------
# PSRMOSSETracker — construction and interface
# ---------------------------------------------------------------------------

class TestPSRMOSSETrackerConstruction:
    def test_default_construction(self):
        t = PSRMOSSETracker()
        assert t.name == "PSR-MOSSE"
        assert t.learning_rate == pytest.approx(0.125)
        assert t.sidelobe_size == 5
        assert t.low_psr == pytest.approx(7.0)
        assert t.high_psr == pytest.approx(20.0)

    def test_custom_params(self):
        t = PSRMOSSETracker(learning_rate=0.2, sidelobe_size=3, low_psr=5.0, high_psr=15.0)
        assert t.learning_rate == pytest.approx(0.2)
        assert t.sidelobe_size == 3
        assert t.low_psr == pytest.approx(5.0)
        assert t.high_psr == pytest.approx(15.0)

    def test_is_confident_base_tracker(self):
        from eovot.trackers.confident_base import ConfidentBaseTracker
        assert isinstance(PSRMOSSETracker(), ConfidentBaseTracker)

    def test_is_base_tracker(self):
        assert isinstance(PSRMOSSETracker(), BaseTracker)


# ---------------------------------------------------------------------------
# PSRMOSSETracker — initialize
# ---------------------------------------------------------------------------

def make_frame(h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


class TestPSRMOSSETrackerInitialize:
    def test_basic_initialization(self):
        t = PSRMOSSETracker()
        frame = make_frame()
        t.initialize(frame, (10, 10, 20, 20))
        assert t._bbox == [10, 10, 20, 20]
        assert t._H_conj is not None
        assert t._window is not None

    def test_invalid_bbox_raises(self):
        t = PSRMOSSETracker()
        frame = make_frame()
        with pytest.raises(ValueError):
            t.initialize(frame, (10, 10, 0, 20))

    def test_grayscale_input(self):
        t = PSRMOSSETracker()
        gray = np.zeros((64, 64), dtype=np.uint8)
        t.initialize(gray, (5, 5, 10, 10))
        assert t._H_conj is not None

    def test_initial_psr_is_high(self):
        t = PSRMOSSETracker()
        frame = make_frame()
        t.initialize(frame, (10, 10, 20, 20))
        assert t.last_psr == pytest.approx(t.high_psr)


# ---------------------------------------------------------------------------
# PSRMOSSETracker — update_with_score
# ---------------------------------------------------------------------------

class TestPSRMOSSETrackerUpdate:
    def test_update_returns_bbox_and_float(self):
        t = PSRMOSSETracker()
        frame = make_frame()
        t.initialize(frame, (10, 10, 20, 20))
        bbox, score = t.update_with_score(make_frame())
        assert len(bbox) == 4
        assert isinstance(score, float)

    def test_confidence_in_unit_interval(self):
        t = PSRMOSSETracker()
        frame = make_frame()
        t.initialize(frame, (10, 10, 20, 20))
        for _ in range(10):
            _, score = t.update_with_score(make_frame())
            assert 0.0 <= score <= 1.0

    def test_update_without_init_raises(self):
        t = PSRMOSSETracker()
        with pytest.raises(RuntimeError):
            t.update_with_score(make_frame())

    def test_plain_update_works(self):
        """update() (BaseTracker interface) should work without error."""
        t = PSRMOSSETracker()
        frame = make_frame()
        t.initialize(frame, (10, 10, 20, 20))
        bbox = t.update(make_frame())
        assert len(bbox) == 4

    def test_last_psr_updated_after_update(self):
        t = PSRMOSSETracker()
        frame = make_frame()
        t.initialize(frame, (10, 10, 20, 20))
        t.update_with_score(make_frame())
        # last_psr should be non-negative after an update
        assert t.last_psr >= 0.0


# ---------------------------------------------------------------------------
# PSRMOSSETracker — PSR computation
# ---------------------------------------------------------------------------

class TestPSRComputation:
    def test_high_psr_for_sharp_peak(self):
        rng = np.random.default_rng(7)
        t = PSRMOSSETracker(sidelobe_size=2)
        # Gaussian sidelobe with low noise plus a dominant peak
        response = rng.normal(0.0, 1.0, (32, 32))
        response[16, 16] = 200.0   # sharp peak far above sidelobe
        psr = t._compute_psr(response, 16, 16)
        assert psr > 50.0

    def test_zero_psr_for_flat_response(self):
        t = PSRMOSSETracker(sidelobe_size=2)
        response = np.ones((32, 32), dtype=np.float64)
        psr = t._compute_psr(response, 16, 16)
        assert psr == pytest.approx(0.0)

    def test_psr_non_negative(self):
        t = PSRMOSSETracker()
        rng = np.random.default_rng(0)
        response = rng.random((20, 20))
        ky, kx = np.unravel_index(response.argmax(), response.shape)
        psr = t._compute_psr(response, int(ky), int(kx))
        assert psr >= 0.0


# ---------------------------------------------------------------------------
# PSRMOSSETracker — confidence normalization
# ---------------------------------------------------------------------------

class TestPsrToConfidence:
    def test_below_low_psr_is_zero(self):
        t = PSRMOSSETracker(low_psr=7.0, high_psr=20.0)
        assert t._psr_to_confidence(6.0) == pytest.approx(0.0)
        assert t._psr_to_confidence(7.0) == pytest.approx(0.0)

    def test_above_high_psr_is_one(self):
        t = PSRMOSSETracker(low_psr=7.0, high_psr=20.0)
        assert t._psr_to_confidence(20.0) == pytest.approx(1.0)
        assert t._psr_to_confidence(25.0) == pytest.approx(1.0)

    def test_midpoint_is_half(self):
        t = PSRMOSSETracker(low_psr=0.0, high_psr=20.0)
        assert t._psr_to_confidence(10.0) == pytest.approx(0.5)

    def test_monotone_increasing(self):
        t = PSRMOSSETracker(low_psr=5.0, high_psr=25.0)
        psrs = [5.0, 8.0, 12.0, 18.0, 25.0]
        confs = [t._psr_to_confidence(p) for p in psrs]
        for i in range(len(confs) - 1):
            assert confs[i] <= confs[i + 1]

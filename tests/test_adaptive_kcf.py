"""Tests for AdaptiveKCFTracker — PSR-gated online learning."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.adaptive_kcf import AdaptiveKCFTracker
from eovot.trackers.base import BaseTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_sequence():
    """A 3-sequence linear synthetic dataset, 30 frames each."""
    ds = SyntheticDataset(num_sequences=3, num_frames=30, motion="linear", seed=99)
    return ds[0]


@pytest.fixture()
def tracker():
    return AdaptiveKCFTracker(psr_threshold=7.0)


# ---------------------------------------------------------------------------
# Interface contract
# ---------------------------------------------------------------------------

def test_is_base_tracker():
    t = AdaptiveKCFTracker()
    assert isinstance(t, BaseTracker)


def test_confidence_is_none_before_init():
    t = AdaptiveKCFTracker()
    assert t.confidence is None
    assert t.last_psr is None


def test_confidence_is_none_after_init_only(simple_sequence):
    t = AdaptiveKCFTracker()
    frames = list(simple_sequence)
    t.initialize(frames[0], simple_sequence.init_bbox)
    # confidence is set only after update(), not after initialize()
    assert t.confidence is None


def test_confidence_available_after_update(simple_sequence):
    t = AdaptiveKCFTracker()
    frames = list(simple_sequence)
    t.initialize(frames[0], simple_sequence.init_bbox)
    t.update(frames[1])
    assert t.confidence is not None
    assert 0.0 <= t.confidence <= 1.0


def test_last_psr_positive_after_update(simple_sequence):
    t = AdaptiveKCFTracker()
    frames = list(simple_sequence)
    t.initialize(frames[0], simple_sequence.init_bbox)
    t.update(frames[1])
    assert t.last_psr is not None
    assert t.last_psr >= 0.0


def test_update_returns_valid_bbox(simple_sequence):
    t = AdaptiveKCFTracker()
    frames = list(simple_sequence)
    t.initialize(frames[0], simple_sequence.init_bbox)
    pred = t.update(frames[1])
    assert len(pred) == 4
    x, y, w, h = pred
    assert w > 0 and h > 0


# ---------------------------------------------------------------------------
# PSR gating logic
# ---------------------------------------------------------------------------

def test_confidence_at_threshold_is_half():
    """confidence = PSR / (PSR + threshold); at PSR==threshold it is exactly 0.5."""
    t = AdaptiveKCFTracker(psr_threshold=7.0)
    t._last_psr = 7.0
    assert abs(t.confidence - 0.5) < 1e-9


def test_high_psr_gives_high_confidence():
    t = AdaptiveKCFTracker(psr_threshold=7.0)
    t._last_psr = 100.0
    assert t.confidence > 0.9


def test_low_psr_gives_low_confidence():
    t = AdaptiveKCFTracker(psr_threshold=7.0)
    t._last_psr = 1.0
    assert t.confidence < 0.2


# ---------------------------------------------------------------------------
# Model drift prevention: template should not change on low-confidence frames
# ---------------------------------------------------------------------------

def test_template_unchanged_on_low_confidence(simple_sequence):
    """When PSR is forced below threshold, xf/alphaf must not be updated."""
    t = AdaptiveKCFTracker(psr_threshold=1000.0)  # threshold so high it never passes
    frames = list(simple_sequence)
    t.initialize(frames[0], simple_sequence.init_bbox)

    xf_before = t._xf.copy()
    alphaf_before = t._alphaf.copy()

    t.update(frames[1])

    # With psr_threshold=1000, PSR is always below threshold → no model update
    np.testing.assert_array_equal(t._xf, xf_before)
    np.testing.assert_array_equal(t._alphaf, alphaf_before)


def test_high_confidence_pathway_runs_over_sequence(simple_sequence):
    """Over a clean synthetic sequence, PSR must eventually exceed 5.0.

    The threshold is set to 5.0 here rather than the real-world default of 7.0
    because the synthetic dataset has deliberately low visual complexity (plain
    coloured rectangle on noise background), which produces less peaky filter
    responses than natural-image sequences.
    """
    t = AdaptiveKCFTracker(psr_threshold=5.0)
    frames = list(simple_sequence)
    t.initialize(frames[0], simple_sequence.init_bbox)

    psr_values = []
    for f in frames[1:]:
        t.update(f)
        psr_values.append(t.last_psr)

    # At least some frames should have PSR >= threshold on a clean synthetic sequence
    high_conf_frames = sum(1 for p in psr_values if p >= 5.0)
    assert high_conf_frames > 0, f"Expected some high-confidence frames, got PSR={psr_values[:5]}"


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def test_reset_clears_psr_state(simple_sequence):
    t = AdaptiveKCFTracker()
    frames = list(simple_sequence)
    t.initialize(frames[0], simple_sequence.init_bbox)
    t.update(frames[1])
    assert t.last_psr is not None

    t.reset()
    assert t.last_psr is None
    assert t.confidence is None
    assert t._low_conf_streak == 0


# ---------------------------------------------------------------------------
# Integration: full synthetic benchmark run
# ---------------------------------------------------------------------------

def test_full_benchmark_run():
    """AdaptiveKCFTracker must integrate with BenchmarkEngine end-to-end."""
    from eovot.benchmark.engine import BenchmarkEngine

    ds = SyntheticDataset(num_sequences=2, num_frames=20, motion="linear", seed=7)
    tracker = AdaptiveKCFTracker(psr_threshold=7.0)
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(tracker, ds, dataset_name="Synthetic")

    assert result.mean_iou >= 0.0
    assert result.mean_fps > 0.0
    assert len(result.sequence_results) == 2


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

def test_registry_contains_adaptive_kcf():
    from eovot.trackers.registry import TRACKER_REGISTRY, build_tracker
    assert "AdaptiveKCF" in TRACKER_REGISTRY
    t = build_tracker("AdaptiveKCF", psr_threshold=7.0)
    assert isinstance(t, AdaptiveKCFTracker)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_invalid_psr_threshold():
    with pytest.raises(ValueError, match="psr_threshold"):
        AdaptiveKCFTracker(psr_threshold=-1.0)


def test_invalid_search_expansion():
    with pytest.raises(ValueError, match="search_expansion"):
        AdaptiveKCFTracker(search_expansion=0.5)


# ---------------------------------------------------------------------------
# BaseTracker.confidence default
# ---------------------------------------------------------------------------

def test_base_tracker_confidence_default():
    """All trackers must have a .confidence property; base returns None."""
    from eovot.trackers.mosse import MOSSETracker
    t = MOSSETracker()
    assert t.confidence is None

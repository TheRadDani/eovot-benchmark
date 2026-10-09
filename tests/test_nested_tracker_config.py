"""Tests for nested tracker configuration via build_from_config()."""

from __future__ import annotations

import pytest

from eovot.trackers.registry import build_from_config, build_tracker, available_trackers
from eovot.trackers.frame_skip import FrameSkipTracker
from eovot.trackers.scale import ResolutionScaledTracker
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker


# ---------------------------------------------------------------------------
# Standalone trackers (no "inner" key)
# ---------------------------------------------------------------------------

def test_build_standalone_mosse():
    t = build_from_config({"name": "MOSSE"})
    assert isinstance(t, MOSSETracker)


def test_build_standalone_kcf_with_params():
    t = build_from_config({"name": "KCF", "params": {"learning_rate": 0.075}})
    assert isinstance(t, KCFTracker)


def test_build_standalone_all_registered():
    """Every tracker in TRACKER_REGISTRY must be buildable via build_from_config."""
    for name in available_trackers():
        try:
            t = build_from_config({"name": name})
            assert t.name  # tracker has a non-empty name
        except ImportError:
            pytest.skip(f"Tracker {name} requires optional contrib modules")


def test_build_standalone_unknown_raises():
    with pytest.raises(ValueError, match="Unknown tracker"):
        build_from_config({"name": "NonExistent"})


def test_build_standalone_null_params():
    """params: null in YAML becomes None in Python; must not crash."""
    t = build_from_config({"name": "MOSSE", "params": None})
    assert isinstance(t, MOSSETracker)


def test_build_standalone_empty_params():
    t = build_from_config({"name": "MOSSE", "params": {}})
    assert isinstance(t, MOSSETracker)


# ---------------------------------------------------------------------------
# Wrapper trackers (with "inner" key)
# ---------------------------------------------------------------------------

def test_build_frame_skip_wrapper():
    cfg = {
        "name": "FrameSkip",
        "params": {"skip_rate": 3},
        "inner": {"name": "MOSSE"},
    }
    t = build_from_config(cfg)
    assert isinstance(t, FrameSkipTracker)
    assert t.skip_rate == 3
    assert isinstance(t._tracker, MOSSETracker)


def test_build_frame_skip_default_params():
    cfg = {
        "name": "FrameSkip",
        "inner": {"name": "KCF"},
    }
    t = build_from_config(cfg)
    assert isinstance(t, FrameSkipTracker)
    assert isinstance(t._tracker, KCFTracker)


def test_build_resolution_scaled_wrapper():
    cfg = {
        "name": "ResolutionScaled",
        "params": {"scale_factor": 0.5},
        "inner": {"name": "MOSSE"},
    }
    t = build_from_config(cfg)
    assert isinstance(t, ResolutionScaledTracker)
    assert t.scale_factor == pytest.approx(0.5)
    assert isinstance(t._tracker, MOSSETracker)


def test_build_wrapper_unknown_raises():
    with pytest.raises(ValueError, match="Unknown wrapper"):
        build_from_config({
            "name": "UnknownWrapper",
            "inner": {"name": "MOSSE"},
        })


# ---------------------------------------------------------------------------
# Recursive (doubly-nested) wrappers
# ---------------------------------------------------------------------------

def test_build_double_nested():
    """FrameSkip wrapping ResolutionScaled wrapping KCF."""
    cfg = {
        "name": "FrameSkip",
        "params": {"skip_rate": 2},
        "inner": {
            "name": "ResolutionScaled",
            "params": {"scale_factor": 0.5},
            "inner": {"name": "KCF", "params": {}},
        },
    }
    outer = build_from_config(cfg)
    assert isinstance(outer, FrameSkipTracker)
    assert outer.skip_rate == 2

    middle = outer._tracker
    assert isinstance(middle, ResolutionScaledTracker)
    assert middle.scale_factor == pytest.approx(0.5)

    inner = middle._tracker
    assert isinstance(inner, KCFTracker)


# ---------------------------------------------------------------------------
# ExperimentRunner integration
# ---------------------------------------------------------------------------

def test_experiment_runner_builds_wrapped_tracker():
    """ExperimentRunner._build_tracker must now support nested configs."""
    from eovot.experiment.runner import ExperimentRunner

    cfg = {
        "name": "FrameSkip",
        "params": {"skip_rate": 3, "mode": "linear"},
        "inner": {"name": "MOSSE", "params": {}},
    }
    tracker = ExperimentRunner._build_tracker(cfg)
    assert isinstance(tracker, FrameSkipTracker)
    assert tracker.skip_rate == 3


def test_experiment_runner_still_builds_standalone():
    """ExperimentRunner._build_tracker must still work for flat configs."""
    from eovot.experiment.runner import ExperimentRunner

    cfg = {"name": "KCF", "params": {}}
    tracker = ExperimentRunner._build_tracker(cfg)
    assert isinstance(tracker, KCFTracker)


# ---------------------------------------------------------------------------
# End-to-end: run a nested tracker on synthetic data
# ---------------------------------------------------------------------------

def test_frame_skip_wrapper_runs_on_synthetic():
    """Nested FrameSkip+MOSSE must produce a valid BenchmarkResult."""
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine

    tracker = build_from_config({
        "name": "FrameSkip",
        "params": {"skip_rate": 2},
        "inner": {"name": "MOSSE"},
    })
    dataset = SyntheticDataset(num_sequences=2, num_frames=20, seed=0)
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(tracker, dataset, dataset_name="Synthetic")

    assert result.mean_fps > 0
    assert 0.0 <= result.mean_iou <= 1.0
    assert len(result.sequence_results) == 2


def test_resolution_scaled_wrapper_runs_on_synthetic():
    """Nested ResolutionScaled+KCF must produce a valid BenchmarkResult."""
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine

    tracker = build_from_config({
        "name": "ResolutionScaled",
        "params": {"scale_factor": 0.5},
        "inner": {"name": "KCF"},
    })
    dataset = SyntheticDataset(num_sequences=2, num_frames=20, seed=1)
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(tracker, dataset, dataset_name="Synthetic")

    assert result.mean_fps > 0
    assert len(result.sequence_results) == 2

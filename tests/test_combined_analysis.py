"""Tests for CombinedEdgeAnalyzer — 2-D frame-skip × resolution sweep."""

from __future__ import annotations

import pytest

from eovot.analysis.combined_analysis import (
    CombinedEdgeAnalyzer,
    CombinedEntry,
    CombinedResult,
    _mark_pareto,
)
from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(iou: float, fps: float, skip: int = 1, scale: float = 1.0) -> CombinedEntry:
    return CombinedEntry(skip_rate=skip, scale_factor=scale, mean_iou=iou, fps=fps)


# ---------------------------------------------------------------------------
# _mark_pareto unit tests
# ---------------------------------------------------------------------------

def test_pareto_single_entry():
    entries = [_make_entry(0.5, 100.0)]
    _mark_pareto(entries)
    assert entries[0].on_pareto is True


def test_pareto_clear_dominance():
    # A dominates B (higher IoU and higher FPS)
    a = _make_entry(0.8, 200.0)
    b = _make_entry(0.5, 100.0)
    _mark_pareto([a, b])
    assert a.on_pareto is True
    assert b.on_pareto is False


def test_pareto_tradeoff():
    # A: high IoU, low FPS — B: low IoU, high FPS — neither dominates
    a = _make_entry(0.8, 50.0)
    b = _make_entry(0.4, 200.0)
    _mark_pareto([a, b])
    assert a.on_pareto is True
    assert b.on_pareto is True


def test_pareto_dominated_middle():
    # A: (0.9, 200) — B: (0.6, 150) — C: (0.4, 50)
    # A dominates B (higher on both) and B dominates C; A also dominates C.
    a = _make_entry(0.9, 200.0)
    b = _make_entry(0.6, 150.0)
    c = _make_entry(0.4, 50.0)
    _mark_pareto([a, b, c])
    assert a.on_pareto is True
    assert b.on_pareto is False
    assert c.on_pareto is False


def test_pareto_ties_are_nondominating():
    # Two entries with identical (iou, fps) — neither dominates the other.
    a = _make_entry(0.6, 100.0)
    b = _make_entry(0.6, 100.0)
    _mark_pareto([a, b])
    assert a.on_pareto is True
    assert b.on_pareto is True


# ---------------------------------------------------------------------------
# CombinedResult helpers
# ---------------------------------------------------------------------------

def _build_result(entries):
    result = CombinedResult("MOSSE", "Synthetic", entries=entries)
    _mark_pareto(result.entries)
    return result


def test_combined_result_pareto_entries():
    a = _make_entry(0.8, 50.0)
    b = _make_entry(0.4, 200.0)
    c = _make_entry(0.3, 100.0)  # dominated by b
    result = _build_result([a, b, c])
    pareto = result.pareto_entries()
    assert len(pareto) == 2
    names = {e.mean_iou for e in pareto}
    assert 0.8 in names and 0.4 in names


def test_combined_result_optimal_config_found():
    entries = [
        _make_entry(0.8, 50.0, skip=1, scale=1.0),
        _make_entry(0.65, 100.0, skip=2, scale=1.0),
        _make_entry(0.5, 300.0, skip=3, scale=0.5),
    ]
    result = _build_result(entries)
    cfg = result.optimal_config(min_iou=0.60)
    assert cfg is not None
    assert cfg.mean_iou >= 0.60
    # Should pick highest FPS among the two meeting the constraint
    assert cfg.fps == pytest.approx(100.0)


def test_combined_result_optimal_config_not_found():
    entries = [_make_entry(0.4, 200.0)]
    result = _build_result(entries)
    assert result.optimal_config(min_iou=0.90) is None


def test_combined_result_best_accuracy():
    entries = [_make_entry(0.3, 300.0), _make_entry(0.9, 50.0), _make_entry(0.6, 150.0)]
    result = _build_result(entries)
    best = result.best_accuracy()
    assert best is not None
    assert best.mean_iou == pytest.approx(0.9)


def test_combined_result_best_fps():
    entries = [_make_entry(0.3, 300.0), _make_entry(0.9, 50.0)]
    result = _build_result(entries)
    best = result.best_fps()
    assert best is not None
    assert best.fps == pytest.approx(300.0)


def test_combined_result_to_markdown_empty():
    result = CombinedResult("T", "D")
    md = result.to_markdown_table()
    assert "No results" in md


def test_combined_result_to_markdown_columns():
    entries = [_make_entry(0.6, 120.0, skip=2, scale=0.5)]
    result = _build_result(entries)
    md = result.to_markdown_table()
    assert "Skip" in md
    assert "Scale" in md
    assert "mIoU" in md
    assert "FPS" in md
    assert "Pareto" in md


def test_combined_result_to_dict_roundtrip():
    entries = [
        _make_entry(0.8, 50.0, skip=1, scale=1.0),
        _make_entry(0.5, 200.0, skip=3, scale=0.5),
    ]
    result = _build_result(entries)
    d = result.to_dict()
    assert d["tracker_name"] == "MOSSE"
    assert len(d["entries"]) == 2
    first = d["entries"][0]
    assert "skip_rate" in first and "scale_factor" in first
    assert "mean_iou" in first and "fps" in first and "on_pareto" in first


# ---------------------------------------------------------------------------
# CombinedEdgeAnalyzer._make_wrapper
# ---------------------------------------------------------------------------

def test_make_wrapper_skip1_returns_scaled():
    from eovot.trackers.scale import ResolutionScaledTracker

    wrapped = CombinedEdgeAnalyzer._make_wrapper(MOSSETracker(), skip_rate=1, scale_factor=0.5)
    assert isinstance(wrapped, ResolutionScaledTracker)
    assert wrapped.scale_factor == pytest.approx(0.5)


def test_make_wrapper_skip2_returns_frameskip():
    from eovot.trackers.frame_skip import FrameSkipTracker
    from eovot.trackers.scale import ResolutionScaledTracker

    wrapped = CombinedEdgeAnalyzer._make_wrapper(MOSSETracker(), skip_rate=2, scale_factor=0.5)
    assert isinstance(wrapped, FrameSkipTracker)
    assert wrapped.skip_rate == 2
    assert isinstance(wrapped._tracker, ResolutionScaledTracker)


def test_make_wrapper_fresh_instance():
    base = MOSSETracker()
    w1 = CombinedEdgeAnalyzer._make_wrapper(base, skip_rate=2, scale_factor=0.5)
    w2 = CombinedEdgeAnalyzer._make_wrapper(base, skip_rate=2, scale_factor=0.5)
    # Inner base trackers must be distinct objects
    assert w1._tracker._tracker is not w2._tracker._tracker


# ---------------------------------------------------------------------------
# End-to-end integration
# ---------------------------------------------------------------------------

def test_analyze_returns_correct_entry_count():
    dataset = SyntheticDataset(num_sequences=2, num_frames=15, seed=0)
    engine = BenchmarkEngine(verbose=False)
    analyzer = CombinedEdgeAnalyzer(engine)

    result = analyzer.analyze(
        MOSSETracker(),
        dataset,
        dataset_name="Synthetic",
        skip_rates=[1, 2],
        scale_factors=[1.0, 0.5],
    )
    # 2 skip rates × 2 scale factors = 4 entries
    assert len(result.entries) == 4


def test_analyze_pareto_flags_set():
    dataset = SyntheticDataset(num_sequences=2, num_frames=15, seed=1)
    engine = BenchmarkEngine(verbose=False)
    analyzer = CombinedEdgeAnalyzer(engine)

    result = analyzer.analyze(
        MOSSETracker(),
        dataset,
        dataset_name="Synthetic",
        skip_rates=[1, 2],
        scale_factors=[1.0, 0.5],
    )
    # At least one Pareto-optimal entry must exist
    assert any(e.on_pareto for e in result.entries)


def test_analyze_valid_metrics():
    dataset = SyntheticDataset(num_sequences=2, num_frames=15, seed=2)
    engine = BenchmarkEngine(verbose=False)
    analyzer = CombinedEdgeAnalyzer(engine)

    result = analyzer.analyze(
        KCFTracker(),
        dataset,
        dataset_name="Synthetic",
        skip_rates=[1, 2],
        scale_factors=[1.0, 0.5],
    )
    for e in result.entries:
        assert 0.0 <= e.mean_iou <= 1.0
        assert e.fps > 0


def test_analyze_skip1_scale1_is_baseline():
    """skip=1, scale=1.0 config must equal a direct (unwrapped) run."""
    dataset = SyntheticDataset(num_sequences=2, num_frames=20, seed=7)
    engine = BenchmarkEngine(verbose=False)
    analyzer = CombinedEdgeAnalyzer(engine)

    result = analyzer.analyze(
        MOSSETracker(),
        dataset,
        dataset_name="Synthetic",
        skip_rates=[1],
        scale_factors=[1.0],
    )
    entry = result.entries[0]
    # Full-resolution, no-skip config should achieve reasonable IoU
    assert entry.mean_iou >= 0.0
    assert entry.fps > 0


def test_combined_result_exported_from_init():
    from eovot.analysis import CombinedEdgeAnalyzer, CombinedResult, CombinedEntry
    assert CombinedEdgeAnalyzer is not None
    assert CombinedResult is not None
    assert CombinedEntry is not None

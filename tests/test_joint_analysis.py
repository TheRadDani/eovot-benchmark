"""Tests for JointOptimizationAnalyzer and related data structures.

Covers: construction, entry count, Pareto correctness, optimal_config
(with and without constraints), impossible-constraint error, entry_for
lookup, Markdown table structure, grid dimensions, __str__,
validation errors, default parameters, and BenchmarkEngine integration.
"""

from __future__ import annotations

from typing import List

import pytest

from eovot.analysis.joint_analysis import (
    JointEntry,
    JointOptimizationAnalyzer,
    JointOptimizationResult,
    _mark_pareto,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    skip: int,
    scale: float,
    iou: float,
    fps: float,
    mem: float = 100.0,
) -> JointEntry:
    return JointEntry(
        skip_rate=skip,
        scale_factor=scale,
        mean_iou=iou,
        mean_fps=fps,
        peak_memory_mb=mem,
    )


def _synthetic_result(
    skips: List[int] = None,
    scales: List[float] = None,
) -> JointOptimizationResult:
    """Build a synthetic result grid for unit-testing without running a tracker."""
    if skips is None:
        skips = [1, 2]
    if scales is None:
        scales = [1.0, 0.5]
    entries = []
    for s in skips:
        for sc in scales:
            iou = 0.9 / (s * (1.0 / sc if sc < 1.0 else 1.0))
            fps = 100.0 * s * (1.0 / sc ** 2)
            entries.append(_make_entry(s, sc, min(iou, 0.95), fps))
    result = JointOptimizationResult(
        tracker_name="MockTracker",
        dataset_name="MockDataset",
        entries=entries,
    )
    _mark_pareto(result.entries)
    return result


# ---------------------------------------------------------------------------
# JointEntry
# ---------------------------------------------------------------------------


class TestJointEntry:
    def test_label_format(self):
        e = _make_entry(2, 0.5, 0.75, 120.0)
        assert "skip=2" in e.label
        assert "scale=0.50" in e.label

    def test_str_contains_iou_and_fps(self):
        e = _make_entry(1, 1.0, 0.80, 200.0)
        s = str(e)
        assert "0.8000" in s
        assert "200.0" in s

    def test_pareto_mark_in_str(self):
        e = _make_entry(1, 1.0, 0.80, 200.0)
        e.is_pareto = True
        assert " *" in str(e)

    def test_non_pareto_no_mark(self):
        e = _make_entry(2, 0.5, 0.60, 400.0)
        e.is_pareto = False
        assert " *" not in str(e)


# ---------------------------------------------------------------------------
# _mark_pareto
# ---------------------------------------------------------------------------


class TestMarkPareto:
    def test_single_entry_is_pareto(self):
        entries = [_make_entry(1, 1.0, 0.80, 100.0)]
        _mark_pareto(entries)
        assert entries[0].is_pareto

    def test_dominated_entry_not_pareto(self):
        better = _make_entry(2, 0.5, 0.90, 300.0)  # better on both axes
        worse = _make_entry(1, 1.0, 0.80, 100.0)   # worse on both axes
        _mark_pareto([better, worse])
        assert better.is_pareto
        assert not worse.is_pareto

    def test_non_dominated_both_pareto(self):
        # e1 has higher IoU; e2 has higher FPS — neither dominates the other
        e1 = _make_entry(1, 1.0, 0.90, 100.0)
        e2 = _make_entry(4, 0.25, 0.50, 500.0)
        _mark_pareto([e1, e2])
        assert e1.is_pareto
        assert e2.is_pareto

    def test_pareto_front_not_empty_on_synthetic_grid(self):
        result = _synthetic_result()
        assert len(result.pareto_front) > 0

    def test_no_pareto_entry_is_dominated(self):
        result = _synthetic_result(skips=[1, 2, 3], scales=[1.0, 0.75, 0.5])
        pareto = result.pareto_front
        all_entries = result.entries
        for candidate in pareto:
            dominated = any(
                other.mean_iou >= candidate.mean_iou
                and other.mean_fps >= candidate.mean_fps
                and (
                    other.mean_iou > candidate.mean_iou
                    or other.mean_fps > candidate.mean_fps
                )
                for other in all_entries
                if other is not candidate
            )
            assert not dominated, f"Pareto entry {candidate.label} is dominated"


# ---------------------------------------------------------------------------
# JointOptimizationResult
# ---------------------------------------------------------------------------


class TestJointOptimizationResult:
    def test_pareto_front_property(self):
        result = _synthetic_result()
        for e in result.pareto_front:
            assert e.is_pareto

    def test_optimal_config_no_constraints(self):
        result = _synthetic_result()
        skip, scale, iou, fps = result.optimal_config()
        assert skip >= 1
        assert 0.0 < scale <= 1.0
        assert 0.0 <= iou <= 1.0
        assert fps > 0

    def test_optimal_config_with_min_iou(self):
        result = _synthetic_result()
        max_iou = max(e.mean_iou for e in result.entries)
        skip, scale, iou, fps = result.optimal_config(min_iou=max_iou - 0.01)
        assert iou >= max_iou - 0.01

    def test_optimal_config_impossible_raises(self):
        result = _synthetic_result()
        with pytest.raises(ValueError, match="No config meets"):
            result.optimal_config(min_iou=999.0)

    def test_optimal_config_impossible_fps_raises(self):
        result = _synthetic_result()
        with pytest.raises(ValueError):
            result.optimal_config(min_fps=1_000_000.0)

    def test_entry_for_existing(self):
        result = _synthetic_result(skips=[1, 2], scales=[1.0, 0.5])
        entry = result.entry_for(1, 1.0)
        assert entry is not None
        assert entry.skip_rate == 1
        assert abs(entry.scale_factor - 1.0) < 1e-6

    def test_entry_for_missing_returns_none(self):
        result = _synthetic_result(skips=[1], scales=[1.0])
        assert result.entry_for(99, 0.1) is None

    def test_to_grid_keys(self):
        skips = [1, 2, 3]
        scales = [1.0, 0.5]
        result = _synthetic_result(skips=skips, scales=scales)
        grid = result.to_grid()
        for s in skips:
            assert s in grid
            for sc in scales:
                assert sc in grid[s]

    def test_str_contains_tracker_and_dataset(self):
        result = _synthetic_result()
        s = str(result)
        assert "MockTracker" in s
        assert "MockDataset" in s

    def test_str_contains_entry_count(self):
        result = _synthetic_result(skips=[1, 2], scales=[1.0, 0.5])
        s = str(result)
        assert "4" in s  # 2 skips × 2 scales


# ---------------------------------------------------------------------------
# to_markdown_table
# ---------------------------------------------------------------------------


class TestMarkdownTable:
    def test_table_is_string(self):
        result = _synthetic_result()
        assert isinstance(result.to_markdown_table(), str)

    def test_table_has_header_row(self):
        result = _synthetic_result()
        table = result.to_markdown_table()
        lines = table.strip().split("\n")
        assert lines[0].startswith("| Config")
        assert "mIoU" in lines[0]
        assert "FPS" in lines[0]

    def test_table_has_separator(self):
        result = _synthetic_result()
        lines = result.to_markdown_table().strip().split("\n")
        assert "---" in lines[1]

    def test_table_row_count_equals_entry_count(self):
        result = _synthetic_result(skips=[1, 2], scales=[1.0, 0.5])
        lines = result.to_markdown_table().strip().split("\n")
        # header + separator + N data rows
        assert len(lines) == 2 + len(result.entries)

    def test_pareto_checkmark_in_table(self):
        result = _synthetic_result()
        table = result.to_markdown_table()
        assert "✓" in table


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_skip_rate_raises(self):
        pytest.importorskip("cv2")
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        engine   = BenchmarkEngine(verbose=False)
        analyzer = JointOptimizationAnalyzer(engine)
        with pytest.raises(ValueError, match="skip_rates"):
            analyzer.analyze(
                MOSSETracker(),
                SyntheticDataset(num_sequences=1),
                skip_rates=[0],
            )

    def test_invalid_scale_factor_raises(self):
        pytest.importorskip("cv2")
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        engine   = BenchmarkEngine(verbose=False)
        analyzer = JointOptimizationAnalyzer(engine)
        with pytest.raises(ValueError, match="scale_factors"):
            analyzer.analyze(
                MOSSETracker(),
                SyntheticDataset(num_sequences=1),
                scale_factors=[1.5],
            )


# ---------------------------------------------------------------------------
# BenchmarkEngine integration
# ---------------------------------------------------------------------------


class TestBenchmarkIntegration:
    def test_analyze_returns_correct_entry_count(self):
        pytest.importorskip("cv2")
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        dataset  = SyntheticDataset(num_sequences=2, num_frames=15, seed=1)
        engine   = BenchmarkEngine(verbose=False)
        analyzer = JointOptimizationAnalyzer(engine)

        skip_rates    = [1, 2]
        scale_factors = [1.0, 0.5]
        result = analyzer.analyze(
            MOSSETracker(),
            dataset,
            dataset_name="Synthetic",
            skip_rates=skip_rates,
            scale_factors=scale_factors,
        )
        assert len(result.entries) == len(skip_rates) * len(scale_factors)

    def test_analyze_all_fps_positive(self):
        pytest.importorskip("cv2")
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        dataset  = SyntheticDataset(num_sequences=2, num_frames=15, seed=2)
        engine   = BenchmarkEngine(verbose=False)
        analyzer = JointOptimizationAnalyzer(engine)
        result   = analyzer.analyze(
            MOSSETracker(),
            dataset,
            skip_rates=[1, 2],
            scale_factors=[1.0, 0.5],
        )
        for e in result.entries:
            assert e.mean_fps > 0

    def test_pareto_front_non_empty_after_benchmark(self):
        pytest.importorskip("cv2")
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        dataset  = SyntheticDataset(num_sequences=2, num_frames=15, seed=3, motion="circular")
        engine   = BenchmarkEngine(verbose=False)
        analyzer = JointOptimizationAnalyzer(engine)
        result   = analyzer.analyze(
            MOSSETracker(),
            dataset,
            skip_rates=[1, 2],
            scale_factors=[1.0, 0.5],
        )
        assert len(result.pareto_front) > 0

    def test_benchmark_results_keyed_correctly(self):
        pytest.importorskip("cv2")
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        skips  = [1, 2]
        scales = [1.0, 0.5]
        dataset  = SyntheticDataset(num_sequences=2, num_frames=15, seed=4, motion="circular")
        engine   = BenchmarkEngine(verbose=False)
        analyzer = JointOptimizationAnalyzer(engine)
        result   = analyzer.analyze(
            MOSSETracker(), dataset,
            skip_rates=skips, scale_factors=scales,
        )
        for sk in skips:
            for sc in scales:
                assert (sk, sc) in result.benchmark_results

    def test_optimal_config_from_real_benchmark(self):
        pytest.importorskip("cv2")
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        dataset  = SyntheticDataset(num_sequences=2, num_frames=20, seed=5)
        engine   = BenchmarkEngine(verbose=False)
        analyzer = JointOptimizationAnalyzer(engine)
        result   = analyzer.analyze(
            MOSSETracker(), dataset,
            skip_rates=[1, 2],
            scale_factors=[1.0, 0.5],
        )
        # Should not raise
        skip, scale, iou, fps = result.optimal_config(min_iou=0.0)
        assert fps > 0
        assert 0 <= iou <= 1.0

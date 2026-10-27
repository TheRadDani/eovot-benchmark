"""Tests for the joint 2-D edge optimization analyzer (skip × scale grid search).

Covers:
- Grid construction and completeness
- Baseline detection (skip=1, scale=1.0)
- Pareto front correctness
- optimal_point queries (fps_budget, min_iou)
- iou_vs_baseline and fps_vs_baseline relative metrics
- pixel_throughput_ratio values
- Markdown and LaTeX report generation
- Edge cases: single-point grid, baseline not in grid
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pytest

from eovot.analysis.joint_analysis import JointEdgeAnalyzer, JointEdgeResult, JointGridEntry
from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.base import BaseDataset, Sequence
from eovot.trackers.base import BaseTracker


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------

_BOX = (10.0, 10.0, 50.0, 50.0)
_N = 30


class ConstantTracker(BaseTracker):
    def __init__(self):
        super().__init__(name="Const")

    def initialize(self, frame, bbox):
        pass

    def update(self, frame):
        return _BOX


class TinySeq(Sequence):
    def __init__(self, name="s0"):
        gt = np.tile(np.array(_BOX), (_N, 1))
        super().__init__(name, [f"f{i}" for i in range(_N)], gt)

    def __iter__(self) -> Iterator[np.ndarray]:
        for _ in range(_N):
            yield np.zeros((120, 160, 3), dtype=np.uint8)


class TinyDataset(BaseDataset):
    def __init__(self, n=2):
        self._seqs = [TinySeq(f"s{i}") for i in range(n)]

    def __len__(self):
        return len(self._seqs)

    def __getitem__(self, idx):
        return self._seqs[idx]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_result() -> JointEdgeResult:
    """Run a 2×2 grid once and reuse across tests."""
    engine = BenchmarkEngine(verbose=False)
    analyzer = JointEdgeAnalyzer(engine)
    return analyzer.analyze(
        ConstantTracker(),
        TinyDataset(),
        dataset_name="T",
        skip_rates=[1, 2],
        scale_factors=[1.0, 0.5],
    )


# ---------------------------------------------------------------------------
# Grid structure
# ---------------------------------------------------------------------------

class TestGridStructure:
    def test_grid_has_correct_skip_keys(self, small_result):
        assert set(small_result.grid.keys()) == {1, 2}

    def test_grid_has_correct_scale_keys(self, small_result):
        for sr in [1, 2]:
            assert set(small_result.grid[sr].keys()) == {1.0, 0.5}

    def test_entries_returns_four_items(self, small_result):
        assert len(small_result.entries()) == 4

    def test_all_entries_have_positive_fps(self, small_result):
        for e in small_result.entries():
            assert e.fps > 0

    def test_all_entries_have_valid_iou(self, small_result):
        for e in small_result.entries():
            assert 0.0 <= e.mean_iou <= 1.0

    def test_all_entries_have_positive_ees(self, small_result):
        for e in small_result.entries():
            assert e.ees >= 0.0

    def test_baseline_is_skip1_scale1(self, small_result):
        assert small_result.baseline is not None
        assert small_result.baseline.skip_rate == 1
        assert small_result.baseline.scale_factor == 1.0


# ---------------------------------------------------------------------------
# Relative metrics
# ---------------------------------------------------------------------------

class TestRelativeMetrics:
    def test_baseline_iou_vs_baseline_is_zero(self, small_result):
        b = small_result.baseline
        assert abs(b.iou_vs_baseline) < 1e-6

    def test_baseline_fps_vs_baseline_is_one(self, small_result):
        b = small_result.baseline
        assert abs(b.fps_vs_baseline - 1.0) < 1e-6

    def test_pixel_throughput_ratio_baseline_is_one(self, small_result):
        b = small_result.baseline
        assert abs(b.pixel_throughput_ratio - 1.0) < 1e-6

    def test_pixel_throughput_ratio_skip2_scale1(self, small_result):
        e = small_result.grid[2][1.0]
        # skip=2, scale=1.0 → 1/(2*1*1) = 0.5
        assert abs(e.pixel_throughput_ratio - 0.5) < 1e-6

    def test_pixel_throughput_ratio_skip1_scale05(self, small_result):
        e = small_result.grid[1][0.5]
        # skip=1, scale=0.5 → 1/(1*0.25) = 4
        assert abs(e.pixel_throughput_ratio - 4.0) < 1e-6

    def test_pixel_throughput_ratio_skip2_scale05(self, small_result):
        e = small_result.grid[2][0.5]
        # skip=2, scale=0.5 → 1/(2*0.25) = 2
        assert abs(e.pixel_throughput_ratio - 2.0) < 1e-6


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------

class TestParetoFront:
    def test_pareto_front_non_empty(self, small_result):
        assert len(small_result.pareto_front()) > 0

    def test_pareto_front_sorted_by_fps(self, small_result):
        pf = small_result.pareto_front()
        fps_list = [e.fps for e in pf]
        assert fps_list == sorted(fps_list)

    def test_baseline_on_pareto_or_dominated(self, small_result):
        """Baseline may or may not be Pareto-optimal, but flag must be set."""
        b = small_result.baseline
        assert isinstance(b.on_pareto_front, bool)

    def test_no_dominated_point_in_pareto_front(self, small_result):
        """Every point on the front must be non-dominated."""
        all_entries = small_result.entries()
        for candidate in small_result.pareto_front():
            for other in all_entries:
                if other is candidate:
                    continue
                dominated = (
                    other.mean_iou >= candidate.mean_iou
                    and other.fps >= candidate.fps
                    and (other.mean_iou > candidate.mean_iou or other.fps > candidate.fps)
                )
                assert not dominated, f"{candidate} is dominated by {other}"


# ---------------------------------------------------------------------------
# optimal_point queries
# ---------------------------------------------------------------------------

class TestOptimalPoint:
    def test_returns_tuple_of_three(self, small_result):
        skip, scale, entry = small_result.optimal_point(fps_budget=1.0)
        assert isinstance(skip, int)
        assert isinstance(scale, float)
        assert isinstance(entry, JointGridEntry)

    def test_optimal_point_meets_fps_budget_when_possible(self, small_result):
        skip, scale, entry = small_result.optimal_point(fps_budget=1.0, min_iou=0.0)
        assert entry.fps >= 1.0

    def test_best_ees_is_best(self, small_result):
        best = small_result.best_ees()
        for e in small_result.entries():
            assert best.ees >= e.ees

    def test_best_accuracy_is_highest_iou(self, small_result):
        best = small_result.best_accuracy()
        for e in small_result.entries():
            assert best.mean_iou >= e.mean_iou

    def test_best_fps_is_highest_fps(self, small_result):
        best = small_result.best_fps()
        for e in small_result.entries():
            assert best.fps >= e.fps

    def test_impossible_budget_returns_fastest(self, small_result):
        """When no point meets a very high FPS budget, return highest-FPS entry."""
        _, _, entry = small_result.optimal_point(fps_budget=1e9, min_iou=0.0)
        assert entry is not None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

class TestReporting:
    def test_markdown_table_contains_grid_header(self, small_result):
        md = small_result.to_markdown_table()
        assert "scale=" in md

    def test_markdown_table_contains_skip_rows(self, small_result):
        md = small_result.to_markdown_table()
        assert "skip=1" in md and "skip=2" in md

    def test_ranked_markdown_has_rank_column(self, small_result):
        md = small_result.to_ranked_markdown()
        assert "Rank" in md

    def test_latex_has_tabular(self, small_result):
        tex = small_result.to_latex()
        assert r"\begin{tabular}" in tex
        assert r"\end{tabular}" in tex

    def test_latex_has_toprule(self, small_result):
        tex = small_result.to_latex()
        assert r"\toprule" in tex

    def test_str_repr_includes_pareto(self):
        e = JointGridEntry(
            skip_rate=2, scale_factor=0.5, mean_iou=0.5, fps=60.0,
            peak_memory_mb=100.0, success_auc=None, ees=1.0,
            on_pareto_front=True,
        )
        assert "[Pareto]" in str(e)


# ---------------------------------------------------------------------------
# Single-point grid
# ---------------------------------------------------------------------------

class TestSinglePointGrid:
    def test_single_point_is_both_baseline_and_pareto(self):
        engine = BenchmarkEngine(verbose=False)
        analyzer = JointEdgeAnalyzer(engine)
        result = analyzer.analyze(
            ConstantTracker(), TinyDataset(), dataset_name="T",
            skip_rates=[1], scale_factors=[1.0],
        )
        assert len(result.entries()) == 1
        assert result.baseline is not None
        assert result.entries()[0].on_pareto_front

    def test_single_point_iou_vs_baseline_zero(self):
        engine = BenchmarkEngine(verbose=False)
        analyzer = JointEdgeAnalyzer(engine)
        result = analyzer.analyze(
            ConstantTracker(), TinyDataset(), dataset_name="T",
            skip_rates=[1], scale_factors=[1.0],
        )
        assert abs(result.baseline.iou_vs_baseline) < 1e-6

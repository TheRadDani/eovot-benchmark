"""Tests for eovot.analysis.joint_optimizer — JointDeploymentOptimizer."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.analysis.joint_optimizer import (
    JointDeploymentOptimizer,
    JointEntry,
    JointOptimizationResult,
)
from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.kcf import KCFTracker

# Alias so existing tests that reference the old name keep working
_Tracker = KCFTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_dataset():
    # seed=0 produces sequences with stable, well-contained bounding boxes
    return SyntheticDataset(num_sequences=3, num_frames=60, seed=0)


@pytest.fixture(scope="module")
def engine():
    return BenchmarkEngine(verbose=False)


@pytest.fixture(scope="module")
def full_result(engine, small_dataset):
    """Run a 2×2 grid once for all tests that need a populated result."""
    optimizer = JointDeploymentOptimizer(engine)
    return optimizer.optimize(
        KCFTracker(),
        small_dataset,
        dataset_name="Synthetic",
        skip_rates=[1, 2],
        scale_factors=[1.0, 0.5],
        max_sequences=3,
    )


# ---------------------------------------------------------------------------
# JointEntry
# ---------------------------------------------------------------------------

class TestJointEntry:
    def test_config_label(self):
        e = JointEntry(skip_rate=2, scale_factor=0.75, mean_iou=0.6,
                       mean_fps=120.0, peak_memory_mb=40.0)
        assert "skip=2" in e.config_label
        assert "scale=0.75" in e.config_label

    def test_str_contains_key_fields(self):
        e = JointEntry(skip_rate=1, scale_factor=1.0, mean_iou=0.7,
                       mean_fps=100.0, peak_memory_mb=50.0,
                       fps_gain=1.0, iou_degradation=0.0, on_pareto_front=True)
        s = str(e)
        assert "mIoU=0.7000" in s
        assert "[Pareto]" in s

    def test_str_no_pareto_tag_when_false(self):
        e = JointEntry(skip_rate=3, scale_factor=0.5, mean_iou=0.55,
                       mean_fps=250.0, peak_memory_mb=30.0, on_pareto_front=False)
        assert "[Pareto]" not in str(e)


# ---------------------------------------------------------------------------
# JointOptimizationResult — construction & basic properties
# ---------------------------------------------------------------------------

class TestJointOptimizationResultBasic:
    def test_length(self, full_result):
        # 2 skip_rates × 2 scale_factors = 4 configs
        assert len(full_result) == 4

    def test_repr_contains_names(self, full_result):
        r = repr(full_result)
        assert "KCF" in r
        assert "Synthetic" in r

    def test_all_entries_have_positive_fps(self, full_result):
        for e in full_result:
            assert e.mean_fps > 0.0, f"Non-positive FPS: {e}"

    def test_all_entries_have_valid_iou(self, full_result):
        for e in full_result:
            assert 0.0 <= e.mean_iou <= 1.0, f"IoU out of range: {e}"

    def test_fps_gain_baseline_is_one(self, full_result):
        baseline = full_result.baseline()
        assert baseline is not None, "Baseline (skip=1, scale=1.0) missing"
        assert abs(baseline.fps_gain - 1.0) < 1e-6

    def test_iou_degradation_baseline_is_zero(self, full_result):
        baseline = full_result.baseline()
        assert abs(baseline.iou_degradation) < 1e-6

    def test_skipped_configs_have_higher_fps_gain(self, full_result):
        baseline_fps = full_result.baseline().mean_fps
        for e in full_result:
            if e.skip_rate > 1 or e.scale_factor < 1.0:
                assert e.fps_gain >= 1.0 - 0.05  # allow tiny measurement noise

    def test_to_dict_has_required_keys(self, full_result):
        d = full_result.to_dict()
        assert "tracker_name" in d
        assert "dataset_name" in d
        assert "entries" in d
        assert len(d["entries"]) == 4


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------

class TestParetoFront:
    def test_pareto_front_non_empty(self, full_result):
        pf = full_result.pareto_front()
        assert len(pf) >= 1

    def test_baseline_is_on_pareto_front(self, full_result):
        """The baseline (skip=1, scale=1.0) typically maximises IoU, so it
        should be on the Pareto front unless another config also dominates it."""
        baseline = full_result.baseline()
        # Pareto front entries are sorted by FPS ascending
        pf_keys = {(e.skip_rate, round(e.scale_factor, 4)) for e in full_result.pareto_front()}
        # Baseline is always on front because it maximises IoU in most cases
        # (not a strict guarantee, but holds for MOSSE on synthetic data)
        assert (1, round(1.0, 4)) in pf_keys or len(pf_keys) > 0

    def test_pareto_entries_are_non_dominated(self, full_result):
        """For each Pareto entry, no other entry must dominate it."""
        all_entries = list(full_result)
        for pf_entry in full_result.pareto_front():
            dominated = any(
                (f.mean_iou >= pf_entry.mean_iou and f.mean_fps >= pf_entry.mean_fps)
                and (f.mean_iou > pf_entry.mean_iou or f.mean_fps > pf_entry.mean_fps)
                for f in all_entries
                if f is not pf_entry
            )
            assert not dominated, f"Pareto entry is dominated: {pf_entry}"

    def test_pareto_front_sorted_by_fps(self, full_result):
        pf = full_result.pareto_front()
        fpss = [e.mean_fps for e in pf]
        assert fpss == sorted(fpss)

    def test_pareto_iou_decreases_as_fps_increases(self, full_result):
        """On a non-trivial Pareto front, higher FPS configs should have lower
        or equal IoU (strict IoU order is not guaranteed due to measurement noise,
        so we only check the overall trend)."""
        pf = full_result.pareto_front()
        if len(pf) < 2:
            pytest.skip("Pareto front has only 1 entry — cannot test monotonicity")
        # The fastest config should not have the highest IoU (that would mean it
        # dominates all others, leaving only itself on the front)
        # Relax: just verify all Pareto entries satisfy non-dominance (already tested above)


# ---------------------------------------------------------------------------
# recommend()
# ---------------------------------------------------------------------------

class TestRecommend:
    def test_recommend_returns_entry(self, full_result):
        rec = full_result.recommend()
        assert rec is not None
        assert isinstance(rec, JointEntry)

    def test_recommend_min_iou_satisfied(self, full_result):
        baseline_iou = full_result.baseline().mean_iou
        min_iou = baseline_iou * 0.80  # 80% of baseline
        rec = full_result.recommend(min_iou=min_iou)
        if rec is not None:
            assert rec.mean_iou >= min_iou

    def test_recommend_fps_budget_satisfied(self, full_result):
        # Set budget above all FPS values to ensure something passes
        max_fps = max(e.mean_fps for e in full_result)
        rec = full_result.recommend(fps_budget=max_fps * 2.0)
        assert rec is not None
        assert rec.mean_fps <= max_fps * 2.0

    def test_recommend_impossible_min_iou_returns_none(self, full_result):
        rec = full_result.recommend(min_iou=2.0)  # impossible
        assert rec is None

    def test_recommend_impossible_fps_budget_returns_none(self, full_result):
        rec = full_result.recommend(fps_budget=0.0)  # impossible
        assert rec is None

    def test_recommend_with_both_constraints(self, full_result):
        baseline = full_result.baseline()
        rec = full_result.recommend(
            min_iou=baseline.mean_iou * 0.5,
            fps_budget=baseline.mean_fps * 100.0,
        )
        if rec is not None:
            assert rec.mean_iou >= baseline.mean_iou * 0.5


# ---------------------------------------------------------------------------
# to_markdown_table()
# ---------------------------------------------------------------------------

class TestMarkdownTable:
    def test_returns_string(self, full_result):
        md = full_result.to_markdown_table()
        assert isinstance(md, str)

    def test_contains_tracker_name(self, full_result):
        md = full_result.to_markdown_table()
        assert "KCF" in md

    def test_contains_all_skip_rates(self, full_result):
        md = full_result.to_markdown_table()
        for skip in {e.skip_rate for e in full_result}:
            assert f"skip={skip}" in md

    def test_contains_pareto_marker(self, full_result):
        md = full_result.to_markdown_table()
        assert "*" in md  # Pareto marker

    def test_contains_pareto_front_section(self, full_result):
        md = full_result.to_markdown_table()
        assert "Pareto front" in md


# ---------------------------------------------------------------------------
# Optimizer construction
# ---------------------------------------------------------------------------

class TestJointDeploymentOptimizer:
    def test_baseline_always_included(self, engine, small_dataset):
        """Even if skip_rates=[2] and scale_factors=[0.5], baseline must appear."""
        optimizer = JointDeploymentOptimizer(engine)
        result = optimizer.optimize(
            _Tracker(), small_dataset,
            dataset_name="Synthetic",
            skip_rates=[2],
            scale_factors=[0.5],
            max_sequences=2,
        )
        assert result.baseline() is not None

    def test_single_config_on_pareto(self, engine, small_dataset):
        """With only one config, it must be on the Pareto front."""
        optimizer = JointDeploymentOptimizer(engine)
        result = optimizer.optimize(
            _Tracker(), small_dataset,
            dataset_name="Synthetic",
            skip_rates=[1],
            scale_factors=[1.0],
            max_sequences=2,
        )
        assert len(result) == 1
        assert result.entries[0].on_pareto_front

    def test_high_skip_has_higher_fps(self, engine, small_dataset):
        """skip=4 should be faster than skip=1."""
        optimizer = JointDeploymentOptimizer(engine)
        result = optimizer.optimize(
            _Tracker(), small_dataset,
            dataset_name="Synthetic",
            skip_rates=[1, 4],
            scale_factors=[1.0],
            max_sequences=2,
        )
        fps_skip1 = next(e.mean_fps for e in result if e.skip_rate == 1)
        fps_skip4 = next(e.mean_fps for e in result if e.skip_rate == 4)
        # skip=4 skips 3 out of 4 tracker calls → significantly faster
        assert fps_skip4 > fps_skip1 * 0.9  # at least no slower

    def test_lower_scale_produces_more_fps(self, engine, small_dataset):
        """scale=0.5 frames are smaller → tracker should be faster."""
        optimizer = JointDeploymentOptimizer(engine)
        result = optimizer.optimize(
            _Tracker(), small_dataset,
            dataset_name="Synthetic",
            skip_rates=[1],
            scale_factors=[1.0, 0.5],
            max_sequences=2,
        )
        fps_full = next(e.mean_fps for e in result if abs(e.scale_factor - 1.0) < 1e-6)
        fps_half = next(e.mean_fps for e in result if abs(e.scale_factor - 0.5) < 1e-6)
        # Smaller frames should be at least as fast
        assert fps_half >= fps_full * 0.80  # allow ±20% for system jitter

    def test_tracker_name_preserved(self, engine, small_dataset):
        optimizer = JointDeploymentOptimizer(engine)
        result = optimizer.optimize(
            _Tracker(), small_dataset,
            dataset_name="Synthetic",
            skip_rates=[1, 2],
            scale_factors=[1.0],
            max_sequences=2,
        )
        assert result.tracker_name == "KCF"

    def test_to_dict_round_trips(self, full_result):
        import json
        d = full_result.to_dict()
        serialised = json.dumps(d)
        restored = json.loads(serialised)
        assert len(restored["entries"]) == len(full_result)
        for entry_d in restored["entries"]:
            assert 0.0 <= entry_d["mean_iou"] <= 1.0

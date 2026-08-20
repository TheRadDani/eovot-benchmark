"""Tests for SequenceDifficultyProfiler (analysis/difficulty.py).

Verifies:
- Metric computation for synthetic GT sequences
- Scoring/tier assignment logic
- analyze_dataset, dataset_summary, filter_by_tier, to_markdown_table
- Edge cases (short sequences, stationary targets, extreme scale change)
"""

from __future__ import annotations

import numpy as np
import pytest

from eovot.analysis.difficulty import SequenceDifficultyProfiler, SequenceDifficulty


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stationary_gt(n: int = 50, x=100, y=80, w=60, h=40) -> np.ndarray:
    """GT where the target never moves and never changes size."""
    return np.tile([float(x), float(y), float(w), float(h)], (n, 1))


def _fast_motion_gt(n: int = 60, frame_h: int = 480, frame_w: int = 640) -> np.ndarray:
    """GT where target centre bounces at ~10 % of the diagonal per frame."""
    diagonal = np.sqrt(frame_h ** 2 + frame_w ** 2)
    step = diagonal * 0.10  # 10 % per frame
    gt = []
    x, y, w, h = 50.0, 50.0, 60.0, 40.0
    dx = step
    for _ in range(n):
        gt.append([x, y, w, h])
        x += dx
        if x + w > frame_w - 1 or x < 0:
            dx = -dx
            x = np.clip(x, 0, frame_w - w - 1)
    return np.array(gt)


def _scale_change_gt(n: int = 50) -> np.ndarray:
    """GT where the target area grows from 30×20 to 150×100 linearly."""
    rows = []
    for i in range(n):
        t = i / max(n - 1, 1)
        w = 30.0 + t * 120.0
        h = 20.0 + t * 80.0
        rows.append([100.0, 80.0, w, h])
    return np.array(rows)


def _small_target_gt(n: int = 50, frame_h: int = 480, frame_w: int = 640) -> np.ndarray:
    """GT where the target is < 0.5 % of frame area throughout."""
    frame_area = frame_h * frame_w
    w = h = np.sqrt(frame_area * 0.005)
    return np.tile([100.0, 100.0, w, h], (n, 1))


# ---------------------------------------------------------------------------
# Basic metric correctness
# ---------------------------------------------------------------------------

class TestMotionMetrics:
    def test_stationary_has_zero_motion(self):
        profiler = SequenceDifficultyProfiler(frame_size=(480, 640))
        gt = _stationary_gt()
        result = profiler.analyze_sequence(gt, "stationary")
        assert result.motion_magnitude == pytest.approx(0.0, abs=1e-8)
        assert result.fast_motion_fraction == pytest.approx(0.0, abs=1e-8)

    def test_fast_motion_detected(self):
        profiler = SequenceDifficultyProfiler(frame_size=(480, 640))
        gt = _fast_motion_gt(frame_h=480, frame_w=640)
        result = profiler.analyze_sequence(gt, "fast")
        assert result.motion_magnitude > 0.05, "Fast motion should exceed threshold"
        assert result.fast_motion_fraction > 0.0

    def test_motion_threshold_custom(self):
        profiler = SequenceDifficultyProfiler(frame_size=(480, 640), fast_motion_threshold=0.20)
        gt = _fast_motion_gt(frame_h=480, frame_w=640)
        result_strict = profiler.analyze_sequence(gt, "fast-strict")
        profiler2 = SequenceDifficultyProfiler(frame_size=(480, 640), fast_motion_threshold=0.01)
        result_lax = profiler2.analyze_sequence(gt, "fast-lax")
        assert result_lax.fast_motion_fraction >= result_strict.fast_motion_fraction


class TestScaleMetrics:
    def test_stationary_has_unit_scale_ratio(self):
        profiler = SequenceDifficultyProfiler(frame_size=(480, 640))
        gt = _stationary_gt()
        result = profiler.analyze_sequence(gt, "stat")
        assert result.scale_change_ratio == pytest.approx(1.0, rel=1e-4)
        assert result.scale_variability == pytest.approx(0.0, abs=1e-8)

    def test_scale_change_detected(self):
        profiler = SequenceDifficultyProfiler(frame_size=(480, 640))
        gt = _scale_change_gt()
        result = profiler.analyze_sequence(gt, "scale")
        # Area grows from 30*20=600 to 150*100=15000 → ratio ~25×
        assert result.scale_change_ratio > 10.0
        assert result.scale_variability > 0.3

    def test_scale_variability_nonnegative(self):
        profiler = SequenceDifficultyProfiler(frame_size=(480, 640))
        gt = _fast_motion_gt()
        result = profiler.analyze_sequence(gt, "fm")
        assert result.scale_variability >= 0.0
        # Constant-size target: ratio is max/min ≈ 1.0 (allow float tolerance)
        assert result.scale_change_ratio >= 1.0 - 1e-4


class TestSmallTargetMetrics:
    def test_small_target_fraction_high(self):
        profiler = SequenceDifficultyProfiler(
            frame_size=(480, 640), small_target_threshold=0.01
        )
        gt = _small_target_gt(frame_h=480, frame_w=640)
        result = profiler.analyze_sequence(gt, "small")
        assert result.small_target_fraction == pytest.approx(1.0)

    def test_large_target_fraction_zero(self):
        profiler = SequenceDifficultyProfiler(
            frame_size=(100, 100), small_target_threshold=0.01
        )
        # Target fills half the frame
        gt = np.tile([0.0, 0.0, 50.0, 50.0], (30, 1))
        result = profiler.analyze_sequence(gt, "large")
        assert result.small_target_fraction == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Difficulty score and tier
# ---------------------------------------------------------------------------

class TestDifficultyScore:
    def test_stationary_is_easy(self):
        profiler = SequenceDifficultyProfiler(frame_size=(480, 640))
        gt = _stationary_gt(n=100)
        result = profiler.analyze_sequence(gt, "stat")
        assert result.difficulty_tier == "easy"
        assert result.difficulty_score < 0.33

    def test_fast_motion_raises_score(self):
        profiler = SequenceDifficultyProfiler(frame_size=(480, 640))
        gt_stat = _stationary_gt()
        gt_fast = _fast_motion_gt(frame_h=480, frame_w=640)
        r_stat = profiler.analyze_sequence(gt_stat, "s")
        r_fast = profiler.analyze_sequence(gt_fast, "f")
        assert r_fast.difficulty_score > r_stat.difficulty_score

    def test_score_in_unit_range(self):
        profiler = SequenceDifficultyProfiler(frame_size=(480, 640))
        for gt in [_stationary_gt(), _fast_motion_gt(), _scale_change_gt()]:
            result = profiler.analyze_sequence(gt, "x")
            assert 0.0 <= result.difficulty_score <= 1.0

    def test_tier_classification(self):
        profiler = SequenceDifficultyProfiler(frame_size=(480, 640))
        # Manufacture a "hard" sequence: fast motion + extreme scale + small target
        n = 80
        gt = _fast_motion_gt(n=n, frame_h=480, frame_w=640)
        # Also make it a small target with big scale change
        gt[:, 2] = 5.0 + np.linspace(0, 200, n)
        gt[:, 3] = 5.0 + np.linspace(0, 100, n)
        result = profiler.analyze_sequence(gt, "hard")
        assert result.difficulty_tier in {"medium", "hard"}


# ---------------------------------------------------------------------------
# analyze_dataset
# ---------------------------------------------------------------------------

class TestAnalyzeDataset:
    def setup_method(self):
        self.profiler = SequenceDifficultyProfiler(frame_size=(480, 640))
        self.ground_truths = {
            "easy-seq": _stationary_gt(50),
            "hard-seq": _fast_motion_gt(60, frame_h=480, frame_w=640),
            "scale-seq": _scale_change_gt(50),
        }

    def test_returns_all_sequences(self):
        results = self.profiler.analyze_dataset(self.ground_truths)
        assert set(results.keys()) == {"easy-seq", "hard-seq", "scale-seq"}

    def test_ordered_by_difficulty_descending(self):
        results = self.profiler.analyze_dataset(self.ground_truths)
        scores = [r.difficulty_score for r in results.values()]
        assert scores == sorted(scores, reverse=True)

    def test_all_results_have_correct_type(self):
        results = self.profiler.analyze_dataset(self.ground_truths)
        for v in results.values():
            assert isinstance(v, SequenceDifficulty)

    def test_frame_size_override(self):
        results = self.profiler.analyze_dataset(
            self.ground_truths, frame_size=(720, 1280)
        )
        assert len(results) == 3


# ---------------------------------------------------------------------------
# dataset_summary
# ---------------------------------------------------------------------------

class TestDatasetSummary:
    def test_summary_keys(self):
        profiler = SequenceDifficultyProfiler(frame_size=(480, 640))
        gts = {
            "a": _stationary_gt(40),
            "b": _fast_motion_gt(40, frame_h=480, frame_w=640),
        }
        results = profiler.analyze_dataset(gts)
        summary = profiler.dataset_summary(results)
        assert "num_sequences" in summary
        assert "mean_difficulty_score" in summary
        assert "tier_counts" in summary
        assert summary["num_sequences"] == 2

    def test_empty_dataset(self):
        profiler = SequenceDifficultyProfiler(frame_size=(480, 640))
        assert profiler.dataset_summary({}) == {}

    def test_tier_counts_sum_to_n(self):
        profiler = SequenceDifficultyProfiler(frame_size=(480, 640))
        gts = {str(i): _stationary_gt(20) for i in range(5)}
        results = profiler.analyze_dataset(gts)
        summary = profiler.dataset_summary(results)
        tier_sum = sum(summary["tier_counts"].values())
        assert tier_sum == 5


# ---------------------------------------------------------------------------
# filter_by_tier
# ---------------------------------------------------------------------------

class TestFilterByTier:
    def test_filter_easy(self):
        profiler = SequenceDifficultyProfiler(frame_size=(480, 640))
        gts = {
            "easy-1": _stationary_gt(50),
            "easy-2": _stationary_gt(50),
            "hard-1": _fast_motion_gt(60, frame_h=480, frame_w=640),
        }
        results = profiler.analyze_dataset(gts)
        easy = profiler.filter_by_tier(results, "easy")
        assert all(r.difficulty_tier == "easy" for r in easy.values())

    def test_invalid_tier_raises(self):
        profiler = SequenceDifficultyProfiler(frame_size=(480, 640))
        with pytest.raises(ValueError, match="Invalid tier"):
            profiler.filter_by_tier({}, "extreme")


# ---------------------------------------------------------------------------
# to_markdown_table
# ---------------------------------------------------------------------------

class TestMarkdownTable:
    def test_table_contains_header(self):
        profiler = SequenceDifficultyProfiler(frame_size=(480, 640))
        gts = {"s1": _stationary_gt(30)}
        results = profiler.analyze_dataset(gts)
        table = profiler.to_markdown_table(results)
        assert "| Sequence" in table
        assert "| Score" in table
        assert "| Tier" in table

    def test_table_row_count(self):
        profiler = SequenceDifficultyProfiler(frame_size=(480, 640))
        gts = {f"seq-{i}": _stationary_gt(20) for i in range(4)}
        results = profiler.analyze_dataset(gts)
        table = profiler.to_markdown_table(results)
        data_rows = [l for l in table.split("\n") if l.startswith("| seq")]
        assert len(data_rows) == 4


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_frame_returns_empty_result(self):
        profiler = SequenceDifficultyProfiler(frame_size=(480, 640))
        gt = np.array([[100.0, 80.0, 60.0, 40.0]])
        result = profiler.analyze_sequence(gt, "one-frame")
        assert result.motion_magnitude == pytest.approx(0.0)
        assert result.difficulty_score == pytest.approx(0.0)
        assert result.difficulty_tier == "easy"

    def test_no_frame_size_falls_back_to_estimation(self):
        profiler = SequenceDifficultyProfiler()  # no frame_size
        gt = _stationary_gt(30, x=100, y=80, w=60, h=40)
        result = profiler.analyze_sequence(gt, "est")
        # Should not raise; frame size is estimated from GT boxes
        assert result.motion_magnitude >= 0.0

    def test_package_import(self):
        from eovot.analysis import SequenceDifficultyProfiler as SDP
        assert SDP is not None

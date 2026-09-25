"""Tests for eovot.metrics.difficulty — SequenceDifficultyAnalyzer."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.metrics.difficulty import (
    SequenceDifficultyAnalyzer,
    SequenceDifficulty,
    StratifiedReport,
    TIER_EASY_THRESHOLD,
    TIER_HARD_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic GT trajectories
# ---------------------------------------------------------------------------

def _static_gt(n: int = 50, x: float = 100.0, y: float = 80.0,
               w: float = 60.0, h: float = 40.0) -> np.ndarray:
    """Stationary target: all boxes identical."""
    return np.tile([x, y, w, h], (n, 1)).astype(np.float64)


def _fast_moving_gt(n: int = 100) -> np.ndarray:
    """Target moving very quickly — 20 px/frame on a 40×30 box (diagonal≈50, 40% per frame)."""
    gt = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        gt[i] = [10.0 + i * 20.0, 50.0 + i * 1.0, 40.0, 30.0]
    return gt


def _scale_varying_gt(n: int = 80) -> np.ndarray:
    """Target that grows from 20×15 to 200×150 over the sequence (100× area change)."""
    gt = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        scale = 1.0 + (i / (n - 1)) * 9.0  # 1× → 10× linear
        gt[i] = [100.0, 100.0, 20.0 * scale, 15.0 * scale]
    return gt


def _long_sequence_gt(n: int = 600) -> np.ndarray:
    """Very long sequence."""
    return _static_gt(n)


# ---------------------------------------------------------------------------
# Unit tests: dimension scores
# ---------------------------------------------------------------------------

class TestMotionScore:
    def test_stationary_target_low_motion(self):
        analyzer = SequenceDifficultyAnalyzer()
        gt = _static_gt(100)
        diff = analyzer.score_sequence(gt, "static")
        assert diff.motion_score < 0.15, "Stationary target should have very low motion score"

    def test_fast_target_high_motion(self):
        analyzer = SequenceDifficultyAnalyzer()
        gt = _fast_moving_gt(100)
        diff = analyzer.score_sequence(gt, "fast")
        assert diff.motion_score > 0.50, "Fast-moving target should have high motion score"

    def test_motion_score_in_range(self):
        analyzer = SequenceDifficultyAnalyzer()
        for gt in [_static_gt(), _fast_moving_gt()]:
            diff = analyzer.score_sequence(gt)
            assert 0.0 <= diff.motion_score <= 1.0


class TestScaleScore:
    def test_fixed_size_low_scale(self):
        analyzer = SequenceDifficultyAnalyzer()
        gt = _static_gt(50)
        diff = analyzer.score_sequence(gt, "static")
        assert diff.scale_score < 0.05, "Constant-size target should have near-zero scale score"

    def test_large_scale_change_high_score(self):
        analyzer = SequenceDifficultyAnalyzer()
        gt = _scale_varying_gt(80)
        diff = analyzer.score_sequence(gt, "scale_var")
        assert diff.scale_score > 0.70, "100× area change should yield high scale score"

    def test_scale_score_in_range(self):
        analyzer = SequenceDifficultyAnalyzer()
        diff = analyzer.score_sequence(_scale_varying_gt())
        assert 0.0 <= diff.scale_score <= 1.0


class TestAspectScore:
    def test_constant_aspect_low_score(self):
        analyzer = SequenceDifficultyAnalyzer()
        gt = _static_gt(60)
        diff = analyzer.score_sequence(gt)
        assert diff.aspect_score < 0.01

    def test_varying_aspect_higher_score(self):
        analyzer = SequenceDifficultyAnalyzer()
        gt = np.zeros((50, 4), dtype=np.float64)
        for i in range(50):
            gt[i] = [50.0, 50.0, 30.0 + i * 2.0, 30.0 - i * 0.4]
        diff = analyzer.score_sequence(gt)
        assert diff.aspect_score > 0.10


class TestLengthScore:
    def test_short_sequence_low_length(self):
        analyzer = SequenceDifficultyAnalyzer()
        gt = _static_gt(10)
        diff = analyzer.score_sequence(gt)
        assert diff.length_score < 0.3

    def test_long_sequence_high_length(self):
        analyzer = SequenceDifficultyAnalyzer()
        gt = _long_sequence_gt(600)
        diff = analyzer.score_sequence(gt)
        assert diff.length_score > 0.7

    def test_midpoint_near_half(self):
        analyzer = SequenceDifficultyAnalyzer(length_midpoint=300)
        gt = _static_gt(300)
        diff = analyzer.score_sequence(gt)
        assert abs(diff.length_score - 0.5) < 0.05


# ---------------------------------------------------------------------------
# Unit tests: overall score and tier assignment
# ---------------------------------------------------------------------------

class TestTierAssignment:
    def test_very_easy_sequence(self):
        analyzer = SequenceDifficultyAnalyzer()
        gt = _static_gt(10)  # short + stationary → easy
        diff = analyzer.score_sequence(gt, "trivial")
        assert diff.tier == "easy"
        assert diff.overall_score < TIER_EASY_THRESHOLD

    def test_hard_sequence(self):
        analyzer = SequenceDifficultyAnalyzer()
        # Combine fast motion + extreme scale change + long
        gt_motion = _fast_moving_gt(600)
        gt_motion[:, 2] = 20.0 + np.arange(600) * 0.25   # scale also varies
        diff = analyzer.score_sequence(gt_motion, "hard")
        # Should be medium or hard
        assert diff.overall_score > 0.3

    def test_tier_thresholds_consistent(self):
        analyzer = SequenceDifficultyAnalyzer()
        for gt in [_static_gt(10), _fast_moving_gt(100), _long_sequence_gt(600)]:
            diff = analyzer.score_sequence(gt)
            if diff.tier == "easy":
                assert diff.overall_score < TIER_EASY_THRESHOLD
            elif diff.tier == "hard":
                assert diff.overall_score >= TIER_HARD_THRESHOLD
            else:
                assert TIER_EASY_THRESHOLD <= diff.overall_score < TIER_HARD_THRESHOLD


# ---------------------------------------------------------------------------
# score_sequence edge cases
# ---------------------------------------------------------------------------

class TestScoreSequenceEdgeCases:
    def test_minimum_two_frames(self):
        analyzer = SequenceDifficultyAnalyzer()
        gt = np.array([[10.0, 10.0, 30.0, 20.0], [15.0, 10.0, 30.0, 20.0]])
        diff = analyzer.score_sequence(gt, "two_frame")
        assert isinstance(diff, SequenceDifficulty)
        assert 0.0 <= diff.overall_score <= 1.0

    def test_raises_on_single_frame(self):
        analyzer = SequenceDifficultyAnalyzer()
        with pytest.raises(ValueError, match="at least 2 GT frames"):
            analyzer.score_sequence(np.array([[10.0, 10.0, 30.0, 20.0]]))

    def test_raises_on_wrong_shape(self):
        analyzer = SequenceDifficultyAnalyzer()
        with pytest.raises(ValueError):
            analyzer.score_sequence(np.ones((10, 3)))

    def test_to_dict_has_all_keys(self):
        analyzer = SequenceDifficultyAnalyzer()
        diff = analyzer.score_sequence(_static_gt(50), "test_seq")
        d = diff.to_dict()
        for key in ("sequence_name", "num_frames", "tier", "overall_score",
                    "motion_score", "scale_score", "aspect_score", "length_score"):
            assert key in d, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# score_dataset
# ---------------------------------------------------------------------------

class TestScoreDataset:
    def test_scores_all_sequences(self):
        analyzer = SequenceDifficultyAnalyzer()
        seqs = {
            "static": _static_gt(50),
            "fast": _fast_moving_gt(100),
            "scale": _scale_varying_gt(80),
        }
        results = analyzer.score_dataset(seqs)
        assert len(results) == 3
        names = {r.sequence_name for r in results}
        assert names == {"static", "fast", "scale"}

    def test_skips_short_sequences(self):
        analyzer = SequenceDifficultyAnalyzer()
        seqs = {
            "ok": _static_gt(50),
            "too_short": np.array([[10.0, 10.0, 20.0, 15.0]]),
        }
        results = analyzer.score_dataset(seqs)
        assert len(results) == 1
        assert results[0].sequence_name == "ok"


# ---------------------------------------------------------------------------
# stratified_report (integration with BenchmarkResult mock)
# ---------------------------------------------------------------------------

class TestStratifiedReport:
    def _make_benchmark_result(self):
        """Create a minimal BenchmarkResult with three sequences."""
        from eovot.benchmark.engine import BenchmarkResult, SequenceResult
        from eovot.profiling.profiler import ProfilingResult
        from eovot.metrics.accuracy import AccuracyMetrics

        def _sr(name, gt, ious_val=0.6, fps=100.0):
            n = len(gt)
            profiling = ProfilingResult(
                tracker_name="MOSSE",
                frame_count=n,
                fps=fps,
                latency_mean_ms=1000.0 / fps,
                latency_std_ms=0.5,
                latency_p95_ms=1000.0 / fps * 1.2,
                latency_p99_ms=1000.0 / fps * 1.3,
                latency_cv=0.1,
                peak_memory_mb=50.0,
            )
            accuracy = AccuracyMetrics(
                mean_iou=ious_val,
                success_auc=ious_val - 0.05,
                precision_auc=ious_val + 0.02,
            )
            return SequenceResult(
                sequence_name=name,
                ious=np.full(n, ious_val),
                profiling=profiling,
                ground_truths=gt,
                accuracy_metrics=accuracy,
            )

        result = BenchmarkResult(tracker_name="MOSSE", dataset_name="Synthetic")
        result.sequence_results = [
            _sr("static_short", _static_gt(10), ious_val=0.80),
            _sr("fast_medium", _fast_moving_gt(100), ious_val=0.55),
            _sr("scale_long", _long_sequence_gt(600), ious_val=0.40),
        ]
        return result

    def test_stratified_report_creates_tiers(self):
        analyzer = SequenceDifficultyAnalyzer()
        bench = self._make_benchmark_result()
        report = analyzer.stratified_report(bench)
        assert isinstance(report, StratifiedReport)
        assert report.tracker_name == "MOSSE"
        assert report.dataset_name == "Synthetic"
        assert set(report.tier_stats.keys()) == {"easy", "medium", "hard"}

    def test_total_sequences_match(self):
        analyzer = SequenceDifficultyAnalyzer()
        bench = self._make_benchmark_result()
        report = analyzer.stratified_report(bench)
        total = sum(
            report.tier_stats[t].get("num_sequences", 0)
            for t in ("easy", "medium", "hard")
        )
        assert total == len(bench.sequence_results)

    def test_to_markdown_returns_string(self):
        analyzer = SequenceDifficultyAnalyzer()
        bench = self._make_benchmark_result()
        report = analyzer.stratified_report(bench)
        md = report.to_markdown()
        assert isinstance(md, str)
        assert "MOSSE" in md
        assert "Synthetic" in md
        for tier in ("Easy", "Medium", "Hard"):
            assert tier in md

    def test_per_tier_mean_iou_plausible(self):
        analyzer = SequenceDifficultyAnalyzer()
        bench = self._make_benchmark_result()
        report = analyzer.stratified_report(bench)
        for tier, stats in report.tier_stats.items():
            if stats.get("num_sequences", 0) > 0:
                assert 0.0 <= stats["mean_iou"] <= 1.0


# ---------------------------------------------------------------------------
# Custom weights
# ---------------------------------------------------------------------------

class TestCustomWeights:
    def test_motion_only_weights(self):
        # With all weight on motion, fast-moving sequence should outscore static
        analyzer = SequenceDifficultyAnalyzer(
            weights={"motion": 1.0, "scale": 0.0, "aspect": 0.0, "length": 0.0}
        )
        static_diff = analyzer.score_sequence(_static_gt(50), "static")
        fast_diff = analyzer.score_sequence(_fast_moving_gt(100), "fast")
        assert fast_diff.overall_score > static_diff.overall_score

    def test_weights_renormalised(self):
        # Unnormalised weights should still produce scores in [0, 1]
        analyzer = SequenceDifficultyAnalyzer(
            weights={"motion": 2.0, "scale": 3.0, "aspect": 1.0, "length": 4.0}
        )
        diff = analyzer.score_sequence(_fast_moving_gt(100))
        assert 0.0 <= diff.overall_score <= 1.0


# ---------------------------------------------------------------------------
# __str__ representation
# ---------------------------------------------------------------------------

class TestStringRepresentation:
    def test_str_contains_key_info(self):
        analyzer = SequenceDifficultyAnalyzer()
        diff = analyzer.score_sequence(_static_gt(50), "myseq")
        s = str(diff)
        assert "myseq" in s
        assert diff.tier in s
        assert "overall" in s

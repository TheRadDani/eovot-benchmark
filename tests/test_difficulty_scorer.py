"""Tests for eovot.metrics.difficulty — stratified benchmark analysis.

Tests the new stratified analysis additions:
- TierStats dataclass
- StratifiedBenchmarkReport
- stratify_benchmark_result()
- TIER_EASY_THRESHOLD / TIER_HARD_THRESHOLD constants

The existing SequenceDifficultyScorer and DifficultyReport are
tested in tests/test_difficulty.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from eovot.metrics.difficulty import (
    DifficultyFactors,
    SequenceDifficultyScorer,
    SequenceDifficultyEntry,
    DifficultyReport,
    StratifiedBenchmarkReport,
    TierStats,
    stratify_benchmark_result,
    TIER_EASY_THRESHOLD,
    TIER_HARD_THRESHOLD,
    _assign_tier,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic ground-truth trajectories
# ---------------------------------------------------------------------------

def _static_gt(n: int = 50) -> np.ndarray:
    """Stationary target: all boxes identical."""
    return np.tile([100.0, 80.0, 60.0, 40.0], (n, 1)).astype(np.float64)


def _dynamic_gt(n: int = 80) -> np.ndarray:
    """Target with high motion and scale variation."""
    gt = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        scale = 1.0 + (i / (n - 1)) * 9.0
        gt[i] = [10.0 + i * 10.0, 50.0, 20.0 * scale, 15.0 * scale]
    return gt


def _make_benchmark_result(sequences=None):
    """Build a minimal BenchmarkResult with synthetic sequences."""
    from eovot.benchmark.engine import BenchmarkResult, SequenceResult
    from eovot.profiling.profiler import ProfilingResult
    from eovot.metrics.accuracy import AccuracyMetrics

    if sequences is None:
        sequences = [
            ("seq_easy", _static_gt(30), 0.80, 150.0),
            ("seq_medium", _static_gt(100), 0.60, 100.0),
            ("seq_dynamic", _dynamic_gt(80), 0.45, 80.0),
        ]

    result = BenchmarkResult(tracker_name="MOSSE", dataset_name="Synthetic")
    for name, gt, iou_val, fps in sequences:
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
            mean_iou=iou_val,
            success_auc=iou_val - 0.05,
            precision_auc=iou_val + 0.02,
        )
        result.sequence_results.append(
            SequenceResult(
                sequence_name=name,
                ious=np.full(n, iou_val),
                profiling=profiling,
                ground_truths=gt,
                accuracy_metrics=accuracy,
            )
        )
    return result


# ---------------------------------------------------------------------------
# Tests: _assign_tier helper
# ---------------------------------------------------------------------------

class TestAssignTier:
    def test_below_easy_threshold(self):
        assert _assign_tier(TIER_EASY_THRESHOLD - 0.01) == "easy"

    def test_at_easy_threshold(self):
        assert _assign_tier(TIER_EASY_THRESHOLD) == "medium"

    def test_between_thresholds(self):
        mid = (TIER_EASY_THRESHOLD + TIER_HARD_THRESHOLD) / 2
        assert _assign_tier(mid) == "medium"

    def test_at_hard_threshold(self):
        assert _assign_tier(TIER_HARD_THRESHOLD) == "hard"

    def test_above_hard_threshold(self):
        assert _assign_tier(TIER_HARD_THRESHOLD + 0.1) == "hard"

    def test_zero_score_is_easy(self):
        assert _assign_tier(0.0) == "easy"

    def test_one_score_is_hard(self):
        assert _assign_tier(1.0) == "hard"


# ---------------------------------------------------------------------------
# Tests: TierStats
# ---------------------------------------------------------------------------

class TestTierStats:
    def test_str_contains_tier(self):
        stats = TierStats(tier="medium", num_sequences=5, mean_iou=0.55, mean_fps=100.0)
        assert "medium" in str(stats)

    def test_to_dict_required_keys(self):
        stats = TierStats(tier="hard", num_sequences=3, mean_iou=0.40, mean_fps=80.0,
                          success_auc=0.38, precision_auc=0.52)
        d = stats.to_dict()
        assert d["tier"] == "hard"
        assert d["num_sequences"] == 3
        assert d["mean_iou"] == round(0.40, 4)
        assert "success_auc" in d
        assert "precision_auc" in d

    def test_to_dict_optional_keys_absent_when_none(self):
        stats = TierStats(tier="easy", num_sequences=2, mean_iou=0.70, mean_fps=200.0)
        d = stats.to_dict()
        assert "success_auc" not in d
        assert "precision_auc" not in d


# ---------------------------------------------------------------------------
# Tests: stratify_benchmark_result
# ---------------------------------------------------------------------------

class TestStratifyBenchmarkResult:
    def test_returns_stratified_report(self):
        bench = _make_benchmark_result()
        report = stratify_benchmark_result(bench, frame_size=(240, 320))
        assert isinstance(report, StratifiedBenchmarkReport)

    def test_tracker_and_dataset_names_preserved(self):
        bench = _make_benchmark_result()
        report = stratify_benchmark_result(bench)
        assert report.tracker_name == "MOSSE"
        assert report.dataset_name == "Synthetic"

    def test_all_tiers_present_in_stats(self):
        bench = _make_benchmark_result()
        report = stratify_benchmark_result(bench)
        assert set(report.tier_stats.keys()) == {"easy", "medium", "hard"}

    def test_total_sequences_equals_input(self):
        bench = _make_benchmark_result()
        report = stratify_benchmark_result(bench)
        total = sum(report.tier_stats[t].num_sequences for t in ("easy", "medium", "hard"))
        assert total == len(bench.sequence_results)

    def test_mean_iou_in_valid_range(self):
        bench = _make_benchmark_result()
        report = stratify_benchmark_result(bench)
        for tier, stats in report.tier_stats.items():
            if stats.num_sequences > 0:
                assert 0.0 <= stats.mean_iou <= 1.0, f"Bad mIoU for tier {tier}"

    def test_mean_fps_positive(self):
        bench = _make_benchmark_result()
        report = stratify_benchmark_result(bench)
        for tier, stats in report.tier_stats.items():
            if stats.num_sequences > 0:
                assert stats.mean_fps > 0.0

    def test_success_auc_populated_when_metrics_available(self):
        bench = _make_benchmark_result()
        report = stratify_benchmark_result(bench)
        nonempty = [t for t in ("easy", "medium", "hard")
                    if report.tier_stats[t].num_sequences > 0]
        for tier in nonempty:
            # Sequences in our fixture have accuracy_metrics set
            assert report.tier_stats[tier].success_auc is not None

    def test_difficulty_entries_count_matches_sequences(self):
        bench = _make_benchmark_result()
        report = stratify_benchmark_result(bench)
        assert len(report.difficulty_entries) == len(bench.sequence_results)

    def test_sequences_without_gt_go_to_easy(self):
        """Sequences with no ground_truths should default to easy tier."""
        from eovot.benchmark.engine import BenchmarkResult, SequenceResult
        from eovot.profiling.profiler import ProfilingResult

        result = BenchmarkResult(tracker_name="T", dataset_name="D")
        profiling = ProfilingResult(
            tracker_name="T", frame_count=10, fps=100.0,
            latency_mean_ms=10.0, latency_std_ms=0.5,
            latency_p95_ms=11.0, latency_p99_ms=12.0,
            latency_cv=0.05, peak_memory_mb=30.0,
        )
        result.sequence_results.append(
            SequenceResult(
                sequence_name="no_gt",
                ious=np.full(10, 0.5),
                profiling=profiling,
                ground_truths=None,
            )
        )
        report = stratify_benchmark_result(result)
        assert report.tier_stats["easy"].num_sequences == 1

    def test_with_custom_scorer(self):
        scorer = SequenceDifficultyScorer(frame_size=(480, 640))
        bench = _make_benchmark_result()
        report = stratify_benchmark_result(bench, scorer=scorer)
        total = sum(report.tier_stats[t].num_sequences for t in ("easy", "medium", "hard"))
        assert total == len(bench.sequence_results)


# ---------------------------------------------------------------------------
# Tests: StratifiedBenchmarkReport.to_markdown
# ---------------------------------------------------------------------------

class TestStratifiedReportMarkdown:
    def test_to_markdown_is_string(self):
        bench = _make_benchmark_result()
        report = stratify_benchmark_result(bench)
        md = report.to_markdown()
        assert isinstance(md, str)
        assert len(md) > 50

    def test_markdown_contains_tracker_and_dataset(self):
        bench = _make_benchmark_result()
        report = stratify_benchmark_result(bench)
        md = report.to_markdown()
        assert "MOSSE" in md
        assert "Synthetic" in md

    def test_markdown_contains_all_tiers(self):
        bench = _make_benchmark_result()
        report = stratify_benchmark_result(bench)
        md = report.to_markdown()
        for tier in ("Easy", "Medium", "Hard"):
            assert tier in md, f"Tier '{tier}' not found in markdown"

    def test_markdown_contains_threshold_note(self):
        bench = _make_benchmark_result()
        report = stratify_benchmark_result(bench)
        md = report.to_markdown()
        assert str(TIER_EASY_THRESHOLD) in md or "easy" in md.lower()


# ---------------------------------------------------------------------------
# Tests: StratifiedBenchmarkReport.to_dict
# ---------------------------------------------------------------------------

class TestStratifiedReportDict:
    def test_to_dict_has_required_keys(self):
        bench = _make_benchmark_result()
        report = stratify_benchmark_result(bench)
        d = report.to_dict()
        assert "tracker_name" in d
        assert "dataset_name" in d
        assert "tier_stats" in d
        assert "difficulty_entries" in d

    def test_tier_stats_has_all_tiers(self):
        bench = _make_benchmark_result()
        report = stratify_benchmark_result(bench)
        d = report.to_dict()
        assert set(d["tier_stats"].keys()) == {"easy", "medium", "hard"}

    def test_difficulty_entries_length(self):
        bench = _make_benchmark_result()
        report = stratify_benchmark_result(bench)
        d = report.to_dict()
        assert len(d["difficulty_entries"]) == len(bench.sequence_results)


# ---------------------------------------------------------------------------
# Tests: repr and str
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_contains_tracker(self):
        bench = _make_benchmark_result()
        report = stratify_benchmark_result(bench)
        assert "MOSSE" in repr(report)

    def test_repr_contains_tier_counts(self):
        bench = _make_benchmark_result()
        report = stratify_benchmark_result(bench)
        r = repr(report)
        assert "easy=" in r
        assert "medium=" in r
        assert "hard=" in r

"""Tests for ComprehensiveLeaderboard.

Verifies:
- Weight validation
- Single and multi-tracker ranking
- All three metric dimensions (accuracy, robustness, efficiency)
- Pareto front computation
- Composite score ordering
- Markdown and JSON output formats
- Edge cases (empty input, single tracker, missing AUC data)
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkResult, SequenceResult
from eovot.profiling.profiler import ProfilingResult
from eovot.metrics.accuracy import AccuracyMetrics
from eovot.reporting.comprehensive import ComprehensiveLeaderboard, LeaderboardEntry


# ---------------------------------------------------------------------------
# Fixtures: minimal BenchmarkResult objects
# ---------------------------------------------------------------------------

def _make_profiling(fps: float = 30.0, mem_mb: float = 200.0) -> ProfilingResult:
    lat = 1000.0 / fps if fps > 0 else 100.0
    return ProfilingResult(
        tracker_name="tracker",
        frame_count=50,
        fps=fps,
        latency_mean_ms=lat,
        latency_std_ms=lat * 0.1,
        latency_p95_ms=lat * 1.2,
        latency_p99_ms=lat * 1.5,
        latency_cv=0.1,
        peak_memory_mb=mem_mb,
    )


def _make_sequence(
    name: str,
    mean_iou: float,
    fps: float = 30.0,
    mem_mb: float = 200.0,
    success_auc: float = None,
    n_frames: int = 50,
) -> SequenceResult:
    if success_auc is None:
        success_auc = mean_iou * 0.9
    ious = np.full(n_frames, mean_iou, dtype=np.float64)
    acc = AccuracyMetrics(
        mean_iou=mean_iou,
        success_auc=success_auc,
        precision_auc=mean_iou * 0.85,
        normalized_precision_auc=mean_iou * 0.80,
    )
    return SequenceResult(
        sequence_name=name,
        ious=ious,
        profiling=_make_profiling(fps=fps, mem_mb=mem_mb),
        accuracy_metrics=acc,
    )


def _make_result(
    tracker_name: str,
    dataset_name: str = "Synthetic",
    mean_iou: float = 0.6,
    fps: float = 30.0,
    mem_mb: float = 200.0,
    n_seq: int = 3,
) -> BenchmarkResult:
    r = BenchmarkResult(tracker_name=tracker_name, dataset_name=dataset_name)
    for i in range(n_seq):
        r.sequence_results.append(
            _make_sequence(f"seq_{i:02d}", mean_iou=mean_iou, fps=fps, mem_mb=mem_mb)
        )
    return r


# ---------------------------------------------------------------------------
# Weight validation
# ---------------------------------------------------------------------------

class TestWeightValidation:
    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            ComprehensiveLeaderboard(
                accuracy_weight=0.5,
                robustness_weight=0.3,
                efficiency_weight=0.3,
            )

    def test_valid_custom_weights(self):
        lb = ComprehensiveLeaderboard(
            accuracy_weight=0.4,
            robustness_weight=0.4,
            efficiency_weight=0.2,
        )
        assert lb.accuracy_weight == 0.4
        assert lb.robustness_weight == 0.4
        assert lb.efficiency_weight == 0.2

    def test_default_weights(self):
        lb = ComprehensiveLeaderboard()
        total = lb.accuracy_weight + lb.robustness_weight + lb.efficiency_weight
        assert abs(total - 1.0) < 1e-9

    def test_equal_weights(self):
        lb = ComprehensiveLeaderboard(
            accuracy_weight=1 / 3,
            robustness_weight=1 / 3,
            efficiency_weight=1 / 3,
        )
        total = lb.accuracy_weight + lb.robustness_weight + lb.efficiency_weight
        assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

class TestEmptyInput:
    def test_rank_empty_returns_empty(self):
        lb = ComprehensiveLeaderboard()
        assert lb.rank([]) == []

    def test_markdown_empty(self):
        lb = ComprehensiveLeaderboard()
        md = lb.to_markdown([])
        assert "No results" in md


# ---------------------------------------------------------------------------
# Single tracker
# ---------------------------------------------------------------------------

class TestSingleTracker:
    def test_single_tracker_rank_1(self):
        lb = ComprehensiveLeaderboard()
        r = _make_result("MOSSE")
        entries = lb.rank([r])
        assert len(entries) == 1
        assert entries[0].rank == 1

    def test_single_tracker_pareto_front(self):
        lb = ComprehensiveLeaderboard()
        r = _make_result("MOSSE")
        entries = lb.rank([r])
        assert entries[0].on_pareto_front is True

    def test_single_tracker_composite_in_range(self):
        lb = ComprehensiveLeaderboard()
        r = _make_result("MOSSE", mean_iou=0.7)
        entries = lb.rank([r])
        assert 0.0 <= entries[0].composite_score <= 1.0 + 1e-9

    def test_single_tracker_ees_positive(self):
        lb = ComprehensiveLeaderboard()
        r = _make_result("KCF", mean_iou=0.5, fps=50.0, mem_mb=100.0)
        entries = lb.rank([r])
        assert entries[0].ees > 0


# ---------------------------------------------------------------------------
# Multi-tracker ranking
# ---------------------------------------------------------------------------

class TestMultiTrackerRanking:
    def test_higher_iou_ranks_higher(self):
        lb = ComprehensiveLeaderboard(
            accuracy_weight=0.9,
            robustness_weight=0.05,
            efficiency_weight=0.05,
        )
        r_high = _make_result("HighIoU", mean_iou=0.8, fps=25.0)
        r_low = _make_result("LowIoU", mean_iou=0.3, fps=25.0)
        entries = lb.rank([r_low, r_high])
        assert entries[0].tracker_name == "HighIoU"
        assert entries[1].tracker_name == "LowIoU"

    def test_ranks_are_sequential(self):
        lb = ComprehensiveLeaderboard()
        results = [_make_result(f"T{i}", mean_iou=0.3 + i * 0.1) for i in range(5)]
        entries = lb.rank(results)
        ranks = [e.rank for e in entries]
        assert ranks == list(range(1, 6))

    def test_higher_fps_ranks_higher_when_accuracy_equal(self):
        lb = ComprehensiveLeaderboard(
            accuracy_weight=0.0,
            robustness_weight=0.0,
            efficiency_weight=1.0,
        )
        r_fast = _make_result("FastTracker", mean_iou=0.5, fps=100.0, mem_mb=200.0)
        r_slow = _make_result("SlowTracker", mean_iou=0.5, fps=5.0, mem_mb=200.0)
        entries = lb.rank([r_slow, r_fast])
        assert entries[0].tracker_name == "FastTracker"

    def test_composite_scores_sorted_descending(self):
        lb = ComprehensiveLeaderboard()
        results = [_make_result(f"T{i}", mean_iou=0.2 + i * 0.15, fps=20 + i * 10)
                   for i in range(4)]
        entries = lb.rank(results)
        scores = [e.composite_score for e in entries]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------

class TestParetoFront:
    def test_pareto_front_identified(self):
        lb = ComprehensiveLeaderboard()
        # One tracker dominates all others in both accuracy and EES
        r_best = _make_result("Best", mean_iou=0.9, fps=200.0, mem_mb=50.0)
        r_mid = _make_result("Mid", mean_iou=0.5, fps=50.0, mem_mb=200.0)
        r_worst = _make_result("Worst", mean_iou=0.2, fps=10.0, mem_mb=500.0)
        entries = lb.rank([r_worst, r_mid, r_best])
        pareto = {e.tracker_name: e.on_pareto_front for e in entries}
        assert pareto["Best"] is True
        # A tracker that is strictly dominated in BOTH accuracy and EES should not be Pareto
        # (depends on exact EES values, but with these extreme parameters it should hold)
        assert not all(pareto.values()) or len(entries) == 1

    def test_all_pareto_when_tradeoff(self):
        lb = ComprehensiveLeaderboard()
        # AccuracyFocused: very high IoU but extremely slow (EES stays low due to
        # log1p(fps) ≈ log1p(0.5) ≈ 0.4).  SpeedFocused: low IoU but very fast
        # (EES = 0.25 × log1p(2000) / ... ≈ high).
        # This guarantees neither dominates the other in both (accuracy, EES).
        r_a = _make_result("AccuracyFocused", mean_iou=0.95, fps=0.5, mem_mb=200.0)
        r_b = _make_result("SpeedFocused", mean_iou=0.25, fps=2000.0, mem_mb=200.0)
        entries = lb.rank([r_a, r_b])
        pareto = {e.tracker_name: e.on_pareto_front for e in entries}
        # AccuracyFocused wins accuracy; SpeedFocused wins EES.
        # Neither dominates the other in BOTH dimensions → both Pareto-optimal.
        assert pareto["AccuracyFocused"] is True
        assert pareto["SpeedFocused"] is True


# ---------------------------------------------------------------------------
# Metric field correctness
# ---------------------------------------------------------------------------

class TestMetricFields:
    def test_success_auc_from_accuracy_metrics(self):
        lb = ComprehensiveLeaderboard()
        r = _make_result("T", mean_iou=0.6)
        # The sequence's AccuracyMetrics has success_auc = 0.6 * 0.9 = 0.54
        entries = lb.rank([r])
        assert abs(entries[0].success_auc - 0.54) < 1e-4

    def test_fps_preserved(self):
        lb = ComprehensiveLeaderboard()
        r = _make_result("FastKCF", fps=120.0)
        entries = lb.rank([r])
        assert abs(entries[0].fps - 120.0) < 0.1

    def test_memory_preserved(self):
        lb = ComprehensiveLeaderboard()
        r = _make_result("SmallMem", mem_mb=64.0)
        entries = lb.rank([r])
        assert abs(entries[0].peak_memory_mb - 64.0) < 0.1

    def test_eao_non_negative(self):
        lb = ComprehensiveLeaderboard()
        r = _make_result("T", mean_iou=0.5)
        entries = lb.rank([r])
        assert entries[0].eao >= 0.0

    def test_survival_rate_in_range(self):
        lb = ComprehensiveLeaderboard()
        r = _make_result("T", mean_iou=0.5)
        entries = lb.rank([r])
        assert 0.0 <= entries[0].mean_survival_rate <= 1.0

    def test_robustness_computed_from_ious(self):
        lb = ComprehensiveLeaderboard()
        # Build a result with sequences that have clear failures
        r = BenchmarkResult(tracker_name="FlickeringTracker", dataset_name="Synthetic")
        # Alternating 0.8/0.0 IoU — many failures
        ious_bad = np.array([0.9] + [0.8, 0.0] * 24, dtype=np.float64)
        ious_good = np.full(50, 0.8, dtype=np.float64)
        for seq_name, iou_arr in [("seq0", ious_bad), ("seq1", ious_good)]:
            r.sequence_results.append(
                SequenceResult(
                    sequence_name=seq_name,
                    ious=iou_arr,
                    profiling=_make_profiling(),
                )
            )
        entries = lb.rank([r])
        assert entries[0].num_failures >= 1


# ---------------------------------------------------------------------------
# Output formats
# ---------------------------------------------------------------------------

class TestOutputFormats:
    def _default_entries(self) -> List[LeaderboardEntry]:
        lb = ComprehensiveLeaderboard()
        results = [
            _make_result("MOSSE", mean_iou=0.55, fps=200.0, mem_mb=80.0),
            _make_result("KCF", mean_iou=0.62, fps=120.0, mem_mb=120.0),
            _make_result("CSRT", mean_iou=0.72, fps=40.0, mem_mb=250.0),
        ]
        return lb.rank(results), lb

    def test_markdown_contains_header(self):
        entries, lb = self._default_entries()
        md = lb.to_markdown(entries)
        assert "Comprehensive Leaderboard" in md

    def test_markdown_has_all_trackers(self):
        entries, lb = self._default_entries()
        md = lb.to_markdown(entries)
        for name in ["MOSSE", "KCF", "CSRT"]:
            assert name in md

    def test_markdown_rank_column(self):
        entries, lb = self._default_entries()
        md = lb.to_markdown(entries)
        assert "| 1 |" in md
        assert "| 2 |" in md
        assert "| 3 |" in md

    def test_save_json_creates_file(self):
        entries, lb = self._default_entries()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "lb.json"
            lb.save_json(entries, out)
            assert out.exists()
            with open(out) as f:
                payload = json.load(f)
            assert "entries" in payload
            assert "weights" in payload
            assert len(payload["entries"]) == 3

    def test_save_json_adds_extension(self):
        entries, lb = self._default_entries()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_noext = Path(tmpdir) / "leaderboard"
            p = lb.save_json(entries, out_noext)
            assert p.suffix == ".json"
            assert p.exists()

    def test_json_entry_fields(self):
        entries, lb = self._default_entries()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "lb.json"
            lb.save_json(entries, out)
            with open(out) as f:
                payload = json.load(f)
            required = {
                "tracker_name", "dataset_name", "mean_iou", "success_auc",
                "eao", "fps", "peak_memory_mb", "ees", "composite_score", "rank",
            }
            for entry_d in payload["entries"]:
                assert required.issubset(entry_d.keys()), (
                    f"Missing fields: {required - entry_d.keys()}"
                )

    def test_to_dict_rounds_floats(self):
        lb = ComprehensiveLeaderboard()
        r = _make_result("T", mean_iou=0.123456789)
        entries = lb.rank([r])
        d = entries[0].to_dict()
        # Floats should be rounded to 4 decimal places
        assert d["mean_iou"] == round(entries[0].mean_iou, 4)

    def test_leaderboard_entry_str(self):
        lb = ComprehensiveLeaderboard()
        r = _make_result("MOSSETracker")
        entries = lb.rank([r])
        s = str(entries[0])
        assert "MOSSETracker" in s
        assert "Comp=" in s


# ---------------------------------------------------------------------------
# Composite score properties
# ---------------------------------------------------------------------------

class TestCompositeScore:
    def test_composite_normalised_ees_single_tracker(self):
        """With one tracker, normalised EES = 0 or 1; composite is bounded."""
        lb = ComprehensiveLeaderboard()
        r = _make_result("T", mean_iou=0.5, fps=30.0)
        entries = lb.rank([r])
        assert 0.0 <= entries[0].composite_score <= 1.0 + 1e-9

    def test_pure_accuracy_weight(self):
        """All weight on accuracy: composite = success_AUC."""
        lb = ComprehensiveLeaderboard(
            accuracy_weight=1.0,
            robustness_weight=0.0,
            efficiency_weight=0.0,
        )
        r = _make_result("T", mean_iou=0.7)
        entries = lb.rank([r])
        expected_auc = 0.7 * 0.9   # from _make_sequence
        assert abs(entries[0].composite_score - expected_auc) < 1e-6

    def test_pure_robustness_weight(self):
        """All weight on robustness: composite = EAO."""
        lb = ComprehensiveLeaderboard(
            accuracy_weight=0.0,
            robustness_weight=1.0,
            efficiency_weight=0.0,
        )
        r = _make_result("T", mean_iou=0.6)
        entries = lb.rank([r])
        assert abs(entries[0].composite_score - entries[0].eao) < 1e-6

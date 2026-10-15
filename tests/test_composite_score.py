"""Tests for the CompositeScoreEngine multi-dimensional ranking system."""

import numpy as np
import pytest

from eovot.metrics.composite import CompositeScoreEngine, CompositeEntry, DimensionScores
from eovot.benchmark.engine import BenchmarkResult, SequenceResult
from eovot.profiling.profiler import ProfilingResult
from eovot.metrics.accuracy import AccuracyMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    tracker_name: str,
    mean_iou: float = 0.5,
    fps: float = 30.0,
    peak_mem: float = 200.0,
    success_auc: float = 0.45,
    norm_precision_auc: float = 0.40,
    n_failures: int = 2,
    n_frames: int = 100,
) -> BenchmarkResult:
    """Build a minimal BenchmarkResult with realistic metrics."""
    profiling = ProfilingResult(
        tracker_name=tracker_name,
        frame_count=n_frames,
        fps=fps,
        latency_mean_ms=1000.0 / fps,
        latency_std_ms=1.0,
        latency_p95_ms=1000.0 / fps * 1.5,
        latency_p99_ms=1000.0 / fps * 2.0,
        latency_cv=0.05,
        peak_memory_mb=peak_mem,
    )

    # Build IoU array: mostly good, with n_failures short failure windows
    ious = np.full(n_frames, mean_iou)
    step = max(n_frames // (n_failures + 1), 1)
    for k in range(1, n_failures + 1):
        fail_idx = k * step
        ious[fail_idx : fail_idx + 3] = 0.05

    accuracy = AccuracyMetrics(
        mean_iou=mean_iou,
        success_auc=success_auc,
        precision_auc=0.35,
        normalized_precision_auc=norm_precision_auc,
    )

    seq = SequenceResult(
        sequence_name="seq_0",
        ious=ious,
        profiling=profiling,
        accuracy_metrics=accuracy,
    )
    result = BenchmarkResult(tracker_name=tracker_name, dataset_name="test")
    result.sequence_results = [seq]
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCompositeScoreEngineInit:
    def test_default_weights_sum_to_one(self):
        engine = CompositeScoreEngine()
        total = engine.weight_accuracy + engine.weight_efficiency + engine.weight_robustness
        assert abs(total - 1.0) < 1e-9

    def test_custom_weights_accepted(self):
        engine = CompositeScoreEngine(
            weight_accuracy=0.5, weight_efficiency=0.3, weight_robustness=0.2
        )
        assert engine.weight_accuracy == 0.5

    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError, match="sum to 1"):
            CompositeScoreEngine(weight_accuracy=0.5, weight_efficiency=0.3, weight_robustness=0.1)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            CompositeScoreEngine(weight_accuracy=-0.1, weight_efficiency=0.6, weight_robustness=0.5)


class TestCompositeScoreEngineRank:
    def test_empty_input_returns_empty(self):
        engine = CompositeScoreEngine()
        assert engine.rank([]) == []

    def test_single_tracker_scores_one(self):
        engine = CompositeScoreEngine()
        result = _make_result("TrackerA", mean_iou=0.6)
        ranking = engine.rank([result])
        assert len(ranking) == 1
        entry = ranking[0]
        assert entry.rank == 1
        # With a single tracker, all normalised scores are 1.0
        assert abs(entry.dimensions.accuracy - 1.0) < 1e-9
        assert abs(entry.dimensions.efficiency - 1.0) < 1e-9
        assert abs(entry.dimensions.robustness - 1.0) < 1e-9
        assert abs(entry.eovot_score - 1.0) < 1e-9

    def test_better_tracker_ranked_higher(self):
        engine = CompositeScoreEngine()
        good = _make_result("Good",  mean_iou=0.75, fps=60.0, peak_mem=100.0, n_failures=0)
        bad  = _make_result("Bad",   mean_iou=0.30, fps=10.0, peak_mem=400.0, n_failures=8)
        ranking = engine.rank([bad, good])
        assert ranking[0].tracker_name == "Good"
        assert ranking[1].tracker_name == "Bad"
        assert ranking[0].eovot_score > ranking[1].eovot_score

    def test_ranks_are_one_based_contiguous(self):
        engine = CompositeScoreEngine()
        results = [_make_result(f"T{i}", mean_iou=0.4 + i * 0.05) for i in range(4)]
        ranking = engine.rank(results)
        assert [e.rank for e in ranking] == [1, 2, 3, 4]

    def test_eovot_score_in_unit_interval(self):
        engine = CompositeScoreEngine()
        results = [
            _make_result("A", mean_iou=0.6, fps=50),
            _make_result("B", mean_iou=0.3, fps=15),
            _make_result("C", mean_iou=0.8, fps=80),
        ]
        for entry in engine.rank(results):
            assert 0.0 <= entry.eovot_score <= 1.0

    def test_dimension_scores_in_unit_interval(self):
        engine = CompositeScoreEngine()
        results = [_make_result("X", mean_iou=0.5), _make_result("Y", mean_iou=0.7)]
        for entry in engine.rank(results):
            assert 0.0 <= entry.dimensions.accuracy <= 1.0
            assert 0.0 <= entry.dimensions.efficiency <= 1.0
            assert 0.0 <= entry.dimensions.robustness <= 1.0

    def test_identical_trackers_same_score(self):
        engine = CompositeScoreEngine()
        r1 = _make_result("Clone1", mean_iou=0.55, fps=25, peak_mem=150)
        r2 = _make_result("Clone2", mean_iou=0.55, fps=25, peak_mem=150)
        ranking = engine.rank([r1, r2])
        assert abs(ranking[0].eovot_score - ranking[1].eovot_score) < 1e-9


class TestCompositeMarkdown:
    def test_markdown_has_header_and_rows(self):
        engine = CompositeScoreEngine()
        results = [_make_result("A"), _make_result("B")]
        ranking = engine.rank(results)
        md = engine.to_markdown(ranking)
        assert "| Rank |" in md
        assert "EOVOT" in md
        assert "| 1 |" in md
        assert "| 2 |" in md

    def test_markdown_row_count(self):
        engine = CompositeScoreEngine()
        results = [_make_result(f"T{i}") for i in range(5)]
        md = engine.to_markdown(engine.rank(results))
        # header + separator + 5 data rows = 7 lines
        lines = [l for l in md.splitlines() if l.strip()]
        assert len(lines) == 7


class TestCompositeDictExport:
    def test_to_dict_has_required_keys(self):
        engine = CompositeScoreEngine()
        ranking = engine.rank([_make_result("T")])
        d = engine.to_dict(ranking)
        assert len(d) == 1
        entry = d[0]
        for key in ("rank", "tracker_name", "eovot_score", "dimensions", "raw"):
            assert key in entry
        for rk in ("mean_iou", "fps", "peak_memory_mb", "ees", "eao", "survival_rate"):
            assert rk in entry["raw"]

    def test_to_dict_rounds_correctly(self):
        engine = CompositeScoreEngine()
        ranking = engine.rank([_make_result("T1"), _make_result("T2")])
        d = engine.to_dict(ranking)
        for e in d:
            assert isinstance(e["eovot_score"], float)
            assert isinstance(e["raw"]["fps"], float)

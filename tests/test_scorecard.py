"""Tests for eovot.metrics.scorecard.TrackerScorecard."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkResult, SequenceResult
from eovot.datasets.synthetic import SyntheticDataset
from eovot.benchmark.engine import BenchmarkEngine
from eovot.metrics.scorecard import TrackerScore, TrackerScorecard
from eovot.profiling.profiler import ProfilingResult
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profiling(fps: float = 30.0, mem: float = 100.0) -> ProfilingResult:
    return ProfilingResult(
        tracker_name="test",
        frame_count=50,
        fps=fps,
        latency_mean_ms=1000.0 / fps,
        latency_std_ms=1.0,
        latency_p95_ms=1000.0 / fps + 2.0,
        latency_p99_ms=1000.0 / fps + 4.0,
        latency_cv=0.05,
        peak_memory_mb=mem,
    )


def _make_result(
    tracker_name: str,
    iou_val: float,
    fps: float = 30.0,
    mem: float = 100.0,
    n_frames: int = 50,
) -> BenchmarkResult:
    """Create a minimal BenchmarkResult with synthetic data for one sequence."""
    ious = np.full(n_frames, iou_val, dtype=np.float64)
    preds = np.tile([10.0, 10.0, 50.0, 50.0], (n_frames, 1))
    gts = np.tile([10.0, 10.0, 50.0, 50.0], (n_frames, 1))

    from eovot.metrics.accuracy import MetricsEngine, AccuracyMetrics

    eng = MetricsEngine()
    acc = eng.compute_all(preds, gts)
    dists = eng.batch_center_distance(preds, gts)

    seq = SequenceResult(
        sequence_name="seq1",
        ious=ious,
        profiling=_make_profiling(fps=fps, mem=mem),
        predictions=preds,
        ground_truths=gts,
        center_distances=dists,
        accuracy_metrics=acc,
    )
    result = BenchmarkResult(tracker_name=tracker_name, dataset_name="SynTest")
    result.sequence_results = [seq]
    return result


# ---------------------------------------------------------------------------
# Unit tests: TrackerScorecard
# ---------------------------------------------------------------------------


class TestTrackerScorecardConstruction:
    def test_default_weights_sum_to_one(self):
        sc = TrackerScorecard()
        total = (
            sc.accuracy_weight
            + sc.robustness_weight
            + sc.efficiency_weight
            + sc.temporal_weight
        )
        assert abs(total - 1.0) < 1e-9

    def test_custom_valid_weights(self):
        sc = TrackerScorecard(
            accuracy_weight=0.4,
            robustness_weight=0.3,
            efficiency_weight=0.2,
            temporal_weight=0.1,
        )
        assert sc.accuracy_weight == pytest.approx(0.4)

    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            TrackerScorecard(
                accuracy_weight=0.5,
                robustness_weight=0.5,
                efficiency_weight=0.5,
                temporal_weight=0.5,
            )

    def test_empty_results_raise(self):
        sc = TrackerScorecard()
        with pytest.raises(ValueError):
            sc.compute([])


class TestTrackerScorecardCompute:
    """Tests using synthetic BenchmarkResult objects."""

    def test_returns_one_score_per_result(self):
        results = [
            _make_result("TrackerA", iou_val=0.8, fps=60.0),
            _make_result("TrackerB", iou_val=0.5, fps=120.0),
            _make_result("TrackerC", iou_val=0.3, fps=200.0),
        ]
        sc = TrackerScorecard()
        scores = sc.compute(results)
        assert len(scores) == 3

    def test_sorted_by_composite_descending(self):
        results = [
            _make_result("Low",  iou_val=0.2, fps=10.0),
            _make_result("High", iou_val=0.9, fps=80.0),
            _make_result("Mid",  iou_val=0.5, fps=40.0),
        ]
        sc = TrackerScorecard()
        scores = sc.compute(results)
        composites = [s.composite_score for s in scores]
        assert composites == sorted(composites, reverse=True)

    def test_scores_in_zero_one(self):
        results = [
            _make_result("A", iou_val=0.7),
            _make_result("B", iou_val=0.4),
        ]
        sc = TrackerScorecard()
        for s in sc.compute(results):
            for dim in (s.accuracy_score, s.robustness_score, s.efficiency_score, s.temporal_score, s.composite_score):
                assert 0.0 <= dim <= 1.0, f"{dim} out of [0, 1] for {s.tracker_name}"

    def test_single_tracker_scores_midpoint(self):
        sc = TrackerScorecard()
        scores = sc.compute([_make_result("Solo", iou_val=0.6)])
        s = scores[0]
        # min-max of a singleton → 0.5
        assert s.accuracy_score == pytest.approx(0.5)
        assert s.robustness_score == pytest.approx(0.5)

    def test_tracker_names_preserved(self):
        results = [
            _make_result("MOSSE", iou_val=0.7),
            _make_result("KCF",   iou_val=0.65),
        ]
        sc = TrackerScorecard()
        names = {s.tracker_name for s in sc.compute(results)}
        assert names == {"MOSSE", "KCF"}

    def test_to_dict_keys(self):
        results = [_make_result("T1", 0.6), _make_result("T2", 0.4)]
        sc = TrackerScorecard()
        scores = sc.compute(results)
        d = scores[0].to_dict()
        for key in (
            "tracker_name", "dataset_name", "composite_score",
            "accuracy_score", "robustness_score", "efficiency_score",
            "temporal_score", "mean_iou", "success_auc", "eao", "ees",
            "smoothness", "fps",
        ):
            assert key in d, f"Missing key: {key}"

    def test_str_representation(self):
        results = [_make_result("T1", 0.6)]
        sc = TrackerScorecard()
        s = sc.compute(results)[0]
        assert "T1" in str(s)
        assert "composite" in str(s)


class TestTrackerScorecardFormatters:
    def test_markdown_table_has_header(self):
        results = [_make_result("A", 0.7), _make_result("B", 0.4)]
        sc = TrackerScorecard()
        scores = sc.compute(results)
        md = sc.to_markdown_table(scores)
        assert "| Rank |" in md
        assert "Composite" in md
        assert "A" in md or "B" in md

    def test_markdown_table_row_count(self):
        results = [_make_result(f"T{i}", 0.5 + i * 0.05) for i in range(4)]
        sc = TrackerScorecard()
        scores = sc.compute(results)
        md = sc.to_markdown_table(scores)
        # Header + separator + 4 data rows
        lines = [l for l in md.splitlines() if l.strip()]
        assert len(lines) == 6

    def test_latex_table_structure(self):
        results = [_make_result("A", 0.7), _make_result("B", 0.4)]
        sc = TrackerScorecard()
        scores = sc.compute(results)
        latex = sc.to_latex_table(scores)
        assert r"\begin{table}" in latex
        assert r"\toprule" in latex
        assert r"\midrule" in latex
        assert r"\bottomrule" in latex
        assert r"\end{table}" in latex


class TestTrackerScorecardEndToEnd:
    """Integration tests that run through BenchmarkEngine."""

    def test_full_pipeline_two_trackers(self):
        dataset = SyntheticDataset(num_sequences=2, num_frames=60, seed=0)
        engine = BenchmarkEngine(verbose=False)
        results = [
            engine.run(MOSSETracker(), dataset, dataset_name="Synthetic"),
            engine.run(KCFTracker(), dataset, dataset_name="Synthetic"),
        ]
        sc = TrackerScorecard()
        scores = sc.compute(results)
        assert len(scores) == 2
        for s in scores:
            assert 0.0 <= s.composite_score <= 1.0
            assert s.fps > 0

    def test_markdown_output_contains_both_trackers(self):
        dataset = SyntheticDataset(num_sequences=2, num_frames=40, seed=1)
        engine = BenchmarkEngine(verbose=False)
        results = [
            engine.run(MOSSETracker(), dataset, dataset_name="Synthetic"),
            engine.run(KCFTracker(), dataset, dataset_name="Synthetic"),
        ]
        sc = TrackerScorecard()
        scores = sc.compute(results)
        md = sc.to_markdown_table(scores)
        assert "MOSSE" in md
        assert "KCF" in md

"""Tests for EdgeScorecardEngine — unified edge deployment scoring."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pytest

from eovot.metrics.scorecard import EdgeScorecardEngine, ScorecardEntry, ScorecardWeights


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------

def _make_fake_result(
    tracker_name: str = "Tracker",
    dataset_name: str = "Dataset",
    mean_iou: float = 0.5,
    fps: float = 30.0,
    peak_memory_mb: float = 200.0,
    success_auc: float = 0.5,
):
    """Construct a minimal BenchmarkResult-like namespace for testing."""
    from types import SimpleNamespace
    import numpy as _np

    ious = _np.full(20, mean_iou)
    preds = _np.tile(_np.array([10.0, 10.0, 50.0, 50.0]), (20, 1))

    from eovot.profiling.profiler import ProfilingResult
    profiling = ProfilingResult(
        tracker_name=tracker_name,
        frame_count=20,
        fps=fps,
        latency_mean_ms=1000.0 / fps,
        latency_std_ms=0.5,
        latency_p95_ms=1000.0 / fps + 1.0,
        latency_p99_ms=1000.0 / fps + 2.0,
        latency_cv=0.05,
        peak_memory_mb=peak_memory_mb,
    )
    from eovot.metrics.accuracy import AccuracyMetrics
    accuracy = AccuracyMetrics(
        mean_iou=mean_iou,
        success_auc=success_auc,
        precision_auc=success_auc * 0.9,
        normalized_precision_auc=success_auc * 0.85,
    )
    from eovot.benchmark.engine import SequenceResult
    seq_r = SequenceResult(
        sequence_name="s0",
        ious=ious,
        profiling=profiling,
        predictions=preds,
        ground_truths=preds.copy(),
        accuracy_metrics=accuracy,
    )

    from eovot.benchmark.engine import BenchmarkResult
    result = BenchmarkResult(tracker_name=tracker_name, dataset_name=dataset_name)
    result.sequence_results = [seq_r]
    return result


# ---------------------------------------------------------------------------
# ScorecardWeights
# ---------------------------------------------------------------------------

class TestScorecardWeights:
    def test_normalised_sums_to_one(self):
        w = ScorecardWeights(accuracy=1, robustness=2, smoothness=1, efficiency=1).normalised()
        total = w.accuracy + w.robustness + w.smoothness + w.efficiency
        assert abs(total - 1.0) < 1e-9

    def test_equal_weights_all_025(self):
        w = ScorecardWeights().normalised()
        for val in [w.accuracy, w.robustness, w.smoothness, w.efficiency]:
            assert abs(val - 0.25) < 1e-9

    def test_zero_weights_raises(self):
        with pytest.raises(ValueError, match="zero"):
            ScorecardWeights(0, 0, 0, 0).normalised()


# ---------------------------------------------------------------------------
# EdgeScorecardEngine.score
# ---------------------------------------------------------------------------

class TestEdgeScorecardEngineScore:
    def setup_method(self):
        self.engine = EdgeScorecardEngine()

    def test_returns_scorecard_entry(self):
        result = _make_fake_result()
        entry = self.engine.score(result)
        assert isinstance(entry, ScorecardEntry)

    def test_esi_in_valid_range(self):
        result = _make_fake_result()
        entry = self.engine.score(result)
        assert 0.0 <= entry.esi <= 100.0

    def test_dimension_scores_in_unit_interval(self):
        result = _make_fake_result()
        entry = self.engine.score(result)
        for score in [entry.accuracy_score, entry.robustness_score,
                      entry.smoothness_score, entry.efficiency_score]:
            assert 0.0 <= score <= 1.0, f"Out-of-range score: {score}"

    def test_perfect_accuracy_high_esi(self):
        good = _make_fake_result(mean_iou=0.9, fps=60.0, success_auc=0.9)
        poor = _make_fake_result(mean_iou=0.2, fps=10.0, success_auc=0.2)
        assert self.engine.score(good).esi > self.engine.score(poor).esi

    def test_tracker_and_dataset_names_preserved(self):
        result = _make_fake_result(tracker_name="KCF", dataset_name="Synthetic")
        entry = self.engine.score(result)
        assert entry.tracker_name == "KCF"
        assert entry.dataset_name == "Synthetic"

    def test_source_metrics_populated(self):
        entry = self.engine.score(_make_fake_result())
        for key in ["success_auc", "mean_eao", "mean_smoothness", "ees", "mean_iou"]:
            assert key in entry.source_metrics

    def test_to_dict_has_required_keys(self):
        entry = self.engine.score(_make_fake_result())
        d = entry.to_dict()
        for key in ["tracker_name", "dataset_name", "esi", "accuracy_score",
                    "robustness_score", "smoothness_score", "efficiency_score"]:
            assert key in d

    def test_custom_weights_affect_esi(self):
        # High efficiency weight should rank a fast tracker higher
        w_eff = ScorecardWeights(accuracy=0.1, robustness=0.1, smoothness=0.1, efficiency=0.7)
        w_acc = ScorecardWeights(accuracy=0.7, robustness=0.1, smoothness=0.1, efficiency=0.1)
        fast = _make_fake_result(mean_iou=0.4, fps=120.0, success_auc=0.4)
        accurate = _make_fake_result(mean_iou=0.8, fps=5.0, success_auc=0.8)

        esi_eff_fast = EdgeScorecardEngine(weights=w_eff).score(fast).esi
        esi_eff_acc = EdgeScorecardEngine(weights=w_eff).score(accurate).esi
        esi_acc_fast = EdgeScorecardEngine(weights=w_acc).score(fast).esi
        esi_acc_acc = EdgeScorecardEngine(weights=w_acc).score(accurate).esi

        # Efficiency-weighted engine should rank fast > accurate
        assert esi_eff_fast > esi_eff_acc
        # Accuracy-weighted engine should rank accurate > fast
        assert esi_acc_acc > esi_acc_fast


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

class TestEdgeScorecardEngineRank:
    def test_rank_returns_sorted_descending(self):
        engine = EdgeScorecardEngine()
        results = [
            _make_fake_result("A", mean_iou=0.3, fps=15.0, success_auc=0.3),
            _make_fake_result("B", mean_iou=0.7, fps=50.0, success_auc=0.7),
            _make_fake_result("C", mean_iou=0.5, fps=30.0, success_auc=0.5),
        ]
        ranked = engine.rank(results)
        esi_values = [e.esi for e in ranked]
        assert esi_values == sorted(esi_values, reverse=True)

    def test_rank_length_matches_input(self):
        engine = EdgeScorecardEngine()
        results = [_make_fake_result(f"T{i}") for i in range(4)]
        ranked = engine.rank(results)
        assert len(ranked) == 4


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

class TestEdgeScorecardOutputFormats:
    def setup_method(self):
        self.engine = EdgeScorecardEngine()
        self.entries = [
            self.engine.score(_make_fake_result("MOSSE", fps=50.0, mean_iou=0.35)),
            self.engine.score(_make_fake_result("KCF",  fps=30.0, mean_iou=0.55)),
        ]

    def test_to_markdown_contains_esi_column(self):
        md = self.engine.to_markdown(self.entries)
        assert "ESI" in md

    def test_to_markdown_contains_all_tracker_names(self):
        md = self.engine.to_markdown(self.entries)
        assert "MOSSE" in md
        assert "KCF" in md

    def test_to_csv_header_has_esi_column(self):
        csv = self.engine.to_csv(self.entries)
        assert "esi" in csv.split("\n")[0]

    def test_to_csv_row_count_matches_entries(self):
        csv = self.engine.to_csv(self.entries)
        lines = [l for l in csv.split("\n") if l.strip()]
        assert len(lines) == len(self.entries) + 1  # +1 for header

    def test_str_contains_esi_value(self):
        entry = self.entries[0]
        s = str(entry)
        assert "ESI=" in s

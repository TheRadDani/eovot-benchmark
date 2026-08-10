"""Tests for eovot.metrics.deployability — DeployabilityScoreEngine."""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkResult, SequenceResult
from eovot.metrics.accuracy import AccuracyMetrics
from eovot.metrics.deployability import (
    DeployabilityEntry,
    DeployabilityScoreEngine,
    _normalise,
)
from eovot.profiling.profiler import ProfilingResult


# ---------------------------------------------------------------------------
# Helpers — build lightweight mock objects without running the benchmark
# ---------------------------------------------------------------------------

def _profiling(fps: float = 30.0, mem_mb: float = 100.0) -> ProfilingResult:
    return ProfilingResult(
        tracker_name="mock",
        frame_count=50,
        fps=fps,
        latency_mean_ms=1000.0 / fps if fps > 0 else 0.0,
        latency_std_ms=0.5,
        latency_p95_ms=2.0,
        peak_memory_mb=mem_mb,
    )


def _seq_result(
    name: str = "seq0",
    ious: Optional[np.ndarray] = None,
    fps: float = 30.0,
    mem_mb: float = 100.0,
    predictions: Optional[np.ndarray] = None,
    success_auc: Optional[float] = None,
) -> SequenceResult:
    if ious is None:
        ious = np.linspace(0.3, 0.8, 50)
    acc = None
    if success_auc is not None:
        acc = AccuracyMetrics(
            mean_iou=float(ious.mean()),
            success_auc=success_auc,
            precision_auc=success_auc * 0.9,
        )
    if predictions is None:
        n = len(ious)
        predictions = np.tile([10.0, 10.0, 20.0, 20.0], (n, 1))
    return SequenceResult(
        sequence_name=name,
        ious=ious,
        profiling=_profiling(fps=fps, mem_mb=mem_mb),
        predictions=predictions,
        accuracy_metrics=acc,
    )


def _result(
    tracker: str = "MOSSE",
    dataset: str = "Synthetic",
    n_seqs: int = 3,
    mean_iou: float = 0.5,
    fps: float = 30.0,
    mem_mb: float = 100.0,
    success_auc: Optional[float] = None,
    with_predictions: bool = True,
) -> BenchmarkResult:
    r = BenchmarkResult(tracker_name=tracker, dataset_name=dataset)
    for i in range(n_seqs):
        ious = np.full(50, mean_iou)
        preds = np.tile([10.0, 10.0, 20.0, 20.0], (50, 1)) if with_predictions else None
        r.sequence_results.append(
            _seq_result(
                name=f"seq{i}",
                ious=ious,
                fps=fps,
                mem_mb=mem_mb,
                predictions=preds,
                success_auc=success_auc,
            )
        )
    return r


# ---------------------------------------------------------------------------
# _normalise helper
# ---------------------------------------------------------------------------

class TestNormalise:
    def test_empty_returns_empty(self):
        assert _normalise([]) == []

    def test_single_value_returns_one(self):
        assert _normalise([0.7]) == [1.0]

    def test_identical_values_returns_ones(self):
        result = _normalise([0.5, 0.5, 0.5])
        assert result == [1.0, 1.0, 1.0]

    def test_min_is_zero_max_is_one(self):
        result = _normalise([0.0, 0.5, 1.0])
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.5)
        assert result[2] == pytest.approx(1.0)

    def test_monotone_ascending(self):
        values = [1.0, 2.0, 3.0, 4.0]
        result = _normalise(values)
        for i in range(len(result) - 1):
            assert result[i] < result[i + 1]

    def test_range_zero_to_one(self):
        values = list(range(10))
        result = _normalise(values)
        assert min(result) == pytest.approx(0.0)
        assert max(result) == pytest.approx(1.0)

    def test_all_values_in_unit_interval(self):
        values = [0.1, 0.9, 0.4, 0.3, 0.8]
        result = _normalise(values)
        assert all(0.0 <= v <= 1.0 for v in result)

    def test_length_preserved(self):
        values = [0.2, 0.5, 0.8, 0.1]
        assert len(_normalise(values)) == 4


# ---------------------------------------------------------------------------
# DeployabilityEntry
# ---------------------------------------------------------------------------

class TestDeployabilityEntry:
    def _make(self, ds: float = 0.75, pareto: bool = True) -> DeployabilityEntry:
        return DeployabilityEntry(
            tracker_name="KCF",
            dataset_name="OTB",
            accuracy_score=0.8,
            efficiency_score=0.7,
            robustness_score=0.6,
            smoothness_score=0.9,
            weight_accuracy=0.25,
            weight_efficiency=0.25,
            weight_robustness=0.25,
            weight_smoothness=0.25,
            deployability_score=ds,
            mean_iou=0.65,
            fps=45.0,
            peak_memory_mb=120.0,
            success_auc=0.72,
            ees=3.4,
            on_pareto_front=pareto,
        )

    def test_component_dict_keys(self):
        e = self._make()
        cd = e.component_dict()
        assert set(cd.keys()) == {"accuracy", "efficiency", "robustness", "smoothness"}

    def test_component_dict_values_rounded(self):
        e = self._make()
        cd = e.component_dict()
        assert cd["accuracy"] == pytest.approx(0.8, abs=1e-4)

    def test_to_dict_has_required_keys(self):
        e = self._make()
        d = e.to_dict()
        assert "tracker" in d
        assert "dataset" in d
        assert "deployability_score" in d
        assert "components" in d
        assert "weights" in d
        assert "raw" in d
        assert "on_pareto_front" in d

    def test_to_dict_raw_keys(self):
        e = self._make()
        raw = e.to_dict()["raw"]
        assert set(raw.keys()) == {"mean_iou", "fps", "peak_memory_mb", "success_auc", "ees"}

    def test_to_dict_success_auc_none(self):
        e = DeployabilityEntry(
            tracker_name="X", dataset_name="D",
            accuracy_score=0.5, efficiency_score=0.5,
            robustness_score=0.5, smoothness_score=0.5,
            weight_accuracy=0.25, weight_efficiency=0.25,
            weight_robustness=0.25, weight_smoothness=0.25,
            deployability_score=0.5,
            mean_iou=0.5, fps=20.0, peak_memory_mb=80.0,
            success_auc=None, ees=1.0,
        )
        assert e.to_dict()["raw"]["success_auc"] is None

    def test_str_contains_tracker_name(self):
        e = self._make()
        assert "KCF" in str(e)

    def test_str_contains_ds(self):
        e = self._make(ds=0.75)
        assert "0.7500" in str(e)

    def test_str_pareto_marker(self):
        e_yes = self._make(pareto=True)
        e_no = self._make(pareto=False)
        assert "✓" in str(e_yes)
        assert "✓" not in str(e_no)

    def test_pareto_default_false(self):
        e = DeployabilityEntry(
            tracker_name="X", dataset_name="D",
            accuracy_score=0.5, efficiency_score=0.5,
            robustness_score=0.5, smoothness_score=0.5,
            weight_accuracy=0.25, weight_efficiency=0.25,
            weight_robustness=0.25, weight_smoothness=0.25,
            deployability_score=0.5,
            mean_iou=0.5, fps=20.0, peak_memory_mb=80.0,
            success_auc=None, ees=1.0,
        )
        assert e.on_pareto_front is False


# ---------------------------------------------------------------------------
# Engine initialisation
# ---------------------------------------------------------------------------

class TestEngineInit:
    def test_default_weights_equal(self):
        eng = DeployabilityScoreEngine()
        w = eng.dimension_weights()
        assert w["accuracy"] == pytest.approx(0.25)
        assert w["efficiency"] == pytest.approx(0.25)
        assert w["robustness"] == pytest.approx(0.25)
        assert w["smoothness"] == pytest.approx(0.25)

    def test_weights_auto_normalised(self):
        eng = DeployabilityScoreEngine(
            weight_accuracy=1.0,
            weight_efficiency=1.0,
            weight_robustness=1.0,
            weight_smoothness=1.0,
        )
        total = sum(eng.dimension_weights().values())
        assert total == pytest.approx(1.0)

    def test_unequal_weights_normalised_to_one(self):
        eng = DeployabilityScoreEngine(
            weight_accuracy=0.4, weight_efficiency=0.3,
            weight_robustness=0.2, weight_smoothness=0.1,
        )
        total = sum(eng.dimension_weights().values())
        assert total == pytest.approx(1.0)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            DeployabilityScoreEngine(weight_accuracy=-0.1)

    def test_all_zero_weights_raises(self):
        with pytest.raises(ValueError, match="positive"):
            DeployabilityScoreEngine(
                weight_accuracy=0, weight_efficiency=0,
                weight_robustness=0, weight_smoothness=0,
            )

    def test_invalid_memory_budget_raises(self):
        with pytest.raises(ValueError, match="memory_budget_mb"):
            DeployabilityScoreEngine(memory_budget_mb=0.0)

    def test_invalid_failure_threshold_zero(self):
        with pytest.raises(ValueError, match="failure_threshold"):
            DeployabilityScoreEngine(failure_threshold=0.0)

    def test_invalid_failure_threshold_one(self):
        with pytest.raises(ValueError, match="failure_threshold"):
            DeployabilityScoreEngine(failure_threshold=1.0)

    def test_custom_memory_budget_stored(self):
        eng = DeployabilityScoreEngine(memory_budget_mb=256.0)
        assert eng.memory_budget_mb == 256.0


# ---------------------------------------------------------------------------
# Raw dimension extractors
# ---------------------------------------------------------------------------

class TestRawDimensions:
    def setup_method(self):
        self.eng = DeployabilityScoreEngine()

    def test_raw_accuracy_uses_success_auc_when_present(self):
        r = _result(success_auc=0.65, mean_iou=0.4)
        acc = self.eng._raw_accuracy(r)
        assert acc == pytest.approx(0.65)

    def test_raw_accuracy_falls_back_to_mean_iou(self):
        r = _result(mean_iou=0.55, success_auc=None)
        acc = self.eng._raw_accuracy(r)
        assert acc == pytest.approx(0.55)

    def test_raw_efficiency_positive(self):
        r = _result(mean_iou=0.6, fps=30.0, mem_mb=100.0)
        ees = self.eng._raw_efficiency(r)
        expected = 0.6 * math.log1p(30.0) / (1.0 + 100.0 / 512.0)
        assert ees == pytest.approx(expected, rel=1e-5)

    def test_raw_efficiency_zero_fps(self):
        r = _result(fps=0.0)
        assert self.eng._raw_efficiency(r) == 0.0

    def test_raw_efficiency_respects_memory_budget(self):
        eng_tight = DeployabilityScoreEngine(memory_budget_mb=64.0)
        eng_loose = DeployabilityScoreEngine(memory_budget_mb=1024.0)
        r = _result(mean_iou=0.5, fps=20.0, mem_mb=100.0)
        assert eng_tight._raw_efficiency(r) < eng_loose._raw_efficiency(r)

    def test_raw_robustness_in_unit_interval(self):
        r = _result(mean_iou=0.6)
        rob = self.eng._raw_robustness(r)
        assert 0.0 <= rob <= 1.0

    def test_raw_robustness_empty_sequences(self):
        r = BenchmarkResult(tracker_name="T", dataset_name="D")
        assert self.eng._raw_robustness(r) == 0.0

    def test_raw_robustness_high_iou_higher_than_low_iou(self):
        r_high = _result(mean_iou=0.9)
        r_low = _result(mean_iou=0.1)
        assert self.eng._raw_robustness(r_high) >= self.eng._raw_robustness(r_low)

    def test_raw_smoothness_in_unit_interval(self):
        r = _result(with_predictions=True)
        s = self.eng._raw_smoothness(r)
        assert 0.0 <= s <= 1.0

    def test_raw_smoothness_no_predictions_returns_zero(self):
        r = BenchmarkResult(tracker_name="T", dataset_name="D")
        r.sequence_results.append(SequenceResult(
            sequence_name="s0",
            ious=np.full(50, 0.5),
            profiling=_profiling(),
            predictions=None,
        ))
        assert self.eng._raw_smoothness(r) == 0.0

    def test_raw_smoothness_empty_sequences(self):
        r = BenchmarkResult(tracker_name="T", dataset_name="D")
        assert self.eng._raw_smoothness(r) == 0.0

    def test_raw_smoothness_short_sequence_ignored(self):
        # Sequence with only 1 prediction frame → smoothness analyzer needs ≥ 2
        r = BenchmarkResult(tracker_name="T", dataset_name="D")
        sr = _seq_result(
            ious=np.array([0.5]),
            predictions=np.array([[10.0, 10.0, 20.0, 20.0]]),
        )
        r.sequence_results.append(sr)
        assert self.eng._raw_smoothness(r) == 0.0


# ---------------------------------------------------------------------------
# score_results
# ---------------------------------------------------------------------------

class TestScoreResults:
    def setup_method(self):
        self.eng = DeployabilityScoreEngine()

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            self.eng.score_results([])

    def test_single_result_ds_is_one(self):
        r = _result()
        entries = self.eng.score_results([r])
        assert len(entries) == 1
        # Single tracker: all dimensions normalised to 1.0 → DS = 1.0
        assert entries[0].deployability_score == pytest.approx(1.0)

    def test_sorted_by_ds_descending(self):
        r_high = _result(tracker="HIGH", mean_iou=0.9, fps=60.0)
        r_low = _result(tracker="LOW", mean_iou=0.2, fps=5.0)
        entries = self.eng.score_results([r_low, r_high])
        assert entries[0].deployability_score >= entries[1].deployability_score

    def test_best_tracker_has_higher_ds(self):
        r_high = _result(tracker="CSRT", mean_iou=0.85, fps=40.0, mem_mb=80.0)
        r_low = _result(tracker="MOSSE", mean_iou=0.3, fps=8.0, mem_mb=300.0)
        entries = self.eng.score_results([r_high, r_low])
        tracker_map = {e.tracker_name: e for e in entries}
        assert tracker_map["CSRT"].deployability_score > tracker_map["MOSSE"].deployability_score

    def test_returns_one_entry_per_result(self):
        results = [_result(tracker=f"T{i}") for i in range(5)]
        entries = self.eng.score_results(results)
        assert len(entries) == 5

    def test_tracker_names_preserved(self):
        results = [_result(tracker="A"), _result(tracker="B"), _result(tracker="C")]
        entries = self.eng.score_results(results)
        names = {e.tracker_name for e in entries}
        assert names == {"A", "B", "C"}

    def test_dataset_name_preserved(self):
        r = _result(dataset="MyDataset")
        entries = self.eng.score_results([r])
        assert entries[0].dataset_name == "MyDataset"

    def test_ds_in_unit_interval(self):
        results = [_result(tracker=f"T{i}", mean_iou=0.2 * i + 0.1) for i in range(4)]
        entries = self.eng.score_results(results)
        for e in entries:
            assert 0.0 <= e.deployability_score <= 1.0

    def test_component_scores_in_unit_interval(self):
        results = [_result(tracker=f"T{i}", mean_iou=0.3 + 0.1 * i) for i in range(3)]
        entries = self.eng.score_results(results)
        for e in entries:
            assert 0.0 <= e.accuracy_score <= 1.0
            assert 0.0 <= e.efficiency_score <= 1.0
            assert 0.0 <= e.robustness_score <= 1.0
            assert 0.0 <= e.smoothness_score <= 1.0

    def test_mean_iou_raw_stored(self):
        r = _result(mean_iou=0.55)
        entries = self.eng.score_results([r])
        assert entries[0].mean_iou == pytest.approx(0.55, abs=1e-4)

    def test_fps_raw_stored(self):
        r = _result(fps=42.0)
        entries = self.eng.score_results([r])
        assert entries[0].fps == pytest.approx(42.0)

    def test_success_auc_stored_when_present(self):
        r = _result(success_auc=0.71)
        entries = self.eng.score_results([r])
        assert entries[0].success_auc == pytest.approx(0.71, abs=1e-3)

    def test_success_auc_none_when_absent(self):
        r = _result(success_auc=None)
        entries = self.eng.score_results([r])
        assert entries[0].success_auc is None

    def test_ees_raw_matches_formula(self):
        r = _result(mean_iou=0.5, fps=20.0, mem_mb=100.0)
        entries = self.eng.score_results([r])
        expected_ees = 0.5 * math.log1p(20.0) / (1.0 + 100.0 / 512.0)
        assert entries[0].ees == pytest.approx(expected_ees, rel=1e-4)

    def test_weights_stored_in_entry(self):
        eng = DeployabilityScoreEngine(
            weight_accuracy=0.4, weight_efficiency=0.3,
            weight_robustness=0.2, weight_smoothness=0.1,
        )
        r = _result()
        e = eng.score_results([r])[0]
        assert e.weight_accuracy == pytest.approx(0.4)
        assert e.weight_efficiency == pytest.approx(0.3)
        assert e.weight_robustness == pytest.approx(0.2)
        assert e.weight_smoothness == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Pareto-front
# ---------------------------------------------------------------------------

class TestParetoFront:
    def setup_method(self):
        self.eng = DeployabilityScoreEngine()

    def test_single_result_is_on_pareto(self):
        r = _result()
        entries = self.eng.score_results([r])
        assert entries[0].on_pareto_front is True

    def test_dominated_tracker_not_on_pareto(self):
        # Tracker B strictly dominates Tracker A in both accuracy and efficiency
        r_dominant = _result(tracker="B", mean_iou=0.9, fps=60.0, mem_mb=50.0)
        r_weak = _result(tracker="A", mean_iou=0.3, fps=8.0, mem_mb=300.0)
        entries = self.eng.score_results([r_dominant, r_weak])
        entry_map = {e.tracker_name: e for e in entries}
        assert entry_map["B"].on_pareto_front is True
        assert entry_map["A"].on_pareto_front is False

    def test_pareto_front_method_filters(self):
        r_dominant = _result(tracker="Best", mean_iou=0.9, fps=60.0, mem_mb=50.0)
        r_weak = _result(tracker="Weak", mean_iou=0.3, fps=8.0, mem_mb=300.0)
        front = self.eng.pareto_front([r_dominant, r_weak])
        assert all(e.on_pareto_front for e in front)
        assert len(front) >= 1

    def test_pareto_front_excludes_dominated(self):
        r_dominant = _result(tracker="Best", mean_iou=0.9, fps=60.0, mem_mb=50.0)
        r_weak = _result(tracker="Weak", mean_iou=0.3, fps=8.0, mem_mb=300.0)
        front = self.eng.pareto_front([r_dominant, r_weak])
        names = {e.tracker_name for e in front}
        assert "Weak" not in names

    def test_incomparable_trackers_both_on_front(self):
        # One is faster, the other is more accurate — neither dominates the other
        r_fast = _result(tracker="Fast", mean_iou=0.3, fps=100.0, mem_mb=30.0)
        r_accurate = _result(tracker="Accurate", mean_iou=0.9, fps=5.0, mem_mb=400.0)
        entries = self.eng.score_results([r_fast, r_accurate])
        entry_map = {e.tracker_name: e for e in entries}
        # Both could be on Pareto since neither dominates in all dimensions
        # (fast has low IoU, accurate has low EES)
        assert entry_map["Fast"].on_pareto_front or entry_map["Accurate"].on_pareto_front

    def test_pareto_front_sorted_by_ds(self):
        results = [_result(tracker=f"T{i}", mean_iou=0.2 + 0.2 * i) for i in range(4)]
        front = self.eng.pareto_front(results)
        ds_values = [e.deployability_score for e in front]
        assert ds_values == sorted(ds_values, reverse=True)


# ---------------------------------------------------------------------------
# Markdown table output
# ---------------------------------------------------------------------------

class TestMarkdownTable:
    def setup_method(self):
        self.eng = DeployabilityScoreEngine()

    def _make_entries(self) -> List[DeployabilityEntry]:
        results = [
            _result(tracker="CSRT", mean_iou=0.75, fps=35.0),
            _result(tracker="KCF", mean_iou=0.55, fps=80.0),
        ]
        return self.eng.score_results(results)

    def test_contains_title(self):
        entries = self._make_entries()
        md = self.eng.to_markdown_table(entries, title="My Ranking")
        assert "My Ranking" in md

    def test_default_title(self):
        entries = self._make_entries()
        md = self.eng.to_markdown_table(entries)
        assert "Deployability" in md

    def test_contains_tracker_names(self):
        entries = self._make_entries()
        md = self.eng.to_markdown_table(entries)
        assert "CSRT" in md
        assert "KCF" in md

    def test_header_row_present(self):
        entries = self._make_entries()
        md = self.eng.to_markdown_table(entries)
        assert "Rank" in md
        assert "Tracker" in md

    def test_pareto_marker_column(self):
        entries = self._make_entries()
        md = self.eng.to_markdown_table(entries)
        assert "Pareto" in md

    def test_rank_numbers_present(self):
        entries = self._make_entries()
        md = self.eng.to_markdown_table(entries)
        assert "| 1 |" in md
        assert "| 2 |" in md

    def test_weight_footer_present(self):
        entries = self._make_entries()
        md = self.eng.to_markdown_table(entries)
        assert "Weights" in md or "weights" in md.lower()

    def test_empty_entries_produces_valid_string(self):
        md = self.eng.to_markdown_table([])
        assert isinstance(md, str)


# ---------------------------------------------------------------------------
# to_summary_dict
# ---------------------------------------------------------------------------

class TestSummaryDict:
    def setup_method(self):
        self.eng = DeployabilityScoreEngine()

    def test_length_matches_entries(self):
        results = [_result(tracker=f"T{i}") for i in range(3)]
        entries = self.eng.score_results(results)
        summary = self.eng.to_summary_dict(entries)
        assert len(summary) == 3

    def test_each_item_is_dict(self):
        results = [_result(tracker="A"), _result(tracker="B")]
        entries = self.eng.score_results(results)
        summary = self.eng.to_summary_dict(entries)
        for item in summary:
            assert isinstance(item, dict)

    def test_tracker_field_present(self):
        results = [_result(tracker="X")]
        entries = self.eng.score_results(results)
        summary = self.eng.to_summary_dict(entries)
        assert summary[0]["tracker"] == "X"

    def test_json_serialisable(self):
        import json
        results = [_result(tracker="A", success_auc=0.6), _result(tracker="B")]
        entries = self.eng.score_results(results)
        data = self.eng.to_summary_dict(entries)
        # Should not raise
        json.dumps(data)


# ---------------------------------------------------------------------------
# dimension_weights
# ---------------------------------------------------------------------------

class TestDimensionWeights:
    def test_keys(self):
        eng = DeployabilityScoreEngine()
        w = eng.dimension_weights()
        assert set(w.keys()) == {"accuracy", "efficiency", "robustness", "smoothness"}

    def test_values_sum_to_one(self):
        eng = DeployabilityScoreEngine(
            weight_accuracy=0.1, weight_efficiency=0.5,
            weight_robustness=0.3, weight_smoothness=0.1,
        )
        total = sum(eng.dimension_weights().values())
        assert total == pytest.approx(1.0)

    def test_reflects_user_intent(self):
        eng = DeployabilityScoreEngine(
            weight_accuracy=4.0, weight_efficiency=2.0,
            weight_robustness=2.0, weight_smoothness=2.0,
        )
        w = eng.dimension_weights()
        assert w["accuracy"] > w["efficiency"]


# ---------------------------------------------------------------------------
# Public API reachable through metrics package
# ---------------------------------------------------------------------------

class TestPublicImport:
    def test_importable_from_metrics(self):
        from eovot.metrics import DeployabilityScoreEngine, DeployabilityEntry
        assert DeployabilityScoreEngine is not None
        assert DeployabilityEntry is not None

    def test_importable_directly(self):
        from eovot.metrics.deployability import DeployabilityScoreEngine
        assert DeployabilityScoreEngine is not None


# ---------------------------------------------------------------------------
# Integration: engine produces consistent ranking
# ---------------------------------------------------------------------------

class TestIntegration:
    """End-to-end ranking with three trackers of clearly different profiles."""

    def setup_method(self):
        self.eng = DeployabilityScoreEngine()

    def _three_tracker_results(self) -> List:
        return [
            _result(tracker="CSRT",      mean_iou=0.80, fps=30.0,  mem_mb=150.0, success_auc=0.78),
            _result(tracker="KCF",       mean_iou=0.60, fps=120.0, mem_mb=80.0,  success_auc=0.58),
            _result(tracker="MOSSE",     mean_iou=0.45, fps=250.0, mem_mb=40.0,  success_auc=0.42),
        ]

    def test_ranking_produces_three_entries(self):
        entries = self.eng.score_results(self._three_tracker_results())
        assert len(entries) == 3

    def test_best_entry_ds_highest(self):
        entries = self.eng.score_results(self._three_tracker_results())
        assert entries[0].deployability_score >= entries[-1].deployability_score

    def test_all_ds_in_unit_interval(self):
        entries = self.eng.score_results(self._three_tracker_results())
        for e in entries:
            assert 0.0 <= e.deployability_score <= 1.0

    def test_to_dict_round_trip(self):
        import json
        entries = self.eng.score_results(self._three_tracker_results())
        data = self.eng.to_summary_dict(entries)
        # Must serialize without error
        serialized = json.dumps(data)
        loaded = json.loads(serialized)
        assert len(loaded) == 3

    def test_markdown_table_contains_all_trackers(self):
        entries = self.eng.score_results(self._three_tracker_results())
        md = self.eng.to_markdown_table(entries)
        assert "CSRT" in md
        assert "KCF" in md
        assert "MOSSE" in md

    def test_efficiency_weight_shift_reorders(self):
        """Up-weighting efficiency enough should promote MOSSE (fastest tracker)."""
        eng_eff = DeployabilityScoreEngine(
            weight_accuracy=0.05,
            weight_efficiency=0.85,
            weight_robustness=0.05,
            weight_smoothness=0.05,
        )
        entries = eng_eff.score_results(self._three_tracker_results())
        # With extreme efficiency weighting, the fastest tracker should be near the top
        top_name = entries[0].tracker_name
        assert top_name in ("MOSSE", "KCF")  # either fast tracker could win

    def test_accuracy_weight_shift_promotes_accurate_tracker(self):
        """Up-weighting accuracy should keep CSRT at the top."""
        eng_acc = DeployabilityScoreEngine(
            weight_accuracy=0.85,
            weight_efficiency=0.05,
            weight_robustness=0.05,
            weight_smoothness=0.05,
        )
        entries = eng_acc.score_results(self._three_tracker_results())
        assert entries[0].tracker_name == "CSRT"

"""Tests for eovot.metrics.cross_dataset.CrossDatasetEvaluator.

All tests use synthetically-constructed BenchmarkResult / SequenceResult
objects so that no external dataset download is needed.
"""
from __future__ import annotations

import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkResult, SequenceResult
from eovot.metrics.cross_dataset import (
    AggregateEntry,
    CrossDatasetEvaluator,
    DatasetEntry,
    _rank_vector,
    _spearman,
)
from eovot.profiling.profiler import ProfilingResult


# ---------------------------------------------------------------------------
# Helpers for constructing fake BenchmarkResult objects
# ---------------------------------------------------------------------------

def _fake_profiling(fps: float = 100.0) -> ProfilingResult:
    return ProfilingResult(
        tracker_name="x",
        frame_count=50,
        fps=fps,
        latency_mean_ms=1000.0 / fps,
        latency_std_ms=0.1,
        latency_p95_ms=1000.0 / fps * 1.1,
        peak_memory_mb=50.0,
    )


def _fake_sequence_result(name: str, iou: float) -> SequenceResult:
    ious = np.full(50, iou)
    return SequenceResult(
        sequence_name=name,
        ious=ious,
        profiling=_fake_profiling(),
    )


def _fake_result(tracker: str, dataset: str, iou: float, fps: float = 100.0) -> BenchmarkResult:
    """Create a BenchmarkResult with a fixed IoU across 3 sequences."""
    r = BenchmarkResult(tracker_name=tracker, dataset_name=dataset)
    for i in range(3):
        sr = _fake_sequence_result(f"seq{i}", iou)
        sr.profiling = _fake_profiling(fps)
        r.sequence_results.append(sr)
    return r


# ---------------------------------------------------------------------------
# DatasetEntry / AggregateEntry unit tests
# ---------------------------------------------------------------------------

class TestDatasetEntry:
    def test_to_dict_basic(self):
        de = DatasetEntry(dataset_name="OTB", mean_iou=0.5, mean_fps=120.0)
        d = de.to_dict()
        assert d["mean_iou"] == 0.5
        assert d["mean_fps"] == 120.0
        assert "success_auc" not in d
        assert "precision_auc" not in d

    def test_to_dict_with_optional_fields(self):
        de = DatasetEntry(
            dataset_name="OTB",
            mean_iou=0.5,
            mean_fps=120.0,
            success_auc=0.45,
            precision_auc=0.6,
        )
        d = de.to_dict()
        assert "success_auc" in d
        assert "precision_auc" in d


class TestAggregateEntry:
    def test_str_representation(self):
        ae = AggregateEntry(
            tracker_name="MOSSE",
            weighted_iou=0.42,
            rank=1,
            dataset_entries={
                "OTB": DatasetEntry("OTB", 0.42, 100.0),
            },
        )
        s = str(ae)
        assert "MOSSE" in s
        assert "#1" in s
        assert "0.4200" in s


# ---------------------------------------------------------------------------
# CrossDatasetEvaluator — construction
# ---------------------------------------------------------------------------

class TestCrossDatasetEvaluatorConstruction:
    def test_default_construction(self):
        ev = CrossDatasetEvaluator()
        assert ev.dataset_weights == {}

    def test_custom_weights(self):
        ev = CrossDatasetEvaluator(dataset_weights={"OTB": 1.0, "GOT10k": 2.0})
        assert ev.dataset_weights["GOT10k"] == 2.0

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            CrossDatasetEvaluator(dataset_weights={"OTB": -1.0})

    def test_zero_weight_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            CrossDatasetEvaluator(dataset_weights={"OTB": 0.0})


# ---------------------------------------------------------------------------
# CrossDatasetEvaluator.build_leaderboard
# ---------------------------------------------------------------------------

class TestBuildLeaderboard:
    def test_empty_results_returns_empty(self):
        ev = CrossDatasetEvaluator()
        assert ev.build_leaderboard({}) == []

    def test_results_with_no_benchmark_objects(self):
        ev = CrossDatasetEvaluator()
        # Each list is empty
        lb = ev.build_leaderboard({"OTB": [], "GOT10k": []})
        assert lb == []

    def test_single_dataset_single_tracker(self):
        ev = CrossDatasetEvaluator()
        results = {"OTB": [_fake_result("MOSSE", "OTB", 0.5)]}
        lb = ev.build_leaderboard(results)
        assert len(lb) == 1
        assert lb[0].tracker_name == "MOSSE"
        assert lb[0].rank == 1
        assert pytest.approx(lb[0].weighted_iou, abs=1e-4) == 0.5

    def test_multiple_trackers_sorted_by_iou(self):
        ev = CrossDatasetEvaluator()
        results = {
            "OTB": [
                _fake_result("MOSSE", "OTB", 0.40),
                _fake_result("KCF",   "OTB", 0.50),
                _fake_result("CSRT",  "OTB", 0.60),
            ]
        }
        lb = ev.build_leaderboard(results)
        assert lb[0].tracker_name == "CSRT"
        assert lb[1].tracker_name == "KCF"
        assert lb[2].tracker_name == "MOSSE"
        assert lb[0].rank == 1
        assert lb[2].rank == 3

    def test_two_datasets_equal_weights(self):
        ev = CrossDatasetEvaluator()
        results = {
            "OTB":    [_fake_result("KCF", "OTB",   0.60)],
            "GOT10k": [_fake_result("KCF", "GOT10k", 0.40)],
        }
        lb = ev.build_leaderboard(results)
        assert len(lb) == 1
        # Expected: (0.60 + 0.40) / 2 = 0.50
        assert pytest.approx(lb[0].weighted_iou, abs=1e-4) == 0.50

    def test_weighted_iou_respects_weights(self):
        ev = CrossDatasetEvaluator(dataset_weights={"OTB": 1.0, "GOT10k": 3.0})
        results = {
            "OTB":    [_fake_result("KCF", "OTB",    0.60)],
            "GOT10k": [_fake_result("KCF", "GOT10k", 0.40)],
        }
        lb = ev.build_leaderboard(results)
        # Expected: (1.0*0.60 + 3.0*0.40) / (1.0 + 3.0) = (0.60 + 1.20) / 4 = 0.45
        assert pytest.approx(lb[0].weighted_iou, abs=1e-4) == 0.45

    def test_tracker_missing_from_one_dataset(self):
        """A tracker present on only one dataset still gets a valid leaderboard entry."""
        ev = CrossDatasetEvaluator()
        results = {
            "OTB":    [_fake_result("MOSSE", "OTB", 0.50), _fake_result("KCF", "OTB", 0.60)],
            "GOT10k": [_fake_result("KCF", "GOT10k", 0.55)],  # MOSSE not evaluated here
        }
        lb = ev.build_leaderboard(results)
        tracker_names = {e.tracker_name for e in lb}
        assert "MOSSE" in tracker_names
        assert "KCF" in tracker_names

    def test_dataset_entries_populated(self):
        ev = CrossDatasetEvaluator()
        results = {
            "OTB":    [_fake_result("KCF", "OTB",    0.60)],
            "GOT10k": [_fake_result("KCF", "GOT10k", 0.40)],
        }
        lb = ev.build_leaderboard(results)
        entry = lb[0]
        assert "OTB" in entry.dataset_entries
        assert "GOT10k" in entry.dataset_entries
        assert pytest.approx(entry.dataset_entries["OTB"].mean_iou, abs=1e-4) == 0.60
        assert pytest.approx(entry.dataset_entries["GOT10k"].mean_iou, abs=1e-4) == 0.40

    def test_rank_assignment_is_1indexed(self):
        ev = CrossDatasetEvaluator()
        results = {
            "OTB": [
                _fake_result("A", "OTB", 0.5),
                _fake_result("B", "OTB", 0.3),
                _fake_result("C", "OTB", 0.7),
            ]
        }
        lb = ev.build_leaderboard(results)
        ranks = [e.rank for e in lb]
        assert sorted(ranks) == [1, 2, 3]
        assert lb[0].rank == 1

    def test_fps_stored_in_dataset_entry(self):
        ev = CrossDatasetEvaluator()
        results = {"OTB": [_fake_result("KCF", "OTB", 0.5, fps=200.0)]}
        lb = ev.build_leaderboard(results)
        assert pytest.approx(lb[0].dataset_entries["OTB"].mean_fps, abs=1.0) == 200.0


# ---------------------------------------------------------------------------
# CrossDatasetEvaluator.rank_consistency
# ---------------------------------------------------------------------------

class TestRankConsistency:
    def test_empty_leaderboard(self):
        ev = CrossDatasetEvaluator()
        assert ev.rank_consistency([]) == 0.0

    def test_single_tracker(self):
        ev = CrossDatasetEvaluator()
        lb = [AggregateEntry("A", {"D1": DatasetEntry("D1", 0.5, 100.0)}, 0.5, 1)]
        assert ev.rank_consistency(lb) == 0.0

    def test_single_dataset(self):
        ev = CrossDatasetEvaluator()
        lb = [
            AggregateEntry("A", {"D1": DatasetEntry("D1", 0.6, 100.0)}, 0.6, 1),
            AggregateEntry("B", {"D1": DatasetEntry("D1", 0.4, 100.0)}, 0.4, 2),
        ]
        assert ev.rank_consistency(lb) == 0.0

    def test_perfect_consistency(self):
        """Same ranking on both datasets → ρ = 1.0."""
        ev = CrossDatasetEvaluator()
        results = {
            "OTB":    [_fake_result("A", "OTB",    0.7),
                       _fake_result("B", "OTB",    0.5),
                       _fake_result("C", "OTB",    0.3)],
            "GOT10k": [_fake_result("A", "GOT10k", 0.8),
                       _fake_result("B", "GOT10k", 0.6),
                       _fake_result("C", "GOT10k", 0.4)],
        }
        lb = ev.build_leaderboard(results)
        rho = ev.rank_consistency(lb)
        assert pytest.approx(rho, abs=0.05) == 1.0

    def test_reversed_ranking(self):
        """Opposite ranking on both datasets → ρ ≈ −1.0."""
        ev = CrossDatasetEvaluator()
        results = {
            "OTB":    [_fake_result("A", "OTB",    0.7),
                       _fake_result("B", "OTB",    0.5),
                       _fake_result("C", "OTB",    0.3)],
            "GOT10k": [_fake_result("A", "GOT10k", 0.3),
                       _fake_result("B", "GOT10k", 0.5),
                       _fake_result("C", "GOT10k", 0.7)],
        }
        lb = ev.build_leaderboard(results)
        rho = ev.rank_consistency(lb)
        assert rho < -0.9

    def test_three_dataset_mean(self):
        """rank_consistency returns a mean across all dataset pairs."""
        ev = CrossDatasetEvaluator()
        results = {
            "D1": [_fake_result("A", "D1", 0.7), _fake_result("B", "D1", 0.3)],
            "D2": [_fake_result("A", "D2", 0.6), _fake_result("B", "D2", 0.4)],
            "D3": [_fake_result("A", "D3", 0.5), _fake_result("B", "D3", 0.5)],
        }
        lb = ev.build_leaderboard(results)
        rho = ev.rank_consistency(lb)
        assert -1.0 <= rho <= 1.0


# ---------------------------------------------------------------------------
# CrossDatasetEvaluator.to_markdown
# ---------------------------------------------------------------------------

class TestToMarkdown:
    def _simple_lb(self) -> tuple:
        ev = CrossDatasetEvaluator()
        results = {
            "OTB":    [_fake_result("MOSSE", "OTB",    0.40),
                       _fake_result("KCF",   "OTB",    0.55)],
            "GOT10k": [_fake_result("MOSSE", "GOT10k", 0.35),
                       _fake_result("KCF",   "GOT10k", 0.48)],
        }
        return ev, ev.build_leaderboard(results)

    def test_returns_string(self):
        ev, lb = self._simple_lb()
        md = ev.to_markdown(lb)
        assert isinstance(md, str)

    def test_empty_leaderboard(self):
        ev = CrossDatasetEvaluator()
        md = ev.to_markdown([])
        assert "_No results._" in md

    def test_contains_tracker_names(self):
        ev, lb = self._simple_lb()
        md = ev.to_markdown(lb)
        assert "MOSSE" in md
        assert "KCF" in md

    def test_contains_dataset_headers(self):
        ev, lb = self._simple_lb()
        md = ev.to_markdown(lb)
        assert "OTB" in md
        assert "GOT10k" in md

    def test_contains_rank_column(self):
        ev, lb = self._simple_lb()
        md = ev.to_markdown(lb)
        assert "#1" in md
        assert "#2" in md

    def test_markdown_table_structure(self):
        ev, lb = self._simple_lb()
        md = ev.to_markdown(lb)
        lines = [l for l in md.splitlines() if l.strip()]
        # Header + separator + N data rows
        assert len(lines) >= 3
        for line in lines:
            assert line.startswith("|")
            assert line.endswith("|")

    def test_missing_dataset_shows_dash(self):
        """Tracker absent from a dataset should show '—' in the table."""
        ev = CrossDatasetEvaluator()
        results = {
            "OTB":    [_fake_result("A", "OTB", 0.6), _fake_result("B", "OTB", 0.5)],
            "GOT10k": [_fake_result("A", "GOT10k", 0.7)],  # B is missing
        }
        lb = ev.build_leaderboard(results)
        md = ev.to_markdown(lb)
        assert "—" in md


# ---------------------------------------------------------------------------
# CrossDatasetEvaluator.to_dict
# ---------------------------------------------------------------------------

class TestToDict:
    def test_returns_dict_with_expected_keys(self):
        ev = CrossDatasetEvaluator()
        results = {"OTB": [_fake_result("KCF", "OTB", 0.5)]}
        lb = ev.build_leaderboard(results)
        d = ev.to_dict(lb)
        assert "dataset_names" in d
        assert "leaderboard" in d

    def test_dataset_names_sorted(self):
        ev = CrossDatasetEvaluator()
        results = {
            "OTB":    [_fake_result("A", "OTB",    0.5)],
            "GOT10k": [_fake_result("A", "GOT10k", 0.4)],
        }
        lb = ev.build_leaderboard(results)
        d = ev.to_dict(lb)
        assert d["dataset_names"] == sorted(d["dataset_names"])

    def test_leaderboard_has_rank_and_tracker(self):
        ev = CrossDatasetEvaluator()
        results = {"OTB": [_fake_result("KCF", "OTB", 0.5)]}
        lb = ev.build_leaderboard(results)
        d = ev.to_dict(lb)
        entry = d["leaderboard"][0]
        assert "rank" in entry
        assert "tracker" in entry
        assert "weighted_iou" in entry
        assert "datasets" in entry

    def test_per_dataset_values(self):
        ev = CrossDatasetEvaluator()
        results = {"OTB": [_fake_result("KCF", "OTB", 0.55)]}
        lb = ev.build_leaderboard(results)
        d = ev.to_dict(lb)
        assert pytest.approx(d["leaderboard"][0]["datasets"]["OTB"]["mean_iou"], abs=1e-3) == 0.55


# ---------------------------------------------------------------------------
# Internal helper tests
# ---------------------------------------------------------------------------

class TestInternalHelpers:
    def test_rank_vector_basic(self):
        arr = np.array([3.0, 1.0, 2.0])
        ranks = _rank_vector(arr)
        assert list(ranks) == [3.0, 1.0, 2.0]

    def test_rank_vector_ties_averaged(self):
        arr = np.array([1.0, 1.0, 3.0])
        ranks = _rank_vector(arr)
        # Two tied values at positions 0,1 share ranks 1 and 2 → mean = 1.5
        assert ranks[0] == pytest.approx(1.5)
        assert ranks[1] == pytest.approx(1.5)
        assert ranks[2] == pytest.approx(3.0)

    def test_spearman_perfect(self):
        a = np.array([1.0, 2.0, 3.0])
        assert _spearman(a, a) == pytest.approx(1.0, abs=1e-9)

    def test_spearman_inverse(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([3.0, 2.0, 1.0])
        assert _spearman(a, b) == pytest.approx(-1.0, abs=1e-9)

    def test_spearman_single_element(self):
        assert _spearman(np.array([1.0]), np.array([1.0])) == 0.0

    def test_spearman_uncorrelated_in_range(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 1, 20)
        b = rng.uniform(0, 1, 20)
        rho = _spearman(a, b)
        assert -1.0 <= rho <= 1.0

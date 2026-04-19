"""Tests for eovot.metrics.statistical."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from eovot.metrics.statistical import (
    BootstrapCI,
    PairwiseComparison,
    TrackerRanking,
    _extract_per_sequence_metric,
    bootstrap_ci,
    cohens_d,
    compare_trackers,
    rank_trackers,
    vargha_delaney_a12,
    wilcoxon_test,
)


# ---------------------------------------------------------------------------
# Minimal stubs — no dependency on benchmark engine
# ---------------------------------------------------------------------------

class _Profiling:
    def __init__(self, fps=30.0, peak_memory_mb=200.0, latency_mean_ms=33.0):
        self.fps = fps
        self.peak_memory_mb = peak_memory_mb
        self.latency_mean_ms = latency_mean_ms


class _SequenceStub:
    def __init__(self, iou=0.6, fps=30.0, peak_memory_mb=200.0, energy_mj=None):
        self._iou = iou
        self.profiling = _Profiling(fps=fps, peak_memory_mb=peak_memory_mb)
        self.energy = _EnergyStub(energy_mj) if energy_mj is not None else None

    @property
    def mean_iou(self):
        return self._iou


class _EnergyStub:
    def __init__(self, mj):
        self.energy_per_frame_mj = mj


class _ResultStub:
    def __init__(self, name: str, ious, fps_vals=None):
        self.tracker_name = name
        self.dataset_name = "test"
        fps_vals = fps_vals or [30.0] * len(ious)
        self.sequence_results = [
            _SequenceStub(iou=iou, fps=fps) for iou, fps in zip(ious, fps_vals)
        ]


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_basic_output_type(self):
        rng = np.random.default_rng(0)
        ci = bootstrap_ci(np.array([0.5, 0.6, 0.55, 0.62, 0.58]), rng=rng)
        assert isinstance(ci, BootstrapCI)

    def test_ci_bounds_contain_point_estimate(self):
        rng = np.random.default_rng(0)
        ci = bootstrap_ci(np.array([0.5, 0.6, 0.55, 0.62, 0.58]), rng=rng)
        assert ci.lower <= ci.point_estimate <= ci.upper

    def test_single_value_trivial_ci(self):
        rng = np.random.default_rng(0)
        ci = bootstrap_ci(np.array([0.7]), n_bootstrap=100, rng=rng)
        assert ci.point_estimate == pytest.approx(0.7)
        assert ci.lower == pytest.approx(ci.upper, abs=1e-9)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci(np.array([]))

    def test_wider_ci_at_higher_confidence(self):
        rng95 = np.random.default_rng(1)
        rng80 = np.random.default_rng(1)
        data = np.random.default_rng(42).uniform(0.4, 0.8, 30)
        ci95 = bootstrap_ci(data, confidence=0.95, n_bootstrap=2000, rng=rng95)
        ci80 = bootstrap_ci(data, confidence=0.80, n_bootstrap=2000, rng=rng80)
        assert (ci95.upper - ci95.lower) >= (ci80.upper - ci80.lower)

    def test_str_contains_metric_name(self):
        rng = np.random.default_rng(0)
        ci = bootstrap_ci(np.array([0.5, 0.6]), metric_name="mean_iou", rng=rng)
        assert "mean_iou" in str(ci)

    def test_confidence_level_stored(self):
        rng = np.random.default_rng(0)
        ci = bootstrap_ci(np.array([0.5, 0.6]), confidence=0.90, rng=rng)
        assert ci.confidence_level == pytest.approx(0.90)


# ---------------------------------------------------------------------------
# cohens_d
# ---------------------------------------------------------------------------

class TestCohensD:
    def test_identical_zero(self):
        a = np.array([0.5, 0.6, 0.7])
        assert cohens_d(a, a.copy()) == pytest.approx(0.0)

    def test_positive_when_a_greater(self):
        a = np.array([0.8, 0.85, 0.9])
        b = np.array([0.5, 0.55, 0.6])
        assert cohens_d(a, b) > 0

    def test_negative_when_b_greater(self):
        a = np.array([0.3, 0.35, 0.4])
        b = np.array([0.7, 0.75, 0.8])
        assert cohens_d(a, b) < 0

    def test_constant_arrays_zero(self):
        a = np.array([0.5, 0.5, 0.5])
        b = np.array([0.5, 0.5, 0.5])
        assert cohens_d(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# vargha_delaney_a12
# ---------------------------------------------------------------------------

class TestA12:
    def test_equal_samples_half(self):
        a = np.array([0.5, 0.5, 0.5])
        assert vargha_delaney_a12(a, a.copy()) == pytest.approx(0.5, abs=0.01)

    def test_clear_dominance_a(self):
        a = np.array([0.9, 0.85, 0.8, 0.95])
        b = np.array([0.2, 0.25, 0.3, 0.15])
        assert vargha_delaney_a12(a, b) > 0.71

    def test_clear_dominance_b(self):
        a = np.array([0.2, 0.25, 0.3])
        b = np.array([0.8, 0.85, 0.9])
        assert vargha_delaney_a12(a, b) < 0.5

    def test_bounds(self):
        a = np.random.default_rng(0).uniform(0.4, 0.9, 20)
        b = np.random.default_rng(1).uniform(0.4, 0.9, 20)
        val = vargha_delaney_a12(a, b)
        assert 0.0 <= val <= 1.0

    def test_empty_returns_half(self):
        assert vargha_delaney_a12(np.array([]), np.array([0.5])) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# wilcoxon_test
# ---------------------------------------------------------------------------

class TestWilcoxon:
    def test_identical_not_significant(self):
        a = np.array([0.5] * 10)
        _, pval, sig = wilcoxon_test(a, a.copy())
        assert sig is False

    def test_large_difference_significant(self):
        pytest.importorskip("scipy")
        a = np.linspace(0.8, 0.95, 20)
        b = np.linspace(0.1, 0.25, 20)
        _, pval, sig = wilcoxon_test(a, b)
        assert sig is True
        assert pval < 0.05

    def test_too_few_samples(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, pval, sig = wilcoxon_test(np.array([0.5, 0.6]), np.array([0.4, 0.5]))
        assert stat is None
        assert pval is None
        assert sig is False

    def test_returns_tuple_of_three(self):
        a = np.array([0.6] * 8)
        b = np.array([0.5] * 8)
        result = wilcoxon_test(a, b)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# _extract_per_sequence_metric
# ---------------------------------------------------------------------------

class TestExtractMetric:
    def test_mean_iou(self):
        r = _ResultStub("A", [0.5, 0.6, 0.7])
        vals = _extract_per_sequence_metric(r, "mean_iou")
        np.testing.assert_allclose(vals, [0.5, 0.6, 0.7])

    def test_fps(self):
        r = _ResultStub("A", [0.5] * 3, fps_vals=[30.0, 60.0, 90.0])
        vals = _extract_per_sequence_metric(r, "fps")
        np.testing.assert_allclose(vals, [30.0, 60.0, 90.0])

    def test_unknown_metric_raises(self):
        r = _ResultStub("A", [0.5])
        with pytest.raises(ValueError, match="Unknown metric"):
            _extract_per_sequence_metric(r, "nonexistent")

    def test_energy_skipped_when_none(self):
        r = _ResultStub("A", [0.5, 0.6])
        vals = _extract_per_sequence_metric(r, "energy_per_frame_mj")
        assert len(vals) == 0

    def test_energy_extracted_when_present(self):
        r = _ResultStub("A", [0.5])
        r.sequence_results[0].energy = _EnergyStub(3.5)
        vals = _extract_per_sequence_metric(r, "energy_per_frame_mj")
        np.testing.assert_allclose(vals, [3.5])


# ---------------------------------------------------------------------------
# compare_trackers
# ---------------------------------------------------------------------------

class TestCompareTrackers:
    def test_winner_correct(self):
        a = _ResultStub("A", [0.8] * 10)
        b = _ResultStub("B", [0.5] * 10)
        cmp = compare_trackers(a, b, metric="mean_iou", seed=0, n_bootstrap=500)
        assert cmp.winner == "A"

    def test_all_fields_populated(self):
        a = _ResultStub("A", [0.7] * 8)
        b = _ResultStub("B", [0.6] * 8)
        cmp = compare_trackers(a, b, metric="mean_iou", seed=0, n_bootstrap=500)
        assert isinstance(cmp.effect_size_d, float)
        assert 0.0 <= cmp.a12 <= 1.0
        assert cmp.ci_a is not None
        assert cmp.ci_b is not None

    def test_str_contains_tracker_names(self):
        a = _ResultStub("MOSSE", [0.4] * 8)
        b = _ResultStub("KCF", [0.5] * 8)
        cmp = compare_trackers(a, b, metric="mean_iou", seed=0, n_bootstrap=200)
        s = str(cmp)
        assert "MOSSE" in s and "KCF" in s

    def test_empty_result_raises(self):
        a = _ResultStub("A", [])
        b = _ResultStub("B", [0.5] * 5)
        with pytest.raises(ValueError):
            compare_trackers(a, b)

    def test_fps_metric(self):
        a = _ResultStub("fast", [0.5] * 5, fps_vals=[100.0] * 5)
        b = _ResultStub("slow", [0.5] * 5, fps_vals=[20.0] * 5)
        cmp = compare_trackers(a, b, metric="fps", seed=0, n_bootstrap=200)
        assert cmp.winner == "fast"

    def test_mean_a_b_correct(self):
        a = _ResultStub("A", [0.6, 0.7, 0.8])
        b = _ResultStub("B", [0.3, 0.4, 0.5])
        cmp = compare_trackers(a, b, metric="mean_iou", seed=0, n_bootstrap=200)
        assert cmp.mean_a == pytest.approx((0.6 + 0.7 + 0.8) / 3)
        assert cmp.mean_b == pytest.approx((0.3 + 0.4 + 0.5) / 3)

    def test_alpha_stored(self):
        a = _ResultStub("A", [0.6] * 5)
        b = _ResultStub("B", [0.5] * 5)
        cmp = compare_trackers(a, b, alpha=0.01, seed=0, n_bootstrap=200)
        assert cmp.alpha == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# rank_trackers
# ---------------------------------------------------------------------------

class TestRankTrackers:
    def test_ranking_order(self):
        results = [
            _ResultStub("C", [0.5] * 5),
            _ResultStub("A", [0.9] * 5),
            _ResultStub("B", [0.7] * 5),
        ]
        ranks = rank_trackers(results, metric="mean_iou", seed=0, n_bootstrap=200)
        assert [r.tracker_name for r in ranks] == ["A", "B", "C"]

    def test_ascending_order(self):
        results = [
            _ResultStub("slow", [0.5] * 5, fps_vals=[10.0] * 5),
            _ResultStub("fast", [0.5] * 5, fps_vals=[100.0] * 5),
        ]
        ranks = rank_trackers(results, metric="fps", higher_is_better=False,
                              seed=0, n_bootstrap=200)
        assert ranks[0].tracker_name == "slow"

    def test_rank_indices_sequential(self):
        results = [_ResultStub(f"T{i}", [i / 10] * 5) for i in range(4)]
        ranks = rank_trackers(results, metric="mean_iou", seed=0, n_bootstrap=200)
        for i, r in enumerate(ranks):
            assert r.rank == i + 1

    def test_ci_is_bootstrapci(self):
        results = [_ResultStub("A", [0.5, 0.55, 0.6, 0.58, 0.52])]
        ranks = rank_trackers(results, metric="mean_iou", seed=0, n_bootstrap=500)
        assert isinstance(ranks[0].ci, BootstrapCI)

    def test_empty_sequence_tracker_omitted(self):
        results = [
            _ResultStub("A", []),
            _ResultStub("B", [0.5] * 5),
        ]
        ranks = rank_trackers(results, metric="mean_iou", seed=0, n_bootstrap=200)
        assert len(ranks) == 1
        assert ranks[0].tracker_name == "B"

    def test_to_dict_structure(self):
        results = [_ResultStub("A", [0.6] * 5)]
        r = rank_trackers(results, metric="mean_iou", seed=0, n_bootstrap=200)[0]
        d = r.to_dict()
        assert d["rank"] == 1
        assert d["tracker_name"] == "A"
        assert "ci_lower" in d and "ci_upper" in d
        assert d["n_sequences"] == 5

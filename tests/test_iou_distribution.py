"""Unit tests for IoUDistributionAnalyzer."""

import numpy as np
import pytest

from eovot.analysis.iou_distribution import (
    IoUDistributionAnalyzer,
    IoUDistributionResult,
    BIMODALITY_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analyzer():
    return IoUDistributionAnalyzer()


@pytest.fixture
def uniform_ious():
    """Uniformly distributed IoU -- borderline bimodal (BC ~= 5/9)."""
    rng = np.random.default_rng(0)
    return rng.uniform(0.0, 1.0, 200)


@pytest.fixture
def bimodal_ious():
    """Bimodal: half near 0.0, half near 1.0 -- clearly bimodal."""
    rng = np.random.default_rng(1)
    low = rng.uniform(0.0, 0.15, 50)
    high = rng.uniform(0.85, 1.0, 50)
    return np.concatenate([low, high])


@pytest.fixture
def unimodal_ious():
    """Gaussian centred at 0.7 with small std -- unimodal, consistent tracker."""
    rng = np.random.default_rng(2)
    return np.clip(rng.normal(0.7, 0.05, 100), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_invalid_thresholds_raises():
    with pytest.raises(ValueError):
        IoUDistributionAnalyzer(failure_threshold=0.6, excellent_threshold=0.5)


def test_invalid_n_bins_raises():
    with pytest.raises(ValueError):
        IoUDistributionAnalyzer(n_bins=1)


# ---------------------------------------------------------------------------
# Bimodality coefficient
# ---------------------------------------------------------------------------

def test_bc_unimodal_is_low(analyzer, unimodal_ious):
    bc = analyzer.bimodality_coefficient(unimodal_ious)
    assert bc < BIMODALITY_THRESHOLD, f"Expected BC < {BIMODALITY_THRESHOLD}, got {bc}"


def test_bc_bimodal_is_high(analyzer, bimodal_ious):
    bc = analyzer.bimodality_coefficient(bimodal_ious)
    assert bc > BIMODALITY_THRESHOLD, f"Expected BC > {BIMODALITY_THRESHOLD}, got {bc}"


def test_bc_constant_array_returns_zero(analyzer):
    ious = np.full(50, 0.5)
    assert analyzer.bimodality_coefficient(ious) == 0.0


def test_bc_short_array_returns_zero(analyzer):
    assert analyzer.bimodality_coefficient(np.array([0.5, 0.6])) == 0.0


def test_bc_non_negative(analyzer, uniform_ious, bimodal_ious, unimodal_ious):
    for arr in [uniform_ious, bimodal_ious, unimodal_ious]:
        bc = analyzer.bimodality_coefficient(arr)
        assert bc >= 0.0, f"BC must be non-negative, got {bc}"


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------

def test_histogram_bins_count(analyzer):
    ious = np.linspace(0.0, 1.0, 100)
    counts, edges = analyzer.compute_histogram(ious)
    assert len(counts) == analyzer.n_bins
    assert len(edges) == analyzer.n_bins + 1


def test_histogram_sum_equals_n_frames(analyzer, uniform_ious):
    counts, _ = analyzer.compute_histogram(uniform_ious)
    assert sum(counts) == len(uniform_ious)


def test_histogram_empty_input(analyzer):
    counts, edges = analyzer.compute_histogram(np.array([]))
    assert all(c == 0 for c in counts)


# ---------------------------------------------------------------------------
# analyze_sequence
# ---------------------------------------------------------------------------

def test_analyze_sequence_basic(analyzer, unimodal_ious):
    result = analyzer.analyze_sequence(unimodal_ious, "MOSSE", "car1")
    assert isinstance(result, IoUDistributionResult)
    assert result.tracker_name == "MOSSE"
    assert result.sequence_name == "car1"
    assert result.num_frames == len(unimodal_ious)
    assert 0.0 <= result.mean_iou <= 1.0
    assert 0.0 <= result.failure_ratio <= 1.0
    assert 0.0 <= result.excellent_ratio <= 1.0


def test_unimodal_not_bimodal(analyzer, unimodal_ious):
    result = analyzer.analyze_sequence(unimodal_ious)
    assert not result.is_bimodal


def test_bimodal_is_detected(analyzer, bimodal_ious):
    result = analyzer.analyze_sequence(bimodal_ious)
    assert result.is_bimodal


def test_failure_ratio_all_zero(analyzer):
    ious = np.zeros(50)
    result = analyzer.analyze_sequence(ious)
    assert result.failure_ratio == 1.0
    assert result.excellent_ratio == 0.0


def test_excellent_ratio_all_one(analyzer):
    ious = np.ones(50)
    result = analyzer.analyze_sequence(ious)
    assert result.failure_ratio == 0.0
    assert result.excellent_ratio == 1.0


def test_quartiles_ordering(analyzer, uniform_ious):
    result = analyzer.analyze_sequence(uniform_ious)
    assert result.q1_iou <= result.median_iou <= result.q3_iou


def test_empty_ious(analyzer):
    result = analyzer.analyze_sequence(np.array([]))
    assert result.num_frames == 0
    assert result.mean_iou == 0.0
    assert not result.is_bimodal


def test_to_dict_roundtrip(analyzer, unimodal_ious):
    result = analyzer.analyze_sequence(unimodal_ious, "KCF", "seq1")
    d = result.to_dict()
    assert d["tracker_name"] == "KCF"
    assert d["sequence_name"] == "seq1"
    assert isinstance(d["histogram_counts"], list)
    assert isinstance(d["is_bimodal"], bool)


# ---------------------------------------------------------------------------
# analyze_benchmark
# ---------------------------------------------------------------------------

def test_analyze_benchmark_aggregate_keys(analyzer, unimodal_ious, bimodal_ious):
    seq_ious = {"seq1": unimodal_ious, "seq2": bimodal_ious}
    out = analyzer.analyze_benchmark(seq_ious, tracker_name="KCF")
    agg = out["aggregate"]
    assert agg["tracker_name"] == "KCF"
    assert agg["num_sequences"] == 2
    assert "mean_bimodality_coefficient" in agg
    assert "bimodal_sequence_count" in agg
    assert "mean_failure_ratio" in agg
    assert "mean_excellent_ratio" in agg


def test_analyze_benchmark_bimodal_count(analyzer, unimodal_ious, bimodal_ious):
    seq_ious = {
        "seq1": unimodal_ious,
        "seq2": unimodal_ious,
        "seq3": bimodal_ious,
    }
    out = analyzer.analyze_benchmark(seq_ious)
    agg = out["aggregate"]
    assert agg["bimodal_sequence_count"] == 1
    assert abs(agg["bimodal_sequence_fraction"] - 1.0 / 3.0) < 1e-4


def test_analyze_benchmark_empty(analyzer):
    out = analyzer.analyze_benchmark({})
    assert out["per_sequence"] == {}
    assert out["aggregate"] == {}


# ---------------------------------------------------------------------------
# compare_trackers and Markdown table
# ---------------------------------------------------------------------------

def test_compare_trackers_sorted_by_bc(analyzer, unimodal_ious, bimodal_ious):
    tracker_benchmarks = {
        "Good": {"seq1": unimodal_ious, "seq2": unimodal_ious},
        "Bad": {"seq1": bimodal_ious, "seq2": bimodal_ious},
    }
    rows = analyzer.compare_trackers(tracker_benchmarks)
    assert rows[0]["tracker_name"] == "Good"


def test_markdown_table_format(analyzer, unimodal_ious, bimodal_ious):
    tracker_benchmarks = {
        "MOSSE": {"s1": unimodal_ious},
        "KCF": {"s1": bimodal_ious},
    }
    rows = analyzer.compare_trackers(tracker_benchmarks)
    table = analyzer.to_markdown_table(rows)
    assert "| Tracker |" in table
    assert "MOSSE" in table
    assert "KCF" in table

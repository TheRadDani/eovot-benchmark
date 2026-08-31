"""Tests for TrackerCorrelationAnalyzer."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.analysis.tracker_correlation import (
    CorrelationReport,
    TrackerCorrelationAnalyzer,
)
from eovot.benchmark.engine import BenchmarkResult, SequenceResult
from eovot.profiling.profiler import ProfilingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(name: str, ious: list[np.ndarray]) -> BenchmarkResult:
    """Build a minimal BenchmarkResult from a list of per-sequence IoU arrays."""
    result = BenchmarkResult(tracker_name=name, dataset_name="test")
    for idx, iou_arr in enumerate(ious):
        profiling = ProfilingResult(
            tracker_name=name,
            frame_count=len(iou_arr),
            fps=50.0,
            latency_mean_ms=20.0,
            latency_std_ms=1.0,
            latency_p95_ms=22.0,
            latency_p99_ms=24.0,
            latency_cv=0.05,
            peak_memory_mb=128.0,
        )
        result.sequence_results.append(
            SequenceResult(
                sequence_name=f"seq_{idx}",
                ious=iou_arr,
                profiling=profiling,
            )
        )
    return result


def _rng_ious(seed: int, n: int = 200) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, n)


# ---------------------------------------------------------------------------
# Basic tests
# ---------------------------------------------------------------------------

class TestTrackerCorrelationAnalyzer:
    def test_requires_at_least_two_trackers(self):
        analyzer = TrackerCorrelationAnalyzer()
        r = _make_result("MOSSE", [_rng_ious(0)])
        with pytest.raises(ValueError, match="at least 2"):
            analyzer.analyze({"MOSSE": r})

    def test_self_correlation_is_one(self):
        ious = _rng_ious(1)
        r1 = _make_result("A", [ious.copy()])
        r2 = _make_result("B", [ious.copy()])
        analyzer = TrackerCorrelationAnalyzer()
        report = analyzer.analyze({"A": r1, "B": r2})
        assert abs(report.correlation("A", "B") - 1.0) < 1e-9

    def test_independent_trackers_low_correlation(self):
        r1 = _make_result("A", [_rng_ious(10)])
        r2 = _make_result("B", [_rng_ious(99)])
        analyzer = TrackerCorrelationAnalyzer()
        report = analyzer.analyze({"A": r1, "B": r2})
        # Independent random sequences should have |r| << 1
        assert abs(report.correlation("A", "B")) < 0.5

    def test_anticorrelated_high_ensemble_gain(self):
        n = 300
        base = _rng_ious(42, n)
        r1 = _make_result("A", [base])
        # Invert: where A is high, B is low and vice-versa
        r2 = _make_result("B", [1.0 - base])
        analyzer = TrackerCorrelationAnalyzer()
        report = analyzer.analyze({"A": r1, "B": r2})
        # Anti-correlated pair should have negative Pearson r
        assert report.correlation("A", "B") < -0.9
        # And a large ensemble gain (oracle always picks the winner)
        assert report.ensemble_potential("A", "B") > 0.1

    def test_matrix_shape(self):
        results = {
            "MOSSE": _make_result("MOSSE", [_rng_ious(0)]),
            "KCF": _make_result("KCF", [_rng_ious(1)]),
            "CSRT": _make_result("CSRT", [_rng_ious(2)]),
        }
        analyzer = TrackerCorrelationAnalyzer()
        report = analyzer.analyze(results)
        assert report.correlation_matrix.shape == (3, 3)
        assert report.ensemble_gain_matrix.shape == (3, 3)
        # Diagonal of correlation matrix should be 1.0
        np.testing.assert_allclose(np.diag(report.correlation_matrix), 1.0, atol=1e-9)
        # Diagonal of gain matrix should be 0.0 (no gain from pairing with yourself)
        np.testing.assert_allclose(np.diag(report.ensemble_gain_matrix), 0.0, atol=1e-9)

    def test_most_complementary_pair_is_symmetric(self):
        results = {
            "A": _make_result("A", [_rng_ious(10)]),
            "B": _make_result("B", [_rng_ious(11)]),
            "C": _make_result("C", [_rng_ious(12)]),
        }
        analyzer = TrackerCorrelationAnalyzer()
        report = analyzer.analyze(results)
        t1, t2 = report.most_complementary_pair()
        # Symmetry: correlation(t1, t2) == correlation(t2, t1)
        assert abs(report.correlation(t1, t2) - report.correlation(t2, t1)) < 1e-9

    def test_cluster_labels_count_matches_trackers(self):
        results = {
            name: _make_result(name, [_rng_ious(seed)])
            for seed, name in enumerate(["A", "B", "C", "D"])
        }
        analyzer = TrackerCorrelationAnalyzer(cluster_threshold=0.95)
        report = analyzer.analyze(results)
        assert len(report.cluster_labels) == 4

    def test_perfectly_correlated_pair_same_cluster(self):
        ious = _rng_ious(7)
        r1 = _make_result("X", [ious.copy()])
        r2 = _make_result("Y", [ious.copy()])
        r3 = _make_result("Z", [_rng_ious(99)])
        analyzer = TrackerCorrelationAnalyzer(cluster_threshold=0.99)
        report = analyzer.analyze({"X": r1, "Y": r2, "Z": r3})
        # X and Y are perfectly correlated — they must be in the same cluster
        idx_x = report.tracker_names.index("X")
        idx_y = report.tracker_names.index("Y")
        assert report.cluster_labels[idx_x] == report.cluster_labels[idx_y]

    def test_to_markdown_contains_key_sections(self):
        results = {
            "A": _make_result("A", [_rng_ious(0)]),
            "B": _make_result("B", [_rng_ious(1)]),
        }
        analyzer = TrackerCorrelationAnalyzer()
        report = analyzer.analyze(results)
        md = report.to_markdown()
        assert "Correlation Matrix" in md
        assert "Ensemble" in md
        assert "complementary" in md.lower()
        assert "redundant" in md.lower()

    def test_invalid_cluster_threshold_raises(self):
        with pytest.raises(ValueError):
            TrackerCorrelationAnalyzer(cluster_threshold=1.5)

    def test_multiple_sequences_concatenated(self):
        """IoU arrays from all sequences should be concatenated before correlation."""
        seqs_a = [_rng_ious(i, 50) for i in range(5)]
        seqs_b = [_rng_ious(i + 50, 50) for i in range(5)]
        r1 = _make_result("A", seqs_a)
        r2 = _make_result("B", seqs_b)
        analyzer = TrackerCorrelationAnalyzer()
        report = analyzer.analyze({"A": r1, "B": r2})
        # Should complete without error; result has expected shape
        assert report.correlation_matrix.shape == (2, 2)

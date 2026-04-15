"""Unit tests for eovot.metrics.robustness.

Tests cover:
- compute_failure_rate: edge cases, known values
- compute_eao: short sequences, perfect tracker, partial failures
- compute_robustness_metrics: integration, dataclass fields
- RobustnessMetrics.to_dict: serialisation contract
"""

from __future__ import annotations

import numpy as np
import pytest

from eovot.metrics.robustness import (
    RobustnessMetrics,
    compute_eao,
    compute_failure_rate,
    compute_robustness_metrics,
)


# ---------------------------------------------------------------------------
# compute_failure_rate
# ---------------------------------------------------------------------------

class TestComputeFailureRate:
    def test_empty_array_returns_zero(self):
        assert compute_failure_rate(np.array([])) == pytest.approx(0.0)

    def test_all_above_threshold(self):
        ious = np.array([0.5, 0.6, 0.7, 0.8])
        assert compute_failure_rate(ious, threshold=0.1) == pytest.approx(0.0)

    def test_all_below_threshold(self):
        ious = np.zeros(10)
        assert compute_failure_rate(ious, threshold=0.1) == pytest.approx(1.0)

    def test_half_failed(self):
        ious = np.array([0.0, 0.0, 0.5, 0.5])
        assert compute_failure_rate(ious, threshold=0.1) == pytest.approx(0.5)

    def test_threshold_boundary_is_exclusive(self):
        # IoU equal to threshold is NOT a failure (ious < threshold)
        ious = np.array([0.1, 0.1, 0.1])
        assert compute_failure_rate(ious, threshold=0.1) == pytest.approx(0.0)

    def test_custom_threshold(self):
        ious = np.array([0.3, 0.4, 0.5, 0.6])
        # Values < 0.5: [0.3, 0.4] → 2/4 = 0.5
        assert compute_failure_rate(ious, threshold=0.5) == pytest.approx(0.5)

    def test_single_frame_failure(self):
        assert compute_failure_rate(np.array([0.05]), threshold=0.1) == pytest.approx(1.0)

    def test_single_frame_success(self):
        assert compute_failure_rate(np.array([0.9]), threshold=0.1) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_eao
# ---------------------------------------------------------------------------

class TestComputeEAO:
    def test_empty_returns_zero(self):
        assert compute_eao(np.array([])) == pytest.approx(0.0)

    def test_short_sequence_fallback(self):
        # Sequence shorter than min_len → returns plain mean
        ious = np.array([0.6, 0.7, 0.8])
        result = compute_eao(ious, min_len=10, max_len=100)
        assert result == pytest.approx(float(ious.mean()), abs=1e-9)

    def test_perfect_tracker_eao_is_one(self):
        ious = np.ones(50)
        result = compute_eao(ious, min_len=5, max_len=30)
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_zero_tracker_eao_is_zero(self):
        ious = np.zeros(50)
        result = compute_eao(ious, min_len=5, max_len=30)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_eao_in_range(self):
        rng = np.random.default_rng(42)
        ious = rng.uniform(0, 1, 80)
        result = compute_eao(ious, min_len=10, max_len=50)
        assert 0.0 <= result <= 1.0

    def test_eao_lower_for_more_failures(self):
        # A sequence with many failures should score lower than one with few failures.
        n = 60
        mostly_good = np.concatenate([np.ones(50), np.zeros(10)])
        mostly_bad = np.concatenate([np.zeros(50), np.ones(10)])
        eao_good = compute_eao(mostly_good, min_len=10, max_len=40)
        eao_bad = compute_eao(mostly_bad, min_len=10, max_len=40)
        assert eao_good > eao_bad

    def test_max_len_capped_at_n(self):
        # max_len larger than sequence should not raise; result should be defined.
        ious = np.full(20, 0.5)
        result = compute_eao(ious, min_len=5, max_len=200)
        assert result == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# compute_robustness_metrics
# ---------------------------------------------------------------------------

class TestComputeRobustnessMetrics:
    def test_returns_robustness_metrics_instance(self):
        ious = np.array([0.8, 0.7, 0.05, 0.0, 0.75])
        result = compute_robustness_metrics(ious)
        assert isinstance(result, RobustnessMetrics)

    def test_robustness_plus_failure_rate_equals_one(self):
        ious = np.random.default_rng(0).uniform(0, 1, 30)
        m = compute_robustness_metrics(ious)
        assert m.robustness + m.failure_rate == pytest.approx(1.0)

    def test_perfect_tracker(self):
        ious = np.ones(40)
        m = compute_robustness_metrics(ious)
        assert m.failure_rate == pytest.approx(0.0)
        assert m.robustness == pytest.approx(1.0)
        assert m.eao == pytest.approx(1.0)
        assert m.mean_iou_active == pytest.approx(1.0)

    def test_always_failed_tracker(self):
        ious = np.zeros(40)
        m = compute_robustness_metrics(ious, threshold=0.1)
        assert m.failure_rate == pytest.approx(1.0)
        assert m.robustness == pytest.approx(0.0)
        assert m.mean_iou_active is None

    def test_threshold_stored_in_result(self):
        ious = np.ones(20) * 0.5
        m = compute_robustness_metrics(ious, threshold=0.25)
        assert m.threshold == pytest.approx(0.25)

    def test_mean_iou_active_excludes_failures(self):
        # Frames 0–2: IoU = 0.8; frames 3–4: IoU = 0.0 (failures at threshold 0.1)
        ious = np.array([0.8, 0.8, 0.8, 0.0, 0.0])
        m = compute_robustness_metrics(ious, threshold=0.1)
        assert m.mean_iou_active == pytest.approx(0.8)

    def test_eao_bounds(self):
        rng = np.random.default_rng(7)
        ious = rng.uniform(0, 1, 60)
        m = compute_robustness_metrics(ious)
        assert 0.0 <= m.eao <= 1.0

    def test_to_dict_keys(self):
        ious = np.array([0.6, 0.7, 0.8, 0.05, 0.9])
        m = compute_robustness_metrics(ious)
        d = m.to_dict()
        assert set(d.keys()) == {"failure_rate", "robustness", "eao", "mean_iou_active", "threshold"}

    def test_to_dict_values_are_rounded(self):
        ious = np.array([0.1234567, 0.9876543, 0.0, 0.7654321])
        m = compute_robustness_metrics(ious, threshold=0.1)
        d = m.to_dict()
        # Rounded to 4 decimal places — check no more than 4 significant decimal digits
        assert d["failure_rate"] == round(m.failure_rate, 4)
        assert d["robustness"] == round(m.robustness, 4)

    def test_str_representation_contains_key_fields(self):
        ious = np.array([0.5, 0.6, 0.0, 0.7])
        m = compute_robustness_metrics(ious)
        s = str(m)
        assert "failure_rate" in s
        assert "robustness" in s
        assert "eao" in s


# ---------------------------------------------------------------------------
# Integration: metrics package re-exports
# ---------------------------------------------------------------------------

class TestMetricsPackageExports:
    def test_imports_from_package(self):
        """All robustness symbols must be importable from eovot.metrics."""
        from eovot.metrics import (  # noqa: F401
            RobustnessMetrics,
            compute_eao,
            compute_failure_rate,
            compute_robustness_metrics,
        )

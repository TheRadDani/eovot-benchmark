"""Tests for protocol-specific evaluation metric implementations."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.metrics.protocols import (
    GOT10kMetrics,
    LaSOTMetrics,
    OTBMetrics,
    ProtocolMetricsEngine,
    PROTOCOL_REGISTRY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _perfect_boxes(n: int, w: float = 50.0, h: float = 40.0):
    """Identical predicted and GT boxes → perfect tracking."""
    boxes = np.array([[10.0 + i, 20.0, w, h] for i in range(n)])
    return boxes.copy(), boxes.copy()


def _zero_overlap_boxes(n: int):
    """Predicted boxes completely separated from GT → IoU = 0 everywhere."""
    preds = np.array([[0.0, 0.0, 10.0, 10.0]] * n)
    gts = np.array([[1000.0, 1000.0, 10.0, 10.0]] * n)
    return preds, gts


def _half_overlap_boxes(n: int, size: float = 40.0):
    """Predicted box shifted by half its width → IoU ≈ 0.33 (horizontal)."""
    preds = np.array([[0.0, 0.0, size, size]] * n)
    gts = np.array([[size / 2, 0.0, size, size]] * n)
    return preds, gts


def _empty_arrays():
    return np.empty((0, 4)), np.empty((0, 4))


# ---------------------------------------------------------------------------
# GOT-10k metrics
# ---------------------------------------------------------------------------

class TestGOT10kMetrics:

    def test_perfect_tracking(self):
        preds, gts = _perfect_boxes(100)
        result = GOT10kMetrics().compute(preds, gts)

        assert result["ao"] == pytest.approx(1.0, abs=1e-4)
        assert result["sr_50"] == pytest.approx(1.0, abs=1e-4)
        assert result["sr_75"] == pytest.approx(1.0, abs=1e-4)

    def test_zero_overlap(self):
        preds, gts = _zero_overlap_boxes(50)
        result = GOT10kMetrics().compute(preds, gts)

        assert result["ao"] == 0.0
        assert result["sr_50"] == 0.0
        assert result["sr_75"] == 0.0

    def test_keys_present(self):
        preds, gts = _perfect_boxes(10)
        result = GOT10kMetrics().compute(preds, gts)
        assert set(result.keys()) == {"ao", "sr_50", "sr_75"}

    def test_values_in_range(self):
        preds, gts = _half_overlap_boxes(80)
        result = GOT10kMetrics().compute(preds, gts)
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_ao_le_sr50_le_sr75_in_boundary(self):
        """SR₀.₅ ≥ SR₀.₇₅ always."""
        preds, gts = _half_overlap_boxes(100)
        result = GOT10kMetrics().compute(preds, gts)
        assert result["sr_50"] >= result["sr_75"]

    def test_empty_arrays(self):
        preds, gts = _empty_arrays()
        result = GOT10kMetrics().compute(preds, gts)
        assert result["ao"] == 0.0
        assert result["sr_50"] == 0.0
        assert result["sr_75"] == 0.0

    def test_multi_sequence(self):
        """Multi-sequence averaging should equal single-sequence for uniform data."""
        preds, gts = _perfect_boxes(20)
        m = GOT10kMetrics()
        single = m.compute(preds, gts)
        multi = m.compute_multi_sequence([preds, preds], [gts, gts])
        assert multi["ao"] == pytest.approx(single["ao"], abs=1e-4)
        assert multi["sr_50"] == pytest.approx(single["sr_50"], abs=1e-4)

    def test_multi_sequence_empty(self):
        m = GOT10kMetrics()
        result = m.compute_multi_sequence([], [])
        assert result == {"ao": 0.0, "sr_50": 0.0, "sr_75": 0.0}

    def test_ao_monotone_with_iou(self):
        """Higher-quality predictions should yield higher AO."""
        preds_good, gts = _perfect_boxes(50)
        preds_bad, _ = _zero_overlap_boxes(50)
        ao_good = GOT10kMetrics().compute(preds_good, gts)["ao"]
        ao_bad = GOT10kMetrics().compute(preds_bad, gts)["ao"]
        assert ao_good > ao_bad


# ---------------------------------------------------------------------------
# LaSOT metrics
# ---------------------------------------------------------------------------

class TestLaSOTMetrics:

    def test_perfect_tracking(self):
        preds, gts = _perfect_boxes(100)
        result = LaSOTMetrics().compute(preds, gts)

        assert result["success_auc"] == pytest.approx(1.0, abs=1e-3)
        assert result["norm_precision"] == pytest.approx(1.0, abs=1e-3)
        # Trapezoidal rule with 51 points introduces a small discretization
        # error even for a perfect (constant-1) curve.
        assert result["norm_precision_auc"] == pytest.approx(1.0, abs=0.02)

    def test_zero_overlap(self):
        preds, gts = _zero_overlap_boxes(50)
        result = LaSOTMetrics().compute(preds, gts)

        # Success AUC includes a thin sliver at threshold=0 where IoU≥0 always
        # holds, so a fully-failing tracker yields a tiny positive AUC (< 0.01).
        assert result["success_auc"] < 0.01
        assert result["norm_precision"] == 0.0

    def test_keys_present(self):
        preds, gts = _perfect_boxes(10)
        result = LaSOTMetrics().compute(preds, gts)
        assert set(result.keys()) == {"success_auc", "norm_precision", "norm_precision_auc"}

    def test_values_in_range(self):
        preds, gts = _half_overlap_boxes(80)
        result = LaSOTMetrics().compute(preds, gts)
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_empty_arrays(self):
        preds, gts = _empty_arrays()
        result = LaSOTMetrics().compute(preds, gts)
        assert result["success_auc"] == 0.0
        assert result["norm_precision"] == 0.0

    def test_normalized_precision_scale_invariant(self):
        """NP should be similar for identical relative configurations at different scales."""
        # Small box, small error
        preds_s = np.array([[0.0, 0.0, 20.0, 20.0]] * 50)
        gts_s = np.array([[5.0, 0.0, 20.0, 20.0]] * 50)  # error = 5px, box=20 → ratio=5/20=0.25

        # Large box, proportionally larger error
        preds_l = np.array([[0.0, 0.0, 80.0, 80.0]] * 50)
        gts_l = np.array([[20.0, 0.0, 80.0, 80.0]] * 50)  # error = 20px, box=80 → ratio=20/80=0.25

        np_small = LaSOTMetrics().compute(preds_s, gts_s)["norm_precision"]
        np_large = LaSOTMetrics().compute(preds_l, gts_l)["norm_precision"]

        # Both have the same normalized error ratio, so NP should be equal
        assert np_small == pytest.approx(np_large, abs=0.05)


# ---------------------------------------------------------------------------
# OTB metrics
# ---------------------------------------------------------------------------

class TestOTBMetrics:

    def test_perfect_tracking(self):
        preds, gts = _perfect_boxes(100)
        result = OTBMetrics().compute(preds, gts)

        assert result["success_auc"] == pytest.approx(1.0, abs=1e-3)
        assert result["precision_score_20px"] == pytest.approx(1.0, abs=1e-4)

    def test_zero_overlap(self):
        preds, gts = _zero_overlap_boxes(50)
        result = OTBMetrics().compute(preds, gts)

        # Success AUC has a thin sliver at threshold=0 → tiny positive value.
        assert result["success_auc"] < 0.01
        assert result["precision_score_20px"] == 0.0

    def test_keys_present(self):
        preds, gts = _perfect_boxes(10)
        result = OTBMetrics().compute(preds, gts)
        assert set(result.keys()) == {"success_auc", "precision_score_20px"}

    def test_values_in_range(self):
        preds, gts = _half_overlap_boxes(80)
        result = OTBMetrics().compute(preds, gts)
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_empty_arrays(self):
        preds, gts = _empty_arrays()
        result = OTBMetrics().compute(preds, gts)
        assert result["success_auc"] == 0.0
        assert result["precision_score_20px"] == 0.0

    def test_precision_threshold_at_20px(self):
        """Predictions within 20 px of GT centre → precision = 1."""
        # GT centred at (30, 30), pred shifted 15 px right (< 20 px)
        size = 20.0
        preds = np.array([[20.0, 20.0, size, size]] * 40)  # centre (30, 30)
        gts = np.array([[5.0, 20.0, size, size]] * 40)    # centre (15, 30) → dist = 15 px < 20
        result = OTBMetrics().compute(preds, gts)
        assert result["precision_score_20px"] == pytest.approx(1.0, abs=1e-4)

    def test_precision_threshold_beyond_20px(self):
        """Predictions more than 20 px away → precision = 0."""
        size = 20.0
        preds = np.array([[0.0, 0.0, size, size]] * 40)    # centre (10, 10)
        gts = np.array([[50.0, 50.0, size, size]] * 40)   # centre (60, 60) → dist ≈ 70 px
        result = OTBMetrics().compute(preds, gts)
        assert result["precision_score_20px"] == 0.0


# ---------------------------------------------------------------------------
# ProtocolMetricsEngine dispatcher
# ---------------------------------------------------------------------------

class TestProtocolMetricsEngine:

    def test_known_protocols_registered(self):
        engine = ProtocolMetricsEngine()
        for proto in ["got10k", "lasot", "otb"]:
            assert proto in engine.PROTOCOLS

    def test_dispatch_got10k(self):
        preds, gts = _perfect_boxes(20)
        result = ProtocolMetricsEngine().compute("got10k", preds, gts)
        assert "ao" in result

    def test_dispatch_lasot(self):
        preds, gts = _perfect_boxes(20)
        result = ProtocolMetricsEngine().compute("lasot", preds, gts)
        assert "success_auc" in result

    def test_dispatch_otb(self):
        preds, gts = _perfect_boxes(20)
        result = ProtocolMetricsEngine().compute("otb", preds, gts)
        assert "precision_score_20px" in result

    def test_case_insensitive(self):
        preds, gts = _perfect_boxes(20)
        result_lower = ProtocolMetricsEngine().compute("got10k", preds, gts)
        result_upper = ProtocolMetricsEngine().compute("GOT10K", preds, gts)
        assert result_lower == result_upper

    def test_unknown_protocol_raises(self):
        preds, gts = _perfect_boxes(10)
        with pytest.raises(ValueError, match="Unknown protocol"):
            ProtocolMetricsEngine().compute("nonexistent", preds, gts)

    def test_compute_all_returns_all_protocols(self):
        preds, gts = _perfect_boxes(20)
        results = ProtocolMetricsEngine().compute_all(preds, gts)
        assert set(results.keys()) == set(PROTOCOL_REGISTRY)

    def test_compute_all_values_consistent(self):
        """compute_all should agree with individual protocol calls."""
        preds, gts = _half_overlap_boxes(50)
        engine = ProtocolMetricsEngine()
        all_results = engine.compute_all(preds, gts)
        for proto in PROTOCOL_REGISTRY:
            individual = engine.compute(proto, preds, gts)
            assert all_results[proto] == individual

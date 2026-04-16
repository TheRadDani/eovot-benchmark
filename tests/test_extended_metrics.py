"""Unit tests for eovot.metrics.extended — GOT-10k and VOT metrics."""

import numpy as np
import pytest

from eovot.metrics.extended import (
    ExtendedMetrics,
    ExtendedMetricsEngine,
    eao_score,
    robustness_rate,
    success_at_threshold,
)


# ---------------------------------------------------------------------------
# success_at_threshold
# ---------------------------------------------------------------------------


class TestSuccessAtThreshold:
    def test_all_above(self):
        ious = np.ones(10) * 0.9
        assert success_at_threshold(ious, 0.5) == pytest.approx(1.0)

    def test_all_below(self):
        ious = np.ones(10) * 0.3
        assert success_at_threshold(ious, 0.5) == pytest.approx(0.0)

    def test_half_above(self):
        ious = np.array([0.6, 0.4, 0.6, 0.4, 0.6, 0.4])
        assert success_at_threshold(ious, 0.5) == pytest.approx(0.5)

    def test_empty_array(self):
        assert success_at_threshold(np.array([]), 0.5) == pytest.approx(0.0)

    def test_sr_0_5_vs_sr_0_75(self):
        ious = np.linspace(0.0, 1.0, 100)
        sr_05 = success_at_threshold(ious, 0.5)
        sr_075 = success_at_threshold(ious, 0.75)
        # SR_0.5 must be >= SR_0.75 since fewer frames exceed the higher threshold.
        assert sr_05 >= sr_075

    def test_threshold_boundary_strict(self):
        # IoU exactly equal to threshold should NOT count (strict ">").
        ious = np.array([0.5, 0.5, 0.5])
        assert success_at_threshold(ious, 0.5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# robustness_rate
# ---------------------------------------------------------------------------


class TestRobustnessRate:
    def test_all_above_threshold(self):
        ious = np.ones(20) * 0.5
        assert robustness_rate(ious, failure_threshold=0.1) == pytest.approx(1.0)

    def test_all_below_threshold(self):
        ious = np.zeros(20)
        assert robustness_rate(ious, failure_threshold=0.1) == pytest.approx(0.0)

    def test_half_failures(self):
        # 5 frames above, 5 frames below threshold=0.1
        ious = np.array([0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0])
        assert robustness_rate(ious, failure_threshold=0.1) == pytest.approx(0.5)

    def test_boundary_inclusive(self):
        # IoU exactly equal to failure_threshold should count as success (>=).
        ious = np.array([0.1, 0.1, 0.1])
        assert robustness_rate(ious, failure_threshold=0.1) == pytest.approx(1.0)

    def test_empty_array(self):
        assert robustness_rate(np.array([])) == pytest.approx(0.0)

    def test_custom_threshold(self):
        ious = np.array([0.4, 0.4, 0.4, 0.2, 0.2])
        # With threshold=0.3: first 3 succeed (0.4 >= 0.3), last 2 fail
        assert robustness_rate(ious, failure_threshold=0.3) == pytest.approx(3 / 5)


# ---------------------------------------------------------------------------
# eao_score
# ---------------------------------------------------------------------------


class TestEAOScore:
    def test_empty_array(self):
        assert eao_score(np.array([])) == pytest.approx(0.0)

    def test_perfect_tracking(self):
        ious = np.ones(200)
        score = eao_score(ious, min_len=10, max_len=100)
        assert score == pytest.approx(1.0)

    def test_zero_tracking(self):
        ious = np.zeros(200)
        score = eao_score(ious, min_len=10, max_len=100)
        assert score == pytest.approx(0.0)

    def test_range_in_0_1(self):
        rng = np.random.default_rng(0)
        ious = rng.uniform(0.0, 1.0, 150)
        score = eao_score(ious, min_len=10, max_len=100)
        assert 0.0 <= score <= 1.0

    def test_short_sequence_clamping(self):
        # Sequence shorter than max_len — should not raise.
        ious = np.ones(30) * 0.7
        score = eao_score(ious, min_len=5, max_len=100)
        assert score == pytest.approx(0.7, abs=1e-4)

    def test_degenerate_window(self):
        # min_len >= max_len — should return a valid scalar.
        ious = np.ones(20) * 0.5
        score = eao_score(ious, min_len=15, max_len=10)
        assert 0.0 <= score <= 1.0

    def test_monotone_with_better_tracker(self):
        # A tracker with higher IoU should have a higher EAO.
        ious_good = np.ones(100) * 0.8
        ious_bad = np.ones(100) * 0.4
        assert eao_score(ious_good) > eao_score(ious_bad)


# ---------------------------------------------------------------------------
# ExtendedMetricsEngine
# ---------------------------------------------------------------------------


class TestExtendedMetricsEngine:
    def setup_method(self):
        self.engine = ExtendedMetricsEngine()

    def test_perfect_preds_all_ones(self):
        boxes = np.tile([0.0, 0.0, 10.0, 10.0], (50, 1))
        result = self.engine.compute_extended(boxes, boxes)
        assert result.mean_iou == pytest.approx(1.0)
        assert result.sr_0_5 == pytest.approx(1.0)
        assert result.sr_0_75 == pytest.approx(1.0)
        assert result.robustness == pytest.approx(1.0)
        assert 0.0 <= result.eao <= 1.0

    def test_no_overlap_zero_scores(self):
        preds = np.tile([0.0, 0.0, 10.0, 10.0], (30, 1))
        gts = np.tile([50.0, 50.0, 10.0, 10.0], (30, 1))
        result = self.engine.compute_extended(preds, gts)
        assert result.mean_iou == pytest.approx(0.0)
        assert result.sr_0_5 == pytest.approx(0.0)
        assert result.sr_0_75 == pytest.approx(0.0)
        assert result.robustness == pytest.approx(0.0)
        assert result.eao == pytest.approx(0.0)

    def test_sr_ordering(self):
        # SR_0.5 must be >= SR_0.75 for any input.
        rng = np.random.default_rng(7)
        preds = rng.uniform(0, 80, (40, 4))
        gts = rng.uniform(0, 80, (40, 4))
        preds[:, 2:] = np.abs(preds[:, 2:]) + 2
        gts[:, 2:] = np.abs(gts[:, 2:]) + 2
        result = self.engine.compute_extended(preds, gts)
        assert result.sr_0_5 >= result.sr_0_75

    def test_to_dict_keys(self):
        boxes = np.tile([5.0, 5.0, 10.0, 10.0], (20, 1))
        result = self.engine.compute_extended(boxes, boxes)
        d = result.to_dict()
        for key in ("mean_iou", "success_auc", "precision_auc",
                    "sr_0_5", "sr_0_75", "robustness", "eao"):
            assert key in d, f"Missing key in to_dict(): {key}"

    def test_str_representation(self):
        boxes = np.tile([0.0, 0.0, 5.0, 5.0], (10, 1))
        result = self.engine.compute_extended(boxes, boxes)
        s = str(result)
        assert "AO=" in s
        assert "SR_0.5=" in s
        assert "EAO=" in s

    def test_values_in_range(self):
        rng = np.random.default_rng(42)
        preds = rng.uniform(0, 100, (60, 4))
        gts = rng.uniform(0, 100, (60, 4))
        preds[:, 2:] = np.abs(preds[:, 2:]) + 1
        gts[:, 2:] = np.abs(gts[:, 2:]) + 1
        result = self.engine.compute_extended(preds, gts)
        for field_name, value in result.to_dict().items():
            assert 0.0 <= value <= 1.0, f"{field_name}={value} out of [0, 1]"


# ---------------------------------------------------------------------------
# ExtendedMetrics dataclass
# ---------------------------------------------------------------------------


class TestExtendedMetricsDataclass:
    def _make(self, **overrides):
        defaults = dict(
            mean_iou=0.5,
            success_auc=0.45,
            precision_auc=0.7,
            sr_0_5=0.6,
            sr_0_75=0.3,
            robustness=0.8,
            eao=0.4,
        )
        defaults.update(overrides)
        return ExtendedMetrics(**defaults)

    def test_to_dict_rounds_to_4dp(self):
        m = self._make(mean_iou=0.123456789)
        assert m.to_dict()["mean_iou"] == 0.1235  # rounded to 4 dp

    def test_str_contains_all_fields(self):
        m = self._make()
        s = str(m)
        for tag in ("AO=", "AUC=", "SR_0.5=", "SR_0.75=", "robustness=", "EAO="):
            assert tag in s

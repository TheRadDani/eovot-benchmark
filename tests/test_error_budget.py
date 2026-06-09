"""Tests for eovot.metrics.error_budget — ErrorBudgetAnalyzer."""

import numpy as np
import pytest

from eovot.metrics.error_budget import (
    AggregateErrorBudget,
    ErrorBudget,
    ErrorBudgetAnalyzer,
    FrameErrorBudget,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _perfect_predictions(n: int = 20) -> tuple:
    """GT and predictions are identical — zero IoU loss."""
    gt = np.array([[10.0, 20.0, 40.0, 30.0]] * n)
    return gt.copy(), gt.copy()


def _center_shifted(n: int = 20, shift: float = 20.0) -> tuple:
    """Predictions shifted right by `shift` pixels — pure centre error."""
    gt = np.array([[50.0, 50.0, 40.0, 30.0]] * n)
    preds = gt.copy()
    preds[:, 0] += shift   # shift x → pure centre displacement
    return preds, gt


def _scale_wrong(n: int = 20, factor: float = 2.0) -> tuple:
    """Predictions have `factor`× the GT width and height — scale error."""
    gt = np.array([[50.0, 50.0, 40.0, 30.0]] * n)
    # Centre the enlarged box on the same GT centre to isolate scale error.
    pred_cx = gt[:, 0] + gt[:, 2] / 2.0
    pred_cy = gt[:, 1] + gt[:, 3] / 2.0
    new_w = gt[:, 2] * factor
    new_h = gt[:, 3] * factor
    preds = np.column_stack([
        pred_cx - new_w / 2.0,
        pred_cy - new_h / 2.0,
        new_w,
        new_h,
    ])
    return preds, gt


def _ar_wrong(n: int = 20) -> tuple:
    """Predictions with GT area but wrong aspect ratio — AR error."""
    gt = np.array([[50.0, 50.0, 40.0, 20.0]] * n)   # GT: AR = 2.0
    # Same area, AR = 1.0 (square), centred on GT centre
    gt_area = gt[0, 2] * gt[0, 3]  # 800
    side = float(np.sqrt(gt_area))  # ≈ 28.28
    pred_cx = gt[:, 0] + gt[:, 2] / 2.0
    pred_cy = gt[:, 1] + gt[:, 3] / 2.0
    preds = np.column_stack([
        pred_cx - side / 2.0,
        pred_cy - side / 2.0,
        np.full(n, side),
        np.full(n, side),
    ])
    return preds, gt


# ---------------------------------------------------------------------------
# ErrorBudget dataclass
# ---------------------------------------------------------------------------

class TestErrorBudget:
    def _make_budget(self, **kwargs):
        defaults = dict(
            tracker_name="T",
            sequence_name="S",
            num_frames=10,
            mean_iou_loss=0.3,
            center_error=0.5,
            scale_error=0.3,
            aspect_ratio_error=0.1,
            residual_error=0.1,
            dominant_axis="center",
            center_contributions=np.ones(10) * 0.5,
            scale_contributions=np.ones(10) * 0.3,
            aspect_ratio_contributions=np.ones(10) * 0.1,
        )
        defaults.update(kwargs)
        return ErrorBudget(**defaults)

    def test_to_dict_keys(self):
        b = self._make_budget()
        d = b.to_dict()
        assert set(d) == {
            "tracker", "sequence", "num_frames", "mean_iou_loss",
            "center_error", "scale_error", "aspect_ratio_error",
            "residual_error", "dominant_axis",
        }

    def test_to_dict_no_arrays(self):
        b = self._make_budget()
        d = b.to_dict()
        for v in d.values():
            assert not isinstance(v, np.ndarray)

    def test_str_contains_tracker(self):
        b = self._make_budget(tracker_name="MOSSE")
        assert "MOSSE" in str(b)

    def test_str_contains_dominant_axis(self):
        b = self._make_budget(dominant_axis="scale")
        assert "scale" in str(b)


# ---------------------------------------------------------------------------
# ErrorBudgetAnalyzer — input validation
# ---------------------------------------------------------------------------

class TestAnalyzerInputValidation:
    def setup_method(self):
        self.analyzer = ErrorBudgetAnalyzer()

    def test_empty_predictions_raises(self):
        with pytest.raises(ValueError, match="at least one frame"):
            self.analyzer.analyze(np.zeros((0, 4)), np.zeros((0, 4)))

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            self.analyzer.analyze(np.ones((5, 3)), np.ones((5, 3)))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            self.analyzer.analyze(np.ones((5, 4)), np.ones((4, 4)))


# ---------------------------------------------------------------------------
# ErrorBudgetAnalyzer.analyze — correctness on synthetic cases
# ---------------------------------------------------------------------------

class TestAnalyzeCorrectness:
    def setup_method(self):
        self.analyzer = ErrorBudgetAnalyzer()

    def test_perfect_predictions_zero_loss(self):
        preds, gts = _perfect_predictions(30)
        budget = self.analyzer.analyze(preds, gts)
        assert budget.mean_iou_loss == pytest.approx(0.0, abs=1e-6)
        assert budget.num_frames == 30

    def test_center_shift_increases_center_error(self):
        """Correcting only the centre should recover most IoU."""
        preds, gts = _center_shifted(30, shift=15.0)
        budget = self.analyzer.analyze(preds, gts)
        assert budget.center_error > budget.scale_error
        assert budget.center_error > budget.aspect_ratio_error

    def test_scale_error_dominates_for_scale_wrong(self):
        """Boxes with 2× the GT size — scale axis should dominate."""
        preds, gts = _scale_wrong(30, factor=2.0)
        budget = self.analyzer.analyze(preds, gts)
        assert budget.scale_error >= budget.center_error
        assert budget.dominant_axis == "scale"

    def test_ar_error_for_ar_wrong(self):
        """Wrong aspect ratio — AR error should be non-trivial."""
        preds, gts = _ar_wrong(20)
        budget = self.analyzer.analyze(preds, gts)
        assert budget.aspect_ratio_error > 0.0

    def test_dominant_axis_reflects_max_fraction(self):
        preds, gts = _center_shifted(20, shift=20.0)
        budget = self.analyzer.analyze(preds, gts)
        axis_map = {
            "center": budget.center_error,
            "scale": budget.scale_error,
            "aspect_ratio": budget.aspect_ratio_error,
            "residual": budget.residual_error,
        }
        assert budget.dominant_axis == max(axis_map, key=lambda k: axis_map[k])

    def test_fractions_non_negative(self):
        for preds, gts in [_center_shifted(20), _scale_wrong(20), _ar_wrong(20)]:
            budget = self.analyzer.analyze(preds, gts)
            assert budget.center_error >= 0.0
            assert budget.scale_error >= 0.0
            assert budget.aspect_ratio_error >= 0.0
            assert budget.residual_error >= 0.0

    def test_iou_loss_in_range(self):
        preds, gts = _center_shifted(20, shift=15.0)
        budget = self.analyzer.analyze(preds, gts)
        assert 0.0 <= budget.mean_iou_loss <= 1.0

    def test_per_frame_arrays_shape(self):
        preds, gts = _center_shifted(25, shift=10.0)
        budget = self.analyzer.analyze(preds, gts)
        assert len(budget.center_contributions) == 25
        assert len(budget.scale_contributions) == 25
        assert len(budget.aspect_ratio_contributions) == 25

    def test_zero_area_gt_frames_excluded(self):
        """Degenerate GT frames (w or h = 0) must not count in num_frames."""
        preds, gts = _center_shifted(10, shift=5.0)
        gts[3, 2] = 0.0   # zero width → degenerate
        gts[7, 3] = 0.0   # zero height → degenerate
        budget = self.analyzer.analyze(preds, gts)
        assert budget.num_frames == 8  # 10 - 2 degenerate

    def test_all_degenerate_gt_returns_zero_frames(self):
        """All-zero GT areas → graceful fallback result."""
        gt = np.zeros((10, 4))  # all zero area
        preds = np.tile([10.0, 10.0, 40.0, 30.0], (10, 1)).astype(float)
        budget = self.analyzer.analyze(preds, gt)
        assert budget.num_frames == 0
        assert budget.mean_iou_loss == pytest.approx(0.0, abs=1e-9)

    def test_tracker_name_and_sequence_name_stored(self):
        preds, gts = _perfect_predictions(5)
        budget = self.analyzer.analyze(preds, gts, tracker_name="KCF", sequence_name="car1")
        assert budget.tracker_name == "KCF"
        assert budget.sequence_name == "car1"

    def test_to_dict_roundtrip(self):
        preds, gts = _center_shifted(15, shift=8.0)
        budget = self.analyzer.analyze(preds, gts, tracker_name="T", sequence_name="S")
        d = budget.to_dict()
        assert d["tracker"] == "T"
        assert d["sequence"] == "S"
        assert 0.0 <= d["center_error"] <= 1.0


# ---------------------------------------------------------------------------
# Private geometric helpers
# ---------------------------------------------------------------------------

class TestGeometricHelpers:
    def test_batch_iou_identical(self):
        boxes = np.array([[10.0, 20.0, 40.0, 30.0]] * 5)
        ious = ErrorBudgetAnalyzer._batch_iou(boxes, boxes)
        np.testing.assert_allclose(ious, 1.0, atol=1e-6)

    def test_batch_iou_no_overlap(self):
        a = np.array([[0.0, 0.0, 10.0, 10.0]])
        b = np.array([[100.0, 100.0, 10.0, 10.0]])
        ious = ErrorBudgetAnalyzer._batch_iou(a, b)
        assert ious[0] == pytest.approx(0.0, abs=1e-6)

    def test_batch_iou_half_overlap(self):
        a = np.array([[0.0, 0.0, 20.0, 10.0]])
        b = np.array([[10.0, 0.0, 20.0, 10.0]])
        ious = ErrorBudgetAnalyzer._batch_iou(a, b)
        # Intersection = 10×10 = 100; Union = 20×10 + 20×10 - 100 = 300
        assert ious[0] == pytest.approx(100.0 / 300.0, abs=1e-6)

    def test_correct_center_preserves_size(self):
        preds, gts = _center_shifted(10, shift=10.0)
        corrected = ErrorBudgetAnalyzer._correct_center(preds, gts)
        np.testing.assert_allclose(corrected[:, 2], preds[:, 2], atol=1e-9)
        np.testing.assert_allclose(corrected[:, 3], preds[:, 3], atol=1e-9)

    def test_correct_center_aligns_center(self):
        preds, gts = _center_shifted(5, shift=15.0)
        corrected = ErrorBudgetAnalyzer._correct_center(preds, gts)
        corr_cx = corrected[:, 0] + corrected[:, 2] / 2.0
        gt_cx = gts[:, 0] + gts[:, 2] / 2.0
        np.testing.assert_allclose(corr_cx, gt_cx, atol=1e-9)

    def test_correct_scale_preserves_center(self):
        preds, gts = _scale_wrong(5, factor=2.0)
        corrected = ErrorBudgetAnalyzer._correct_scale(preds, gts)
        pred_cx = preds[:, 0] + preds[:, 2] / 2.0
        corr_cx = corrected[:, 0] + corrected[:, 2] / 2.0
        np.testing.assert_allclose(corr_cx, pred_cx, atol=1e-6)

    def test_correct_scale_matches_gt_area(self):
        preds, gts = _scale_wrong(5, factor=2.0)
        corrected = ErrorBudgetAnalyzer._correct_scale(preds, gts)
        gt_areas = gts[:, 2] * gts[:, 3]
        corr_areas = corrected[:, 2] * corrected[:, 3]
        np.testing.assert_allclose(corr_areas, gt_areas, rtol=1e-5)

    def test_correct_ar_preserves_area(self):
        preds, gts = _ar_wrong(5)
        corrected = ErrorBudgetAnalyzer._correct_aspect_ratio(preds, gts)
        pred_areas = preds[:, 2] * preds[:, 3]
        corr_areas = corrected[:, 2] * corrected[:, 3]
        np.testing.assert_allclose(corr_areas, pred_areas, rtol=1e-4)

    def test_correct_ar_matches_gt_aspect_ratio(self):
        preds, gts = _ar_wrong(5)
        corrected = ErrorBudgetAnalyzer._correct_aspect_ratio(preds, gts)
        gt_ar = gts[:, 2] / gts[:, 3]
        corr_ar = corrected[:, 2] / corrected[:, 3]
        np.testing.assert_allclose(corr_ar, gt_ar, rtol=1e-4)


# ---------------------------------------------------------------------------
# ErrorBudgetAnalyzer.aggregate
# ---------------------------------------------------------------------------

class TestAggregate:
    def setup_method(self):
        self.analyzer = ErrorBudgetAnalyzer()

    def _make_seq_budget(self, n_frames, center, scale, ar, residual, iou_loss):
        empty = np.array([], dtype=np.float64)
        return ErrorBudget(
            tracker_name="T",
            sequence_name="S",
            num_frames=n_frames,
            mean_iou_loss=iou_loss,
            center_error=center,
            scale_error=scale,
            aspect_ratio_error=ar,
            residual_error=residual,
            dominant_axis=max(
                {"center": center, "scale": scale, "aspect_ratio": ar, "residual": residual},
                key=lambda k: {"center": center, "scale": scale,
                               "aspect_ratio": ar, "residual": residual}[k]
            ),
            center_contributions=np.full(n_frames, center),
            scale_contributions=np.full(n_frames, scale),
            aspect_ratio_contributions=np.full(n_frames, ar),
        )

    def test_aggregate_single(self):
        b = self._make_seq_budget(100, 0.5, 0.3, 0.1, 0.1, 0.4)
        agg = self.analyzer.aggregate([b], tracker_name="T", dataset_name="D")
        assert agg.center_error == pytest.approx(0.5, abs=1e-6)
        assert agg.num_frames == 100
        assert agg.num_sequences == 1

    def test_aggregate_weighted(self):
        """Frame-weighted mean — heavier sequence dominates."""
        b1 = self._make_seq_budget(10, 0.8, 0.1, 0.05, 0.05, 0.5)  # small
        b2 = self._make_seq_budget(90, 0.2, 0.6, 0.1, 0.1, 0.3)   # large
        agg = self.analyzer.aggregate([b1, b2], tracker_name="T", dataset_name="D")
        # Expected weighted center: (10*0.8 + 90*0.2) / 100 = 0.26
        assert agg.center_error == pytest.approx((10 * 0.8 + 90 * 0.2) / 100, abs=1e-6)
        assert agg.num_frames == 100
        assert agg.num_sequences == 2

    def test_aggregate_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            self.analyzer.aggregate([], tracker_name="T", dataset_name="D")

    def test_aggregate_dominant_axis(self):
        b = self._make_seq_budget(50, 0.1, 0.7, 0.1, 0.1, 0.4)
        agg = self.analyzer.aggregate([b])
        assert agg.dominant_axis == "scale"

    def test_aggregate_to_dict(self):
        b = self._make_seq_budget(20, 0.5, 0.3, 0.1, 0.1, 0.4)
        agg = self.analyzer.aggregate([b], tracker_name="KCF", dataset_name="OTB")
        d = agg.to_dict()
        assert d["tracker"] == "KCF"
        assert d["dataset"] == "OTB"
        assert "center_error" in d

    def test_aggregate_str_representation(self):
        b = self._make_seq_budget(10, 0.5, 0.3, 0.1, 0.1, 0.4)
        agg = self.analyzer.aggregate([b], tracker_name="MOSSE")
        s = str(agg)
        assert "MOSSE" in s


# ---------------------------------------------------------------------------
# to_markdown_table
# ---------------------------------------------------------------------------

class TestMarkdownTable:
    def _make_agg(self, tracker, center, scale, ar, residual, dominant):
        return AggregateErrorBudget(
            tracker_name=tracker,
            dataset_name="OTB100",
            num_sequences=10,
            num_frames=1000,
            mean_iou_loss=0.3,
            center_error=center,
            scale_error=scale,
            aspect_ratio_error=ar,
            residual_error=residual,
            dominant_axis=dominant,
        )

    def test_table_contains_tracker_names(self):
        aggs = [
            self._make_agg("MOSSE", 0.7, 0.2, 0.05, 0.05, "center"),
            self._make_agg("KCF", 0.4, 0.45, 0.08, 0.07, "scale"),
        ]
        table = ErrorBudgetAnalyzer.to_markdown_table(aggs)
        assert "MOSSE" in table
        assert "KCF" in table

    def test_table_has_header_row(self):
        table = ErrorBudgetAnalyzer.to_markdown_table([
            self._make_agg("T", 0.5, 0.3, 0.1, 0.1, "center")
        ])
        assert "Tracker" in table
        assert "Center %" in table
        assert "Scale %" in table

    def test_empty_list(self):
        table = ErrorBudgetAnalyzer.to_markdown_table([])
        assert "Tracker" in table
        lines = [l for l in table.splitlines() if l.strip()]
        assert len(lines) == 2  # header + separator only

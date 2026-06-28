"""Tests for eovot.metrics.challenge — challenge attribute analysis."""

import numpy as np
import pytest

from eovot.metrics.challenge import (
    ATTRIBUTES,
    FM,
    SV,
    LR,
    ARC,
    OV,
    ChallengeDetector,
    ChallengeAnalyzer,
    SequenceAttributeLabels,
    AttributeAccuracy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_static_gt(n: int, x=100, y=100, w=50, h=50) -> np.ndarray:
    """Static target at fixed position."""
    gt = np.zeros((n, 4), dtype=np.float64)
    gt[:] = [x, y, w, h]
    return gt


def _make_fast_motion_gt(n: int, velocity: float = 30.0) -> np.ndarray:
    """Target moving fast in x direction."""
    gt = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        gt[i] = [velocity * i, 100.0, 50.0, 50.0]
    return gt


def _make_scale_variation_gt(n: int) -> np.ndarray:
    """Target alternating between large and small scale."""
    gt = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        size = 80.0 if i % 2 == 0 else 30.0
        gt[i] = [100.0, 100.0, size, size]
    return gt


def _make_low_res_gt(n: int) -> np.ndarray:
    """Tiny target (10×10 = 100 px²) always below LR threshold."""
    gt = np.zeros((n, 4), dtype=np.float64)
    gt[:] = [100, 100, 10, 10]
    return gt


# ---------------------------------------------------------------------------
# ChallengeDetector tests
# ---------------------------------------------------------------------------

class TestChallengeDetector:
    def test_output_shape(self):
        detector = ChallengeDetector()
        gt = _make_static_gt(20)
        labels = detector.label_sequence(gt, sequence_name="test")
        assert labels.labels.shape == (20, len(ATTRIBUTES))
        assert labels.labels.dtype == bool
        assert labels.sequence_name == "test"

    def test_static_target_no_fm_sv(self):
        detector = ChallengeDetector()
        gt = _make_static_gt(30)
        labels = detector.label_sequence(gt)
        assert not labels.labels[:, ATTRIBUTES.index(FM)].any(), "No FM on static target"
        assert not labels.labels[:, ATTRIBUTES.index(SV)].any(), "No SV on static target"

    def test_fast_motion_detected(self):
        detector = ChallengeDetector(fm_velocity_px=20.0)
        gt = _make_fast_motion_gt(10, velocity=40.0)
        labels = detector.label_sequence(gt)
        fm_col = labels.labels[:, ATTRIBUTES.index(FM)]
        # Frame 0 has no prior, but frames 1+ should be FM
        assert fm_col[1:].all(), "All frames after first should be FastMotion"
        assert not fm_col[0], "First frame cannot be FastMotion"

    def test_no_fast_motion_when_slow(self):
        detector = ChallengeDetector(fm_velocity_px=50.0)
        gt = _make_fast_motion_gt(10, velocity=5.0)
        labels = detector.label_sequence(gt)
        fm_col = labels.labels[:, ATTRIBUTES.index(FM)]
        assert not fm_col.any(), "No FM when moving slowly"

    def test_scale_variation_detected(self):
        detector = ChallengeDetector(sv_area_ratio=0.25)
        gt = _make_scale_variation_gt(10)
        labels = detector.label_sequence(gt)
        sv_col = labels.labels[:, ATTRIBUTES.index(SV)]
        # Frames 1, 3, 5, 7, 9 alternate scale dramatically
        assert sv_col[1:].any(), "SV should be detected on scale-alternating target"

    def test_low_resolution_detected(self):
        detector = ChallengeDetector(lr_area_threshold=400.0)
        gt = _make_low_res_gt(10)
        labels = detector.label_sequence(gt)
        lr_col = labels.labels[:, ATTRIBUTES.index(LR)]
        assert lr_col.all(), "All frames should be LowResolution for 10×10 target"

    def test_low_resolution_not_triggered_for_normal_target(self):
        detector = ChallengeDetector(lr_area_threshold=400.0)
        gt = _make_static_gt(10, w=50, h=50)  # 2500 px² >> 400
        labels = detector.label_sequence(gt)
        lr_col = labels.labels[:, ATTRIBUTES.index(LR)]
        assert not lr_col.any()

    def test_aspect_ratio_change_detected(self):
        detector = ChallengeDetector(arc_ratio_threshold=0.4)
        gt = np.zeros((6, 4))
        # Alternate between wide (100×20) and tall (20×100)
        for i in range(6):
            if i % 2 == 0:
                gt[i] = [0, 0, 100, 20]
            else:
                gt[i] = [0, 0, 20, 100]
        labels = detector.label_sequence(gt)
        arc_col = labels.labels[:, ATTRIBUTES.index(ARC)]
        assert arc_col[1:].any()

    def test_out_of_view_with_frame_dims(self):
        detector = ChallengeDetector(
            frame_width=640, frame_height=480, ov_border_fraction=0.05
        )
        # Target near top-left corner — should trigger OV
        gt = np.array([[2.0, 2.0, 10.0, 10.0]] * 5)
        labels = detector.label_sequence(gt)
        ov_col = labels.labels[:, ATTRIBUTES.index(OV)]
        assert ov_col.all(), "Target in top-left corner should be OutOfView"

    def test_out_of_view_not_triggered_in_frame_centre(self):
        detector = ChallengeDetector(
            frame_width=640, frame_height=480, ov_border_fraction=0.05
        )
        gt = np.array([[290.0, 220.0, 50.0, 50.0]] * 5)
        labels = detector.label_sequence(gt)
        ov_col = labels.labels[:, ATTRIBUTES.index(OV)]
        assert not ov_col.any()

    def test_out_of_view_disabled_without_frame_dims(self):
        detector = ChallengeDetector(frame_width=None, frame_height=None)
        gt = np.array([[0.0, 0.0, 5.0, 5.0]] * 5)
        labels = detector.label_sequence(gt)
        ov_col = labels.labels[:, ATTRIBUTES.index(OV)]
        assert not ov_col.any(), "OV requires frame dims to be set"

    def test_invalid_gt_shape_raises(self):
        detector = ChallengeDetector()
        with pytest.raises(ValueError):
            detector.label_sequence(np.ones((10, 3)))

    def test_single_frame_no_crash(self):
        detector = ChallengeDetector()
        gt = _make_static_gt(1)
        labels = detector.label_sequence(gt)
        assert labels.labels.shape == (1, len(ATTRIBUTES))

    def test_coverage_dict(self):
        detector = ChallengeDetector(lr_area_threshold=400.0)
        gt = _make_low_res_gt(10)
        labels = detector.label_sequence(gt)
        cov = labels.coverage()
        assert set(cov.keys()) == set(ATTRIBUTES)
        assert abs(cov[LR] - 1.0) < 1e-9

    def test_active_frames_returns_correct_indices(self):
        detector = ChallengeDetector(fm_velocity_px=20.0)
        gt = _make_fast_motion_gt(5, velocity=40.0)
        labels = detector.label_sequence(gt)
        fm_frames = labels.active_frames(FM)
        assert 0 not in fm_frames
        assert 1 in fm_frames

    def test_active_frames_unknown_attribute_raises(self):
        detector = ChallengeDetector()
        labels = detector.label_sequence(_make_static_gt(5))
        with pytest.raises(KeyError):
            labels.active_frames("Nonexistent")


# ---------------------------------------------------------------------------
# ChallengeAnalyzer tests
# ---------------------------------------------------------------------------

class TestChallengeAnalyzer:
    def _make_perfect_preds(self, gt: np.ndarray) -> np.ndarray:
        return gt.copy()

    def _make_random_preds(self, gt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        noise = rng.uniform(-15, 15, size=gt.shape)
        return np.clip(gt + noise, 0, None)

    def test_breakdown_returns_list_of_attribute_accuracy(self):
        rng = np.random.default_rng(0)
        gt = _make_fast_motion_gt(20, velocity=40.0)
        preds = self._make_random_preds(gt, rng)

        detector = ChallengeDetector(fm_velocity_px=20.0)
        labels = detector.label_sequence(gt, "seq1")

        analyzer = ChallengeAnalyzer()
        report = analyzer.compute_breakdown(preds, gt, labels, tracker_name="T1")

        assert isinstance(report, list)
        assert all(isinstance(e, AttributeAccuracy) for e in report)

    def test_breakdown_perfect_predictions_iou_one(self):
        gt = _make_fast_motion_gt(15, velocity=30.0)
        preds = self._make_perfect_preds(gt)

        detector = ChallengeDetector(fm_velocity_px=20.0)
        labels = detector.label_sequence(gt)

        analyzer = ChallengeAnalyzer()
        report = analyzer.compute_breakdown(preds, gt, labels, tracker_name="Perfect")

        for entry in report:
            assert abs(entry.mean_iou - 1.0) < 1e-6, (
                f"{entry.attribute}: expected mIoU=1.0, got {entry.mean_iou}"
            )

    def test_breakdown_skips_inactive_attributes(self):
        gt = _make_static_gt(20)  # No FM, no SV, no ARC
        preds = gt.copy()
        detector = ChallengeDetector(fm_velocity_px=20.0)
        labels = detector.label_sequence(gt)

        analyzer = ChallengeAnalyzer()
        report = analyzer.compute_breakdown(preds, gt, labels)
        active_attrs = {e.attribute for e in report}

        assert FM not in active_attrs, "No FastMotion on static target"
        assert SV not in active_attrs, "No ScaleVariation on static target"

    def test_breakdown_n_frames_matches_active_count(self):
        gt = _make_fast_motion_gt(10, velocity=40.0)
        preds = gt.copy()
        detector = ChallengeDetector(fm_velocity_px=20.0)
        labels = detector.label_sequence(gt)

        analyzer = ChallengeAnalyzer()
        report = analyzer.compute_breakdown(preds, gt, labels)

        for entry in report:
            expected = int(labels.labels[:, ATTRIBUTES.index(entry.attribute)].sum())
            assert entry.n_frames == expected

    def test_aggregate_weighted_average(self):
        gt1 = _make_fast_motion_gt(10, velocity=40.0)
        gt2 = _make_fast_motion_gt(20, velocity=40.0)
        preds1 = gt1.copy()
        preds2 = gt2.copy()

        detector = ChallengeDetector(fm_velocity_px=20.0)
        labels1 = detector.label_sequence(gt1, "s1")
        labels2 = detector.label_sequence(gt2, "s2")

        analyzer = ChallengeAnalyzer()
        r1 = analyzer.compute_breakdown(preds1, gt1, labels1, tracker_name="T")
        r2 = analyzer.compute_breakdown(preds2, gt2, labels2, tracker_name="T")

        agg = analyzer.aggregate([r1, r2])
        assert FM in agg
        assert abs(agg[FM].mean_iou - 1.0) < 1e-6

    def test_to_markdown_table_returns_string(self):
        gt = _make_fast_motion_gt(15, velocity=40.0)
        preds = gt.copy()
        detector = ChallengeDetector(fm_velocity_px=20.0)
        labels = detector.label_sequence(gt)
        analyzer = ChallengeAnalyzer()
        report = analyzer.compute_breakdown(preds, gt, labels, "T")
        agg = analyzer.aggregate([report])
        table = analyzer.to_markdown_table(agg, tracker_name="T")
        assert "| Attribute |" in table
        assert "FastMotion" in table

    def test_compare_trackers_table(self):
        gt = _make_fast_motion_gt(15, velocity=40.0)
        preds_a = gt.copy()
        preds_b = gt.copy()
        preds_b[:, 0] += 5  # slight offset

        detector = ChallengeDetector(fm_velocity_px=20.0)
        labels = detector.label_sequence(gt)
        analyzer = ChallengeAnalyzer()

        r_a = analyzer.compute_breakdown(preds_a, gt, labels, "A")
        r_b = analyzer.compute_breakdown(preds_b, gt, labels, "B")

        comparison = analyzer.compare_trackers({
            "A": analyzer.aggregate([r_a]),
            "B": analyzer.aggregate([r_b]),
        })
        assert "mIoU A" in comparison
        assert "mIoU B" in comparison

    def test_empty_reports_aggregate(self):
        analyzer = ChallengeAnalyzer()
        agg = analyzer.aggregate([])
        assert agg == {}

    def test_breakdown_length_mismatch_handled(self):
        gt = _make_fast_motion_gt(20, velocity=40.0)
        preds = gt[:15].copy()  # shorter than gt

        detector = ChallengeDetector(fm_velocity_px=20.0)
        labels = detector.label_sequence(gt)
        analyzer = ChallengeAnalyzer()
        # Should not crash despite length mismatch
        report = analyzer.compute_breakdown(preds, gt, labels, tracker_name="T")
        assert isinstance(report, list)

"""Tests for :mod:`eovot.utils.bbox_utils`."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.utils import (
    bbox_area,
    bbox_center,
    batch_giou,
    batch_iou,
    clip_bbox,
    is_valid_bbox,
    xywh_to_xyxy,
    xyxy_to_xywh,
)


# ---------------------------------------------------------------------- #
# Format conversion                                                        #
# ---------------------------------------------------------------------- #


class TestFormatConversion:
    def test_xywh_to_xyxy_single(self):
        out = xywh_to_xyxy((10.0, 20.0, 30.0, 40.0))
        np.testing.assert_allclose(out, [10.0, 20.0, 40.0, 60.0])

    def test_xyxy_to_xywh_single(self):
        out = xyxy_to_xywh((10.0, 20.0, 40.0, 60.0))
        np.testing.assert_allclose(out, [10.0, 20.0, 30.0, 40.0])

    def test_roundtrip_batch(self):
        rng = np.random.default_rng(0)
        boxes = rng.uniform(0, 100, size=(20, 4))
        boxes[:, 2:] = np.maximum(boxes[:, 2:], 1e-3)   # positive w, h
        recovered = xyxy_to_xywh(xywh_to_xyxy(boxes))
        np.testing.assert_allclose(recovered, boxes, atol=1e-9)

    def test_xywh_to_xyxy_batch_shape(self):
        boxes = np.array([[0, 0, 10, 10], [5, 5, 20, 20]], dtype=np.float64)
        out = xywh_to_xyxy(boxes)
        assert out.shape == (2, 4)
        np.testing.assert_allclose(out, [[0, 0, 10, 10], [5, 5, 25, 25]])


# ---------------------------------------------------------------------- #
# Geometry                                                                 #
# ---------------------------------------------------------------------- #


class TestBboxArea:
    def test_scalar(self):
        assert bbox_area((0.0, 0.0, 4.0, 5.0)) == 20.0

    def test_negative_clamped(self):
        assert bbox_area((0.0, 0.0, -3.0, 5.0)) == 0.0

    def test_batch(self):
        boxes = np.array([[0, 0, 4, 5], [0, 0, -3, 5], [1, 2, 2, 2]])
        np.testing.assert_allclose(bbox_area(boxes), [20.0, 0.0, 4.0])


class TestBboxCenter:
    def test_scalar(self):
        c = bbox_center((10.0, 20.0, 4.0, 6.0))
        np.testing.assert_allclose(c, [12.0, 23.0])

    def test_batch(self):
        boxes = np.array([[0, 0, 10, 10], [5, 5, 4, 6]], dtype=np.float64)
        c = bbox_center(boxes)
        np.testing.assert_allclose(c, [[5.0, 5.0], [7.0, 8.0]])


class TestClipBbox:
    def test_inside_frame_unchanged(self):
        out = clip_bbox((10.0, 10.0, 20.0, 20.0), frame_shape=(100, 100))
        np.testing.assert_allclose(out, [10.0, 10.0, 20.0, 20.0])

    def test_right_edge_trimmed(self):
        out = clip_bbox((90.0, 10.0, 30.0, 20.0), frame_shape=(100, 100))
        # x2 = 90 + 30 = 120 → clamped to 100 → w = 10
        np.testing.assert_allclose(out, [90.0, 10.0, 10.0, 20.0])

    def test_bottom_edge_trimmed(self):
        out = clip_bbox((10.0, 90.0, 20.0, 30.0), frame_shape=(100, 100))
        np.testing.assert_allclose(out, [10.0, 90.0, 20.0, 10.0])

    def test_negative_origin_shifted_in(self):
        # top-left of the box is off-screen; clip should shrink, not shift.
        out = clip_bbox((-5.0, -10.0, 20.0, 25.0), frame_shape=(100, 100))
        # x1 clipped to 0, x2 stays 15 → w = 15
        np.testing.assert_allclose(out, [0.0, 0.0, 15.0, 15.0])

    def test_completely_off_screen_becomes_degenerate(self):
        out = clip_bbox((200.0, 200.0, 30.0, 30.0), frame_shape=(100, 100))
        # Both corners clipped to (100, 100) → w = h = 0
        assert out[2] == 0.0 and out[3] == 0.0

    def test_batch(self):
        boxes = np.array(
            [[10, 10, 20, 20], [90, 90, 30, 30], [-5, -10, 20, 25]],
            dtype=np.float64,
        )
        out = clip_bbox(boxes, frame_shape=(100, 100))
        np.testing.assert_allclose(
            out,
            [[10, 10, 20, 20], [90, 90, 10, 10], [0, 0, 15, 15]],
        )


class TestIsValidBbox:
    def test_valid_scalar(self):
        assert is_valid_bbox((0.0, 0.0, 4.0, 5.0)) is True

    def test_zero_dim_scalar(self):
        assert is_valid_bbox((0.0, 0.0, 0.0, 5.0)) is False

    def test_batch(self):
        boxes = np.array([[0, 0, 4, 5], [0, 0, 0, 5], [1, 2, -1, 3]])
        v = is_valid_bbox(boxes)
        np.testing.assert_array_equal(v, [True, False, False])


# ---------------------------------------------------------------------- #
# Overlap metrics                                                          #
# ---------------------------------------------------------------------- #


class TestBatchIou:
    def test_identical_boxes_iou_one(self):
        p = np.array([[10.0, 20.0, 30.0, 40.0]])
        g = p.copy()
        np.testing.assert_allclose(batch_iou(p, g), [1.0])

    def test_disjoint_boxes_iou_zero(self):
        p = np.array([[0.0, 0.0, 10.0, 10.0]])
        g = np.array([[100.0, 100.0, 10.0, 10.0]])
        np.testing.assert_allclose(batch_iou(p, g), [0.0])

    def test_half_overlap(self):
        p = np.array([[0.0, 0.0, 10.0, 10.0]])
        g = np.array([[5.0, 0.0, 10.0, 10.0]])
        # Intersection = 5 × 10 = 50; union = 100 + 100 - 50 = 150
        np.testing.assert_allclose(batch_iou(p, g), [50.0 / 150.0])

    def test_empty_batch(self):
        p = np.zeros((0, 4))
        g = np.zeros((0, 4))
        out = batch_iou(p, g)
        assert out.shape == (0,)

    def test_matches_metrics_engine(self):
        """Values must agree with the existing MetricsEngine.batch_iou."""
        from eovot.metrics.accuracy import MetricsEngine

        rng = np.random.default_rng(42)
        boxes = rng.uniform(0, 100, size=(50, 4))
        boxes[:, 2:] = np.maximum(boxes[:, 2:], 1.0)
        preds = boxes + rng.normal(0, 5, size=boxes.shape)
        preds[:, 2:] = np.maximum(preds[:, 2:], 1.0)

        engine = MetricsEngine()
        np.testing.assert_allclose(batch_iou(preds, boxes), engine.batch_iou(preds, boxes))


class TestBatchGIoU:
    def test_identical_boxes_giou_one(self):
        p = np.array([[10.0, 20.0, 30.0, 40.0]])
        g = p.copy()
        np.testing.assert_allclose(batch_giou(p, g), [1.0])

    def test_disjoint_giou_negative(self):
        """Non-overlapping boxes must produce GIoU in [-1, 0)."""
        p = np.array([[0.0, 0.0, 10.0, 10.0]])
        g = np.array([[100.0, 100.0, 10.0, 10.0]])
        val = batch_giou(p, g)[0]
        assert -1.0 <= val < 0.0

    def test_giou_leq_iou(self):
        """GIoU must never exceed IoU (equal only for full overlap)."""
        rng = np.random.default_rng(7)
        boxes = rng.uniform(0, 100, size=(30, 4))
        boxes[:, 2:] = np.maximum(boxes[:, 2:], 5.0)
        preds = boxes + rng.normal(0, 20, size=boxes.shape)
        preds[:, 2:] = np.maximum(preds[:, 2:], 5.0)
        iou = batch_iou(preds, boxes)
        giou = batch_giou(preds, boxes)
        assert np.all(giou <= iou + 1e-9)

    def test_giou_bounds(self):
        rng = np.random.default_rng(11)
        boxes = rng.uniform(-50, 150, size=(50, 4))
        boxes[:, 2:] = np.maximum(boxes[:, 2:], 1.0)
        preds = rng.uniform(-50, 150, size=(50, 4))
        preds[:, 2:] = np.maximum(preds[:, 2:], 1.0)
        giou = batch_giou(preds, boxes)
        assert np.all(giou >= -1.0 - 1e-9)
        assert np.all(giou <= 1.0 + 1e-9)

    def test_far_disjoint_giou_approaches_neg1(self):
        """Very distant boxes drive GIoU close to -1."""
        p = np.array([[0.0, 0.0, 1.0, 1.0]])
        g = np.array([[10_000.0, 10_000.0, 1.0, 1.0]])
        val = batch_giou(p, g)[0]
        assert val < -0.999

    def test_empty_input(self):
        out = batch_giou(np.zeros((0, 4)), np.zeros((0, 4)))
        assert out.shape == (0,)

    def test_degenerate_boxes_return_zero(self):
        p = np.array([[0.0, 0.0, 0.0, 10.0]])
        g = np.array([[0.0, 0.0, 10.0, 10.0]])
        np.testing.assert_allclose(batch_giou(p, g), [0.0])

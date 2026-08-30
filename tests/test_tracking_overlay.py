"""Tests for the tracking overlay renderer.

All tests run without a real dataset or GPU — synthetic BGR frames are
constructed in-memory so the suite runs offline in CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest

from eovot.visualization.overlay import (
    OverlayConfig,
    TrackingOverlayRenderer,
    _iou,
    render_frame,
)

BBox = Tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_frame(h: int = 120, w: int = 160) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _frames(n: int, h: int = 120, w: int = 160) -> List[np.ndarray]:
    return [_blank_frame(h, w) for _ in range(n)]


def _gt(n: int) -> List[BBox]:
    return [(10.0, 10.0, 40.0, 40.0) for _ in range(n)]


def _preds(n: int, offset: float = 0.0) -> List[Optional[BBox]]:
    return [(10.0 + offset, 10.0, 40.0, 40.0) for _ in range(n)]


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

class TestIou:
    def test_identical_boxes(self):
        assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_non_overlapping_boxes(self):
        assert _iou((0, 0, 10, 10), (20, 20, 10, 10)) == pytest.approx(0.0)

    def test_partial_overlap(self):
        iou = _iou((0, 0, 10, 10), (5, 0, 10, 10))
        assert 0 < iou < 1

    def test_symmetry(self):
        a = (2, 3, 8, 7)
        b = (5, 5, 6, 9)
        assert _iou(a, b) == pytest.approx(_iou(b, a))

    def test_zero_area_box(self):
        assert _iou((0, 0, 0, 0), (0, 0, 10, 10)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# render_frame
# ---------------------------------------------------------------------------

class TestRenderFrame:
    def test_returns_ndarray(self):
        frame = _blank_frame()
        result = render_frame(frame, (10, 10, 40, 40), {}, frame_idx=0)
        assert isinstance(result, np.ndarray)

    def test_does_not_modify_input(self):
        frame = _blank_frame()
        original = frame.copy()
        render_frame(frame, (10, 10, 40, 40), {"T": (10, 10, 40, 40)})
        np.testing.assert_array_equal(frame, original)

    def test_output_shape_matches_input(self):
        frame = _blank_frame(80, 100)
        result = render_frame(frame, None, {})
        assert result.shape == frame.shape

    def test_no_gt_no_crash(self):
        frame = _blank_frame()
        result = render_frame(frame, None, {"MOSSE": (10, 10, 40, 40)})
        assert result.shape == frame.shape

    def test_none_prediction_skipped(self):
        frame = _blank_frame()
        result = render_frame(frame, (10, 10, 30, 30), {"MOSSE": None})
        assert result.shape == frame.shape

    def test_multiple_trackers(self):
        frame = _blank_frame()
        preds = {"A": (10, 10, 30, 30), "B": (15, 15, 25, 25), "C": None}
        result = render_frame(frame, (10, 10, 30, 30), preds)
        assert result.shape == frame.shape

    def test_show_iou_false(self):
        cfg = OverlayConfig(show_iou=False, show_legend=False, show_frame_number=False)
        frame = _blank_frame()
        result = render_frame(frame, (10, 10, 30, 30), {"T": (10, 10, 30, 30)}, config=cfg)
        assert result.shape == frame.shape

    def test_show_legend_false(self):
        cfg = OverlayConfig(show_legend=False)
        frame = _blank_frame()
        result = render_frame(frame, (10, 10, 30, 30), {"T": (10, 10, 30, 30)}, config=cfg)
        assert result.shape == frame.shape

    def test_empty_predictions(self):
        frame = _blank_frame()
        result = render_frame(frame, (10, 10, 30, 30), {})
        assert result.shape == frame.shape

    def test_annotated_frame_differs_from_blank(self):
        frame = _blank_frame()
        result = render_frame(frame, (10, 10, 50, 50), {"T": (10, 10, 50, 50)})
        assert not np.array_equal(result, frame)

    def test_frame_idx_shown(self):
        frame = _blank_frame()
        cfg = OverlayConfig(show_frame_number=True)
        result = render_frame(frame, None, {}, frame_idx=42, config=cfg)
        # Any annotation changes at least one pixel
        assert not np.array_equal(result, frame)


# ---------------------------------------------------------------------------
# OverlayConfig
# ---------------------------------------------------------------------------

class TestOverlayConfig:
    def test_default_gt_color_is_green(self):
        cfg = OverlayConfig()
        assert cfg.gt_color == (0, 255, 0)

    def test_custom_palette_applied(self):
        custom_color = (100, 200, 50)
        cfg = OverlayConfig(pred_colors=[custom_color])
        assert cfg.pred_colors[0] == custom_color

    def test_show_flags(self):
        cfg = OverlayConfig(show_gt=False, show_iou=False, show_legend=False)
        assert not cfg.show_gt
        assert not cfg.show_iou
        assert not cfg.show_legend


# ---------------------------------------------------------------------------
# TrackingOverlayRenderer — render_sequence
# ---------------------------------------------------------------------------

class TestRenderSequence:
    def test_creates_output_directory(self, tmp_path: Path):
        renderer = TrackingOverlayRenderer(output_dir=str(tmp_path / "out"))
        renderer.render_sequence(
            "seq1",
            frames=_frames(5),
            gt_bboxes=_gt(5),
            predictions={"T": _preds(5)},
            save_video=False,
            save_frames=False,
        )
        assert (tmp_path / "out" / "seq1").is_dir()

    def test_save_frames_writes_jpegs(self, tmp_path: Path):
        renderer = TrackingOverlayRenderer(output_dir=str(tmp_path))
        renderer.render_sequence(
            "seq1",
            frames=_frames(4),
            save_video=False,
            save_frames=True,
        )
        jpegs = list((tmp_path / "seq1" / "frames").glob("*.jpg"))
        assert len(jpegs) == 4

    def test_returns_path_to_seq_dir(self, tmp_path: Path):
        renderer = TrackingOverlayRenderer(output_dir=str(tmp_path))
        out = renderer.render_sequence(
            "seq2",
            frames=_frames(3),
            save_video=False,
        )
        assert isinstance(out, Path)
        assert out.name == "seq2"

    def test_empty_frames_raises(self, tmp_path: Path):
        renderer = TrackingOverlayRenderer(output_dir=str(tmp_path))
        with pytest.raises(ValueError, match="empty"):
            renderer.render_sequence("empty_seq", frames=[])

    def test_none_gt_bboxes_accepted(self, tmp_path: Path):
        renderer = TrackingOverlayRenderer(output_dir=str(tmp_path))
        renderer.render_sequence(
            "seq_no_gt",
            frames=_frames(3),
            gt_bboxes=None,
            save_video=False,
        )
        assert (tmp_path / "seq_no_gt").is_dir()

    def test_multiple_tracker_predictions(self, tmp_path: Path):
        renderer = TrackingOverlayRenderer(output_dir=str(tmp_path))
        preds = {
            "MOSSE": _preds(5),
            "KCF": _preds(5, offset=2.0),
            "CSRT": [None] * 5,
        }
        renderer.render_sequence(
            "multi_tracker",
            frames=_frames(5),
            gt_bboxes=_gt(5),
            predictions=preds,
            save_video=False,
        )
        assert (tmp_path / "multi_tracker").is_dir()

    def test_gt_shorter_than_frames(self, tmp_path: Path):
        renderer = TrackingOverlayRenderer(output_dir=str(tmp_path))
        renderer.render_sequence(
            "short_gt",
            frames=_frames(10),
            gt_bboxes=_gt(5),       # only 5 GT for 10 frames
            save_video=False,
        )
        assert (tmp_path / "short_gt").is_dir()


# ---------------------------------------------------------------------------
# TrackingOverlayRenderer — render_comparison_strip
# ---------------------------------------------------------------------------

class TestRenderComparisonStrip:
    def test_returns_ndarray(self, tmp_path: Path):
        renderer = TrackingOverlayRenderer(output_dir=str(tmp_path))
        strip = renderer.render_comparison_strip(
            _blank_frame(80, 100),
            _gt(5),
            {"A": _preds(5), "B": _preds(5, 2.0)},
            frame_idx=0,
        )
        assert isinstance(strip, np.ndarray)

    def test_strip_is_wider_than_single_frame(self, tmp_path: Path):
        renderer = TrackingOverlayRenderer(output_dir=str(tmp_path))
        frame = _blank_frame(80, 100)
        strip = renderer.render_comparison_strip(
            frame,
            _gt(5),
            {"A": _preds(5), "B": _preds(5)},
            frame_idx=0,
        )
        assert strip.shape[1] > frame.shape[1]

    def test_single_tracker_strip_same_width(self, tmp_path: Path):
        renderer = TrackingOverlayRenderer(output_dir=str(tmp_path))
        frame = _blank_frame(80, 100)
        strip = renderer.render_comparison_strip(
            frame,
            _gt(5),
            {"A": _preds(5)},
            frame_idx=0,
        )
        assert strip.shape[0] == frame.shape[0]

    def test_empty_predictions_returns_copy(self, tmp_path: Path):
        renderer = TrackingOverlayRenderer(output_dir=str(tmp_path))
        frame = _blank_frame()
        strip = renderer.render_comparison_strip(frame, None, {}, frame_idx=0)
        np.testing.assert_array_equal(strip, frame)

    def test_no_gt_bboxes(self, tmp_path: Path):
        renderer = TrackingOverlayRenderer(output_dir=str(tmp_path))
        strip = renderer.render_comparison_strip(
            _blank_frame(),
            None,
            {"T": _preds(5)},
            frame_idx=2,
        )
        assert isinstance(strip, np.ndarray)


# ---------------------------------------------------------------------------
# Integration: import from top-level visualization package
# ---------------------------------------------------------------------------

def test_overlay_importable_from_package():
    from eovot.visualization import TrackingOverlayRenderer, OverlayConfig, render_frame
    assert TrackingOverlayRenderer is not None
    assert OverlayConfig is not None
    assert render_frame is not None

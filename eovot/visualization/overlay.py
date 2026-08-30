"""Tracking overlay renderer for qualitative benchmark inspection.

Renders per-frame ground-truth and predicted bounding boxes onto raw video
frames, enabling direct visual comparison of tracker behaviour across
challenge conditions.  Results can be saved as individual JPEG frames or
as an MP4 video.

Typical usage::

    from eovot.visualization.overlay import TrackingOverlayRenderer

    renderer = TrackingOverlayRenderer(output_dir="out/overlays")
    seq_out = renderer.render_sequence(
        sequence_name="ball_seq",
        frames=[frame1, frame2, ...],         # BGR NumPy arrays
        gt_bboxes=[(x, y, w, h), ...],        # per-frame GT
        predictions={
            "MOSSE": [(x, y, w, h), ...],
            "KCF":   [(x, y, w, h), ...],
        },
        save_video=True,
    )
    print("Overlay saved to", seq_out)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BBox = Tuple[float, float, float, float]  # x, y, w, h (top-left + size)
_Color = Tuple[int, int, int]             # BGR

# Default palette — perceptually distinct colours for up to 8 trackers.
_PALETTE: List[_Color] = [
    (0, 165, 255),   # orange
    (255,  80,  80),  # blue
    (80, 200,  80),  # green
    (200,  80, 200),  # magenta
    (0, 220, 220),   # yellow
    (220, 220,   0),  # cyan
    (80,  80, 200),  # red-ish
    (180, 100, 255),  # purple
]
_GT_COLOR: _Color = (0, 255, 0)           # bright green for ground truth


@dataclass
class OverlayConfig:
    """Visual configuration for the overlay renderer.

    All colour values are in BGR (OpenCV convention).

    Attributes:
        show_gt:            Draw the ground-truth bounding box when available.
        gt_color:           BGR colour for the GT box and label background.
        gt_label:           Text label drawn above the GT box.
        gt_thickness:       Line thickness of the GT rectangle.
        pred_colors:        Cycle of BGR colours for predicted boxes.
        pred_thickness:     Line thickness of predicted rectangles.
        show_iou:           Append the IoU score to each tracker label.
        show_frame_number:  Overlay the frame index in the top-left corner.
        show_legend:        Draw a semi-transparent legend panel listing
                            each tracker name and its current IoU.
        font_scale:         OpenCV font scale applied to all labels.
        font_thickness:     OpenCV font stroke thickness.
    """

    show_gt: bool = True
    gt_color: _Color = field(default_factory=lambda: _GT_COLOR)
    gt_label: str = "GT"
    gt_thickness: int = 2

    pred_colors: List[_Color] = field(default_factory=lambda: list(_PALETTE))
    pred_thickness: int = 2

    show_iou: bool = True
    show_frame_number: bool = True
    show_legend: bool = True

    font_scale: float = 0.50
    font_thickness: int = 1


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def _iou(a: BBox, b: BBox) -> float:
    """Compute intersection-over-union for two (x, y, w, h) boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Low-level draw helpers
# ---------------------------------------------------------------------------

def _draw_bbox(
    canvas: np.ndarray,
    bbox: BBox,
    color: _Color,
    label: str,
    thickness: int,
    font_face: int,
    font_scale: float,
    font_thickness: int,
) -> None:
    x, y, w, h = (int(round(v)) for v in bbox)
    # Bounding rectangle
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
    # Label background + text
    (tw, th), baseline = cv2.getTextSize(label, font_face, font_scale, font_thickness)
    ly = max(y - 4, th + baseline + 2)
    cv2.rectangle(canvas, (x, ly - th - baseline), (x + tw + 6, ly + baseline + 2), color, -1)
    cv2.putText(canvas, label, (x + 3, ly), font_face, font_scale, (0, 0, 0), font_thickness,
                cv2.LINE_AA)


def _draw_legend(
    canvas: np.ndarray,
    tracker_names: List[str],
    ious: Dict[str, float],
    config: OverlayConfig,
) -> None:
    if not tracker_names:
        return
    entry_h = 22
    pad = 8
    legend_w = 175
    legend_h = pad * 2 + len(tracker_names) * entry_h
    h, w = canvas.shape[:2]
    x0 = w - legend_w - pad
    y0 = pad

    # Semi-transparent dark background
    roi = canvas[y0: y0 + legend_h, x0: x0 + legend_w]
    dark = np.full_like(roi, 30)
    cv2.addWeighted(dark, 0.55, roi, 0.45, 0, roi)
    canvas[y0: y0 + legend_h, x0: x0 + legend_w] = roi

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, name in enumerate(tracker_names):
        color = config.pred_colors[i % len(config.pred_colors)]
        ey = y0 + pad + i * entry_h
        cv2.rectangle(canvas, (x0 + 5, ey + 3), (x0 + 17, ey + 15), color, -1)
        iou_val = ious.get(name, 0.0)
        text = f"{name[:12]} {iou_val:.2f}"
        cv2.putText(canvas, text, (x0 + 22, ey + 14), font,
                    config.font_scale * 0.9, (225, 225, 225), config.font_thickness,
                    cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_frame(
    image: np.ndarray,
    gt_bbox: Optional[BBox],
    predictions: Dict[str, Optional[BBox]],
    frame_idx: int = 0,
    config: Optional[OverlayConfig] = None,
) -> np.ndarray:
    """Annotate a single frame with GT and predicted bounding boxes.

    Args:
        image:       BGR frame as a ``(H, W, 3)`` uint8 NumPy array.
        gt_bbox:     Ground-truth ``(x, y, w, h)`` box, or ``None`` if unavailable.
        predictions: Dict of tracker name → predicted ``(x, y, w, h)`` box (``None``
                     indicates a tracker failure for this frame).
        frame_idx:   Frame index shown in the top-left corner when
                     ``config.show_frame_number`` is True.
        config:      :class:`OverlayConfig` instance; defaults used when ``None``.

    Returns:
        Annotated copy of *image* as a ``(H, W, 3)`` uint8 BGR array.
    """
    cfg = config or OverlayConfig()
    font = cv2.FONT_HERSHEY_SIMPLEX
    canvas = image.copy()

    # Frame counter
    if cfg.show_frame_number:
        label = f"#{frame_idx}"
        cv2.putText(canvas, label, (8, 22), font, cfg.font_scale,
                    (255, 255, 255), cfg.font_thickness + 1, cv2.LINE_AA)
        cv2.putText(canvas, label, (8, 22), font, cfg.font_scale,
                    (0, 0, 0), cfg.font_thickness, cv2.LINE_AA)

    # Ground truth
    if cfg.show_gt and gt_bbox is not None:
        _draw_bbox(canvas, gt_bbox, cfg.gt_color, cfg.gt_label,
                   cfg.gt_thickness, font, cfg.font_scale, cfg.font_thickness)

    # Compute IoUs for legend / labels
    ious: Dict[str, float] = {}
    tracker_names = list(predictions.keys())
    for name, pred in predictions.items():
        if pred is not None and gt_bbox is not None:
            ious[name] = _iou(gt_bbox, pred)
        else:
            ious[name] = 0.0

    # Tracker predictions
    for i, (name, pred) in enumerate(predictions.items()):
        if pred is None:
            continue
        color = cfg.pred_colors[i % len(cfg.pred_colors)]
        iou_str = f" IoU={ious[name]:.2f}" if cfg.show_iou else ""
        label = f"{name}{iou_str}"
        _draw_bbox(canvas, pred, color, label,
                   cfg.pred_thickness, font, cfg.font_scale, cfg.font_thickness)

    # Legend panel
    if cfg.show_legend and tracker_names:
        _draw_legend(canvas, tracker_names, ious, cfg)

    return canvas


class TrackingOverlayRenderer:
    """Save tracking results as annotated video or image frames.

    Renders ground-truth and per-tracker predicted bounding boxes onto raw
    video frames.  Supports multiple trackers on the same frame, an optional
    IoU legend, and both MP4 video output and per-frame JPEG saves.

    Args:
        output_dir: Root directory for output files.  Created if it doesn't
            exist.  Each call to :meth:`render_sequence` creates a
            subdirectory named after the sequence.
        config:     Visual configuration; :class:`OverlayConfig` defaults used
            when ``None``.
        fps:        Frames-per-second for video output (default 25.0).

    Example::

        renderer = TrackingOverlayRenderer("out/overlays")
        renderer.render_sequence(
            sequence_name="seq01",
            frames=[bgr_frame_1, bgr_frame_2, ...],
            gt_bboxes=[(10, 20, 50, 60), ...],
            predictions={"MOSSE": [(10, 20, 50, 60), ...], "KCF": [...]},
            save_video=True,
        )
    """

    def __init__(
        self,
        output_dir: str = "overlay_output",
        config: Optional[OverlayConfig] = None,
        fps: float = 25.0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.config = config or OverlayConfig()
        self.fps = fps

    def render_sequence(
        self,
        sequence_name: str,
        frames: List[np.ndarray],
        gt_bboxes: Optional[List[Optional[BBox]]] = None,
        predictions: Optional[Dict[str, List[Optional[BBox]]]] = None,
        save_video: bool = True,
        save_frames: bool = False,
    ) -> Path:
        """Render and save an annotated sequence.

        Args:
            sequence_name: Used to name the output subdirectory and video file.
            frames:        Ordered list of BGR frames (``H×W×3`` uint8 arrays).
            gt_bboxes:     Per-frame ground-truth boxes; ``None`` entries are
                           treated as "no GT for this frame".
            predictions:   Tracker name → list of per-frame predicted boxes
                           (``None`` entries = tracker failure).
            save_video:    Encode all annotated frames to
                           ``<output_dir>/<sequence_name>/<sequence_name>_overlay.mp4``.
            save_frames:   Save each annotated frame as a JPEG under
                           ``<output_dir>/<sequence_name>/frames/``.

        Returns:
            Path to the sequence output directory.

        Raises:
            ValueError: If *frames* is empty.
        """
        if not frames:
            raise ValueError("frames list must not be empty")

        predictions = predictions or {}
        seq_dir = self.output_dir / sequence_name
        seq_dir.mkdir(parents=True, exist_ok=True)

        h, w = frames[0].shape[:2]
        writer: Optional[cv2.VideoWriter] = None
        video_path: Optional[Path] = None

        if save_video:
            video_path = seq_dir / f"{sequence_name}_overlay.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, (w, h))
            if not writer.isOpened():
                logger.warning(
                    "VideoWriter failed to open '%s' — falling back to frame saves.",
                    video_path,
                )
                writer = None
                save_frames = True

        if save_frames:
            frames_dir = seq_dir / "frames"
            frames_dir.mkdir(exist_ok=True)

        for i, frame in enumerate(frames):
            gt = gt_bboxes[i] if gt_bboxes and i < len(gt_bboxes) else None
            preds: Dict[str, Optional[BBox]] = {
                name: pred_list[i] if i < len(pred_list) else None
                for name, pred_list in predictions.items()
            }
            annotated = render_frame(frame, gt, preds, frame_idx=i, config=self.config)

            if writer is not None:
                writer.write(annotated)
            if save_frames:
                cv2.imwrite(str(frames_dir / f"frame_{i:05d}.jpg"), annotated)

        if writer is not None:
            writer.release()
            logger.info("Saved overlay video: %s", video_path)

        return seq_dir

    def render_comparison_strip(
        self,
        frame: np.ndarray,
        gt_bboxes: Optional[List[Optional[BBox]]],
        predictions: Dict[str, List[Optional[BBox]]],
        frame_idx: int,
    ) -> np.ndarray:
        """Render a horizontal strip with one panel per tracker.

        Each panel shows the same *frame* annotated with a single tracker's
        prediction alongside the ground-truth box.  Panels are concatenated
        horizontally so all trackers can be compared side-by-side.

        Args:
            frame:       Raw BGR frame.
            gt_bboxes:   Per-frame GT list (indexed by *frame_idx*).
            predictions: Tracker name → per-frame prediction list.
            frame_idx:   Index into *gt_bboxes* and each prediction list.

        Returns:
            Wide BGR image (H × (W × N_trackers) × 3).
        """
        gt = gt_bboxes[frame_idx] if gt_bboxes and frame_idx < len(gt_bboxes) else None
        panels: List[np.ndarray] = []

        for i, (name, pred_list) in enumerate(predictions.items()):
            pred = pred_list[frame_idx] if frame_idx < len(pred_list) else None
            color = self.config.pred_colors[i % len(self.config.pred_colors)]
            cfg = OverlayConfig(
                gt_color=self.config.gt_color,
                pred_colors=[color],
                show_legend=False,
                show_iou=self.config.show_iou,
                show_frame_number=self.config.show_frame_number,
                font_scale=self.config.font_scale,
            )
            panel = render_frame(frame, gt, {name: pred}, frame_idx=frame_idx, config=cfg)
            panels.append(panel)

        if not panels:
            return frame.copy()

        target_h = panels[0].shape[0]
        resized = [
            cv2.resize(p, (int(p.shape[1] * target_h / p.shape[0]), target_h))
            if p.shape[0] != target_h else p
            for p in panels
        ]
        return np.concatenate(resized, axis=1)

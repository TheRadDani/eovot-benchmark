"""Resolution-scaling tracker wrapper for edge deployment analysis.

Edge devices often cannot process full-resolution video at the target frame
rate.  Resolution downscaling is a principled alternative to frame skipping:
instead of processing fewer frames, every frame is processed at a smaller
resolution, trading accuracy for throughput.

:class:`ResolutionScaledTracker` wraps any :class:`~.base.BaseTracker` and
transparently downscales frames before passing them to the underlying tracker,
then upscales the predicted bounding box back to original-frame coordinates.

Combined with :class:`~eovot.trackers.frame_skip.FrameSkipTracker`, this
enables a 2-D optimization space for edge deployment:

    throughput_gain ≈ skip_rate / scale_factor²

Example::

    from eovot.trackers.scale import ResolutionScaledTracker
    from eovot.trackers.kcf import KCFTracker
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine

    base    = KCFTracker()
    scaled  = ResolutionScaledTracker(base, scale_factor=0.5)

    dataset = SyntheticDataset(num_sequences=5)
    engine  = BenchmarkEngine(verbose=False)
    result  = engine.run(scaled, dataset, dataset_name="Synthetic")
    print(result)
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import BaseTracker, BBox


class ResolutionScaledTracker(BaseTracker):
    """Tracker wrapper that downscales frames before processing.

    Every call to :meth:`initialize` and :meth:`update` resizes the input
    frame by ``scale_factor`` (both dimensions), runs the underlying tracker
    on the smaller image, and maps the predicted bounding box back to
    original-resolution coordinates.

    Args:
        tracker:      The :class:`~.base.BaseTracker` to wrap.  Must
                      implement ``initialize()`` and ``update()``.
        scale_factor: Linear scale factor applied to both frame dimensions.
                      Must be in ``(0, 1]``.  ``1.0`` is a no-op pass-through.
                      Common values: ``0.75``, ``0.5``, ``0.25``.

    Raises:
        ValueError: If ``scale_factor`` is not in ``(0, 1]``.

    Example::

        tracker = ResolutionScaledTracker(MOSSETracker(), scale_factor=0.5)
        # tracker.name is "MOSSE_scale0.50"
        tracker.initialize(frame, bbox)
        pred = tracker.update(next_frame)
    """

    def __init__(self, tracker: BaseTracker, scale_factor: float = 0.5) -> None:
        if not (0.0 < scale_factor <= 1.0):
            raise ValueError(
                f"scale_factor must be in (0, 1], got {scale_factor}"
            )
        super().__init__(name=f"{tracker.name}_scale{scale_factor:.2f}")
        self._tracker = tracker
        self.scale_factor = scale_factor

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the wrapped tracker on a downscaled version of *frame*.

        Args:
            frame: BGR image ``(H, W, 3)`` at original resolution.
            bbox:  Ground-truth bounding box ``(x, y, w, h)`` in original
                   pixel coordinates.
        """
        scaled_frame = self._scale_frame(frame)
        scaled_bbox = self._scale_bbox_down(bbox)
        self._tracker.initialize(scaled_frame, scaled_bbox)

    def update(self, frame: np.ndarray) -> BBox:
        """Predict target location by running the wrapped tracker at reduced resolution.

        Args:
            frame: BGR image ``(H, W, 3)`` at original resolution.

        Returns:
            Predicted bounding box ``(x, y, w, h)`` in *original* pixel
            coordinates (upscaled from the tracker's reduced-resolution output).
        """
        scaled_frame = self._scale_frame(frame)
        scaled_pred = self._tracker.update(scaled_frame)
        return self._scale_bbox_up(scaled_pred)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scale_frame(self, frame: np.ndarray) -> np.ndarray:
        """Return a resized copy of *frame*, or the original if scale_factor=1."""
        if self.scale_factor == 1.0:
            return frame
        h, w = frame.shape[:2]
        new_w = max(1, int(round(w * self.scale_factor)))
        new_h = max(1, int(round(h * self.scale_factor)))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def _scale_bbox_down(self, bbox: BBox) -> BBox:
        """Scale bbox from original coordinates to reduced-resolution coordinates."""
        x, y, w, h = bbox
        s = self.scale_factor
        return (x * s, y * s, w * s, h * s)

    def _scale_bbox_up(self, bbox: BBox) -> BBox:
        """Scale bbox from reduced-resolution coordinates back to original."""
        x, y, w, h = bbox
        s = self.scale_factor
        return (x / s, y / s, w / s, h / s)

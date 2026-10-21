"""ConfidentBaseTracker — extended tracker interface with per-frame confidence.

For edge deployment the system needs to know *when* the tracker is uncertain
so it can trigger re-detection, raise an alert, or fall back to a slower but
more robust algorithm.  The base :class:`~eovot.trackers.base.BaseTracker`
returns only a bounding box; :class:`ConfidentBaseTracker` extends that
contract by requiring an accompanying confidence score in ``[0, 1]``.

The default :meth:`update` implementation calls :meth:`update_with_score`
and discards the score, so subclasses remain fully compatible with every
existing benchmark loop that only reads ``update()``.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Tuple

import numpy as np

from .base import BaseTracker, BBox


class ConfidentBaseTracker(BaseTracker):
    """Abstract tracker that exposes per-frame confidence alongside the bbox.

    Subclasses implement :meth:`update_with_score`, which returns both the
    predicted bounding box and a confidence value in ``[0, 1]``.

    Confidence semantics are tracker-specific — a correlation filter might
    use Peak-to-Sidelobe Ratio (PSR), a deep tracker might use its final
    softmax score, and a classical tracker might use appearance similarity.
    What matters is monotonicity: higher score ↔ higher expected IoU.

    Example::

        class MyTracker(ConfidentBaseTracker):
            def initialize(self, frame, bbox):
                ...
            def update_with_score(self, frame):
                ...
                return predicted_bbox, confidence  # float in [0, 1]
    """

    @abstractmethod
    def update_with_score(self, frame: np.ndarray) -> Tuple[BBox, float]:
        """Predict the target location and return a confidence score.

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 numpy array.

        Returns:
            ``(bbox, confidence)`` where *bbox* is ``(x, y, w, h)`` and
            *confidence* is a float in ``[0, 1]``.  A value near 1 means the
            tracker is confident; near 0 means it may have lost the target.
        """
        ...

    def update(self, frame: np.ndarray) -> BBox:
        """Calls :meth:`update_with_score` and discards the confidence score.

        This preserves full backwards compatibility with the plain
        :class:`~eovot.trackers.base.BaseTracker` interface used by the
        benchmark engine.
        """
        bbox, _ = self.update_with_score(frame)
        return bbox

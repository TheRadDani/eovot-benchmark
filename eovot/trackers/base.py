"""Abstract base class that every EOVOT tracker must implement."""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

# Bounding box type: (x, y, width, height) in pixel coordinates.
BBox = Tuple[float, float, float, float]


class BaseTracker(ABC):
    """Abstract interface for visual object trackers.

    All trackers in EOVOT must subclass ``BaseTracker`` and implement
    :meth:`initialize` and :meth:`update`.  This contract ensures the
    benchmark engine can drive any tracker in a uniform way.

    Bounding boxes use the ``(x, y, w, h)`` convention throughout, where
    ``(x, y)`` is the top-left corner.

    Trackers that compute an internal confidence signal (e.g. MOSSE's PSR)
    may override :meth:`confidence` to expose it.  The default returns
    ``1.0`` so that wrappers like
    :class:`~eovot.trackers.adaptive_skip.AdaptiveSkipTracker` degrade
    gracefully for trackers that do not implement it.

    Example::

        class MyTracker(BaseTracker):
            def initialize(self, frame, bbox):
                ...
            def update(self, frame):
                ...
                return predicted_bbox
    """

    def __init__(self, name: str) -> None:
        """
        Args:
            name: Human-readable identifier used in benchmark reports.
        """
        self.name = name

    @abstractmethod
    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the tracker on the first frame of a sequence.

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 numpy array.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        ...

    @abstractmethod
    def update(self, frame: np.ndarray) -> BBox:
        """Predict the target location in the current frame.

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 numpy array.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.
        """
        ...

    def confidence(self) -> float:
        """Return a scalar confidence score for the most recent prediction.

        Implementations should normalise the score to ``[0, 1]`` where
        ``1.0`` means the tracker is highly certain about its last prediction
        and ``0.0`` means it has likely lost the target.

        The default implementation returns ``1.0`` (always confident), which
        makes all existing trackers compatible with confidence-aware wrappers
        without any changes.  Trackers with an internal reliability signal
        (e.g. PSR for correlation filters) should override this method.

        Returns:
            Float confidence in ``[0, 1]``.  Higher is more confident.
        """
        return 1.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

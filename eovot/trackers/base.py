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

    Trackers that produce a native confidence signal (e.g. correlation filters
    computing PSR) should also override :meth:`update_with_confidence` to expose
    it without redundant computation.  The default implementation calls
    :meth:`update` and returns a sentinel confidence of ``-1.0``.

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

    def update_with_confidence(self, frame: np.ndarray) -> Tuple[BBox, float]:
        """Predict the target location and return a confidence score.

        The confidence is a scalar in ``[0, 1]`` where 1 means very certain
        and 0 means the tracker has likely lost the target.

        Trackers that derive a native confidence signal from their internal
        state (e.g. PSR from a correlation filter response) should override
        this method to return it efficiently.

        The default implementation delegates to :meth:`update` and returns
        ``-1.0`` as a sentinel indicating that no confidence is available.

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 numpy array.

        Returns:
            Tuple ``(bbox, confidence)`` where ``bbox`` is ``(x, y, w, h)``
            and ``confidence`` is in ``[0, 1]`` or ``-1.0`` if unsupported.
        """
        return self.update(frame), -1.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

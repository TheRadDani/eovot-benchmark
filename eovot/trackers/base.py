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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

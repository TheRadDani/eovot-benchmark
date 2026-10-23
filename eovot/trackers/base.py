"""Abstract base classes that every EOVOT tracker must implement."""

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


class ConfidenceTracker(BaseTracker):
    """Extension of :class:`BaseTracker` for trackers that expose a confidence score.

    Correlation-filter trackers compute a response map on every frame; the
    **Peak-to-Sidelobe Ratio (PSR)** of that map is a reliable, zero-overhead
    proxy for tracking confidence that requires no ground-truth.

    PSR interpretation (Bolme et al., CVPR 2010):

    * PSR ≥ 7.0 — tracker is confident; target is well-localised.
    * 4.0 ≤ PSR < 7.0 — uncertain; possible appearance change or occlusion.
    * PSR < 4.0 — likely tracking failure; consider re-detection.

    Subclasses must implement :meth:`last_confidence` (a property returning the
    PSR computed during the most recent :meth:`~BaseTracker.update` call) in
    addition to the standard :meth:`~BaseTracker.initialize` and
    :meth:`~BaseTracker.update` methods.

    The :meth:`update_with_confidence` convenience method calls
    :meth:`~BaseTracker.update` and returns ``(bbox, confidence)`` together,
    which is useful when the caller needs both without two separate calls.

    Example::

        from eovot.trackers.mosse import MOSSETracker

        tracker = MOSSETracker()
        tracker.initialize(frame, bbox)
        for frame in sequence:
            predicted_bbox, psr = tracker.update_with_confidence(frame)
            if psr < 4.0:
                print("Low confidence — possible tracking failure")
    """

    @property
    @abstractmethod
    def last_confidence(self) -> float:
        """PSR or equivalent confidence score from the most recent :meth:`update` call.

        Returns ``0.0`` before the first :meth:`update` call or after
        :meth:`~BaseTracker.initialize`.
        """
        ...

    def update_with_confidence(self, frame: np.ndarray) -> Tuple[BBox, float]:
        """Run :meth:`update` and return ``(predicted_bbox, confidence)`` together.

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 numpy array.

        Returns:
            Tuple of ``(BBox, float)`` — the predicted bounding box and the
            PSR confidence score for this frame.
        """
        bbox = self.update(frame)
        return bbox, self.last_confidence


def compute_psr(response: np.ndarray, win_size: int = 11) -> float:
    """Compute the Peak-to-Sidelobe Ratio of a correlation response map.

    PSR is defined as::

        PSR = (peak_value - mean_sidelobe) / std_sidelobe

    where the *sidelobe region* is everything outside an ``win_size × win_size``
    window centred on the response peak.  This function is the canonical PSR
    implementation shared by :class:`~eovot.trackers.mosse.MOSSETracker` and
    :class:`~eovot.trackers.kcf.KCFTracker`.

    Args:
        response: 2-D real-valued correlation response map (any shape).
        win_size: Side length of the exclusion window centred on the peak.
            The original paper uses 11; larger values are more conservative.

    Returns:
        PSR as a non-negative float.  Returns ``0.0`` when the response is
        constant (zero variance) or when all pixels fall inside the peak window.

    References:
        Bolme et al., "Visual object tracking using adaptive correlation
        filters," CVPR 2010, Section 4.
    """
    ky, kx = np.unravel_index(response.argmax(), response.shape)
    peak_val = float(response[ky, kx])

    h, w = response.shape
    half = win_size // 2
    y1 = max(0, ky - half)
    y2 = min(h, ky + half + 1)
    x1 = max(0, kx - half)
    x2 = min(w, kx + half + 1)

    mask = np.ones_like(response, dtype=bool)
    mask[y1:y2, x1:x2] = False
    sidelobe = response[mask]

    if sidelobe.size == 0:
        return 0.0

    std_sl = float(sidelobe.std())
    if std_sl < 1e-10:
        return 0.0

    return (peak_val - float(sidelobe.mean())) / std_sl

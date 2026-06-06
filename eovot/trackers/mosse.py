"""MOSSE — Minimum Output Sum of Squared Error correlation filter tracker.

Reference
---------
Bolme, D. S., Beveridge, J. R., Draper, B. A., & Lui, Y. M. (2010).
Visual object tracking using adaptive correlation filters.
IEEE CVPR 2010, pp. 2544–2550.

Design notes
------------
* Pure NumPy + OpenCV — no deep-learning framework required.
* Runs at >500 FPS on a modern CPU core, making it ideal as an
  edge-deployment baseline.
* Uses a cosine window to reduce boundary effects in the FFT.
* Learns the filter online with exponential moving average (EMA) update.
* Gracefully handles partial out-of-bounds patches via ``cv2.resize``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox
from .confidence import compute_psr, psr_to_confidence


class MOSSETracker(BaseTracker):
    """MOSSE correlation filter tracker.

    Args:
        learning_rate: EMA weight for online filter update (0 = no update,
                       1 = replace entirely).  Default 0.125 matches the
                       original paper.
        sigma:         Standard deviation of the Gaussian target response
                       used during initialisation (pixels).  Smaller values
                       produce a sharper, more localised peak.
        psr_threshold: Peak-to-Sidelobe Ratio threshold below which the
                       tracker is considered to have lost the target.  Set
                       to ``None`` to disable failure detection.

    Example::

        import cv2
        from eovot.trackers.mosse import MOSSETracker

        tracker = MOSSETracker()
        cap     = cv2.VideoCapture("video.mp4")
        ret, frame = cap.read()
        tracker.initialize(frame, (100, 80, 60, 60))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            bbox = tracker.update(frame)
            print(bbox)
    """

    def __init__(
        self,
        learning_rate: float = 0.125,
        sigma: float = 2.0,
        psr_threshold: Optional[float] = None,
    ) -> None:
        super().__init__(name="MOSSE")
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.psr_threshold = psr_threshold

        # Internal state — set in initialize()
        self._H_conj: Optional[np.ndarray] = None  # conjugate of learned filter
        self._window: Optional[np.ndarray] = None   # cosine (Hann) window
        self._bbox: Optional[list] = None           # current [x, y, w, h]
        self._last_psr: float = 0.0                 # PSR from the most recent update()

    # ------------------------------------------------------------------ #
    # BaseTracker interface                                                #
    # ------------------------------------------------------------------ #

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the filter on the first frame.

        Args:
            frame: BGR or grayscale image.
            bbox:  Ground-truth box ``(x, y, w, h)``.

        Raises:
            ValueError: If the bounding box yields an empty patch.
        """
        x, y, w, h = [int(round(v)) for v in bbox]
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid bounding box {bbox}: width and height must be positive.")

        gray = self._to_gray(frame)
        self._bbox = [x, y, w, h]
        self._window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)

        # Desired response: Gaussian centred at patch centre
        G = np.fft.fft2(self._gaussian_response(h, w))

        # Extract and preprocess the initial patch
        patch = self._extract_patch(gray, x, y, w, h)
        Fi = np.fft.fft2(self._preprocess(patch))

        # Least-squares MOSSE solution: H* = G / F  (in frequency domain)
        self._H_conj = (G * np.conj(Fi)) / (Fi * np.conj(Fi) + 1e-5)

    def update(self, frame: np.ndarray) -> BBox:
        """Localise the target in the next frame and update the filter.

        Args:
            frame: BGR or grayscale image.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.
        """
        if self._H_conj is None or self._bbox is None:
            raise RuntimeError("Tracker not initialised. Call initialize() first.")

        x, y, w, h = self._bbox
        gray = self._to_gray(frame)

        patch = self._extract_patch(gray, x, y, w, h)
        Fi = np.fft.fft2(self._preprocess(patch))

        # Correlation response
        response = np.real(np.fft.ifft2(self._H_conj * Fi))

        # Compute PSR before shifting — used by update_with_confidence().
        self._last_psr = compute_psr(response)

        # Locate the peak
        ky, kx = np.unravel_index(response.argmax(), response.shape)
        # Shift: peak at (0,0) means no displacement
        dy = ky if ky < h // 2 else ky - h
        dx = kx if kx < w // 2 else kx - w

        x_new = x + dx
        y_new = y + dy
        self._bbox = [x_new, y_new, w, h]

        # Online filter update with EMA
        patch_new = self._extract_patch(gray, x_new, y_new, w, h)
        if patch_new.shape == (h, w):
            G = np.fft.fft2(self._gaussian_response(h, w))
            Fi_new = np.fft.fft2(self._preprocess(patch_new))
            H_conj_new = (G * np.conj(Fi_new)) / (Fi_new * np.conj(Fi_new) + 1e-5)
            self._H_conj = (
                (1.0 - self.learning_rate) * self._H_conj
                + self.learning_rate * H_conj_new
            )

        return (float(x_new), float(y_new), float(w), float(h))

    def update_with_confidence(self, frame: np.ndarray) -> Tuple[BBox, float]:
        """Track and return PSR-derived confidence.

        Calls :meth:`update` (which updates ``_last_psr``) and converts the
        stored PSR to a ``[0, 1]`` confidence score via
        :func:`~eovot.trackers.confidence.psr_to_confidence`.

        Returns:
            ``(bbox, confidence)`` where confidence is in ``[0, 1]``.
        """
        bbox = self.update(frame)
        confidence = psr_to_confidence(self._last_psr)
        return bbox, confidence

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        """Convert BGR or BGRA image to single-channel grayscale."""
        if frame.ndim == 2:
            return frame
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _preprocess(self, patch: np.ndarray) -> np.ndarray:
        """Log-normalise the patch and apply a cosine window."""
        f = np.log1p(patch.astype(np.float32))
        f = (f - f.mean()) / (f.std() + 1e-5)
        return f * self._window

    def _gaussian_response(self, h: int, w: int) -> np.ndarray:
        """2-D Gaussian centred at (h//2, w//2) with std=sigma."""
        cy, cx = h // 2, w // 2
        ys, xs = np.ogrid[:h, :w]
        g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * self.sigma ** 2))
        return (g / (g.sum() + 1e-6)).astype(np.float32)

    def _extract_patch(
        self, gray: np.ndarray, x: int, y: int, w: int, h: int
    ) -> np.ndarray:
        """Extract a ``(h, w)`` patch from *gray*, resizing if needed.

        Clips coordinates to image boundaries so the patch is always
        well-defined, even when the target is partially out of frame.
        """
        ih, iw = gray.shape
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(iw, x + w)
        y2 = min(ih, y + h)
        patch = gray[y1:y2, x1:x2]
        if patch.shape != (h, w):
            patch = cv2.resize(patch, (w, h), interpolation=cv2.INTER_LINEAR)
        return patch

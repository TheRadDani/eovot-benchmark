"""PSR-MOSSE: MOSSE tracker extended with Peak-to-Sidelobe Ratio confidence.

PSR is the standard confidence metric for correlation filter trackers.  It
measures how sharply peaked the response map is relative to its surrounding
sidelobe — a high PSR indicates the filter has confidently localised the
target; a low PSR signals drift or occlusion.

Formula::

    PSR = (peak_value - sidelobe_mean) / sidelobe_std

where the sidelobe is the full response map with an ``(2r+1) × (2r+1)``
window around the peak set to zero.  Default exclusion half-width r = 5.

Typical PSR ranges (empirical, from Bolme et al. 2010):
  > 20  — confident tracking
  7–20  — uncertain, possible drift
  < 7   — likely failure

This tracker inherits the full MOSSE filter implementation and exposes PSR
as a normalised ``[0, 1]`` confidence via :meth:`update_with_score`.

Reference
---------
Bolme, D. S. et al. (2010). Visual object tracking using adaptive
correlation filters. IEEE CVPR, pp. 2544–2550.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .base import BBox
from .confident_base import ConfidentBaseTracker


class PSRMOSSETracker(ConfidentBaseTracker):
    """MOSSE correlation filter tracker with PSR-based confidence output.

    Implements :class:`~eovot.trackers.confident_base.ConfidentBaseTracker`
    so it is usable everywhere a :class:`~eovot.trackers.base.BaseTracker`
    is expected, while also providing per-frame confidence via
    :meth:`update_with_score`.

    Args:
        learning_rate: EMA weight for online filter update.  Default ``0.125``
            matches the original MOSSE paper.
        sigma:         Std of the Gaussian desired response (pixels).
        sidelobe_size: Half-width *r* of the sidelobe exclusion window.
            The masked region is ``(2r+1) × (2r+1)``.  Default ``5``.
        low_psr:       PSR at or below which confidence is clipped to 0.
            Default ``7.0`` (empirical failure threshold).
        high_psr:      PSR at or above which confidence is clipped to 1.
            Default ``20.0`` (empirical confident-tracking threshold).

    Example::

        tracker = PSRMOSSETracker()
        tracker.initialize(first_frame, (100, 80, 60, 60))
        for frame in video:
            bbox, confidence = tracker.update_with_score(frame)
            if confidence < 0.4:
                print("Low confidence — possible occlusion or drift")
    """

    def __init__(
        self,
        learning_rate: float = 0.125,
        sigma: float = 2.0,
        sidelobe_size: int = 5,
        low_psr: float = 7.0,
        high_psr: float = 20.0,
    ) -> None:
        super().__init__(name="PSR-MOSSE")
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.sidelobe_size = sidelobe_size
        self.low_psr = low_psr
        self.high_psr = high_psr

        # Internal state — set in initialize()
        self._H_conj: np.ndarray | None = None
        self._window: np.ndarray | None = None
        self._bbox: list | None = None
        self._last_psr: float = 0.0

    # ------------------------------------------------------------------ #
    # ConfidentBaseTracker interface                                       #
    # ------------------------------------------------------------------ #

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the MOSSE filter on the first frame.

        Args:
            frame: BGR or grayscale image.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.

        Raises:
            ValueError: If the bbox yields an empty patch.
        """
        x, y, w, h = [int(round(v)) for v in bbox]
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid bounding box {bbox}: w and h must be positive.")

        gray = self._to_gray(frame)
        self._bbox = [x, y, w, h]
        self._window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)

        G = np.fft.fft2(self._gaussian_response(h, w))
        Fi = np.fft.fft2(self._preprocess(self._extract_patch(gray, x, y, w, h)))
        self._H_conj = (G * np.conj(Fi)) / (Fi * np.conj(Fi) + 1e-5)
        self._last_psr = float(self.high_psr)  # first frame is confident by construction

    def update_with_score(self, frame: np.ndarray) -> Tuple[BBox, float]:
        """Localise the target and return ``(bbox, confidence)``.

        Confidence is derived from the PSR of the correlation response map,
        linearly normalised to ``[0, 1]`` using ``low_psr`` and ``high_psr``.

        Args:
            frame: BGR or grayscale image.

        Returns:
            ``(bbox, confidence)`` — bbox is ``(x, y, w, h)``, confidence is
            in ``[0, 1]``.
        """
        if self._H_conj is None or self._bbox is None:
            raise RuntimeError("Tracker not initialised. Call initialize() first.")

        x, y, w, h = self._bbox
        gray = self._to_gray(frame)
        patch = self._extract_patch(gray, x, y, w, h)
        Fi = np.fft.fft2(self._preprocess(patch))
        response = np.real(np.fft.ifft2(self._H_conj * Fi))

        ky, kx = np.unravel_index(response.argmax(), response.shape)
        psr = self._compute_psr(response, ky, kx)
        self._last_psr = psr

        dy = ky if ky < h // 2 else ky - h
        dx = kx if kx < w // 2 else kx - w
        x_new, y_new = x + dx, y + dy
        self._bbox = [x_new, y_new, w, h]

        # Online filter update
        patch_new = self._extract_patch(gray, x_new, y_new, w, h)
        if patch_new.shape == (h, w):
            G = np.fft.fft2(self._gaussian_response(h, w))
            Fi_new = np.fft.fft2(self._preprocess(patch_new))
            H_new = (G * np.conj(Fi_new)) / (Fi_new * np.conj(Fi_new) + 1e-5)
            self._H_conj = (
                (1.0 - self.learning_rate) * self._H_conj
                + self.learning_rate * H_new
            )

        confidence = self._psr_to_confidence(psr)
        return (float(x_new), float(y_new), float(w), float(h)), confidence

    @property
    def last_psr(self) -> float:
        """PSR from the most recent :meth:`update_with_score` call."""
        return self._last_psr

    # ------------------------------------------------------------------ #
    # PSR computation                                                      #
    # ------------------------------------------------------------------ #

    def _compute_psr(self, response: np.ndarray, peak_y: int, peak_x: int) -> float:
        """Compute the Peak-to-Sidelobe Ratio of a 2-D response map.

        The sidelobe is the response outside a ``(2r+1) × (2r+1)`` exclusion
        window centred on the peak.  PSR = (peak − sidelobe_mean) / sidelobe_std.

        Args:
            response: 2-D float array of the correlation response.
            peak_y:   Row index of the response peak.
            peak_x:   Column index of the response peak.

        Returns:
            PSR as a non-negative float.  Returns ``0.0`` for degenerate
            (zero-variance) response maps.
        """
        h, w = response.shape
        mask = np.ones((h, w), dtype=bool)
        r = self.sidelobe_size
        y1 = max(0, peak_y - r)
        y2 = min(h, peak_y + r + 1)
        x1 = max(0, peak_x - r)
        x2 = min(w, peak_x + r + 1)
        mask[y1:y2, x1:x2] = False

        sidelobe = response[mask]
        if len(sidelobe) == 0:
            return 0.0
        mean_sl = float(sidelobe.mean())
        std_sl = float(sidelobe.std())
        if std_sl < 1e-10:
            return 0.0
        return max(0.0, (float(response[peak_y, peak_x]) - mean_sl) / std_sl)

    def _psr_to_confidence(self, psr: float) -> float:
        """Map PSR linearly to a ``[0, 1]`` confidence score."""
        if psr <= self.low_psr:
            return 0.0
        if psr >= self.high_psr:
            return 1.0
        return (psr - self.low_psr) / (self.high_psr - self.low_psr)

    # ------------------------------------------------------------------ #
    # Internal helpers (identical logic to MOSSETracker)                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _preprocess(self, patch: np.ndarray) -> np.ndarray:
        f = np.log1p(patch.astype(np.float32))
        f = (f - f.mean()) / (f.std() + 1e-5)
        return f * self._window

    def _gaussian_response(self, h: int, w: int) -> np.ndarray:
        cy, cx = h // 2, w // 2
        ys, xs = np.ogrid[:h, :w]
        g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * self.sigma ** 2))
        return (g / (g.sum() + 1e-6)).astype(np.float32)

    def _extract_patch(
        self, gray: np.ndarray, x: int, y: int, w: int, h: int
    ) -> np.ndarray:
        ih, iw = gray.shape
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(iw, x + w), min(ih, y + h)
        patch = gray[y1:y2, x1:x2]
        if patch.shape != (h, w):
            patch = cv2.resize(patch, (w, h), interpolation=cv2.INTER_LINEAR)
        return patch

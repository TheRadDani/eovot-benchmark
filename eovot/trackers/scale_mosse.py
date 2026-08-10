"""Scale-Adaptive MOSSE Tracker.

Extends the baseline MOSSE correlation filter tracker with a multi-scale
response search: at each frame the position filter is applied across a
scale pyramid, and the bounding box size is updated according to the
scale level that produces the highest Peak-to-Sidelobe Ratio (PSR).

This enables the tracker to handle objects that grow or shrink across
frames (zoom, perspective change, approaching/receding motion) — a
critical failure mode for fixed-size correlation filter trackers on
real-world edge deployments.

Algorithm
---------
1. **Position update** — same as MOSSE: maximise the correlation response
   over a fixed-size search window centred at the previous estimate.
2. **Scale search** — at the estimated position, extract patches at
   ``num_scales`` different sizes spanning
   ``scale_step^(-(num_scales//2))`` to ``scale_step^(num_scales//2)``
   of the current canonical size; each patch is resized to the canonical
   dimensions and scored with the existing correlation filter via PSR.
3. **Scale selection** — the scale level with the highest PSR wins.
4. **EMA size update** — the canonical bounding box dimensions are
   updated with coefficient ``scale_lr``, preventing abrupt jumps on
   noisy single-frame responses.

Reference
---------
Martin Danelljan, Gustav Hager, Fahad Shahbaz Khan, Michael Felsberg.
"Accurate Scale Estimation for Robust Visual Tracking."
BMVC 2014.  (This tracker implements the key scale-search insight from
DSST without the full 1-D scale correlation filter formulation.)

Example::

    from eovot.trackers.scale_mosse import ScaleMOSSETracker
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine

    tracker = ScaleMOSSETracker(num_scales=17, scale_step=1.02)
    dataset = SyntheticDataset(num_sequences=5, motion="linear")
    engine  = BenchmarkEngine()
    result  = engine.run(tracker, dataset, dataset_name="Synthetic")
    print(result)
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .base import BaseTracker, BBox


class ScaleMOSSETracker(BaseTracker):
    """MOSSE correlation filter with multi-scale response for adaptive scale estimation.

    Args:
        learning_rate: EMA weight for online filter update in ``(0, 1]``.
            Default: ``0.125``.
        sigma: Standard deviation of the Gaussian regression target used
            to initialise the filter (pixels).  Default: ``2.0``.
        num_scales: Number of scale levels in the pyramid.  Should be odd
            so that scale factor 1 (current size) is always included.
            Default: ``11``.
        scale_step: Multiplicative step between consecutive scale levels.
            The scale pool spans
            ``scale_step^(-(num_scales//2))`` to ``scale_step^(num_scales//2)``.
            Default: ``1.03`` — gives roughly ±16 % size range with 11 levels.
        scale_lr: EMA coefficient for bounding-box size update.  A smaller
            value smooths size changes; a larger value reacts faster.
            Default: ``0.25``.
        psr_min_sidelobe: Minimum sidelobe pixels required for a reliable
            PSR estimate.  Tiny patches with fewer pixels default to scale 1.
            Default: ``10``.

    Example::

        tracker = ScaleMOSSETracker(num_scales=11, scale_step=1.03)
        tracker.initialize(first_frame, (x, y, w, h))
        bbox = tracker.update(next_frame)
    """

    def __init__(
        self,
        learning_rate: float = 0.125,
        sigma: float = 2.0,
        num_scales: int = 11,
        scale_step: float = 1.03,
        scale_lr: float = 0.25,
        psr_min_sidelobe: int = 10,
    ) -> None:
        super().__init__(name="ScaleMOSSE")
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.num_scales = num_scales
        self.scale_step = scale_step
        self.scale_lr = scale_lr
        self.psr_min_sidelobe = psr_min_sidelobe

        self._H_conj: Optional[np.ndarray] = None
        self._window: Optional[np.ndarray] = None
        self._cx: float = 0.0
        self._cy: float = 0.0
        self._w: float = 0.0
        self._h: float = 0.0
        self._scale_factors: list = []

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the scale-adaptive filter on the first frame.

        Args:
            frame: BGR or grayscale image ``(H, W[, C])``.
            bbox:  Initial bounding box ``(x, y, w, h)`` in pixel coords.

        Raises:
            ValueError: If width or height in *bbox* is not positive.
        """
        x, y, w, h = (float(v) for v in bbox)
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid bbox {bbox}: width and height must be positive.")

        self._w = max(1.0, w)
        self._h = max(1.0, h)
        self._cx = x + w / 2.0
        self._cy = y + h / 2.0

        half = self.num_scales // 2
        self._scale_factors = [self.scale_step ** (i - half) for i in range(self.num_scales)]

        gray = self._to_gray(frame)
        iw, ih = int(round(self._w)), int(round(self._h))
        self._window = np.outer(np.hanning(ih), np.hanning(iw)).astype(np.float32)
        G = np.fft.fft2(self._gaussian_response(ih, iw))
        patch = self._extract_patch(gray, self._cx, self._cy, iw, ih)
        Fi = np.fft.fft2(self._preprocess(patch, ih, iw))
        self._H_conj = (G * np.conj(Fi)) / (Fi * np.conj(Fi) + 1e-5)

    def update(self, frame: np.ndarray) -> BBox:
        """Track the target in a new frame with scale adaptation.

        Args:
            frame: BGR or grayscale image ``(H, W[, C])``.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._H_conj is None:
            raise RuntimeError("ScaleMOSSETracker not initialised. Call initialize() first.")

        gray = self._to_gray(frame)
        iw, ih = int(round(self._w)), int(round(self._h))

        # Step 1: Position estimation (standard MOSSE)
        patch = self._extract_patch(gray, self._cx, self._cy, iw, ih)
        Fi = np.fft.fft2(self._preprocess(patch, ih, iw))
        response = np.real(np.fft.ifft2(self._H_conj * Fi))
        ky, kx = np.unravel_index(response.argmax(), response.shape)
        dy = ky if ky < ih // 2 else ky - ih
        dx = kx if kx < iw // 2 else kx - iw
        self._cx += float(dx)
        self._cy += float(dy)

        # Step 2: Scale estimation via PSR across scale pyramid
        best_scale = self._estimate_best_scale(gray, iw, ih)

        # Step 3: EMA size update
        self._w = (1.0 - self.scale_lr) * self._w + self.scale_lr * (self._w * best_scale)
        self._h = (1.0 - self.scale_lr) * self._h + self.scale_lr * (self._h * best_scale)
        self._w = max(1.0, self._w)
        self._h = max(1.0, self._h)
        new_iw, new_ih = int(round(self._w)), int(round(self._h))

        # Step 4: Filter update (rebuild window if size changed)
        if new_iw != iw or new_ih != ih:
            self._window = np.outer(np.hanning(new_ih), np.hanning(new_iw)).astype(np.float32)

        new_patch = self._extract_patch(gray, self._cx, self._cy, new_iw, new_ih)
        G = np.fft.fft2(self._gaussian_response(new_ih, new_iw))
        Fi_new = np.fft.fft2(self._preprocess(new_patch, new_ih, new_iw))
        H_conj_new = (G * np.conj(Fi_new)) / (Fi_new * np.conj(Fi_new) + 1e-5)
        if H_conj_new.shape == self._H_conj.shape:
            self._H_conj = (
                (1.0 - self.learning_rate) * self._H_conj
                + self.learning_rate * H_conj_new
            )
        else:
            self._H_conj = H_conj_new

        x0 = self._cx - self._w / 2.0
        y0 = self._cy - self._h / 2.0
        return (float(x0), float(y0), float(self._w), float(self._h))

    # ------------------------------------------------------------------
    # Scale estimation
    # ------------------------------------------------------------------

    def _estimate_best_scale(self, gray: np.ndarray, iw: int, ih: int) -> float:
        """Return the scale factor from the pyramid with the highest PSR.

        For each candidate scale *s* the patch ``(iw*s, ih*s)`` is extracted
        at the current estimated centre and resized back to ``(iw, ih)``
        before being scored by the existing position filter.  The scale
        giving the best PSR is returned.

        Args:
            gray: Single-channel grayscale image.
            iw:   Current canonical patch width (pixels).
            ih:   Current canonical patch height (pixels).

        Returns:
            Winning scale factor from ``self._scale_factors``.
        """
        best_psr = -float("inf")
        best_scale = 1.0

        for s in self._scale_factors:
            sw = max(1, int(round(iw * s)))
            sh = max(1, int(round(ih * s)))

            patch_s = self._extract_patch(gray, self._cx, self._cy, sw, sh)
            if sw != iw or sh != ih:
                patch_s = cv2.resize(patch_s, (iw, ih), interpolation=cv2.INTER_LINEAR)

            proc = self._preprocess(patch_s, ih, iw)
            Fi = np.fft.fft2(proc)
            response = np.real(np.fft.ifft2(self._H_conj * Fi))

            psr = self._psr(response, ih, iw)
            if psr > best_psr:
                best_psr = psr
                best_scale = s

        return best_scale

    def _psr(self, response: np.ndarray, ih: int, iw: int) -> float:
        """Peak-to-Sidelobe Ratio of a 2-D correlation response map.

        The sidelobe is all pixels outside an 11x11 neighbourhood around
        the peak.  Higher PSR indicates a cleaner, more confident response.

        Args:
            response: 2-D response array of shape ``(ih, iw)``.
            ih: Response height.
            iw: Response width.

        Returns:
            PSR scalar (higher is better).
        """
        peak_val = float(response.max())
        py, px = np.unravel_index(response.argmax(), response.shape)
        ys, xs = np.ogrid[:ih, :iw]
        sidelobe_mask = (np.abs(ys - py) > 5) | (np.abs(xs - px) > 5)
        sidelobe = response[sidelobe_mask]
        if sidelobe.size < self.psr_min_sidelobe:
            return 0.0
        return float((peak_val - sidelobe.mean()) / (sidelobe.std() + 1e-6))

    # ------------------------------------------------------------------
    # Shared preprocessing (mirrors MOSSETracker internals)
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _preprocess(self, patch: np.ndarray, ih: int, iw: int) -> np.ndarray:
        """Log-normalise the patch and apply the Hann window."""
        f = np.log1p(patch.astype(np.float32))
        f = (f - f.mean()) / (f.std() + 1e-5)
        if self._window is not None and f.shape == self._window.shape:
            win = self._window
        else:
            win = np.outer(np.hanning(ih), np.hanning(iw)).astype(np.float32)
        return f * win

    def _gaussian_response(self, h: int, w: int) -> np.ndarray:
        cy, cx = h // 2, w // 2
        ys, xs = np.ogrid[:h, :w]
        g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * self.sigma ** 2))
        return (g / (g.sum() + 1e-6)).astype(np.float32)

    @staticmethod
    def _extract_patch(
        gray: np.ndarray, cx: float, cy: float, w: int, h: int
    ) -> np.ndarray:
        """Extract a ``(h, w)`` patch centred at ``(cx, cy)``, with edge clipping."""
        fh, fw = gray.shape[:2]
        x1 = max(0, int(cx - w / 2))
        y1 = max(0, int(cy - h / 2))
        x2 = min(fw, x1 + w)
        y2 = min(fh, y1 + h)
        patch = gray[y1:y2, x1:x2]
        if patch.shape != (h, w):
            patch = cv2.resize(patch, (w, h), interpolation=cv2.INTER_LINEAR)
        return patch

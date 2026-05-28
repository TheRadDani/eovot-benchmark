"""Scale-Adaptive MOSSE (SA-MOSSE) correlation filter tracker.

Extends the standard MOSSE tracker with a discrete scale pool that searches
for the best target scale at every frame.  The correlation filter is kept at
a fixed template resolution; candidate patches are extracted at different
physical sizes and resized to that resolution before correlation, so the
filter itself never changes shape.

Algorithm overview
------------------
1. **Translation** — same as MOSSE: compute the correlation response on a patch
   extracted at the current (x, y, w, h) and locate the displacement peak.
2. **Scale search** — after updating the translation, extract ``n_scales``
   candidate patches whose physical size ranges from
   ``scale_step ** (-(n_scales//2))`` to ``scale_step ** (n_scales//2)``
   times the current dimensions.  Each candidate is resized to the fixed
   template shape and correlated with the learned filter; the candidate with
   the highest response peak is chosen.
3. **Scale smoothing** — the raw scale candidate is blended with the previous
   scale using an exponential moving average (``scale_lr``) to suppress
   frame-to-frame jitter.
4. **Filter update** — the filter is updated using the new scale patch (EMA
   with ``learning_rate``), exactly as in standard MOSSE.

References
----------
* Bolme et al., "Visual object tracking using adaptive correlation filters,"
  CVPR 2010. (original MOSSE)
* Li et al., "A Scale Adaptive Kernel Correlation Filter Tracker with Feature
  Integration," ECCV Workshops 2014. (scale search motivation)
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox


class ScaleAdaptiveMOSSETracker(BaseTracker):
    """MOSSE tracker extended with discrete-scale-pool search.

    The tracker maintains a fixed-resolution correlation filter (initialised
    from the first-frame bounding box) and performs a multi-scale patch search
    at every subsequent frame.  Aspect ratio is always preserved.

    Args:
        learning_rate: EMA weight for online filter update (0 = frozen,
            1 = full replacement).  Default ``0.125`` matches the original
            MOSSE paper.
        sigma: Standard deviation of the Gaussian target response used during
            initialisation (pixels).  Default ``2.0``.
        n_scales: Number of candidate scales to evaluate each frame.  Must be
            a positive odd integer so the pool is symmetric around scale 1.
            Default ``7`` (i.e. scale_step ** {-3, -2, -1, 0, +1, +2, +3}).
        scale_step: Multiplicative step between consecutive scale candidates.
            Default ``1.03`` (3 % increment per step).
        scale_lr: EMA weight for the scale update.  Lower values produce
            smoother but slower scale adaptation.  Default ``0.35``.

    Example::

        import cv2
        from eovot.trackers.scale_adaptive_mosse import ScaleAdaptiveMOSSETracker

        tracker = ScaleAdaptiveMOSSETracker()
        cap = cv2.VideoCapture("video.mp4")
        ret, frame = cap.read()
        tracker.initialize(frame, (100, 80, 60, 60))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            bbox = tracker.update(frame)
            print(bbox)
    """

    _MIN_DIM: int = 4  # minimum width / height in pixels

    def __init__(
        self,
        learning_rate: float = 0.125,
        sigma: float = 2.0,
        n_scales: int = 7,
        scale_step: float = 1.03,
        scale_lr: float = 0.35,
    ) -> None:
        super().__init__(name="ScaleAdaptiveMOSSE")
        if n_scales < 1:
            raise ValueError("n_scales must be >= 1.")
        if scale_step <= 1.0:
            raise ValueError("scale_step must be > 1.0.")
        if not (0.0 < learning_rate <= 1.0):
            raise ValueError("learning_rate must be in (0, 1].")
        if not (0.0 < scale_lr <= 1.0):
            raise ValueError("scale_lr must be in (0, 1].")

        self.learning_rate = learning_rate
        self.sigma = sigma
        self.n_scales = n_scales
        self.scale_step = scale_step
        self.scale_lr = scale_lr

        # Build the symmetric scale pool once; reused every frame.
        half = n_scales // 2
        self._scale_pool: np.ndarray = np.array(
            [scale_step ** k for k in range(-half, half + 1)], dtype=np.float64
        )

        # State initialised in initialize()
        self._H_conj: Optional[np.ndarray] = None  # conjugate of learned filter
        self._window: Optional[np.ndarray] = None   # Hann window (th, tw)
        self._bbox: Optional[list] = None           # current [x, y, w, h]
        self._template_size: Optional[Tuple[int, int]] = None  # (th, tw) fixed
        self._init_w: int = 0                        # initial width (for scale calc)
        self._init_h: int = 0                        # initial height (for scale calc)
        self._current_scale: float = 1.0             # accumulated scale factor

    # ------------------------------------------------------------------ #
    # BaseTracker interface                                                #
    # ------------------------------------------------------------------ #

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the scale-adaptive filter on the first frame.

        Args:
            frame: BGR or grayscale image ``(H, W[, C])``.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.

        Raises:
            ValueError: If the bounding box yields an empty patch.
        """
        x, y, w, h = [int(round(v)) for v in bbox]
        if w < self._MIN_DIM or h < self._MIN_DIM:
            raise ValueError(
                f"Bounding box {bbox} too small; minimum dimension is {self._MIN_DIM} px."
            )

        gray = _to_gray(frame)
        self._bbox = [x, y, w, h]
        self._init_w = w
        self._init_h = h
        self._current_scale = 1.0
        self._template_size = (h, w)  # (th, tw) — fixed for the life of the track

        self._window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)

        G = np.fft.fft2(_gaussian_response(h, w, self.sigma))
        patch = _extract_and_resize(gray, x, y, w, h, w, h)
        Fi = np.fft.fft2(self._preprocess(patch))
        self._H_conj = (G * np.conj(Fi)) / (Fi * np.conj(Fi) + 1e-5)

    def update(self, frame: np.ndarray) -> BBox:
        """Localise the target in the next frame with scale adaptation.

        Args:
            frame: BGR or grayscale image ``(H, W[, C])``.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If :meth:`initialize` has not been called.
        """
        if self._H_conj is None or self._bbox is None:
            raise RuntimeError(
                "Tracker not initialised — call initialize() before update()."
            )

        x, y, w, h = self._bbox
        th, tw = self._template_size  # type: ignore[misc]
        gray = _to_gray(frame)

        # ---- Step 1: Translation estimation --------------------------------
        patch = _extract_and_resize(gray, x, y, w, h, tw, th)
        Fi = np.fft.fft2(self._preprocess(patch))
        response = np.real(np.fft.ifft2(self._H_conj * Fi))

        ky, kx = np.unravel_index(response.argmax(), response.shape)
        # Wrap-around: peaks beyond the half-size encode negative displacements.
        dy_tmpl = ky if ky < th // 2 else ky - th
        dx_tmpl = kx if kx < tw // 2 else kx - tw

        # Scale template-space displacement back to image space.
        scale_y = h / th
        scale_x = w / tw
        x_new = x + int(round(dx_tmpl * scale_x))
        y_new = y + int(round(dy_tmpl * scale_y))

        # ---- Step 2: Scale search ------------------------------------------
        best_scale_factor = 1.0
        best_peak = -np.inf

        for sf in self._scale_pool:
            cand_w = max(self._MIN_DIM, int(round(w * sf)))
            cand_h = max(self._MIN_DIM, int(round(h * sf)))
            cand = _extract_and_resize(gray, x_new, y_new, cand_w, cand_h, tw, th)
            resp = np.real(np.fft.ifft2(self._H_conj * np.fft.fft2(self._preprocess(cand))))
            peak = float(resp.max())
            if peak > best_peak:
                best_peak = peak
                best_scale_factor = sf

        # ---- Step 3: Scale EMA smoothing -----------------------------------
        # Blend the raw winner scale toward 1.0 (i.e. no change) at rate scale_lr
        # to suppress per-frame noise while still tracking genuine size changes.
        self._current_scale *= (1.0 - self.scale_lr) + self.scale_lr * best_scale_factor

        w_new = max(self._MIN_DIM, int(round(self._init_w * self._current_scale)))
        h_new = max(self._MIN_DIM, int(round(self._init_h * self._current_scale)))
        self._bbox = [x_new, y_new, w_new, h_new]

        # ---- Step 4: Online filter update ----------------------------------
        new_patch = _extract_and_resize(gray, x_new, y_new, w_new, h_new, tw, th)
        G = np.fft.fft2(_gaussian_response(th, tw, self.sigma))
        Fi_new = np.fft.fft2(self._preprocess(new_patch))
        H_conj_new = (G * np.conj(Fi_new)) / (Fi_new * np.conj(Fi_new) + 1e-5)
        self._H_conj = (
            (1.0 - self.learning_rate) * self._H_conj
            + self.learning_rate * H_conj_new
        )

        return (float(x_new), float(y_new), float(w_new), float(h_new))

    # ------------------------------------------------------------------ #
    # Preprocessing                                                        #
    # ------------------------------------------------------------------ #

    def _preprocess(self, patch: np.ndarray) -> np.ndarray:
        """Log-normalise the patch and apply the Hann window."""
        f = np.log1p(patch.astype(np.float32))
        f = (f - f.mean()) / (f.std() + 1e-5)
        return f * self._window  # type: ignore[operator]


# ------------------------------------------------------------------ #
# Module-level helpers (shared with MOSSETracker by convention)       #
# ------------------------------------------------------------------ #

def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert BGR / BGRA / grayscale image to single-channel uint8."""
    if frame.ndim == 2:
        return frame
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _gaussian_response(h: int, w: int, sigma: float) -> np.ndarray:
    """2-D Gaussian centred at ``(h//2, w//2)`` with standard deviation *sigma*."""
    cy, cx = h // 2, w // 2
    ys, xs = np.ogrid[:h, :w]
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma ** 2))
    return (g / (g.sum() + 1e-6)).astype(np.float32)


def _extract_and_resize(
    gray: np.ndarray,
    x: int,
    y: int,
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> np.ndarray:
    """Extract a ``(src_h, src_w)`` region and resize to ``(dst_h, dst_w)``.

    Coordinates are clipped to the image boundary so the result is always
    well-defined, even when the target is partially out of frame.
    """
    ih, iw = gray.shape
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(iw, x + src_w)
    y2 = min(ih, y + src_h)
    patch = gray[y1:y2, x1:x2]
    if patch.size == 0:
        return np.zeros((dst_h, dst_w), dtype=np.float32)
    if patch.shape != (dst_h, dst_w):
        patch = cv2.resize(patch, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
    return patch

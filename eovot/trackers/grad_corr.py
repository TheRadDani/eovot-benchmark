"""GradCorr — Gradient-Channel Correlation Filter tracker.

A lightweight multi-channel correlation filter that uses dense gradient
features instead of raw grayscale pixels.  Unlike HOG-based DCF variants
that build a sparse cell grid, GradCorr operates at full pixel resolution,
preserving fine spatial detail while remaining entirely pure NumPy + OpenCV.

Motivation
----------
MOSSE and linear KCF use raw grayscale intensity as the sole feature channel.
This representation is sensitive to illumination changes and low-contrast
targets where intensity differences are small.  Adding gradient magnitude and
orientation channels provides:

* **Illumination invariance** — gradients respond to edges, not absolute
  brightness, so the filter remains effective under lighting change.
* **Shape information** — the orientation channels capture the angular
  structure of the target boundary, complementing pure magnitude.
* **No cell quantisation** — unlike HOG, orientation bins are not quantised
  to discrete histogram cells; the continuous doubled-angle encoding retains
  sub-pixel orientation precision in the frequency domain.

Feature representation
----------------------
Four channels per patch, extracted at full pixel resolution:

1. **Gray** — contrast-normalised grayscale (zero mean, unit std).
2. **Magnitude** — L2 gradient magnitude, contrast-normalised.
3. **CosOrient** — ``cos(2θ) × |∇|``: captures magnitude-weighted horizontal
   edge energy.  The factor of 2 makes the encoding unsigned (180°-periodic)
   matching the direction-insensitive nature of image edges.
4. **SinOrient** — ``sin(2θ) × |∇|``: orthogonal orientation complement.

Channels 3 and 4 together form a 2-D complex gradient representation first
described by Felzenszwalb et al. (2010).  Doubling the angle before taking
cos/sin achieves orientation invariance to gradient sign flips: a dark-on-light
and light-on-dark edge at the same location produce identical (cos, sin) values.

Filter training and detection
------------------------------
One independent optimal filter per channel, trained in closed form:

    Hk = Ak / (Bk + λ),    Ak = Y* ⊙ Xk̂,    Bk = Xk̂* ⊙ Xk̂

Response: ``r = Σk ℜ{IFFT2(Hk ⊙ Ẑk)}``

Per-frame complexity: O(4 × N log N), N = patch area — identical to a single
kernel-correlation pass in KCF but with a 4-channel feature instead of a
kernel function.

Online update
-------------
EMA on per-channel accumulators:

    Ak ← (1−α) Ak + α (Y* ⊙ Ẑk)
    Bk ← (1−α) Bk + α (Ẑk* ⊙ Ẑk)

Distinguishing properties vs. HOG-DCF
--------------------------------------
* **Full pixel resolution** — HOG-DCF maps a (pw, ph) patch to a
  (pw/cell_size, ph/cell_size) cell grid, coarsening spatial resolution.
  GradCorr keeps the full (pw, ph) response map.
* **No bin quantisation** — HOG bins gradient angles into discrete buckets
  (typically 9 bins); GradCorr uses the exact continuous cos/sin encoding.
* **Simpler implementation** — no cell aggregation, block normalisation, or
  orientation binning; the entire feature extraction is 10 NumPy lines.
* **Lower overhead** — no HOG cell grid allocation or bin accumulation loop.

Edge performance
----------------
* No GPU or external library required.
* Expected throughput: ~200–500 FPS depending on patch size and CPU.
* Memory: 2 × 4 × H × W × 8 bytes for the complex filter state; ≈ 0.5 MB
  for a 128×128 search window.

Example::

    from eovot.trackers.grad_corr import GradCorrTracker
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine

    dataset = SyntheticDataset(num_sequences=10, num_frames=100, motion="random")
    tracker = GradCorrTracker(learning_rate=0.10, padding=1.5)
    engine  = BenchmarkEngine(verbose=True)
    result  = engine.run(tracker, dataset, dataset_name="Synthetic-Random")
    print(result.summary())
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox

_N_CHANNELS = 4  # gray, magnitude, cos-orient, sin-orient


class GradCorrTracker(BaseTracker):
    """Gradient-Channel Correlation Filter tracker.

    Extends MOSSE with four dense gradient feature channels instead of raw
    grayscale, providing illumination invariance and orientation sensitivity
    without requiring HOG cell quantisation or deep features.

    Unlike HOG-based DCF trackers, GradCorr operates at full pixel resolution
    and uses continuous doubled-angle orientation encoding, making it both
    simpler to implement and more spatially precise.

    Args:
        learning_rate: EMA weight for online filter update, in ``(0, 1]``.
            Lower values (0.05–0.10) give stable long-term models; higher
            values (0.15–0.25) adapt faster to appearance change.
            Default: ``0.10``.
        lambda_: Tikhonov regularisation term.  Prevents filter singularities
            when feature energy is near zero.  Default: ``1e-4``.
        padding: Context factor — each side of the search window extends the
            target box by this fraction, so ``padding=1.5`` means the search
            window is 2.5× wider/taller than the target.  Default: ``1.5``.
        sigma_frac: Standard deviation of the Gaussian regression target as
            a fraction of the search window dimension.  Default: ``0.1``.

    Raises:
        RuntimeError: If :meth:`update` is called before :meth:`initialize`.
        ValueError: If ``bbox`` has non-positive width or height.
    """

    def __init__(
        self,
        learning_rate: float = 0.10,
        lambda_: float = 1e-4,
        padding: float = 1.5,
        sigma_frac: float = 0.1,
    ) -> None:
        super().__init__(name="GradCorr")
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.padding = padding
        self.sigma_frac = sigma_frac

        self._pos: Optional[Tuple[float, float]] = None
        self._target_sz: Optional[Tuple[int, int]] = None  # (tw, th)
        self._search_sz: Optional[Tuple[int, int]] = None  # (sw, sh)
        self._window: Optional[np.ndarray] = None           # (sh, sw) Hann window
        self._Yf: Optional[np.ndarray] = None               # (sh, sw) complex
        self._A: Optional[np.ndarray] = None                # (4, sh, sw) complex
        self._B: Optional[np.ndarray] = None                # (4, sh, sw) float64

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the filter on the first frame.

        Args:
            frame: BGR ``(H, W, 3)`` or grayscale ``(H, W)`` uint8 array.
            bbox: Ground-truth box ``(x, y, w, h)`` in pixel coordinates.

        Raises:
            ValueError: If the bbox has non-positive width or height.
        """
        x, y, w, h = (float(v) for v in bbox)
        if w <= 0 or h <= 0:
            raise ValueError(
                f"GradCorrTracker: invalid bbox {bbox} — width and height must be positive."
            )

        cx, cy = x + w / 2.0, y + h / 2.0
        self._pos = (cx, cy)
        self._target_sz = (max(1, int(round(w))), max(1, int(round(h))))

        sw = max(1, int(round(w * (1.0 + self.padding))))
        sh = max(1, int(round(h * (1.0 + self.padding))))
        self._search_sz = (sw, sh)

        self._window = self._hann2d(sh, sw)
        self._Yf = np.fft.fft2(self._gaussian_labels(sh, sw, self.sigma_frac))

        feats = self._extract_features(frame, cx, cy)   # (4, sh, sw)
        C, H, W = feats.shape
        self._A = np.zeros((C, H, W), dtype=np.complex128)
        self._B = np.zeros((C, H, W), dtype=np.float64)
        for c in range(C):
            Xf = np.fft.fft2(feats[c])
            self._A[c] = np.conj(self._Yf) * Xf
            self._B[c] = np.real(np.conj(Xf) * Xf)

    def update(self, frame: np.ndarray) -> BBox:
        """Locate the target in a new frame and update the filter.

        Args:
            frame: BGR or grayscale image.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._pos is None:
            raise RuntimeError(
                "GradCorrTracker not initialised. Call initialize() first."
            )

        cx, cy = self._pos
        feats = self._extract_features(frame, cx, cy)
        C, H, W = feats.shape

        # Sum response maps across channels
        response = np.zeros((H, W), dtype=np.float64)
        for c in range(C):
            Zf = np.fft.fft2(feats[c])
            Hf = self._A[c] / (self._B[c] + self.lambda_)
            response += np.real(np.fft.ifft2(Hf * Zf))

        # Peak in cyclic-shift convention
        dy, dx = np.unravel_index(np.argmax(response), response.shape)
        if dy > H // 2:
            dy -= H
        if dx > W // 2:
            dx -= W

        new_cx = cx + float(dx)
        new_cy = cy + float(dy)
        self._pos = (new_cx, new_cy)

        # EMA online update
        new_feats = self._extract_features(frame, new_cx, new_cy)
        lr = self.learning_rate
        for c in range(C):
            Zf = np.fft.fft2(new_feats[c])
            self._A[c] = (1.0 - lr) * self._A[c] + lr * (np.conj(self._Yf) * Zf)
            self._B[c] = (1.0 - lr) * self._B[c] + lr * np.real(np.conj(Zf) * Zf)

        tw, th = self._target_sz
        return (new_cx - tw / 2.0, new_cy - th / 2.0, float(tw), float(th))

    def reset(self) -> None:
        """Reset all internal state so the tracker can be re-initialised."""
        self._pos = None
        self._target_sz = None
        self._search_sz = None
        self._window = None
        self._Yf = None
        self._A = None
        self._B = None

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(
        self, frame: np.ndarray, cx: float, cy: float
    ) -> np.ndarray:
        """Extract 4-channel gradient features centred at (cx, cy).

        Returns:
            Float64 array of shape ``(4, sh, sw)`` with Hann window applied.
        """
        sw, sh = self._search_sz
        patch = self._crop_patch(frame, cx, cy, sw, sh)   # (sh, sw) float32

        c0 = patch.astype(np.float64)
        c0 = (c0 - c0.mean()) / (c0.std() + 1e-5)

        gx = np.gradient(c0, axis=1)
        gy = np.gradient(c0, axis=0)
        mag = np.hypot(gx, gy)
        angle = np.arctan2(gy, gx)

        c1 = (mag - mag.mean()) / (mag.std() + 1e-5)
        c2 = np.cos(2.0 * angle) * mag   # unsigned orientation (180° periodic)
        c3 = np.sin(2.0 * angle) * mag

        features = np.stack([c0, c1, c2, c3], axis=0)     # (4, sh, sw)
        features *= self._window[np.newaxis]
        return features

    def _crop_patch(
        self, frame: np.ndarray, cx: float, cy: float, w: int, h: int
    ) -> np.ndarray:
        """Extract a (h, w) grayscale patch centred at (cx, cy).

        Out-of-bounds regions are edge-replicated so the patch is always
        well-defined even when the target is partially outside the frame.

        Returns:
            Float32 array of shape ``(h, w)``.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        x1 = int(round(cx - w / 2.0))
        y1 = int(round(cy - h / 2.0))
        x2, y2 = x1 + w, y1 + h

        fh, fw = gray.shape[:2]
        pl = max(0, -x1)
        pt = max(0, -y1)
        pr = max(0, x2 - fw)
        pb = max(0, y2 - fh)

        if pl or pt or pr or pb:
            gray = np.pad(gray, ((pt, pb), (pl, pr)), mode="edge")
            x1 += pl
            y1 += pt
            x2 += pl
            y2 += pt

        patch = gray[y1:y2, x1:x2]
        if patch.shape != (h, w):
            patch = cv2.resize(patch, (w, h))
        return patch.astype(np.float32)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hann2d(h: int, w: int) -> np.ndarray:
        return np.outer(np.hanning(h), np.hanning(w))

    @staticmethod
    def _gaussian_labels(h: int, w: int, sigma_frac: float) -> np.ndarray:
        """Gaussian regression target at index (0,0) — FFT cyclic convention."""
        sy, sx = sigma_frac * h, sigma_frac * w
        ys = np.arange(h) - h // 2
        xs = np.arange(w) - w // 2
        xx, yy = np.meshgrid(xs, ys)
        labels = np.exp(-(xx ** 2 / (2.0 * sx ** 2) + yy ** 2 / (2.0 * sy ** 2)))
        labels = np.roll(np.roll(labels, -h // 2, axis=0), -w // 2, axis=1)
        return labels.astype(np.float64)

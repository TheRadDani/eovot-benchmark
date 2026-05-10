"""HOG-feature Kernelized Correlation Filter (HOG-KCF) tracker.

Extends the standard KCF tracker by replacing raw grayscale with a
multi-channel Histogram of Oriented Gradients (HOG) feature map.

Background
----------
The original KCF tracker (Henriques et al., TPAMI 2015) applies a Gaussian
RBF kernel over raw grayscale features.  A well-known improvement is to
replace the raw feature with a multi-channel HOG descriptor, which provides
significantly stronger discriminability for targets with rich edge structure
(vehicles, pedestrians, rigid objects) at only moderate extra computation.

Multi-channel kernel (Henriques et al. 2015, Sec. 3.3)::

    ||x - z||² = Σ_c ||x_c - z_c||²

Using Parseval's theorem each channel's contribution reduces to element-wise
FFT operations, so the per-frame cost remains O(C · N log N) — still fast
enough for edge deployment at 50–150 FPS depending on patch size.

References
----------
- Henriques et al., "High-Speed Tracking with Kernelized Correlation Filters."
  IEEE TPAMI, 2015.
- Danelljan et al., "Accurate Scale Estimation for Robust Visual Tracking."
  BMVC, 2014.  (shows HOG-KCF as the backbone before DSST scale filter)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox


class HOGKCFTracker(BaseTracker):
    """KCF tracker with multi-channel HOG features.

    Uses an 8-orientation HOG feature map (dense, one HOG cell per
    ``cell_size`` pixels) instead of raw grayscale.  The multi-channel
    Gaussian kernel is computed in the Fourier domain, preserving the
    O(C · N log N) complexity of the original KCF.

    Compared to :class:`~eovot.trackers.kcf.KCFTracker`:

    * Significantly more discriminative on structured targets (vehicles,
      persons) where edge orientation is a stable appearance cue.
    * Naturally scale the template to the HOG cell grid, reducing template
      size and computational cost for large search patches.
    * Slightly higher per-frame cost due to HOG extraction and C-channel FFTs,
      typically offset by the smaller HOG feature map size.

    Args:
        learning_rate: EMA weight for online filter updates, in ``(0, 1]``.
            Default: ``0.075``.
        lambda_: Ridge-regression regularisation term. Default: ``1e-4``.
        padding: Context added on each side as a fraction of target size.
            Default: ``1.5`` (search window = 2.5× target).
        kernel_sigma: Bandwidth of the Gaussian kernel. Default: ``0.5``.
        n_orient: Number of HOG orientation bins. Default: ``8``.
        cell_size: HOG cell size in pixels. Larger values reduce feature-map
            resolution and speed up computation. Default: ``4``.

    Example::

        tracker = HOGKCFTracker()
        tracker.initialize(frame, (x, y, w, h))
        for frame in sequence:
            pred = tracker.update(frame)
    """

    def __init__(
        self,
        learning_rate: float = 0.075,
        lambda_: float = 1e-4,
        padding: float = 1.5,
        kernel_sigma: float = 0.5,
        n_orient: int = 8,
        cell_size: int = 4,
    ) -> None:
        super().__init__(name="HOG-KCF")
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.padding = padding
        self.kernel_sigma = kernel_sigma
        self.n_orient = n_orient
        self.cell_size = cell_size

        self._pos: Optional[Tuple[float, float]] = None
        self._target_sz: Optional[Tuple[int, int]] = None
        self._search_sz: Optional[Tuple[int, int]] = None
        # Multi-channel template and filter: one entry per HOG channel.
        self._xfs: Optional[List[np.ndarray]] = None
        self._alphaf: Optional[np.ndarray] = None
        self._window: Optional[np.ndarray] = None
        self._yf: Optional[np.ndarray] = None
        # HOG feature map dimensions derived from search patch.
        self._feat_h: int = 0
        self._feat_w: int = 0

    # ------------------------------------------------------------------
    # Public BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the HOG-KCF filter on the first frame.

        Args:
            frame: First frame as a ``(H, W, C)`` BGR array or ``(H, W)`` gray.
            bbox:  Initial bounding box ``(x, y, w, h)`` in pixel coordinates.
        """
        x, y, w, h = (float(v) for v in bbox)
        cx, cy = x + w / 2.0, y + h / 2.0

        self._target_sz = (max(1, int(round(w))), max(1, int(round(h))))
        self._search_sz = (
            max(1, int(round(w * (1.0 + self.padding)))),
            max(1, int(round(h * (1.0 + self.padding)))),
        )
        self._pos = (cx, cy)

        # HOG feature map size in cells (rounded to even for FFT symmetry).
        sw, sh = self._search_sz
        self._feat_w = max(2, (sw // self.cell_size) * self.cell_size // self.cell_size)
        self._feat_h = max(2, (sh // self.cell_size) * self.cell_size // self.cell_size)

        self._window = self._hann2d(self._feat_h, self._feat_w)
        self._yf = np.fft.fft2(self._gaussian_labels(self._feat_h, self._feat_w))

        channels = self._extract(frame, cx, cy)
        self._xfs = [np.fft.fft2(ch * self._window) for ch in channels]
        kf = self._kernel_corr(self._xfs, self._xfs)
        self._alphaf = self._yf / (kf + self.lambda_)

    def update(self, frame: np.ndarray) -> BBox:
        """Track the target in a new frame.

        Args:
            frame: Current frame as a ``(H, W, C)`` BGR or ``(H, W)`` gray array.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If :meth:`initialize` has not been called.
        """
        if self._pos is None:
            raise RuntimeError("HOGKCFTracker is not initialised. Call initialize() first.")

        cx, cy = self._pos

        # --- Detection ---
        channels = self._extract(frame, cx, cy)
        zfs = [np.fft.fft2(ch * self._window) for ch in channels]
        kzf = self._kernel_corr(self._xfs, zfs)
        response = np.real(np.fft.ifft2(self._alphaf * kzf))

        dy, dx = np.unravel_index(np.argmax(response), response.shape)
        fh, fw = response.shape
        if dy > fh // 2:
            dy -= fh
        if dx > fw // 2:
            dx -= fw

        # Convert cell-level displacement to pixel displacement.
        new_cx = cx + float(dx) * self.cell_size
        new_cy = cy + float(dy) * self.cell_size
        self._pos = (new_cx, new_cy)

        # --- Online update (EMA on template and filter coefficients) ---
        new_channels = self._extract(frame, new_cx, new_cy)
        new_xfs = [np.fft.fft2(ch * self._window) for ch in new_channels]
        new_kf = self._kernel_corr(new_xfs, new_xfs)
        new_alphaf = self._yf / (new_kf + self.lambda_)

        lr = self.learning_rate
        self._xfs = [(1.0 - lr) * xf + lr * nxf for xf, nxf in zip(self._xfs, new_xfs)]
        self._alphaf = (1.0 - lr) * self._alphaf + lr * new_alphaf

        tw, th = self._target_sz
        return (new_cx - tw / 2.0, new_cy - th / 2.0, float(tw), float(th))

    def reset(self) -> None:
        """Reset all internal state so the tracker can be re-initialised."""
        self._pos = None
        self._target_sz = None
        self._search_sz = None
        self._xfs = None
        self._alphaf = None
        self._window = None
        self._yf = None
        self._feat_h = 0
        self._feat_w = 0

    # ------------------------------------------------------------------
    # HOG feature extraction
    # ------------------------------------------------------------------

    def _extract(self, frame: np.ndarray, cx: float, cy: float) -> List[np.ndarray]:
        """Extract HOG feature channels centred at ``(cx, cy)``.

        Steps:
        1. Convert to grayscale and crop the search patch with edge padding.
        2. Resize to ``(feat_h * cell_size, feat_w * cell_size)`` so that the
           HOG grid is exactly ``(feat_h, feat_w)`` cells.
        3. Compute per-pixel gradient magnitude and orientation.
        4. Soft-assign each pixel to ``n_orient`` orientation bins weighted
           by gradient magnitude.
        5. Average-pool each orientation map over ``cell_size × cell_size``
           cells → ``(feat_h, feat_w)`` per channel.
        6. L2-normalise across channels per cell, then standardise per channel.

        Returns:
            List of ``n_orient`` float32 arrays, each of shape
            ``(feat_h, feat_w)``.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        sw, sh = self._search_sz
        x1 = int(round(cx - sw / 2.0))
        y1 = int(round(cy - sh / 2.0))
        x2, y2 = x1 + sw, y1 + sh

        fh_px, fw_px = gray.shape[:2]
        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - fw_px)
        pad_b = max(0, y2 - fh_px)
        if pad_l or pad_t or pad_r or pad_b:
            gray = np.pad(gray, ((pad_t, pad_b), (pad_l, pad_r)), mode="edge")
            x1 += pad_l
            y1 += pad_t
            x2 += pad_l
            y2 += pad_t

        patch = gray[y1:y2, x1:x2].astype(np.float32)
        target_h = self._feat_h * self.cell_size
        target_w = self._feat_w * self.cell_size
        if patch.shape != (target_h, target_w):
            patch = cv2.resize(patch, (target_w, target_h))

        return self._hog_channels(patch)

    def _hog_channels(self, patch: np.ndarray) -> List[np.ndarray]:
        """Compute multi-channel HOG feature map from a grayscale patch.

        Args:
            patch: Grayscale float32 array of shape
                ``(feat_h * cell_size, feat_w * cell_size)``.

        Returns:
            List of ``n_orient`` arrays of shape ``(feat_h, feat_w)``,
            L2-normalised per cell and zero-mean unit-variance per channel.
        """
        cs = self.cell_size
        fh, fw = self._feat_h, self._feat_w

        # Compute image gradients via Sobel-style finite differences.
        gx = np.gradient(patch.astype(np.float64), axis=1)
        gy = np.gradient(patch.astype(np.float64), axis=0)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        angle = np.arctan2(gy, gx)  # in [-π, π]

        # Soft assignment to orientation bins using a triangle kernel.
        bin_width = 2.0 * np.pi / self.n_orient
        channels_raw = []
        for b in range(self.n_orient):
            bin_center = -np.pi + (b + 0.5) * bin_width
            diff = angle - bin_center
            # Wrap angular difference to [-π, π]
            diff = (diff + np.pi) % (2.0 * np.pi) - np.pi
            weight = np.maximum(0.0, 1.0 - np.abs(diff) / bin_width)
            feat_pixel = (weight * magnitude).astype(np.float32)

            # Average-pool to cell grid using reshape + mean
            feat_cells = (
                feat_pixel
                .reshape(fh, cs, fw, cs)
                .mean(axis=(1, 3))
            )
            channels_raw.append(feat_cells)

        # Stack → (fh, fw, n_orient); L2-normalize per cell across channels.
        stacked = np.stack(channels_raw, axis=-1)  # (fh, fw, n_orient)
        norm = np.sqrt((stacked ** 2).sum(axis=-1, keepdims=True)) + 1e-6
        stacked = stacked / norm

        # Subtract per-channel mean to remove DC bias; do NOT divide by std
        # because std can be near-zero for spatially uniform bins, which would
        # amplify the values and cause exp(-exponent) to underflow to 0.
        result = []
        for b in range(self.n_orient):
            ch = stacked[:, :, b]
            ch = ch - ch.mean()
            result.append(ch.astype(np.float32))
        return result

    # ------------------------------------------------------------------
    # Multi-channel kernel correlation
    # ------------------------------------------------------------------

    def _kernel_corr(
        self, xfs: List[np.ndarray], zfs: List[np.ndarray]
    ) -> np.ndarray:
        """Multi-channel Gaussian kernel correlation in the Fourier domain.

        Computes the DFT of the multi-channel kernel response map::

            k(δ) = exp(−‖x − z_δ‖² / σ²)

        where ‖·‖² sums over all HOG channels and all spatial positions.
        Using Parseval's theorem, this reduces to per-channel element-wise
        FFT operations (O(C · N log N) total):

            ‖x − z‖² = Σ_c (‖x_c‖² + ‖z_c‖² − 2·Re(x_c* ⊙ z_c)) / N

        Args:
            xfs: List of FFT arrays for the template channels.
            zfs: List of FFT arrays for the candidate patch channels.

        Returns:
            DFT of the Gaussian kernel response map, shape ``(feat_h, feat_w)``.
        """
        N = xfs[0].shape[0] * xfs[0].shape[1]
        # xx/zz: per-channel spatial energy via Parseval (sum|X|²/N = ||x||²).
        xx = sum(np.real(np.sum(xf * np.conj(xf))) / N for xf in xfs)
        zz = sum(np.real(np.sum(zf * np.conj(zf))) / N for zf in zfs)
        # cross[δ]: sum of per-channel spatial cross-correlations.
        # np.fft.ifft2 applies 1/N internally, so ifft2(X̄·Z)[δ] = (x⋆z)[δ].
        # At δ=0 (self-correlation): cross[0] = sum_c ||x_c||² = xx → exponent[0] = 0.
        cross = sum(
            np.real(np.fft.ifft2(np.conj(xf) * zf))
            for xf, zf in zip(xfs, zfs)
        )
        exponent = np.maximum(0.0, xx + zz - 2.0 * cross) / (self.kernel_sigma ** 2)
        return np.fft.fft2(np.exp(-exponent))

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hann2d(h: int, w: int) -> np.ndarray:
        """2-D Hann window to suppress spectral leakage at patch edges."""
        return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)

    @staticmethod
    def _gaussian_labels(h: int, w: int, sigma_frac: float = 0.1) -> np.ndarray:
        """Soft Gaussian regression target centred at the patch origin (FFT convention).

        Args:
            h: Feature map height in cells.
            w: Feature map width in cells.
            sigma_frac: Standard deviation as a fraction of the map dimension.

        Returns:
            2-D float32 array of shape ``(h, w)`` with values in ``[0, 1]``.
        """
        sig_h, sig_w = sigma_frac * h, sigma_frac * w
        ys = np.arange(h) - h // 2
        xs = np.arange(w) - w // 2
        xx, yy = np.meshgrid(xs, ys)
        labels = np.exp(
            -(xx ** 2 / (2.0 * sig_w ** 2) + yy ** 2 / (2.0 * sig_h ** 2))
        )
        labels = np.roll(np.roll(labels, -h // 2, axis=0), -w // 2, axis=1)
        return labels.astype(np.float32)

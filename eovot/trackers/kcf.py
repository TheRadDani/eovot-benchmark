"""KCF (Kernelized Correlation Filter) tracker.

Reference:
    Henriques et al., "High-Speed Tracking with Kernelized Correlation
    Filters." IEEE Transactions on Pattern Analysis and Machine
    Intelligence (TPAMI), 2015.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox


class KCFTracker(BaseTracker):
    """Gaussian-kernel Correlation Filter (KCF) tracker.

    Extends the linear/MOSSE correlation filter with a non-linear Gaussian
    (RBF) kernel.  Thanks to the circulant structure of the kernel matrix,
    all matrix inversions reduce to element-wise division in the Fourier
    domain, keeping the per-frame cost at O(N log N) — identical to MOSSE
    in algorithmic complexity but more discriminative in practice.

    Key differences from MOSSE:

    * **Gaussian kernel** maps features into an infinite-dimensional RKHS,
      making the classifier non-linear and more robust to appearance change.
    * **Larger search window** (``padding`` factor) reduces drift on
      fast-moving targets without extra cost.
    * **Lower learning rate** (0.075 vs 0.125) prevents over-fitting on a
      single frame — beneficial for longer sequences.

    Suitable for edge deployment: pure NumPy + OpenCV, no GPU required.
    Expected throughput: ~150–350 FPS on a single modern CPU core.

    Args:
        learning_rate: EMA weight for online filter updates, in (0, 1].
            Default: ``0.075``.
        lambda_: Ridge-regression regularisation term. Default: ``1e-4``.
        padding: Context size as a fraction of the target size added on
            each side. Default: ``1.5`` (search window is 2.5× target).
        kernel_sigma: Bandwidth of the Gaussian (RBF) kernel. Smaller
            values give a sharper, more localised kernel response.
            Default: ``0.5``.
    """

    def __init__(
        self,
        learning_rate: float = 0.075,
        lambda_: float = 1e-4,
        padding: float = 1.5,
        kernel_sigma: float = 0.5,
    ) -> None:
        super().__init__(name="KCF")
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.padding = padding
        self.kernel_sigma = kernel_sigma

        self._pos: Optional[Tuple[float, float]] = None
        self._target_sz: Optional[Tuple[int, int]] = None
        self._search_sz: Optional[Tuple[int, int]] = None
        self._alphaf: Optional[np.ndarray] = None
        self._xf: Optional[np.ndarray] = None
        self._window: Optional[np.ndarray] = None
        self._yf: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the KCF filter on the first frame.

        Args:
            frame: First frame as a (H, W, C) BGR array or (H, W) grayscale.
            bbox: Initial bounding box ``(x, y, w, h)`` in pixel coordinates.
        """
        x, y, w, h = (float(v) for v in bbox)
        cx, cy = x + w / 2.0, y + h / 2.0

        self._target_sz = (max(1, int(round(w))), max(1, int(round(h))))
        self._search_sz = (
            max(1, int(round(w * (1.0 + self.padding)))),
            max(1, int(round(h * (1.0 + self.padding)))),
        )
        self._pos = (cx, cy)

        sw, sh = self._search_sz
        self._window = self._hann2d(sh, sw)
        self._yf = np.fft.fft2(self._gaussian_labels(sh, sw))

        patch = self._extract(frame, cx, cy)
        xf = np.fft.fft2(patch * self._window)
        kf = self._kernel_corr(xf, xf)
        self._alphaf = self._yf / (kf + self.lambda_)
        self._xf = xf

    def update(self, frame: np.ndarray) -> BBox:
        """Track the target in a new frame.

        Args:
            frame: Current frame (H, W, C) BGR or (H, W) grayscale.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._pos is None:
            raise RuntimeError("KCFTracker is not initialised. Call initialize() first.")

        cx, cy = self._pos

        # --- Detection ---
        patch = self._extract(frame, cx, cy)
        zf = np.fft.fft2(patch * self._window)
        kzf = self._kernel_corr(self._xf, zf)
        response = np.real(np.fft.ifft2(self._alphaf * kzf))

        dy, dx = np.unravel_index(np.argmax(response), response.shape)
        sh, sw = response.shape
        if dy > sh // 2:
            dy -= sh
        if dx > sw // 2:
            dx -= sw

        new_cx = cx + float(dx)
        new_cy = cy + float(dy)
        self._pos = (new_cx, new_cy)

        # --- Online update (EMA on template and filter coefficients) ---
        new_patch = self._extract(frame, new_cx, new_cy)
        new_xf = np.fft.fft2(new_patch * self._window)
        new_kf = self._kernel_corr(new_xf, new_xf)
        new_alphaf = self._yf / (new_kf + self.lambda_)

        lr = self.learning_rate
        self._xf = (1.0 - lr) * self._xf + lr * new_xf
        self._alphaf = (1.0 - lr) * self._alphaf + lr * new_alphaf

        tw, th = self._target_sz
        return (new_cx - tw / 2.0, new_cy - th / 2.0, float(tw), float(th))

    def reset(self) -> None:
        """Reset all internal state so the tracker can be re-initialised."""
        self._pos = None
        self._target_sz = None
        self._search_sz = None
        self._alphaf = None
        self._xf = None
        self._window = None
        self._yf = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract(self, frame: np.ndarray, cx: float, cy: float) -> np.ndarray:
        """Extract and pre-process a grayscale patch centred at ``(cx, cy)``.

        Steps: convert to grayscale → crop with edge-padding for
        out-of-bounds regions → resize to ``search_sz`` →
        log-normalise → zero-mean unit-variance standardisation.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        sw, sh = self._search_sz
        x1 = int(round(cx - sw / 2.0))
        y1 = int(round(cy - sh / 2.0))
        x2, y2 = x1 + sw, y1 + sh

        fh, fw = gray.shape[:2]
        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - fw)
        pad_b = max(0, y2 - fh)
        if pad_l or pad_t or pad_r or pad_b:
            gray = np.pad(gray, ((pad_t, pad_b), (pad_l, pad_r)), mode="edge")
            x1 += pad_l
            y1 += pad_t
            x2 += pad_l
            y2 += pad_t

        patch = gray[y1:y2, x1:x2].astype(np.float32)
        if patch.shape != (sh, sw):
            patch = cv2.resize(patch, (sw, sh))

        patch = np.log1p(patch)
        patch = (patch - patch.mean()) / (patch.std() + 1e-5)
        return patch

    def _kernel_corr(self, xf: np.ndarray, zf: np.ndarray) -> np.ndarray:
        """Gaussian (RBF) kernel correlation in the Fourier domain.

        Computes the DFT of the kernel response map evaluated at all cyclic
        shifts of the search window::

            k(delta) = exp(-||x - z_delta||^2 / sigma^2)

        Using Parseval's theorem this reduces to element-wise FFT operations,
        keeping the cost at O(N log N).

        Args:
            xf: DFT of the template patch.
            zf: DFT of the candidate patch.

        Returns:
            DFT of the Gaussian kernel response map.
        """
        N = xf.shape[0] * xf.shape[1]
        xx = np.real(np.sum(xf * np.conj(xf))) / N
        zz = np.real(np.sum(zf * np.conj(zf))) / N
        cross = np.real(np.fft.ifft2(np.conj(xf) * zf)) / N
        exponent = np.maximum(0.0, xx + zz - 2.0 * cross) / (self.kernel_sigma ** 2)
        return np.fft.fft2(np.exp(-exponent))

    @staticmethod
    def _hann2d(h: int, w: int) -> np.ndarray:
        """2-D Hann (cosine) window to suppress spectral leakage at patch edges."""
        return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)

    @staticmethod
    def _gaussian_labels(h: int, w: int, sigma_frac: float = 0.1) -> np.ndarray:
        """Soft Gaussian regression target centred at the search-window origin.

        The peak is placed at index ``(0, 0)`` via :func:`numpy.roll` to match
        the circular-shift convention used by :func:`numpy.fft.fft2`.

        Args:
            h: Patch height in pixels.
            w: Patch width in pixels.
            sigma_frac: Standard deviation as a fraction of patch dimension.

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

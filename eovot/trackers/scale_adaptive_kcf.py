"""Scale-Adaptive KCF tracker (DSST-lite).

Extends the vanilla KCF translation filter with a discrete scale-pool
search inspired by the Discriminative Scale Space Tracker (DSST,
Danelljan et al., BMVC 2014).

Vanilla KCF always returns a bounding box of the same size as the
initialisation frame.  This is fine for targets with constant size but
degrades measurably on sequences with scale variation, which is a
standard VOT challenge attribute.  The scale-pool search adds only
``n_scales - 1`` extra patch extractions per frame, keeping throughput
suitable for edge deployment.

Algorithm
---------
Each ``update()`` call runs three sequential steps:

1. **Translation detection** — standard KCF correlation filter response
   to find the new target centre ``(cx, cy)``.  The filter template is
   *not* updated yet.

2. **Scale pool search** — the translation filter response is evaluated
   at ``n_scales`` candidate patch sizes around the current estimate
   (e.g. ×0.9025, ×0.95, ×1.0, ×1.05, ×1.1025 for ``n_scales=5``,
   ``scale_step=1.05``).  For each candidate size the raw patch is
   extracted, resized to the fixed filter dimensions, then preprocessed
   before correlation.  The candidate with the highest peak response is
   chosen and blended with the running scale estimate via EMA.

3. **Template update** — a new search patch is extracted at the found
   position; both the feature template and the filter coefficients are
   updated with EMA (same as vanilla KCF).

Example::

    from eovot.trackers.scale_adaptive_kcf import ScaleAdaptiveKCFTracker

    tracker = ScaleAdaptiveKCFTracker(n_scales=5, scale_step=1.05)
    tracker.initialize(frame0, (x, y, w, h))
    for frame in frames[1:]:
        bbox = tracker.update(frame)   # returns (x, y, w, h) with updated scale
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox


class ScaleAdaptiveKCFTracker(BaseTracker):
    """KCF translation filter augmented with discrete scale-pool estimation.

    Unlike vanilla KCF (which fixes the returned bbox size at initialisation),
    this tracker maintains a running scale estimate updated each frame via a
    small pool search.  The extra cost is proportional to ``n_scales``, each
    requiring one raw patch extraction and one FFT-based response evaluation.

    Args:
        learning_rate: EMA weight for filter updates in ``(0, 1]``.
            Default: ``0.075``.
        lambda_: Ridge-regression regularisation term. Default: ``1e-4``.
        padding: Context size as a fraction of the target size added on
            each side (search window = target × ``1 + padding``).
            Default: ``1.5``.
        kernel_sigma: Bandwidth of the Gaussian RBF kernel. Default: ``0.5``.
        scale_step: Multiplicative step between adjacent scale candidates.
            Must be > 1.  Default: ``1.05`` (5 % per step).
        n_scales: Number of candidate scales to evaluate per frame.  Must
            be a positive odd integer so the current scale (factor 1.0) is
            always included.  Default: ``5``.
        scale_lr: EMA weight for the running scale estimate in ``(0, 1]``.
            Smaller values smooth rapid scale changes; ``1.0`` accepts the
            current frame's estimate without smoothing.  Default: ``0.35``.
        min_scale: Minimum allowed accumulated scale factor (relative to
            initialisation size).  Prevents runaway shrinkage.
            Default: ``0.1``.
        max_scale: Maximum allowed accumulated scale factor.
            Default: ``10.0``.
    """

    def __init__(
        self,
        learning_rate: float = 0.075,
        lambda_: float = 1e-4,
        padding: float = 1.5,
        kernel_sigma: float = 0.5,
        scale_step: float = 1.05,
        n_scales: int = 5,
        scale_lr: float = 0.35,
        min_scale: float = 0.1,
        max_scale: float = 10.0,
    ) -> None:
        super().__init__(name="ScaleAdaptiveKCF")
        if scale_step <= 1.0:
            raise ValueError(f"scale_step must be > 1.0, got {scale_step}.")
        if n_scales < 1 or n_scales % 2 == 0:
            raise ValueError(f"n_scales must be a positive odd integer, got {n_scales}.")
        if not (0.0 < learning_rate <= 1.0):
            raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}.")
        if min_scale <= 0 or max_scale <= min_scale:
            raise ValueError(
                f"Require 0 < min_scale < max_scale, got [{min_scale}, {max_scale}]."
            )

        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.padding = padding
        self.kernel_sigma = kernel_sigma
        self.scale_step = scale_step
        self.n_scales = n_scales
        self.scale_lr = scale_lr
        self.min_scale = min_scale
        self.max_scale = max_scale

        # Scale pool: step^(-(n//2)), ..., step^0 (=1.0), ..., step^(n//2)
        half = n_scales // 2
        self._scale_pool: List[float] = [scale_step ** (i - half) for i in range(n_scales)]

        # State initialised by initialize()
        self._pos: Optional[Tuple[float, float]] = None
        self._init_sz: Optional[Tuple[float, float]] = None   # target w/h at init
        self._search_sz: Optional[Tuple[int, int]] = None     # fixed search window (sw, sh)
        self._current_scale: float = 1.0                      # running scale estimate
        self._alphaf: Optional[np.ndarray] = None             # filter coefficients (freq)
        self._xf: Optional[np.ndarray] = None                 # feature template (freq)
        self._window: Optional[np.ndarray] = None             # 2-D Hann window
        self._yf: Optional[np.ndarray] = None                 # regression target (freq)

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the tracker on the first frame.

        Args:
            frame: BGR image as ``(H, W, 3)`` uint8 array, or grayscale.
            bbox:  Initial bounding box ``(x, y, w, h)``.
        """
        x, y, w, h = (float(v) for v in bbox)
        cx, cy = x + w / 2.0, y + h / 2.0

        self._init_sz = (w, h)
        self._current_scale = 1.0
        self._pos = (cx, cy)

        sw = max(1, int(round(w * (1.0 + self.padding))))
        sh = max(1, int(round(h * (1.0 + self.padding))))
        self._search_sz = (sw, sh)

        self._window = self._hann2d(sh, sw)
        self._yf = np.fft.fft2(self._gaussian_labels(sh, sw))

        patch = self._preprocess(self._crop(frame, cx, cy, sw, sh))
        xf = np.fft.fft2(patch * self._window)
        kf = self._kernel_corr(xf, xf)
        self._alphaf = self._yf / (kf + self.lambda_)
        self._xf = xf

    def update(self, frame: np.ndarray) -> BBox:
        """Predict target location and scale in a new frame.

        Args:
            frame: BGR image as ``(H, W, 3)`` uint8 array, or grayscale.

        Returns:
            Predicted bounding box ``(x, y, w, h)`` with updated scale.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._pos is None:
            raise RuntimeError(
                "ScaleAdaptiveKCFTracker is not initialised — call initialize() first."
            )

        cx, cy = self._pos
        sw, sh = self._search_sz

        # ----------------------------------------------------------
        # Step 1: Translation detection (no template update yet)
        # ----------------------------------------------------------
        patch = self._preprocess(self._crop(frame, cx, cy, sw, sh))
        zf = np.fft.fft2(patch * self._window)
        kzf = self._kernel_corr(self._xf, zf)
        response = np.real(np.fft.ifft2(self._alphaf * kzf))

        dy, dx = np.unravel_index(np.argmax(response), response.shape)
        if dy > sh // 2:
            dy -= sh
        if dx > sw // 2:
            dx -= sw

        new_cx = cx + float(dx)
        new_cy = cy + float(dy)

        # ----------------------------------------------------------
        # Step 2: Scale pool search at the new position
        #
        # For each candidate scale factor s we extract a raw patch of
        # size proportional to s × current_scale × init_sz, resize it
        # to the fixed filter dimensions (sw, sh), preprocess, and
        # compute the correlation response.  The scale with the highest
        # peak is selected and EMA-blended into the running estimate.
        # ----------------------------------------------------------
        best_scale_factor = 1.0
        best_peak = -np.inf

        iw, ih = self._init_sz  # type: ignore[misc]
        for s in self._scale_pool:
            cand_scale = self._current_scale * s
            cand_w = max(1, int(round(iw * cand_scale * (1.0 + self.padding))))
            cand_h = max(1, int(round(ih * cand_scale * (1.0 + self.padding))))

            # Extract raw grayscale patch at candidate dimensions
            raw = self._crop(frame, new_cx, new_cy, cand_w, cand_h)
            if raw is None or raw.size == 0:
                continue

            # Resize to fixed filter dimensions, THEN preprocess.
            # (Preprocessing before resize would distort spatial frequencies.)
            if raw.shape != (sh, sw):
                raw = cv2.resize(raw, (sw, sh))

            cand_patch = self._preprocess(raw)
            if cand_patch is None:
                continue

            zf_cand = np.fft.fft2(cand_patch * self._window)
            kzf_cand = self._kernel_corr(self._xf, zf_cand)
            resp_cand = np.real(np.fft.ifft2(self._alphaf * kzf_cand))
            peak = float(np.nanmax(resp_cand))

            if peak > best_peak:
                best_peak = peak
                best_scale_factor = s

        # EMA-blend the winning scale factor into the running estimate
        new_scale = self._current_scale * (
            (1.0 - self.scale_lr) + self.scale_lr * best_scale_factor
        )
        self._current_scale = float(np.clip(new_scale, self.min_scale, self.max_scale))
        self._pos = (new_cx, new_cy)

        # ----------------------------------------------------------
        # Step 3: Template update at new position (fixed search size)
        # ----------------------------------------------------------
        new_patch = self._preprocess(self._crop(frame, new_cx, new_cy, sw, sh))
        new_xf = np.fft.fft2(new_patch * self._window)
        new_kf = self._kernel_corr(new_xf, new_xf)
        new_alphaf = self._yf / (new_kf + self.lambda_)

        lr = self.learning_rate
        self._xf = (1.0 - lr) * self._xf + lr * new_xf
        self._alphaf = (1.0 - lr) * self._alphaf + lr * new_alphaf

        # ----------------------------------------------------------
        # Return bbox at new position with updated scale
        # ----------------------------------------------------------
        tw = max(1.0, iw * self._current_scale)
        th = max(1.0, ih * self._current_scale)
        return (new_cx - tw / 2.0, new_cy - th / 2.0, tw, th)

    def reset(self) -> None:
        """Reset all internal state so the tracker can be re-initialised."""
        self._pos = None
        self._init_sz = None
        self._search_sz = None
        self._current_scale = 1.0
        self._alphaf = None
        self._xf = None
        self._window = None
        self._yf = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _crop(
        frame: np.ndarray,
        cx: float,
        cy: float,
        w: int,
        h: int,
    ) -> Optional[np.ndarray]:
        """Extract a raw float32 grayscale region of size ``(h, w)`` centred at ``(cx, cy)``.

        Regions that extend beyond the frame boundary are filled by edge-replication
        (same convention as vanilla KCF).  Returns ``None`` if the patch is empty.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        x1 = int(round(cx - w / 2.0))
        y1 = int(round(cy - h / 2.0))
        x2, y2 = x1 + w, y1 + h

        fh, fw = gray.shape[:2]
        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - fw)
        pad_b = max(0, y2 - fh)
        if pad_l or pad_t or pad_r or pad_b:
            gray = np.pad(gray, ((pad_t, pad_b), (pad_l, pad_r)), mode="edge")
            x1 += pad_l
            y1 += pad_t

        patch = gray[y1 : y1 + h, x1 : x1 + w].astype(np.float32)
        if patch.size == 0:
            return None
        return patch

    @staticmethod
    def _preprocess(patch: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Log-normalise and zero-mean, unit-variance standardise a float32 patch."""
        if patch is None:
            return None
        patch = np.log1p(np.maximum(patch, 0.0))  # clamp to [0, ∞) before log
        std = patch.std()
        patch = (patch - patch.mean()) / (std + 1e-5)
        return patch

    def _kernel_corr(self, xf: np.ndarray, zf: np.ndarray) -> np.ndarray:
        """Gaussian (RBF) kernel correlation in the Fourier domain."""
        N = xf.shape[0] * xf.shape[1]
        xx = np.real(np.sum(xf * np.conj(xf))) / N
        zz = np.real(np.sum(zf * np.conj(zf))) / N
        cross = np.real(np.fft.ifft2(np.conj(xf) * zf)) / N
        exponent = np.maximum(0.0, xx + zz - 2.0 * cross) / (self.kernel_sigma ** 2)
        return np.fft.fft2(np.exp(-exponent))

    @staticmethod
    def _hann2d(h: int, w: int) -> np.ndarray:
        """2-D Hann window for spectral leakage suppression."""
        return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)

    @staticmethod
    def _gaussian_labels(h: int, w: int, sigma_frac: float = 0.1) -> np.ndarray:
        """Soft Gaussian regression target at the circulant-shift origin."""
        sig_h, sig_w = sigma_frac * h, sigma_frac * w
        ys = np.arange(h) - h // 2
        xs = np.arange(w) - w // 2
        xx, yy = np.meshgrid(xs, ys)
        labels = np.exp(
            -(xx ** 2 / (2.0 * sig_w ** 2) + yy ** 2 / (2.0 * sig_h ** 2))
        )
        labels = np.roll(np.roll(labels, -h // 2, axis=0), -w // 2, axis=1)
        return labels.astype(np.float32)

"""Discriminative Scale Space Tracker (DSST) for EOVOT.

Reference
---------
Danelljan, M., Häger, G., Khan, F. S., & Felsberg, M. (2014).
Accurate Scale Estimation for Robust Visual Tracking.
BMVC 2014.

Architecture
~~~~~~~~~~~~
1. **Translation filter** — identical to KCF: Gaussian-kernel CF localises
   the target centre in a padded grayscale search window.
2. **Scale estimation** — at each frame a scale pyramid of S candidate
   patches is evaluated.  The candidate whose log-normalised L2 feature
   vector has the highest cosine similarity to the learned appearance
   template is selected as the new scale.  The template is updated online
   with exponential moving average (EMA).

Why cosine-similarity scale estimation?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The original DSST paper uses multi-channel HOG features fed into a separate
1-D correlation filter.  Reproducing that exactly requires either a scipy
dependency (for HOG) or significant extra code.  Cosine-similarity on
log-normalised patches is a simpler, dependency-free alternative that
correctly captures scale by measuring appearance similarity: when the patch
is extracted at the *natural* target scale and resized to the fixed analysis
size, it is most similar to the template extracted under the same conditions
at initialisation.

Edge deployment profile
~~~~~~~~~~~~~~~~~~~~~~~
* Pure NumPy + OpenCV — no GPU or deep-learning framework required.
* Per-frame cost: O(N log N) translation FFT + S × O(M) cosine similarity
  evaluations, where N = search-window pixels, M = 32×32 = 1024.
* Expected throughput: 120–250 FPS on a modern CPU core.

Usage::

    from eovot.trackers.dsst import DSSTTracker

    tracker = DSSTTracker()
    tracker.initialize(first_frame, (x, y, w, h))
    for frame in subsequent_frames:
        x, y, w, h = tracker.update(frame)
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox

# Fixed analysis size for scale-pyramid patches (height, width).
_SCALE_FEAT_SIZE: Tuple[int, int] = (32, 32)


class DSSTTracker(BaseTracker):
    """Scale-adaptive correlation filter tracker (DSST).

    Extends KCF with a scale estimation step that adapts the bounding-box
    size every frame, resolving the core weakness of fixed-box trackers
    (MOSSE, KCF) on sequences with target-size variation.

    Args:
        translation_lr:   EMA rate for the translation filter (0 < lr ≤ 1).
                          Default: 0.075.
        scale_lr:         EMA rate for the appearance scale template.
                          Default: 0.025 (slow adaptation prevents drift).
        padding:          Fractional padding on each side of the target to
                          form the translation search window.  Default: 1.5.
        lambda_:          Ridge-regression regularisation.  Default: 1e-4.
        kernel_sigma:     RBF kernel bandwidth (translation).  Default: 0.5.
        num_scales:       Number of levels in the scale pyramid (must be
                          odd).  Default: 17 (levels −8 … 0 … +8).
        scale_step:       Multiplicative ratio between adjacent scale levels.
                          Default: 1.02 (~2 % per level).
        min_scale_factor: Hard lower bound on the accumulated scale.
                          Default: 0.2.
        max_scale_factor: Hard upper bound on the accumulated scale.
                          Default: 5.0.

    Example::

        tracker = DSSTTracker(num_scales=17, scale_step=1.02)
        tracker.initialize(frame0, (100, 80, 60, 60))
        for frame in frames[1:]:
            x, y, w, h = tracker.update(frame)
    """

    def __init__(
        self,
        translation_lr: float = 0.075,
        scale_lr: float = 0.025,
        padding: float = 1.5,
        lambda_: float = 1e-4,
        kernel_sigma: float = 0.5,
        num_scales: int = 17,
        scale_step: float = 1.02,
        min_scale_factor: float = 0.2,
        max_scale_factor: float = 5.0,
    ) -> None:
        super().__init__(name="DSST")

        if num_scales % 2 == 0:
            raise ValueError(f"num_scales must be odd, got {num_scales}.")
        if not 0 < translation_lr <= 1:
            raise ValueError(f"translation_lr must be in (0, 1], got {translation_lr}.")
        if not 0 < scale_lr <= 1:
            raise ValueError(f"scale_lr must be in (0, 1], got {scale_lr}.")

        self.translation_lr = translation_lr
        self.scale_lr = scale_lr
        self.padding = padding
        self.lambda_ = lambda_
        self.kernel_sigma = kernel_sigma
        self.num_scales = num_scales
        self.scale_step = scale_step
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor

        # Scale-pyramid factors: [step^(−half), …, step^0, …, step^(+half)]
        half = (num_scales - 1) // 2
        self._scale_factors: np.ndarray = (
            np.float64(scale_step) ** np.arange(-half, half + 1, dtype=np.float64)
        )

        # --- Translation filter state ---
        self._pos: Optional[Tuple[float, float]] = None
        self._base_target_sz: Optional[Tuple[int, int]] = None  # (w, h) fixed at init
        self._search_sz: Optional[Tuple[int, int]] = None       # (sw, sh) fixed at init
        self._alphaf: Optional[np.ndarray] = None
        self._xf: Optional[np.ndarray] = None
        self._window: Optional[np.ndarray] = None
        self._yf: Optional[np.ndarray] = None

        # --- Scale estimation state ---
        # Normalized appearance template (flattened 32×32 patch), shape (1024,).
        self._scale_template: Optional[np.ndarray] = None

        # Accumulated scale multiplier (1.0 at init, updated each frame).
        self._current_scale: float = 1.0

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the tracker on the first frame.

        Args:
            frame: BGR image (H, W, 3) uint8, or (H, W) grayscale.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        x, y, w, h = (float(v) for v in bbox)
        cx, cy = x + w / 2.0, y + h / 2.0
        self._pos = (cx, cy)
        self._base_target_sz = (max(1, int(round(w))), max(1, int(round(h))))
        self._current_scale = 1.0

        bw, bh = self._base_target_sz
        sw = max(1, int(round(bw * (1.0 + self.padding))))
        sh = max(1, int(round(bh * (1.0 + self.padding))))
        self._search_sz = (sw, sh)

        gray = self._to_gray(frame)

        # --- Translation filter ---
        self._window = self._hann2d(sh, sw)
        self._yf = np.fft.fft2(self._gaussian_labels(sh, sw))
        patch = self._extract_trans(gray, cx, cy)
        xf = np.fft.fft2(patch * self._window)
        kf = self._kernel_corr(xf, xf)
        self._alphaf = self._yf / (kf + self.lambda_)
        self._xf = xf

        # --- Scale appearance template ---
        self._scale_template = self._build_scale_feat(gray, cx, cy, self._current_scale)

    def update(self, frame: np.ndarray) -> BBox:
        """Estimate target location and scale in the next frame.

        Args:
            frame: BGR image (H, W, 3) uint8, or (H, W) grayscale.

        Returns:
            Predicted bounding box ``(x, y, w, h)`` with updated scale.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._pos is None:
            raise RuntimeError(
                "DSSTTracker is not initialised. Call initialize() first."
            )

        cx, cy = self._pos
        gray = self._to_gray(frame)

        # --- Step 1: Translation detection ---
        patch = self._extract_trans(gray, cx, cy)
        zf = np.fft.fft2(patch * self._window)
        kzf = self._kernel_corr(self._xf, zf)
        response = np.real(np.fft.ifft2(self._alphaf * kzf))

        sh, sw = response.shape
        dy_raw, dx_raw = np.unravel_index(response.argmax(), response.shape)
        dy = int(dy_raw) if dy_raw < sh // 2 else int(dy_raw) - sh
        dx = int(dx_raw) if dx_raw < sw // 2 else int(dx_raw) - sw
        cx += float(dx)
        cy += float(dy)
        self._pos = (cx, cy)

        # --- Step 2: Scale estimation via cosine similarity ---
        best_idx = self._estimate_scale(gray, cx, cy)
        scale_delta = float(self._scale_factors[best_idx])
        self._current_scale = float(
            np.clip(
                self._current_scale * scale_delta,
                self.min_scale_factor,
                self.max_scale_factor,
            )
        )

        # --- Step 3: Online filter update ---
        new_patch = self._extract_trans(gray, cx, cy)
        new_xf = np.fft.fft2(new_patch * self._window)
        new_kf = self._kernel_corr(new_xf, new_xf)
        new_alphaf = self._yf / (new_kf + self.lambda_)

        lr_t = self.translation_lr
        self._xf = (1.0 - lr_t) * self._xf + lr_t * new_xf
        self._alphaf = (1.0 - lr_t) * self._alphaf + lr_t * new_alphaf

        # Update appearance template toward the detected scale
        new_feat = self._build_scale_feat(gray, cx, cy, self._current_scale)
        lr_s = self.scale_lr
        tmpl = (1.0 - lr_s) * self._scale_template + lr_s * new_feat
        norm = np.linalg.norm(tmpl)
        self._scale_template = tmpl / (norm + 1e-9)

        # --- Output: scaled bounding box centred at updated position ---
        bw, bh = self._base_target_sz
        tw = float(bw) * self._current_scale
        th = float(bh) * self._current_scale
        return (cx - tw / 2.0, cy - th / 2.0, tw, th)

    def reset(self) -> None:
        """Clear internal state so the tracker can be re-initialised."""
        self._pos = None
        self._base_target_sz = None
        self._search_sz = None
        self._alphaf = None
        self._xf = None
        self._window = None
        self._yf = None
        self._scale_template = None
        self._current_scale = 1.0

    # ------------------------------------------------------------------
    # Scale estimation
    # ------------------------------------------------------------------

    def _estimate_scale(self, gray: np.ndarray, cx: float, cy: float) -> int:
        """Return the index in ``_scale_factors`` with the best scale match.

        For each of the ``num_scales`` pyramid levels, a patch is extracted at
        the candidate scale, resized to ``_SCALE_FEAT_SIZE``, log-normalised,
        and L2-normalised.  The index whose feature vector has the highest
        cosine similarity to the stored appearance template is returned.

        Args:
            gray: Single-channel grayscale frame.
            cx, cy: Current target centre estimate.

        Returns:
            Index in ``[0, num_scales)`` corresponding to the best scale.
        """
        similarities = np.empty(self.num_scales, dtype=np.float64)
        for i, sf in enumerate(self._scale_factors):
            feat = self._build_scale_feat(gray, cx, cy, self._current_scale * float(sf))
            similarities[i] = float(np.dot(self._scale_template, feat))
        return int(np.argmax(similarities))

    def _build_scale_feat(
        self,
        gray: np.ndarray,
        cx: float,
        cy: float,
        scale: float,
    ) -> np.ndarray:
        """Build a normalised appearance feature vector at the given scale.

        Extracts a patch of size ``base_target_sz × scale``, resizes it to
        ``_SCALE_FEAT_SIZE``, applies log-normalisation and zero-mean centering,
        then L2-normalises the result.

        Args:
            gray: Single-channel grayscale frame.
            cx, cy: Target centre.
            scale: Effective scale multiplier applied to ``base_target_sz``.

        Returns:
            Float64 array of shape ``(32×32,)`` with L2 norm ≤ 1.
        """
        bw, bh = self._base_target_sz
        pw = max(1, int(round(bw * scale)))
        ph = max(1, int(round(bh * scale)))

        patch = self._extract_region(gray, cx, cy, pw, ph)
        fw, fh = _SCALE_FEAT_SIZE
        patch_r = cv2.resize(
            patch.astype(np.float32), (fw, fh), interpolation=cv2.INTER_LINEAR
        ).astype(np.float64)
        patch_r = np.log1p(patch_r)
        patch_r -= patch_r.mean()
        norm = np.linalg.norm(patch_r)
        return patch_r.ravel() / (norm + 1e-9)

    # ------------------------------------------------------------------
    # Translation filter helpers
    # ------------------------------------------------------------------

    def _extract_trans(self, gray: np.ndarray, cx: float, cy: float) -> np.ndarray:
        """Extract the log-normalised translation search patch."""
        sw, sh = self._search_sz
        patch = self._extract_region(gray, cx, cy, sw, sh)
        patch = np.log1p(patch.astype(np.float64))
        patch -= patch.mean()
        std = patch.std()
        if std > 1e-7:
            patch /= std
        return patch

    def _extract_region(
        self,
        gray: np.ndarray,
        cx: float,
        cy: float,
        w: int,
        h: int,
    ) -> np.ndarray:
        """Extract an ``(h, w)`` region centred at ``(cx, cy)`` with edge padding.

        Out-of-bounds areas are filled by replicating the border pixel values
        so the FFT sees a smooth extension rather than a black border.
        """
        fh, fw = gray.shape[:2]
        x1 = int(round(cx - w / 2.0))
        y1 = int(round(cy - h / 2.0))
        x2 = x1 + w
        y2 = y1 + h

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

        patch = gray[y1:y2, x1:x2]
        if patch.shape != (h, w):
            patch = cv2.resize(patch, (w, h), interpolation=cv2.INTER_LINEAR)
        return patch.astype(np.float64)

    def _kernel_corr(self, xf: np.ndarray, zf: np.ndarray) -> np.ndarray:
        """Gaussian (RBF) kernel correlation in the Fourier domain.

        Computes the DFT of ``k(x, z_delta) = exp(−||x − z_delta||² / σ²)``
        for all cyclic shifts ``delta`` via Parseval's theorem, keeping the
        cost at O(N log N).

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

    # ------------------------------------------------------------------
    # Shared static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        """Convert a BGR or BGRA frame to single-channel uint8 grayscale."""
        if frame.ndim == 2:
            return frame
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _hann2d(h: int, w: int) -> np.ndarray:
        """2-D Hann window to suppress spectral leakage at patch boundaries."""
        return np.outer(np.hanning(h), np.hanning(w)).astype(np.float64)

    @staticmethod
    def _gaussian_labels(h: int, w: int, sigma_frac: float = 0.1) -> np.ndarray:
        """Soft 2-D Gaussian regression target for the translation filter.

        Peak at index ``(0, 0)`` to match the circular-shift convention of
        :func:`numpy.fft.fft2`.

        Args:
            h, w: Patch dimensions.
            sigma_frac: Std as a fraction of the patch dimension.

        Returns:
            Float64 array of shape ``(h, w)`` with values in ``(0, 1]``.
        """
        sig_h = sigma_frac * h + 1e-9
        sig_w = sigma_frac * w + 1e-9
        ys = np.arange(h, dtype=np.float64) - h // 2
        xs = np.arange(w, dtype=np.float64) - w // 2
        xx, yy = np.meshgrid(xs, ys)
        labels = np.exp(-(xx ** 2 / (2.0 * sig_w ** 2) + yy ** 2 / (2.0 * sig_h ** 2)))
        return np.roll(
            np.roll(labels, -h // 2, axis=0), -w // 2, axis=1
        ).astype(np.float64)

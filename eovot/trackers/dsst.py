"""Discriminative Scale Space Tracker (DSST).

Reference:
    Danelljan et al., "Accurate Scale Estimation for Robust Visual
    Tracking." BMVC 2014.  https://www.bmva.org/bmvc/2014/2014/papers/056

Architecture
~~~~~~~~~~~~
Two independent correlation filters run in sequence every frame:

1. **Translation filter** — 2-D Gaussian-kernel correlation filter (identical
   in structure to KCF).  A patch is always extracted at a size proportional
   to ``current_scale × search_window``; it is then *resized* to the fixed
   initial window size ``(sw, sh)`` before going into the filter.  As a
   result, the filter coefficients are always ``(sh, sw)`` regardless of
   scale, and the peak offset ``(dy, dx)`` must be multiplied by
   ``current_scale`` to recover the actual pixel displacement.

2. **Scale filter** — 1-D linear DCF-style filter (no kernel trick).  A
   pyramid of ``n_scales`` patches is extracted at ``base_size × scale_step^k``
   for ``k ∈ {-n//2, …, 0, …, n//2}``.  Each patch is resized to a small
   fixed template (``scale_model_sz``) and flattened; together they form a
   ``(n_feat, n_scales)`` feature matrix.  A 1-D DFT along the scale axis
   locates the best scale level without exhaustive search.

Both filters are updated online with exponential moving averages:
``(1-lr) * old + lr * new``.  The scale filter uses numerator/denominator
EMA so it can be evaluated as ``num / (den + λ)`` at inference time, matching
the original DSST paper.

Suitable for edge deployment: pure NumPy + OpenCV, no GPU required.
Expected throughput: ~80–180 FPS on a single modern CPU core, approximately
1.5–2× slower than KCF due to the scale pyramid overhead.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox


class DSSTTracker(BaseTracker):
    """Scale-adaptive correlation filter tracker (DSST).

    Adds a separate 1-D scale correlation filter on top of the KCF translation
    filter.  At every frame the tracker:

    1. Localises the target centre with the 2-D translation filter.
    2. Searches a pyramid of ``n_scales`` scale candidates with the 1-D scale
       filter and selects the one that maximises the response.
    3. Updates both filters online via EMA.
    4. Returns a bounding box whose *size* reflects the new scale estimate.

    Args:
        n_scales:             Number of scale pyramid levels (forced to odd).
                              Default: ``33``.
        scale_step:           Geometric ratio between consecutive levels.
                              Default: ``1.02`` (±2 % per level).
        scale_lr:             EMA learning rate for the scale filter.
                              Default: ``0.025``.
        translation_lr:       EMA learning rate for the translation filter.
                              Default: ``0.075``.
        lambda_:              Tikhonov regularisation for both filters.
                              Default: ``1e-4``.
        padding:              Fractional padding added to the translation search
                              window on each side. Default: ``1.0``.
        scale_model_max_area: Maximum pixel area of scale sample patches used
                              inside the scale filter.  Smaller values trade
                              accuracy for speed.  Default: ``512``.
        kernel_sigma:         RBF bandwidth for the translation kernel.
                              Default: ``0.5``.

    Example::

        from eovot.trackers.dsst import DSSTTracker
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.benchmark.engine import BenchmarkEngine

        ds = SyntheticDataset(num_sequences=3, motion="circular")
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(DSSTTracker(), ds, dataset_name="Synthetic-Circular")
        print(result)
    """

    def __init__(
        self,
        n_scales: int = 33,
        scale_step: float = 1.02,
        scale_lr: float = 0.025,
        translation_lr: float = 0.075,
        lambda_: float = 1e-4,
        padding: float = 1.0,
        scale_model_max_area: int = 512,
        kernel_sigma: float = 0.5,
    ) -> None:
        super().__init__(name="DSST")
        if n_scales % 2 == 0:
            n_scales += 1  # enforce odd so center level is well-defined
        self.n_scales = n_scales
        self.scale_step = scale_step
        self.scale_lr = scale_lr
        self.translation_lr = translation_lr
        self.lambda_ = lambda_
        self.padding = padding
        self.scale_model_max_area = scale_model_max_area
        self.kernel_sigma = kernel_sigma

        half = n_scales // 2
        # Relative scale factors: [step^(-half), …, 1.0, …, step^(+half)]
        self._scale_factors: np.ndarray = scale_step ** np.arange(-half, half + 1, dtype=np.float64)

        # Mutable state — all initialised in initialize()
        self._pos: Optional[Tuple[float, float]] = None
        self._base_target_sz: Optional[Tuple[int, int]] = None
        self._current_scale: float = 1.0
        # Translation filter
        self._trans_sz: Optional[Tuple[int, int]] = None
        self._trans_window: Optional[np.ndarray] = None
        self._trans_yf: Optional[np.ndarray] = None
        self._trans_alphaf: Optional[np.ndarray] = None
        self._trans_xf: Optional[np.ndarray] = None
        # Scale filter
        self._scale_model_sz: Optional[Tuple[int, int]] = None
        self._scale_yf: Optional[np.ndarray] = None
        self._scale_window: Optional[np.ndarray] = None
        self._scale_num: Optional[np.ndarray] = None  # numerator EMA
        self._scale_den: Optional[np.ndarray] = None  # denominator EMA

    # -------------------------------------------------------------------------
    # BaseTracker interface
    # -------------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise DSST on the first frame.

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 numpy array.
            bbox:  Initial bounding box ``(x, y, w, h)``.
        """
        x, y, w, h = (float(v) for v in bbox)
        cx, cy = x + w / 2.0, y + h / 2.0
        self._pos = (cx, cy)
        self._base_target_sz = (max(1, int(round(w))), max(1, int(round(h))))
        self._current_scale = 1.0

        btw, bth = self._base_target_sz
        sw = max(1, int(round(btw * (1.0 + self.padding))))
        sh = max(1, int(round(bth * (1.0 + self.padding))))
        self._trans_sz = (sw, sh)
        self._trans_window = np.outer(np.hanning(sh), np.hanning(sw)).astype(np.float64)
        self._trans_yf = np.fft.fft2(self._gaussian_labels_2d(sh, sw))

        patch = self._get_trans_patch(frame, cx, cy)
        xf = np.fft.fft2(patch)
        kf = self._kernel_corr(xf, xf)
        self._trans_alphaf = self._trans_yf / (kf + self.lambda_)
        self._trans_xf = xf

        self._scale_model_sz = self._compute_scale_model_sz(btw, bth)
        self._scale_yf = np.fft.fft(self._gaussian_1d(self.n_scales))
        self._scale_window = np.hanning(self.n_scales).astype(np.float64)

        xs = self._get_scale_features(frame, cx, cy)
        xsf = np.fft.fft(xs, axis=1)
        self._scale_num = np.conj(xsf) * self._scale_yf[np.newaxis, :]
        self._scale_den = np.real(np.sum(xsf * np.conj(xsf), axis=0))

    def update(self, frame: np.ndarray) -> BBox:
        """Predict the target location in the current frame.

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 numpy array.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._pos is None:
            raise RuntimeError("DSSTTracker is not initialised. Call initialize() first.")

        cx, cy = self._pos
        sw, sh = self._trans_sz

        # ------------------------------------------------------------------
        # Step 1 — localise target centre with the translation filter
        # ------------------------------------------------------------------
        patch = self._get_trans_patch(frame, cx, cy)
        zf = np.fft.fft2(patch)
        kzf = self._kernel_corr(self._trans_xf, zf)
        resp = np.real(np.fft.ifft2(self._trans_alphaf * kzf))

        dy, dx = np.unravel_index(np.argmax(resp), resp.shape)
        if dy > sh // 2:
            dy -= sh
        if dx > sw // 2:
            dx -= sw
        # Offset in fixed-scale space must be multiplied by current_scale.
        new_cx = cx + float(dx) * self._current_scale
        new_cy = cy + float(dy) * self._current_scale

        # ------------------------------------------------------------------
        # Step 2 — estimate scale with the scale filter
        # ------------------------------------------------------------------
        xs = self._get_scale_features(frame, new_cx, new_cy)
        xsf = np.fft.fft(xs, axis=1)
        alphaf = self._scale_num / (self._scale_den[np.newaxis, :] + self.lambda_)
        scale_resp = np.real(np.fft.ifft(np.sum(alphaf * xsf, axis=0)))
        best_k = int(np.argmax(scale_resp))
        # The IFFT output uses circular DFT convention: argmax=k means the
        # best matching scale column is at index (n//2 + k) % n_scales.
        # Mapping best_k → actual column index recovers the true scale factor.
        half = self.n_scales // 2
        actual_idx = (half + best_k) % self.n_scales
        raw_scale = self._current_scale * self._scale_factors[actual_idx]
        # Guard against unbounded drift
        new_scale = float(np.clip(raw_scale, 0.01, 100.0))

        # ------------------------------------------------------------------
        # Step 3 — update translation filter at the new scale / position
        # ------------------------------------------------------------------
        lr_t = self.translation_lr
        new_patch = self._get_trans_patch(frame, new_cx, new_cy, scale=new_scale)
        new_xf = np.fft.fft2(new_patch)
        new_kf = self._kernel_corr(new_xf, new_xf)
        new_alphaf = self._trans_yf / (new_kf + self.lambda_)
        self._trans_xf = (1.0 - lr_t) * self._trans_xf + lr_t * new_xf
        self._trans_alphaf = (1.0 - lr_t) * self._trans_alphaf + lr_t * new_alphaf

        # ------------------------------------------------------------------
        # Step 4 — update scale filter
        # ------------------------------------------------------------------
        lr_s = self.scale_lr
        new_xs = self._get_scale_features(frame, new_cx, new_cy, scale=new_scale)
        new_xsf = np.fft.fft(new_xs, axis=1)
        self._scale_num = (
            (1.0 - lr_s) * self._scale_num
            + lr_s * np.conj(new_xsf) * self._scale_yf[np.newaxis, :]
        )
        self._scale_den = (
            (1.0 - lr_s) * self._scale_den
            + lr_s * np.real(np.sum(new_xsf * np.conj(new_xsf), axis=0))
        )

        self._pos = (new_cx, new_cy)
        self._current_scale = new_scale

        btw, bth = self._base_target_sz
        tw = max(1, int(round(btw * new_scale)))
        th = max(1, int(round(bth * new_scale)))
        return (new_cx - tw / 2.0, new_cy - th / 2.0, float(tw), float(th))

    def reset(self) -> None:
        """Clear all internal state so the tracker can be re-initialised."""
        self._pos = None
        self._base_target_sz = None
        self._current_scale = 1.0
        self._trans_sz = None
        self._trans_window = None
        self._trans_yf = None
        self._trans_alphaf = None
        self._trans_xf = None
        self._scale_model_sz = None
        self._scale_yf = None
        self._scale_window = None
        self._scale_num = None
        self._scale_den = None

    # -------------------------------------------------------------------------
    # Translation-filter helpers
    # -------------------------------------------------------------------------

    def _get_trans_patch(
        self,
        frame: np.ndarray,
        cx: float,
        cy: float,
        scale: Optional[float] = None,
    ) -> np.ndarray:
        """Extract a scale-normalised translation patch of fixed size ``(sw, sh)``.

        The patch is taken from a region of size ``(sw * scale, sh * scale)``
        so the translation filter always sees target content at a canonical
        scale regardless of the current scale estimate.
        """
        if scale is None:
            scale = self._current_scale
        sw, sh = self._trans_sz
        actual_w = max(1, int(round(sw * scale)))
        actual_h = max(1, int(round(sh * scale)))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        crop = self._crop_padded(gray, cx, cy, actual_w, actual_h)
        if crop.shape != (actual_h, actual_w):
            crop = cv2.resize(crop, (actual_w, actual_h))
        patch = cv2.resize(crop, (sw, sh)).astype(np.float64)
        patch = np.log1p(patch)
        patch -= patch.mean()
        std = patch.std()
        if std > 1e-5:
            patch /= std
        patch *= self._trans_window
        return patch

    def _kernel_corr(self, xf: np.ndarray, zf: np.ndarray) -> np.ndarray:
        """Gaussian (RBF) kernel correlation in the Fourier domain.

        Identical to the KCF kernel but operates on whatever 2-D FFT arrays
        are passed in.  Uses Parseval's theorem to avoid the O(N²) kernel
        matrix computation.
        """
        N = xf.shape[0] * xf.shape[1]
        xx = np.real(np.sum(xf * np.conj(xf))) / N
        zz = np.real(np.sum(zf * np.conj(zf))) / N
        cross = np.real(np.fft.ifft2(np.conj(xf) * zf)) / N
        exp_arg = np.maximum(0.0, xx + zz - 2.0 * cross) / (self.kernel_sigma ** 2)
        return np.fft.fft2(np.exp(-exp_arg))

    @staticmethod
    def _gaussian_labels_2d(h: int, w: int, sigma_frac: float = 0.1) -> np.ndarray:
        """2-D Gaussian regression target with the peak at index (0, 0)."""
        sig_h, sig_w = sigma_frac * h, sigma_frac * w
        ys = np.arange(h) - h // 2
        xs = np.arange(w) - w // 2
        xx, yy = np.meshgrid(xs, ys)
        g = np.exp(-(xx ** 2 / (2 * sig_w ** 2) + yy ** 2 / (2 * sig_h ** 2)))
        g = np.roll(np.roll(g, -h // 2, axis=0), -w // 2, axis=1)
        return g.astype(np.float64)

    # -------------------------------------------------------------------------
    # Scale-filter helpers
    # -------------------------------------------------------------------------

    def _compute_scale_model_sz(self, tw: int, th: int) -> Tuple[int, int]:
        """Compute scale sample template size, capped by ``scale_model_max_area``."""
        area = float(tw * th)
        if area > self.scale_model_max_area:
            factor = float(np.sqrt(self.scale_model_max_area / area))
            return max(1, int(round(tw * factor))), max(1, int(round(th * factor)))
        return max(1, tw), max(1, th)

    def _get_scale_features(
        self,
        frame: np.ndarray,
        cx: float,
        cy: float,
        scale: Optional[float] = None,
    ) -> np.ndarray:
        """Build the ``(n_feat, n_scales)`` scale feature matrix.

        For each of the ``n_scales`` scale levels:

        1. Compute patch dimensions ``base_sz × (current_scale × scale_factor_k)``.
        2. Extract that patch from *frame* (with edge padding where needed).
        3. Resize to the fixed ``scale_model_sz``.
        4. Log-normalise and weight by the 1-D Hann window.
        5. Flatten to a ``n_feat``-element feature vector.

        The resulting matrix ``xs[i, k]`` holds feature element *i* for scale
        level *k* and is ready for a 1-D FFT along axis 1.
        """
        if scale is None:
            scale = self._current_scale
        btw, bth = self._base_target_sz
        sm_w, sm_h = self._scale_model_sz
        n_feat = sm_w * sm_h

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        xs = np.zeros((n_feat, self.n_scales), dtype=np.float64)

        for i, sf in enumerate(self._scale_factors):
            s = scale * sf
            pw = max(1, int(round(btw * s)))
            ph = max(1, int(round(bth * s)))
            crop = self._crop_padded(gray, cx, cy, pw, ph)
            patch = cv2.resize(crop, (sm_w, sm_h)).astype(np.float64)
            patch = np.log1p(patch)
            patch -= patch.mean()
            std = patch.std()
            if std > 1e-5:
                patch /= std
            xs[:, i] = patch.ravel() * self._scale_window[i]

        return xs

    @staticmethod
    def _gaussian_1d(n: int, sigma_frac: float = 0.1) -> np.ndarray:
        """1-D Gaussian scale target with the peak at DFT index 0."""
        sigma = sigma_frac * n
        xs = np.arange(n) - n // 2
        g = np.exp(-(xs ** 2) / (2 * sigma ** 2))
        g = np.roll(g, -(n // 2))
        return g.astype(np.float64)

    @staticmethod
    def _crop_padded(
        gray: np.ndarray, cx: float, cy: float, pw: int, ph: int
    ) -> np.ndarray:
        """Crop a ``(ph × pw)`` region centred at ``(cx, cy)`` with edge padding."""
        x1 = int(round(cx - pw / 2.0))
        y1 = int(round(cy - ph / 2.0))
        x2, y2 = x1 + pw, y1 + ph
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
        return gray[y1:y2, x1:x2]

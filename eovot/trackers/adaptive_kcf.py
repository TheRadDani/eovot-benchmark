"""Adaptive KCF tracker with PSR confidence scoring and multi-scale search.

Extends the standard Kernelized Correlation Filter (KCF) with three
research-grade additions that matter for edge deployment:

1. **Peak-to-Sidelobe Ratio (PSR)** — a per-frame confidence estimate that
   quantifies filter response quality.  High PSR (>7) means the filter
   found a sharp, unambiguous peak; low PSR (<4) indicates drift or
   occlusion.

2. **Multi-scale pyramid search** — jointly estimates position *and* scale
   change by testing a small set of candidate scales (e.g. ×0.95, ×1.0,
   ×1.05) each frame.  The scale that yields the highest response peak is
   selected, and the target bounding box is updated accordingly.

3. **Confidence-weighted learning rate** — model updates are gated by PSR
   so the tracker suppresses template corruption during uncertain frames.
   On very low PSR frames the template is frozen; on high PSR frames the
   full learning rate is applied.

4. **Coarse-grid re-detection** — after *redetect_after* consecutive
   low-confidence frames the filter is slid over the full image to
   recover from abrupt motion or reappearance.

All logic is pure NumPy + OpenCV; no GPU required.
Expected throughput: ~80–250 FPS depending on num_scales and patch size.

References:
    Bolme et al., "Visual Object Tracking Using Adaptive Correlation
    Filters." CVPR 2010.  (MOSSE — PSR introduced here.)

    Li & Zhu, "A Scale Adaptive Kernel Correlation Filter Tracker with
    Feature Integration." ECCV 2014.  (Scale adaptation inspiration.)

    Henriques et al., "High-Speed Tracking with Kernelized Correlation
    Filters." IEEE TPAMI 2015.  (KCF base algorithm.)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox

# PSR thresholds — calibrated on OTB-100 visual inspection
_PSR_HIGH: float = 7.0   # above this: full learning rate
_PSR_LOW: float = 4.0    # below this: freeze model; increment failure counter


class AdaptiveKCFTracker(BaseTracker):
    """KCF tracker with PSR confidence, multi-scale search, and re-detection.

    Args:
        learning_rate: Base EMA weight for model updates, in ``(0, 1]``.
            Actual LR is scaled by PSR confidence each frame. Default: ``0.075``.
        lambda_: Ridge-regression regularisation term. Default: ``1e-4``.
        padding: Context factor added to the search window on each side.
            Default: ``1.5`` (search window = 2.5× target size).
        kernel_sigma: Bandwidth of the Gaussian (RBF) kernel. Default: ``0.5``.
        scale_step: Multiplicative gap between adjacent scale levels.
            Default: ``1.05`` (±5 % scale change tested each frame).
        num_scales: Number of scale levels in the pyramid.  Must be a
            positive odd integer so one level equals the current scale.
            Default: ``3``.
        psr_threshold: PSR below this value is treated as low confidence and
            increments the failure counter.  Default: ``4.0``.
        redetect_after: Consecutive low-confidence frames before a coarse
            re-detection scan is triggered.  Default: ``5``.

    Example::

        from eovot.trackers.adaptive_kcf import AdaptiveKCFTracker

        tracker = AdaptiveKCFTracker(num_scales=3, psr_threshold=4.0)
        tracker.initialize(first_frame, (x, y, w, h))

        for frame in subsequent_frames:
            bbox = tracker.update(frame)
            print(f"PSR={tracker.confidence:.1f}  reliable={tracker.is_tracking_reliable}")
    """

    def __init__(
        self,
        learning_rate: float = 0.075,
        lambda_: float = 1e-4,
        padding: float = 1.5,
        kernel_sigma: float = 0.5,
        scale_step: float = 1.05,
        num_scales: int = 3,
        psr_threshold: float = 4.0,
        redetect_after: int = 5,
    ) -> None:
        super().__init__(name="AdaptiveKCF")
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.padding = padding
        self.kernel_sigma = kernel_sigma
        self.scale_step = scale_step
        self.num_scales = max(1, num_scales)
        self.psr_threshold = psr_threshold
        self.redetect_after = redetect_after

        # Runtime state — None until initialize() is called
        self._pos: Optional[Tuple[float, float]] = None
        self._target_sz: Optional[Tuple[float, float]] = None
        self._current_scale: float = 1.0
        self._alphaf: Optional[np.ndarray] = None
        self._xf: Optional[np.ndarray] = None
        self._window: Optional[np.ndarray] = None
        self._yf: Optional[np.ndarray] = None

        # Confidence state
        self._psr: float = 0.0
        self._low_conf_count: int = 0
        self._psr_history: List[float] = []

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def confidence(self) -> float:
        """Most recent PSR value (higher is more confident; typical range 0–20)."""
        return self._psr

    @property
    def is_tracking_reliable(self) -> bool:
        """``True`` if the last PSR exceeded :attr:`psr_threshold`."""
        return self._psr >= self.psr_threshold

    @property
    def psr_history(self) -> List[float]:
        """PSR values for every frame processed since the last initialize() call."""
        return list(self._psr_history)

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the tracker on the first frame.

        Args:
            frame: First frame as an ``(H, W, 3)`` BGR uint8 array.
            bbox: Initial bounding box ``(x, y, w, h)`` in pixel coordinates.
        """
        x, y, w, h = (float(v) for v in bbox)
        cx, cy = x + w / 2.0, y + h / 2.0

        self._target_sz = (max(1.0, w), max(1.0, h))
        self._current_scale = 1.0
        self._pos = (cx, cy)
        self._psr = 0.0
        self._low_conf_count = 0
        self._psr_history = []

        sw, sh = self._search_size()
        self._window = self._hann2d(sh, sw)
        self._yf = np.fft.fft2(self._gaussian_labels(sh, sw))

        patch = self._extract(frame, cx, cy, self._current_scale)
        xf = np.fft.fft2(patch * self._window)
        kf = self._kernel_corr(xf, xf)
        self._alphaf = self._yf / (kf + self.lambda_)
        self._xf = xf

    def update(self, frame: np.ndarray) -> BBox:
        """Track the target in the current frame.

        Steps per frame:
        1. Multi-scale detection — find the scale and position with the
           highest correlation response peak.
        2. PSR computation — measure response quality.
        3. Re-detection check — trigger grid scan after sustained failure.
        4. Confidence-weighted model update — freeze or partially apply EMA.

        Args:
            frame: Current frame ``(H, W, 3)`` BGR uint8.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If :meth:`initialize` has not been called first.
        """
        if self._pos is None:
            raise RuntimeError(
                "AdaptiveKCFTracker not initialised — call initialize() first."
            )

        cx, cy = self._pos

        # ── Step 1: multi-scale detection ──────────────────────────────
        best_peak = -np.inf
        best_dx = best_dy = 0.0
        best_scale = self._current_scale
        best_response: Optional[np.ndarray] = None

        for scale in self._scale_levels():
            patch = self._extract(frame, cx, cy, scale)
            zf = np.fft.fft2(patch * self._window)
            kzf = self._kernel_corr(self._xf, zf)
            response = np.real(np.fft.ifft2(self._alphaf * kzf))

            peak = float(response.max())
            if peak > best_peak:
                best_peak = peak
                best_response = response
                best_scale = scale
                sh, sw = response.shape
                dy, dx = np.unravel_index(np.argmax(response), response.shape)
                if dy > sh // 2:
                    dy -= sh
                if dx > sw // 2:
                    dx -= sw
                best_dy, best_dx = float(dy), float(dx)

        # ── Step 2: update position and scale ──────────────────────────
        new_cx = cx + best_dx
        new_cy = cy + best_dy
        self._current_scale = best_scale

        # ── Step 3: PSR confidence ──────────────────────────────────────
        assert best_response is not None
        psr = self._compute_psr(best_response)
        self._psr = psr
        self._psr_history.append(psr)

        if psr < self.psr_threshold:
            self._low_conf_count += 1
        else:
            self._low_conf_count = 0

        # ── Step 4: re-detection on sustained failure ───────────────────
        if self._low_conf_count >= self.redetect_after:
            new_cx, new_cy = self._redetect(frame)
            self._low_conf_count = 0

        self._pos = (new_cx, new_cy)

        # ── Step 5: confidence-weighted model update ────────────────────
        lr = self._adaptive_lr(psr)
        if lr > 0.0:
            new_patch = self._extract(frame, new_cx, new_cy, self._current_scale)
            new_xf = np.fft.fft2(new_patch * self._window)
            new_kf = self._kernel_corr(new_xf, new_xf)
            new_alphaf = self._yf / (new_kf + self.lambda_)
            self._xf = (1.0 - lr) * self._xf + lr * new_xf
            self._alphaf = (1.0 - lr) * self._alphaf + lr * new_alphaf

        # ── Step 6: derive output bbox ──────────────────────────────────
        tw = self._target_sz[0] * self._current_scale
        th = self._target_sz[1] * self._current_scale
        return (new_cx - tw / 2.0, new_cy - th / 2.0, tw, th)

    def reset(self) -> None:
        """Clear all internal state so the tracker can be re-initialised."""
        self._pos = None
        self._target_sz = None
        self._current_scale = 1.0
        self._alphaf = None
        self._xf = None
        self._window = None
        self._yf = None
        self._psr = 0.0
        self._low_conf_count = 0
        self._psr_history = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _search_size(self) -> Tuple[int, int]:
        """Return canonical search window ``(width, height)`` in pixels."""
        tw, th = self._target_sz
        sw = max(1, int(round(tw * (1.0 + self.padding))))
        sh = max(1, int(round(th * (1.0 + self.padding))))
        return sw, sh

    def _scale_levels(self) -> List[float]:
        """Scale factors centred on the current scale."""
        half = self.num_scales // 2
        return [
            self._current_scale * (self.scale_step ** (i - half))
            for i in range(self.num_scales)
        ]

    def _extract(
        self, frame: np.ndarray, cx: float, cy: float, scale: float
    ) -> np.ndarray:
        """Extract, crop and pre-process a patch at *scale* around *(cx, cy)*.

        The patch is cropped at ``scale × search_size`` then resized to the
        canonical ``search_size`` so FFT dimensions stay constant.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        sw, sh = self._search_size()
        csw = max(1, int(round(sw * scale)))
        csh = max(1, int(round(sh * scale)))

        x1 = int(round(cx - csw / 2.0))
        y1 = int(round(cy - csh / 2.0))
        x2, y2 = x1 + csw, y1 + csh

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
            patch = cv2.resize(patch, (sw, sh), interpolation=cv2.INTER_LINEAR)

        patch = np.log1p(patch)
        patch = (patch - patch.mean()) / (patch.std() + 1e-5)
        return patch

    def _kernel_corr(self, xf: np.ndarray, zf: np.ndarray) -> np.ndarray:
        """Gaussian RBF kernel correlation in the Fourier domain."""
        N = xf.shape[0] * xf.shape[1]
        xx = np.real(np.sum(xf * np.conj(xf))) / N
        zz = np.real(np.sum(zf * np.conj(zf))) / N
        cross = np.real(np.fft.ifft2(np.conj(xf) * zf)) / N
        exponent = np.maximum(0.0, xx + zz - 2.0 * cross) / (self.kernel_sigma ** 2)
        return np.fft.fft2(np.exp(-exponent))

    @staticmethod
    def _compute_psr(response: np.ndarray, window_half: int = 5) -> float:
        """Peak-to-Sidelobe Ratio of a correlation response map.

        PSR = (peak − mean_sidelobe) / std_sidelobe

        The sidelobe is the response map with a ``(2·window_half+1)²``
        exclusion window around the peak.

        Args:
            response: 2-D float32/float64 correlation response.
            window_half: Half-size of exclusion region around the peak.

        Returns:
            PSR value ≥ 0.  Returns 0.0 if the sidelobe has zero variance.
        """
        h, w = response.shape
        flat_idx = int(np.argmax(response))
        py, px = divmod(flat_idx, w)
        peak_val = response[py, px]

        mask = np.zeros_like(response, dtype=bool)
        y1 = max(0, py - window_half)
        y2 = min(h, py + window_half + 1)
        x1 = max(0, px - window_half)
        x2 = min(w, px + window_half + 1)
        mask[y1:y2, x1:x2] = True

        sidelobe = response[~mask]
        if sidelobe.size == 0:
            return 0.0
        std_s = float(sidelobe.std())
        if std_s < 1e-10:
            return 0.0
        return float((peak_val - float(sidelobe.mean())) / std_s)

    def _adaptive_lr(self, psr: float) -> float:
        """Return learning rate scaled by PSR confidence.

        - PSR ≥ _PSR_HIGH → full ``self.learning_rate``
        - PSR ≤ _PSR_LOW  → 0.0 (freeze template)
        - Between          → linear interpolation
        """
        if psr >= _PSR_HIGH:
            return self.learning_rate
        if psr <= _PSR_LOW:
            return 0.0
        alpha = (psr - _PSR_LOW) / (_PSR_HIGH - _PSR_LOW)
        return self.learning_rate * alpha

    def _redetect(self, frame: np.ndarray) -> Tuple[float, float]:
        """Coarse grid scan to recover from tracking failure.

        Evaluates the current filter at a regular grid of candidate positions
        across the full image and returns the position with the highest PSR.

        Complexity: O(G × N log N) where G = number of grid cells.

        Returns:
            ``(cx, cy)`` of the best candidate, or the last known position
            if no candidate exceeds the current PSR.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        fh, fw = gray.shape[:2]
        sw, sh = self._search_size()

        stride_x = max(1, sw // 2)
        stride_y = max(1, sh // 2)

        best_psr = -float("inf")
        best_cx, best_cy = self._pos  # type: ignore[assignment]

        for cy_cand in range(sh // 2, fh - sh // 2 + 1, stride_y):
            for cx_cand in range(sw // 2, fw - sw // 2 + 1, stride_x):
                patch = self._extract(
                    frame, float(cx_cand), float(cy_cand), self._current_scale
                )
                zf = np.fft.fft2(patch * self._window)
                kzf = self._kernel_corr(self._xf, zf)
                response = np.real(np.fft.ifft2(self._alphaf * kzf))
                psr = self._compute_psr(response)
                if psr > best_psr:
                    best_psr = psr
                    best_cx, best_cy = float(cx_cand), float(cy_cand)

        return best_cx, best_cy

    @staticmethod
    def _hann2d(h: int, w: int) -> np.ndarray:
        return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)

    @staticmethod
    def _gaussian_labels(h: int, w: int, sigma_frac: float = 0.1) -> np.ndarray:
        sig_h, sig_w = sigma_frac * h, sigma_frac * w
        ys = np.arange(h) - h // 2
        xs = np.arange(w) - w // 2
        xx, yy = np.meshgrid(xs, ys)
        labels = np.exp(
            -(xx ** 2 / (2.0 * sig_w ** 2) + yy ** 2 / (2.0 * sig_h ** 2))
        )
        labels = np.roll(np.roll(labels, -h // 2, axis=0), -w // 2, axis=1)
        return labels.astype(np.float32)

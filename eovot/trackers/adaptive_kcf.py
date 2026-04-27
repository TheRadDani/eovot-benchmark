"""AdaptiveKCF — budget-aware correlation filter tracker.

Dynamically switches between a Gaussian (RBF) kernel and a linear kernel
depending on whether the measured per-frame latency is within a user-supplied
budget.  This makes the tracker self-regulating on constrained edge hardware:
it attempts the more discriminative Gaussian mode by default and falls back to
the faster linear mode only when the device cannot keep up.

Algorithm
---------
* Initialisation: always use the Gaussian kernel.
* Per-frame update:
  1. Time the update internally.
  2. Maintain an exponential moving average (EMA) of recent latencies.
  3. Switch to **fast mode** (linear kernel) when ``ema > 0.85 * budget``.
  4. Switch back to **accurate mode** (Gaussian kernel) when
     ``ema < 0.55 * budget``.
* The hysteresis band (0.55–0.85) prevents rapid oscillation near the budget
  boundary.

Both modes keep the same O(N log N) per-frame cost thanks to the circulant
structure of correlation filters; the difference is purely in the kernel
evaluation, which is ~2× cheaper in linear mode.

Reference
---------
Henriques et al., "High-Speed Tracking with Kernelized Correlation
Filters." IEEE TPAMI, 2015.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox

# Mode hysteresis thresholds (fraction of latency budget)
_SWITCH_TO_FAST = 0.85
_SWITCH_TO_ACCURATE = 0.55


class AdaptiveKCFTracker(BaseTracker):
    """Budget-aware KCF tracker that adapts its kernel to meet a latency target.

    In **accurate mode** (default) a Gaussian (RBF) kernel is used, matching
    the full KCF formulation.  When the internal latency EMA exceeds
    ``latency_budget_ms * 0.85``, the tracker drops to **fast mode** which
    uses a linear (dot-product) kernel — equivalent to a MOSSE-style filter
    but sharing the same search-window and label infrastructure.

    The transition back to accurate mode only happens when latency is well
    under budget (``< 0.55 * budget``), preventing rapid mode oscillation.

    Args:
        latency_budget_ms: Target per-frame latency in milliseconds.
            Default ``33.0`` targets ~30 FPS.
        learning_rate: EMA weight for online filter update, in (0, 1].
            Default: ``0.075``.
        lambda_: Ridge-regression regularisation term. Default: ``1e-4``.
        padding: Context padding as a fraction of target size on each side.
            Default: ``1.5``.
        kernel_sigma: Bandwidth of the Gaussian (RBF) kernel (accurate mode
            only). Default: ``0.5``.
        ema_alpha: Smoothing factor for the internal latency estimator.
            Higher values react faster but are noisier. Default: ``0.2``.

    Example::

        tracker = AdaptiveKCFTracker(latency_budget_ms=20.0)  # 50 FPS budget
        tracker.initialize(frame, bbox)
        for frame in frames:
            bbox = tracker.update(frame)
        stats = tracker.adaptation_stats
        print(f"Mode switches: {stats['mode_switches']}")
        print(f"Fast-mode frames: {stats['fast_mode_frames']}")
    """

    def __init__(
        self,
        latency_budget_ms: float = 33.0,
        learning_rate: float = 0.075,
        lambda_: float = 1e-4,
        padding: float = 1.5,
        kernel_sigma: float = 0.5,
        ema_alpha: float = 0.2,
    ) -> None:
        super().__init__(name="AdaptiveKCF")
        self.latency_budget_ms = latency_budget_ms
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.padding = padding
        self.kernel_sigma = kernel_sigma
        self.ema_alpha = ema_alpha

        self._mode: str = "accurate"
        self._latency_ema: float = 0.0
        self._mode_switches: int = 0
        self._latency_history: List[float] = []
        self._mode_history: List[str] = []

        self._pos: Optional[Tuple[float, float]] = None
        self._target_sz: Optional[Tuple[int, int]] = None
        self._search_sz: Optional[Tuple[int, int]] = None
        self._alphaf: Optional[np.ndarray] = None
        self._xf: Optional[np.ndarray] = None
        self._window: Optional[np.ndarray] = None
        self._yf: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        """Current operating mode: ``"accurate"`` or ``"fast"``."""
        return self._mode

    @property
    def mode_switches(self) -> int:
        """Total number of mode transitions since last :meth:`initialize`."""
        return self._mode_switches

    @property
    def adaptation_stats(self) -> dict:
        """Summary dict of adaptation behaviour for post-run analysis."""
        total = len(self._mode_history)
        fast_frames = self._mode_history.count("fast")
        return {
            "mode": self._mode,
            "latency_budget_ms": self.latency_budget_ms,
            "latency_ema_ms": round(self._latency_ema, 3),
            "mode_switches": self._mode_switches,
            "total_frames": total,
            "fast_mode_frames": fast_frames,
            "accurate_mode_frames": total - fast_frames,
            "fast_mode_pct": round(100.0 * fast_frames / total, 1) if total else 0.0,
        }

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the filter on the first frame using the Gaussian kernel.

        Args:
            frame: BGR image ``(H, W, 3)`` or grayscale ``(H, W)``.
            bbox: Initial bounding box ``(x, y, w, h)``.
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
        kf = self._gaussian_kernel(xf, xf)
        self._alphaf = self._yf / (kf + self.lambda_)
        self._xf = xf

        self._mode = "accurate"
        self._latency_ema = 0.0
        self._mode_switches = 0
        self._latency_history.clear()
        self._mode_history.clear()

    def update(self, frame: np.ndarray) -> BBox:
        """Track the target in a new frame, adapting the kernel to the budget.

        Args:
            frame: Current frame ``(H, W, 3)`` BGR or ``(H, W)`` grayscale.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._pos is None:
            raise RuntimeError(
                "AdaptiveKCFTracker is not initialised. Call initialize() first."
            )

        t0 = time.perf_counter()

        cx, cy = self._pos
        patch = self._extract(frame, cx, cy)
        zf = np.fft.fft2(patch * self._window)

        if self._mode == "accurate":
            kzf = self._gaussian_kernel(self._xf, zf)
        else:
            kzf = self._linear_kernel(self._xf, zf)

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

        new_patch = self._extract(frame, new_cx, new_cy)
        new_xf = np.fft.fft2(new_patch * self._window)
        if self._mode == "accurate":
            new_kf = self._gaussian_kernel(new_xf, new_xf)
        else:
            new_kf = self._linear_kernel(new_xf, new_xf)
        new_alphaf = self._yf / (new_kf + self.lambda_)

        lr = self.learning_rate
        self._xf = (1.0 - lr) * self._xf + lr * new_xf
        self._alphaf = (1.0 - lr) * self._alphaf + lr * new_alphaf

        elapsed_ms = (time.perf_counter() - t0) * 1_000.0
        self._adapt(elapsed_ms)

        tw, th = self._target_sz
        return (new_cx - tw / 2.0, new_cy - th / 2.0, float(tw), float(th))

    def reset(self) -> None:
        """Clear all internal state so the tracker can be re-initialised."""
        self._pos = None
        self._target_sz = None
        self._search_sz = None
        self._alphaf = None
        self._xf = None
        self._window = None
        self._yf = None
        self._mode = "accurate"
        self._latency_ema = 0.0
        self._mode_switches = 0
        self._latency_history.clear()
        self._mode_history.clear()

    # ------------------------------------------------------------------
    # Adaptation logic
    # ------------------------------------------------------------------

    def _adapt(self, elapsed_ms: float) -> None:
        """Update latency EMA and switch modes via hysteresis."""
        if self._latency_ema == 0.0:
            self._latency_ema = elapsed_ms
        else:
            a = self.ema_alpha
            self._latency_ema = (1.0 - a) * self._latency_ema + a * elapsed_ms

        self._latency_history.append(elapsed_ms)
        prev_mode = self._mode

        budget = self.latency_budget_ms
        if self._mode == "accurate" and self._latency_ema > budget * _SWITCH_TO_FAST:
            self._mode = "fast"
        elif self._mode == "fast" and self._latency_ema < budget * _SWITCH_TO_ACCURATE:
            self._mode = "accurate"

        if self._mode != prev_mode:
            self._mode_switches += 1

        self._mode_history.append(self._mode)

    # ------------------------------------------------------------------
    # Kernel implementations
    # ------------------------------------------------------------------

    def _gaussian_kernel(self, xf: np.ndarray, zf: np.ndarray) -> np.ndarray:
        """Gaussian (RBF) kernel correlation in the Fourier domain."""
        N = xf.shape[0] * xf.shape[1]
        xx = np.real(np.sum(xf * np.conj(xf))) / N
        zz = np.real(np.sum(zf * np.conj(zf))) / N
        cross = np.real(np.fft.ifft2(np.conj(xf) * zf)) / N
        exponent = np.maximum(0.0, xx + zz - 2.0 * cross) / (self.kernel_sigma ** 2)
        return np.fft.fft2(np.exp(-exponent))

    @staticmethod
    def _linear_kernel(xf: np.ndarray, zf: np.ndarray) -> np.ndarray:
        """Linear (dot-product) kernel in the Fourier domain.

        Equivalent to MOSSE's direct correlation but expressed in the same
        KCF framework, enabling a drop-in swap without restructuring the
        detection/update loop.
        """
        N = xf.shape[0] * xf.shape[1]
        return np.conj(xf) * zf / N

    # ------------------------------------------------------------------
    # Feature extraction helpers (shared with KCFTracker)
    # ------------------------------------------------------------------

    def _extract(self, frame: np.ndarray, cx: float, cy: float) -> np.ndarray:
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

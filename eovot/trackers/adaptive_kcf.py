"""Adaptive KCF tracker with Peak-to-Sidelobe Ratio (PSR) confidence gating.

Extends the baseline :class:`~eovot.trackers.kcf.KCFTracker` with a per-frame
confidence estimate derived from the filter response map.  When the response
map's PSR drops below a configurable threshold, the tracker considers itself
uncertain and freezes its internal state — returning the last reliable bounding
box rather than propagating a noisy, potentially drift-corrupted prediction.

**Why PSR matters on the edge**

Correlation-filter trackers like KCF are highly efficient but suffer silently
during occlusions, fast motion, or background clutter: the filter continues to
produce a prediction without any signal that the prediction is unreliable.  On
edge devices, downstream consumers (robot controllers, video-analytics pipelines,
alert systems) cannot afford to act on bad predictions.  PSR makes this failure
mode visible and controllable.

Peak-to-Sidelobe Ratio (PSR) definition::

    PSR = (peak_response − μ_sidelobe) / σ_sidelobe

where the *sidelobe* is the set of response-map pixels outside a square of
``(2 × psr_window + 1)`` pixels centred on the peak.  A PSR above ~7 typically
indicates reliable tracking; below ~5 the filter has likely lost the target.
The default threshold of 7.0 is conservative (prioritises stability).

Usage::

    from eovot.trackers.adaptive_kcf import AdaptiveKCFTracker

    tracker = AdaptiveKCFTracker(min_psr=7.0)
    tracker.initialize(frame0, bbox)

    for frame in frames[1:]:
        bbox = tracker.update(frame)
        print(f"PSR={tracker.last_psr:.2f}  confident={tracker.last_confident}")

    # Post-hoc confidence analysis
    from eovot.metrics.confidence import ConfidenceAnalyzer
    analyzer = ConfidenceAnalyzer()
    result = analyzer.analyze(tracker.psr_history, per_frame_ious)
    print(result)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import BBox
from .kcf import KCFTracker


class AdaptiveKCFTracker(KCFTracker):
    """KCF tracker with PSR-based confidence gating.

    Inherits all KCF parameters (learning_rate, lambda_, padding, kernel_sigma)
    and adds two confidence-specific hyperparameters.

    When a frame's PSR falls below ``min_psr``, the tracker:

    - **Skips the model update** — the internal correlation filter is not
      polluted with a bad appearance template.
    - **Returns the last reliable bounding box** — downstream consumers
      receive a stable, trusted position rather than a noisy estimate.
    - **Records a low-confidence flag** — accessible via :attr:`last_confident`
      and the full :attr:`psr_history` for off-line analysis.

    Args:
        min_psr: Minimum acceptable PSR to accept a frame's update and
            displacement estimate.  Frames below this value cause the
            tracker to freeze. Default: ``7.0``.
        psr_window: Half-size of the peak exclusion window for sidelobe
            computation.  Excludes a ``(2w+1) × (2w+1)`` square.
            Default: ``2`` (5 × 5 exclusion zone).
        learning_rate: EMA weight for online filter updates. Default: ``0.075``.
        lambda_: Ridge-regression regularisation term. Default: ``1e-4``.
        padding: Context padding as a fraction of target size. Default: ``1.5``.
        kernel_sigma: Bandwidth of the Gaussian (RBF) kernel. Default: ``0.5``.
    """

    def __init__(
        self,
        min_psr: float = 7.0,
        psr_window: int = 2,
        learning_rate: float = 0.075,
        lambda_: float = 1e-4,
        padding: float = 1.5,
        kernel_sigma: float = 0.5,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            lambda_=lambda_,
            padding=padding,
            kernel_sigma=kernel_sigma,
        )
        self.name = "AdaptiveKCF"
        self.min_psr = min_psr
        self.psr_window = psr_window

        self._last_psr: float = 0.0
        self._last_confident: bool = True
        self._last_reliable_bbox: Optional[BBox] = None
        self._psr_history: List[float] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def last_psr(self) -> float:
        """PSR of the response map computed on the most recently processed frame."""
        return self._last_psr

    @property
    def last_confident(self) -> bool:
        """``True`` if the last update was accepted (PSR ≥ ``min_psr``)."""
        return self._last_confident

    @property
    def psr_history(self) -> List[float]:
        """Per-frame PSR values recorded since :meth:`initialize` (read-only copy)."""
        return list(self._psr_history)

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the tracker and reset confidence history.

        Args:
            frame: First frame as a ``(H, W, 3)`` BGR or ``(H, W)`` grayscale array.
            bbox: Initial bounding box ``(x, y, w, h)`` in pixel coordinates.
        """
        super().initialize(frame, bbox)
        self._last_reliable_bbox = bbox
        self._last_psr = 0.0
        self._last_confident = True
        self._psr_history.clear()

    def update(self, frame: np.ndarray) -> BBox:
        """Track the target with confidence-gated filter updates.

        Computes the KCF response map, evaluates its PSR, and either:

        - **Accepts** the frame (PSR ≥ ``min_psr``): shifts the position,
          updates the appearance model, records a new reliable bbox.
        - **Rejects** the frame (PSR < ``min_psr``): freezes all state,
          returns the last reliable bbox unchanged.

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

        cx, cy = self._pos

        # Compute response map (identical path to KCFTracker.update, before peak shift).
        patch = self._extract(frame, cx, cy)
        zf = np.fft.fft2(patch * self._window)
        kzf = self._kernel_corr(self._xf, zf)
        response = np.real(np.fft.ifft2(self._alphaf * kzf))

        psr = self._compute_psr(response, self.psr_window)
        self._last_psr = psr
        self._psr_history.append(psr)

        if psr >= self.min_psr:
            # High-confidence: locate peak, shift position, update model.
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
            new_kf = self._kernel_corr(new_xf, new_xf)
            new_alphaf = self._yf / (new_kf + self.lambda_)

            lr = self.learning_rate
            self._xf = (1.0 - lr) * self._xf + lr * new_xf
            self._alphaf = (1.0 - lr) * self._alphaf + lr * new_alphaf

            tw, th = self._target_sz
            bbox: BBox = (
                new_cx - tw / 2.0,
                new_cy - th / 2.0,
                float(tw),
                float(th),
            )
            self._last_reliable_bbox = bbox
            self._last_confident = True
        else:
            # Low-confidence: freeze, return last known-good position.
            self._last_confident = False
            bbox = self._last_reliable_bbox  # type: ignore[assignment]

        return bbox

    def reset(self) -> None:
        """Reset all internal state so the tracker can be re-initialised."""
        super().reset()
        self._last_reliable_bbox = None
        self._last_psr = 0.0
        self._last_confident = True
        self._psr_history.clear()

    # ------------------------------------------------------------------
    # PSR computation
    # ------------------------------------------------------------------

    def _kernel_corr(self, xf: np.ndarray, zf: np.ndarray) -> np.ndarray:
        """Corrected Gaussian kernel correlation in the Fourier domain.

        The base :class:`~eovot.trackers.kcf.KCFTracker` divides ``cross``
        by N but does not apply the matching N normalisation to the exponent,
        producing exponent values ~10 000× too large and an effectively
        all-zero response map.  This override uses the reference formula from
        Henriques et al. (MATLAB code)::

            k(δ) = exp(−(||x||² + ||z||² − 2⟨x, z_δ⟩) / (N · σ²))

        which yields properly scaled exponents and a discriminative response
        map suitable for PSR analysis.

        Args:
            xf: DFT of the template patch.
            zf: DFT of the candidate patch.

        Returns:
            DFT of the Gaussian kernel response map with well-scaled values.
        """
        N = xf.shape[0] * xf.shape[1]
        xx = np.real(np.sum(xf * np.conj(xf))) / N
        zz = np.real(np.sum(zf * np.conj(zf))) / N
        # IFFT gives inner product ⟨x, z_δ⟩ at each cyclic shift δ.
        cross = np.real(np.fft.ifft2(np.conj(xf) * zf))
        exponent = np.maximum(0.0, xx + zz - 2.0 * cross) / (N * self.kernel_sigma ** 2)
        return np.fft.fft2(np.exp(-exponent))

    @staticmethod
    def _compute_psr(response: np.ndarray, window: int) -> float:
        """Peak-to-Sidelobe Ratio of a 2-D filter response map.

        Args:
            response: 2-D float array (the inverse-FFT output).
            window: Half-size of the peak exclusion zone; all pixels inside
                the ``(2w+1) × (2w+1)`` square centred on the peak are
                excluded from the sidelobe statistics.

        Returns:
            PSR scalar ≥ 0. Returns ``0.0`` when the sidelobe region is empty
            or has zero standard deviation.
        """
        h, w = response.shape
        peak_idx = int(np.argmax(response))
        py, px = divmod(peak_idx, w)

        mask = np.ones((h, w), dtype=bool)
        y1 = max(0, py - window)
        y2 = min(h, py + window + 1)
        x1 = max(0, px - window)
        x2 = min(w, px + window + 1)
        mask[y1:y2, x1:x2] = False

        sidelobe = response[mask]
        if len(sidelobe) == 0:
            return 0.0
        mu = float(sidelobe.mean())
        sigma = float(sidelobe.std())
        if sigma < 1e-8:
            return 0.0
        return float((float(response[py, px]) - mu) / sigma)

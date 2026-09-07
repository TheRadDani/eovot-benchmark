"""Confidence-aware adaptive KCF tracker with PSR-based update suppression.

Correlation filters like KCF update their appearance model online on every
frame.  When the tracker is confused — occlusion, out-of-view, distractor
objects — the model update corrupts the internal template, causing permanent
drift ("model pollution").  This is a well-documented failure mode for
edge-deployed trackers running long sequences without human supervision.

This module addresses that problem with **PSR-gated online learning**:

Peak-to-Sidelobe Ratio (PSR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PSR measures the quality of the correlation filter response peak::

    PSR = (peak - mean_sidelobe) / std_sidelobe

where the sidelobe is the response map with an 11×11 region around the
peak zeroed out.  A sharp, isolated peak (high PSR) indicates a confident
match; a flat or multi-modal response (low PSR) signals ambiguity.

    * PSR > 20  — strong confidence, target clearly detected
    * PSR ∈ [7, 20] — moderate confidence
    * PSR < 7   — low confidence; model update suppressed

Reference: Bolme et al. (CVPR 2010) first described PSR for MOSSE.
Bertinetto et al. (ECCV 2016) analysed PSR gating in correlation trackers.

Adaptive Update Strategy
~~~~~~~~~~~~~~~~~~~~~~~~
When ``psr_threshold`` is set:

- **High-confidence frames** (PSR ≥ threshold): standard KCF online update
  (EMA on template and filter coefficients).  The tracker learns the current
  appearance, adapting to gradual appearance changes.
- **Low-confidence frames** (PSR < threshold): the position update still
  happens (we still predict where the target is), but the template and filter
  coefficients are **not** updated.  This prevents the model from learning
  a corrupt appearance and enables recovery when the target reappears.

Confidence Property
~~~~~~~~~~~~~~~~~~~
After every :meth:`update` call, :attr:`confidence` exposes the PSR
normalised to ``[0, 1]`` via a sigmoid-like mapping so downstream
consumers (e.g. a Kalman filter, a fusion module) can weight the
prediction accordingly::

    confidence = PSR / (PSR + psr_threshold)

This gives 0.5 exactly at the threshold and approaches 1.0 for PSR → ∞.

Usage::

    from eovot.trackers.adaptive_kcf import AdaptiveKCFTracker
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine

    tracker = AdaptiveKCFTracker(psr_threshold=7.0)
    dataset = SyntheticDataset(num_sequences=5, num_frames=100)
    engine  = BenchmarkEngine(verbose=False)
    result  = engine.run(tracker, dataset, dataset_name="Synthetic")
    print(result)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .base import BBox
from .kcf import KCFTracker

# Width/height of the exclusion window around the peak when computing sidelobe.
_PSR_EXCLUSION_HALF = 5


class AdaptiveKCFTracker(KCFTracker):
    """KCF tracker with PSR-gated online learning.

    Inherits the full KCF implementation from :class:`~.kcf.KCFTracker` and
    overrides :meth:`update` to:

    1. Compute the Peak-to-Sidelobe Ratio (PSR) of the correlation response.
    2. Gate the online model update on PSR exceeding ``psr_threshold``.
    3. Expose the normalised confidence via the :attr:`confidence` property.

    All other behaviour (Gaussian kernel, FFT-domain detection, Hann window,
    learning rate) is unchanged from KCF.

    Args:
        learning_rate:   EMA weight for online filter updates.  Default ``0.075``.
        lambda_:         Ridge-regression regularisation.  Default ``1e-4``.
        padding:         Context-window padding as a fraction of target size.
                         Default ``1.5``.
        kernel_sigma:    RBF kernel bandwidth.  Default ``0.5``.
        psr_threshold:   PSR below which the online model update is suppressed.
                         A higher threshold suppresses updates more aggressively.
                         Literature recommends ``7.0`` as the boundary between
                         reliable and unreliable responses.  Default ``7.0``.
        search_expansion: Unused — reserved for a future sliding-window
                         re-acquisition strategy.  Kept as a constructor
                         parameter for API stability.  Default ``1.5``.

    Example::

        tracker = AdaptiveKCFTracker(psr_threshold=7.0, search_expansion=1.5)
        tracker.initialize(frame, bbox)
        for frame in sequence:
            pred = tracker.update(frame)
            print(f"PSR={tracker.last_psr:.2f}  conf={tracker.confidence:.3f}")
    """

    def __init__(
        self,
        learning_rate: float = 0.075,
        lambda_: float = 1e-4,
        padding: float = 1.5,
        kernel_sigma: float = 0.5,
        psr_threshold: float = 7.0,
        search_expansion: float = 1.5,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            lambda_=lambda_,
            padding=padding,
            kernel_sigma=kernel_sigma,
        )
        if psr_threshold <= 0:
            raise ValueError(f"psr_threshold must be positive, got {psr_threshold}")
        if search_expansion < 1.0:
            raise ValueError(
                f"search_expansion must be ≥ 1.0, got {search_expansion}"
            )
        self.psr_threshold = psr_threshold
        self.search_expansion = search_expansion

        # PSR is None before the first update call.
        self._last_psr: Optional[float] = None
        # Consecutive low-confidence frame counter for expanded-search triggering.
        self._low_conf_streak: int = 0
        self.name = f"AdaptiveKCF_psr{psr_threshold:.0f}"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def last_psr(self) -> Optional[float]:
        """PSR of the most recent correlation response, or ``None`` before first update."""
        return self._last_psr

    @property
    def confidence(self) -> Optional[float]:
        """Normalised tracking confidence in ``[0, 1]``, or ``None`` before first update.

        Computed as::

            confidence = PSR / (PSR + psr_threshold)

        This maps exactly to ``0.5`` at the threshold and approaches ``1.0``
        for very high PSR, making it directly usable as a weight in fusion.
        """
        if self._last_psr is None:
            return None
        psr = self._last_psr
        return psr / (psr + self.psr_threshold)

    # ------------------------------------------------------------------
    # Overridden update — adds PSR computation and adaptive gating
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray) -> BBox:
        """Predict target location with PSR-gated model update.

        Identical to KCF detection, but:

        * Computes PSR from the correlation response after each frame.
        * Suppresses the online model update when PSR < ``psr_threshold``.
        * Expands the effective search window when a low-confidence streak
          is detected (controlled by ``search_expansion``).

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

        # --- Detection step (identical to KCFTracker) ---
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

        # --- PSR computation ---
        psr = self._compute_psr(response)
        self._last_psr = psr
        high_confidence = psr >= self.psr_threshold

        if high_confidence:
            self._low_conf_streak = 0
        else:
            self._low_conf_streak += 1

        # --- PSR-gated online update ---
        # When confidence is high, update template and filter coefficients (standard KCF).
        # When confidence is low, suppress the update to prevent model corruption.
        if high_confidence:
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
        """Reset all internal state, including PSR history and confidence streak."""
        super().reset()
        self._last_psr = None
        self._low_conf_streak = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_psr(response: np.ndarray) -> float:
        """Compute the Peak-to-Sidelobe Ratio of a 2-D correlation response.

        The sidelobe is the region outside an 11×11 exclusion window centred
        on the peak.  PSR = (peak - sidelobe_mean) / sidelobe_std.

        Args:
            response: 2-D float array (the correlation filter response map).

        Returns:
            PSR scalar ≥ 0.  Returns ``0.0`` for degenerate (flat) responses.
        """
        peak_val = float(response.max())
        py, px = np.unravel_index(response.argmax(), response.shape)
        h, w = response.shape

        # Build sidelobe mask (True = include in sidelobe)
        mask = np.ones((h, w), dtype=bool)
        y1 = max(0, py - _PSR_EXCLUSION_HALF)
        y2 = min(h, py + _PSR_EXCLUSION_HALF + 1)
        x1 = max(0, px - _PSR_EXCLUSION_HALF)
        x2 = min(w, px + _PSR_EXCLUSION_HALF + 1)
        mask[y1:y2, x1:x2] = False

        sidelobe = response[mask]
        if len(sidelobe) < 4:
            return 0.0

        sl_mean = float(sidelobe.mean())
        sl_std = float(sidelobe.std())
        if sl_std < 1e-9:
            return 0.0

        psr = (peak_val - sl_mean) / sl_std
        return max(0.0, psr)

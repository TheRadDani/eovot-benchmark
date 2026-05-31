"""Scale-adaptive MOSSE correlation filter tracker.

Extends the base MOSSE tracker with a multi-scale search pyramid that lets
the filter adapt to target size changes over time.  At each frame, correlation
responses are computed at S discrete scale levels; the scale yielding the
highest Peak-to-Sidelobe Ratio (PSR) is selected and used for the online
filter update.

Scale-change is one of the hardest tracking challenges in OTB, LaSOT, and
GOT-10k.  Plain MOSSE uses a fixed-size template and fails when a target
grows or shrinks significantly.  This tracker closes that gap with only an
O(S·N log N) cost increase per frame — still fully viable on edge hardware.

Algorithm
---------
1. **Initialization** — train a standard MOSSE filter on the initial patch
   at canonical template size ``(tw, th)``.
2. **Detection (per frame)**:
   a. Compute scale levels:
      ``s_k = scale_step^k  for  k ∈ {-⌊n//2⌋, …, 0, …, ⌊n//2⌋}``
   b. For each scale ``s_k``, extract a patch of size
      ``(round(tw·s_k), round(th·s_k))`` and **resize to ``(tw, th)``** before
      computing the DFT.  This preserves the filter dimensions.
   c. Compute the MOSSE response map for each resized patch.
   d. Evaluate PSR for each response; keep the scale with the highest PSR.
3. **Online update** — update the MOSSE filter using the patch extracted at
   the winning scale (after converting the scale offset to a smooth EMA on
   the running template size).

PSR definition::

    PSR = (peak_val − μ_sidelobe) / σ_sidelobe

where the sidelobe is the response map with an ``(11×11)`` window around the
peak masked out.  Values above ~7 typically indicate confident detections.

References
----------
Bolme et al. "Visual Object Tracking using Adaptive Correlation Filters."
    IEEE CVPR 2010.

Danelljan et al. "Accurate Scale Estimation for Robust Visual Tracking."
    BMVC 2014  (motivation for scale pyramid in correlation filters).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox


class ScaleAdaptiveMOSSETracker(BaseTracker):
    """MOSSE tracker with scale-pyramid adaptation.

    At each frame the tracker evaluates correlation responses at ``n_scales``
    discrete scale levels and selects the scale that yields the highest
    Peak-to-Sidelobe Ratio (PSR).  The running template size is updated with
    an exponential moving average so scale transitions are smooth rather than
    abrupt.

    Args:
        learning_rate: EMA weight for online MOSSE filter update.
            Default ``0.125`` (matches original paper).
        sigma: Standard deviation of the Gaussian regression target in pixels.
            Default ``2.0``.
        n_scales: Number of scale levels in the pyramid.  Must be a positive
            odd integer so the current scale (factor 1.0) is always included.
            Default ``5``.
        scale_step: Multiplicative gap between adjacent scale levels.
            With ``scale_step=1.05`` and ``n_scales=5`` the tested factors are
            ``[0.90, 0.95, 1.00, 1.05, 1.10]``.  Default ``1.05``.
        scale_lr: Exponential moving-average weight for updating the running
            template size after a scale change is detected.  Lower values
            give smoother (but slower) adaptation.  Default ``0.35``.
        psr_threshold: Minimum PSR below which a frame's detection is treated
            as unreliable.  The tracker keeps its previous prediction when PSR
            is below this threshold.  Set to ``0`` to disable.
            Default ``6.0``.

    Example::

        from eovot.trackers.scale_adaptive_mosse import ScaleAdaptiveMOSSETracker
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.benchmark.engine import BenchmarkEngine

        tracker = ScaleAdaptiveMOSSETracker(n_scales=5, scale_step=1.05)
        dataset = SyntheticDataset(num_sequences=3, motion="linear")
        engine  = BenchmarkEngine(verbose=False)
        result  = engine.run(tracker, dataset, dataset_name="Synthetic")
        print(result)
    """

    # PSR computation: mask a window of this half-size around the peak.
    _PSR_MASK_HALFSIZE: int = 5

    def __init__(
        self,
        learning_rate: float = 0.125,
        sigma: float = 2.0,
        n_scales: int = 5,
        scale_step: float = 1.05,
        scale_lr: float = 0.35,
        psr_threshold: float = 6.0,
    ) -> None:
        super().__init__(name="ScaleAdaptiveMOSSE")
        if n_scales < 1 or n_scales % 2 == 0:
            raise ValueError(
                f"n_scales must be a positive odd integer, got {n_scales}."
            )
        if scale_step <= 1.0:
            raise ValueError(
                f"scale_step must be > 1.0, got {scale_step}."
            )
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.n_scales = n_scales
        self.scale_step = scale_step
        self.scale_lr = scale_lr
        self.psr_threshold = psr_threshold

        # Pre-compute scale factors: symmetric around 1.0.
        half = n_scales // 2
        self._scale_factors: List[float] = [
            scale_step ** k for k in range(-half, half + 1)
        ]

        # Internal state — populated by initialize().
        self._H_conj: Optional[np.ndarray] = None
        self._window: Optional[np.ndarray] = None
        self._yf: Optional[np.ndarray] = None
        self._pos: Optional[Tuple[float, float]] = None
        self._target_w: float = 0.0
        self._target_h: float = 0.0
        self._template_w: int = 0
        self._template_h: int = 0

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the scale-adaptive MOSSE filter on the first frame.

        Args:
            frame: BGR or grayscale image, shape ``(H, W[, C])``.
            bbox: Ground-truth box ``(x, y, w, h)`` in pixel coordinates.
        """
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid bbox {bbox}: w and h must be positive.")

        cx, cy = x + w / 2.0, y + h / 2.0
        self._pos = (cx, cy)
        self._target_w = float(w)
        self._target_h = float(h)
        self._template_w = max(1, int(round(w)))
        self._template_h = max(1, int(round(h)))

        tw, th = self._template_w, self._template_h
        self._window = np.outer(np.hanning(th), np.hanning(tw)).astype(np.float32)
        G = np.fft.fft2(self._gaussian_response(th, tw))
        self._yf = G

        gray = self._to_gray(frame)
        patch = self._extract_and_resize(gray, cx, cy, tw, th)
        Fi = np.fft.fft2(self._preprocess(patch))
        self._H_conj = (G * np.conj(Fi)) / (Fi * np.conj(Fi) + 1e-5)

    def update(self, frame: np.ndarray) -> BBox:
        """Predict and track the target with scale adaptation.

        Args:
            frame: Current BGR or grayscale frame, shape ``(H, W[, C])``.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._H_conj is None or self._pos is None:
            raise RuntimeError(
                "ScaleAdaptiveMOSSETracker not initialised. Call initialize() first."
            )

        cx, cy = self._pos
        gray = self._to_gray(frame)
        tw, th = self._template_w, self._template_h

        # ------------------------------------------------------------------
        # 1. Multi-scale correlation responses
        # ------------------------------------------------------------------
        best_psr = -1.0
        best_dx = 0.0
        best_dy = 0.0
        best_scale_idx = self.n_scales // 2  # default: current scale (factor 1.0)

        for k, sf in enumerate(self._scale_factors):
            scaled_w = max(1, int(round(self._target_w * sf)))
            scaled_h = max(1, int(round(self._target_h * sf)))
            patch = self._extract_and_resize(gray, cx, cy, scaled_w, scaled_h, out_w=tw, out_h=th)
            Zf = np.fft.fft2(self._preprocess(patch))
            response = np.real(np.fft.ifft2(self._H_conj * Zf))
            psr, peak_y, peak_x = self._compute_psr(response)
            if psr > best_psr:
                best_psr = psr
                best_scale_idx = k
                dy = float(peak_y) if peak_y <= th // 2 else float(peak_y - th)
                dx = float(peak_x) if peak_x <= tw // 2 else float(peak_x - tw)
                best_dy, best_dx = dy, dx

        # ------------------------------------------------------------------
        # 2. Accept or reject the detection based on PSR confidence
        # ------------------------------------------------------------------
        if best_psr >= self.psr_threshold:
            new_cx = cx + best_dx
            new_cy = cy + best_dy
            self._pos = (new_cx, new_cy)

            # Smooth scale update via EMA.
            winning_sf = self._scale_factors[best_scale_idx]
            self._target_w = (
                (1.0 - self.scale_lr) * self._target_w
                + self.scale_lr * self._target_w * winning_sf
            )
            self._target_h = (
                (1.0 - self.scale_lr) * self._target_h
                + self.scale_lr * self._target_h * winning_sf
            )
        else:
            # Low confidence: hold position, no scale update.
            new_cx, new_cy = cx, cy

        # ------------------------------------------------------------------
        # 3. Online MOSSE filter update at the (possibly updated) scale
        # ------------------------------------------------------------------
        new_patch = self._extract_and_resize(
            gray, new_cx, new_cy,
            max(1, int(round(self._target_w))),
            max(1, int(round(self._target_h))),
            out_w=tw, out_h=th,
        )
        Fi_new = np.fft.fft2(self._preprocess(new_patch))
        H_conj_new = (self._yf * np.conj(Fi_new)) / (Fi_new * np.conj(Fi_new) + 1e-5)
        lr = self.learning_rate
        self._H_conj = (1.0 - lr) * self._H_conj + lr * H_conj_new

        # Return bounding box centred at the updated position.
        return (
            new_cx - self._target_w / 2.0,
            new_cy - self._target_h / 2.0,
            self._target_w,
            self._target_h,
        )

    def reset(self) -> None:
        """Clear all internal state for re-initialisation."""
        self._H_conj = None
        self._window = None
        self._yf = None
        self._pos = None
        self._target_w = 0.0
        self._target_h = 0.0
        self._template_w = 0
        self._template_h = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_psr(
        self, response: np.ndarray
    ) -> Tuple[float, int, int]:
        """Compute Peak-to-Sidelobe Ratio and peak location.

        The sidelobe region is the response map with a rectangular mask of
        half-size ``_PSR_MASK_HALFSIZE`` around the peak zeroed out.

        Args:
            response: Real-valued correlation response map, shape ``(H, W)``.

        Returns:
            ``(psr, peak_y, peak_x)`` — PSR scalar and integer peak indices.
        """
        h, w = response.shape
        peak_idx = int(response.argmax())
        peak_y, peak_x = divmod(peak_idx, w)
        peak_val = float(response.flat[peak_idx])

        # Build sidelobe mask — exclude the peak neighbourhood.
        mask = np.ones_like(response, dtype=bool)
        hs = self._PSR_MASK_HALFSIZE
        y1 = max(0, peak_y - hs)
        y2 = min(h, peak_y + hs + 1)
        x1 = max(0, peak_x - hs)
        x2 = min(w, peak_x + hs + 1)
        mask[y1:y2, x1:x2] = False

        sidelobe = response[mask]
        if sidelobe.size == 0:
            return 0.0, peak_y, peak_x

        mu = float(sidelobe.mean())
        std = float(sidelobe.std())
        psr = (peak_val - mu) / (std + 1e-6)
        return psr, peak_y, peak_x

    def _extract_and_resize(
        self,
        gray: np.ndarray,
        cx: float,
        cy: float,
        src_w: int,
        src_h: int,
        out_w: Optional[int] = None,
        out_h: Optional[int] = None,
    ) -> np.ndarray:
        """Extract a patch of size ``(src_h, src_w)`` centred at ``(cx, cy)``.

        If ``out_w`` / ``out_h`` differ from ``src_w`` / ``src_h`` the patch
        is resized to ``(out_h, out_w)`` using bilinear interpolation.  This
        is the key operation that maps scale-pyramid patches back to the
        canonical filter dimensions.

        Edge-padding (``mode="edge"``) prevents black borders for patches that
        extend beyond frame boundaries.
        """
        fh, fw = gray.shape[:2]
        x1 = int(round(cx - src_w / 2.0))
        y1 = int(round(cy - src_h / 2.0))
        x2 = x1 + src_w
        y2 = y1 + src_h

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

        target_w = out_w if out_w is not None else src_w
        target_h = out_h if out_h is not None else src_h
        if patch.shape != (target_h, target_w):
            patch = cv2.resize(
                patch, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )
        return patch

    def _preprocess(self, patch: np.ndarray) -> np.ndarray:
        """Log-normalise and apply the Hann window."""
        f = np.log1p(patch.astype(np.float32))
        f = (f - f.mean()) / (f.std() + 1e-5)
        return f * self._window

    def _gaussian_response(self, h: int, w: int) -> np.ndarray:
        """2-D Gaussian centred at ``(h//2, w//2)`` with std = ``sigma``."""
        cy, cx = h // 2, w // 2
        ys, xs = np.ogrid[:h, :w]
        g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * self.sigma ** 2))
        return (g / (g.sum() + 1e-6)).astype(np.float32)

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        """Convert BGR / BGRA image to single-channel grayscale."""
        if frame.ndim == 2:
            return frame
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

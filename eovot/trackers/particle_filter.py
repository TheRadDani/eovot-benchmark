"""Particle filter (Sequential Monte Carlo) visual tracker for EOVOT.

The SIR (Sampling Importance Resampling) filter decomposes visual tracking
into three steps repeated each frame:

1. **Propagate** — perturb each particle's state with Gaussian noise drawn
   from a constant-velocity motion prior.
2. **Weigh** — score each candidate patch against the reference colour
   histogram using the Bhattacharyya similarity coefficient.
3. **Resample** — when the Effective Sample Size (ESS) falls below
   ``resample_threshold × N``, draw a fresh set of N particles with
   replacement proportional to their weights, preventing filter degeneracy.

The tracker also maintains an exponentially weighted moving average of the
colour template so it adapts gradually to appearance change (illumination
shift, partial occlusion, deformation).

Suitable for edge deployment: pure NumPy + OpenCV, no GPU or pre-trained
model files required.  Expected throughput: ~30–80 FPS with 200 particles on
a modern CPU core, scaling linearly with ``n_particles``.

Reference
---------
Isard & Blake (1998).  CONDENSATION — Conditional Density Propagation for
Visual Tracking.  International Journal of Computer Vision, 29(1), 5–28.

Example::

    from eovot.trackers.particle_filter import ParticleFilterTracker
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset

    tracker = ParticleFilterTracker(n_particles=300, motion_noise_px=8.0)
    dataset = SyntheticDataset(num_sequences=5, num_frames=150, motion="random")
    engine  = BenchmarkEngine(verbose=False)
    result  = engine.run(tracker, dataset, dataset_name="Synthetic-Random")
    print(result)
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .base import BaseTracker, BBox


class ParticleFilterTracker(BaseTracker):
    """SIR particle filter visual tracker using HSV colour histogram likelihood.

    Each particle represents a hypothesis about the target's bounding box
    ``(cx, cy, w, h)``.  The likelihood of each hypothesis is measured by
    the Bhattacharyya coefficient between the particle's local colour
    histogram and a reference histogram built at initialisation.

    Args:
        n_particles: Number of Monte Carlo samples.  More particles give
            better state-space coverage at the cost of higher compute.
            Default: ``200``.
        motion_noise_px: Standard deviation (pixels) of the Gaussian
            random-walk applied to the ``cx`` and ``cy`` components each
            frame.  Increase for fast-moving targets. Default: ``10.0``.
        scale_noise: Standard deviation of multiplicative scale noise
            applied to ``w`` and ``h`` each frame (fraction of current
            size, e.g. 0.02 = 2 % per frame). Default: ``0.02``.
        n_bins: Histogram bins per HSV channel.  Finer bins are more
            discriminative but sensitive to illumination noise. Default: ``16``.
        resample_threshold: Resampling is triggered when
            ``ESS / n_particles < resample_threshold``.  Lower values delay
            resampling (preserving diversity); higher values are more
            aggressive. Default: ``0.5``.
        update_rate: EMA weight for the online template update each frame.
            Set to ``0.0`` to freeze the template after initialisation.
            Default: ``0.05``.
        seed: Optional integer seed for the internal NumPy random generator.
            Fixes randomness for reproducible benchmarking. Default: ``None``.
    """

    def __init__(
        self,
        n_particles: int = 200,
        motion_noise_px: float = 10.0,
        scale_noise: float = 0.02,
        n_bins: int = 16,
        resample_threshold: float = 0.5,
        update_rate: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(name="ParticleFilter")
        self.n_particles = n_particles
        self.motion_noise_px = motion_noise_px
        self.scale_noise = scale_noise
        self.n_bins = n_bins
        self.resample_threshold = resample_threshold
        self.update_rate = update_rate
        self.seed = seed

        # Internal state — None before initialise() is called.
        self._particles: Optional[np.ndarray] = None   # (N, 4): [cx, cy, w, h]
        self._weights: Optional[np.ndarray] = None     # (N,) normalised
        self._template_hist: Optional[np.ndarray] = None
        self._rng: Optional[np.random.Generator] = None

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise particles around the ground-truth bounding box.

        Args:
            frame: First frame as a ``(H, W, 3)`` BGR uint8 array.
            bbox:  Ground-truth box ``(x, y, w, h)`` in pixel coordinates.
        """
        x, y, w, h = (float(v) for v in bbox)
        cx, cy = x + w / 2.0, y + h / 2.0

        self._rng = np.random.default_rng(self.seed)

        # Scatter particles around the initial centre with small noise.
        init_noise_xy = self._rng.normal(
            0.0, self.motion_noise_px / 4.0, (self.n_particles, 2)
        )
        init_noise_wh = self._rng.normal(
            0.0, self.scale_noise * max(w, h, 1.0), (self.n_particles, 2)
        )
        fh, fw = frame.shape[:2]
        self._particles = np.column_stack([
            cx + init_noise_xy[:, 0],
            cy + init_noise_xy[:, 1],
            np.clip(w + init_noise_wh[:, 0], 4.0, float(fw)),
            np.clip(h + init_noise_wh[:, 1], 4.0, float(fh)),
        ])
        self._weights = np.full(self.n_particles, 1.0 / self.n_particles)

        # Reference histogram from the target patch at frame 0.
        self._template_hist = self._colour_hist(frame, cx, cy, w, h)

    def update(self, frame: np.ndarray) -> BBox:
        """Predict target location in the next frame via SIR filtering.

        Args:
            frame: Current frame as a ``(H, W, 3)`` BGR uint8 array.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._particles is None or self._rng is None:
            raise RuntimeError(
                "ParticleFilterTracker is not initialised. Call initialize() first."
            )

        fh, fw = frame.shape[:2]
        n = self.n_particles

        # ------ Step 1: Propagate (motion prior — Gaussian random walk) ------
        self._particles[:, 0] += self._rng.normal(0.0, self.motion_noise_px, n)
        self._particles[:, 1] += self._rng.normal(0.0, self.motion_noise_px, n)
        # Multiplicative scale jitter keeps particles at plausible sizes.
        scale = self._rng.normal(1.0, self.scale_noise, (n, 2))
        self._particles[:, 2:4] *= scale
        # Clamp to frame bounds.
        self._particles[:, 0] = np.clip(self._particles[:, 0], 0.0, float(fw))
        self._particles[:, 1] = np.clip(self._particles[:, 1], 0.0, float(fh))
        self._particles[:, 2] = np.clip(self._particles[:, 2], 4.0, float(fw))
        self._particles[:, 3] = np.clip(self._particles[:, 3], 4.0, float(fh))

        # ------ Step 2: Weigh (likelihood = Bhattacharyya similarity) ------
        raw = np.empty(n, dtype=np.float64)
        for i in range(n):
            cx, cy, pw, ph = self._particles[i]
            hist = self._colour_hist(frame, cx, cy, pw, ph)
            raw[i] = _bhattacharyya(self._template_hist, hist)
        raw += 1e-12  # guard against all-zero degeneracy
        self._weights = raw / raw.sum()

        # ------ Step 3: Resample (systematic, when ESS is low) ------
        ess = 1.0 / float(np.sum(self._weights ** 2))
        if ess < self.resample_threshold * n:
            indices = _systematic_resample(self._weights, self._rng)
            self._particles = self._particles[indices]
            self._weights = np.full(n, 1.0 / n)

        # ------ Step 4: State estimate (weighted mean) ------
        cx_est = float(np.dot(self._weights, self._particles[:, 0]))
        cy_est = float(np.dot(self._weights, self._particles[:, 1]))
        w_est = float(np.dot(self._weights, self._particles[:, 2]))
        h_est = float(np.dot(self._weights, self._particles[:, 3]))

        # ------ Step 5: Online template adaptation (EMA) ------
        if self.update_rate > 0.0:
            new_hist = self._colour_hist(frame, cx_est, cy_est, w_est, h_est)
            lr = self.update_rate
            blended = (1.0 - lr) * self._template_hist + lr * new_hist
            total = blended.sum()
            self._template_hist = blended / total if total > 0 else blended

        return (cx_est - w_est / 2.0, cy_est - h_est / 2.0, w_est, h_est)

    def reset(self) -> None:
        """Reset all internal state so the tracker can be re-initialised."""
        self._particles = None
        self._weights = None
        self._template_hist = None
        self._rng = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _colour_hist(
        self,
        frame: np.ndarray,
        cx: float,
        cy: float,
        w: float,
        h: float,
    ) -> np.ndarray:
        """Extract a normalised HSV colour histogram from a patch.

        Converts the crop to HSV and concatenates per-channel histograms.
        Returns a flat float64 array of length ``3 × n_bins``.

        If the crop degenerates to an empty region (zero area), returns a
        uniform histogram so downstream weights are finite.
        """
        fh, fw = frame.shape[:2]
        x1 = max(0, int(cx - w / 2.0))
        y1 = max(0, int(cy - h / 2.0))
        x2 = min(fw, int(cx + w / 2.0) + 1)
        y2 = min(fh, int(cy + h / 2.0) + 1)

        if x2 <= x1 or y2 <= y1:
            size = self.n_bins * 3
            return np.full(size, 1.0 / size)

        patch = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

        hists = []
        ranges = [(0, 180), (0, 256), (0, 256)]  # H, S, V ranges
        for ch, (lo, hi) in enumerate(ranges):
            h_arr = cv2.calcHist([hsv], [ch], None, [self.n_bins], [lo, hi])
            hists.append(h_arr.flatten())

        combined = np.concatenate(hists).astype(np.float64)
        total = combined.sum()
        return combined / total if total > 0 else combined + 1e-12


# ------------------------------------------------------------------
# Module-level helpers (pure functions, no class state)
# ------------------------------------------------------------------

def _bhattacharyya(p: np.ndarray, q: np.ndarray) -> float:
    """Bhattacharyya similarity coefficient between two normalised histograms.

    Returns a value in ``[0, 1]``, where 1 indicates identical distributions.
    Both arrays must be non-negative and sum to 1 (or close to 1).

    Args:
        p: Reference histogram (flat float64 array).
        q: Candidate histogram (same shape as ``p``).
    """
    return float(np.sqrt(p * q).sum())


def _systematic_resample(
    weights: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Low-variance (systematic) resampling.

    Draws a single random offset in ``[0, 1/N)`` and places N equally-spaced
    strata on the cumulative weight axis.  Compared to multinomial resampling,
    systematic resampling has lower variance and preserves more diversity for
    the same particle count.

    Args:
        weights: Normalised importance weights, shape ``(N,)``.
        rng:     NumPy random generator for reproducibility.

    Returns:
        Integer index array of shape ``(N,)`` selecting the resampled particles.
    """
    n = len(weights)
    cumsum = np.cumsum(weights)
    positions = (rng.uniform(0.0, 1.0) + np.arange(n)) / n
    return np.searchsorted(cumsum, positions)

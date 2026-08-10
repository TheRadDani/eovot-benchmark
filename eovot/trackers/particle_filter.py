"""Particle Filter Tracker for robust probabilistic visual object tracking.

Implements Sequential Monte Carlo (SMC) tracking using:

* **Motion model**: Gaussian random walk on the target centre ``(cx, cy)``.
* **Appearance model**: Normalised HSV color histogram with Bhattacharyya
  coefficient as the likelihood function.
* **Resampling**: Systematic resampling triggered when the Effective Sample
  Size (ESS) drops below ``num_particles * ess_threshold_ratio``.
* **Template update**: Exponential Moving Average (EMA) with a configurable
  learning rate to adapt to gradual appearance changes.

The number of particles is the primary speed/quality tradeoff knob —
making this tracker ideal for edge deployment analysis:

* ``num_particles=50``  — fast, suitable for heavily constrained hardware
* ``num_particles=150`` — balanced (default), good accuracy
* ``num_particles=500`` — high-accuracy, slower

This tracker is fundamentally different from the correlation-filter
trackers in EOVOT (MOSSE, KCF): it maintains an explicit probability
distribution over target locations and naturally handles multi-modal
distributions arising from occlusion and clutter.

Reference
---------
Isard, M. and Blake, A. "CONDENSATION — Conditional Density Propagation
for Visual Tracking." International Journal of Computer Vision 29(1),
5–28, 1998.

Example::

    from eovot.trackers.particle_filter import ParticleFilterTracker
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine

    tracker = ParticleFilterTracker(num_particles=200)
    dataset = SyntheticDataset(num_sequences=5)
    engine  = BenchmarkEngine()
    result  = engine.run(tracker, dataset, dataset_name="Synthetic")
    print(result)
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import BaseTracker, BBox


class ParticleFilterTracker(BaseTracker):
    """Probabilistic tracker using Sequential Monte Carlo sampling.

    Args:
        num_particles: Number of particles (hypotheses) maintained per frame.
            More particles improve robustness at higher compute cost.
            Default: ``150``.
        motion_std: Standard deviation of the Gaussian random walk motion
            model (pixels).  Larger values allow tracking faster-moving
            targets at the cost of higher positional uncertainty.
            Default: ``10.0``.
        num_hue_bins: Number of histogram bins for the H channel of HSV.
            Default: ``16``.
        num_sat_bins: Number of histogram bins for the S channel of HSV.
            Default: ``8``.
        hist_update_lr: EMA learning rate for the online appearance model
            update in ``[0, 1]``.  ``0`` freezes the template; ``1`` replaces
            it entirely each frame.  Default: ``0.05``.
        ess_threshold_ratio: Resampling is triggered when
            ``ESS < num_particles * ess_threshold_ratio``.
            Default: ``0.5``.
        bbox_scale: Fraction of the initial bounding box used to extract
            appearance patches (``1.0`` = exact bbox, ``0.8`` = inner 80 %).
            Default: ``1.0``.
        seed: RNG seed for reproducible particle initialisation.
            Default: ``42``.

    Example::

        tracker = ParticleFilterTracker(num_particles=100, motion_std=8.0)
        tracker.initialize(first_frame, (x, y, w, h))
        bbox = tracker.update(next_frame)
    """

    def __init__(
        self,
        num_particles: int = 150,
        motion_std: float = 10.0,
        num_hue_bins: int = 16,
        num_sat_bins: int = 8,
        hist_update_lr: float = 0.05,
        ess_threshold_ratio: float = 0.5,
        bbox_scale: float = 1.0,
        seed: int = 42,
    ) -> None:
        super().__init__(name="ParticleFilter")
        self.num_particles = num_particles
        self.motion_std = motion_std
        self.num_hue_bins = num_hue_bins
        self.num_sat_bins = num_sat_bins
        self.hist_update_lr = hist_update_lr
        self.ess_threshold_ratio = ess_threshold_ratio
        self.bbox_scale = bbox_scale

        self._particles: np.ndarray = np.empty((0, 2), dtype=np.float64)
        self._weights: np.ndarray = np.empty(0, dtype=np.float64)
        self._target_hist: np.ndarray = np.empty(0, dtype=np.float64)
        self._bbox_w: float = 0.0
        self._bbox_h: float = 0.0
        self._rng = np.random.default_rng(seed=seed)

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the particle filter around the initial bounding box.

        Particles are sampled from a Gaussian centred at the target centre
        with ``motion_std`` as initial spread to cover local uncertainty.

        Args:
            frame: BGR image ``(H, W, 3)``.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.

        Raises:
            ValueError: If width or height in *bbox* is not positive.
        """
        x, y, w, h = (float(v) for v in bbox)
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid bbox {bbox}: width and height must be positive.")

        self._bbox_w = w
        self._bbox_h = h
        cx, cy = x + w / 2.0, y + h / 2.0

        self._particles = self._rng.normal(
            loc=[cx, cy],
            scale=self.motion_std,
            size=(self.num_particles, 2),
        )
        self._weights = np.ones(self.num_particles, dtype=np.float64) / self.num_particles
        self._target_hist = self._extract_hist(frame, cx, cy)

    def update(self, frame: np.ndarray) -> BBox:
        """Propagate particles, weight by likelihood, and return MAP estimate.

        Steps:

        1. **Predict** — perturb each particle with Gaussian motion noise.
        2. **Observe** — compute Bhattacharyya likelihood for each particle.
        3. **Estimate** — weighted mean of particles gives the position.
        4. **Update** — EMA template update at the estimated position.
        5. **Resample** — systematic resampling if ESS is low.

        Args:
            frame: BGR image ``(H, W, 3)``.

        Returns:
            Predicted bounding box ``(x, y, w, h)`` as the weighted mean
            position of all particles with the fixed initial bbox size.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._particles.size == 0:
            raise RuntimeError(
                "ParticleFilterTracker not initialised. Call initialize() first."
            )

        fh, fw = frame.shape[:2]

        # Step 1: Predict via motion model
        noise = self._rng.normal(0.0, self.motion_std, self._particles.shape)
        self._particles = self._particles + noise
        self._particles[:, 0] = np.clip(self._particles[:, 0], 0.0, fw - 1.0)
        self._particles[:, 1] = np.clip(self._particles[:, 1], 0.0, fh - 1.0)

        # Step 2: Compute likelihoods
        likelihoods = self._compute_likelihoods(frame)
        self._weights = self._weights * likelihoods
        total = self._weights.sum()
        if total < 1e-12:
            # Catastrophic weight collapse — reset to uniform
            self._weights = np.ones(self.num_particles, dtype=np.float64) / self.num_particles
        else:
            self._weights /= total

        # Step 3: Weighted mean estimate
        cx = float(np.dot(self._weights, self._particles[:, 0]))
        cy = float(np.dot(self._weights, self._particles[:, 1]))

        # Step 4: Online appearance model update
        if self.hist_update_lr > 0:
            obs_hist = self._extract_hist(frame, cx, cy)
            self._target_hist = (
                (1.0 - self.hist_update_lr) * self._target_hist
                + self.hist_update_lr * obs_hist
            )

        # Step 5: Systematic resampling when ESS is low
        ess = 1.0 / float(np.sum(self._weights ** 2))
        if ess < self.num_particles * self.ess_threshold_ratio:
            self._particles = self._systematic_resample()
            self._weights = np.ones(self.num_particles, dtype=np.float64) / self.num_particles

        x0 = cx - self._bbox_w / 2.0
        y0 = cy - self._bbox_h / 2.0
        return (float(x0), float(y0), float(self._bbox_w), float(self._bbox_h))

    # ------------------------------------------------------------------
    # Appearance model
    # ------------------------------------------------------------------

    def _extract_hist(self, frame: np.ndarray, cx: float, cy: float) -> np.ndarray:
        """Extract and normalise a 2-D HS histogram from the target region.

        The region is a ``bbox_scale``-scaled box around ``(cx, cy)``.
        The Hue and Saturation channels of HSV are used because they are
        more illumination-invariant than raw BGR intensities.

        Args:
            frame: BGR image.
            cx, cy: Centre of the target region.

        Returns:
            1-D normalised histogram of length
            ``num_hue_bins * num_sat_bins`` with values summing to 1.
        """
        fh, fw = frame.shape[:2]
        half_w = max(1, int(self._bbox_w * self.bbox_scale / 2.0))
        half_h = max(1, int(self._bbox_h * self.bbox_scale / 2.0))
        x1 = max(0, int(cx) - half_w)
        y1 = max(0, int(cy) - half_h)
        x2 = min(fw, int(cx) + half_w)
        y2 = min(fh, int(cy) + half_h)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return np.zeros(self.num_hue_bins * self.num_sat_bins, dtype=np.float64)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1], None,
            [self.num_hue_bins, self.num_sat_bins],
            [0, 180, 0, 256],
        ).flatten().astype(np.float64)
        total = hist.sum()
        return hist / (total + 1e-8)

    def _compute_likelihoods(self, frame: np.ndarray) -> np.ndarray:
        """Compute Bhattacharyya likelihood for every particle.

        The Bhattacharyya coefficient ``BC = sum(sqrt(p * q))`` measures
        the overlap of two normalised histograms; it lies in ``[0, 1]``
        and is used directly as the observation likelihood.

        Args:
            frame: BGR image.

        Returns:
            Float64 array of shape ``(num_particles,)`` with values in
            ``[0, 1]``.
        """
        likelihoods = np.empty(self.num_particles, dtype=np.float64)
        for i in range(self.num_particles):
            cx = float(self._particles[i, 0])
            cy = float(self._particles[i, 1])
            hist = self._extract_hist(frame, cx, cy)
            likelihoods[i] = float(np.sum(np.sqrt(self._target_hist * hist + 1e-12)))
        return likelihoods

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    def _systematic_resample(self) -> np.ndarray:
        """Systematic resampling: O(N) and lower variance than multinomial.

        Draws ``num_particles`` samples from the current weighted particle
        set using a single uniform random number and evenly-spaced offsets,
        ensuring at least one sample per high-weight particle.

        Returns:
            New particle array of shape ``(num_particles, 2)``.
        """
        N = self.num_particles
        cumsum = np.cumsum(self._weights)
        u0 = self._rng.uniform(0.0, 1.0 / N)
        positions = u0 + np.arange(N, dtype=np.float64) / N
        indices = np.searchsorted(cumsum, positions)
        indices = np.clip(indices, 0, N - 1)
        return self._particles[indices].copy()

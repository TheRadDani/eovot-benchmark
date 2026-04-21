"""Edge deployment scoring for EOVOT benchmarks.

Computes a hardware-aware composite score for each tracker-on-device
combination, enabling fair cross-device comparison that reflects real
deployment constraints rather than pure accuracy.

The :class:`EdgeScorer` weighs four normalised sub-scores:

* **Accuracy**  — mean IoU normalised to ``[0, 1]``.
* **Speed**     — FPS relative to the device ``target_fps`` (capped at 1.0).
* **Memory**    — fraction of device ``memory_limit_mb`` *not* consumed.
* **Energy**    — energy efficiency per frame normalised against the device's
  per-frame budget at full TDP.

Weights are configurable via :class:`EdgeScoreWeights`.  When energy data is
absent the energy weight is redistributed proportionally across the remaining
three dimensions.

Usage::

    from eovot.profiling.hardware_profiles import PROFILES
    from eovot.metrics.edge_score import EdgeScorer

    scorer = EdgeScorer(profile=PROFILES["jetson_nano"])
    score = scorer.compute(
        mean_iou=0.55,
        mean_fps=45.0,
        peak_memory_mb=180.0,
        energy_per_frame_mj=2.5,
    )
    print(score)
    # EdgeScore[NVIDIA Jetson Nano] composite=0.7123 ...

Reference:
    Efficiency scoring philosophy inspired by MLPerf Edge and EfficientNet's
    accuracy-vs-FLOPs trade-off analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..profiling.hardware_profiles import HardwareProfile


@dataclass
class EdgeScoreWeights:
    """Relative importance of each dimension in the composite EdgeScore.

    Weights are automatically normalised to sum to 1.0 before use.

    Attributes:
        accuracy: Weight for mean IoU (accuracy dimension).
        speed: Weight for FPS relative to device ``target_fps``.
        memory: Weight for memory efficiency (lower RSS is better).
        energy: Weight for per-frame energy efficiency.  Set to 0.0
            to ignore energy (e.g. when energy profiling is unavailable).
    """

    accuracy: float = 0.40
    speed: float = 0.30
    memory: float = 0.15
    energy: float = 0.15

    def normalised(self) -> "EdgeScoreWeights":
        """Return a copy with weights normalised to sum to 1.0.

        Raises:
            ValueError: If all weights are zero.
        """
        total = self.accuracy + self.speed + self.memory + self.energy
        if total <= 0:
            raise ValueError("At least one EdgeScoreWeight must be positive.")
        return EdgeScoreWeights(
            accuracy=self.accuracy / total,
            speed=self.speed / total,
            memory=self.memory / total,
            energy=self.energy / total,
        )


@dataclass
class EdgeScoreResult:
    """Composite edge deployment score and its individual components.

    All sub-scores are in ``[0, 1]``, where 1.0 is the best possible
    performance on the target device.

    Attributes:
        composite: Weighted composite edge score.
        accuracy_score: Normalised IoU (equals ``mean_iou`` clamped to [0, 1]).
        speed_score: ``min(mean_fps / target_fps, 1.0)``.
        memory_score: ``1 − peak_memory_mb / memory_limit_mb``, clamped to [0, 1].
        energy_score: Energy efficiency score, or 0.0 if energy data is absent.
        fits_on_device: ``True`` when FPS ≥ target_fps AND memory ≤ memory_limit_mb.
        hardware_profile: The :class:`~eovot.profiling.hardware_profiles.HardwareProfile`
            used for scoring.
    """

    composite: float
    accuracy_score: float
    speed_score: float
    memory_score: float
    energy_score: float
    fits_on_device: bool
    hardware_profile: HardwareProfile

    def __str__(self) -> str:
        fit_tag = "YES" if self.fits_on_device else "NO"
        return (
            f"EdgeScore[{self.hardware_profile.name}] "
            f"composite={self.composite:.4f}  "
            f"acc={self.accuracy_score:.3f}  "
            f"speed={self.speed_score:.3f}  "
            f"mem={self.memory_score:.3f}  "
            f"energy={self.energy_score:.3f}  "
            f"fits={fit_tag}"
        )

    def to_dict(self) -> dict:
        """Return a plain dict suitable for JSON serialisation."""
        return {
            "hardware_profile": self.hardware_profile.name,
            "composite": round(self.composite, 4),
            "accuracy_score": round(self.accuracy_score, 4),
            "speed_score": round(self.speed_score, 4),
            "memory_score": round(self.memory_score, 4),
            "energy_score": round(self.energy_score, 4),
            "fits_on_device": self.fits_on_device,
        }


class EdgeScorer:
    """Compute hardware-aware edge deployment scores from benchmark metrics.

    Args:
        profile: Target device as a
            :class:`~eovot.profiling.hardware_profiles.HardwareProfile`.
        weights: Contribution of each dimension.
            Defaults to ``EdgeScoreWeights()`` (accuracy-heavy preset).

    Example::

        from eovot.profiling.hardware_profiles import PROFILES
        from eovot.metrics.edge_score import EdgeScorer, EdgeScoreWeights

        # Default weights (accuracy-heavy)
        scorer = EdgeScorer(profile=PROFILES["raspberry_pi4"])

        # Equal weights
        scorer = EdgeScorer(
            profile=PROFILES["jetson_nano"],
            weights=EdgeScoreWeights(accuracy=0.25, speed=0.25, memory=0.25, energy=0.25),
        )
    """

    def __init__(
        self,
        profile: HardwareProfile,
        weights: Optional[EdgeScoreWeights] = None,
    ) -> None:
        self.profile = profile
        self.weights = (weights or EdgeScoreWeights()).normalised()

    def compute(
        self,
        mean_iou: float,
        mean_fps: float,
        peak_memory_mb: float,
        energy_per_frame_mj: Optional[float] = None,
    ) -> EdgeScoreResult:
        """Compute the composite edge score for one tracker result.

        Args:
            mean_iou: Mean IoU across all evaluated frames (0–1).
            mean_fps: Mean frames per second.
            peak_memory_mb: Peak RSS memory consumption in MB.
            energy_per_frame_mj: Mean energy per frame in milli-Joules,
                or ``None`` if energy profiling was disabled.

        Returns:
            :class:`EdgeScoreResult` with the composite score and all
            per-dimension sub-scores.
        """
        w = self.weights
        p = self.profile

        # --- Sub-scores (each clamped to [0, 1]) ---
        acc = float(min(max(mean_iou, 0.0), 1.0))

        speed = float(min(mean_fps / p.target_fps, 1.0))

        mem = float(min(max(1.0 - peak_memory_mb / p.memory_limit_mb, 0.0), 1.0))

        # Energy: per-frame budget at full TDP for one target-fps interval.
        if energy_per_frame_mj is not None and energy_per_frame_mj >= 0:
            budget_mj = (p.tdp_watts / p.target_fps) * 1_000.0
            energy = float(min(max(1.0 - energy_per_frame_mj / budget_mj, 0.0), 1.0))
        else:
            energy = 0.0

        # --- Composite ---
        if energy_per_frame_mj is None:
            # Redistribute energy weight across the other three dimensions.
            other_total = w.accuracy + w.speed + w.memory
            if other_total > 0:
                composite = (
                    w.accuracy * acc + w.speed * speed + w.memory * mem
                ) / other_total
            else:
                composite = 0.0
        else:
            composite = (
                w.accuracy * acc
                + w.speed * speed
                + w.memory * mem
                + w.energy * energy
            )

        fits = (mean_fps >= p.target_fps) and (peak_memory_mb <= p.memory_limit_mb)

        return EdgeScoreResult(
            composite=float(composite),
            accuracy_score=acc,
            speed_score=speed,
            memory_score=mem,
            energy_score=energy,
            fits_on_device=fits,
            hardware_profile=p,
        )

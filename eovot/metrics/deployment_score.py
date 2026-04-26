"""Device-specific deployment scoring for EOVOT benchmark results.

A *deployment score* is a single scalar in [0, 1] that summarises how
suitable a tracker is for a specific edge device.  It combines three
sub-scores, each normalised independently to [0, 1]:

* **accuracy_score** — mean IoU over evaluated sequences.
* **fps_score**      — fraction of the device's target FPS achieved,
                       clipped to [0, 1].  A score of 1.0 means the
                       tracker runs at or above the target frame-rate.
* **memory_score**   — head-room fraction ``1 − (peak_mem / limit)``,
                       clipped to [0, 1].  A score of 1.0 means the
                       tracker uses negligible memory.

Composite score (weighted average)::

    deployment_score = w_acc * accuracy + w_fps * fps + w_mem * memory

Default weights: ``w_acc=0.5, w_fps=0.3, w_mem=0.2``.

A separate boolean flag ``deployable`` is set only when *both* hard
constraints are satisfied simultaneously:

  * ``fps >= device.target_fps``
  * ``peak_memory_mb <= device.memory_limit_mb``

Typical usage::

    from eovot.metrics.deployment_score import DeploymentScorer, score_all_devices
    from eovot.profiling.device_profiles import get_device

    scorer = DeploymentScorer(device=get_device("raspberry_pi_4"))
    result = scorer.score(mean_iou=0.55, fps=22.0, peak_memory_mb=180.0,
                          tracker_name="MOSSE")
    print(result)
    # DeploymentScore[raspberry_pi_4] total=0.677 (acc=0.550, fps=1.000,
    #   mem=0.956) [DEPLOYABLE]

    # Score against every registered device in one call:
    all_scores = score_all_devices(mean_iou=0.55, fps=22.0,
                                   peak_memory_mb=180.0, tracker_name="MOSSE")
    for key, sc in all_scores.items():
        print(key, sc.total_score, sc.deployable)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from ..profiling.device_profiles import DEVICE_PROFILES, DeviceProfile, get_device


@dataclass
class DeploymentScore:
    """Deployment suitability scores for one tracker–device pair.

    Attributes:
        device_key:      Key of the target :class:`~eovot.profiling.device_profiles.DeviceProfile`.
        tracker_name:    Tracker identifier.
        accuracy_score:  Sub-score from mean IoU (0–1).
        fps_score:       Sub-score from FPS relative to device target (0–1).
        memory_score:    Sub-score from memory head-room (0–1).
        total_score:     Weighted composite score (0–1).
        deployable:      ``True`` iff both FPS and memory hard constraints
                         are met simultaneously.
        fps_achieved:    Observed FPS value passed to :meth:`~DeploymentScorer.score`.
        peak_memory_mb:  Observed peak memory (MB) passed to
                         :meth:`~DeploymentScorer.score`.
    """

    device_key: str
    tracker_name: str
    accuracy_score: float
    fps_score: float
    memory_score: float
    total_score: float
    deployable: bool
    fps_achieved: float
    peak_memory_mb: float

    def __str__(self) -> str:
        tag = "DEPLOYABLE" if self.deployable else "NOT DEPLOYABLE"
        return (
            f"DeploymentScore[{self.device_key}] "
            f"total={self.total_score:.3f}  "
            f"(acc={self.accuracy_score:.3f}, "
            f"fps={self.fps_score:.3f}, "
            f"mem={self.memory_score:.3f})  "
            f"[{tag}]"
        )

    def to_dict(self) -> Dict:
        """Serialize to a plain dict for JSON export."""
        return {
            "device_key": self.device_key,
            "tracker_name": self.tracker_name,
            "total_score": round(self.total_score, 4),
            "accuracy_score": round(self.accuracy_score, 4),
            "fps_score": round(self.fps_score, 4),
            "memory_score": round(self.memory_score, 4),
            "deployable": self.deployable,
            "fps_achieved": round(self.fps_achieved, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
        }


class DeploymentScorer:
    """Compute deployment suitability scores for a specific edge device.

    Args:
        device:      Target :class:`~eovot.profiling.device_profiles.DeviceProfile`.
        weight_acc:  Weight for accuracy sub-score (default 0.5).
        weight_fps:  Weight for FPS sub-score (default 0.3).
        weight_mem:  Weight for memory sub-score (default 0.2).

    Raises:
        ValueError: If weights do not sum to 1.0 (tolerance ±0.01).
    """

    def __init__(
        self,
        device: DeviceProfile,
        weight_acc: float = 0.5,
        weight_fps: float = 0.3,
        weight_mem: float = 0.2,
    ) -> None:
        total_w = weight_acc + weight_fps + weight_mem
        if abs(total_w - 1.0) > 0.01:
            raise ValueError(
                f"Weights must sum to 1.0, got {total_w:.4f} "
                f"(acc={weight_acc}, fps={weight_fps}, mem={weight_mem})"
            )
        self.device = device
        self.weight_acc = weight_acc
        self.weight_fps = weight_fps
        self.weight_mem = weight_mem

    def score(
        self,
        mean_iou: float,
        fps: float,
        peak_memory_mb: float,
        tracker_name: str = "unknown",
    ) -> DeploymentScore:
        """Compute the deployment score for a tracker on this device.

        Args:
            mean_iou:        Mean intersection-over-union across all evaluated
                             sequences, in [0, 1].
            fps:             Mean measured frames per second.
            peak_memory_mb:  Peak RSS memory usage in megabytes.
            tracker_name:    Label embedded in the returned result.

        Returns:
            :class:`DeploymentScore` with all sub-scores and composite total.
        """
        acc_score = float(max(0.0, min(1.0, mean_iou)))

        if self.device.target_fps > 0:
            fps_score = float(min(1.0, fps / self.device.target_fps))
        else:
            fps_score = 1.0

        if self.device.memory_limit_mb > 0:
            mem_fraction = peak_memory_mb / self.device.memory_limit_mb
            mem_score = float(max(0.0, 1.0 - mem_fraction))
        else:
            mem_score = 1.0

        total = (
            self.weight_acc * acc_score
            + self.weight_fps * fps_score
            + self.weight_mem * mem_score
        )

        deployable = (
            self.device.meets_fps(fps)
            and self.device.fits_memory(peak_memory_mb)
        )

        return DeploymentScore(
            device_key=self.device.key,
            tracker_name=tracker_name,
            accuracy_score=acc_score,
            fps_score=fps_score,
            memory_score=mem_score,
            total_score=round(total, 4),
            deployable=deployable,
            fps_achieved=fps,
            peak_memory_mb=peak_memory_mb,
        )

    @classmethod
    def for_device_key(cls, key: str, **kwargs) -> "DeploymentScorer":
        """Convenience constructor that looks up a device by registry key.

        Args:
            key:      Key in :data:`~eovot.profiling.device_profiles.DEVICE_PROFILES`.
            **kwargs: Forwarded to :class:`DeploymentScorer` (weight overrides).
        """
        return cls(device=get_device(key), **kwargs)


def score_all_devices(
    mean_iou: float,
    fps: float,
    peak_memory_mb: float,
    tracker_name: str = "unknown",
    weight_acc: float = 0.5,
    weight_fps: float = 0.3,
    weight_mem: float = 0.2,
) -> Dict[str, DeploymentScore]:
    """Score a tracker against every registered device profile.

    Args:
        mean_iou:        Mean IoU across evaluated sequences.
        fps:             Mean frames per second.
        peak_memory_mb:  Peak memory usage in MB.
        tracker_name:    Label embedded in each result.
        weight_acc:      Accuracy weight applied uniformly to all devices.
        weight_fps:      FPS weight applied uniformly to all devices.
        weight_mem:      Memory weight applied uniformly to all devices.

    Returns:
        Dict mapping device key → :class:`DeploymentScore`.
    """
    out: Dict[str, DeploymentScore] = {}
    for key, profile in DEVICE_PROFILES.items():
        scorer = DeploymentScorer(
            device=profile,
            weight_acc=weight_acc,
            weight_fps=weight_fps,
            weight_mem=weight_mem,
        )
        out[key] = scorer.score(
            mean_iou=mean_iou,
            fps=fps,
            peak_memory_mb=peak_memory_mb,
            tracker_name=tracker_name,
        )
    return out

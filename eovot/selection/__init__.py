"""Hardware-aware tracker selection for EOVOT.

This subpackage provides utilities to characterise the current hardware
environment and recommend the best available tracker given deployment
constraints (target FPS, memory budget, accuracy priority).

Key classes
-----------
HardwareProfile
    Describes a device's CPU, memory, and estimated power envelope.
TrackerSelector
    Ranks and filters trackers against a :class:`HardwareProfile` and a
    set of user-specified constraints.

Example::

    from eovot.selection import HardwareProfile, TrackerSelector

    profile = HardwareProfile.detect()
    selector = TrackerSelector()
    recommendation = selector.recommend(
        profile,
        target_fps=30.0,
        max_memory_mb=256.0,
        accuracy_weight=0.6,
    )
    print(recommendation)
"""

from .hardware_profile import DeviceClass, HardwareProfile
from .tracker_selector import TrackerConstraints, TrackerRecommendation, TrackerSelector

__all__ = [
    "DeviceClass",
    "HardwareProfile",
    "TrackerConstraints",
    "TrackerRecommendation",
    "TrackerSelector",
]

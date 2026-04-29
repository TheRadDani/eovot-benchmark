from .base import BaseTracker, BBox
from .mosse import MOSSETracker
from .kcf import KCFTracker
from .csrt import CSRTTracker
from .median_flow import MedianFlowTracker

try:
    from .siamfc import SiamFCTracker
    _SIAMFC_AVAILABLE = True
except ImportError:
    _SIAMFC_AVAILABLE = False

__all__ = [
    "BaseTracker",
    "BBox",
    "MOSSETracker",
    "KCFTracker",
    "CSRTTracker",
    "MedianFlowTracker",
]

if _SIAMFC_AVAILABLE:
    __all__.append("SiamFCTracker")

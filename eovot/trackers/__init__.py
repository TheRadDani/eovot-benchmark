from .base import BaseTracker, BBox
from .mosse import MOSSETracker
from .kcf import KCFTracker
from .csrt import CSRTTracker
from .median_flow import MedianFlowTracker
from .mil import MILTracker
from .template import TemplatePatchTracker

#: Central registry mapping short names to tracker classes.
#: Used by CLI tools and experiment configs for programmatic tracker discovery.
TRACKER_REGISTRY = {
    "MOSSE": MOSSETracker,
    "KCF": KCFTracker,
    "CSRT": CSRTTracker,
    "MedianFlow": MedianFlowTracker,
    "MIL": MILTracker,
    "TemplateMatch": TemplatePatchTracker,
}

__all__ = [
    "BaseTracker",
    "BBox",
    "MOSSETracker",
    "KCFTracker",
    "CSRTTracker",
    "MedianFlowTracker",
    "MILTracker",
    "TemplatePatchTracker",
    "TRACKER_REGISTRY",
]

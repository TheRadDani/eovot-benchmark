"""Analysis sub-package — sequence attribute analysis and per-attribute reporting.

Two complementary attribute systems are provided:

1. **Auto-detection** (``sequence_attributes``, ``attribute_report``):
   Computes difficulty flags directly from ground-truth bounding boxes.
   Useful when no external annotation file is available.

2. **Annotation-based** (``attribute``):
   Loads attribute tags from OTB-style CSV files or manual registration,
   then stratifies accuracy metrics (IoU, success AUC, precision AUC) by
   attribute across one or more trackers.
"""

from .attribute import (
    STANDARD_ATTRIBUTES,
    AttributeAnalysis,
    AttributeAnalyzer,
    AttributeResult,
)
from .attribute_report import AttributeReport, generate_attribute_report
from .sequence_attributes import (
    AttributeFlags,
    SequenceAttributes,
    compute_sequence_attributes,
    tag_sequences,
)

__all__ = [
    # Annotation-based analyser
    "AttributeAnalyzer",
    "AttributeAnalysis",
    "AttributeResult",
    "STANDARD_ATTRIBUTES",
    # Auto-detection
    "AttributeFlags",
    "SequenceAttributes",
    "compute_sequence_attributes",
    "tag_sequences",
    # Per-attribute report
    "AttributeReport",
    "generate_attribute_report",
]

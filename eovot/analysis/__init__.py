"""Attribute-based performance analysis for EOVOT.

This sub-package provides tools for decomposing tracker performance by
tracking-challenge *attributes* — motion patterns and appearance
conditions derived directly from ground-truth trajectories.

Attribute breakdown is the standard evaluation methodology used in
OTB-100, VOT-20XX, and LaSOT papers.  It pinpoints *why* a tracker
fails (e.g. fast motion vs. scale variation) rather than just *how much*.

Typical usage::

    from eovot.analysis import AttributeDetector, AttributeBreakdown

    detector = AttributeDetector()
    breakdown = AttributeBreakdown()

    # Per-sequence analysis
    profiles = detector.detect(gt_boxes)               # {attr: bool array}
    results  = breakdown.compute(gt_boxes, ious=ious)  # {attr: BreakdownResult}

    # Multi-tracker comparison table
    table = breakdown.compare_trackers(
        gt_boxes=gt_boxes,
        tracker_ious={"MOSSE": mosse_ious, "KCF": kcf_ious},
    )
    print(table.to_markdown())
"""

from .attributes import SequenceAttribute, AttributeProfile, AttributeDetector
from .breakdown import BreakdownResult, AttributeBreakdown, TrackerAttributeComparison

__all__ = [
    "SequenceAttribute",
    "AttributeProfile",
    "AttributeDetector",
    "BreakdownResult",
    "AttributeBreakdown",
    "TrackerAttributeComparison",
]

"""Analysis sub-package — attribute-stratified benchmarking and diagnostics."""

from eovot.analysis.attribute import (
    STANDARD_ATTRIBUTES,
    AttributeAnalysis,
    AttributeAnalyzer,
    AttributeResult,
)

__all__ = [
    "AttributeAnalyzer",
    "AttributeAnalysis",
    "AttributeResult",
    "STANDARD_ATTRIBUTES",
]

"""Analysis sub-package — per-attribute and cross-experiment tracker evaluation."""

from .attribute_analyzer import (
    AttributeStats,
    AttributeReport,
    AttributeAnalyzer,
)

__all__ = [
    "AttributeStats",
    "AttributeReport",
    "AttributeAnalyzer",
]

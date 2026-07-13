"""Sequence analysis utilities for EOVOT.

This sub-package provides tools to characterize tracking sequences and
benchmark results beyond aggregate accuracy numbers, enabling stratified
evaluation and failure analysis.

Modules
-------
sequence_analyzer
    :class:`~eovot.analysis.sequence_analyzer.SequenceAnalyzer` — computes
    per-sequence difficulty attributes (motion speed, scale change, aspect-
    ratio instability) from ground-truth arrays and classifies sequences into
    difficulty tiers.  Produces per-tier performance breakdowns that reveal
    where a tracker succeeds or fails.
"""

from .sequence_analyzer import SequenceAnalyzer, SequenceAttributes, DifficultyTier

__all__ = ["SequenceAnalyzer", "SequenceAttributes", "DifficultyTier"]

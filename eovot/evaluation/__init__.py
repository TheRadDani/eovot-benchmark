"""Unified evaluation pipeline for EOVOT.

Chains BenchmarkEngine through all metric layers — robustness, temporal
consistency, attribute analysis, efficiency scoring, and statistical testing —
into a single :class:`EvaluationPipeline` call that produces a structured
:class:`EvaluationReport` and a ready-to-read Markdown document.
"""

from .pipeline import EvaluationPipeline, EvaluationReport

__all__ = ["EvaluationPipeline", "EvaluationReport"]

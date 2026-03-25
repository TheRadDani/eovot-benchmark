"""Visualisation utilities for EOVOT benchmark results.

Provides publication-ready success and precision curve plots that follow
the standard VOT evaluation protocol used in academic papers.
"""

from .curves import CurvePlotter

__all__ = ["CurvePlotter"]

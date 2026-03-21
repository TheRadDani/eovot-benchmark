"""Tracker sub-package — base interface and built-in implementations."""

from .base import BaseTracker, BBox
from .mosse import MOSSETracker

__all__ = ["BaseTracker", "BBox", "MOSSETracker"]

"""Analysis sub-package — dataset and sequence characterisation tools.

Provides utilities for understanding dataset properties and sequence
difficulty without requiring any tracker inference.

Modules
-------
sequence_profiler
    :class:`~eovot.analysis.sequence_profiler.SequenceDifficultyProfiler`
    and :class:`~eovot.analysis.sequence_profiler.SequenceDifficulty` —
    characterise sequences by motion, scale change, and deformation, then
    rank them by composite difficulty score.

Quick start::

    from eovot.analysis import SequenceDifficultyProfiler

    profiler = SequenceDifficultyProfiler()
    diffs    = profiler.profile_dataset(dataset)
    ranked   = profiler.sort_by_difficulty(diffs)   # hardest first
    stats    = profiler.summary_stats(diffs)
"""

from .sequence_profiler import SequenceDifficulty, SequenceDifficultyProfiler

__all__ = ["SequenceDifficulty", "SequenceDifficultyProfiler"]

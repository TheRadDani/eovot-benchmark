"""Hyperparameter tuning utilities for EOVOT trackers.

This module provides two complementary analysis tools:

:class:`~eovot.tuning.grid_search.GridSearchRunner`
    Exhaustive grid search over a discrete parameter grid.  Evaluates every
    combination of tracker hyperparameters on a dataset and ranks the results
    by a chosen scalar metric (e.g. ``mean_iou``, ``mean_fps``).

:class:`~eovot.tuning.sensitivity.SensitivityAnalyzer`
    One-At-a-Time (OAT) sensitivity analysis.  Holds all parameters fixed at
    their baseline values and varies each one independently across a provided
    range.  Reports a normalised sensitivity score for each parameter, making
    it easy to identify which knobs matter most.

Both tools integrate with the existing :class:`~eovot.benchmark.engine.BenchmarkEngine`
and produce :class:`~eovot.tuning.grid_search.TuningResult` /
:class:`~eovot.tuning.sensitivity.SensitivityReport` objects that can be
exported as Markdown or CSV.

Typical usage::

    from eovot.tuning.grid_search import GridSearchRunner
    from eovot.trackers.mosse import MOSSETracker

    runner = GridSearchRunner(
        tracker_class=MOSSETracker,
        param_grid={"learning_rate": [0.05, 0.125, 0.2], "sigma": [1.0, 2.0, 3.0]},
        metric="mean_iou",
        verbose=True,
    )
    result = runner.run(dataset, dataset_name="OTB100")
    print(result.best_params)
    print(result.to_markdown())
"""

from .grid_search import GridSearchRunner, TuningEntry, TuningResult
from .sensitivity import SensitivityAnalyzer, SensitivityReport, ParameterSensitivity

__all__ = [
    "GridSearchRunner",
    "TuningEntry",
    "TuningResult",
    "SensitivityAnalyzer",
    "SensitivityReport",
    "ParameterSensitivity",
]

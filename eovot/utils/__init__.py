"""Utility sub-package for EOVOT.

Currently includes:

- :mod:`~eovot.utils.prediction_io` — read / write tracker predictions in
  OTB, GOT-10k, VOT, and EOVOT-JSON formats for interoperability and
  offline metric computation.
"""

from .prediction_io import (
    PredictionFormat,
    PredictionWriter,
    PredictionReader,
    load_predictions_from_benchmark_result,
)

__all__ = [
    "PredictionFormat",
    "PredictionWriter",
    "PredictionReader",
    "load_predictions_from_benchmark_result",
]

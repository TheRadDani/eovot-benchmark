"""Edge deployment constraint system for EOVOT.

Provides :class:`~eovot.constraints.profiles.EdgeProfile` dataclass for
declaring hardware constraints and :class:`~eovot.constraints.evaluator.ConstraintEvaluator`
for checking benchmark results against those constraints.

Quick start::

    from eovot.constraints import ConstraintEvaluator, RASPBERRY_PI_4

    ev = ConstraintEvaluator()
    report = ev.evaluate(benchmark_result, RASPBERRY_PI_4)
    print(report.summary())
"""

from .evaluator import ConstraintCheck, ConstraintEvaluator, ConstraintReport
from .profiles import (
    EMBEDDED_MICRO,
    JETSON_NANO,
    LAPTOP_CPU,
    MOBILE_CLASS,
    PREDEFINED_PROFILES,
    RASPBERRY_PI_4,
    EdgeProfile,
)

__all__ = [
    # Profile types and singletons
    "EdgeProfile",
    "RASPBERRY_PI_4",
    "JETSON_NANO",
    "MOBILE_CLASS",
    "EMBEDDED_MICRO",
    "LAPTOP_CPU",
    "PREDEFINED_PROFILES",
    # Evaluator types
    "ConstraintCheck",
    "ConstraintReport",
    "ConstraintEvaluator",
]

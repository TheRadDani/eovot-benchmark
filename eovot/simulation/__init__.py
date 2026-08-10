"""Frame budget simulation for edge-constrained tracker evaluation.

Provides :class:`~eovot.simulation.frame_budget.FrameBudgetSimulator` for
evaluating tracker accuracy under temporal subsampling constraints — the
scenario where an edge device cannot process every incoming camera frame.

Quick start::

    from eovot.simulation import FrameBudgetSimulator

    sim = FrameBudgetSimulator(budget_rates=[1.0, 0.5, 0.25, 0.1])
    curve = sim.simulate(tracker, sequence, native_fps=200.0)
    FrameBudgetSimulator.print_curve(curve)
"""

from .frame_budget import BudgetCurve, BudgetPoint, FrameBudgetSimulator

__all__ = ["FrameBudgetSimulator", "BudgetCurve", "BudgetPoint"]

"""Multi-objective tracker scoring for edge deployment trade-off analysis.

Provides a principled way to compare trackers across four competing objectives:
accuracy, throughput, memory efficiency, and energy consumption.  This is
essential for edge deployment research where optimising for accuracy alone
produces models that are impractical on constrained hardware.

Components
----------
:class:`ScoringWeights`
    Configurable per-objective weights that must sum to 1.0.  Four presets
    cover common research vs deployment scenarios.

:func:`compute_composite_scores`
    Min-max normalises each objective across the tracker set, then computes
    a weighted composite score.  Returns a ranked :class:`pandas.DataFrame`.

:func:`pareto_frontier`
    Identifies trackers on the Pareto frontier for any pair of objectives,
    exposing the trade-off curve without collapsing it to a single scalar.

Example::

    from eovot.metrics.scoring import compute_composite_scores, EDGE_WEIGHTS, pareto_frontier

    metrics = {
        "MOSSE":      {"auc": 0.48, "fps": 520.0, "peak_memory_mb": 42.0},
        "KCF":        {"auc": 0.55, "fps": 280.0, "peak_memory_mb": 68.0},
        "CSRT":       {"auc": 0.68, "fps":  45.0, "peak_memory_mb": 130.0},
        "MedianFlow": {"auc": 0.51, "fps": 150.0, "peak_memory_mb":  55.0},
    }

    df = compute_composite_scores(metrics, weights=EDGE_WEIGHTS)
    print(df[["tracker", "composite_score"]])

    pareto_names, pareto_df = pareto_frontier(metrics, "fps", "auc")
    print("Pareto-optimal trackers:", pareto_names)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# ScoringWeights
# ---------------------------------------------------------------------------


@dataclass
class ScoringWeights:
    """Per-objective importance weights for composite tracker scoring.

    All four weights must sum to exactly 1.0 (validated in ``__post_init__``).

    Attributes:
        accuracy: Weight for the accuracy objective (AUC / precision).
        speed:    Weight for the speed objective (FPS / latency).
        memory:   Weight for the memory-efficiency objective.
        energy:   Weight for the energy-consumption objective.
    """

    accuracy: float = 0.40
    speed: float = 0.30
    memory: float = 0.20
    energy: float = 0.10

    _TOL: float = 1e-6

    def __post_init__(self) -> None:
        total = self.accuracy + self.speed + self.memory + self.energy
        if abs(total - 1.0) > self._TOL:
            raise ValueError(
                f"ScoringWeights must sum to 1.0, got {total:.6f}. "
                f"(accuracy={self.accuracy}, speed={self.speed}, "
                f"memory={self.memory}, energy={self.energy})"
            )


# ---------------------------------------------------------------------------
# Preset weight configurations
# ---------------------------------------------------------------------------

RESEARCH_WEIGHTS = ScoringWeights(accuracy=0.70, speed=0.15, memory=0.10, energy=0.05)
"""Accuracy-first.  Suitable for academic benchmarking where deployment is not constrained."""

EDGE_WEIGHTS = ScoringWeights(accuracy=0.30, speed=0.35, memory=0.25, energy=0.10)
"""Speed and memory-first.  Suitable for evaluating trackers for embedded / IoT deployment."""

BALANCED_WEIGHTS = ScoringWeights(accuracy=0.40, speed=0.30, memory=0.20, energy=0.10)
"""Default balanced weighting.  A reasonable starting point for most evaluations."""

ENERGY_WEIGHTS = ScoringWeights(accuracy=0.30, speed=0.20, memory=0.15, energy=0.35)
"""Energy-first.  Suitable for battery-constrained edge devices where power is the bottleneck."""

PRESET_WEIGHTS: Dict[str, ScoringWeights] = {
    "research": RESEARCH_WEIGHTS,
    "edge": EDGE_WEIGHTS,
    "balanced": BALANCED_WEIGHTS,
    "energy": ENERGY_WEIGHTS,
}


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _minmax_normalize(values: np.ndarray, higher_is_better: bool, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalise *values* to [0, 1].

    If ``higher_is_better`` is False the direction is flipped so that lower
    raw values receive higher normalised scores.

    When all values are identical the function returns a uniform array of 1.0
    so that no tracker is penalised for a degenerate input.
    """
    lo, hi = values.min(), values.max()
    if hi - lo < eps:
        return np.ones_like(values, dtype=float)
    normalised = (values - lo) / (hi - lo)
    return normalised if higher_is_better else 1.0 - normalised


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------


def compute_composite_scores(
    tracker_metrics: Dict[str, Dict],
    weights: Optional[ScoringWeights] = None,
) -> pd.DataFrame:
    """Compute weighted composite scores for a set of trackers.

    Each objective is independently min-max normalised across the tracker set,
    then combined using the provided weights.  This makes the score relative:
    a composite_score of 1.0 means the tracker is best on all objectives in
    this comparison set.

    Args:
        tracker_metrics: Mapping of tracker name to a dict containing any
            subset of the following keys:

            - ``'auc'``            float in [0, 1] — success-curve AUC
            - ``'precision'``      float in [0, 1] — precision @ 20 px
            - ``'fps'``            float — frames per second
            - ``'mean_latency_ms'`` float — mean per-frame latency (ms)
            - ``'peak_memory_mb'``  float — peak RSS memory (MB)
            - ``'mean_energy_j'``   float — mean per-sequence energy (J)

            Missing keys are treated as 0.0 (for higher-is-better metrics) or
            ``inf`` (for lower-is-better metrics).

        weights: :class:`ScoringWeights` instance.  Defaults to
            :data:`BALANCED_WEIGHTS`.

    Returns:
        :class:`pandas.DataFrame` sorted by ``composite_score`` descending with
        columns: ``tracker``, ``auc``, ``precision``, ``fps``,
        ``mean_latency_ms``, ``peak_memory_mb``, ``mean_energy_j``,
        ``accuracy_score``, ``speed_score``, ``memory_score``,
        ``energy_score``, ``composite_score``.
    """
    if weights is None:
        weights = BALANCED_WEIGHTS

    names = list(tracker_metrics.keys())
    if not names:
        return pd.DataFrame()

    n = len(names)

    auc = np.array([tracker_metrics[t].get("auc", 0.0) for t in names], dtype=float)
    precision = np.array([tracker_metrics[t].get("precision", 0.0) for t in names], dtype=float)
    fps = np.array([tracker_metrics[t].get("fps", 0.0) for t in names], dtype=float)
    latency = np.array(
        [tracker_metrics[t].get("mean_latency_ms", float("inf")) for t in names], dtype=float
    )
    memory = np.array(
        [tracker_metrics[t].get("peak_memory_mb", float("inf")) for t in names], dtype=float
    )
    energy = np.array([tracker_metrics[t].get("mean_energy_j", 0.0) for t in names], dtype=float)

    # Replace inf with max-finite + 1 so normalisation doesn't degenerate.
    def _clean_lower_is_better(arr: np.ndarray) -> np.ndarray:
        finite = arr[np.isfinite(arr)]
        fill = float(finite.max()) * 2.0 + 1.0 if len(finite) else 1.0
        result = arr.copy()
        result[~np.isfinite(result)] = fill
        return result

    latency = _clean_lower_is_better(latency)
    memory = _clean_lower_is_better(memory)

    accuracy_raw = (auc + precision) / 2.0
    acc_score = _minmax_normalize(accuracy_raw, higher_is_better=True)
    speed_score = _minmax_normalize(fps, higher_is_better=True)
    memory_score = _minmax_normalize(memory, higher_is_better=False)

    # Energy normalisation: if all values are zero (not profiled), give full marks.
    if energy.sum() == 0.0:
        energy_score = np.ones(n, dtype=float)
    else:
        energy_score = _minmax_normalize(energy, higher_is_better=False)

    composite = (
        weights.accuracy * acc_score
        + weights.speed * speed_score
        + weights.memory * memory_score
        + weights.energy * energy_score
    )

    df = pd.DataFrame(
        {
            "tracker": names,
            "auc": np.round(auc, 4),
            "precision": np.round(precision, 4),
            "fps": np.round(fps, 2),
            "mean_latency_ms": np.round(latency, 3),
            "peak_memory_mb": np.round(memory, 2),
            "mean_energy_j": np.round(energy, 6),
            "accuracy_score": np.round(acc_score, 4),
            "speed_score": np.round(speed_score, 4),
            "memory_score": np.round(memory_score, 4),
            "energy_score": np.round(energy_score, 4),
            "composite_score": np.round(composite, 4),
        }
    )
    return df.sort_values("composite_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------


def pareto_frontier(
    tracker_metrics: Dict[str, Dict],
    objective_x: str = "fps",
    objective_y: str = "auc",
    x_higher_is_better: bool = True,
    y_higher_is_better: bool = True,
) -> Tuple[List[str], pd.DataFrame]:
    """Identify trackers on the Pareto frontier for two objectives.

    A tracker is Pareto-optimal when no other tracker is at least as good on
    both objectives *and* strictly better on at least one.  The frontier
    exposes the fundamental trade-off between the two objectives without
    collapsing them to a single scalar.

    Args:
        tracker_metrics: Mapping of tracker name to metric dict.  Must contain
            the keys specified by *objective_x* and *objective_y*.
        objective_x:          Metric key for the x-axis (e.g. ``'fps'``).
        objective_y:          Metric key for the y-axis (e.g. ``'auc'``).
        x_higher_is_better:   Whether a higher x value is preferred.
        y_higher_is_better:   Whether a higher y value is preferred.

    Returns:
        A 2-tuple ``(pareto_names, df)`` where:

        - *pareto_names* is the list of tracker names on the frontier.
        - *df* is a :class:`pandas.DataFrame` with columns
          ``['tracker', objective_x, objective_y, 'on_pareto_frontier']``.
    """
    names = list(tracker_metrics.keys())
    if not names:
        return [], pd.DataFrame()

    x_raw = np.array([tracker_metrics[t].get(objective_x, 0.0) for t in names], dtype=float)
    y_raw = np.array([tracker_metrics[t].get(objective_y, 0.0) for t in names], dtype=float)

    # Flip so "higher is always better" in comparison space.
    x_cmp = x_raw if x_higher_is_better else -x_raw
    y_cmp = y_raw if y_higher_is_better else -y_raw

    n = len(names)
    is_dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i: at least as good in both, strictly better in one.
            if (
                x_cmp[j] >= x_cmp[i]
                and y_cmp[j] >= y_cmp[i]
                and (x_cmp[j] > x_cmp[i] or y_cmp[j] > y_cmp[i])
            ):
                is_dominated[i] = True
                break

    pareto_mask = ~is_dominated
    pareto_names = [names[i] for i in range(n) if pareto_mask[i]]

    df = pd.DataFrame(
        {
            "tracker": names,
            objective_x: x_raw,
            objective_y: y_raw,
            "on_pareto_frontier": pareto_mask,
        }
    )
    return pareto_names, df

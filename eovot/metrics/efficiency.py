"""Edge-aware composite efficiency scoring for EOVOT tracker evaluation.

Bridges the gap between pure accuracy rankings (mIoU-only) and practical
edge deployment decisions by computing a weighted composite *edge score* that
penalises resource-hungry trackers and rewards those achieving good accuracy
within hardware constraints.

Edge Score formula::

    edge_score = (
        w_accuracy * iou
        + w_speed   * min(fps / target_fps, 1.0)
        + w_memory  * max(0, 1 - memory_mb / memory_budget_mb)
        + w_energy  * max(0, 1 - energy_mj / energy_budget_mj)   # optional
    ) / (w_accuracy + w_speed + w_memory + w_energy)

All component weights must be non-negative and are automatically normalised
to sum to 1.  The energy weight is excluded from normalisation when energy
data is unavailable.

Typical usage::

    from eovot.metrics.efficiency import EdgeScoreConfig, compute_edge_score

    config = EdgeScoreConfig(
        target_fps=30.0,
        memory_budget_mb=512.0,
        energy_budget_mj=5.0,
        w_accuracy=0.40,
        w_speed=0.30,
        w_memory=0.20,
        w_energy=0.10,
    )
    score = compute_edge_score(result_dict["summary"], config)
    print(score)

    # Rank multiple trackers
    ranked = rank_by_edge_score(all_result_dicts, config)
    md = edge_score_leaderboard_md(all_result_dicts, config)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class EdgeScoreConfig:
    """Configuration for edge-score computation.

    Attributes:
        target_fps: FPS above which the speed component is capped at 1.0.
            Represents the minimum acceptable real-time frame rate for the
            target application (e.g. 30 fps for drone control, 15 fps for
            IoT cameras).
        memory_budget_mb: Maximum acceptable RSS memory (MiB).  A tracker
            that stays at or below this value scores 1.0 on the memory
            component; trackers exceeding it score 0.
        energy_budget_mj: Per-frame energy budget in milli-Joules.  Set to
            ``None`` to disable the energy component.  Meaningful only when
            the benchmark was run with ``tdp_watts`` configured.
        w_accuracy: Weight for the IoU accuracy component.
        w_speed: Weight for the FPS throughput component.
        w_memory: Weight for the memory efficiency component.
        w_energy: Weight for the energy efficiency component.  Only active
            when energy data are available and ``energy_budget_mj`` is set.
    """

    target_fps: float = 30.0
    memory_budget_mb: float = 512.0
    energy_budget_mj: Optional[float] = None

    w_accuracy: float = 0.40
    w_speed: float = 0.30
    w_memory: float = 0.20
    w_energy: float = 0.10


@dataclass
class EdgeScoreResult:
    """Per-tracker edge-score with full component breakdown.

    All component values are in ``[0, 1]``; higher is better.
    The ``edge_score`` field is the weighted, normalised composite.

    Attributes:
        tracker: Tracker identifier string.
        edge_score: Final composite score in ``[0, 1]`` (higher is better).
        accuracy_component: Normalised IoU contribution.
        speed_component: Normalised FPS contribution (capped at 1.0 when
            FPS exceeds ``target_fps``).
        memory_component: ``max(0, 1 - memory_mb / memory_budget_mb)``.
        energy_component: ``max(0, 1 - energy_mj / energy_budget_mj)``,
            or ``None`` when energy data or budget are unavailable.
        mean_iou: Raw mean IoU value.
        fps: Raw mean FPS value.
        memory_mb: Raw peak memory in MiB.
        energy_mj: Raw mean energy per frame in mJ, or ``None``.
    """

    tracker: str
    edge_score: float
    accuracy_component: float
    speed_component: float
    memory_component: float
    energy_component: Optional[float]
    mean_iou: float
    fps: float
    memory_mb: float
    energy_mj: Optional[float]

    def __str__(self) -> str:
        e_str = (
            f"  energy_comp={self.energy_component:.3f}"
            if self.energy_component is not None
            else ""
        )
        return (
            f"EdgeScoreResult[{self.tracker}] "
            f"edge_score={self.edge_score:.4f}  "
            f"(acc={self.accuracy_component:.3f}  "
            f"spd={self.speed_component:.3f}  "
            f"mem={self.memory_component:.3f}"
            f"{e_str})"
        )


def compute_edge_score(
    summary: Dict[str, Any],
    config: EdgeScoreConfig,
) -> EdgeScoreResult:
    """Compute an :class:`EdgeScoreResult` from a benchmark summary dict.

    Args:
        summary: The ``"summary"`` sub-dict returned by
            :meth:`~eovot.benchmark.engine.BenchmarkResult.to_dict`.
            Recognised keys: ``"tracker"``, ``"tracker_name"``,
            ``"mean_iou"``, ``"mean_fps"``, ``"peak_memory_mb"``,
            ``"mean_energy_per_frame_mj"``.
        config: :class:`EdgeScoreConfig` controlling weights and device
            budgets.

    Returns:
        :class:`EdgeScoreResult` with the composite score and each
        normalised component.
    """
    tracker = summary.get("tracker") or summary.get("tracker_name", "?")
    iou = float(summary.get("mean_iou", 0.0))
    fps = float(summary.get("mean_fps", 0.0))
    memory_mb = float(summary.get("peak_memory_mb", 0.0))
    energy_mj_raw = summary.get("mean_energy_per_frame_mj")
    energy_mj: Optional[float] = float(energy_mj_raw) if energy_mj_raw is not None else None

    # Normalised components, each in [0, 1].
    acc_comp = float(max(0.0, min(1.0, iou)))

    if config.target_fps > 0:
        spd_comp = float(min(fps / config.target_fps, 1.0))
    else:
        spd_comp = 0.0

    if config.memory_budget_mb > 0:
        mem_comp = float(max(0.0, 1.0 - memory_mb / config.memory_budget_mb))
    else:
        mem_comp = 0.0

    use_energy = (
        energy_mj is not None
        and config.energy_budget_mj is not None
        and config.energy_budget_mj > 0
    )
    if use_energy:
        assert energy_mj is not None and config.energy_budget_mj is not None
        e_comp: Optional[float] = float(
            max(0.0, 1.0 - energy_mj / config.energy_budget_mj)
        )
    else:
        e_comp = None

    # Normalise weights so they sum to 1 (drop energy weight when unused).
    w_acc = float(config.w_accuracy)
    w_spd = float(config.w_speed)
    w_mem = float(config.w_memory)
    w_ene = float(config.w_energy) if use_energy else 0.0
    total_w = w_acc + w_spd + w_mem + w_ene
    if total_w <= 0.0:
        total_w = 1.0

    score = (
        w_acc * acc_comp
        + w_spd * spd_comp
        + w_mem * mem_comp
        + w_ene * (e_comp or 0.0)
    ) / total_w

    return EdgeScoreResult(
        tracker=tracker,
        edge_score=round(float(score), 6),
        accuracy_component=round(acc_comp, 6),
        speed_component=round(spd_comp, 6),
        memory_component=round(mem_comp, 6),
        energy_component=round(e_comp, 6) if e_comp is not None else None,
        mean_iou=round(iou, 6),
        fps=round(fps, 4),
        memory_mb=round(memory_mb, 4),
        energy_mj=round(energy_mj, 6) if energy_mj is not None else None,
    )


def rank_by_edge_score(
    results: List[Dict[str, Any]],
    config: Optional[EdgeScoreConfig] = None,
) -> List[EdgeScoreResult]:
    """Compute and rank edge scores for a list of benchmark result dicts.

    Args:
        results: List of result dicts, each containing a ``"summary"``
            sub-dict as produced by
            :meth:`~eovot.benchmark.engine.BenchmarkResult.to_dict`.
            Bare summary dicts (without the ``"summary"`` wrapper) are also
            accepted for convenience.
        config: :class:`EdgeScoreConfig`.  Defaults to
            ``EdgeScoreConfig()`` (equal-ish weights, 30 fps / 512 MB
            budgets) when ``None``.

    Returns:
        List of :class:`EdgeScoreResult` sorted by ``edge_score``
        descending (best tracker first).
    """
    if config is None:
        config = EdgeScoreConfig()
    summaries = [r.get("summary", r) for r in results]
    scores = [compute_edge_score(s, config) for s in summaries]
    scores.sort(key=lambda x: x.edge_score, reverse=True)
    return scores


def edge_score_leaderboard_md(
    results: List[Dict[str, Any]],
    config: Optional[EdgeScoreConfig] = None,
    title: str = "EOVOT Edge-Score Leaderboard",
) -> str:
    """Generate a Markdown leaderboard table ranked by edge score.

    Args:
        results: List of result dicts (same format as
            :func:`rank_by_edge_score`).
        config: Scoring configuration.  Defaults to ``EdgeScoreConfig()``.
        title: Markdown heading text for the leaderboard section.

    Returns:
        Multi-line Markdown string ready for appending to a ``.md`` file.
    """
    if config is None:
        config = EdgeScoreConfig()

    if not results:
        return "No results to display.\n"

    ranked = rank_by_edge_score(results, config)
    has_energy = any(r.energy_component is not None for r in ranked)

    header = ["Rank", "Tracker", "EdgeScore", "mIoU", "FPS", "Mem (MB)",
               "Acc ▸", "Spd ▸", "Mem ▸"]
    sep = [":---:", ":---", "---:", "---:", "---:", "---:",
           "---:", "---:", "---:"]
    if has_energy:
        header += ["Nrg ▸", "E (mJ/fr)"]
        sep += ["---:", "---:"]

    budget_info = (
        f"> Config — target_fps={config.target_fps} fps, "
        f"memory_budget={config.memory_budget_mb} MB"
    )
    if config.energy_budget_mj is not None:
        budget_info += f", energy_budget={config.energy_budget_mj} mJ/fr"
    budget_info += "\n"

    lines = [
        f"\n## {title}\n",
        budget_info,
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
    ]

    for rank, r in enumerate(ranked, start=1):
        row = (
            f"| {rank} | {r.tracker} | **{r.edge_score:.4f}** "
            f"| {r.mean_iou:.4f} | {r.fps:.1f} | {r.memory_mb:.1f} "
            f"| {r.accuracy_component:.3f} | {r.speed_component:.3f} "
            f"| {r.memory_component:.3f}"
        )
        if has_energy:
            e_c = f"{r.energy_component:.3f}" if r.energy_component is not None else "N/A"
            e_j = f"{r.energy_mj:.3f}" if r.energy_mj is not None else "N/A"
            row += f" | {e_c} | {e_j}"
        row += " |"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)

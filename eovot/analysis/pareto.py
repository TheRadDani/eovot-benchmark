"""Pareto-optimal frontier analysis for edge deployment trade-off evaluation.

Tracker selection for edge deployment involves simultaneous optimisation
across at least four conflicting objectives:

- **Accuracy** (IoU) — maximise
- **Throughput** (FPS) — maximise
- **Latency** (ms/frame) — minimise
- **Memory** (MB) — minimise
- **Energy** (mJ/frame) — minimise (when profiled)

No single tracker dominates all others across all objectives; practitioners
must choose based on their deployment constraints.  This module provides:

1. **Pareto front detection** — identifies the set of non-dominated trackers
   so that users know which trade-offs are worth considering.
2. **Composite edge score** — a single weighted score that aggregates all
   objectives given a specific hardware budget, enabling ranked comparison.
3. **ParetoAnalyzer** — loads benchmark JSON results, runs both analyses,
   and renders console or Markdown leaderboards.

Example::

    analyzer = ParetoAnalyzer(fps_target=30.0, memory_budget_mb=512.0)
    profiles = analyzer.load_from_json("results/comparison.json")
    result   = analyzer.analyze(profiles)
    analyzer.print_leaderboard(result)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class TrackerProfile:
    """Multi-dimensional performance profile for one tracker.

    All fields are derived from :meth:`BenchmarkResult.to_dict` / ``summary``
    output produced by :class:`~eovot.benchmark.engine.BenchmarkEngine`.

    Attributes:
        name:           Tracker identifier.
        mean_iou:       Mean IoU across all evaluated frames [0, 1].
        fps:            Mean throughput in frames per second.
        latency_ms:     Mean per-frame latency in milliseconds.
        memory_mb:      Peak memory usage in megabytes.
        energy_mj:      Mean energy per frame in milli-Joules (0 = not measured).
        success_auc:    AUC of the IoU success curve [0, 1].
        precision_auc:  Normalised AUC of the centre-distance precision curve [0, 1].
    """

    name: str
    mean_iou: float
    fps: float
    latency_ms: float
    memory_mb: float
    energy_mj: float = 0.0
    success_auc: float = 0.0
    precision_auc: float = 0.0

    def __str__(self) -> str:
        return (
            f"TrackerProfile({self.name}: "
            f"IoU={self.mean_iou:.3f}, FPS={self.fps:.1f}, "
            f"latency={self.latency_ms:.1f}ms, mem={self.memory_mb:.0f}MB)"
        )


@dataclass
class ParetoResult:
    """Output of a Pareto analysis run.

    Attributes:
        profiles:     All tracker profiles that were analysed.
        pareto_front: Names of Pareto-optimal (non-dominated) trackers.
        rankings:     Tracker names → composite edge score, sorted descending.
        dominated:    Names of trackers that are dominated by at least one other.
    """

    profiles: List[TrackerProfile]
    pareto_front: List[str]
    rankings: Dict[str, float]
    dominated: List[str]


# ---------------------------------------------------------------------------
# Pareto front computation
# ---------------------------------------------------------------------------


def _dominates(q: TrackerProfile, p: TrackerProfile) -> bool:
    """Return True if *q* Pareto-dominates *p*.

    *q* dominates *p* when *q* is at least as good as *p* in every objective
    and strictly better in at least one.  Objectives:

    * Maximise: ``mean_iou``, ``fps``
    * Minimise: ``latency_ms``, ``memory_mb``

    Energy is excluded from Pareto comparison because it is often unavailable
    (value 0.0); accuracy/speed/memory are universally measured.

    Args:
        q: Potential dominator.
        p: Candidate being checked.

    Returns:
        True if q dominates p.
    """
    at_least_as_good = (
        q.mean_iou >= p.mean_iou
        and q.fps >= p.fps
        and q.latency_ms <= p.latency_ms
        and q.memory_mb <= p.memory_mb
    )
    if not at_least_as_good:
        return False
    strictly_better = (
        q.mean_iou > p.mean_iou
        or q.fps > p.fps
        or q.latency_ms < p.latency_ms
        or q.memory_mb < p.memory_mb
    )
    return strictly_better


def compute_pareto_front(profiles: List[TrackerProfile]) -> List[str]:
    """Identify the Pareto-optimal subset of tracker profiles.

    A tracker is on the Pareto front if and only if no other tracker
    dominates it (i.e., is at least as good in *all* objectives and
    strictly better in at least one).

    Args:
        profiles: List of tracker profiles to evaluate.

    Returns:
        Names of non-dominated trackers, in their original order.
    """
    pareto: List[str] = []
    for i, p in enumerate(profiles):
        if not any(_dominates(profiles[j], p) for j in range(len(profiles)) if j != i):
            pareto.append(p.name)
    return pareto


# ---------------------------------------------------------------------------
# Composite edge score
# ---------------------------------------------------------------------------


def compute_edge_score(
    profile: TrackerProfile,
    w_accuracy: float = 0.40,
    w_speed: float = 0.30,
    w_memory: float = 0.20,
    w_energy: float = 0.10,
    fps_target: float = 30.0,
    memory_budget_mb: float = 512.0,
    energy_budget_mj: float = 100.0,
) -> float:
    """Compute a composite edge-deployment score for a tracker.

    Each objective is normalised to ``[0, 1]`` relative to the given hardware
    budget, then combined as a weighted sum.  A score of 1.0 means the tracker
    achieves perfect accuracy, meets the FPS target, uses zero memory, and
    consumes zero energy.

    When energy is not profiled (``profile.energy_mj == 0``), its weight is
    redistributed to accuracy so the total still sums to 1.

    Args:
        profile:            Tracker performance profile.
        w_accuracy:         Accuracy weight (default 0.40).
        w_speed:            Speed / throughput weight (default 0.30).
        w_memory:           Memory efficiency weight (default 0.20).
        w_energy:           Energy efficiency weight (default 0.10).
        fps_target:         Target FPS for normalisation (default 30.0).
        memory_budget_mb:   Memory ceiling in MB (default 512.0).
        energy_budget_mj:   Energy ceiling per frame in mJ (default 100.0).

    Returns:
        Composite edge score in ``[0, 1]``, higher is better.
    """
    accuracy_score = float(np.clip(profile.mean_iou, 0.0, 1.0))
    speed_score = float(np.clip(profile.fps / max(fps_target, 1e-6), 0.0, 1.0))
    memory_score = float(
        np.clip(1.0 - profile.memory_mb / max(memory_budget_mb, 1e-6), 0.0, 1.0)
    )

    # Graceful degradation when energy is not measured
    if profile.energy_mj > 0.0 and w_energy > 0.0:
        energy_score = float(
            np.clip(1.0 - profile.energy_mj / max(energy_budget_mj, 1e-6), 0.0, 1.0)
        )
    else:
        # Redistribute energy weight to accuracy
        w_accuracy = w_accuracy + w_energy
        w_energy = 0.0
        energy_score = 0.0

    total_weight = w_accuracy + w_speed + w_memory + w_energy
    if total_weight <= 0.0:
        return 0.0

    score = (
        w_accuracy * accuracy_score
        + w_speed * speed_score
        + w_memory * memory_score
        + w_energy * energy_score
    ) / total_weight

    return float(score)


def rank_trackers(
    profiles: List[TrackerProfile],
    **edge_score_kwargs: Any,
) -> Dict[str, float]:
    """Rank trackers by composite edge score, descending.

    Args:
        profiles:             Tracker profiles to rank.
        **edge_score_kwargs:  Forwarded to :func:`compute_edge_score`.

    Returns:
        ``{tracker_name: edge_score}`` sorted by score descending.
    """
    scores = {p.name: compute_edge_score(p, **edge_score_kwargs) for p in profiles}
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


# ---------------------------------------------------------------------------
# High-level analyser
# ---------------------------------------------------------------------------


class ParetoAnalyzer:
    """Load benchmark results, run Pareto analysis, and generate leaderboards.

    Example::

        analyzer = ParetoAnalyzer(fps_target=30.0, memory_budget_mb=256.0)
        profiles = analyzer.load_from_json("results/comparison.json")
        result   = analyzer.analyze(profiles)
        analyzer.print_leaderboard(result)
        md = analyzer.to_markdown(result)

    Args:
        **edge_score_kwargs: Keyword arguments forwarded to
            :func:`compute_edge_score` (e.g., ``fps_target``,
            ``memory_budget_mb``, ``w_accuracy``).
    """

    def __init__(self, **edge_score_kwargs: Any) -> None:
        self._score_kwargs = edge_score_kwargs

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_from_json(self, path: str) -> List[TrackerProfile]:
        """Load tracker profiles from a benchmark JSON result file.

        Accepts both single-tracker output and multi-tracker comparison
        dicts, as produced by :class:`~eovot.reporting.reporter.BenchmarkReporter`.

        Args:
            path: Path to a JSON file written by BenchmarkReporter.

        Returns:
            List of :class:`TrackerProfile` objects.

        Raises:
            FileNotFoundError: If *path* does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        with open(path) as fh:
            data = json.load(fh)

        if isinstance(data, dict) and "tracker" in data:
            # Single-tracker result: {"tracker": "...", "summary": {...}, ...}
            return [self._parse_entry(data)]

        if isinstance(data, dict):
            # Multi-tracker: {tracker_name: {result_dict}, ...}
            profiles = []
            for name, entry in data.items():
                if isinstance(entry, dict):
                    profiles.append(self._parse_entry(entry, name_override=name))
            return profiles

        return []

    def load_from_dict(self, data: Dict[str, Any]) -> List[TrackerProfile]:
        """Load profiles from an in-memory result dict.

        Useful for integration with :class:`~eovot.benchmark.engine.BenchmarkEngine`
        without writing to disk.

        Args:
            data: Dict in the format produced by ``BenchmarkResult.to_dict()``.

        Returns:
            List of :class:`TrackerProfile` objects (typically one).
        """
        if "tracker" in data:
            return [self._parse_entry(data)]
        profiles = []
        for name, entry in data.items():
            if isinstance(entry, dict):
                profiles.append(self._parse_entry(entry, name_override=name))
        return profiles

    def _parse_entry(
        self,
        data: Dict[str, Any],
        name_override: Optional[str] = None,
    ) -> TrackerProfile:
        """Convert a raw result dict into a TrackerProfile."""
        name = name_override or str(data.get("tracker", "unknown"))
        # Prefer nested "summary" key; fall back to top-level
        summary: Dict[str, Any] = data.get("summary", data)
        return TrackerProfile(
            name=name,
            mean_iou=float(summary.get("mean_iou", 0.0)),
            fps=float(summary.get("mean_fps", 0.0)),
            latency_ms=float(summary.get("mean_latency_ms", 0.0)),
            memory_mb=float(
                summary.get("peak_memory_mb", summary.get("mean_memory_mb", 0.0))
            ),
            energy_mj=float(summary.get("mean_energy_per_frame_mj", 0.0)),
            success_auc=float(summary.get("success_auc", 0.0)),
            precision_auc=float(summary.get("precision_auc", 0.0)),
        )

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze(self, profiles: List[TrackerProfile]) -> ParetoResult:
        """Run Pareto analysis on a collection of tracker profiles.

        Args:
            profiles: Tracker profiles to analyse.

        Returns:
            :class:`ParetoResult` with Pareto front, ranked scores, and
            list of dominated trackers.
        """
        if not profiles:
            return ParetoResult(
                profiles=[],
                pareto_front=[],
                rankings={},
                dominated=[],
            )

        pareto_front = compute_pareto_front(profiles)
        rankings = rank_trackers(profiles, **self._score_kwargs)
        dominated = [p.name for p in profiles if p.name not in pareto_front]

        return ParetoResult(
            profiles=profiles,
            pareto_front=pareto_front,
            rankings=rankings,
            dominated=dominated,
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def print_leaderboard(self, result: ParetoResult) -> None:
        """Print a formatted leaderboard table to stdout.

        Pareto-optimal trackers are marked with ``*`` in the last column.

        Args:
            result: :class:`ParetoResult` from :meth:`analyze`.
        """
        sep = "=" * 74
        print(f"\n{sep}")
        print("EOVOT — Edge Deployment Leaderboard")
        print(sep)
        print(
            f"{'Rank':<5} {'Tracker':<22} {'Score':>6} "
            f"{'IoU':>6} {'FPS':>7} {'Mem MB':>7} {'Pareto':>7}"
        )
        print("-" * 74)
        for rank, (name, score) in enumerate(result.rankings.items(), start=1):
            prof = next((p for p in result.profiles if p.name == name), None)
            if prof is None:
                continue
            flag = "*" if name in result.pareto_front else ""
            print(
                f"{rank:<5} {name:<22} {score:>6.3f} "
                f"{prof.mean_iou:>6.3f} {prof.fps:>7.1f} "
                f"{prof.memory_mb:>7.1f} {flag:>7}"
            )
        print("-" * 74)
        print(
            f"* Pareto-optimal  "
            f"({len(result.pareto_front)}/{len(result.profiles)} trackers)"
        )
        print(f"{sep}\n")

    def to_markdown(self, result: ParetoResult) -> str:
        """Export the leaderboard as a Markdown table.

        Suitable for inclusion in research papers and GitHub wikis.

        Args:
            result: :class:`ParetoResult` from :meth:`analyze`.

        Returns:
            Markdown string with a formatted leaderboard table.
        """
        lines = [
            "## EOVOT — Edge Deployment Leaderboard",
            "",
            "| Rank | Tracker | Edge Score | IoU | FPS | Memory (MB) | Pareto |",
            "|------|---------|:----------:|:---:|:---:|:-----------:|:------:|",
        ]
        for rank, (name, score) in enumerate(result.rankings.items(), start=1):
            prof = next((p for p in result.profiles if p.name == name), None)
            if prof is None:
                continue
            flag = "✓" if name in result.pareto_front else ""
            lines.append(
                f"| {rank} | `{name}` | {score:.3f} | {prof.mean_iou:.3f} | "
                f"{prof.fps:.1f} | {prof.memory_mb:.1f} | {flag} |"
            )
        return "\n".join(lines)

    def to_json(self, result: ParetoResult) -> str:
        """Serialise the leaderboard to JSON.

        Args:
            result: :class:`ParetoResult` from :meth:`analyze`.

        Returns:
            JSON string with rankings, Pareto front, and per-tracker profiles.
        """
        output = {
            "pareto_front": result.pareto_front,
            "dominated": result.dominated,
            "rankings": [
                {
                    "rank": rank,
                    "name": name,
                    "edge_score": score,
                    "pareto_optimal": name in result.pareto_front,
                    "mean_iou": next(
                        (p.mean_iou for p in result.profiles if p.name == name), 0.0
                    ),
                    "fps": next(
                        (p.fps for p in result.profiles if p.name == name), 0.0
                    ),
                    "memory_mb": next(
                        (p.memory_mb for p in result.profiles if p.name == name), 0.0
                    ),
                    "energy_mj": next(
                        (p.energy_mj for p in result.profiles if p.name == name), 0.0
                    ),
                }
                for rank, (name, score) in enumerate(result.rankings.items(), start=1)
            ],
        }
        return json.dumps(output, indent=2)

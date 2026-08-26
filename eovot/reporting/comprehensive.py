"""Comprehensive multi-metric leaderboard for EOVOT.

Combines the three evaluation dimensions that previously lived in separate,
disconnected modules into one unified ranking:

- **Accuracy** — success AUC and mean IoU from
  :class:`~eovot.metrics.accuracy.MetricsEngine`
- **Robustness** — Expected Average Overlap (EAO), failure count, and
  survival rate from :class:`~eovot.metrics.robustness.RobustnessAnalyzer`
- **Efficiency** — Edge Efficiency Score (EES), Pareto optimality, and
  FPS from :class:`~eovot.metrics.efficiency.EfficiencyMetricsEngine`

The three dimensions are combined into a single **Composite Score**::

    CompositeScore = w_acc × success_AUC
                   + w_rob × EAO
                   + w_eff × normalize(EES)

where ``w_acc + w_rob + w_eff = 1`` and EES is normalised to ``[0, 1]``
across the comparison set so the score is always bounded.

The leaderboard is designed to answer the key EOVOT research question:
"Which tracker is best for *edge deployment*, accounting for accuracy,
failure behaviour, and hardware constraints simultaneously?"

Typical usage::

    from eovot.reporting.comprehensive import ComprehensiveLeaderboard

    lb = ComprehensiveLeaderboard(memory_budget_mb=512.0)
    entries = lb.rank(benchmark_results)         # list[LeaderboardEntry]
    print(lb.to_markdown(entries))
    lb.save_json(entries, "results/leaderboard.json")
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union, TYPE_CHECKING

import numpy as np

from ..metrics.robustness import RobustnessAnalyzer
from ..metrics.efficiency import EfficiencyMetricsEngine

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult


@dataclass
class LeaderboardEntry:
    """Per-tracker summary across all three evaluation dimensions.

    Attributes:
        tracker_name: Human-readable tracker identifier.
        dataset_name: Dataset on which the tracker was evaluated.
        mean_iou: Mean IoU across all evaluated frames.
        success_auc: Area under the Success Curve (primary accuracy scalar).
        precision_auc: Normalised AUC of the Precision Curve.
        eao: Expected Average Overlap — simplified robustness/accuracy scalar.
        num_failures: Total failure events detected across all sequences.
        mean_failures_per_sequence: Average failure count per sequence.
        mean_survival_rate: Fraction of non-burn-in frames with IoU ≥ threshold.
        fps: Mean throughput in frames-per-second.
        peak_memory_mb: Peak RSS memory footprint in megabytes.
        ees: Edge Efficiency Score (accuracy × log(FPS) / memory penalty).
        on_pareto_front: True when no other tracker in the set dominates
            this one in both accuracy and EES simultaneously.
        composite_score: Weighted combination of success_AUC, EAO, and
            normalised EES.  Higher is better.  Range: ``[0, 1]`` when all
            components are in ``[0, 1]``.
        rank: Final rank in the leaderboard (1 = best composite score).
    """

    tracker_name: str
    dataset_name: str
    mean_iou: float = 0.0
    success_auc: float = 0.0
    precision_auc: float = 0.0
    eao: float = 0.0
    num_failures: int = 0
    mean_failures_per_sequence: float = 0.0
    mean_survival_rate: float = 0.0
    fps: float = 0.0
    peak_memory_mb: float = 0.0
    ees: float = 0.0
    on_pareto_front: bool = False
    composite_score: float = 0.0
    rank: int = 0

    def to_dict(self) -> Dict:
        """Return a JSON-serialisable plain dict."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d

    def __str__(self) -> str:
        pareto = "✓" if self.on_pareto_front else " "
        return (
            f"[{self.rank:>2}] {self.tracker_name:<16s} "
            f"Comp={self.composite_score:.4f}  "
            f"AUC={self.success_auc:.4f}  EAO={self.eao:.4f}  "
            f"EES={self.ees:.4f}  FPS={self.fps:.1f}  "
            f"Fail/seq={self.mean_failures_per_sequence:.2f}  "
            f"Pareto={pareto}"
        )


class ComprehensiveLeaderboard:
    """Unified ranking across accuracy, robustness, and efficiency.

    Args:
        memory_budget_mb: Acceptable peak-memory ceiling used by the EES
            formula.  Trackers within this budget receive full memory credit.
            Default: ``512.0`` MB.
        accuracy_weight: Weight for success AUC in the composite score.
            Default: ``0.5``.
        robustness_weight: Weight for EAO in the composite score.
            Default: ``0.3``.
        efficiency_weight: Weight for normalised EES in the composite score.
            Default: ``0.2``.
        failure_threshold: IoU threshold for the robustness analyser's
            failure detection.  Default: ``0.1`` (standard VOT).
        burn_in_frames: Frames to skip before counting failures.
            Default: ``5``.

    Raises:
        ValueError: If weights do not sum to 1 (within 1e-6 tolerance).
    """

    def __init__(
        self,
        memory_budget_mb: float = 512.0,
        accuracy_weight: float = 0.5,
        robustness_weight: float = 0.3,
        efficiency_weight: float = 0.2,
        failure_threshold: float = 0.1,
        burn_in_frames: int = 5,
    ) -> None:
        total = accuracy_weight + robustness_weight + efficiency_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"accuracy_weight + robustness_weight + efficiency_weight must sum to 1.0, "
                f"got {total:.6f}."
            )
        self.memory_budget_mb = memory_budget_mb
        self.accuracy_weight = accuracy_weight
        self.robustness_weight = robustness_weight
        self.efficiency_weight = efficiency_weight
        self._robustness = RobustnessAnalyzer(
            failure_threshold=failure_threshold,
            burn_in_frames=burn_in_frames,
        )
        self._efficiency = EfficiencyMetricsEngine(memory_budget_mb=memory_budget_mb)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def rank(self, results: List["BenchmarkResult"]) -> List[LeaderboardEntry]:
        """Build a ranked leaderboard from a list of benchmark results.

        Each entry in ``results`` represents one tracker evaluated on one
        dataset.  The method computes robustness and efficiency metrics,
        combines them with the stored accuracy metrics, and sorts by
        composite score descending.

        Args:
            results: One :class:`~eovot.benchmark.engine.BenchmarkResult`
                per tracker/dataset combination.

        Returns:
            List of :class:`LeaderboardEntry` sorted by composite score
            (highest first), with ``rank`` and ``on_pareto_front`` populated.
        """
        if not results:
            return []

        entries: List[LeaderboardEntry] = []
        for r in results:
            entry = self._build_entry(r)
            entries.append(entry)

        # Mark Pareto-optimal trackers in (accuracy, EES) space
        self._mark_pareto(entries)

        # Normalise EES to [0, 1] across the comparison set
        ees_vals = [e.ees for e in entries]
        ees_max = max(ees_vals) if ees_vals else 1.0
        ees_min = min(ees_vals) if ees_vals else 0.0
        ees_range = max(ees_max - ees_min, 1e-9)

        for e in entries:
            norm_ees = (e.ees - ees_min) / ees_range
            e.composite_score = (
                self.accuracy_weight * e.success_auc
                + self.robustness_weight * e.eao
                + self.efficiency_weight * norm_ees
            )

        entries.sort(key=lambda x: x.composite_score, reverse=True)
        for rank, e in enumerate(entries, start=1):
            e.rank = rank

        return entries

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def to_markdown(self, entries: List[LeaderboardEntry]) -> str:
        """Format the leaderboard as a Markdown table.

        The table groups columns by dimension:
        - **Accuracy**: success AUC, mIoU
        - **Robustness**: EAO, failures/seq, survival rate
        - **Efficiency**: FPS, memory (MB), EES, Pareto flag
        - **Overall**: composite score

        Args:
            entries: Output of :meth:`rank`.

        Returns:
            Multi-line Markdown string.
        """
        if not entries:
            return "No results to display.\n"

        w_acc = self.accuracy_weight
        w_rob = self.robustness_weight
        w_eff = self.efficiency_weight
        lines = [
            "# EOVOT Comprehensive Leaderboard\n",
            f"> Weights — Accuracy: {w_acc:.0%}  "
            f"| Robustness: {w_rob:.0%}  "
            f"| Efficiency: {w_eff:.0%}  "
            f"| Memory budget: {self.memory_budget_mb:.0f} MB\n",
            "| Rank | Tracker | Dataset "
            "| Composite ↓ "
            "| AUC | mIoU "
            "| EAO | Fail/seq | Survival "
            "| FPS | Mem (MB) | EES | Pareto |",
            "|------|---------|---------|"
            "-----------:|"
            "----:|-----:|"
            "----:|---------:|--------:|"
            "----:|---------:|----:|:------:|",
        ]
        for e in entries:
            pareto = "✓" if e.on_pareto_front else ""
            lines.append(
                f"| {e.rank} | {e.tracker_name} | {e.dataset_name} "
                f"| **{e.composite_score:.4f}** "
                f"| {e.success_auc:.4f} | {e.mean_iou:.4f} "
                f"| {e.eao:.4f} | {e.mean_failures_per_sequence:.2f} "
                f"| {e.mean_survival_rate:.3f} "
                f"| {e.fps:.1f} | {e.peak_memory_mb:.1f} "
                f"| {e.ees:.4f} | {pareto} |"
            )
        lines.append("")
        return "\n".join(lines)

    def save_json(
        self,
        entries: List[LeaderboardEntry],
        path: Union[str, Path],
    ) -> Path:
        """Serialise the leaderboard to a JSON file.

        Args:
            entries: Output of :meth:`rank`.
            path:    Destination file path.  Parent directories are created
                     automatically.  A ``.json`` extension is appended when
                     the path has none.

        Returns:
            The resolved :class:`pathlib.Path` that was written.
        """
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".json")
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "weights": {
                "accuracy": self.accuracy_weight,
                "robustness": self.robustness_weight,
                "efficiency": self.efficiency_weight,
            },
            "memory_budget_mb": self.memory_budget_mb,
            "entries": [e.to_dict() for e in entries],
        }
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        return p

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_entry(self, result: "BenchmarkResult") -> LeaderboardEntry:
        """Compute all metrics for one benchmark result."""
        # --- Accuracy (already stored in BenchmarkResult) ---
        success_auc = result.mean_success_auc or result.mean_iou
        precision_auc = result.mean_precision_auc or 0.0

        # --- Robustness: aggregate RobustnessAnalyzer over all sequences ---
        seq_ious: Dict[str, "np.ndarray"] = {}
        for sr in result.sequence_results:
            if sr.ious is not None and len(sr.ious) > 0:
                seq_ious[sr.sequence_name] = sr.ious

        rob_agg: Dict = {}
        if seq_ious:
            rob_out = self._robustness.analyze_benchmark(
                seq_ious, tracker_name=result.tracker_name
            )
            rob_agg = rob_out.get("aggregate", {})

        eao = float(rob_agg.get("mean_eao", result.mean_iou))
        num_failures = int(rob_agg.get("total_failures", 0))
        fail_per_seq = float(rob_agg.get("mean_failures_per_sequence", 0.0))
        survival = float(rob_agg.get("mean_survival_rate", 0.0))

        # --- Efficiency ---
        ees = self._efficiency.edge_efficiency_score(
            mean_iou=result.mean_iou,
            fps=result.mean_fps,
            peak_memory_mb=result.peak_memory_mb,
        )

        return LeaderboardEntry(
            tracker_name=result.tracker_name,
            dataset_name=result.dataset_name,
            mean_iou=round(result.mean_iou, 4),
            success_auc=round(success_auc, 4),
            precision_auc=round(precision_auc, 4),
            eao=round(eao, 4),
            num_failures=num_failures,
            mean_failures_per_sequence=round(fail_per_seq, 3),
            mean_survival_rate=round(survival, 4),
            fps=round(result.mean_fps, 2),
            peak_memory_mb=round(result.peak_memory_mb, 2),
            ees=round(ees, 6),
        )

    @staticmethod
    def _mark_pareto(entries: List[LeaderboardEntry]) -> None:
        """Set ``on_pareto_front`` in-place in the (accuracy, EES) space."""
        for i, candidate in enumerate(entries):
            dominated = any(
                other.success_auc >= candidate.success_auc
                and other.ees >= candidate.ees
                and (other.success_auc > candidate.success_auc or other.ees > candidate.ees)
                for j, other in enumerate(entries)
                if j != i
            )
            candidate.on_pareto_front = not dominated

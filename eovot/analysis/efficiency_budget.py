"""Hardware budget constraint analysis for edge-device tracker deployment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class HardwareBudget:
    """Constraints that a tracker must satisfy on a target edge device."""

    min_fps: Optional[float] = None
    max_latency_ms: Optional[float] = None
    max_latency_p95_ms: Optional[float] = None
    max_memory_mb: Optional[float] = None
    max_energy_mj_per_frame: Optional[float] = None
    label: str = "target device"

    def is_satisfied_by(self, result) -> Tuple[bool, Dict[str, str]]:
        """Return (feasible, violations) for a BenchmarkResult-like object.

        The result object must expose: fps, latency_ms, latency_p95_ms,
        memory_mb, energy_mj_per_frame as numeric attributes (None -> skip).
        """
        violations: Dict[str, str] = {}

        def _check(limit, value, fmt, key):
            if limit is None or value is None:
                return
            if value > limit:
                violations[key] = fmt.format(value=value, limit=limit)

        if self.min_fps is not None:
            fps = getattr(result, "fps", None)
            if fps is not None and fps < self.min_fps:
                violations["fps"] = (
                    f"fps={fps:.1f} < min_fps={self.min_fps:.1f}"
                )

        _check(
            self.max_latency_ms,
            getattr(result, "latency_ms", None),
            "latency_ms={value:.2f} > max={limit:.2f}",
            "latency_ms",
        )
        _check(
            self.max_latency_p95_ms,
            getattr(result, "latency_p95_ms", None),
            "latency_p95_ms={value:.2f} > max={limit:.2f}",
            "latency_p95_ms",
        )
        _check(
            self.max_memory_mb,
            getattr(result, "memory_mb", None),
            "memory_mb={value:.1f} > max={limit:.1f}",
            "memory_mb",
        )
        _check(
            self.max_energy_mj_per_frame,
            getattr(result, "energy_mj_per_frame", None),
            "energy_mj={value:.4f} > max={limit:.4f}",
            "energy_mj_per_frame",
        )

        return len(violations) == 0, violations


@dataclass
class BudgetEntry:
    """Per-tracker feasibility record produced by BudgetAnalyzer."""

    tracker_name: str
    dataset_name: str
    mean_iou: float
    success_auc: float
    fps: Optional[float]
    memory_mb: Optional[float]
    energy_mj_per_frame: Optional[float]
    satisfies: bool
    violations: Dict[str, str] = field(default_factory=dict)
    accuracy_rank: int = 0   # rank by mean_iou among all trackers (1 = best)
    budget_rank: int = 0     # rank by mean_iou among feasible trackers only


@dataclass
class BudgetReport:
    """Aggregated result of a single budget analysis run."""

    budget: HardwareBudget
    entries: List[BudgetEntry]
    feasible: List[BudgetEntry]
    infeasible: List[BudgetEntry]

    @property
    def best_tracker(self) -> Optional[str]:
        """Name of the most accurate tracker that satisfies the budget."""
        if not self.feasible:
            return None
        return max(self.feasible, key=lambda e: e.mean_iou).tracker_name

    @property
    def accuracy_loss_at_budget(self) -> Optional[float]:
        """IoU drop vs. unconstrained best; None when results list is empty."""
        if not self.entries:
            return None
        best_overall = max(self.entries, key=lambda e: e.mean_iou).mean_iou
        if not self.feasible:
            return best_overall
        best_feasible = max(self.feasible, key=lambda e: e.mean_iou).mean_iou
        return round(best_overall - best_feasible, 6)

    def to_markdown_table(self) -> str:
        """Render a Markdown table sorted by accuracy rank."""
        header = (
            "| Tracker | Dataset | IoU | AUC | FPS | Mem (MB) | "
            "Energy (mJ/f) | Feasible | Violations |\n"
            "|---------|---------|-----|-----|-----|----------|"
            "--------------|----------|------------|\n"
        )
        rows = []
        for e in sorted(self.entries, key=lambda x: x.accuracy_rank):
            fps_str = f"{e.fps:.1f}" if e.fps is not None else "—"
            mem_str = f"{e.memory_mb:.1f}" if e.memory_mb is not None else "—"
            energy_str = (
                f"{e.energy_mj_per_frame:.4f}"
                if e.energy_mj_per_frame is not None
                else "—"
            )
            feasible_str = "✓" if e.satisfies else "✗"
            violation_str = "; ".join(e.violations.values()) if e.violations else ""
            rows.append(
                f"| {e.tracker_name} | {e.dataset_name} | {e.mean_iou:.4f} | "
                f"{e.success_auc:.4f} | {fps_str} | {mem_str} | {energy_str} | "
                f"{feasible_str} | {violation_str} |"
            )
        return header + "\n".join(rows)


class BudgetAnalyzer:
    """Evaluate which trackers satisfy a hardware deployment budget.

    All configuration is passed per call, so one instance can be reused
    across multiple budgets and result sets.
    """

    def analyze(self, results, budget: HardwareBudget) -> BudgetReport:
        """Classify *results* against *budget* and return a BudgetReport.

        Parameters
        ----------
        results:
            Iterable of objects with tracker_name, dataset_name, mean_iou,
            success_auc, fps, memory_mb, energy_mj_per_frame attributes.
        budget:
            HardwareBudget with the target-device constraints.
        """
        entries: List[BudgetEntry] = []
        for r in results:
            satisfies, violations = budget.is_satisfied_by(r)
            entries.append(
                BudgetEntry(
                    tracker_name=getattr(r, "tracker_name", "unknown"),
                    dataset_name=getattr(r, "dataset_name", "unknown"),
                    mean_iou=float(getattr(r, "mean_iou", 0.0)),
                    success_auc=float(getattr(r, "success_auc", 0.0)),
                    fps=getattr(r, "fps", None),
                    memory_mb=getattr(r, "memory_mb", None),
                    energy_mj_per_frame=getattr(r, "energy_mj_per_frame", None),
                    satisfies=satisfies,
                    violations=violations,
                )
            )

        for rank, e in enumerate(
            sorted(entries, key=lambda e: e.mean_iou, reverse=True), start=1
        ):
            e.accuracy_rank = rank

        feasible = [e for e in entries if e.satisfies]
        infeasible = [e for e in entries if not e.satisfies]

        for rank, e in enumerate(
            sorted(feasible, key=lambda e: e.mean_iou, reverse=True), start=1
        ):
            e.budget_rank = rank

        return BudgetReport(
            budget=budget,
            entries=entries,
            feasible=feasible,
            infeasible=infeasible,
        )

    def sweep_budgets(
        self, results, budgets: Dict[str, HardwareBudget]
    ) -> Dict[str, BudgetReport]:
        """Run analyze() for each named budget and return a keyed mapping."""
        result_list = list(results)
        return {
            name: self.analyze(result_list, budget)
            for name, budget in budgets.items()
        }

    def accuracy_cost_curve(
        self, results, min_fps_values: List[float]
    ) -> List[Tuple[float, Optional[float], Optional[str]]]:
        """Compute the accuracy cost of increasing FPS constraints.

        Returns a list of (min_fps, accuracy_loss, best_tracker_name) tuples
        sorted ascending by min_fps. Useful for plotting the tradeoff curve.
        """
        result_list = list(results)
        curve = []
        for fps in sorted(min_fps_values):
            report = self.analyze(result_list, HardwareBudget(min_fps=fps))
            curve.append((fps, report.accuracy_loss_at_budget, report.best_tracker))
        return curve

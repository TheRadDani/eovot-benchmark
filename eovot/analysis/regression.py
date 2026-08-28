"""Benchmark regression detector for iterative tracker development.

When tuning tracker hyperparameters or changing an algorithm, you need to
know not just whether the aggregate metrics improved, but *which sequences*
improved and which regressed.  A tracker that gains 0.02 mIoU on average
might have regressed severely on 30 % of sequences — a pattern invisible in
aggregate tables.

This module compares two :class:`~eovot.benchmark.engine.BenchmarkResult`
objects (baseline vs. candidate) and produces:

- **Per-sequence delta table** — IoU change for every sequence, sorted by
  magnitude, colour-coded as improvement / regression / stable.
- **Regression summary** — count and fraction of sequences that regressed
  beyond a configurable threshold.
- **Speedup / slowdown summary** — FPS change between runs per sequence.
- **Overall verdict** — whether the candidate should be accepted based on
  aggregate accuracy and regression rate constraints.

Typical workflow::

    from eovot.analysis.regression import BenchmarkDiff

    baseline  = BenchmarkResult.load("results/kcf_baseline.json")
    candidate = BenchmarkResult.load("results/kcf_tuned.json")

    diff = BenchmarkDiff(regression_threshold=0.02, regression_rate_limit=0.20)
    report = diff.compare(baseline, candidate)

    print(report)                  # text summary
    print(report.to_markdown())    # Markdown table for GitHub PR body
    report.raise_if_regressed()    # raises RegressionError in CI pipelines
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult


class RegressionError(Exception):
    """Raised by :meth:`DiffReport.raise_if_regressed` when the candidate fails."""


# ---------------------------------------------------------------------------
# Per-sequence delta
# ---------------------------------------------------------------------------

@dataclass
class SequenceDelta:
    """IoU and FPS delta for a single sequence.

    Attributes:
        sequence_name: Sequence identifier.
        baseline_iou:  Mean IoU of the baseline run on this sequence.
        candidate_iou: Mean IoU of the candidate run on this sequence.
        iou_delta:     ``candidate_iou - baseline_iou``.  Positive = improvement.
        baseline_fps:  FPS of the baseline run on this sequence.
        candidate_fps: FPS of the candidate run on this sequence.
        fps_delta:     ``candidate_fps - baseline_fps``.
        status:        ``"improved"``, ``"regressed"``, or ``"stable"``.
    """

    sequence_name: str
    baseline_iou: float
    candidate_iou: float
    iou_delta: float
    baseline_fps: float
    candidate_fps: float
    fps_delta: float
    status: str

    def __str__(self) -> str:
        arrow = "↑" if self.iou_delta > 0 else ("↓" if self.iou_delta < 0 else "=")
        return (
            f"{self.sequence_name:<30s} [{self.status:<9s}] "
            f"IoU: {self.baseline_iou:.3f} → {self.candidate_iou:.3f} "
            f"(Δ={self.iou_delta:+.3f} {arrow})  "
            f"FPS: {self.baseline_fps:.1f} → {self.candidate_fps:.1f} "
            f"(Δ={self.fps_delta:+.1f})"
        )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class DiffReport:
    """Full regression report comparing baseline to candidate.

    Attributes:
        tracker_name:       Tracker name (expected to be the same for both runs).
        dataset_name:       Dataset name.
        baseline_mean_iou:  Aggregate mean IoU for the baseline run.
        candidate_mean_iou: Aggregate mean IoU for the candidate run.
        iou_delta:          ``candidate_mean_iou - baseline_mean_iou``.
        baseline_mean_fps:  Aggregate mean FPS for the baseline.
        candidate_mean_fps: Aggregate mean FPS for the candidate.
        fps_delta:          ``candidate_mean_fps - baseline_mean_fps``.
        sequence_deltas:    Per-sequence breakdown.
        num_improved:       Sequences where IoU improved beyond threshold.
        num_regressed:      Sequences where IoU decreased beyond threshold.
        num_stable:         Sequences within the stable band.
        regression_rate:    ``num_regressed / total_sequences``.
        regression_threshold: IoU change magnitude treated as a regression/improvement.
        accepted:           Whether the candidate passes acceptance criteria.
    """

    tracker_name: str
    dataset_name: str
    baseline_mean_iou: float
    candidate_mean_iou: float
    iou_delta: float
    baseline_mean_fps: float
    candidate_mean_fps: float
    fps_delta: float
    sequence_deltas: List[SequenceDelta]
    num_improved: int
    num_regressed: int
    num_stable: int
    regression_rate: float
    regression_threshold: float
    accepted: bool

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        verdict = "ACCEPTED" if self.accepted else "REJECTED"
        lines = [
            f"DiffReport [{self.tracker_name} on {self.dataset_name}]  [{verdict}]",
            f"  mIoU : {self.baseline_mean_iou:.4f} → {self.candidate_mean_iou:.4f}  "
            f"(Δ={self.iou_delta:+.4f})",
            f"  FPS  : {self.baseline_mean_fps:.1f} → {self.candidate_mean_fps:.1f}  "
            f"(Δ={self.fps_delta:+.1f})",
            f"  Sequences: {len(self.sequence_deltas)} total  "
            f"improved={self.num_improved}  stable={self.num_stable}  "
            f"regressed={self.num_regressed} ({self.regression_rate:.1%})",
            "",
        ]
        sorted_deltas = sorted(
            self.sequence_deltas, key=lambda d: d.iou_delta
        )
        for sd in sorted_deltas:
            lines.append("  " + str(sd))
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Return a Markdown regression table suitable for a PR body or wiki."""
        verdict = "✅ ACCEPTED" if self.accepted else "❌ REJECTED"
        lines = [
            f"## Benchmark Regression Report  {verdict}",
            "",
            f"**Tracker:** {self.tracker_name}  |  **Dataset:** {self.dataset_name}",
            "",
            "### Aggregate",
            "",
            "| Metric | Baseline | Candidate | Delta |",
            "|--------|--------:|----------:|------:|",
            f"| mIoU | {self.baseline_mean_iou:.4f} | {self.candidate_mean_iou:.4f} "
            f"| {self.iou_delta:+.4f} |",
            f"| FPS | {self.baseline_mean_fps:.1f} | {self.candidate_mean_fps:.1f} "
            f"| {self.fps_delta:+.1f} |",
            "",
            f"Sequences: {len(self.sequence_deltas)} total — "
            f"**improved:** {self.num_improved}, "
            f"**stable:** {self.num_stable}, "
            f"**regressed:** {self.num_regressed} "
            f"({self.regression_rate:.1%})",
            "",
            "### Per-Sequence Breakdown",
            "",
            "| Sequence | Status | Baseline IoU | Candidate IoU | ΔIoU | Baseline FPS | Candidate FPS | ΔFPS |",
            "|----------|:------:|-------------:|--------------:|-----:|-------------:|--------------:|-----:|",
        ]
        sorted_deltas = sorted(self.sequence_deltas, key=lambda d: d.iou_delta)
        for sd in sorted_deltas:
            icon = {"improved": "↑", "regressed": "↓", "stable": "="}.get(sd.status, "")
            lines.append(
                f"| {sd.sequence_name} | {icon} {sd.status} "
                f"| {sd.baseline_iou:.4f} | {sd.candidate_iou:.4f} | {sd.iou_delta:+.4f} "
                f"| {sd.baseline_fps:.1f} | {sd.candidate_fps:.1f} | {sd.fps_delta:+.1f} |"
            )
        return "\n".join(lines)

    def raise_if_regressed(self) -> None:
        """Raise :class:`RegressionError` when the candidate is not accepted.

        Useful in CI pipelines to block merges when a tracker change degrades
        performance beyond the configured thresholds.

        Raises:
            RegressionError: With a message describing the failure reason.
        """
        if not self.accepted:
            raise RegressionError(
                f"Candidate rejected: "
                f"mIoU Δ={self.iou_delta:+.4f}, "
                f"regression_rate={self.regression_rate:.1%} on "
                f"{self.num_regressed}/{len(self.sequence_deltas)} sequences."
            )


# ---------------------------------------------------------------------------
# BenchmarkDiff
# ---------------------------------------------------------------------------

class BenchmarkDiff:
    """Compare two :class:`~eovot.benchmark.engine.BenchmarkResult` objects.

    Args:
        regression_threshold: Minimum absolute IoU decrease to count as a
            regression (and minimum increase to count as an improvement).
            Default: ``0.02`` (2 percentage points).
        regression_rate_limit: Maximum fraction of sequences allowed to regress
            before the candidate is considered rejected.
            Default: ``0.25`` (25 %).
        require_iou_improvement: When ``True``, the candidate is also rejected
            if aggregate mIoU does not improve.  Default: ``False`` (aggregate
            IoU can be equal; only regression rate is enforced).
    """

    def __init__(
        self,
        regression_threshold: float = 0.02,
        regression_rate_limit: float = 0.25,
        require_iou_improvement: bool = False,
    ) -> None:
        if regression_threshold < 0:
            raise ValueError("regression_threshold must be ≥ 0.")
        if not 0.0 <= regression_rate_limit <= 1.0:
            raise ValueError("regression_rate_limit must be in [0, 1].")
        self.regression_threshold = regression_threshold
        self.regression_rate_limit = regression_rate_limit
        self.require_iou_improvement = require_iou_improvement

    def compare(
        self,
        baseline: "BenchmarkResult",
        candidate: "BenchmarkResult",
    ) -> DiffReport:
        """Compare *baseline* to *candidate* and return a :class:`DiffReport`.

        Sequences present only in *baseline* or only in *candidate* are
        excluded from the per-sequence breakdown; a warning is included in the
        report.  The aggregate metrics always use all sequences in each result.

        Args:
            baseline:  Reference :class:`~eovot.benchmark.engine.BenchmarkResult`.
            candidate: New :class:`~eovot.benchmark.engine.BenchmarkResult` to
                evaluate.

        Returns:
            Populated :class:`DiffReport`.
        """
        # Index sequences by name for alignment
        base_seqs: Dict[str, object] = {
            r.sequence_name: r for r in baseline.sequence_results
        }
        cand_seqs: Dict[str, object] = {
            r.sequence_name: r for r in candidate.sequence_results
        }
        common = sorted(set(base_seqs) & set(cand_seqs))

        deltas: List[SequenceDelta] = []
        for name in common:
            b = base_seqs[name]
            c = cand_seqs[name]
            b_iou = float(b.ious.mean()) if len(b.ious) else 0.0  # type: ignore[union-attr]
            c_iou = float(c.ious.mean()) if len(c.ious) else 0.0  # type: ignore[union-attr]
            iou_delta = c_iou - b_iou
            b_fps = b.profiling.fps  # type: ignore[union-attr]
            c_fps = c.profiling.fps  # type: ignore[union-attr]
            fps_delta = c_fps - b_fps

            if iou_delta > self.regression_threshold:
                status = "improved"
            elif iou_delta < -self.regression_threshold:
                status = "regressed"
            else:
                status = "stable"

            deltas.append(SequenceDelta(
                sequence_name=name,
                baseline_iou=b_iou,
                candidate_iou=c_iou,
                iou_delta=iou_delta,
                baseline_fps=b_fps,
                candidate_fps=c_fps,
                fps_delta=fps_delta,
                status=status,
            ))

        num_improved = sum(1 for d in deltas if d.status == "improved")
        num_regressed = sum(1 for d in deltas if d.status == "regressed")
        num_stable = sum(1 for d in deltas if d.status == "stable")
        total = len(deltas)
        regression_rate = num_regressed / total if total > 0 else 0.0

        iou_delta = candidate.mean_iou - baseline.mean_iou
        fps_delta = candidate.mean_fps - baseline.mean_fps

        accepted = regression_rate <= self.regression_rate_limit
        if self.require_iou_improvement:
            accepted = accepted and iou_delta >= 0

        return DiffReport(
            tracker_name=candidate.tracker_name,
            dataset_name=candidate.dataset_name,
            baseline_mean_iou=baseline.mean_iou,
            candidate_mean_iou=candidate.mean_iou,
            iou_delta=iou_delta,
            baseline_mean_fps=baseline.mean_fps,
            candidate_mean_fps=candidate.mean_fps,
            fps_delta=fps_delta,
            sequence_deltas=deltas,
            num_improved=num_improved,
            num_regressed=num_regressed,
            num_stable=num_stable,
            regression_rate=regression_rate,
            regression_threshold=self.regression_threshold,
            accepted=accepted,
        )

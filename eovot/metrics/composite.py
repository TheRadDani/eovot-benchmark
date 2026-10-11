"""Composite Deployability Score (CDS) for multi-dimensional edge tracker ranking.

Standard VOT benchmarks rank trackers by a single accuracy scalar (success AUC).
EOVOT's core thesis is that edge deployment quality has *four* orthogonal
dimensions — accuracy, robustness, efficiency, and temporal smoothness — and
that a single scalar combining all four is necessary for deployment decisions.

The Composite Deployability Score operationalises this thesis::

    CDS = w_acc × success_auc
        + w_rob × eao
        + w_eff × ees_normalized
        + w_smooth × smoothness_score

All inputs are in [0, 1]:

* ``success_auc``    — Area Under the Success Curve (accuracy, IoU-threshold sweep).
* ``eao``            — Expected Average Overlap (robustness + accuracy combined).
* ``ees_normalized`` — Edge Efficiency Score normalised to [0, 1] across the comparison
                       set via MinMax scaling.  Absolute EES values are not comparable
                       across datasets; normalisation makes the ranking set-relative.
* ``smoothness_score`` — Temporal consistency composite (position jitter, scale jitter,
                          velocity outliers); available only when live BenchmarkResult
                          objects are passed (predictions are not stored in JSON).

Default weights are calibrated for embedded vision deployment:

* ``w_acc = 0.40``    Accuracy remains the primary quality signal.
* ``w_rob = 0.20``    Robustness matters for long autonomous sequences.
* ``w_eff = 0.30``    Efficiency is critical on constrained hardware.
* ``w_smooth = 0.10`` Smoothness matters for downstream consumers (controllers, pipelines).

When temporal data is unavailable (JSON-only workflow), the ``w_smooth`` budget
is redistributed proportionally among the remaining three dimensions so the
weights still sum to 1.0.

Typical usage::

    from eovot.metrics.composite import CompositeDeployabilityScore

    # From live BenchmarkResult objects (full 4-dimension CDS)
    engine = CompositeDeployabilityScore()
    entries = engine.rank(benchmark_results)
    print(engine.to_markdown_table(entries))

    # From saved JSON files (3-dimension CDS, temporal excluded)
    result_dicts = [json.load(open(p)) for p in json_paths]
    entries = engine.rank_from_dicts(result_dicts)
    print(engine.to_markdown_table(entries))
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class CDSEntry:
    """Per-tracker Composite Deployability Score summary.

    Attributes:
        tracker_name:      Human-readable tracker identifier.
        dataset_name:      Dataset on which the tracker was evaluated.
        cds:               Composite Deployability Score (primary ranking key).
        success_auc:       Area Under the Success Curve (accuracy).
        eao:               Expected Average Overlap (robustness).
        ees_normalized:    EES normalised to [0, 1] across the comparison set.
        smoothness_score:  Temporal consistency composite; 1.0 when unavailable.
        mean_iou:          Mean IoU across all evaluated frames.
        fps:               Mean frames-per-second throughput.
        peak_memory_mb:    Peak RSS memory footprint in megabytes.
        num_failures:      Total tracking failures across all sequences.
        survival_rate:     Fraction of frames where IoU ≥ failure_threshold.
        has_temporal:      True when smoothness was computed from live predictions.
    """

    tracker_name: str
    dataset_name: str
    cds: float
    success_auc: float
    eao: float
    ees_normalized: float
    smoothness_score: float
    mean_iou: float
    fps: float
    peak_memory_mb: float
    num_failures: int = 0
    survival_rate: float = 0.0
    has_temporal: bool = False

    def __str__(self) -> str:
        temporal = "yes" if self.has_temporal else "no"
        return (
            f"CDSEntry({self.tracker_name}  "
            f"CDS={self.cds:.4f}  AUC={self.success_auc:.4f}  "
            f"EAO={self.eao:.4f}  EES_n={self.ees_normalized:.4f}  "
            f"S={self.smoothness_score:.4f}  temporal={temporal})"
        )


class CompositeDeployabilityScore:
    """Rank trackers by a weighted composite of accuracy, robustness, efficiency,
    and temporal consistency.

    Args:
        w_acc:             Weight for success_auc. Default ``0.40``.
        w_rob:             Weight for EAO. Default ``0.20``.
        w_eff:             Weight for normalised EES. Default ``0.30``.
        w_smooth:          Weight for smoothness score. Default ``0.10``.
        memory_budget_mb:  Memory budget (MB) for EES computation. Default ``512.0``.
        failure_threshold: IoU below which a frame counts as a failure. Default ``0.1``.

    Raises:
        ValueError: If the four weights do not sum to 1.0 (±1e-6 tolerance).

    Example::

        engine = CompositeDeployabilityScore(w_acc=0.5, w_rob=0.2, w_eff=0.2, w_smooth=0.1)
        entries = engine.rank(results)
        for e in entries:
            print(e)
    """

    def __init__(
        self,
        w_acc: float = 0.40,
        w_rob: float = 0.20,
        w_eff: float = 0.30,
        w_smooth: float = 0.10,
        memory_budget_mb: float = 512.0,
        failure_threshold: float = 0.1,
    ) -> None:
        total = w_acc + w_rob + w_eff + w_smooth
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.6f}. "
                f"(w_acc={w_acc}, w_rob={w_rob}, w_eff={w_eff}, w_smooth={w_smooth})"
            )
        if memory_budget_mb <= 0:
            raise ValueError(f"memory_budget_mb must be positive, got {memory_budget_mb}.")
        self.w_acc = w_acc
        self.w_rob = w_rob
        self.w_eff = w_eff
        self.w_smooth = w_smooth
        self.memory_budget_mb = memory_budget_mb
        self.failure_threshold = failure_threshold

    # ------------------------------------------------------------------
    # Public ranking entry points
    # ------------------------------------------------------------------

    def rank(self, results: List) -> List[CDSEntry]:
        """Build a CDS ranking from live :class:`~eovot.benchmark.engine.BenchmarkResult` objects.

        This path has access to per-frame predictions and therefore computes
        the full 4-dimension CDS including temporal smoothness.

        Args:
            results: List of :class:`~eovot.benchmark.engine.BenchmarkResult`,
                one per tracker / dataset combination.

        Returns:
            :class:`CDSEntry` list sorted by CDS descending.
        """
        from .robustness import RobustnessAnalyzer
        from .temporal import TemporalConsistencyAnalyzer

        rob_analyzer = RobustnessAnalyzer(failure_threshold=self.failure_threshold)
        temp_analyzer = TemporalConsistencyAnalyzer()

        raw: List[Dict] = []
        for r in results:
            seq_ious = {sr.sequence_name: sr.ious for sr in r.sequence_results}
            rob = rob_analyzer.analyze_benchmark(seq_ious, tracker_name=r.tracker_name)
            agg_rob = rob["aggregate"]

            has_temporal = any(
                sr.predictions is not None and len(sr.predictions) > 1
                for sr in r.sequence_results
            )
            smoothness = 1.0
            if has_temporal:
                seq_preds = {
                    sr.sequence_name: sr.predictions
                    for sr in r.sequence_results
                    if sr.predictions is not None
                }
                temp = temp_analyzer.analyze_benchmark(seq_preds, tracker_name=r.tracker_name)
                smoothness = float(temp["aggregate"].get("mean_smoothness_score", 1.0))

            ees_raw = self._ees(r.mean_iou, r.mean_fps, r.peak_memory_mb)
            sauc = r.mean_success_auc if r.mean_success_auc is not None else r.mean_iou

            raw.append({
                "tracker_name": r.tracker_name,
                "dataset_name": r.dataset_name,
                "success_auc": sauc,
                "eao": float(agg_rob.get("mean_eao", 0.0)),
                "ees_raw": ees_raw,
                "smoothness_score": smoothness,
                "has_temporal": has_temporal,
                "mean_iou": r.mean_iou,
                "fps": r.mean_fps,
                "peak_memory_mb": r.peak_memory_mb,
                "num_failures": int(agg_rob.get("total_failures", 0)),
                "survival_rate": float(agg_rob.get("mean_survival_rate", 0.0)),
            })

        return self._finalize(raw)

    def rank_from_dicts(self, result_dicts: List[Dict]) -> List[CDSEntry]:
        """Build a CDS ranking from JSON-serialised result dicts.

        Temporal smoothness is unavailable (predictions are not stored in
        JSON), so the ``w_smooth`` weight is redistributed proportionally
        among the remaining three dimensions.

        Args:
            result_dicts: List of dicts from
                :meth:`~eovot.benchmark.engine.BenchmarkResult.to_dict`.

        Returns:
            :class:`CDSEntry` list sorted by CDS descending.
        """
        from .robustness import RobustnessAnalyzer

        rob_analyzer = RobustnessAnalyzer(failure_threshold=self.failure_threshold)

        raw: List[Dict] = []
        for d in result_dicts:
            summary = d.get("summary", {})
            sequences = d.get("sequences", [])

            tracker_name = summary.get("tracker", "?")
            dataset_name = summary.get("dataset", "?")
            mean_iou = float(summary.get("mean_iou", 0.0))
            fps = float(summary.get("mean_fps", 0.0))
            memory_mb = float(summary.get("peak_memory_mb", 0.0))
            success_auc = float(summary.get("success_auc", mean_iou))

            seq_ious: Dict[str, np.ndarray] = {}
            for seq in sequences:
                name = seq.get("sequence_name", "")
                ious_list = seq.get("ious", [seq.get("mean_iou", 0.0)])
                seq_ious[name] = np.array(ious_list, dtype=np.float64)

            rob = rob_analyzer.analyze_benchmark(seq_ious, tracker_name=tracker_name)
            agg_rob = rob["aggregate"]

            raw.append({
                "tracker_name": tracker_name,
                "dataset_name": dataset_name,
                "success_auc": success_auc,
                "eao": float(agg_rob.get("mean_eao", 0.0)),
                "ees_raw": self._ees(mean_iou, fps, memory_mb),
                "smoothness_score": 1.0,
                "has_temporal": False,
                "mean_iou": mean_iou,
                "fps": fps,
                "peak_memory_mb": memory_mb,
                "num_failures": int(agg_rob.get("total_failures", 0)),
                "survival_rate": float(agg_rob.get("mean_survival_rate", 0.0)),
            })

        return self._finalize(raw, force_no_temporal=True)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def to_markdown_table(self, entries: List[CDSEntry]) -> str:
        """Format a CDS ranking as a Markdown table.

        Args:
            entries: Output of :meth:`rank` or :meth:`rank_from_dicts`.

        Returns:
            Multi-line Markdown string ready to paste into a report or README.
        """
        has_temporal = any(e.has_temporal for e in entries)

        if has_temporal:
            header = (
                "| Rank | Tracker | Dataset | CDS | Success AUC | EAO | EES (norm) | Smoothness | mIoU | FPS | Failures |\n"
                "|------|---------|---------|----:|------------:|----:|-----------:|-----------:|-----:|----:|---------:|"
            )
            rows = [
                f"| {rank} | {e.tracker_name} | {e.dataset_name} "
                f"| **{e.cds:.4f}** | {e.success_auc:.4f} | {e.eao:.4f} "
                f"| {e.ees_normalized:.4f} | {e.smoothness_score:.4f} "
                f"| {e.mean_iou:.4f} | {e.fps:.1f} | {e.num_failures} |"
                for rank, e in enumerate(entries, start=1)
            ]
        else:
            header = (
                "| Rank | Tracker | Dataset | CDS | Success AUC | EAO | EES (norm) | mIoU | FPS | Failures |\n"
                "|------|---------|---------|----:|------------:|----:|-----------:|-----:|----:|---------:|"
            )
            rows = [
                f"| {rank} | {e.tracker_name} | {e.dataset_name} "
                f"| **{e.cds:.4f}** | {e.success_auc:.4f} | {e.eao:.4f} "
                f"| {e.ees_normalized:.4f} "
                f"| {e.mean_iou:.4f} | {e.fps:.1f} | {e.num_failures} |"
                for rank, e in enumerate(entries, start=1)
            ]

        return header + "\n" + "\n".join(rows)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ees(self, mean_iou: float, fps: float, peak_memory_mb: float) -> float:
        """Raw Edge Efficiency Score (before normalisation)."""
        if fps <= 0 or mean_iou < 0:
            return 0.0
        return (mean_iou * math.log1p(fps)) / (1.0 + peak_memory_mb / self.memory_budget_mb)

    def _finalize(
        self,
        raw: List[Dict],
        force_no_temporal: bool = False,
    ) -> List[CDSEntry]:
        """Normalise EES, compute CDS, and sort by CDS descending."""
        if not raw:
            return []

        # MinMax-normalise EES across the comparison set.
        # When all trackers have identical EES (e.g. a single-tracker run), everyone
        # gets full EES credit (1.0) rather than 0.0.
        ees_values = [e["ees_raw"] for e in raw]
        ees_max = max(ees_values) if ees_values else 1.0
        ees_min = min(ees_values) if ees_values else 0.0
        ees_range = ees_max - ees_min if ees_max > ees_min else None

        # Redistribute w_smooth when temporal is unavailable
        use_temporal = not force_no_temporal and any(e["has_temporal"] for e in raw)
        if use_temporal:
            w_acc, w_rob, w_eff, w_smooth = self.w_acc, self.w_rob, self.w_eff, self.w_smooth
        else:
            total_3 = self.w_acc + self.w_rob + self.w_eff
            w_acc = self.w_acc / total_3
            w_rob = self.w_rob / total_3
            w_eff = self.w_eff / total_3
            w_smooth = 0.0

        entries: List[CDSEntry] = []
        for e in raw:
            ees_norm = (e["ees_raw"] - ees_min) / ees_range if ees_range is not None else 1.0
            cds = (
                w_acc * e["success_auc"]
                + w_rob * e["eao"]
                + w_eff * ees_norm
                + w_smooth * e["smoothness_score"]
            )
            entries.append(CDSEntry(
                tracker_name=e["tracker_name"],
                dataset_name=e["dataset_name"],
                cds=round(cds, 4),
                success_auc=round(e["success_auc"], 4),
                eao=round(e["eao"], 4),
                ees_normalized=round(ees_norm, 4),
                smoothness_score=round(e["smoothness_score"], 4),
                mean_iou=round(e["mean_iou"], 4),
                fps=round(e["fps"], 2),
                peak_memory_mb=round(e["peak_memory_mb"], 2),
                num_failures=e["num_failures"],
                survival_rate=round(e["survival_rate"], 4),
                has_temporal=bool(e.get("has_temporal", False)),
            ))

        entries.sort(key=lambda x: x.cds, reverse=True)
        return entries

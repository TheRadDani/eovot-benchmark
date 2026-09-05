"""Consolidated research report generator for EOVOT benchmark runs.

Bridges the gap between individual metrics modules and actionable research
output by combining accuracy, robustness, efficiency, and per-attribute
performance into a **single, self-contained Markdown report** from any
collection of :class:`~eovot.benchmark.engine.BenchmarkResult` objects.

The report has five sections:

1. **Experiment Summary** — metadata (trackers, dataset, date, sequences).
2. **Accuracy Leaderboard** — mIoU, success AUC, precision AUC, normalized
   precision AUC ranked by success AUC.
3. **Efficiency Comparison** — FPS, latency, memory, and Edge Efficiency
   Score (EES) with Pareto-front annotation.
4. **Robustness Analysis** — EAO, failure rate, survival rate, and mean
   recovery lag per tracker.
5. **Per-Attribute Breakdown** — per-tracker mIoU on each VOT attribute
   (scale variation, fast motion, occlusion, etc.), automatically detected
   from the stored ground-truth boxes.

Typical usage::

    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kcf import KCFTracker
    from eovot.reporting.full_report import FullReportGenerator

    ds = SyntheticDataset(num_sequences=10, num_frames=200)
    engine = BenchmarkEngine(verbose=False)
    results = [
        engine.run(MOSSETracker(), ds, dataset_name="Synthetic"),
        engine.run(KCFTracker(),   ds, dataset_name="Synthetic"),
    ]

    gen = FullReportGenerator()
    report = gen.generate(results)
    print(report)
    gen.save(results, path="results/full_report.md")

Loading from saved JSON files::

    import json
    from eovot.benchmark.engine import BenchmarkResult
    from eovot.reporting.full_report import FullReportGenerator

    results = [
        BenchmarkResult.load("results/MOSSE.json"),
        BenchmarkResult.load("results/KCF.json"),
    ]
    gen = FullReportGenerator(memory_budget_mb=512.0, failure_threshold=0.1)
    gen.save(results, "results/full_report.md")
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..benchmark.engine import BenchmarkResult
from ..metrics.accuracy import MetricsEngine
from ..metrics.attributes import AttributeAnalyzer, AttributeDetector
from ..metrics.efficiency import EfficiencyMetricsEngine
from ..metrics.robustness import RobustnessAnalyzer


class FullReportGenerator:
    """Generate a consolidated Markdown research report from benchmark results.

    Args:
        memory_budget_mb: Memory budget used for the Edge Efficiency Score
            denominator.  Default: ``512.0`` MB.
        failure_threshold: IoU threshold below which a frame counts as a
            tracking failure in the robustness analysis.  Default: ``0.1``.
        burn_in_frames: Frames to skip at the start of each sequence before
            looking for robustness failures.  Default: ``5``.
        title: Optional custom report title.  When omitted a generic title is
            generated from the dataset name.
    """

    def __init__(
        self,
        memory_budget_mb: float = 512.0,
        failure_threshold: float = 0.1,
        burn_in_frames: int = 5,
        title: Optional[str] = None,
    ) -> None:
        self._efficiency = EfficiencyMetricsEngine(memory_budget_mb=memory_budget_mb)
        self._robustness = RobustnessAnalyzer(
            failure_threshold=failure_threshold,
            burn_in_frames=burn_in_frames,
        )
        self._attr_detector = AttributeDetector()
        self._attr_analyzer = AttributeAnalyzer(detector=self._attr_detector)
        self._title = title

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, results: List[BenchmarkResult]) -> str:
        """Build the full Markdown report string from a list of benchmark results.

        Args:
            results: One :class:`~eovot.benchmark.engine.BenchmarkResult` per
                tracker.  All results must be for the same dataset.

        Returns:
            Multi-section Markdown string ready to write to a ``.md`` file.

        Raises:
            ValueError: If *results* is empty.
        """
        if not results:
            raise ValueError("results must be non-empty.")

        dataset_name = results[0].dataset_name
        tracker_names = [r.tracker_name for r in results]
        title = self._title or f"EOVOT Benchmark Report — {dataset_name}"

        sections = [
            self._header(title, dataset_name, tracker_names, results),
            self._accuracy_section(results),
            self._efficiency_section(results),
            self._robustness_section(results),
            self._attribute_section(results),
            self._footer(),
        ]
        return "\n\n".join(s for s in sections if s)

    def save(
        self,
        results: List[BenchmarkResult],
        path: str = "results/full_report.md",
    ) -> Path:
        """Generate the report and write it to *path*.

        Args:
            results: Benchmark results (same as :meth:`generate`).
            path: Destination file path.  Parent directories are created
                automatically.  A ``.md`` extension is appended if absent.

        Returns:
            The resolved :class:`pathlib.Path` that was written.
        """
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".md")
        p.parent.mkdir(parents=True, exist_ok=True)
        content = self.generate(results)
        p.write_text(content, encoding="utf-8")
        return p

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _header(
        self,
        title: str,
        dataset_name: str,
        tracker_names: List[str],
        results: List[BenchmarkResult],
    ) -> str:
        n_seqs = max((len(r.sequence_results) for r in results), default=0)
        total_frames = sum(
            sum(len(sr.ious) for sr in r.sequence_results) for r in results
        )
        date_str = datetime.date.today().isoformat()

        lines = [
            f"# {title}",
            "",
            f"**Date:** {date_str}  ",
            f"**Dataset:** {dataset_name}  ",
            f"**Sequences:** {n_seqs}  ",
            f"**Total frames evaluated:** {total_frames:,}  ",
            f"**Trackers:** {', '.join(tracker_names)}",
            "",
            "---",
        ]
        return "\n".join(lines)

    def _accuracy_section(self, results: List[BenchmarkResult]) -> str:
        lines = [
            "## 1. Accuracy Leaderboard",
            "",
            "Ranked by **Success AUC** (area under the success curve, IoU thresholds 0→1).  "
            "Higher is better for all columns.",
            "",
        ]

        # Build rows — sort by success AUC descending (fall back to mIoU).
        rows = []
        for r in results:
            rows.append({
                "tracker": r.tracker_name,
                "mIoU": r.mean_iou,
                "success_auc": r.mean_success_auc,
                "precision_auc": r.mean_precision_auc,
                "norm_prec_auc": r.mean_normalized_precision_auc,
            })

        rows.sort(key=lambda x: (x["success_auc"] or 0.0), reverse=True)

        # Detect which optional columns are populated.
        has_sauc = any(row["success_auc"] is not None for row in rows)
        has_pauc = any(row["precision_auc"] is not None for row in rows)
        has_npauc = any(row["norm_prec_auc"] is not None for row in rows)

        # Header
        header_parts = ["| Rank | Tracker | mIoU"]
        sep_parts = ["|------|---------|-----:"]
        if has_sauc:
            header_parts.append("Success AUC")
            sep_parts.append("-----------:")
        if has_pauc:
            header_parts.append("Precision AUC")
            sep_parts.append("-------------:")
        if has_npauc:
            header_parts.append("Norm Prec AUC |")
            sep_parts.append("-------------:|")
        else:
            header_parts[-1] += " |"
            sep_parts[-1] += "|"

        lines.append(" | ".join(header_parts))
        lines.append(" | ".join(sep_parts))

        for rank, row in enumerate(rows, 1):
            parts = [f"| {rank}", row["tracker"], f"{row['mIoU']:.4f}"]
            if has_sauc:
                v = row["success_auc"]
                parts.append(f"{v:.4f}" if v is not None else "—")
            if has_pauc:
                v = row["precision_auc"]
                parts.append(f"{v:.4f}" if v is not None else "—")
            if has_npauc:
                v = row["norm_prec_auc"]
                parts.append(f"{v:.4f} |" if v is not None else "— |")
            else:
                parts[-1] += " |"
            lines.append(" | ".join(parts))

        return "\n".join(lines)

    def _efficiency_section(self, results: List[BenchmarkResult]) -> str:
        lines = [
            "## 2. Efficiency Comparison",
            "",
            "**EES** = `mean_iou × log1p(fps) / (1 + peak_memory_mb / budget)`.  "
            "Pareto-optimal trackers (★) are not dominated in both mIoU and EES.",
            "",
        ]

        entries = self._efficiency.rank_trackers(results)

        lines.append(
            "| Rank | Tracker | FPS | Latency (ms) | Mem (MB) | EES | Pareto |"
        )
        lines.append(
            "|------|---------|----:|-------------:|---------:|----:|:------:|"
        )

        # Build a latency lookup from results
        latency_map: Dict[str, float] = {
            r.tracker_name: float(
                np.mean([sr.profiling.latency_mean_ms for sr in r.sequence_results])
            )
            for r in results
            if r.sequence_results
        }

        for rank, e in enumerate(entries, 1):
            lat = latency_map.get(e.tracker_name, 0.0)
            pareto = "★" if e.on_pareto_front else ""
            lines.append(
                f"| {rank} | {e.tracker_name} "
                f"| {e.fps:.1f} "
                f"| {lat:.2f} "
                f"| {e.peak_memory_mb:.1f} "
                f"| {e.ees:.4f} "
                f"| {pareto} |"
            )

        return "\n".join(lines)

    def _robustness_section(self, results: List[BenchmarkResult]) -> str:
        lines = [
            "## 3. Robustness Analysis",
            "",
            "Failures are detected when IoU drops below the failure threshold.  "
            "**EAO** here is the raw (no re-initialization) expected average overlap.  "
            "**Survival** is the fraction of non-burn-in frames with IoU ≥ threshold.",
            "",
            "| Tracker | EAO | Failures (total) | Failures/seq | Survival | Recov. lag (fr) |",
            "|---------|----:|-----------------:|-------------:|---------:|----------------:|",
        ]

        for r in results:
            if not r.sequence_results:
                continue
            seq_ious = {sr.sequence_name: sr.ious for sr in r.sequence_results}
            agg = self._robustness.analyze_benchmark(
                seq_ious, tracker_name=r.tracker_name
            )["aggregate"]

            lines.append(
                f"| {r.tracker_name} "
                f"| {agg['mean_eao']:.4f} "
                f"| {agg['total_failures']} "
                f"| {agg['mean_failures_per_sequence']:.2f} "
                f"| {agg['mean_survival_rate']:.3f} "
                f"| {agg['mean_recovery_lag_frames']:.1f} |"
            )

        return "\n".join(lines)

    def _attribute_section(self, results: List[BenchmarkResult]) -> str:
        lines = [
            "## 4. Per-Attribute Performance",
            "",
            "mIoU broken down by automatically-detected sequence challenge attributes.  "
            "Only attributes present in ≥ 1 sequence are shown.",
            "",
        ]

        for r in results:
            # AttributeAnalyzer.breakdown expects a BenchmarkResult with
            # ground_truths stored on each SequenceResult.
            table = self._attr_analyzer.breakdown(r)
            if not table.entries:
                lines.append(
                    f"_No ground-truth boxes stored for **{r.tracker_name}** "
                    f"— run the benchmark with `BenchmarkEngine` to enable this section._"
                )
                lines.append("")
                continue
            lines.append(table.to_markdown())
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _footer() -> str:
        return (
            "---\n"
            "_Report generated by [EOVOT](https://github.com/TheRadDani/eovot-benchmark)._"
        )

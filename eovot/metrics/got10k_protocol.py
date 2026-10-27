"""GOT-10k standard evaluation protocol: AO, SR0.50, SR0.75.

The GOT-10k benchmark defines three canonical scalars for ranking trackers:

AO  (Average Overlap)
    Mean IoU over all frames in a sequence, then averaged across sequences.
    This differs subtly from pooling all frames first — GOT-10k weights each
    sequence equally regardless of length.

SR0.50  (Success Rate at threshold 0.50)
    Fraction of frames in a sequence with IoU ≥ 0.50, then averaged across
    sequences.  Measures whether the tracker keeps a usable prediction (box
    at least half-overlapping with ground truth) throughout.

SR0.75  (Success Rate at threshold 0.75)
    Same as SR0.50 but at the stricter IoU ≥ 0.75 threshold.  Distinguishes
    high-precision trackers that maintain tight localisation from those that
    merely stay near the target.

Reference:
    Huang et al., "GOT-10k: A Large High-Diversity Benchmark for Generic
    Object Tracking in the Wild." TPAMI 2021.  Tables 2–4 report all three
    metrics.  https://arxiv.org/abs/1810.11981

Usage::

    import numpy as np
    from eovot.metrics.got10k_protocol import GOT10kEvaluator, GOT10kResult

    evaluator = GOT10kEvaluator()

    # Single-sequence evaluation
    preds = np.array(...)   # shape (N, 4)
    gts   = np.array(...)   # shape (N, 4)
    result = evaluator.evaluate_sequence(preds, gts, sequence_name="airplane-1")
    print(result)

    # Dataset-level aggregation across many sequences
    seq_results = [evaluator.evaluate_sequence(p, g, name=n)
                   for p, g, n in zip(all_preds, all_gts, names)]
    summary = evaluator.aggregate(seq_results)
    print(summary)  # GOT10kSummary(AO=0.512, SR50=0.624, SR75=0.380)

    # Markdown table for reports and README files
    print(evaluator.to_markdown([("MOSSE", summary_mosse), ("KCF", summary_kcf)]))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Bounding-box type alias used throughout EOVOT
BBox = Tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GOT10kSequenceResult:
    """GOT-10k evaluation metrics for a single tracking sequence.

    Attributes:
        sequence_name: Identifier of the evaluated sequence.
        ao: Average Overlap — mean IoU across all frames.
        sr50: Success Rate at IoU ≥ 0.50.
        sr75: Success Rate at IoU ≥ 0.75.
        num_frames: Number of frames evaluated (excluding the init frame).
    """

    sequence_name: str
    ao: float
    sr50: float
    sr75: float
    num_frames: int

    def __str__(self) -> str:
        return (
            f"GOT10kSequenceResult[{self.sequence_name}] "
            f"AO={self.ao:.4f}  SR0.50={self.sr50:.4f}  SR0.75={self.sr75:.4f}  "
            f"frames={self.num_frames}"
        )


@dataclass
class GOT10kSummary:
    """Dataset-level GOT-10k metrics aggregated across all sequences.

    All three scalars are the *sequence-averaged* values (each sequence
    contributes equally regardless of length), matching the official
    GOT-10k evaluation protocol.

    Attributes:
        ao: Average Overlap averaged across sequences.
        sr50: SR0.50 averaged across sequences.
        sr75: SR0.75 averaged across sequences.
        num_sequences: Number of sequences in the evaluation.
        per_sequence: Per-sequence results keyed by sequence name.
    """

    ao: float
    sr50: float
    sr75: float
    num_sequences: int
    per_sequence: Dict[str, GOT10kSequenceResult] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"GOT10kSummary(AO={self.ao:.4f}, SR0.50={self.sr50:.4f}, "
            f"SR0.75={self.sr75:.4f}, n_seq={self.num_sequences})"
        )

    def to_dict(self) -> Dict:
        """Return a JSON-serialisable representation."""
        return {
            "ao": round(self.ao, 4),
            "sr50": round(self.sr50, 4),
            "sr75": round(self.sr75, 4),
            "num_sequences": self.num_sequences,
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class GOT10kEvaluator:
    """Compute GOT-10k standard metrics from predicted and ground-truth boxes.

    The evaluator is stateless; instantiate once and call :meth:`evaluate_sequence`
    for each sequence, then :meth:`aggregate` to obtain dataset-level numbers.

    Example::

        evaluator = GOT10kEvaluator()
        seq_results = []
        for preds, gts, name in zip(all_preds, all_gts, seq_names):
            seq_results.append(evaluator.evaluate_sequence(preds, gts, name))
        summary = evaluator.aggregate(seq_results)
    """

    # ------------------------------------------------------------------
    # Per-sequence evaluation
    # ------------------------------------------------------------------

    def evaluate_sequence(
        self,
        preds: np.ndarray,
        gts: np.ndarray,
        sequence_name: str = "",
    ) -> GOT10kSequenceResult:
        """Evaluate a single sequence and return AO, SR0.50, SR0.75.

        The first frame is the initialisation frame; by GOT-10k convention the
        tracker receives the ground-truth box on frame 0 and is evaluated from
        frame 1 onwards.  If *preds* includes a frame-0 entry (as produced by
        :class:`~eovot.benchmark.engine.BenchmarkEngine`, which records the
        init bbox as the first prediction), it is skipped automatically.

        Args:
            preds: ``(N, 4)`` float array of predicted boxes ``(x, y, w, h)``.
                May include the initialisation-frame box as the first row.
            gts:   ``(N, 4)`` float array of ground-truth boxes ``(x, y, w, h)``.
            sequence_name: Identifier embedded in the returned result.

        Returns:
            :class:`GOT10kSequenceResult` with AO, SR0.50, and SR0.75.
        """
        preds = np.asarray(preds, dtype=np.float64)
        gts = np.asarray(gts, dtype=np.float64)

        n = min(len(preds), len(gts))
        if n == 0:
            return GOT10kSequenceResult(
                sequence_name=sequence_name, ao=0.0, sr50=0.0, sr75=0.0, num_frames=0
            )

        # Skip frame-0 (init frame) when both arrays have the same length and
        # the first GT / pred match exactly (engine records the init bbox).
        start = 0
        if n > 1 and np.allclose(preds[0], gts[0], atol=1e-3):
            start = 1

        p = preds[start:n]
        g = gts[start:n]
        n_eval = len(p)

        if n_eval == 0:
            return GOT10kSequenceResult(
                sequence_name=sequence_name, ao=0.0, sr50=0.0, sr75=0.0, num_frames=0
            )

        ious = self._batch_iou(p, g)
        ao = float(ious.mean())
        sr50 = float((ious >= 0.50).mean())
        sr75 = float((ious >= 0.75).mean())

        return GOT10kSequenceResult(
            sequence_name=sequence_name,
            ao=ao,
            sr50=sr50,
            sr75=sr75,
            num_frames=n_eval,
        )

    # ------------------------------------------------------------------
    # Dataset-level aggregation
    # ------------------------------------------------------------------

    def aggregate(
        self, sequence_results: List[GOT10kSequenceResult]
    ) -> GOT10kSummary:
        """Average per-sequence metrics across a full dataset evaluation.

        Per GOT-10k protocol each sequence contributes equally regardless of
        length — this is a sequence-level (not frame-level) average.

        Args:
            sequence_results: List of :class:`GOT10kSequenceResult` objects
                produced by :meth:`evaluate_sequence`.

        Returns:
            :class:`GOT10kSummary` with dataset-level AO, SR0.50, SR0.75.
        """
        if not sequence_results:
            return GOT10kSummary(ao=0.0, sr50=0.0, sr75=0.0, num_sequences=0)

        aos = np.array([r.ao for r in sequence_results])
        sr50s = np.array([r.sr50 for r in sequence_results])
        sr75s = np.array([r.sr75 for r in sequence_results])

        per_seq = {r.sequence_name: r for r in sequence_results}

        return GOT10kSummary(
            ao=float(aos.mean()),
            sr50=float(sr50s.mean()),
            sr75=float(sr75s.mean()),
            num_sequences=len(sequence_results),
            per_sequence=per_seq,
        )

    # ------------------------------------------------------------------
    # Convenience: evaluate a BenchmarkResult directly
    # ------------------------------------------------------------------

    def evaluate_benchmark(self, benchmark_result) -> GOT10kSummary:
        """Compute GOT-10k metrics directly from a :class:`~eovot.benchmark.engine.BenchmarkResult`.

        Args:
            benchmark_result: Fully populated result from
                :class:`~eovot.benchmark.engine.BenchmarkEngine`.

        Returns:
            :class:`GOT10kSummary` with AO, SR0.50, SR0.75 for the full run.

        Raises:
            ValueError: If the benchmark result contains no sequences or
                per-frame predictions were not stored.
        """
        seq_results: List[GOT10kSequenceResult] = []

        for sr in benchmark_result.sequence_results:
            if sr.predictions is None or sr.ground_truths is None:
                # Fall back to IoU-only AO when per-frame arrays aren't stored.
                ao = sr.mean_iou
                seq_results.append(
                    GOT10kSequenceResult(
                        sequence_name=sr.sequence_name,
                        ao=ao,
                        sr50=float((sr.ious >= 0.50).mean()) if len(sr.ious) else 0.0,
                        sr75=float((sr.ious >= 0.75).mean()) if len(sr.ious) else 0.0,
                        num_frames=len(sr.ious),
                    )
                )
            else:
                seq_results.append(
                    self.evaluate_sequence(
                        sr.predictions,
                        sr.ground_truths,
                        sequence_name=sr.sequence_name,
                    )
                )

        return self.aggregate(seq_results)

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def to_markdown(
        self,
        tracker_summaries: List[Tuple[str, GOT10kSummary]],
        title: str = "GOT-10k Evaluation Results",
    ) -> str:
        """Format a multi-tracker comparison table as Markdown.

        Args:
            tracker_summaries: List of ``(tracker_name, GOT10kSummary)`` pairs.
                Sorted by AO descending before rendering.
            title: Optional heading prepended to the table.

        Returns:
            Multi-line Markdown string ready for README or report embedding.
        """
        lines = []
        if title:
            lines.append(f"### {title}\n")

        lines += [
            "| Rank | Tracker | AO ↑ | SR0.50 ↑ | SR0.75 ↑ | Sequences |",
            "|------|---------|-----:|---------:|---------:|----------:|",
        ]

        sorted_summaries = sorted(
            tracker_summaries, key=lambda x: x[1].ao, reverse=True
        )

        for rank, (name, summary) in enumerate(sorted_summaries, start=1):
            lines.append(
                f"| {rank} | {name} "
                f"| {summary.ao:.4f} "
                f"| {summary.sr50:.4f} "
                f"| {summary.sr75:.4f} "
                f"| {summary.num_sequences} |"
            )

        return "\n".join(lines)

    def to_latex(
        self, tracker_summaries: List[Tuple[str, GOT10kSummary]]
    ) -> str:
        """Format results as a LaTeX ``tabular`` fragment.

        Best value in each column is typeset in bold for direct inclusion
        in conference and journal papers following the GOT-10k format.

        Args:
            tracker_summaries: List of ``(tracker_name, GOT10kSummary)`` pairs.

        Returns:
            LaTeX ``tabular`` string (no surrounding ``table`` float).
        """
        sorted_summaries = sorted(
            tracker_summaries, key=lambda x: x[1].ao, reverse=True
        )

        best_ao = max(s.ao for _, s in sorted_summaries)
        best_sr50 = max(s.sr50 for _, s in sorted_summaries)
        best_sr75 = max(s.sr75 for _, s in sorted_summaries)

        def _fmt(val: float, best: float) -> str:
            s = f"{val:.4f}"
            return f"\\textbf{{{s}}}" if abs(val - best) < 1e-6 else s

        lines = [
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Tracker & AO $\uparrow$ & SR$_{0.50}$ $\uparrow$ & SR$_{0.75}$ $\uparrow$ \\",
            r"\midrule",
        ]

        for name, summary in sorted_summaries:
            ao_s = _fmt(summary.ao, best_ao)
            sr50_s = _fmt(summary.sr50, best_sr50)
            sr75_s = _fmt(summary.sr75, best_sr75)
            lines.append(f"{name} & {ao_s} & {sr50_s} & {sr75_s} \\\\")

        lines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _batch_iou(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
        """Vectorised per-frame IoU using NumPy broadcasting."""
        p = preds
        g = gts

        ix1 = np.maximum(p[:, 0], g[:, 0])
        iy1 = np.maximum(p[:, 1], g[:, 1])
        ix2 = np.minimum(p[:, 0] + p[:, 2], g[:, 0] + g[:, 2])
        iy2 = np.minimum(p[:, 1] + p[:, 3], g[:, 1] + g[:, 3])

        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        union = p[:, 2] * p[:, 3] + g[:, 2] * g[:, 3] - inter

        valid = (
            (p[:, 2] > 0) & (p[:, 3] > 0)
            & (g[:, 2] > 0) & (g[:, 3] > 0)
            & (union > 0)
        )
        return np.where(valid, inter / union, 0.0)

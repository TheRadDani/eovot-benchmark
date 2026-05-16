"""GOT-10k evaluation protocol for EOVOT.

Implements the official GOT-10k evaluation metrics defined in:

    Huang et al., "GOT-10k: A Large High-Diversity Benchmark for Generic
    Object Tracking in the Wild." IEEE TPAMI 2021.

Metrics
-------
* **AO** (Average Overlap): mean IoU across all non-initialisation frames
  (frame 0 is excluded — it is the init frame and trivially has IoU = 1).
* **SR₅₀** (Success Rate at τ = 0.5): fraction of non-init frames with
  IoU ≥ 0.5.  Required by the official leaderboard.
* **SR₇₅** (Success Rate at τ = 0.75): fraction of non-init frames with
  IoU ≥ 0.75.  Discriminates high-accuracy trackers.

The evaluator also exports predictions in the official GOT-10k submission
format (one TXT file per sequence, starting from frame 2) for upload to
the evaluation server at http://got-10k.aitestunion.com/

Typical usage::

    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.metrics.got10k_eval import GOT10kEvaluator

    engine = BenchmarkEngine()
    result = engine.run(tracker, got10k_val_dataset, dataset_name="GOT-10k-val")

    evaluator = GOT10kEvaluator(split="val")
    report = evaluator.evaluate(result)
    print(report)
    # GOT-10k [MOSSE / val] AO=0.3421  SR50=0.3012  SR75=0.0987  (180 sequences)

    # Export for submission to the evaluation server
    evaluator.export_submission(report, result, output_dir="submissions/")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class GOT10kSequenceResult:
    """Per-sequence GOT-10k evaluation result."""

    sequence_name: str
    ao: float
    """Average Overlap — mean IoU over all non-init frames."""
    sr50: float
    """Success Rate at IoU ≥ 0.5."""
    sr75: float
    """Success Rate at IoU ≥ 0.75."""
    num_frames: int
    """Total frame count including the initialisation frame."""

    def __str__(self) -> str:
        return (
            f"{self.sequence_name}: AO={self.ao:.4f}  "
            f"SR50={self.sr50:.4f}  SR75={self.sr75:.4f}  "
            f"frames={self.num_frames}"
        )

    def to_dict(self) -> Dict:
        return {
            "name": self.sequence_name,
            "ao": round(self.ao, 4),
            "sr50": round(self.sr50, 4),
            "sr75": round(self.sr75, 4),
            "frames": self.num_frames,
        }


@dataclass
class GOT10kReport:
    """Aggregate GOT-10k evaluation report across all sequences.

    Attributes:
        tracker_name:  Identifier of the evaluated tracker.
        dataset_split: Dataset split (``"val"`` or ``"test"``).
        ao:            Mean AO across sequences — primary leaderboard metric.
        sr50:          Mean SR₅₀ across sequences.
        sr75:          Mean SR₇₅ across sequences.
        num_sequences: Number of sequences evaluated.
        per_sequence:  Per-sequence breakdown.
    """

    tracker_name: str
    dataset_split: str
    ao: float
    sr50: float
    sr75: float
    num_sequences: int
    per_sequence: List[GOT10kSequenceResult] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"GOT-10k [{self.tracker_name} / {self.dataset_split}] "
            f"AO={self.ao:.4f}  SR50={self.sr50:.4f}  SR75={self.sr75:.4f}  "
            f"({self.num_sequences} sequences)"
        )

    def to_dict(self) -> Dict:
        return {
            "tracker": self.tracker_name,
            "split": self.dataset_split,
            "ao": round(self.ao, 4),
            "sr50": round(self.sr50, 4),
            "sr75": round(self.sr75, 4),
            "num_sequences": self.num_sequences,
            "per_sequence": [s.to_dict() for s in self.per_sequence],
        }


def compute_ao(ious: np.ndarray) -> float:
    """Compute Average Overlap (AO) following the GOT-10k protocol.

    The initialisation frame (index 0) is excluded; it is trivially
    IoU = 1.0 by construction and would inflate the score.

    Args:
        ious: Per-frame IoU array of shape ``(N,)`` where ``N`` is the
            total sequence length including the init frame.

    Returns:
        Mean IoU over frames ``1 … N-1``, or ``0.0`` for sequences
        shorter than 2 frames.
    """
    if len(ious) < 2:
        return 0.0
    return float(np.mean(ious[1:]))


def compute_sr(ious: np.ndarray, threshold: float = 0.5) -> float:
    """Compute Success Rate (SR) at a given IoU threshold.

    Args:
        ious:      Per-frame IoU array of shape ``(N,)``.
        threshold: IoU threshold in ``[0, 1]``.  Default: ``0.5`` (SR₅₀).

    Returns:
        Fraction of non-init frames with IoU ≥ ``threshold``, or ``0.0``
        for sequences shorter than 2 frames.
    """
    if len(ious) < 2:
        return 0.0
    return float(np.mean(ious[1:] >= threshold))


class GOT10kEvaluator:
    """Evaluate tracker results using the official GOT-10k protocol.

    Computes AO, SR₅₀, and SR₇₅ from
    :class:`~eovot.benchmark.engine.BenchmarkResult` objects and optionally
    exports results in the official submission format.

    Args:
        split: Dataset split being evaluated — used for report labelling only.
            Default: ``"val"``.

    Example::

        evaluator = GOT10kEvaluator(split="val")
        report = evaluator.evaluate(benchmark_result)
        print(report)

        # Save JSON report + submission TXT files
        evaluator.export_submission(report, benchmark_result, output_dir="submit/")

        # Compare multiple trackers
        reports = [evaluator.evaluate(r) for r in results]
        print(evaluator.to_markdown_table(reports))
    """

    def __init__(self, split: str = "val") -> None:
        self.split = split

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(self, result) -> GOT10kReport:
        """Compute GOT-10k metrics from a :class:`~eovot.benchmark.engine.BenchmarkResult`.

        Args:
            result: Output of :meth:`~eovot.benchmark.engine.BenchmarkEngine.run`.

        Returns:
            :class:`GOT10kReport` with aggregate and per-sequence statistics.
        """
        per_seq: List[GOT10kSequenceResult] = []

        for sr in result.sequence_results:
            ious = np.asarray(sr.ious, dtype=np.float64)
            per_seq.append(
                GOT10kSequenceResult(
                    sequence_name=sr.sequence_name,
                    ao=compute_ao(ious),
                    sr50=compute_sr(ious, 0.5),
                    sr75=compute_sr(ious, 0.75),
                    num_frames=len(ious),
                )
            )

        n = len(per_seq)
        mean_ao = float(np.mean([s.ao for s in per_seq])) if n else 0.0
        mean_sr50 = float(np.mean([s.sr50 for s in per_seq])) if n else 0.0
        mean_sr75 = float(np.mean([s.sr75 for s in per_seq])) if n else 0.0

        return GOT10kReport(
            tracker_name=result.tracker_name,
            dataset_split=self.split,
            ao=mean_ao,
            sr50=mean_sr50,
            sr75=mean_sr75,
            num_sequences=n,
            per_sequence=per_seq,
        )

    # ------------------------------------------------------------------
    # Submission format export
    # ------------------------------------------------------------------

    def export_submission(
        self,
        report: GOT10kReport,
        result,
        output_dir: str,
    ) -> Path:
        """Export results in the GOT-10k official submission format.

        The official server expects a directory named ``<tracker_name>``
        containing one TXT file per sequence.  Each file holds predicted
        bounding boxes from frame 2 onwards (frame 1 is the init frame),
        one ``x,y,w,h`` box per line (comma-separated, three decimal places).

        A JSON summary of AO / SR₅₀ / SR₇₅ is also written alongside
        the per-sequence files for local record-keeping.

        Args:
            report:     :class:`GOT10kReport` returned by :meth:`evaluate`.
            result:     Original
                :class:`~eovot.benchmark.engine.BenchmarkResult` — needed
                for the raw per-frame predictions.
            output_dir: Root directory.  A ``<tracker_name>/`` subdirectory
                is created inside it.

        Returns:
            Path to the tracker submission directory.
        """
        submit_dir = Path(output_dir) / report.tracker_name
        submit_dir.mkdir(parents=True, exist_ok=True)

        for sr in result.sequence_results:
            if sr.predictions is None or len(sr.predictions) < 2:
                continue
            pred_txt = submit_dir / f"{sr.sequence_name}.txt"
            with open(pred_txt, "w") as fh:
                for box in sr.predictions[1:]:
                    fh.write(
                        f"{box[0]:.3f},{box[1]:.3f},{box[2]:.3f},{box[3]:.3f}\n"
                    )

        report_path = submit_dir / "got10k_report.json"
        with open(report_path, "w") as fh:
            json.dump(report.to_dict(), fh, indent=2)

        return submit_dir

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def to_markdown_table(self, reports: List[GOT10kReport]) -> str:
        """Format a list of GOT-10k reports as a Markdown comparison table.

        Trackers are ranked by AO (descending).

        Args:
            reports: One :class:`GOT10kReport` per tracker.

        Returns:
            Multi-line Markdown string ready for pasting into a README or paper.
        """
        ranked = sorted(reports, key=lambda r: r.ao, reverse=True)
        lines = [
            "| Rank | Tracker | AO | SR₅₀ | SR₇₅ | Sequences |",
            "|------|---------|---:|-----:|-----:|----------:|",
        ]
        for rank, r in enumerate(ranked, 1):
            lines.append(
                f"| {rank} | {r.tracker_name} | {r.ao:.4f} "
                f"| {r.sr50:.4f} | {r.sr75:.4f} | {r.num_sequences} |"
            )
        return "\n".join(lines)

    def save_report(self, report: GOT10kReport, output_dir: str) -> Path:
        """Save a :class:`GOT10kReport` as a JSON file.

        Args:
            report:     Report to save.
            output_dir: Destination directory (created if absent).

        Returns:
            Path to the written JSON file.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"got10k_{report.tracker_name}_{report.dataset_split}.json"
        with open(path, "w") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        return path

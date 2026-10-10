"""Cold-start analysis for edge-deployment tracker evaluation.

Edge devices (Raspberry Pi, Jetson Nano, microcontrollers) face a significant
cold-start penalty on the first tracking frame: model weights must be loaded,
memory must be allocated, and JIT compilers may trigger.  This cost is entirely
invisible when reporting only mean-update FPS.

:class:`ColdStartAnalyzer` consumes a :class:`~eovot.benchmark.engine.BenchmarkResult`
and produces a :class:`ColdStartReport` that separates first-frame initialization
latency from steady-state update latency, enabling researchers to make deployment
decisions with full cost visibility.

Typical usage::

    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.analysis.cold_start import ColdStartAnalyzer
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.kcf import KCFTracker

    dataset = SyntheticDataset(num_sequences=5, num_frames=50)
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(KCFTracker(), dataset, dataset_name="Synthetic")

    report = ColdStartAnalyzer().analyze(result)
    print(report)
    print(report.to_markdown())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult


@dataclass
class ColdStartReport:
    """Cold-start analysis summary for one tracker evaluated on a dataset.

    Attributes:
        tracker_name: Name of the evaluated tracker.
        dataset_name: Name of the evaluated dataset.
        mean_init_ms: Mean initialization latency across all sequences (ms).
        std_init_ms: Standard deviation of initialization latency (ms).
        max_init_ms: Worst-case initialization latency observed (ms).
        min_init_ms: Best-case initialization latency observed (ms).
        mean_update_ms: Mean per-frame update latency (ms).
        init_overhead_ratio: ``mean_init_ms / mean_update_ms`` — how many
            update frames worth of cost the cold start represents.
        n_sequences: Number of sequences in the analysis.
        per_sequence: Per-sequence breakdown mapping sequence name to a dict
            with ``init_latency_ms``, ``mean_update_ms``, and
            ``overhead_ratio``.
    """

    tracker_name: str
    dataset_name: str
    mean_init_ms: float
    std_init_ms: float
    max_init_ms: float
    min_init_ms: float
    mean_update_ms: float
    init_overhead_ratio: float
    n_sequences: int
    per_sequence: Dict[str, Dict] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"ColdStartReport[{self.tracker_name} on {self.dataset_name}]  "
            f"init={self.mean_init_ms:.2f}±{self.std_init_ms:.2f} ms  "
            f"(max={self.max_init_ms:.2f} ms)  "
            f"update={self.mean_update_ms:.2f} ms  "
            f"overhead={self.init_overhead_ratio:.1f}× update frames  "
            f"({self.n_sequences} sequences)"
        )

    def to_markdown(self) -> str:
        """Format the cold-start report as a Markdown string.

        Returns:
            Multi-line Markdown suitable for experiment reports.
        """
        lines = [
            f"## Cold-Start Analysis: {self.tracker_name} on {self.dataset_name}\n",
            "| Metric | Value |",
            "|--------|------:|",
            f"| Mean init latency (ms) | {self.mean_init_ms:.3f} |",
            f"| Std init latency (ms) | {self.std_init_ms:.3f} |",
            f"| Max init latency (ms) | {self.max_init_ms:.3f} |",
            f"| Min init latency (ms) | {self.min_init_ms:.3f} |",
            f"| Mean update latency (ms) | {self.mean_update_ms:.3f} |",
            f"| Init overhead ratio | {self.init_overhead_ratio:.2f}× |",
            f"| Sequences evaluated | {self.n_sequences} |",
            "",
            "### Per-Sequence Breakdown",
            "",
            "| Sequence | Init (ms) | Update (ms) | Overhead |",
            "|----------|----------:|------------:|---------:|",
        ]
        for seq_name, d in sorted(self.per_sequence.items()):
            lines.append(
                f"| {seq_name} "
                f"| {d['init_latency_ms']:.3f} "
                f"| {d['mean_update_ms']:.3f} "
                f"| {d['overhead_ratio']:.2f}× |"
            )
        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Return a JSON-serialisable plain dict."""
        return {
            "tracker_name": self.tracker_name,
            "dataset_name": self.dataset_name,
            "mean_init_ms": round(self.mean_init_ms, 3),
            "std_init_ms": round(self.std_init_ms, 3),
            "max_init_ms": round(self.max_init_ms, 3),
            "min_init_ms": round(self.min_init_ms, 3),
            "mean_update_ms": round(self.mean_update_ms, 3),
            "init_overhead_ratio": round(self.init_overhead_ratio, 3),
            "n_sequences": self.n_sequences,
            "per_sequence": {
                k: {kk: round(vv, 3) for kk, vv in v.items()}
                for k, v in self.per_sequence.items()
            },
        }


class ColdStartAnalyzer:
    """Analyze tracker cold-start cost from :class:`~eovot.benchmark.engine.BenchmarkResult`.

    The analyzer reads the ``init_latency_ms`` field recorded by
    :class:`~eovot.profiling.profiler.Profiler` during each sequence's first
    frame and compares it against the per-sequence mean update latency.

    Args:
        warn_threshold_ratio: If the mean init-to-update ratio exceeds this
            value, :meth:`analyze` prints a warning.  Default: ``10.0``
            (initialization ≥ 10× a single update frame).

    Example::

        analyzer = ColdStartAnalyzer(warn_threshold_ratio=5.0)
        report = analyzer.analyze(benchmark_result)
        print(report.to_markdown())
    """

    def __init__(self, warn_threshold_ratio: float = 10.0) -> None:
        self.warn_threshold_ratio = warn_threshold_ratio

    def analyze(self, result: "BenchmarkResult") -> ColdStartReport:
        """Build a :class:`ColdStartReport` from a completed benchmark run.

        Sequences for which ``init_latency_ms`` was not recorded (older JSON
        files) are silently skipped.

        Args:
            result: A :class:`~eovot.benchmark.engine.BenchmarkResult` from
                :meth:`~eovot.benchmark.engine.BenchmarkEngine.run`.

        Returns:
            :class:`ColdStartReport` with aggregate and per-sequence stats.

        Raises:
            ValueError: If no sequences with init-latency data are found.
        """
        init_times: List[float] = []
        update_times: List[float] = []
        per_sequence: Dict[str, Dict] = {}

        for sr in result.sequence_results:
            p = sr.profiling
            if p.init_latency_ms <= 0.0:
                continue
            init_ms = p.init_latency_ms
            upd_ms = p.latency_mean_ms
            ratio = (init_ms / upd_ms) if upd_ms > 0.0 else 0.0

            init_times.append(init_ms)
            update_times.append(upd_ms)
            per_sequence[sr.sequence_name] = {
                "init_latency_ms": init_ms,
                "mean_update_ms": upd_ms,
                "overhead_ratio": ratio,
            }

        if not init_times:
            raise ValueError(
                "No initialization-latency data found in the BenchmarkResult.  "
                "Re-run with a current BenchmarkEngine to record init times."
            )

        arr_init = np.array(init_times)
        mean_init = float(arr_init.mean())
        mean_update = float(np.mean(update_times))
        overhead_ratio = mean_init / mean_update if mean_update > 0.0 else 0.0

        if overhead_ratio >= self.warn_threshold_ratio:
            import warnings
            warnings.warn(
                f"[ColdStartAnalyzer] {result.tracker_name} init overhead is "
                f"{overhead_ratio:.1f}× a single update frame — high cold-start "
                f"cost may impact edge deployment latency SLAs.",
                RuntimeWarning,
                stacklevel=2,
            )

        return ColdStartReport(
            tracker_name=result.tracker_name,
            dataset_name=result.dataset_name,
            mean_init_ms=mean_init,
            std_init_ms=float(arr_init.std()),
            max_init_ms=float(arr_init.max()),
            min_init_ms=float(arr_init.min()),
            mean_update_ms=mean_update,
            init_overhead_ratio=overhead_ratio,
            n_sequences=len(init_times),
            per_sequence=per_sequence,
        )

    def compare(
        self, results: List["BenchmarkResult"]
    ) -> List[ColdStartReport]:
        """Analyze multiple benchmark results and return one report each.

        Args:
            results: List of :class:`~eovot.benchmark.engine.BenchmarkResult`,
                one per tracker.

        Returns:
            List of :class:`ColdStartReport` in the same order as *results*.
        """
        return [self.analyze(r) for r in results]

    def to_comparison_markdown(
        self, reports: List[ColdStartReport]
    ) -> str:
        """Format a multi-tracker cold-start comparison table.

        Args:
            reports: Output of :meth:`compare`.

        Returns:
            Markdown string with one row per tracker, sorted by mean init
            latency ascending.
        """
        if not reports:
            return "No cold-start reports to compare.\n"

        rows = sorted(reports, key=lambda r: r.mean_init_ms)
        lines = [
            "## Cold-Start Comparison\n",
            "| Tracker | Mean Init (ms) | Max Init (ms) | Update (ms) | Overhead |",
            "|---------|---------------:|--------------:|------------:|---------:|",
        ]
        for r in rows:
            lines.append(
                f"| {r.tracker_name} "
                f"| {r.mean_init_ms:.3f} "
                f"| {r.max_init_ms:.3f} "
                f"| {r.mean_update_ms:.3f} "
                f"| {r.init_overhead_ratio:.2f}× |"
            )
        lines.append("")
        return "\n".join(lines)

"""Tests for the MultiMetricReporter integration module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkResult, SequenceResult
from eovot.profiling.profiler import ProfilingResult
from eovot.reporting.multi_metric_report import MultiMetricReporter, TrackerMultiMetricSummary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profiling(name: str, fps: float = 30.0, mem_mb: float = 100.0) -> ProfilingResult:
    return ProfilingResult(
        tracker_name=name,
        frame_count=50,
        fps=fps,
        latency_mean_ms=1000.0 / fps if fps > 0 else 0.0,
        latency_std_ms=1.0,
        latency_p95_ms=1000.0 / fps + 5.0 if fps > 0 else 0.0,
        peak_memory_mb=mem_mb,
    )


def _make_sequence_result(
    seq_name: str,
    tracker_name: str,
    iou_value: float = 0.6,
    fps: float = 30.0,
    mem_mb: float = 100.0,
    n_frames: int = 50,
) -> SequenceResult:
    ious = np.full(n_frames, iou_value, dtype=np.float64)
    # Build simple moving predictions so temporal analysis has data
    preds = np.zeros((n_frames, 4), dtype=np.float64)
    for i in range(n_frames):
        preds[i] = [10.0 + i, 10.0, 40.0, 40.0]

    return SequenceResult(
        sequence_name=seq_name,
        ious=ious,
        profiling=_make_profiling(tracker_name, fps=fps, mem_mb=mem_mb),
        predictions=preds,
        ground_truths=preds.copy(),
    )


def _make_benchmark_result(
    tracker_name: str,
    dataset_name: str = "SyntheticDataset",
    iou_value: float = 0.6,
    fps: float = 30.0,
    mem_mb: float = 100.0,
    n_sequences: int = 3,
) -> BenchmarkResult:
    seq_results = [
        _make_sequence_result(
            f"seq_{i}", tracker_name, iou_value=iou_value, fps=fps, mem_mb=mem_mb
        )
        for i in range(n_sequences)
    ]
    return BenchmarkResult(
        tracker_name=tracker_name,
        dataset_name=dataset_name,
        sequence_results=seq_results,
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestMultiMetricReporter:

    def test_summarize_single_tracker(self):
        result = _make_benchmark_result("MOSSE")
        reporter = MultiMetricReporter(memory_budget_mb=512.0)
        summaries = reporter.summarize([result])

        assert len(summaries) == 1
        s = summaries[0]
        assert s.tracker_name == "MOSSE"
        assert s.dataset_name == "SyntheticDataset"
        assert s.num_sequences == 3
        assert 0.0 <= s.mean_iou <= 1.0
        assert s.mean_fps > 0.0
        assert s.peak_memory_mb > 0.0
        assert s.ees >= 0.0
        assert 0.0 <= s.mean_smoothness_score <= 1.0
        assert s.mean_eao >= 0.0

    def test_pareto_front_single_tracker(self):
        result = _make_benchmark_result("MOSSE")
        reporter = MultiMetricReporter()
        summaries = reporter.summarize([result])
        # A single tracker is always on the Pareto front
        assert summaries[0].on_pareto_front is True

    def test_pareto_front_two_trackers(self):
        """Tracker with higher IoU and higher EES should dominate the other."""
        high = _make_benchmark_result("Strong", iou_value=0.8, fps=60.0, mem_mb=80.0)
        weak = _make_benchmark_result("Weak", iou_value=0.3, fps=10.0, mem_mb=300.0)
        reporter = MultiMetricReporter()
        summaries = reporter.summarize([high, weak])

        by_name = {s.tracker_name: s for s in summaries}
        assert by_name["Strong"].on_pareto_front is True
        assert by_name["Weak"].on_pareto_front is False

    def test_to_markdown_contains_trackers(self):
        results = [
            _make_benchmark_result("MOSSE", iou_value=0.55, fps=120.0),
            _make_benchmark_result("KCF", iou_value=0.65, fps=60.0),
        ]
        reporter = MultiMetricReporter()
        summaries = reporter.summarize(results)
        md = reporter.to_markdown(summaries)

        assert "MOSSE" in md
        assert "KCF" in md
        assert "Accuracy" in md
        assert "Robustness" in md
        assert "Temporal" in md

    def test_to_dict_schema(self):
        result = _make_benchmark_result("MOSSE")
        reporter = MultiMetricReporter()
        summaries = reporter.summarize([result])
        d = reporter.to_dict(summaries)

        assert "schema_version" in d
        assert "trackers" in d
        assert len(d["trackers"]) == 1
        t = d["trackers"][0]
        assert "accuracy" in t
        assert "efficiency" in t
        assert "robustness" in t
        assert "temporal" in t

    def test_save_json_roundtrip(self):
        result = _make_benchmark_result("MOSSE")
        reporter = MultiMetricReporter()
        summaries = reporter.summarize([result])

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "report.json")
            reporter.save_json(summaries, out_path)
            loaded = json.loads(Path(out_path).read_text())

        assert loaded["trackers"][0]["tracker"] == "MOSSE"
        assert "accuracy" in loaded["trackers"][0]

    def test_empty_results(self):
        reporter = MultiMetricReporter()
        summaries = reporter.summarize([])
        assert summaries == []
        md = reporter.to_markdown(summaries)
        assert "No tracker" in md

    def test_tracker_multi_metric_summary_to_dict(self):
        s = TrackerMultiMetricSummary(
            tracker_name="KCF",
            dataset_name="OTB100",
            num_sequences=10,
            mean_iou=0.6,
            success_auc=0.55,
            mean_fps=45.0,
            ees=1.2,
        )
        d = s.to_dict()
        assert d["tracker"] == "KCF"
        assert d["accuracy"]["mean_iou"] == 0.6
        assert d["efficiency"]["mean_fps"] == 45.0

    def test_no_predictions_skips_temporal(self):
        """Sequences without stored predictions should not break temporal analysis."""
        seq = SequenceResult(
            sequence_name="seq_0",
            ious=np.full(30, 0.5),
            profiling=_make_profiling("MOSSE"),
            predictions=None,
        )
        result = BenchmarkResult(
            tracker_name="MOSSE",
            dataset_name="OTB100",
            sequence_results=[seq],
        )
        reporter = MultiMetricReporter()
        summaries = reporter.summarize([result])
        assert summaries[0].mean_smoothness_score == 0.0

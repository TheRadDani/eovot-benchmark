"""Tests for the self-contained HTML dashboard reporter."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkResult, SequenceResult
from eovot.profiling.profiler import ProfilingResult
from eovot.reporting.html_reporter import HTMLReporter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_profiling(fps: float = 100.0, lat: float = 10.0, mem: float = 50.0) -> ProfilingResult:
    return ProfilingResult(
        tracker_name="test",
        frame_count=20,
        fps=fps,
        latency_mean_ms=lat,
        latency_std_ms=0.5,
        latency_p95_ms=lat * 1.2,
        peak_memory_mb=mem,
    )


def _make_sequence_result(name: str = "seq_0", mean_iou: float = 0.7) -> SequenceResult:
    ious = np.full(30, mean_iou)
    return SequenceResult(
        sequence_name=name,
        ious=ious,
        profiling=_make_profiling(),
    )


def _make_benchmark_result(
    tracker_name: str = "MOSSE",
    dataset_name: str = "Synthetic",
    n_sequences: int = 3,
    mean_iou: float = 0.7,
) -> BenchmarkResult:
    result = BenchmarkResult(tracker_name=tracker_name, dataset_name=dataset_name)
    for i in range(n_sequences):
        result.sequence_results.append(
            _make_sequence_result(name=f"seq_{i:02d}", mean_iou=mean_iou)
        )
    return result


# ---------------------------------------------------------------------------
# Reporter instantiation
# ---------------------------------------------------------------------------

class TestHTMLReporterInit:
    def test_creates_output_dir(self, tmp_path):
        out = tmp_path / "my_results"
        HTMLReporter(output_dir=str(out))
        assert out.is_dir()

    def test_empty_results_raises(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        with pytest.raises(ValueError, match="at least one"):
            reporter.save_html([])


# ---------------------------------------------------------------------------
# HTML output structure
# ---------------------------------------------------------------------------

class TestHTMLOutput:
    def test_file_created(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        result = _make_benchmark_result()
        path = reporter.save_html([result], name="test_report")
        assert path.exists()
        assert path.suffix == ".html"

    def test_html_has_doctype(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        path = reporter.save_html([_make_benchmark_result()], name="r")
        content = path.read_text()
        assert content.startswith("<!DOCTYPE html>")

    def test_tracker_name_appears_in_html(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        result = _make_benchmark_result(tracker_name="SuperTracker")
        path = reporter.save_html([result])
        assert "SuperTracker" in path.read_text()

    def test_dataset_name_appears_in_html(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        result = _make_benchmark_result(dataset_name="MyDataset")
        path = reporter.save_html([result])
        assert "MyDataset" in path.read_text()

    def test_html_contains_svg_charts(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        path = reporter.save_html([_make_benchmark_result()])
        content = path.read_text()
        assert "<svg" in content
        assert "<polyline" in content   # success curve
        assert "<circle" in content     # scatter dot

    def test_html_contains_table(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        path = reporter.save_html([_make_benchmark_result()])
        content = path.read_text()
        assert "<table" in content
        assert "<thead" in content
        assert "<tbody" in content

    def test_html_is_self_contained(self, tmp_path):
        """No external CDN or script src references."""
        reporter = HTMLReporter(output_dir=str(tmp_path))
        path = reporter.save_html([_make_benchmark_result()])
        content = path.read_text()
        # No remote resource URLs
        assert "cdn.jsdelivr.net" not in content
        assert "unpkg.com" not in content
        assert 'src="http' not in content
        assert 'href="http' not in content

    def test_html_accordion_per_tracker(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        results = [
            _make_benchmark_result("MOSSE"),
            _make_benchmark_result("KCF"),
        ]
        path = reporter.save_html(results)
        content = path.read_text()
        assert content.count("<details") == 2

    def test_sequence_names_in_html(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        result = _make_benchmark_result(n_sequences=2)
        path = reporter.save_html([result])
        content = path.read_text()
        assert "seq_00" in content
        assert "seq_01" in content


# ---------------------------------------------------------------------------
# Multi-tracker comparison
# ---------------------------------------------------------------------------

class TestMultiTrackerHTMLReport:
    def test_multiple_trackers_all_present(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        trackers = ["MOSSE", "KCF", "CSRT"]
        results = [_make_benchmark_result(t, mean_iou=0.5 + 0.1 * i) for i, t in enumerate(trackers)]
        path = reporter.save_html(results, name="multi")
        content = path.read_text()
        for name in trackers:
            assert name in content

    def test_iou_values_appear_in_table(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        result = _make_benchmark_result(mean_iou=0.8765)
        path = reporter.save_html([result])
        content = path.read_text()
        # The formatted IoU should appear somewhere (possibly as 0.8765)
        assert "0.8765" in content

    def test_distinct_colours_per_tracker(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        results = [_make_benchmark_result(f"T{i}") for i in range(4)]
        path = reporter.save_html(results)
        content = path.read_text()
        # At least 3 distinct palette colours should appear
        colours_found = sum(1 for c in ["#2563eb", "#16a34a", "#dc2626", "#d97706"] if c in content)
        assert colours_found >= 3


# ---------------------------------------------------------------------------
# Integration: real BenchmarkEngine output
# ---------------------------------------------------------------------------

class TestHTMLReporterIntegration:
    def test_from_benchmark_engine(self, tmp_path):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        dataset = SyntheticDataset(num_sequences=2, num_frames=15, seed=0)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), dataset, dataset_name="Synthetic")

        reporter = HTMLReporter(output_dir=str(tmp_path))
        path = reporter.save_html([result], name="integration")
        assert path.exists()
        content = path.read_text()
        assert "MOSSE" in content
        assert "Synthetic" in content
        assert "<polyline" in content

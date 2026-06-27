"""Tests for eovot.reporting.html_reporter.HTMLReporter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkResult, SequenceResult
from eovot.profiling.profiler import ProfilingResult
from eovot.reporting.html_reporter import HTMLReporter, _esc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_profiling(fps: float = 100.0, mem_mb: float = 64.0) -> ProfilingResult:
    lat = 1_000.0 / fps
    return ProfilingResult(
        tracker_name="test",
        frame_count=50,
        fps=fps,
        latency_mean_ms=lat,
        latency_std_ms=lat * 0.1,
        latency_p95_ms=lat * 1.5,
        peak_memory_mb=mem_mb,
    )


def _make_seq_result(name: str, mean_iou: float, fps: float = 100.0) -> SequenceResult:
    ious = np.full(50, mean_iou, dtype=np.float64)
    return SequenceResult(
        sequence_name=name,
        ious=ious,
        profiling=_make_profiling(fps=fps),
    )


def _make_result(
    tracker: str = "MOSSE",
    dataset: str = "Synthetic",
    n_seqs: int = 3,
    mean_iou: float = 0.6,
    fps: float = 300.0,
) -> BenchmarkResult:
    result = BenchmarkResult(tracker_name=tracker, dataset_name=dataset)
    for i in range(n_seqs):
        result.sequence_results.append(
            _make_seq_result(f"seq_{i:02d}", mean_iou=mean_iou - i * 0.05, fps=fps)
        )
    return result


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestHTMLReporterConstruction:
    def test_creates_output_dir(self, tmp_path):
        out_dir = tmp_path / "nested" / "reports"
        reporter = HTMLReporter(output_dir=str(out_dir))
        assert out_dir.exists()

    def test_default_title(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        assert reporter.title == "EOVOT Benchmark Report"

    def test_custom_title(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path), title="My Report")
        assert reporter.title == "My Report"

    def test_empty_results_raises(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        with pytest.raises(ValueError, match="at least one"):
            reporter.save([], name="empty")


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------

class TestHTMLReporterOutput:
    def test_save_creates_html_file(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        result = _make_result()
        path = reporter.save([result], name="test_report")
        assert path.exists()
        assert path.suffix == ".html"

    def test_custom_name_used_in_filename(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        result = _make_result()
        path = reporter.save([result], name="my_custom_report")
        assert path.name == "my_custom_report.html"

    def test_output_is_valid_html(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        result = _make_result()
        path = reporter.save([result])
        content = path.read_text(encoding="utf-8")
        assert content.startswith("<!DOCTYPE html>")
        assert "</html>" in content

    def test_multiple_results_produce_one_file(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        results = [
            _make_result(tracker="MOSSE"),
            _make_result(tracker="KCF", fps=200.0),
        ]
        path = reporter.save(results, name="multi")
        assert path.exists()

    def test_returns_path_object(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        path = reporter.save([_make_result()])
        assert isinstance(path, Path)


# ---------------------------------------------------------------------------
# HTML content
# ---------------------------------------------------------------------------

class TestHTMLContent:
    def test_tracker_name_in_html(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        result = _make_result(tracker="MOSSE-Edge")
        content = reporter.save([result]).read_text()
        assert "MOSSE-Edge" in content

    def test_dataset_name_in_html(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        result = _make_result(dataset="OTB100")
        content = reporter.save([result]).read_text()
        assert "OTB100" in content

    def test_custom_title_in_html(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path), title="Edge VOT Report")
        content = reporter.save([_make_result()]).read_text()
        assert "Edge VOT Report" in content

    def test_chart_data_json_embedded(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        content = reporter.save([_make_result()]).read_text()
        assert "const DATA" in content

    def test_chartjs_cdn_referenced(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        content = reporter.save([_make_result()]).read_text()
        assert "chart.js" in content.lower()

    def test_sequence_names_in_html(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        result = _make_result(n_seqs=2)
        content = reporter.save([result]).read_text()
        assert "seq_00" in content
        assert "seq_01" in content

    def test_multiple_trackers_all_named_in_html(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        results = [
            _make_result(tracker="MOSSE"),
            _make_result(tracker="KCF"),
            _make_result(tracker="CSRT"),
        ]
        content = reporter.save(results).read_text()
        assert "MOSSE" in content
        assert "KCF" in content
        assert "CSRT" in content

    def test_chart_data_json_is_valid(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        result = _make_result()
        path = reporter.save([result])
        content = path.read_text()

        # Extract the JSON blob between 'const DATA = ' and ';'
        marker = "const DATA = "
        start = content.index(marker) + len(marker)
        # find the closing ';' after the JSON object
        depth = 0
        end = start
        for i, ch in enumerate(content[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        chart_json = json.loads(content[start:end])
        assert "success" in chart_json
        assert "fps" in chart_json
        assert "scatter" in chart_json
        assert "memory" in chart_json


# ---------------------------------------------------------------------------
# HTML escaping helper
# ---------------------------------------------------------------------------

class TestEscHelper:
    def test_ampersand(self):
        assert "&amp;" in _esc("A & B")

    def test_less_than(self):
        assert "&lt;" in _esc("<script>")

    def test_greater_than(self):
        assert "&gt;" in _esc("x > 0")

    def test_quote(self):
        assert "&quot;" in _esc('"quoted"')

    def test_plain_string_unchanged(self):
        assert _esc("hello world") == "hello world"


# ---------------------------------------------------------------------------
# max_sequences_in_table cap
# ---------------------------------------------------------------------------

class TestSequenceCap:
    def test_cap_limits_rows(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path), max_sequences_in_table=2)
        result = _make_result(n_seqs=5)
        content = reporter.save([result]).read_text()
        # Only 2 of the 5 sequences should appear
        assert content.count("seq_0") <= 2


# ---------------------------------------------------------------------------
# Integration: BenchmarkEngine → HTMLReporter
# ---------------------------------------------------------------------------

class TestHTMLReporterWithEngine:
    def test_full_pipeline(self, tmp_path):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        dataset = SyntheticDataset(num_sequences=2, num_frames=20, motion="linear")
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), dataset, dataset_name="Synthetic-Test")

        reporter = HTMLReporter(output_dir=str(tmp_path))
        path = reporter.save([result], name="engine_test")

        content = path.read_text()
        assert "MOSSE" in content
        assert "Synthetic-Test" in content
        assert "const DATA" in content

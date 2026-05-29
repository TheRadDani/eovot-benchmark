"""Tests for eovot.reporting.html_reporter.HTMLReporter."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkResult, SequenceResult
from eovot.profiling.profiler import ProfilingResult
from eovot.reporting.html_reporter import HTMLReporter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_profiling(tracker: str, fps: float = 120.0, mem: float = 20.0) -> ProfilingResult:
    return ProfilingResult(
        tracker_name=tracker,
        frame_count=50,
        latency_mean_ms=1000.0 / fps,
        latency_std_ms=0.1,
        latency_p95_ms=1000.0 / fps * 1.5,
        fps=fps,
        peak_memory_mb=mem,
    )


def _make_seq_result(
    name: str,
    tracker: str,
    n: int = 50,
    mean_iou: float = 0.6,
    fps: float = 100.0,
    with_cd: bool = True,
) -> SequenceResult:
    ious = np.full(n, mean_iou)
    gt = np.column_stack([
        np.zeros(n), np.zeros(n),
        np.full(n, 60.0), np.full(n, 40.0),
    ])
    preds = gt + np.random.default_rng(0).uniform(-5, 5, (n, 4))
    cd = np.full(n, 5.0) if with_cd else None
    return SequenceResult(
        sequence_name=name,
        ious=ious,
        profiling=_make_profiling(tracker, fps=fps),
        predictions=preds,
        ground_truths=gt,
        center_distances=cd,
    )


def _make_benchmark_result(tracker: str, fps: float = 100.0) -> BenchmarkResult:
    seqs = [
        _make_seq_result(f"seq_{i}", tracker, mean_iou=0.5 + i * 0.05, fps=fps)
        for i in range(3)
    ]
    return BenchmarkResult(
        tracker_name=tracker,
        dataset_name="Synthetic",
        sequence_results=seqs,
    )


# ---------------------------------------------------------------------------
# Basic generation tests
# ---------------------------------------------------------------------------

class TestHTMLReporterGeneration:
    def test_generate_returns_path(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        result = _make_benchmark_result("MOSSE")
        path = reporter.generate([result], name="test_report")
        assert path.exists()
        assert path.suffix == ".html"

    def test_output_dir_created(self, tmp_path):
        out = tmp_path / "nested" / "output"
        reporter = HTMLReporter(output_dir=str(out))
        result = _make_benchmark_result("MOSSE")
        reporter.generate([result])
        assert out.exists()

    def test_html_is_valid_structure(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        result = _make_benchmark_result("KCF")
        path = reporter.generate([result], name="r")
        html = path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert "<canvas" in html

    def test_title_in_html(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        result = _make_benchmark_result("MOSSE")
        path = reporter.generate([result], title="My Custom Title")
        html = path.read_text(encoding="utf-8")
        assert "My Custom Title" in html

    def test_tracker_name_in_html(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        for tracker in ["MOSSE", "KCF", "CSRT"]:
            result = _make_benchmark_result(tracker)
            path = reporter.generate([result], name=tracker.lower())
            html = path.read_text(encoding="utf-8")
            assert tracker in html

    def test_multiple_trackers(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        results = [_make_benchmark_result(t) for t in ["MOSSE", "KCF"]]
        path = reporter.generate(results, name="multi")
        html = path.read_text(encoding="utf-8")
        assert "MOSSE" in html
        assert "KCF" in html

    def test_json_data_embedded(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        result = _make_benchmark_result("MOSSE")
        path = reporter.generate([result])
        html = path.read_text(encoding="utf-8")
        # DATA object must be present for the JS to work
        assert "const DATA" in html

    def test_data_json_parseable(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        result = _make_benchmark_result("MOSSE")
        path = reporter.generate([result])
        html = path.read_text(encoding="utf-8")
        # Extract JSON from `const DATA = {...};`
        match = re.search(r"const DATA\s*=\s*(\{.*?\});", html, re.DOTALL)
        assert match is not None, "DATA JSON not found in HTML"
        data = json.loads(match.group(1))
        assert "trackers" in data
        assert "MOSSE" in data["trackers"]

    def test_success_curves_in_data(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        result = _make_benchmark_result("KCF")
        path = reporter.generate([result])
        html = path.read_text(encoding="utf-8")
        match = re.search(r"const DATA\s*=\s*(\{.*?\});", html, re.DOTALL)
        data = json.loads(match.group(1))
        assert len(data["success_curves"]) == 1
        assert len(data["iou_thresholds"]) == 101
        assert len(data["success_curves"][0]["rates"]) == 101

    def test_precision_curves_in_data(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        result = _make_benchmark_result("KCF")
        path = reporter.generate([result])
        html = path.read_text(encoding="utf-8")
        match = re.search(r"const DATA\s*=\s*(\{.*?\});", html, re.DOTALL)
        data = json.loads(match.group(1))
        assert len(data["precision_curves"]) == 1
        assert len(data["dist_thresholds"]) == 51

    def test_scatter_points_in_data(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        results = [_make_benchmark_result("MOSSE", fps=300), _make_benchmark_result("KCF", fps=150)]
        path = reporter.generate(results)
        html = path.read_text(encoding="utf-8")
        match = re.search(r"const DATA\s*=\s*(\{.*?\});", html, re.DOTALL)
        data = json.loads(match.group(1))
        fps_vals = {p["label"]: p["fps"] for p in data["scatter_points"]}
        assert abs(fps_vals["MOSSE"] - 300.0) < 1.0
        assert abs(fps_vals["KCF"] - 150.0) < 1.0

    def test_per_sequence_breakdown_in_data(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        result = _make_benchmark_result("MOSSE")
        path = reporter.generate([result])
        html = path.read_text(encoding="utf-8")
        match = re.search(r"const DATA\s*=\s*(\{.*?\});", html, re.DOTALL)
        data = json.loads(match.group(1))
        assert len(data["per_tracker_sequences"]) == 1
        assert len(data["per_tracker_sequences"][0]["sequences"]) == 3


# ---------------------------------------------------------------------------
# Energy reporting
# ---------------------------------------------------------------------------

class TestEnergyInReport:
    def test_no_energy_flag_false(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        result = _make_benchmark_result("MOSSE")
        path = reporter.generate([result])
        html = path.read_text(encoding="utf-8")
        match = re.search(r"const DATA\s*=\s*(\{.*?\});", html, re.DOTALL)
        data = json.loads(match.group(1))
        assert data["has_energy"] is False

    def test_energy_flag_true_when_present(self, tmp_path):
        from eovot.profiling.energy import EnergyResult

        reporter = HTMLReporter(output_dir=str(tmp_path))
        result = _make_benchmark_result("MOSSE")
        energy = EnergyResult(
            tracker_name="MOSSE",
            frame_count=50,
            tdp_watts=10.0,
            total_energy_j=0.05,
            mean_power_w=1.0,
            energy_per_frame_mj=1.0,
            peak_cpu_pct=20.0,
            mean_cpu_pct=10.0,
        )
        for sr in result.sequence_results:
            sr.energy = energy

        path = reporter.generate([result])
        html = path.read_text(encoding="utf-8")
        match = re.search(r"const DATA\s*=\s*(\{.*?\});", html, re.DOTALL)
        data = json.loads(match.group(1))
        assert data["has_energy"] is True


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

class TestColourPalette:
    def test_colours_assigned_per_tracker(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        results = [_make_benchmark_result(t) for t in ["A","B","C","D","E","F","G","H","I"]]
        path = reporter.generate(results)
        html = path.read_text(encoding="utf-8")
        match = re.search(r"const DATA\s*=\s*(\{.*?\});", html, re.DOTALL)
        data = json.loads(match.group(1))
        assert len(data["colors"]) == 9   # wraps around palette
        assert all(c.startswith("#") for c in data["colors"])

    def test_default_name(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path))
        result = _make_benchmark_result("MOSSE")
        path = reporter.generate([result])
        assert path.name == "report.html"

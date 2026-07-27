"""Tests for HTMLReportGenerator."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkResult, SequenceResult
from eovot.profiling.profiler import ProfilingResult
from eovot.metrics.accuracy import AccuracyMetrics
from eovot.reporting.html_report import HTMLReportGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_profiling(tracker_name: str = "T", fps: float = 120.0) -> ProfilingResult:
    return ProfilingResult(
        tracker_name=tracker_name,
        frame_count=50,
        fps=fps,
        latency_mean_ms=1000.0 / fps,
        latency_std_ms=0.5,
        latency_p95_ms=(1000.0 / fps) + 2.0,
        peak_memory_mb=80.0,
    )


def _fake_accuracy(iou: float = 0.6) -> AccuracyMetrics:
    return AccuracyMetrics(
        mean_iou=iou,
        success_auc=iou * 0.9,
        precision_auc=iou * 0.85,
    )


def _make_result(
    tracker_name: str = "MOSSE",
    dataset_name: str = "Synthetic",
    n_sequences: int = 3,
    mean_iou: float = 0.6,
) -> BenchmarkResult:
    """Build a minimal BenchmarkResult with synthetic data."""
    result = BenchmarkResult(tracker_name=tracker_name, dataset_name=dataset_name)
    rng = np.random.default_rng(42)
    for i in range(n_sequences):
        ious = rng.uniform(mean_iou - 0.1, mean_iou + 0.1, size=50).clip(0, 1)
        seq = SequenceResult(
            sequence_name=f"seq_{i:02d}",
            ious=ious,
            profiling=_fake_profiling(tracker_name),
            accuracy_metrics=_fake_accuracy(float(ious.mean())),
        )
        result.sequence_results.append(seq)
    return result


# ---------------------------------------------------------------------------
# HTMLReportGenerator tests
# ---------------------------------------------------------------------------

class TestHTMLReportGenerator:
    def test_empty_results_raises(self, tmp_path):
        gen = HTMLReportGenerator(output_dir=str(tmp_path))
        with pytest.raises(ValueError, match="empty"):
            gen.generate([], "report.html")

    def test_generates_html_file(self, tmp_path):
        gen = HTMLReportGenerator(output_dir=str(tmp_path))
        result = _make_result()
        path = gen.generate([result], "test_report.html")
        assert path.exists()
        assert path.suffix == ".html"

    def test_html_file_is_valid_utf8(self, tmp_path):
        gen = HTMLReportGenerator(output_dir=str(tmp_path))
        path = gen.generate([_make_result()], "report.html")
        content = path.read_text(encoding="utf-8")
        assert len(content) > 0

    def test_html_contains_tracker_names(self, tmp_path):
        gen = HTMLReportGenerator(output_dir=str(tmp_path))
        results = [
            _make_result("MOSSE", "Synthetic"),
            _make_result("KCF", "Synthetic"),
        ]
        path = gen.generate(results, "report.html")
        content = path.read_text(encoding="utf-8")
        assert "MOSSE" in content
        assert "KCF" in content

    def test_html_contains_dataset_name(self, tmp_path):
        gen = HTMLReportGenerator(output_dir=str(tmp_path))
        path = gen.generate([_make_result(dataset_name="MyDataset")], "report.html")
        content = path.read_text(encoding="utf-8")
        assert "MyDataset" in content

    def test_html_contains_doctype(self, tmp_path):
        gen = HTMLReportGenerator(output_dir=str(tmp_path))
        path = gen.generate([_make_result()], "report.html")
        content = path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content

    def test_html_contains_table(self, tmp_path):
        gen = HTMLReportGenerator(output_dir=str(tmp_path))
        path = gen.generate([_make_result()], "report.html")
        content = path.read_text(encoding="utf-8")
        assert "<table>" in content
        assert "<thead>" in content
        assert "<tbody>" in content

    def test_html_contains_sortable_js(self, tmp_path):
        gen = HTMLReportGenerator(output_dir=str(tmp_path))
        path = gen.generate([_make_result()], "report.html")
        content = path.read_text(encoding="utf-8")
        assert "data-sortable" in content
        assert "<script>" in content

    def test_sequence_rows_present(self, tmp_path):
        gen = HTMLReportGenerator(output_dir=str(tmp_path))
        result = _make_result(n_sequences=4)
        path = gen.generate([result], "report.html")
        content = path.read_text(encoding="utf-8")
        # Each sequence name should appear in the per-sequence table
        assert "seq_00" in content
        assert "seq_03" in content

    def test_appends_html_extension_when_absent(self, tmp_path):
        gen = HTMLReportGenerator(output_dir=str(tmp_path))
        path = gen.generate([_make_result()], "no_ext")
        assert path.suffix == ".html"

    def test_custom_title_appears_in_html(self, tmp_path):
        gen = HTMLReportGenerator(output_dir=str(tmp_path))
        path = gen.generate([_make_result()], "r.html", title="My Custom Report")
        content = path.read_text(encoding="utf-8")
        assert "My Custom Report" in content

    def test_multi_tracker_both_appear_in_table(self, tmp_path):
        gen = HTMLReportGenerator(output_dir=str(tmp_path))
        results = [_make_result("TrackerA"), _make_result("TrackerB")]
        path = gen.generate(results, "r.html")
        content = path.read_text(encoding="utf-8")
        assert "TrackerA" in content
        assert "TrackerB" in content

    def test_output_dir_created_automatically(self, tmp_path):
        new_dir = tmp_path / "nested" / "reports"
        gen = HTMLReportGenerator(output_dir=str(new_dir))
        path = gen.generate([_make_result()], "r.html")
        assert path.exists()

    def test_iou_badge_classes_present(self, tmp_path):
        gen = HTMLReportGenerator(output_dir=str(tmp_path))
        # Good IoU (>= 0.55)
        path = gen.generate([_make_result(mean_iou=0.65)], "r.html")
        content = path.read_text(encoding="utf-8")
        assert "badge-good" in content

    def test_result_with_energy_shows_value(self, tmp_path):
        from eovot.profiling.energy import EnergyResult

        result = _make_result()
        for seq in result.sequence_results:
            seq.energy = EnergyResult(
                tracker_name="MOSSE",
                frame_count=50,
                tdp_watts=6.0,
                total_energy_j=0.05,
                mean_power_w=3.0,
                energy_per_frame_mj=1.0,
                peak_cpu_pct=50.0,
                mean_cpu_pct=40.0,
            )

        gen = HTMLReportGenerator(output_dir=str(tmp_path))
        path = gen.generate([result], "r.html")
        content = path.read_text(encoding="utf-8")
        # Energy total should appear (not "—")
        assert "0." in content  # some decimal number

    def test_integration_with_benchmark_engine(self, tmp_path):
        """Full pipeline: run engine on synthetic data, generate HTML report."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker
        from eovot.trackers.kcf import KCFTracker

        dataset = SyntheticDataset(num_sequences=2, num_frames=25, motion="linear")
        engine = BenchmarkEngine(verbose=False)

        results = [
            engine.run(MOSSETracker(), dataset, dataset_name="Synthetic"),
            engine.run(KCFTracker(), dataset, dataset_name="Synthetic"),
        ]

        gen = HTMLReportGenerator(output_dir=str(tmp_path))
        path = gen.generate(results, "integration_report.html")

        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "MOSSE" in content
        assert "KCF" in content
        # Should have valid IoU values (not empty)
        assert re.search(r"\d\.\d{3}", content)

"""Tests for the self-contained HTML report generator."""

import re
from pathlib import Path

import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkResult, SequenceResult
from eovot.profiling.profiler import ProfilingResult
from eovot.metrics.accuracy import AccuracyMetrics
from eovot.reporting.html_reporter import HTMLReporter


# ---------------------------------------------------------------------------
# Minimal BenchmarkResult builder
# ---------------------------------------------------------------------------


def _make_profiling(fps: float = 30.0, mem_mb: float = 128.0) -> ProfilingResult:
    lat = 1000.0 / fps if fps > 0 else 0.0
    return ProfilingResult(
        tracker_name="t",
        frame_count=20,
        fps=fps,
        latency_mean_ms=lat,
        latency_std_ms=1.0,
        latency_p95_ms=lat * 1.2,
        peak_memory_mb=mem_mb,
    )


def _make_sequence_result(
    name: str = "seq_001",
    n_frames: int = 20,
    mean_iou: float = 0.7,
    fps: float = 30.0,
    mem_mb: float = 128.0,
) -> SequenceResult:
    bbox = np.array([10.0, 10.0, 30.0, 30.0])
    gt = np.tile(bbox, (n_frames, 1))
    # Add small noise to predictions so precision curve is non-trivial.
    preds = gt + np.random.default_rng(0).uniform(-5, 5, gt.shape)
    # Clip negative dimensions.
    preds[:, 2:] = np.maximum(preds[:, 2:], 1.0)

    # Manually set ious to the desired mean.
    ious = np.full(n_frames, mean_iou)

    return SequenceResult(
        sequence_name=name,
        ious=ious,
        profiling=_make_profiling(fps=fps, mem_mb=mem_mb),
        predictions=preds,
        ground_truths=gt,
        accuracy_metrics=AccuracyMetrics(
            mean_iou=mean_iou,
            success_auc=mean_iou * 0.9,
            precision_auc=mean_iou * 0.85,
        ),
    )


def _make_result(
    tracker_name: str = "MOSSE",
    dataset_name: str = "Synthetic",
    n_seqs: int = 2,
    mean_iou: float = 0.7,
    fps: float = 500.0,
    mem_mb: float = 64.0,
) -> BenchmarkResult:
    result = BenchmarkResult(tracker_name=tracker_name, dataset_name=dataset_name)
    for i in range(n_seqs):
        result.sequence_results.append(
            _make_sequence_result(
                name=f"seq_{i:03d}",
                mean_iou=mean_iou,
                fps=fps,
                mem_mb=mem_mb,
            )
        )
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_reporter(tmp_path):
    return HTMLReporter(output_dir=str(tmp_path))


@pytest.fixture()
def single_result():
    return [_make_result("MOSSE")]


@pytest.fixture()
def multi_result():
    return [
        _make_result("MOSSE", fps=500.0, mem_mb=64.0, mean_iou=0.55),
        _make_result("KCF", fps=200.0, mem_mb=96.0, mean_iou=0.60),
        _make_result("CSRT", fps=40.0, mem_mb=160.0, mean_iou=0.70),
    ]


# ---------------------------------------------------------------------------
# Output file creation
# ---------------------------------------------------------------------------


class TestOutputFile:
    def test_returns_path(self, tmp_reporter, single_result):
        path = tmp_reporter.generate(single_result)
        assert isinstance(path, Path)

    def test_file_exists(self, tmp_reporter, single_result):
        path = tmp_reporter.generate(single_result)
        assert path.exists()

    def test_has_html_extension(self, tmp_reporter, single_result):
        path = tmp_reporter.generate(single_result)
        assert path.suffix == ".html"

    def test_custom_name(self, tmp_reporter, single_result):
        path = tmp_reporter.generate(single_result, name="my_report")
        assert path.name == "my_report.html"

    def test_output_dir_created(self, tmp_path):
        subdir = tmp_path / "nested" / "reports"
        reporter = HTMLReporter(output_dir=str(subdir))
        path = reporter.generate([_make_result("MOSSE")])
        assert subdir.exists()
        assert path.exists()


# ---------------------------------------------------------------------------
# HTML structure
# ---------------------------------------------------------------------------


class TestHTMLStructure:
    def _html(self, reporter, results, title="Test Report"):
        path = reporter.generate(results, title=title)
        return path.read_text(encoding="utf-8")

    def test_doctype(self, tmp_reporter, single_result):
        html = self._html(tmp_reporter, single_result)
        assert html.startswith("<!DOCTYPE html>")

    def test_contains_title(self, tmp_reporter, single_result):
        html = self._html(tmp_reporter, single_result, title="My EOVOT Report")
        assert "My EOVOT Report" in html

    def test_contains_tracker_name(self, tmp_reporter, single_result):
        html = self._html(tmp_reporter, single_result)
        assert "MOSSE" in html

    def test_contains_dataset_name(self, tmp_reporter, single_result):
        html = self._html(tmp_reporter, single_result)
        assert "Synthetic" in html

    def test_contains_multiple_tracker_names(self, tmp_reporter, multi_result):
        html = self._html(tmp_reporter, multi_result)
        for name in ("MOSSE", "KCF", "CSRT"):
            assert name in html

    def test_no_external_resources(self, tmp_reporter, single_result):
        """The report must be fully self-contained (no external CSS/JS/image loads).

        Navigation <a href="https://..."> links are permitted; only resource
        loads (script src, link href, img src) would break offline viewing.
        """
        html = self._html(tmp_reporter, single_result)
        # Matches <script src="http...">, <link href="http...">, <img src="http...">
        # but NOT <a href="http..."> which is a navigation link.
        resource_pattern = re.compile(
            r'<(script|link|img)[^>]+(src|href)=["\']https?://',
            re.IGNORECASE,
        )
        assert not resource_pattern.search(html)

    def test_contains_base64_image(self, tmp_reporter, single_result):
        """Charts should be embedded as base64 data URIs."""
        html = self._html(tmp_reporter, single_result)
        assert "data:image/png;base64," in html

    def test_utf8_encoding(self, tmp_reporter, single_result):
        """File is written with UTF-8 encoding."""
        path = tmp_reporter.generate(single_result)
        content = path.read_bytes()
        content.decode("utf-8")  # Should not raise.

    def test_contains_eovot_attribution(self, tmp_reporter, single_result):
        html = self._html(tmp_reporter, single_result)
        assert "EOVOT" in html


# ---------------------------------------------------------------------------
# Summary table content
# ---------------------------------------------------------------------------


class TestSummaryTable:
    def test_contains_table_element(self, tmp_reporter, single_result):
        path = tmp_reporter.generate(single_result)
        html = path.read_text()
        assert "<table>" in html.lower() or "<table" in html

    def test_contains_miou_column(self, tmp_reporter, single_result):
        path = tmp_reporter.generate(single_result)
        html = path.read_text()
        assert "mIoU" in html or "miou" in html.lower()

    def test_contains_fps_column(self, tmp_reporter, single_result):
        path = tmp_reporter.generate(single_result)
        html = path.read_text()
        assert "FPS" in html

    def test_contains_ees_column(self, tmp_reporter, single_result):
        path = tmp_reporter.generate(single_result)
        html = path.read_text()
        assert "EES" in html

    def test_contains_pareto_column(self, tmp_reporter, multi_result):
        path = tmp_reporter.generate(multi_result)
        html = path.read_text()
        assert "Pareto" in html

    def test_empty_results_no_crash(self, tmp_reporter):
        path = tmp_reporter.generate([])
        html = path.read_text()
        assert "No results" in html or "<p" in html


# ---------------------------------------------------------------------------
# Multiple charts embedded
# ---------------------------------------------------------------------------


class TestCharts:
    def test_multiple_base64_images(self, tmp_reporter, multi_result):
        path = tmp_reporter.generate(multi_result)
        html = path.read_text()
        count = html.count("data:image/png;base64,")
        # Expect at least 3 charts: success, precision, efficiency
        assert count >= 3

    def test_success_curve_section(self, tmp_reporter, single_result):
        path = tmp_reporter.generate(single_result)
        html = path.read_text()
        assert "Success" in html

    def test_precision_curve_section(self, tmp_reporter, single_result):
        path = tmp_reporter.generate(single_result)
        html = path.read_text()
        assert "Precision" in html

    def test_efficiency_section(self, tmp_reporter, single_result):
        path = tmp_reporter.generate(single_result)
        html = path.read_text()
        assert "Efficienc" in html  # Matches "Efficiency" or "Efficiency Analysis"


# ---------------------------------------------------------------------------
# Memory budget and EES computation
# ---------------------------------------------------------------------------


class TestEfficiencyIntegration:
    def test_custom_memory_budget(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path), memory_budget_mb=256.0)
        assert reporter._eff_engine.memory_budget_mb == 256.0

    def test_report_generated_with_custom_budget(self, tmp_path):
        reporter = HTMLReporter(output_dir=str(tmp_path), memory_budget_mb=256.0)
        results = [_make_result("MOSSE"), _make_result("KCF")]
        path = reporter.generate(results)
        assert path.exists()


# ---------------------------------------------------------------------------
# Handling edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_tracker(self, tmp_reporter):
        """Single tracker produces valid HTML."""
        results = [_make_result("MOSSE")]
        path = tmp_reporter.generate(results)
        assert path.exists()
        html = path.read_text()
        assert "MOSSE" in html

    def test_single_sequence_per_tracker(self, tmp_reporter):
        results = [_make_result("MOSSE", n_seqs=1)]
        path = tmp_reporter.generate(results)
        assert path.exists()

    def test_sequences_without_predictions(self, tmp_reporter):
        """If predictions are None, precision curve falls back gracefully."""
        result = _make_result("MOSSE")
        for r in result.sequence_results:
            r.predictions = None
            r.ground_truths = None
        path = tmp_reporter.generate([result])
        assert path.exists()

    def test_overwrite_existing_file(self, tmp_reporter, single_result):
        """Calling generate() twice overwrites the output file without error."""
        path1 = tmp_reporter.generate(single_result, name="overwrite_test")
        path2 = tmp_reporter.generate(single_result, name="overwrite_test")
        assert path1 == path2
        assert path1.exists()


# ---------------------------------------------------------------------------
# Exported from reporting package
# ---------------------------------------------------------------------------


def test_exported_from_reporting_package():
    from eovot.reporting import HTMLReporter as HR
    assert HR is HTMLReporter

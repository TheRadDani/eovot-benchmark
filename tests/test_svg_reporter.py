"""Tests for eovot.reporting.svg_reporter (SvgHtmlReporter)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from eovot.reporting.svg_reporter import (
    SvgHtmlReporter,
    _chart_bars,
    _chart_efficiency_scatter,
    _chart_success_curves,
    _collect_ious,
    _compute_success_curve,
    _summary_table,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_result(
    tracker: str = "MOSSE",
    dataset: str = "Synthetic",
    mean_iou: float = 0.55,
    fps: float = 120.0,
    mem_mb: float = 80.0,
    latency_ms: float = 8.3,
    success_auc: float = 0.48,
    precision_auc: float = 0.60,
    n_frames: int = 50,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed=abs(hash(tracker)) % (2 ** 32))
    ious = np.clip(rng.normal(mean_iou, 0.15, n_frames), 0, 1).tolist()
    return {
        "summary": {
            "tracker": tracker,
            "dataset": dataset,
            "mean_iou": mean_iou,
            "mean_fps": fps,
            "peak_memory_mb": mem_mb,
            "mean_latency_ms": latency_ms,
            "success_auc": success_auc,
            "precision_auc": precision_auc,
            "num_sequences": 5,
        },
        "sequences": [
            {"sequence_name": f"seq{i}", "ious": ious, "mean_iou": mean_iou}
            for i in range(5)
        ],
    }


@pytest.fixture
def two_results():
    return [
        _make_result("MOSSE", mean_iou=0.50, fps=200.0, mem_mb=60.0),
        _make_result("KCF",   mean_iou=0.60, fps=150.0, mem_mb=90.0),
    ]


@pytest.fixture
def single_result():
    return [_make_result("CSRT", mean_iou=0.70, fps=30.0, mem_mb=120.0)]


# ---------------------------------------------------------------------------
# _collect_ious
# ---------------------------------------------------------------------------

class TestCollectIous:
    def test_collects_from_ious_list(self):
        result = _make_result(n_frames=20)
        arr = _collect_ious(result)
        assert len(arr) == 5 * 20

    def test_falls_back_to_mean_iou_scalar(self):
        result = {
            "summary": {"mean_iou": 0.4},
            "sequences": [{"sequence_name": "s", "mean_iou": 0.4}],
        }
        arr = _collect_ious(result)
        assert float(arr[0]) == pytest.approx(0.4)

    def test_empty_sequences_returns_zero_array(self):
        result = {"summary": {}, "sequences": []}
        arr = _collect_ious(result)
        assert float(arr[0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _compute_success_curve
# ---------------------------------------------------------------------------

class TestSuccessCurve:
    def test_output_shape(self):
        ious = np.linspace(0, 1, 100)
        t, r = _compute_success_curve(ious, n_points=101)
        assert t.shape == (101,) and r.shape == (101,)

    def test_rates_in_zero_one(self):
        ious = np.random.default_rng(0).uniform(0, 1, 200)
        _, rates = _compute_success_curve(ious)
        assert (rates >= 0).all() and (rates <= 1).all()

    def test_monotone_non_increasing(self):
        ious = np.linspace(0.1, 0.9, 300)
        _, rates = _compute_success_curve(ious)
        assert all(rates[i] >= rates[i + 1] - 1e-12 for i in range(len(rates) - 1))

    def test_perfect_tracker_auc_is_one(self):
        ious = np.ones(100)
        t, r = _compute_success_curve(ious)
        assert float(np.trapezoid(r, t) if hasattr(np, "trapezoid") else np.trapz(r, t)) == pytest.approx(1.0, abs=0.02)


# ---------------------------------------------------------------------------
# SVG chart helpers
# ---------------------------------------------------------------------------

class TestChartHelpers:
    def test_success_curves_is_svg(self, two_results):
        svg = _chart_success_curves(two_results)
        assert svg.startswith("<svg")
        assert "</svg>" in svg

    def test_success_curves_contains_tracker_names(self, two_results):
        svg = _chart_success_curves(two_results)
        assert "MOSSE" in svg
        assert "KCF" in svg

    def test_efficiency_scatter_is_svg(self, two_results):
        svg = _chart_efficiency_scatter(two_results)
        assert "<svg" in svg
        assert "<circle" in svg

    def test_efficiency_scatter_single_tracker(self, single_result):
        svg = _chart_efficiency_scatter(single_result)
        assert "<circle" in svg

    def test_bar_chart_is_svg(self, two_results):
        svg = _chart_bars(two_results, "mean_iou", "Mean IoU", "mIoU")
        assert "<svg" in svg
        assert "<rect" in svg

    def test_bar_chart_single_tracker(self, single_result):
        svg = _chart_bars(single_result, "mean_fps", "Throughput", "FPS")
        assert "CSRT" in svg

    def test_no_external_resources_in_charts(self, two_results):
        for svg in (
            _chart_success_curves(two_results),
            _chart_efficiency_scatter(two_results),
            _chart_bars(two_results, "mean_iou", "Mean IoU", "mIoU"),
        ):
            # The SVG namespace URI is expected; CDN or script src refs are not
            assert "cdn" not in svg.lower()
            assert 'src="http' not in svg
            assert "jsdelivr" not in svg
            assert "cloudflare" not in svg


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

class TestSummaryTable:
    def test_returns_table_element(self, two_results):
        tbl = _summary_table(two_results)
        assert tbl.startswith("<table")
        assert "</table>" in tbl

    def test_tracker_names_in_table(self, two_results):
        tbl = _summary_table(two_results)
        assert "MOSSE" in tbl
        assert "KCF" in tbl

    def test_best_value_highlighted(self, two_results):
        # KCF has higher mean_iou (0.60); must be highlighted
        tbl = _summary_table(two_results)
        assert "font-weight:bold" in tbl

    def test_missing_optional_field_renders_dash(self):
        result = {"summary": {"tracker": "T"}, "sequences": []}
        tbl = _summary_table([result])
        assert "—" in tbl


# ---------------------------------------------------------------------------
# SvgHtmlReporter
# ---------------------------------------------------------------------------

class TestSvgHtmlReporter:
    def test_creates_html_file(self, two_results, tmp_path):
        reporter = SvgHtmlReporter(output_dir=str(tmp_path))
        path = reporter.generate(two_results, name="test")
        assert path.exists() and path.suffix == ".html"

    def test_output_is_valid_html(self, two_results, tmp_path):
        reporter = SvgHtmlReporter(output_dir=str(tmp_path))
        path = reporter.generate(two_results, name="valid")
        content = path.read_text(encoding="utf-8")
        assert "<!doctype html>" in content.lower()
        assert "</html>" in content

    def test_custom_title_in_output(self, two_results, tmp_path):
        reporter = SvgHtmlReporter(output_dir=str(tmp_path))
        path = reporter.generate(two_results, name="t", title="My Custom Title")
        assert "My Custom Title" in path.read_text(encoding="utf-8")

    def test_default_title_includes_name(self, two_results, tmp_path):
        reporter = SvgHtmlReporter(output_dir=str(tmp_path))
        path = reporter.generate(two_results, name="sweep_abc")
        assert "sweep_abc" in path.read_text(encoding="utf-8")

    def test_tracker_names_in_output(self, two_results, tmp_path):
        reporter = SvgHtmlReporter(output_dir=str(tmp_path))
        content = reporter.generate(two_results, name="n").read_text()
        assert "MOSSE" in content and "KCF" in content

    def test_output_dir_created(self, tmp_path):
        new_dir = tmp_path / "a" / "b"
        SvgHtmlReporter(output_dir=str(new_dir))
        assert new_dir.exists()

    def test_empty_results_raises(self, tmp_path):
        reporter = SvgHtmlReporter(output_dir=str(tmp_path))
        with pytest.raises(ValueError, match="empty"):
            reporter.generate([], name="empty")

    def test_single_tracker(self, single_result, tmp_path):
        reporter = SvgHtmlReporter(output_dir=str(tmp_path))
        content = reporter.generate(single_result, name="single").read_text()
        assert "CSRT" in content

    def test_no_external_links(self, two_results, tmp_path):
        reporter = SvgHtmlReporter(output_dir=str(tmp_path))
        content = reporter.generate(two_results, name="noext").read_text()
        assert 'src="http' not in content
        assert 'href="http' not in content
        assert "cdn.jsdelivr.net" not in content
        assert "cdnjs.cloudflare.com" not in content

    def test_multiple_svg_elements_embedded(self, two_results, tmp_path):
        reporter = SvgHtmlReporter(output_dir=str(tmp_path))
        content = reporter.generate(two_results, name="svgs").read_text()
        assert content.count("<svg") >= 4

    def test_file_size_sanity(self, two_results, tmp_path):
        reporter = SvgHtmlReporter(output_dir=str(tmp_path))
        path = reporter.generate(two_results, name="size")
        size_kb = path.stat().st_size / 1024
        assert 2 < size_kb < 500

    def test_generate_from_json_files(self, two_results, tmp_path):
        json_paths = []
        for i, res in enumerate(two_results):
            p = tmp_path / f"r{i}.json"
            p.write_text(json.dumps(res), encoding="utf-8")
            json_paths.append(str(p))
        reporter = SvgHtmlReporter(output_dir=str(tmp_path))
        path = reporter.generate_from_json_files(json_paths, name="from_json")
        assert path.exists()
        content = path.read_text()
        assert "MOSSE" in content and "KCF" in content

    def test_many_trackers(self, tmp_path):
        results = [
            _make_result(f"Tracker{i}", mean_iou=0.3 + i * 0.05, fps=200 - i * 20)
            for i in range(8)
        ]
        reporter = SvgHtmlReporter(output_dir=str(tmp_path))
        content = reporter.generate(results, name="many").read_text()
        for i in range(8):
            assert f"Tracker{i}" in content

    def test_bare_result_no_optional_fields(self, tmp_path):
        result = {
            "summary": {"tracker": "BareT", "mean_iou": 0.4, "mean_fps": 50.0,
                        "peak_memory_mb": 100.0},
            "sequences": [],
        }
        reporter = SvgHtmlReporter(output_dir=str(tmp_path))
        path = reporter.generate([result], name="bare")
        assert path.exists()

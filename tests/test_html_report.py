"""Tests for eovot.reporting.html_report.HTMLReportGenerator."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from eovot.reporting.html_report import (
    HTMLReportGenerator,
    _build_fps_iou_scatter,
    _build_latency_bars,
    _build_leaderboard_html,
    _build_precision_curves,
    _build_success_curves,
)
from eovot.reporting.reporter import BenchmarkReporter


# ---------------------------------------------------------------------------
# Fixtures — synthetic benchmark result dicts
# ---------------------------------------------------------------------------

def _make_result(
    tracker: str,
    dataset: str = "Synthetic",
    n_seq: int = 3,
    mean_iou: float = 0.65,
    fps: float = 120.0,
    success_auc: float = 0.60,
    precision_auc: float = 0.75,
    include_latency: bool = True,
    include_center_distances: bool = True,
    tdp_watts: float = 0.0,
) -> Dict[str, Any]:
    """Build a minimal benchmark result dict like BenchmarkResult.to_dict()."""
    seqs = []
    for i in range(n_seq):
        seq: Dict[str, Any] = {
            "sequence_name": f"seq_{i:02d}",
            "mean_iou": mean_iou,
            "fps": fps,
            "peak_memory_mb": 50.0,
            "success_auc": success_auc,
            "precision_auc": precision_auc,
            # Per-frame IoU array (10 frames)
            "ious": [max(0.0, mean_iou + 0.05 * (j - 5) / 5) for j in range(10)],
        }
        if include_latency:
            seq["mean_latency_ms"] = 1000.0 / fps
        if include_center_distances:
            seq["center_distances"] = [float(j * 2) for j in range(10)]
        if tdp_watts > 0:
            seq["energy_j"] = 0.01
            seq["energy_per_frame_mj"] = 1.0
        seqs.append(seq)

    summary: Dict[str, Any] = {
        "tracker": tracker,
        "dataset": dataset,
        "num_sequences": n_seq,
        "mean_iou": mean_iou,
        "mean_fps": fps,
        "peak_memory_mb": 55.0,
        "success_auc": success_auc,
        "precision_auc": precision_auc,
    }
    if tdp_watts > 0:
        summary["mean_energy_per_frame_mj"] = 1.0

    return {"summary": summary, "sequences": seqs}


@pytest.fixture
def two_results() -> List[Dict[str, Any]]:
    return [
        _make_result("MOSSE", fps=350.0, mean_iou=0.45, success_auc=0.42),
        _make_result("KCF",   fps=180.0, mean_iou=0.55, success_auc=0.53),
    ]


@pytest.fixture
def single_result() -> List[Dict[str, Any]]:
    return [_make_result("MOSSE")]


# ---------------------------------------------------------------------------
# HTMLReportGenerator.generate()
# ---------------------------------------------------------------------------

def test_generate_returns_string(two_results: List[Dict]) -> None:
    gen = HTMLReportGenerator()
    html = gen.generate(two_results)
    assert isinstance(html, str)


def test_generate_starts_with_doctype(two_results: List[Dict]) -> None:
    gen = HTMLReportGenerator()
    html = gen.generate(two_results)
    assert html.strip().startswith("<!DOCTYPE html>")


def test_generate_contains_title(two_results: List[Dict]) -> None:
    title = "My Custom Report"
    gen = HTMLReportGenerator(title=title)
    html = gen.generate(two_results)
    assert title in html


def test_generate_contains_tracker_names(two_results: List[Dict]) -> None:
    gen = HTMLReportGenerator()
    html = gen.generate(two_results)
    assert "MOSSE" in html
    assert "KCF" in html


def test_generate_contains_leaderboard_table(two_results: List[Dict]) -> None:
    gen = HTMLReportGenerator()
    html = gen.generate(two_results)
    assert '<table class="leaderboard">' in html


def test_generate_contains_svg(two_results: List[Dict]) -> None:
    gen = HTMLReportGenerator()
    html = gen.generate(two_results)
    assert "<svg" in html


def test_generate_valid_html_structure(two_results: List[Dict]) -> None:
    gen = HTMLReportGenerator()
    html = gen.generate(two_results)
    assert "<html" in html
    assert "<head>" in html
    assert "<body>" in html
    assert "</html>" in html


def test_generate_no_external_src(two_results: List[Dict]) -> None:
    """The report must be fully self-contained — no external resource URLs."""
    gen = HTMLReportGenerator()
    html = gen.generate(two_results)
    # No CDN links or external script sources.
    assert "cdn." not in html
    assert 'src="http' not in html
    assert 'href="http' not in html


def test_generate_success_auc_in_html(two_results: List[Dict]) -> None:
    gen = HTMLReportGenerator()
    html = gen.generate(two_results)
    # Both AUC values should appear somewhere in the output.
    assert "0.4200" in html or "0.42" in html
    assert "0.5300" in html or "0.53" in html


def test_generate_with_energy(tmp_path: Path) -> None:
    results = [_make_result("MOSSE", tdp_watts=15.0)]
    gen = HTMLReportGenerator()
    html = gen.generate(results)
    # Energy column header should be present
    assert "mJ" in html or "Energy" in html


def test_generate_single_tracker(single_result: List[Dict]) -> None:
    gen = HTMLReportGenerator()
    html = gen.generate(single_result)
    assert "MOSSE" in html
    assert "<svg" in html


def test_generate_empty_results() -> None:
    gen = HTMLReportGenerator()
    html = gen.generate([])
    assert "<!DOCTYPE html>" in html
    # Should not crash; leaderboard just shows "no results"
    assert html


def test_generate_no_latency_data() -> None:
    """No latency section when sequences have no mean_latency_ms."""
    results = [_make_result("MOSSE", include_latency=False)]
    gen = HTMLReportGenerator()
    html = gen.generate(results)
    # Latency chart should simply be absent from the output.
    assert "p50" not in html or "Latency Percentiles" not in html


def test_generate_no_center_distances() -> None:
    """Precision curves section omitted when center_distances are absent."""
    results = [_make_result("MOSSE", include_center_distances=False)]
    gen = HTMLReportGenerator()
    html = gen.generate(results)
    # Precision curve heading should not appear.
    assert "Precision Curves" not in html


# ---------------------------------------------------------------------------
# HTMLReportGenerator.save()
# ---------------------------------------------------------------------------

def test_save_creates_html_file(two_results: List[Dict], tmp_path: Path) -> None:
    gen = HTMLReportGenerator()
    out = gen.save(two_results, str(tmp_path / "report"))
    assert out.exists()
    assert out.suffix == ".html"


def test_save_returns_path_object(two_results: List[Dict], tmp_path: Path) -> None:
    gen = HTMLReportGenerator()
    out = gen.save(two_results, str(tmp_path / "report.html"))
    assert isinstance(out, Path)


def test_save_appends_html_extension(two_results: List[Dict], tmp_path: Path) -> None:
    gen = HTMLReportGenerator()
    out = gen.save(two_results, str(tmp_path / "my_report"))
    assert out.name == "my_report.html"


def test_save_creates_parent_dirs(two_results: List[Dict], tmp_path: Path) -> None:
    gen = HTMLReportGenerator()
    nested = tmp_path / "a" / "b" / "c" / "report.html"
    out = gen.save(two_results, str(nested))
    assert out.exists()


def test_save_file_content_is_html(two_results: List[Dict], tmp_path: Path) -> None:
    gen = HTMLReportGenerator()
    out = gen.save(two_results, str(tmp_path / "r.html"))
    content = out.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "MOSSE" in content


def test_save_file_is_utf8(two_results: List[Dict], tmp_path: Path) -> None:
    gen = HTMLReportGenerator(title="Café Résumé")
    out = gen.save(two_results, str(tmp_path / "r.html"))
    content = out.read_bytes()
    content.decode("utf-8")  # must not raise


# ---------------------------------------------------------------------------
# BenchmarkReporter.save_html() integration
# ---------------------------------------------------------------------------

def test_reporter_save_html(two_results: List[Dict], tmp_path: Path) -> None:
    reporter = BenchmarkReporter(output_dir=str(tmp_path))
    out = reporter.save_html(two_results, name="comparison")
    assert out.exists()
    assert out.suffix == ".html"
    content = out.read_text()
    assert "MOSSE" in content
    assert "KCF" in content


def test_reporter_save_html_custom_title(two_results: List[Dict], tmp_path: Path) -> None:
    reporter = BenchmarkReporter(output_dir=str(tmp_path))
    out = reporter.save_html(two_results, title="Edge Tracker Shootout")
    content = out.read_text()
    assert "Edge Tracker Shootout" in content


# ---------------------------------------------------------------------------
# Individual chart builder tests
# ---------------------------------------------------------------------------

def test_success_curves_svg_is_valid(two_results: List[Dict]) -> None:
    svg = _build_success_curves(two_results)
    assert svg.startswith("<svg")
    assert "</svg>" in svg
    assert "MOSSE" in svg
    assert "KCF" in svg
    assert "AUC=" in svg


def test_success_curves_no_data() -> None:
    """Empty ious — should not crash, should still return an SVG string."""
    results = [{"summary": {"tracker": "Empty"}, "sequences": [{"mean_iou": 0.0}]}]
    svg = _build_success_curves(results)
    assert svg.startswith("<svg")


def test_precision_curves_svg(two_results: List[Dict]) -> None:
    svg = _build_precision_curves(two_results)
    assert "<svg" in svg
    assert "MOSSE" in svg
    assert "@20px=" in svg


def test_precision_curves_no_center_distances() -> None:
    results = [_make_result("MOSSE", include_center_distances=False)]
    svg = _build_precision_curves(results)
    # No center_distances → no legend entry → MOSSE not in legend
    assert "<svg" in svg


def test_fps_iou_scatter_svg(two_results: List[Dict]) -> None:
    svg = _build_fps_iou_scatter(two_results)
    assert "<svg" in svg
    assert "<circle" in svg
    assert "MOSSE" in svg
    assert "Pareto" in svg


def test_fps_iou_scatter_single_point(single_result: List[Dict]) -> None:
    svg = _build_fps_iou_scatter(single_result)
    assert "<svg" in svg
    assert "<circle" in svg
    # Pareto front not drawn for single point
    assert "dashed" not in svg or "Pareto" not in svg


def test_fps_iou_scatter_empty() -> None:
    svg = _build_fps_iou_scatter([{"summary": {}, "sequences": []}])
    assert "<svg" in svg
    assert "No data" in svg


def test_latency_bars_svg(two_results: List[Dict]) -> None:
    svg = _build_latency_bars(two_results)
    assert "<svg" in svg
    assert "<rect" in svg
    assert "p50" in svg
    assert "p95" in svg
    assert "p99" in svg


def test_latency_bars_no_latency_data() -> None:
    results = [_make_result("MOSSE", include_latency=False)]
    svg = _build_latency_bars(results)
    # No latency data → empty string returned
    assert svg == ""


def test_leaderboard_ranked_by_auc(two_results: List[Dict]) -> None:
    html_str = _build_leaderboard_html(two_results)
    # KCF has higher AUC (0.53) so it should rank #1
    kcf_pos = html_str.index("KCF")
    mosse_pos = html_str.index("MOSSE")
    assert kcf_pos < mosse_pos


def test_leaderboard_contains_all_columns(two_results: List[Dict]) -> None:
    html_str = _build_leaderboard_html(two_results)
    for col in ["Rank", "Tracker", "Dataset", "Sequences", "mIoU", "Success AUC",
                "Precision AUC", "FPS", "Mem"]:
        assert col in html_str


def test_leaderboard_energy_column_dash_when_absent(two_results: List[Dict]) -> None:
    html_str = _build_leaderboard_html(two_results)
    # Energy field absent in two_results → should display em dash
    assert "—" in html_str


def test_leaderboard_energy_column_value_when_present() -> None:
    results = [_make_result("MOSSE", tdp_watts=15.0)]
    html_str = _build_leaderboard_html(results)
    # Energy should appear as a numeric value, not —
    assert "1.000" in html_str


def test_leaderboard_empty_results() -> None:
    html_str = _build_leaderboard_html([])
    assert "No results" in html_str


def test_leaderboard_xss_escaping() -> None:
    """Tracker names with HTML special chars must be escaped."""
    results = [_make_result("<script>alert(1)</script>")]
    html_str = _build_leaderboard_html(results)
    assert "<script>" not in html_str
    assert "&lt;script&gt;" in html_str

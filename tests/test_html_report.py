"""Tests for eovot.reporting.html_report — HTMLReportGenerator."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import numpy as np
import pytest

from eovot.reporting.html_report import (
    HTMLReportGenerator,
    _efficiency_scatter_svg,
    _render_footer,
    _render_header,
    _render_summary_table,
    _success_curves_svg,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_profiling(tracker_name="T", fps=100.0, mem=50.0):
    from eovot.profiling.profiler import ProfilingResult
    return ProfilingResult(
        tracker_name=tracker_name,
        frame_count=30,
        fps=fps,
        latency_mean_ms=1000.0 / fps,
        latency_std_ms=1.0,
        latency_p95_ms=12.0,
        peak_memory_mb=mem,
    )


def _make_seq_result(name, ious, fps=100.0, mem=50.0, tracker_name="T", gt=None):
    from eovot.benchmark.engine import SequenceResult
    return SequenceResult(
        sequence_name=name,
        ious=np.array(ious, dtype=np.float64),
        profiling=_make_profiling(tracker_name, fps=fps, mem=mem),
        ground_truths=gt,
    )


def _make_result(
    tracker_name="MOSSE",
    dataset_name="Synthetic",
    n_seqs=3,
    iou_val=0.7,
    fps=200.0,
    mem=60.0,
):
    from eovot.benchmark.engine import BenchmarkResult
    br = BenchmarkResult(tracker_name=tracker_name, dataset_name=dataset_name)
    br.sequence_results = [
        _make_seq_result(f"seq_{i}", [iou_val] * 20, fps=fps, mem=mem,
                         tracker_name=tracker_name)
        for i in range(n_seqs)
    ]
    return br


def _two_results():
    return [
        _make_result("MOSSE", fps=300.0, mem=40.0, iou_val=0.5),
        _make_result("KCF",   fps=180.0, mem=80.0, iou_val=0.6),
    ]


# ---------------------------------------------------------------------------
# HTMLReportGenerator — constructor
# ---------------------------------------------------------------------------

def test_generator_creates_output_dir(tmp_path):
    out = tmp_path / "reports" / "sub"
    gen = HTMLReportGenerator(output_dir=str(out))
    assert out.exists()


def test_generator_default_output_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    gen = HTMLReportGenerator()
    assert Path("results/reports").exists()


# ---------------------------------------------------------------------------
# HTMLReportGenerator.generate — basic output
# ---------------------------------------------------------------------------

def test_generate_creates_file(tmp_path):
    gen = HTMLReportGenerator(output_dir=str(tmp_path))
    results = [_make_result()]
    path = gen.generate(results, filename="test.html")
    assert path.exists()
    assert path.suffix == ".html"


def test_generate_returns_correct_path(tmp_path):
    gen = HTMLReportGenerator(output_dir=str(tmp_path))
    results = [_make_result()]
    path = gen.generate(results, filename="out.html")
    assert path.name == "out.html"
    assert path.parent == tmp_path


def test_generate_raises_on_empty_results(tmp_path):
    gen = HTMLReportGenerator(output_dir=str(tmp_path))
    with pytest.raises(ValueError, match="at least one"):
        gen.generate([])


def test_generate_file_is_nonempty(tmp_path):
    gen = HTMLReportGenerator(output_dir=str(tmp_path))
    path = gen.generate([_make_result()])
    assert path.stat().st_size > 100


# ---------------------------------------------------------------------------
# HTML structure validation
# ---------------------------------------------------------------------------

def test_html_has_doctype(tmp_path):
    gen = HTMLReportGenerator(output_dir=str(tmp_path))
    path = gen.generate([_make_result()])
    content = path.read_text(encoding="utf-8")
    assert content.strip().startswith("<!DOCTYPE html>")


def test_html_has_closing_tag(tmp_path):
    gen = HTMLReportGenerator(output_dir=str(tmp_path))
    path = gen.generate([_make_result()])
    content = path.read_text(encoding="utf-8")
    assert "</html>" in content


def test_html_has_title_tag(tmp_path):
    gen = HTMLReportGenerator(output_dir=str(tmp_path))
    path = gen.generate([_make_result()], title="My Report")
    content = path.read_text(encoding="utf-8")
    assert "<title>My Report</title>" in content


def test_html_contains_tracker_name(tmp_path):
    gen = HTMLReportGenerator(output_dir=str(tmp_path))
    path = gen.generate([_make_result("SpecialTracker")])
    content = path.read_text(encoding="utf-8")
    assert "SpecialTracker" in content


def test_html_contains_dataset_name(tmp_path):
    gen = HTMLReportGenerator(output_dir=str(tmp_path))
    path = gen.generate([_make_result(dataset_name="OTB100")])
    content = path.read_text(encoding="utf-8")
    assert "OTB100" in content


def test_html_contains_all_tracker_names(tmp_path):
    gen = HTMLReportGenerator(output_dir=str(tmp_path))
    results = _two_results()
    path = gen.generate(results, title="Comparison")
    content = path.read_text(encoding="utf-8")
    assert "MOSSE" in content
    assert "KCF" in content


def test_html_contains_sequence_names(tmp_path):
    gen = HTMLReportGenerator(output_dir=str(tmp_path))
    path = gen.generate([_make_result(n_seqs=3)])
    content = path.read_text(encoding="utf-8")
    assert "seq_0" in content
    assert "seq_2" in content


def test_html_contains_summary_section(tmp_path):
    gen = HTMLReportGenerator(output_dir=str(tmp_path))
    path = gen.generate([_make_result()])
    content = path.read_text(encoding="utf-8")
    assert "Summary" in content


def test_html_contains_success_curves_section(tmp_path):
    gen = HTMLReportGenerator(output_dir=str(tmp_path))
    path = gen.generate([_make_result()])
    content = path.read_text(encoding="utf-8")
    assert "Success Curves" in content


def test_html_contains_efficiency_section(tmp_path):
    gen = HTMLReportGenerator(output_dir=str(tmp_path))
    path = gen.generate([_make_result()])
    content = path.read_text(encoding="utf-8")
    assert "Efficiency" in content


def test_html_contains_per_sequence_section(tmp_path):
    gen = HTMLReportGenerator(output_dir=str(tmp_path))
    path = gen.generate([_make_result()])
    content = path.read_text(encoding="utf-8")
    assert "Per-Sequence" in content


def test_html_contains_embedded_svg(tmp_path):
    gen = HTMLReportGenerator(output_dir=str(tmp_path))
    path = gen.generate([_make_result()])
    content = path.read_text(encoding="utf-8")
    assert "<svg" in content
    assert "</svg>" in content


def test_html_no_external_links(tmp_path):
    gen = HTMLReportGenerator(output_dir=str(tmp_path))
    path = gen.generate([_make_result()])
    content = path.read_text(encoding="utf-8")
    # Must not reference CDN or external resources
    assert 'src="http' not in content
    assert "cdn.jsdelivr" not in content
    assert "unpkg.com" not in content


def test_html_contains_css_variables(tmp_path):
    gen = HTMLReportGenerator(output_dir=str(tmp_path))
    path = gen.generate([_make_result()])
    content = path.read_text(encoding="utf-8")
    assert "--bg" in content
    assert "--fg" in content


def test_html_has_dark_mode_media_query(tmp_path):
    gen = HTMLReportGenerator(output_dir=str(tmp_path))
    path = gen.generate([_make_result()])
    content = path.read_text(encoding="utf-8")
    assert "prefers-color-scheme: dark" in content


# ---------------------------------------------------------------------------
# Summary table content
# ---------------------------------------------------------------------------

def test_summary_table_contains_iou_value(tmp_path):
    gen = HTMLReportGenerator(output_dir=str(tmp_path))
    path = gen.generate([_make_result(iou_val=0.7)])
    content = path.read_text(encoding="utf-8")
    assert "0.7" in content


def test_summary_table_contains_fps_value(tmp_path):
    gen = HTMLReportGenerator(output_dir=str(tmp_path))
    path = gen.generate([_make_result(fps=123.0)])
    content = path.read_text(encoding="utf-8")
    assert "123" in content


def test_summary_table_includes_energy_when_present(tmp_path):
    from eovot.benchmark.engine import BenchmarkResult, SequenceResult
    from eovot.profiling.energy import EnergyResult

    br = BenchmarkResult(tracker_name="T", dataset_name="D")
    sr = _make_seq_result("s", [0.6] * 10, tracker_name="T")
    sr.energy = EnergyResult(
        tracker_name="T",
        frame_count=10,
        tdp_watts=15.0,
        total_energy_j=0.5,
        mean_power_w=5.0,
        energy_per_frame_mj=50.0,
        peak_cpu_pct=80.0,
        mean_cpu_pct=60.0,
    )
    br.sequence_results = [sr]

    gen = HTMLReportGenerator(output_dir=str(tmp_path))
    path = gen.generate([br])
    content = path.read_text(encoding="utf-8")
    assert "mJ" in content or "50." in content


# ---------------------------------------------------------------------------
# _render_header
# ---------------------------------------------------------------------------

def test_render_header_contains_title():
    h = _render_header("My Title", "2025-01-01", "OTB", 2)
    assert "My Title" in h


def test_render_header_contains_dataset():
    h = _render_header("Title", "2025-01-01", "GOT-10k", 3)
    assert "GOT-10k" in h


def test_render_header_escapes_xss():
    h = _render_header("<script>alert(1)</script>", "2025-01-01", "D", 1)
    assert "<script>" not in h


# ---------------------------------------------------------------------------
# _render_summary_table
# ---------------------------------------------------------------------------

def test_summary_table_has_header_row():
    results = [_make_result("T")]
    table = _render_summary_table(results)
    assert "Mean IoU" in table
    assert "FPS" in table


def test_summary_table_multiple_rows(tmp_path):
    results = _two_results()
    table = _render_summary_table(results)
    assert "MOSSE" in table
    assert "KCF" in table


def test_summary_table_escapes_tracker_name():
    from eovot.benchmark.engine import BenchmarkResult
    br = _make_result("<script>alert()</script>")
    table = _render_summary_table([br])
    assert "<script>" not in table


# ---------------------------------------------------------------------------
# SVG chart generators
# ---------------------------------------------------------------------------

def test_success_curves_svg_is_valid_svg():
    results = _two_results()
    svg = _success_curves_svg(results)
    assert svg.startswith("<svg")
    assert "</svg>" in svg


def test_success_curves_svg_contains_tracker_names():
    results = _two_results()
    svg = _success_curves_svg(results)
    assert "MOSSE" in svg
    assert "KCF" in svg


def test_success_curves_svg_contains_auc():
    results = [_make_result(iou_val=1.0)]
    svg = _success_curves_svg(results)
    assert "AUC" in svg


def test_success_curves_svg_contains_polyline():
    results = [_make_result()]
    svg = _success_curves_svg(results)
    assert "<polyline" in svg


def test_efficiency_scatter_svg_is_valid_svg():
    results = _two_results()
    svg = _efficiency_scatter_svg(results)
    assert svg.startswith("<svg")
    assert "</svg>" in svg


def test_efficiency_scatter_contains_tracker_names():
    results = _two_results()
    svg = _efficiency_scatter_svg(results)
    assert "MOSSE" in svg
    assert "KCF" in svg


def test_efficiency_scatter_contains_circles():
    results = _two_results()
    svg = _efficiency_scatter_svg(results)
    assert "<circle" in svg


def test_success_curves_svg_single_tracker():
    results = [_make_result()]
    svg = _success_curves_svg(results)
    assert "<svg" in svg


def test_efficiency_scatter_single_tracker():
    results = [_make_result(fps=42.0)]
    svg = _efficiency_scatter_svg(results)
    assert "<svg" in svg
    assert "42" in svg or "MOSSE" in svg


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

def test_footer_contains_eovot():
    f = _render_footer()
    assert "EOVOT" in f or "eovot" in f.lower()


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------

def test_full_pipeline_engine_to_html(tmp_path):
    """BenchmarkEngine → save() → load() → HTMLReportGenerator."""
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kcf import KCFTracker
    from eovot.datasets.synthetic import SyntheticDataset

    engine = BenchmarkEngine(verbose=False)
    dataset = SyntheticDataset(num_sequences=3, num_frames=25, seed=7)

    results = [
        engine.run(MOSSETracker(), dataset, dataset_name="Synthetic"),
        engine.run(KCFTracker(),   dataset, dataset_name="Synthetic"),
    ]

    gen = HTMLReportGenerator(output_dir=str(tmp_path))
    path = gen.generate(results, filename="integration.html",
                        title="Integration Test Report")

    assert path.exists()
    content = path.read_text(encoding="utf-8")

    assert "MOSSE" in content
    assert "KCF" in content
    assert "Synthetic" in content
    assert "<svg" in content
    assert "</html>" in content
    assert path.stat().st_size > 2000

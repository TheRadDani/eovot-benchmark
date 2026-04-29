"""Tests for HTMLReporter."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from eovot.reporting.html_reporter import HTMLReporter


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _make_result(
    tracker: str = "MOSSE",
    dataset: str = "OTB100",
    n_seq: int = 3,
    mean_iou: float = 0.50,
    mean_fps: float = 210.0,
) -> dict:
    sequences = [
        {
            "sequence_name": f"seq_{i:02d}",
            "mean_iou": round(mean_iou + i * 0.03, 4),
            "fps": round(mean_fps + i * 5.0, 2),
            "mean_latency_ms": 5.0,
            "peak_memory_mb": 120.0 + i * 2,
        }
        for i in range(n_seq)
    ]
    return {
        "summary": {
            "tracker": tracker,
            "dataset": dataset,
            "num_sequences": n_seq,
            "mean_iou": mean_iou,
            "mean_fps": mean_fps,
            "peak_memory_mb": 125.0,
        },
        "sequences": sequences,
    }


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---------------------------------------------------------------------------
# File creation
# ---------------------------------------------------------------------------

def test_save_creates_html_file(tmp_dir):
    reporter = HTMLReporter(output_dir=tmp_dir)
    path = reporter.save([_make_result()], name="test-report")
    assert path.exists()
    assert path.suffix == ".html"
    assert path.name == "test-report.html"


def test_output_dir_created_automatically():
    with tempfile.TemporaryDirectory() as base:
        new_dir = str(Path(base) / "sub" / "reports")
        reporter = HTMLReporter(output_dir=new_dir)
        path = reporter.save([_make_result()], name="report")
        assert path.exists()


# ---------------------------------------------------------------------------
# HTML structure
# ---------------------------------------------------------------------------

def test_html_doctype_present(tmp_dir):
    reporter = HTMLReporter(output_dir=tmp_dir)
    content = reporter.save([_make_result()], name="r").read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content


def test_html_has_body_and_close(tmp_dir):
    reporter = HTMLReporter(output_dir=tmp_dir)
    content = reporter.save([_make_result()], name="r").read_text(encoding="utf-8")
    assert "<body>" in content
    assert "</html>" in content


def test_html_contains_eovot_footer(tmp_dir):
    reporter = HTMLReporter(output_dir=tmp_dir)
    content = reporter.save([_make_result()], name="r").read_text(encoding="utf-8")
    assert "EOVOT" in content


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def test_summary_table_contains_tracker_name(tmp_dir):
    reporter = HTMLReporter(output_dir=tmp_dir)
    content = reporter.save([_make_result(tracker="MY_TRACKER")], name="r").read_text(encoding="utf-8")
    assert "MY_TRACKER" in content


def test_summary_table_headers_present(tmp_dir):
    reporter = HTMLReporter(output_dir=tmp_dir)
    content = reporter.save([_make_result()], name="r").read_text(encoding="utf-8")
    for header in ["mIoU", "FPS", "Mem (MB)"]:
        assert header in content


def test_multi_tracker_both_names_present(tmp_dir):
    reporter = HTMLReporter(output_dir=tmp_dir)
    results = [_make_result("MOSSE"), _make_result("KCF")]
    content = reporter.save(results, name="comparison").read_text(encoding="utf-8")
    assert "MOSSE" in content
    assert "KCF" in content


def test_energy_column_shows_dash_when_absent(tmp_dir):
    reporter = HTMLReporter(output_dir=tmp_dir)
    content = reporter.save([_make_result()], name="r").read_text(encoding="utf-8")
    assert "—" in content


def test_energy_column_shows_value_when_present(tmp_dir):
    reporter = HTMLReporter(output_dir=tmp_dir)
    result = _make_result()
    result["summary"]["total_energy_j"] = 0.1234
    content = reporter.save([result], name="r").read_text(encoding="utf-8")
    assert "0.1234" in content


# ---------------------------------------------------------------------------
# Per-sequence breakdown
# ---------------------------------------------------------------------------

def test_sequence_details_element_present(tmp_dir):
    reporter = HTMLReporter(output_dir=tmp_dir)
    content = reporter.save([_make_result(n_seq=3)], name="r").read_text(encoding="utf-8")
    assert "<details>" in content


def test_sequence_names_in_output(tmp_dir):
    reporter = HTMLReporter(output_dir=tmp_dir)
    content = reporter.save([_make_result(n_seq=3)], name="r").read_text(encoding="utf-8")
    assert "seq_00" in content
    assert "seq_02" in content


def test_empty_sequences_no_details(tmp_dir):
    reporter = HTMLReporter(output_dir=tmp_dir)
    result = _make_result()
    result["sequences"] = []
    content = reporter.save([result], name="r").read_text(encoding="utf-8")
    assert "<details>" not in content


# ---------------------------------------------------------------------------
# IoU colour badges
# ---------------------------------------------------------------------------

def test_green_badge_for_high_iou(tmp_dir):
    reporter = HTMLReporter(output_dir=tmp_dir)
    content = reporter.save([_make_result(mean_iou=0.7)], name="r").read_text(encoding="utf-8")
    assert 'badge g' in content


def test_red_badge_for_low_iou(tmp_dir):
    reporter = HTMLReporter(output_dir=tmp_dir)
    result = _make_result(mean_iou=0.2)
    # Force per-sequence IoU to be low so the red badge appears in sequence breakdown
    result["sequences"] = [{"sequence_name": "s0", "mean_iou": 0.1, "fps": 10, "mean_latency_ms": 100, "peak_memory_mb": 50}]
    content = reporter.save([result], name="r").read_text(encoding="utf-8")
    assert 'badge r' in content


# ---------------------------------------------------------------------------
# Custom title
# ---------------------------------------------------------------------------

def test_custom_title_in_h1(tmp_dir):
    reporter = HTMLReporter(output_dir=tmp_dir)
    content = reporter.save([_make_result()], name="r", title="My Experiment").read_text(encoding="utf-8")
    assert "My Experiment" in content
    assert "<title>My Experiment</title>" in content


# ---------------------------------------------------------------------------
# Plots toggle
# ---------------------------------------------------------------------------

def test_no_plots_flag_excludes_base64(tmp_dir):
    reporter = HTMLReporter(output_dir=tmp_dir)
    content = reporter.save([_make_result()], name="r", embed_plots=False).read_text(encoding="utf-8")
    assert "data:image/png;base64" not in content


def test_with_plots_embeds_base64_if_matplotlib_available(tmp_dir):
    pytest.importorskip("matplotlib")
    reporter = HTMLReporter(output_dir=tmp_dir)
    content = reporter.save([_make_result()], name="r", embed_plots=True).read_text(encoding="utf-8")
    assert "data:image/png;base64" in content


# ---------------------------------------------------------------------------
# generate_report.py CLI smoke test
# ---------------------------------------------------------------------------

def test_cli_generates_html(tmp_dir):
    import subprocess, sys
    result_data = _make_result()
    json_path = Path(tmp_dir) / "result.json"
    json_path.write_text(json.dumps(result_data), encoding="utf-8")

    out_path = Path(tmp_dir) / "report.html"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/generate_report.py",
            "--results", str(json_path),
            "--name", "report",
            "--output-dir", tmp_dir,
            "--no-plots",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out_path.exists()
    assert "EOVOT" in out_path.read_text(encoding="utf-8")

"""Tests for scripts/evaluate_deployability.py CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Path to the script under test
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_deployability.py"
PYTHON = sys.executable


def _make_result_json(tmp_path: Path, tracker: str, fps: float, iou: float) -> Path:
    """Write a minimal BenchmarkResult JSON file and return its path."""
    ious_per_seq = [round(iou + 0.05 * i * (-1 if i % 2 else 1), 3) for i in range(20)]
    data = {
        "summary": {
            "tracker": tracker,
            "dataset": "Synthetic",
            "mean_iou": iou,
            "mean_fps": fps,
            "peak_memory_mb": 150.0,
            "num_sequences": 2,
            "success_auc": iou * 0.95,
            "precision_auc": iou * 0.90,
        },
        "sequences": [
            {
                "sequence_name": f"seq{i}",
                "mean_iou": iou,
                "fps": fps,
                "mean_latency_ms": 1000.0 / fps,
                "latency_std_ms": 1.0,
                "latency_p95_ms": 1000.0 / fps * 1.2,
                "latency_p99_ms": 1000.0 / fps * 1.5,
                "latency_cv": 0.05,
                "peak_memory_mb": 150.0,
                "ious": ious_per_seq,
                "success_auc": iou * 0.95,
                "precision_auc": iou * 0.90,
                "normalized_precision_auc": iou * 0.85,
            }
            for i in range(2)
        ],
    }
    p = tmp_path / f"{tracker}.json"
    p.write_text(json.dumps(data, indent=2))
    return p


# ---------------------------------------------------------------------------
# Basic invocation tests
# ---------------------------------------------------------------------------

def test_cli_runs_with_single_file(tmp_path):
    """Script should exit 0 with a single input file."""
    f = _make_result_json(tmp_path, "MOSSE", fps=80.0, iou=0.55)
    result = subprocess.run(
        [PYTHON, str(SCRIPT), str(f)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "MOSSE" in result.stdout
    assert "CDS" in result.stdout


def test_cli_runs_with_multiple_files(tmp_path):
    """Script should rank multiple trackers and include all names in output."""
    f1 = _make_result_json(tmp_path, "MOSSE", fps=80.0, iou=0.55)
    f2 = _make_result_json(tmp_path, "KCF", fps=50.0, iou=0.45)
    result = subprocess.run(
        [PYTHON, str(SCRIPT), str(f1), str(f2)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "MOSSE" in result.stdout
    assert "KCF" in result.stdout


def test_cli_device_projection(tmp_path):
    """--device flag should add a device simulation section."""
    f = _make_result_json(tmp_path, "MOSSE", fps=80.0, iou=0.55)
    result = subprocess.run(
        [PYTHON, str(SCRIPT), str(f), "--device", "rpi4"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "rpi4" in result.stdout.lower() or "raspberry" in result.stdout.lower()


def test_cli_writes_output_file(tmp_path):
    """--output flag should write a Markdown file to the given path."""
    f = _make_result_json(tmp_path, "MOSSE", fps=80.0, iou=0.55)
    out = tmp_path / "report.md"
    result = subprocess.run(
        [PYTHON, str(SCRIPT), str(f), "--output", str(out)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
    content = out.read_text()
    assert "EOVOT Deployability Analysis Report" in content
    assert "MOSSE" in content


def test_cli_bad_weights_exit_nonzero(tmp_path):
    """Weights that don't sum to 1.0 should cause a non-zero exit."""
    f = _make_result_json(tmp_path, "MOSSE", fps=80.0, iou=0.55)
    result = subprocess.run(
        [PYTHON, str(SCRIPT), str(f),
         "--w-acc", "0.5", "--w-rob", "0.5", "--w-eff", "0.5", "--w-smooth", "0.1"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_cli_missing_file_exit_nonzero(tmp_path):
    """A non-existent input file should cause a non-zero exit."""
    result = subprocess.run(
        [PYTHON, str(SCRIPT), "/nonexistent/path.json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_cli_robustness_section_present(tmp_path):
    """The output should always include a robustness summary section."""
    f = _make_result_json(tmp_path, "CSRT", fps=30.0, iou=0.65)
    result = subprocess.run(
        [PYTHON, str(SCRIPT), str(f)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Robustness" in result.stdout
    assert "EAO" in result.stdout


def test_cli_custom_weights(tmp_path):
    """Custom weights that sum to 1.0 should be accepted."""
    f = _make_result_json(tmp_path, "MOSSE", fps=80.0, iou=0.55)
    result = subprocess.run(
        [PYTHON, str(SCRIPT), str(f),
         "--w-acc", "0.50",
         "--w-rob", "0.20",
         "--w-eff", "0.20",
         "--w-smooth", "0.10"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "MOSSE" in result.stdout


def test_cli_memory_budget(tmp_path):
    """--memory-budget-mb should be forwarded without error."""
    f = _make_result_json(tmp_path, "MOSSE", fps=80.0, iou=0.55)
    result = subprocess.run(
        [PYTHON, str(SCRIPT), str(f), "--memory-budget-mb", "256"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

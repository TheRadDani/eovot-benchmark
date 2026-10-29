"""Tests for scripts/compare_confidence.py — confidence calibration CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
SCRIPT = SCRIPTS_DIR / "compare_confidence.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT)] + args,
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# Tests — CLI
# ---------------------------------------------------------------------------

class TestCompareConfidenceCLI:

    def test_demo_exits_zero(self):
        result = _run(["--demo"])
        assert result.returncode == 0, result.stderr

    def test_demo_stdout_contains_report_sections(self):
        result = _run(["--demo"])
        out = result.stdout
        assert "# EOVOT Confidence Calibration Report" in out
        assert "## 1. Confidence Calibration" in out
        assert "## 2. Robustness" in out
        assert "## 3. Edge Efficiency" in out

    def test_demo_contains_tracker_names(self):
        result = _run(["--demo"])
        assert "AdaptiveKCF" in result.stdout
        assert "KCF" in result.stdout

    def test_demo_output_to_file(self, tmp_path):
        out_file = tmp_path / "report.md"
        result = _run(["--demo", "--output", str(out_file)])
        assert result.returncode == 0
        assert out_file.exists()
        content = out_file.read_text()
        assert "EOVOT Confidence Calibration Report" in content

    def test_demo_no_stdout_suppresses_report(self, tmp_path):
        out_file = tmp_path / "report.md"
        result = _run(["--demo", "--no-stdout", "--output", str(out_file)])
        assert result.returncode == 0
        assert "# EOVOT" not in result.stdout

    def test_demo_custom_failure_threshold(self):
        result = _run(["--demo", "--failure-threshold", "0.3"])
        assert result.returncode == 0
        assert "IoU < 0.3" in result.stdout

    def test_demo_custom_memory_budget(self):
        result = _run(["--demo", "--memory-budget", "256"])
        assert result.returncode == 0
        assert "256 MB" in result.stdout

    def test_no_args_exits_nonzero(self):
        result = _run([])
        assert result.returncode != 0

    def test_results_file_loads_correctly(self, tmp_path):
        """Save a synthetic result to JSON, then load it with the script."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.kcf import KCFTracker

        dataset = SyntheticDataset(num_sequences=2, num_frames=20, motion="linear")
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(KCFTracker(), dataset, dataset_name="Synthetic")
        saved = result.save(tmp_path / "KCF.json")

        cli_result = _run(["--results", str(saved)])
        assert cli_result.returncode == 0
        assert "KCF" in cli_result.stdout

    def test_results_with_psr_sidecar(self, tmp_path):
        """Supply a PSR sidecar JSON alongside a benchmark result file."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.adaptive_kcf import AdaptiveKCFTracker

        dataset = SyntheticDataset(num_sequences=2, num_frames=30, motion="linear")
        engine = BenchmarkEngine(verbose=False)
        tracker = AdaptiveKCFTracker(min_psr=4.0)
        result = engine.run(tracker, dataset, dataset_name="Synthetic")
        saved = result.save(tmp_path / "AdaptiveKCF.json")

        # Build a per-sequence sidecar from the flat PSR history
        psr_flat = tracker.psr_history
        psr_dir = tmp_path / "psr"
        psr_dir.mkdir()
        per_seq: dict = {}
        offset = 0
        for sr in result.sequence_results:
            n = len(sr.ious)
            per_seq[sr.sequence_name] = psr_flat[offset: offset + n]
            offset += n
        (psr_dir / "AdaptiveKCF.json").write_text(json.dumps(per_seq))

        cli_result = _run([
            "--results", str(saved),
            "--psr-dir", str(psr_dir),
        ])
        assert cli_result.returncode == 0
        assert "AdaptiveKCF" in cli_result.stdout


# ---------------------------------------------------------------------------
# Tests — internal analysis helpers
# ---------------------------------------------------------------------------

class TestAnalyzeTracker:

    def _make_result(self, tracker_name: str = "T", mean_iou: float = 0.7):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.kcf import KCFTracker

        dataset = SyntheticDataset(num_sequences=2, num_frames=20, motion="linear")
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(KCFTracker(), dataset, dataset_name="Synthetic")
        result.tracker_name = tracker_name
        return result

    def test_analyze_tracker_no_psr(self):
        sys.path.insert(0, str(SCRIPTS_DIR.parent))
        from scripts.compare_confidence import _analyze_tracker

        result = self._make_result()
        row = _analyze_tracker(
            result=result,
            psr_by_seq=None,
            failure_threshold=0.1,
            memory_budget_mb=512.0,
        )
        assert row["has_confidence"] is False
        assert not np.isfinite(row["auroc"])
        assert row["ees"] >= 0.0

    def test_analyze_tracker_with_psr_returns_finite_values(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.adaptive_kcf import AdaptiveKCFTracker
        from scripts.compare_confidence import _analyze_tracker

        dataset = SyntheticDataset(num_sequences=2, num_frames=40, motion="linear")
        engine = BenchmarkEngine(verbose=False)
        tracker = AdaptiveKCFTracker(min_psr=4.0)
        result = engine.run(tracker, dataset, dataset_name="Synthetic")

        psr_flat = tracker.psr_history
        psr_by_seq: dict = {}
        offset = 0
        for sr in result.sequence_results:
            n = len(sr.ious)
            psr_by_seq[sr.sequence_name] = psr_flat[offset: offset + n]
            offset += n

        row = _analyze_tracker(
            result=result,
            psr_by_seq=psr_by_seq,
            failure_threshold=0.1,
            memory_budget_mb=512.0,
        )
        assert row["has_confidence"] is True
        assert 0.0 <= row["auroc"] <= 1.0
        assert row["ees"] >= 0.0

    def test_render_report_produces_markdown(self):
        from scripts.compare_confidence import _render_report

        rows = [
            {
                "tracker_name": "T1", "dataset_name": "D",
                "mean_iou": 0.6, "mean_fps": 100.0, "peak_memory_mb": 50.0,
                "ees": 2.5, "mean_eao": 0.55, "mean_failures": 1.2,
                "has_confidence": True,
                "pearson_r": 0.45, "auroc": 0.72, "best_threshold": 6.5,
                "best_f1": 0.68, "n_frames": 90, "n_failures": 8,
            },
            {
                "tracker_name": "T2", "dataset_name": "D",
                "mean_iou": 0.5, "mean_fps": 200.0, "peak_memory_mb": 30.0,
                "ees": 2.0, "mean_eao": 0.48, "mean_failures": float("nan"),
                "has_confidence": False,
                "pearson_r": float("nan"), "auroc": float("nan"),
                "best_threshold": float("nan"), "best_f1": float("nan"),
                "n_frames": 0, "n_failures": 0,
            },
        ]
        report = _render_report(rows, failure_threshold=0.1, memory_budget_mb=512.0)
        assert "T1" in report
        assert "T2" in report
        assert "—" in report  # no-PSR placeholder
        assert "0.720" in report  # AUROC formatted to 3dp

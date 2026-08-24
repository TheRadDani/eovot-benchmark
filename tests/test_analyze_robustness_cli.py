"""Tests for the robustness analysis CLI (``scripts/analyze_robustness.py``).

Exercises both the CLI end-to-end (via subprocess) and its internal
helpers directly, since the helpers are what most of the interesting
logic lives in.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

# Make ``scripts`` importable as a package for direct helper testing.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import analyze_robustness  # type: ignore[import-not-found]
from eovot.metrics.robustness import RobustnessAnalyzer, RobustnessResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_result_json(
    tracker_name: str, sequence_ious: Dict[str, List[float]]
) -> Dict:
    """Build the JSON payload produced by :meth:`BenchmarkResult.to_dict`."""
    seq_entries = []
    for name, ious in sequence_ious.items():
        arr = np.asarray(ious, dtype=np.float64)
        seq_entries.append(
            {
                "sequence_name": name,
                "mean_iou": float(arr.mean()) if len(arr) else 0.0,
                "fps": 100.0,
                "mean_latency_ms": 10.0,
                "latency_std_ms": 1.0,
                "latency_p95_ms": 12.0,
                "latency_p99_ms": 14.0,
                "latency_cv": 0.1,
                "peak_memory_mb": 20.0,
                "ious": list(map(float, arr)),
            }
        )
    return {
        "summary": {
            "tracker": tracker_name,
            "dataset": "synthetic",
            "num_sequences": len(seq_entries),
            "mean_iou": float(np.mean([s["mean_iou"] for s in seq_entries])),
            "mean_fps": 100.0,
            "peak_memory_mb": 20.0,
        },
        "sequences": seq_entries,
    }


@pytest.fixture
def result_dir(tmp_path: Path) -> Path:
    """Create a directory with two tracker result files: 'strong' and 'weak'.

    - ``strong`` has all IoUs above the default 0.1 failure threshold.
    - ``weak``   drops to near-zero mid-sequence for one sequence.
    """
    strong_seqs = {
        "seq_a": [0.9] * 30,
        "seq_b": [0.8] * 40,
    }
    weak_seqs = {
        "seq_a": [0.9] * 10 + [0.05] * 20,   # fails at frame 10, never recovers
        "seq_b": [0.7] * 40,
    }

    (tmp_path / "strong-synthetic.json").write_text(
        json.dumps(_make_result_json("strong", strong_seqs)), encoding="utf-8"
    )
    (tmp_path / "weak-synthetic.json").write_text(
        json.dumps(_make_result_json("weak", weak_seqs)), encoding="utf-8"
    )
    return tmp_path


# ---------------------------------------------------------------------------
# Internal helper tests
# ---------------------------------------------------------------------------


class TestLoadResultFile:
    def test_loads_valid_result(self, result_dir: Path):
        parsed = analyze_robustness._load_result_file(
            result_dir / "strong-synthetic.json"
        )
        assert parsed is not None
        name, seq_ious = parsed
        assert name == "strong"
        assert set(seq_ious) == {"seq_a", "seq_b"}
        assert seq_ious["seq_a"].shape == (30,)
        assert np.all(seq_ious["seq_a"] > 0.5)

    def test_falls_back_to_mean_iou_for_legacy_file(self, tmp_path: Path):
        legacy = {
            "summary": {"tracker": "legacy", "dataset": "otb"},
            "sequences": [
                {"sequence_name": "seq_x", "mean_iou": 0.42},
                {"sequence_name": "seq_y", "mean_iou": 0.55},
            ],
        }
        p = tmp_path / "legacy.json"
        p.write_text(json.dumps(legacy))
        parsed = analyze_robustness._load_result_file(p)
        assert parsed is not None
        name, seq_ious = parsed
        assert name == "legacy"
        # No 'ious' key → degenerate one-frame arrays.
        assert seq_ious["seq_x"].shape == (1,)
        assert seq_ious["seq_x"][0] == pytest.approx(0.42)

    def test_malformed_json_returns_none(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text("{not json")
        assert analyze_robustness._load_result_file(p) is None

    def test_missing_sequences_returns_none(self, tmp_path: Path):
        p = tmp_path / "empty.json"
        p.write_text(json.dumps({"summary": {"tracker": "x"}}))
        assert analyze_robustness._load_result_file(p) is None

    def test_tracker_name_falls_back_to_stem(self, tmp_path: Path):
        """When summary omits 'tracker' name, use file stem's first segment."""
        payload = {
            "summary": {},
            "sequences": [
                {"sequence_name": "s", "mean_iou": 0.5, "ious": [0.5]}
            ],
        }
        p = tmp_path / "MyTracker-dataset.json"
        p.write_text(json.dumps(payload))
        parsed = analyze_robustness._load_result_file(p)
        assert parsed is not None
        assert parsed[0] == "MyTracker"


class TestCollectInputs:
    def test_directory_expands_to_json_files(self, result_dir: Path):
        files = analyze_robustness._collect_inputs([str(result_dir)])
        assert len(files) == 2
        assert all(f.suffix == ".json" for f in files)

    def test_explicit_files_returned_as_is(self, result_dir: Path):
        target = result_dir / "strong-synthetic.json"
        files = analyze_robustness._collect_inputs([str(target)])
        assert files == [target]

    def test_missing_path_is_skipped(self, tmp_path: Path):
        files = analyze_robustness._collect_inputs([str(tmp_path / "nope.json")])
        assert files == []


class TestFormatTopFailures:
    def _make_result(self, name: str, failures: int, eao: float) -> RobustnessResult:
        return RobustnessResult(
            tracker_name="t",
            sequence_name=name,
            num_failures=failures,
            failure_frames=[0] * failures,
            recovery_lags=[10] * failures,
            mean_recovery_lag=10.0 if failures else 0.0,
            eao=eao,
            survival_rate=0.5,
        )

    def test_returns_empty_when_top_n_zero(self):
        per_seq = {"a": self._make_result("a", 3, 0.2)}
        assert analyze_robustness._format_top_failures("t", per_seq, 0) == []

    def test_returns_empty_when_no_failures(self):
        per_seq = {"a": self._make_result("a", 0, 0.9)}
        assert analyze_robustness._format_top_failures("t", per_seq, 5) == []

    def test_ranks_by_failures_descending(self):
        per_seq = {
            "a": self._make_result("a", 1, 0.5),
            "b": self._make_result("b", 3, 0.5),
            "c": self._make_result("c", 2, 0.5),
        }
        lines = analyze_robustness._format_top_failures("t", per_seq, 5)
        # Row order in the table (after the two header lines):
        seq_rows = [l for l in lines if l.startswith("|") and "---" not in l][1:]
        assert seq_rows[0].split("|")[1].strip() == "b"
        assert seq_rows[1].split("|")[1].strip() == "c"
        assert seq_rows[2].split("|")[1].strip() == "a"

    def test_ties_broken_by_lowest_eao(self):
        per_seq = {
            "hi_eao": self._make_result("hi_eao", 2, 0.6),
            "lo_eao": self._make_result("lo_eao", 2, 0.3),
        }
        lines = analyze_robustness._format_top_failures("t", per_seq, 5)
        seq_rows = [l for l in lines if l.startswith("|") and "---" not in l][1:]
        assert seq_rows[0].split("|")[1].strip() == "lo_eao"

    def test_top_n_caps_output(self):
        per_seq = {
            f"seq_{i}": self._make_result(f"seq_{i}", 1, 0.5) for i in range(10)
        }
        lines = analyze_robustness._format_top_failures("t", per_seq, 3)
        seq_rows = [l for l in lines if l.startswith("|") and "---" not in l][1:]
        assert len(seq_rows) == 3


class TestRobustnessAnalyzerMarkdown:
    def test_empty_input_returns_placeholder(self):
        assert "No tracker aggregates" in RobustnessAnalyzer.to_markdown_table([])

    def test_rows_sorted_by_eao_descending(self):
        aggs = [
            {
                "tracker_name": "slow",
                "num_sequences": 5,
                "total_failures": 3,
                "mean_failures_per_sequence": 0.6,
                "mean_eao": 0.3,
                "mean_survival_rate": 0.5,
                "mean_recovery_lag_frames": 10.0,
            },
            {
                "tracker_name": "fast",
                "num_sequences": 5,
                "total_failures": 0,
                "mean_failures_per_sequence": 0.0,
                "mean_eao": 0.8,
                "mean_survival_rate": 1.0,
                "mean_recovery_lag_frames": 0.0,
            },
        ]
        table = RobustnessAnalyzer.to_markdown_table(aggs)
        # 'fast' has higher EAO — it must appear before 'slow' in the table.
        assert table.index("fast") < table.index("slow")


# ---------------------------------------------------------------------------
# CLI end-to-end tests
# ---------------------------------------------------------------------------


class TestCLIEndToEnd:
    def test_reports_expected_trackers(self, result_dir: Path, capsys):
        exit_code = analyze_robustness.main([str(result_dir)])
        out = capsys.readouterr().out
        assert exit_code == 0
        assert "EOVOT Robustness Analysis Report" in out
        assert "strong" in out
        assert "weak" in out
        # 'strong' should rank above 'weak' — no failures vs. one failure.
        assert out.index("strong") < out.index("weak")

    def test_weak_tracker_flagged_with_failure(self, result_dir: Path, capsys):
        analyze_robustness.main(
            [str(result_dir), "--top-failures", "5"]
        )
        out = capsys.readouterr().out
        assert "Worst Failure Sequences" in out
        assert "seq_a" in out         # the sequence 'weak' fails on

    def test_top_failures_zero_omits_section(self, result_dir: Path, capsys):
        analyze_robustness.main([str(result_dir), "--top-failures", "0"])
        out = capsys.readouterr().out
        assert "Worst Failure Sequences" not in out

    def test_out_file_written(self, result_dir: Path, tmp_path: Path):
        out_path = tmp_path / "reports" / "robust.md"
        exit_code = analyze_robustness.main(
            [str(result_dir), "--out", str(out_path)]
        )
        assert exit_code == 0
        assert out_path.exists()
        content = out_path.read_text()
        assert "EOVOT Robustness Analysis Report" in content

    def test_no_inputs_errors(self, tmp_path: Path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        exit_code = analyze_robustness.main([str(empty_dir)])
        assert exit_code == 1

    def test_failure_threshold_flag_affects_failure_count(
        self, tmp_path: Path, capsys
    ):
        """A stricter threshold turns a mid-IoU sequence into a failure."""
        seqs = {"seq_x": [0.5] * 30}   # all frames at 0.5
        payload = _make_result_json("borderline", seqs)
        (tmp_path / "borderline-synth.json").write_text(json.dumps(payload))

        # Default (0.1): no failure.
        analyze_robustness.main([str(tmp_path)])
        default_out = capsys.readouterr().out
        # Strict (0.6): every frame is now below threshold → 1 failure.
        analyze_robustness.main(
            [str(tmp_path), "--failure-threshold", "0.6", "--top-failures", "5"]
        )
        strict_out = capsys.readouterr().out
        assert "Worst Failure Sequences" not in default_out
        assert "Worst Failure Sequences" in strict_out
        assert "seq_x" in strict_out

    def test_subprocess_invocation(self, result_dir: Path):
        """Sanity check: script runs when invoked as ``python scripts/...``."""
        script = REPO_ROOT / "scripts" / "analyze_robustness.py"
        cp = subprocess.run(
            [sys.executable, str(script), str(result_dir)],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=60,
        )
        assert cp.returncode == 0, cp.stderr
        assert "EOVOT Robustness Analysis Report" in cp.stdout

"""Tests for the unified EOVOT CLI (eovot.cli).

Covers all four subcommands — run, compare, simulate, list — using synthetic
data so no external dataset download is required.  Uses monkeypatch and
capsys to capture output without spawning subprocesses.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eovot.cli import main, _build_parser


# ---------------------------------------------------------------------------
# Parser smoke tests
# ---------------------------------------------------------------------------

class TestParser:

    def test_help_does_not_crash(self, capsys):
        parser = _build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--help"])
        assert exc.value.code == 0

    def test_no_subcommand_exits_zero(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 0

    def test_run_requires_tracker(self):
        with pytest.raises(SystemExit):
            main(["run"])

    def test_compare_requires_trackers(self):
        with pytest.raises(SystemExit):
            main(["compare"])

    def test_simulate_requires_input(self):
        with pytest.raises(SystemExit):
            main(["simulate"])


# ---------------------------------------------------------------------------
# eovot list
# ---------------------------------------------------------------------------

class TestCmdList:

    def test_list_default_shows_both(self, capsys):
        main(["list"])
        out = capsys.readouterr().out
        assert "trackers" in out.lower() or "MOSSE" in out
        assert "rpi4" in out or "device" in out.lower()

    def test_list_trackers_only(self, capsys):
        main(["list", "--trackers"])
        out = capsys.readouterr().out
        assert "MOSSE" in out
        assert "KCF" in out
        assert "CSRT" in out

    def test_list_devices_only(self, capsys):
        main(["list", "--devices"])
        out = capsys.readouterr().out
        assert "rpi4" in out
        assert "jetson_nano" in out

    def test_list_devices_verbose(self, capsys):
        main(["list", "--devices", "--verbose"])
        out = capsys.readouterr().out
        assert "speed_factor" in out
        assert "TDP" in out

    def test_list_trackers_contains_camshift(self, capsys):
        main(["list", "--trackers"])
        out = capsys.readouterr().out
        assert "CamShift" in out


# ---------------------------------------------------------------------------
# eovot run
# ---------------------------------------------------------------------------

class TestCmdRun:

    def test_run_mosse_synthetic(self, capsys):
        main([
            "run", "--tracker", "MOSSE",
            "--dataset", "synthetic",
            "--sequences", "2",
            "--frames", "30",
            "--motion", "circular",
        ])
        out = capsys.readouterr().out
        assert "mean_iou" in out
        assert "mean_fps" in out

    def test_run_kcf_synthetic(self, capsys):
        main([
            "run", "--tracker", "KCF",
            "--dataset", "synthetic",
            "--sequences", "2",
            "--frames", "30",
        ])
        out = capsys.readouterr().out
        assert "BENCHMARK SUMMARY" in out

    def test_run_saves_json(self, tmp_path, capsys):
        output_file = str(tmp_path / "result.json")
        main([
            "run", "--tracker", "MOSSE",
            "--dataset", "synthetic",
            "--sequences", "2",
            "--frames", "20",
            "--output", output_file,
        ])
        assert Path(output_file).is_file()
        with open(output_file) as f:
            data = json.load(f)
        assert "summary" in data
        assert data["summary"]["tracker"] == "MOSSE"

    def test_run_unknown_tracker_exits_nonzero(self):
        with pytest.raises(SystemExit) as exc:
            main([
                "run", "--tracker", "NONEXISTENT_TRACKER",
                "--dataset", "synthetic",
                "--sequences", "1",
            ])
        assert exc.value.code != 0

    def test_run_with_max_sequences(self, capsys):
        main([
            "run", "--tracker", "MOSSE",
            "--dataset", "synthetic",
            "--sequences", "10",
            "--frames", "20",
            "--max-sequences", "2",
        ])
        out = capsys.readouterr().out
        # Only 2 sequences should be evaluated
        assert "BENCHMARK SUMMARY" in out


# ---------------------------------------------------------------------------
# eovot compare
# ---------------------------------------------------------------------------

class TestCmdCompare:

    def test_compare_two_trackers(self, capsys):
        main([
            "compare",
            "--trackers", "MOSSE", "KCF",
            "--dataset", "synthetic",
            "--sequences", "2",
            "--frames", "25",
            "--motion", "circular",
            "--quiet",
        ])
        out = capsys.readouterr().out
        assert "LEADERBOARD" in out
        assert "MOSSE" in out
        assert "KCF" in out

    def test_compare_saves_json_files(self, tmp_path, capsys):
        main([
            "compare",
            "--trackers", "MOSSE", "KCF",
            "--dataset", "synthetic",
            "--sequences", "2",
            "--frames", "20",
            "--motion", "circular",
            "--quiet",
            "--output-dir", str(tmp_path),
        ])
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 2

    def test_compare_unknown_tracker_exits_nonzero(self):
        with pytest.raises(SystemExit) as exc:
            main([
                "compare",
                "--trackers", "MOSSE", "FAKE_TRACKER",
                "--dataset", "synthetic",
                "--sequences", "1",
            ])
        assert exc.value.code != 0

    def test_compare_efficiency_table_in_output(self, capsys):
        main([
            "compare",
            "--trackers", "MOSSE", "KCF",
            "--dataset", "synthetic",
            "--sequences", "2",
            "--frames", "20",
            "--motion", "circular",
            "--quiet",
        ])
        out = capsys.readouterr().out
        # Markdown table should contain | separators and EES column
        assert "|" in out
        assert "EES" in out


# ---------------------------------------------------------------------------
# eovot simulate
# ---------------------------------------------------------------------------

class TestCmdSimulate:

    @pytest.fixture
    def saved_result(self, tmp_path):
        """Run MOSSE on synthetic data and save the result JSON."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.trackers.mosse import MOSSETracker
        from eovot.datasets.synthetic import SyntheticDataset

        dataset = SyntheticDataset(num_sequences=2, num_frames=20, seed=0, motion="circular")
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), dataset, dataset_name="Synthetic")
        out_path = tmp_path / "mosse_result.json"
        result.save(out_path)
        return str(out_path)

    def test_simulate_all_devices(self, saved_result, capsys):
        main(["simulate", "--input", saved_result])
        out = capsys.readouterr().out
        assert "rpi4" in out or "Raspberry" in out
        assert "FPS" in out

    def test_simulate_specific_devices(self, saved_result, capsys):
        main([
            "simulate", "--input", saved_result,
            "--devices", "rpi4", "jetson_nano",
        ])
        out = capsys.readouterr().out
        assert "rpi4" in out or "Raspberry" in out
        assert "jetson" in out.lower()

    def test_simulate_with_sustained_seconds(self, saved_result, capsys):
        main([
            "simulate", "--input", saved_result,
            "--sustained", "120",
            "--devices", "rpi4",
        ])
        out = capsys.readouterr().out
        assert "120" in out or "throttl" in out.lower() or "Raspberry" in out

    def test_simulate_saves_json(self, saved_result, tmp_path, capsys):
        output_path = str(tmp_path / "sim_out.json")
        main([
            "simulate", "--input", saved_result,
            "--devices", "rpi4",
            "--output", output_path,
        ])
        assert Path(output_path).is_file()
        with open(output_path) as f:
            data = json.load(f)
        assert "devices" in data
        assert len(data["devices"]) == 1
        assert data["devices"][0]["device"] == "rpi4"

    def test_simulate_missing_input_exits_nonzero(self):
        with pytest.raises(SystemExit) as exc:
            main(["simulate", "--input", "/nonexistent/path/result.json"])
        assert exc.value.code != 0

"""Tests for EdgeDeploymentReport and the simulate_edge CLI."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from eovot.reporting.edge_report import EdgeDeploymentReport, DeploymentCell


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_result_dict(
    tracker: str = "MOSSE",
    mean_iou: float = 0.65,
    mean_fps: float = 500.0,
    peak_memory_mb: float = 50.0,
) -> dict:
    return {
        "summary": {
            "tracker_name": tracker,
            "tracker": tracker,
            "dataset": "Synthetic",
            "num_sequences": 5,
            "mean_iou": mean_iou,
            "mean_fps": mean_fps,
            "peak_memory_mb": peak_memory_mb,
        },
        "sequences": [],
    }


TWO_TRACKERS = [
    _make_result_dict("MOSSE", mean_iou=0.65, mean_fps=500.0, peak_memory_mb=50.0),
    _make_result_dict("KCF",   mean_iou=0.72, mean_fps=150.0, peak_memory_mb=80.0),
]


# ---------------------------------------------------------------------------
# DeploymentCell
# ---------------------------------------------------------------------------

class TestDeploymentCell:
    def test_to_dict_keys(self):
        cell = DeploymentCell(
            tracker_name="MOSSE",
            device_name="rpi4",
            device_display="Raspberry Pi 4B (4 GB)",
            host_fps=500.0,
            estimated_fps=60.0,
            estimated_latency_ms=16.7,
            estimated_memory_mb=50.0,
            memory_limit_mb=3800.0,
            fits_in_memory=True,
            estimated_energy_mj=0.5,
            thermal_state="nominal",
            deployable=True,
            mean_iou=0.65,
        )
        d = cell.to_dict()
        for key in ["tracker", "device", "estimated_fps", "deployable", "mean_iou"]:
            assert key in d, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# EdgeDeploymentReport construction
# ---------------------------------------------------------------------------

class TestEdgeDeploymentReport:
    def test_from_result_dicts_two_trackers(self):
        report = EdgeDeploymentReport.from_result_dicts(
            TWO_TRACKERS, min_fps=10.0, device_names=["rpi4", "jetson_nano"]
        )
        assert len(report.tracker_names) == 2
        assert "MOSSE" in report.tracker_names
        assert "KCF" in report.tracker_names
        assert len(report.device_names) == 2
        # 2 trackers × 2 devices = 4 cells
        assert len(report.cells) == 4

    def test_cells_contain_valid_fps(self):
        report = EdgeDeploymentReport.from_result_dicts(
            TWO_TRACKERS, device_names=["rpi4"]
        )
        for cell in report.cells:
            assert cell.estimated_fps > 0.0
            assert cell.estimated_latency_ms > 0.0

    def test_memory_fits_for_small_tracker(self):
        # MOSSE uses only 50 MB — should fit on all devices
        report = EdgeDeploymentReport.from_result_dicts(
            [_make_result_dict("MOSSE", peak_memory_mb=50.0)],
            device_names=["rpi4", "jetson_nano"],
        )
        for cell in report.cells:
            assert cell.fits_in_memory, f"Expected MOSSE to fit on {cell.device_name}"

    def test_oom_for_large_tracker(self):
        # 2 GB tracker should not fit on Coral (1 GB RAM)
        report = EdgeDeploymentReport.from_result_dicts(
            [_make_result_dict("HeavyDL", peak_memory_mb=2000.0)],
            device_names=["coral_board"],
        )
        cell = report.get_cell("HeavyDL", "coral_board")
        assert cell is not None
        assert not cell.fits_in_memory
        assert not cell.deployable

    def test_min_fps_threshold(self):
        # Fast tracker on fast device — check deployability changes with min_fps
        report_low = EdgeDeploymentReport.from_result_dicts(
            TWO_TRACKERS, min_fps=1.0, device_names=["jetson_xnx"]
        )
        report_high = EdgeDeploymentReport.from_result_dicts(
            TWO_TRACKERS, min_fps=999.0, device_names=["jetson_xnx"]
        )
        # With min_fps=1, more should be deployable than with min_fps=999
        assert len(report_low.deployable_combinations()) >= len(report_high.deployable_combinations())

    def test_get_cell(self):
        report = EdgeDeploymentReport.from_result_dicts(
            TWO_TRACKERS, device_names=["rpi4", "rpi5"]
        )
        cell = report.get_cell("MOSSE", "rpi4")
        assert cell is not None
        assert cell.tracker_name == "MOSSE"
        assert cell.device_name == "rpi4"

    def test_get_cell_missing_returns_none(self):
        report = EdgeDeploymentReport.from_result_dicts(
            TWO_TRACKERS, device_names=["rpi4"]
        )
        assert report.get_cell("MOSSE", "nonexistent_device") is None

    def test_deployable_combinations_list(self):
        report = EdgeDeploymentReport.from_result_dicts(
            TWO_TRACKERS, min_fps=0.1, device_names=["rpi4"]
        )
        combos = report.deployable_combinations()
        # All should be deployable at 0.1 FPS (assuming memory fits)
        for tracker, device in combos:
            assert tracker in report.tracker_names
            assert device in report.device_names

    def test_all_built_in_devices_simulated(self):
        report = EdgeDeploymentReport.from_result_dicts(TWO_TRACKERS, min_fps=1.0)
        from eovot.profiling.device_sim import KNOWN_DEVICES
        assert len(report.device_names) == len(KNOWN_DEVICES)


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

class TestMarkdownOutput:
    def test_markdown_contains_tracker_names(self):
        report = EdgeDeploymentReport.from_result_dicts(
            TWO_TRACKERS, device_names=["rpi4"]
        )
        md = report.to_markdown()
        assert "MOSSE" in md
        assert "KCF" in md

    def test_markdown_contains_device_names(self):
        report = EdgeDeploymentReport.from_result_dicts(
            TWO_TRACKERS, device_names=["rpi4", "rpi5"]
        )
        md = report.to_markdown()
        assert "rpi4" in md.lower() or "raspberry pi 4" in md.lower()

    def test_markdown_has_summary_section(self):
        report = EdgeDeploymentReport.from_result_dicts(
            TWO_TRACKERS, device_names=["rpi4"]
        )
        md = report.to_markdown()
        assert "Summary" in md

    def test_to_dict_serializable(self):
        report = EdgeDeploymentReport.from_result_dicts(
            TWO_TRACKERS, device_names=["rpi4"]
        )
        d = report.to_dict()
        # Must be JSON-serializable
        serialized = json.dumps(d)
        restored = json.loads(serialized)
        assert "trackers" in restored
        assert "devices" in restored
        assert "cells" in restored


# ---------------------------------------------------------------------------
# Save / IO
# ---------------------------------------------------------------------------

class TestSave:
    def test_save_creates_files(self):
        report = EdgeDeploymentReport.from_result_dicts(
            TWO_TRACKERS, device_names=["rpi4"]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = report.save(output_dir=tmpdir, prefix="test_report")
            assert saved["markdown"].exists()
            assert saved["json"].exists()
            assert saved["markdown"].stat().st_size > 0
            assert saved["json"].stat().st_size > 0

    def test_saved_json_is_valid(self):
        report = EdgeDeploymentReport.from_result_dicts(
            TWO_TRACKERS, device_names=["rpi4"]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = report.save(output_dir=tmpdir)
            content = saved["json"].read_text()
            data = json.loads(content)
            assert "cells" in data
            assert len(data["cells"]) == 2  # 2 trackers × 1 device


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestSimulateEdgeCLI:
    def _write_result(self, tmpdir: Path, tracker: str, fps: float) -> Path:
        path = tmpdir / f"{tracker}-Synthetic.json"
        path.write_text(
            json.dumps(_make_result_dict(tracker, mean_fps=fps)),
            encoding="utf-8",
        )
        return path

    def test_cli_runs_on_directory(self, tmp_path):
        self._write_result(tmp_path, "MOSSE", 500.0)
        self._write_result(tmp_path, "KCF", 150.0)

        from scripts.simulate_edge import main

        out_dir = str(tmp_path / "edge_out")
        exit_code = main([
            str(tmp_path),
            "--devices", "rpi4",
            "--min-fps", "5",
            "--output-dir", out_dir,
        ])
        assert exit_code == 0
        assert (Path(out_dir) / "edge_report.md").exists()
        assert (Path(out_dir) / "edge_report.json").exists()

    def test_cli_list_devices(self, capsys):
        from scripts.simulate_edge import main
        exit_code = main(["--list-devices"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "rpi4" in captured.out
        assert "jetson_nano" in captured.out

    def test_cli_no_inputs_returns_1(self):
        from scripts.simulate_edge import main
        exit_code = main([])
        assert exit_code == 1

    def test_cli_invalid_device_returns_1(self, tmp_path):
        self._write_result(tmp_path, "MOSSE", 500.0)
        from scripts.simulate_edge import main
        exit_code = main([str(tmp_path), "--devices", "nonexistent_device_xyz"])
        assert exit_code == 1

    def test_cli_sustained_seconds(self, tmp_path):
        self._write_result(tmp_path, "MOSSE", 500.0)
        from scripts.simulate_edge import main
        out_dir = str(tmp_path / "edge_out")
        exit_code = main([
            str(tmp_path),
            "--devices", "rpi4",
            "--sustained-seconds", "120",
            "--output-dir", out_dir,
        ])
        assert exit_code == 0

    def test_cli_explicit_json_files(self, tmp_path):
        p1 = self._write_result(tmp_path, "MOSSE", 500.0)
        p2 = self._write_result(tmp_path, "KCF", 150.0)
        from scripts.simulate_edge import main
        out_dir = str(tmp_path / "edge_out")
        exit_code = main([
            str(p1), str(p2),
            "--devices", "rpi4", "rpi5",
            "--output-dir", out_dir,
        ])
        assert exit_code == 0

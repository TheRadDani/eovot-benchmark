"""Tests for device fleet heatmap visualisation and device_analysis CLI."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from eovot.profiling.device_sim import KNOWN_DEVICES, DeviceSimulator
from eovot.profiling.profiler import ProfilingResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_profiling(
    tracker_name: str = "MOSSE",
    fps: float = 200.0,
    memory_mb: float = 50.0,
) -> ProfilingResult:
    latency = 1_000.0 / fps
    return ProfilingResult(
        tracker_name=tracker_name,
        frame_count=100,
        fps=fps,
        latency_mean_ms=latency,
        latency_std_ms=0.3,
        latency_p95_ms=latency * 1.3,
        peak_memory_mb=memory_mb,
    )


def _build_sim_matrix(
    tracker_names=("MOSSE", "KCF"),
    device_names=("rpi4", "jetson_nano"),
) -> tuple:
    """Return (sim_matrix, device_names) for use in tests."""
    sim = DeviceSimulator()
    matrix: Dict[str, Dict[str, object]] = {}
    fps_map = {"MOSSE": 250.0, "KCF": 150.0, "CSRT": 50.0}
    for name in tracker_names:
        pr = _make_profiling(name, fps=fps_map.get(name, 100.0))
        results = sim.simulate_all(pr, device_names=list(device_names))
        matrix[name] = {r.device_name: r for r in results}
    return matrix, list(device_names)


# ---------------------------------------------------------------------------
# plot_device_fleet_heatmap
# ---------------------------------------------------------------------------

class TestPlotDeviceFleetHeatmap:
    """Tests for eovot.visualization.plots.plot_device_fleet_heatmap."""

    def _import(self):
        pytest.importorskip("matplotlib")
        from eovot.visualization.plots import plot_device_fleet_heatmap
        return plot_device_fleet_heatmap

    def test_invalid_metric_raises(self):
        fn = self._import()
        matrix, devices = _build_sim_matrix()
        with pytest.raises(ValueError, match="metric"):
            fn(matrix, devices, metric="invalid_metric")

    def test_fps_metric_saves_file(self, tmp_path):
        fn = self._import()
        matrix, devices = _build_sim_matrix()
        out = str(tmp_path / "heatmap_fps.png")
        fn(matrix, devices, metric="fps", output_path=out)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    def test_latency_metric_saves_file(self, tmp_path):
        fn = self._import()
        matrix, devices = _build_sim_matrix()
        out = str(tmp_path / "heatmap_latency.png")
        fn(matrix, devices, metric="latency_ms", output_path=out)
        assert Path(out).exists()

    def test_energy_metric_saves_file(self, tmp_path):
        fn = self._import()
        matrix, devices = _build_sim_matrix()
        out = str(tmp_path / "heatmap_energy.png")
        fn(matrix, devices, metric="energy_mj", output_path=out)
        assert Path(out).exists()

    def test_viability_metric_saves_file(self, tmp_path):
        fn = self._import()
        matrix, devices = _build_sim_matrix()
        out = str(tmp_path / "heatmap_viability.png")
        fn(matrix, devices, metric="viability", output_path=out)
        assert Path(out).exists()

    def test_all_builtin_devices(self, tmp_path):
        fn = self._import()
        sim = DeviceSimulator()
        all_devices = sim.list_devices()
        pr = _make_profiling("MOSSE")
        results = sim.simulate_all(pr)
        matrix = {"MOSSE": {r.device_name: r for r in results}}
        out = str(tmp_path / "heatmap_all.png")
        fn(matrix, all_devices, output_path=out)
        assert Path(out).exists()

    def test_oom_tracker_included_gracefully(self, tmp_path):
        fn = self._import()
        # Big memory tracker should still render without error
        pr_big = _make_profiling("BigModel", memory_mb=5000.0)
        sim = DeviceSimulator()
        results = sim.simulate_all(pr_big, device_names=["coral_board", "rpi4"])
        matrix = {"BigModel": {r.device_name: r for r in results}}
        out = str(tmp_path / "heatmap_oom.png")
        fn(matrix, ["coral_board", "rpi4"], output_path=out)
        assert Path(out).exists()

    def test_single_tracker_single_device(self, tmp_path):
        fn = self._import()
        pr = _make_profiling("MOSSE")
        sim = DeviceSimulator()
        r = sim.simulate(pr, "rpi4")
        matrix = {"MOSSE": {"rpi4": r}}
        out = str(tmp_path / "heatmap_single.png")
        fn(matrix, ["rpi4"], output_path=out)
        assert Path(out).exists()


# ---------------------------------------------------------------------------
# device_analysis.py CLI
# ---------------------------------------------------------------------------

def _write_result_json(path: Path, tracker: str, fps: float, mem: float) -> None:
    data = {
        "summary": {
            "tracker_name": tracker,
            "mean_fps": fps,
            "peak_memory_mb": mem,
            "num_sequences": 5,
        },
        "sequences": [
            {"sequence_name": f"seq{i}", "mean_iou": 0.5, "fps": fps}
            for i in range(5)
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


class TestDeviceAnalysisCLI:
    """Integration tests for scripts/device_analysis.py."""

    def _run(self, argv):
        scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
        sys.path.insert(0, str(scripts_dir.parent))
        from scripts.device_analysis import main
        return main(argv)

    def test_single_file_exits_zero(self, tmp_path):
        f = tmp_path / "MOSSE-test.json"
        _write_result_json(f, "MOSSE", fps=200.0, mem=50.0)
        rc = self._run([str(f), "--devices", "rpi4"])
        assert rc == 0

    def test_multiple_files_exits_zero(self, tmp_path):
        for name, fps in [("MOSSE", 200.0), ("KCF", 150.0)]:
            _write_result_json(tmp_path / f"{name}.json", name, fps, 60.0)
        rc = self._run(
            [str(tmp_path / "MOSSE.json"), str(tmp_path / "KCF.json"),
             "--devices", "rpi4", "jetson_nano"]
        )
        assert rc == 0

    def test_all_metrics_exit_zero(self, tmp_path):
        f = tmp_path / "MOSSE.json"
        _write_result_json(f, "MOSSE", fps=200.0, mem=50.0)
        for metric in ("fps", "latency_ms", "energy_mj", "viability"):
            rc = self._run([str(f), "--devices", "rpi4", "--metric", metric])
            assert rc == 0, f"metric={metric} exited with {rc}"

    def test_output_dir_creates_markdown(self, tmp_path):
        f = tmp_path / "MOSSE.json"
        _write_result_json(f, "MOSSE", fps=200.0, mem=50.0)
        out_dir = tmp_path / "analysis"
        rc = self._run([str(f), "--devices", "rpi4", "--output-dir", str(out_dir)])
        assert rc == 0
        assert (out_dir / "device_fleet_analysis.md").exists()

    def test_unknown_device_exits_nonzero(self, tmp_path):
        f = tmp_path / "MOSSE.json"
        _write_result_json(f, "MOSSE", fps=200.0, mem=50.0)
        rc = self._run([str(f), "--devices", "nonexistent_device_xyz"])
        assert rc != 0

    def test_no_inputs_exits_nonzero(self, tmp_path):
        rc = self._run([str(tmp_path / "does_not_exist.json")])
        assert rc != 0

    def test_missing_fps_field_warns_and_skips(self, tmp_path):
        f = tmp_path / "broken.json"
        f.write_text(json.dumps({"summary": {"tracker_name": "X"}}), encoding="utf-8")
        # Second valid file ensures at least one result loads
        f2 = tmp_path / "MOSSE.json"
        _write_result_json(f2, "MOSSE", fps=200.0, mem=50.0)
        rc = self._run([str(f), str(f2), "--devices", "rpi4"])
        assert rc == 0

    def test_sustained_load_flag(self, tmp_path):
        f = tmp_path / "MOSSE.json"
        _write_result_json(f, "MOSSE", fps=200.0, mem=50.0)
        rc = self._run([str(f), "--devices", "rpi4", "--sustained", "120.0"])
        assert rc == 0

    def test_directory_input(self, tmp_path):
        for name, fps in [("MOSSE", 200.0), ("KCF", 150.0)]:
            _write_result_json(tmp_path / f"{name}.json", name, fps, 60.0)
        rc = self._run([str(tmp_path), "--devices", "rpi4"])
        assert rc == 0

    def test_host_calibration_flag(self, tmp_path):
        f = tmp_path / "MOSSE.json"
        _write_result_json(f, "MOSSE", fps=200.0, mem=50.0)
        rc = self._run([str(f), "--devices", "rpi4", "--host-calibration", "1.5"])
        assert rc == 0

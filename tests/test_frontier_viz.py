"""Tests for eovot.visualization.frontier — efficiency frontier plots."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import List

import pytest

# ---------------------------------------------------------------------------
# Lightweight stubs — no dependency on matplotlib at import time
# ---------------------------------------------------------------------------


@dataclass
class _EfficiencyEntry:
    tracker_name: str
    dataset_name: str
    mean_iou: float
    fps: float
    peak_memory_mb: float
    ees: float
    on_pareto_front: bool = False


@dataclass
class _DeviceSimResult:
    device_name: str
    display_name: str
    tracker_name: str
    estimated_fps: float
    estimated_latency_ms: float
    estimated_energy_mj_per_frame: float
    fits_in_memory: bool
    host_fps: float = 0.0
    host_latency_ms: float = 0.0
    host_memory_mb: float = 100.0
    estimated_memory_mb: float = 100.0
    memory_limit_mb: float = 4000.0
    thermal_state: str = "nominal"
    effective_speed_factor: float = 0.12
    notes: str = ""


@dataclass
class _SkipRateResult:
    skip_rate: int
    mean_iou: float
    success_auc: float
    failure_rate: float
    effective_fps_multiplier: float


@dataclass
class _FrameSkipAnalysis:
    tracker_name: str
    dataset_name: str
    baseline_fps: float
    interpolation: str
    skip_results: List[_SkipRateResult] = field(default_factory=list)

    @property
    def optimal_skip_rate(self) -> int:
        if not self.skip_results:
            return 1
        baseline = self.skip_results[0].mean_iou
        best = 1
        for r in self.skip_results:
            if baseline > 0 and (baseline - r.mean_iou) / baseline <= 0.10:
                best = r.skip_rate
        return best


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

pytest.importorskip("matplotlib", reason="matplotlib required for frontier viz")


@pytest.fixture()
def entries():
    return [
        _EfficiencyEntry("MOSSE",  "OTB", 0.42, 320.0, 60.0,  1.80, on_pareto_front=True),
        _EfficiencyEntry("KCF",    "OTB", 0.55, 120.0, 80.0,  2.10, on_pareto_front=True),
        _EfficiencyEntry("CSRT",   "OTB", 0.72,  25.0, 95.0,  1.50, on_pareto_front=False),
        _EfficiencyEntry("MIL",    "OTB", 0.48,  35.0, 70.0,  0.90, on_pareto_front=False),
    ]


@pytest.fixture()
def sim_by_tracker():
    devices = ["Raspberry Pi 4B", "Jetson Nano", "Jetson Xavier NX"]
    data = {}
    for tracker, fps_vals in [
        ("MOSSE", [38.0, 32.0, 115.0]),
        ("KCF",   [14.0, 12.0, 43.0]),
    ]:
        data[tracker] = [
            _DeviceSimResult(
                device_name=f"dev{j}",
                display_name=devices[j],
                tracker_name=tracker,
                estimated_fps=fps_vals[j],
                estimated_latency_ms=1000.0 / fps_vals[j],
                estimated_energy_mj_per_frame=0.5,
                fits_in_memory=True,
            )
            for j in range(len(fps_vals))
        ]
    return data


@pytest.fixture()
def skip_analyses():
    def _make(name, iou_vals):
        return _FrameSkipAnalysis(
            tracker_name=name,
            dataset_name="Synthetic",
            baseline_fps=200.0,
            interpolation="hold_last",
            skip_results=[
                _SkipRateResult(k, v, v * 0.9, 1.0 - v, float(k))
                for k, v in zip([1, 2, 4, 8], iou_vals)
            ],
        )
    return [
        _make("MOSSE", [0.72, 0.68, 0.58, 0.40]),
        _make("KCF",   [0.65, 0.62, 0.50, 0.30]),
    ]


# ---------------------------------------------------------------------------
# plot_efficiency_frontier
# ---------------------------------------------------------------------------


class TestPlotEfficiencyFrontier:
    def test_saves_png_file(self, entries):
        from eovot.visualization.frontier import plot_efficiency_frontier
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "frontier.png")
            plot_efficiency_frontier(entries, output_path=path)
            assert os.path.isfile(path)
            assert os.path.getsize(path) > 1000

    def test_empty_entries_no_error(self):
        from eovot.visualization.frontier import plot_efficiency_frontier
        # Should return silently without raising
        plot_efficiency_frontier([])

    def test_all_pareto_no_error(self, entries):
        from eovot.visualization.frontier import plot_efficiency_frontier
        for e in entries:
            e.on_pareto_front = True
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "frontier.png")
            plot_efficiency_frontier(entries, output_path=path, annotate=False)
            assert os.path.isfile(path)

    def test_no_pareto_no_error(self, entries):
        from eovot.visualization.frontier import plot_efficiency_frontier
        for e in entries:
            e.on_pareto_front = False
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "frontier.png")
            plot_efficiency_frontier(entries, output_path=path)
            assert os.path.isfile(path)

    def test_custom_title_accepted(self, entries):
        from eovot.visualization.frontier import plot_efficiency_frontier
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "frontier.png")
            plot_efficiency_frontier(entries, output_path=path, title="Custom Title")
            assert os.path.isfile(path)

    def test_single_tracker(self):
        from eovot.visualization.frontier import plot_efficiency_frontier
        single = [_EfficiencyEntry("MOSSE", "OTB", 0.5, 100.0, 50.0, 1.0, True)]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "frontier.png")
            plot_efficiency_frontier(single, output_path=path)
            assert os.path.isfile(path)


# ---------------------------------------------------------------------------
# plot_device_projection
# ---------------------------------------------------------------------------


class TestPlotDeviceProjection:
    def test_saves_png_file(self, sim_by_tracker):
        from eovot.visualization.frontier import plot_device_projection
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "devices.png")
            plot_device_projection(sim_by_tracker, output_path=path)
            assert os.path.isfile(path)
            assert os.path.getsize(path) > 1000

    def test_empty_dict_no_error(self):
        from eovot.visualization.frontier import plot_device_projection
        plot_device_projection({})

    def test_latency_metric(self, sim_by_tracker):
        from eovot.visualization.frontier import plot_device_projection
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "latency.png")
            plot_device_projection(
                sim_by_tracker,
                metric="estimated_latency_ms",
                output_path=path,
            )
            assert os.path.isfile(path)

    def test_oom_cell_marked(self, sim_by_tracker):
        from eovot.visualization.frontier import plot_device_projection
        # Mark one device as OOM
        sim_by_tracker["MOSSE"][0].fits_in_memory = False
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "oom.png")
            plot_device_projection(sim_by_tracker, mark_oom=True, output_path=path)
            assert os.path.isfile(path)

    def test_custom_title(self, sim_by_tracker):
        from eovot.visualization.frontier import plot_device_projection
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "devices.png")
            plot_device_projection(
                sim_by_tracker, title="My Title", output_path=path
            )
            assert os.path.isfile(path)

    def test_single_tracker_single_device(self):
        from eovot.visualization.frontier import plot_device_projection
        data = {
            "MOSSE": [_DeviceSimResult(
                device_name="rpi4", display_name="Pi 4B", tracker_name="MOSSE",
                estimated_fps=38.0, estimated_latency_ms=26.0,
                estimated_energy_mj_per_frame=0.3, fits_in_memory=True,
            )]
        }
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "single.png")
            plot_device_projection(data, output_path=path)
            assert os.path.isfile(path)


# ---------------------------------------------------------------------------
# plot_frame_skip_degradation
# ---------------------------------------------------------------------------


class TestPlotFrameSkipDegradation:
    def test_saves_png_file(self, skip_analyses):
        from eovot.visualization.frontier import plot_frame_skip_degradation
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "frame_skip.png")
            plot_frame_skip_degradation(skip_analyses, output_path=path)
            assert os.path.isfile(path)
            assert os.path.getsize(path) > 1000

    def test_empty_analyses_no_error(self):
        from eovot.visualization.frontier import plot_frame_skip_degradation
        plot_frame_skip_degradation([])

    def test_success_auc_metric(self, skip_analyses):
        from eovot.visualization.frontier import plot_frame_skip_degradation
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "auc.png")
            plot_frame_skip_degradation(
                skip_analyses, metric="success_auc", output_path=path
            )
            assert os.path.isfile(path)

    def test_show_optimal_false(self, skip_analyses):
        from eovot.visualization.frontier import plot_frame_skip_degradation
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "noopt.png")
            plot_frame_skip_degradation(
                skip_analyses, show_optimal=False, output_path=path
            )
            assert os.path.isfile(path)

    def test_single_analysis(self, skip_analyses):
        from eovot.visualization.frontier import plot_frame_skip_degradation
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "single.png")
            plot_frame_skip_degradation([skip_analyses[0]], output_path=path)
            assert os.path.isfile(path)

    def test_failure_rate_metric(self, skip_analyses):
        from eovot.visualization.frontier import plot_frame_skip_degradation
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "fail.png")
            plot_frame_skip_degradation(
                skip_analyses, metric="failure_rate", output_path=path
            )
            assert os.path.isfile(path)

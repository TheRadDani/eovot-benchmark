"""Tests for latency tail-risk metrics (p99, CV) and FPS-IoU scatter/latency plots."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from eovot.profiling.profiler import Profiler, ProfilingResult

_matplotlib_available = True
try:
    import matplotlib  # noqa: F401
except ImportError:
    _matplotlib_available = False

skip_no_matplotlib = pytest.mark.skipif(
    not _matplotlib_available, reason="matplotlib not installed"
)


# ---------------------------------------------------------------------------
# Profiler tail-metric tests
# ---------------------------------------------------------------------------

class TestLatencyTailMetrics:
    """Verify that p99 and CV are correctly computed by Profiler.summary()."""

    def _run_profiler(self, latencies_ms):
        profiler = Profiler()
        for lat in latencies_ms:
            profiler.start_frame()
            time.sleep(lat / 1_000.0)
            profiler.end_frame()
        return profiler.summary("tracker")

    def test_p99_field_present(self):
        result = self._run_profiler([1.0] * 5)
        assert hasattr(result, "latency_p99_ms")

    def test_cv_field_present(self):
        result = self._run_profiler([1.0] * 5)
        assert hasattr(result, "latency_cv")

    def test_p99_ge_p95(self):
        result = self._run_profiler([1.0] * 20)
        assert result.latency_p99_ms >= result.latency_p95_ms - 1e-9

    def test_p99_ge_mean(self):
        result = self._run_profiler([1.0] * 10)
        assert result.latency_p99_ms >= result.latency_mean_ms - 1e-9

    def test_cv_nonnegative(self):
        result = self._run_profiler([5.0] * 15)
        assert result.latency_cv >= 0.0

    def test_cv_positive_for_variable_latency(self):
        latencies = [1.0, 10.0] * 15
        result = self._run_profiler(latencies)
        assert result.latency_cv > 0.0

    def test_repr_includes_p99(self):
        result = self._run_profiler([2.0] * 5)
        assert "p99=" in str(result)

    def test_repr_includes_cv(self):
        result = self._run_profiler([2.0] * 5)
        assert "CV=" in str(result)

    def test_default_p99_cv_zero_on_manual_construction(self):
        """ProfilingResult constructed without p99/CV should default to 0.0."""
        result = ProfilingResult(
            tracker_name="manual",
            frame_count=10,
            fps=100.0,
            latency_mean_ms=10.0,
            latency_std_ms=1.0,
            latency_p95_ms=11.5,
            peak_memory_mb=50.0,
        )
        assert result.latency_p99_ms == 0.0
        assert result.latency_cv == 0.0


# ---------------------------------------------------------------------------
# Scatter-plot and latency-percentile plot smoke tests
# ---------------------------------------------------------------------------

def _make_result(tracker, fps, iou, lat_ms=10.0):
    return {
        "summary": {
            "tracker": tracker,
            "mean_fps": fps,
            "mean_iou": iou,
            "peak_memory_mb": 50.0,
        },
        "sequences": [
            {"sequence_name": "seq0", "mean_latency_ms": lat_ms, "fps": fps},
            {"sequence_name": "seq1", "mean_latency_ms": lat_ms * 1.2, "fps": fps * 0.9},
        ],
    }


@skip_no_matplotlib
def test_fps_iou_scatter_creates_file(tmp_path):
    from eovot.visualization.plots import plot_fps_iou_scatter
    results = [
        _make_result("MOSSE", fps=200.0, iou=0.62),
        _make_result("KCF", fps=80.0, iou=0.70),
        _make_result("CSRT", fps=25.0, iou=0.80),
    ]
    out = str(tmp_path / "scatter.png")
    plot_fps_iou_scatter(results, output_path=out)
    assert os.path.isfile(out)


@skip_no_matplotlib
def test_fps_iou_scatter_empty_list(tmp_path):
    from eovot.visualization.plots import plot_fps_iou_scatter
    out = str(tmp_path / "empty_scatter.png")
    plot_fps_iou_scatter([], output_path=out)
    assert os.path.isfile(out)


@skip_no_matplotlib
def test_fps_iou_scatter_no_pareto(tmp_path):
    from eovot.visualization.plots import plot_fps_iou_scatter
    results = [_make_result("MOSSE", 200.0, 0.60)]
    out = str(tmp_path / "no_pareto.png")
    plot_fps_iou_scatter(results, output_path=out, pareto_front=False)
    assert os.path.isfile(out)


@skip_no_matplotlib
def test_latency_percentile_plot_creates_file(tmp_path):
    from eovot.visualization.plots import plot_latency_percentiles
    results = [
        _make_result("MOSSE", 200.0, 0.62, lat_ms=5.0),
        _make_result("KCF", 80.0, 0.70, lat_ms=12.0),
    ]
    out = str(tmp_path / "percentiles.png")
    plot_latency_percentiles(results, output_path=out)
    assert os.path.isfile(out)


@skip_no_matplotlib
def test_latency_percentile_plot_empty_sequences(tmp_path):
    from eovot.visualization.plots import plot_latency_percentiles
    results = [{"summary": {"tracker": "X"}, "sequences": []}]
    out = str(tmp_path / "empty_lat.png")
    # Should not raise; no output file is expected when there are no sequences.
    plot_latency_percentiles(results, output_path=out)

"""Unit tests for RadarChartBuilder and RadarEntry.

Tests cover:
- RadarEntry construction and raw_values()
- RadarEntry.from_benchmark_result() convenience constructor
- RadarChartBuilder normalisation correctness
- Best tracker on each axis scores 1.0
- Normalised values in [0, 1]
- Speed axis is log-normalised (not linear)
- to_dict() structure and key completeness
- to_markdown() produces valid Markdown table
- Empty entries raises ValueError
- Single-entry chart (degenerate case)
- plot() with output_path (saves file)
"""

from __future__ import annotations

import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from eovot.visualization.radar import RadarChartBuilder, RadarEntry, _AXES, _N_AXES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _three_entries() -> list:
    return [
        RadarEntry("MOSSE",   mean_iou=0.42, fps=312.0, survival_rate=0.78,
                   smoothness_score=0.81, ees=0.21),
        RadarEntry("KCF",     mean_iou=0.51, fps=180.0, survival_rate=0.83,
                   smoothness_score=0.74, ees=0.28),
        RadarEntry("CSRT",    mean_iou=0.68, fps=42.0,  survival_rate=0.91,
                   smoothness_score=0.69, ees=0.19),
    ]


# ---------------------------------------------------------------------------
# RadarEntry
# ---------------------------------------------------------------------------

class TestRadarEntry:
    def test_raw_values_length(self):
        e = RadarEntry("T", mean_iou=0.5, fps=100.0, survival_rate=0.8,
                       smoothness_score=0.9, ees=0.3)
        assert len(e.raw_values()) == _N_AXES

    def test_raw_values_order(self):
        e = RadarEntry("T", mean_iou=0.5, fps=100.0, survival_rate=0.8,
                       smoothness_score=0.9, ees=0.3)
        vals = e.raw_values()
        assert vals[0] == pytest.approx(0.5)
        assert vals[1] == pytest.approx(100.0)
        assert vals[2] == pytest.approx(0.8)
        assert vals[3] == pytest.approx(0.9)
        assert vals[4] == pytest.approx(0.3)

    def test_from_benchmark_result_uses_name_and_iou(self):
        class _FakeBR:
            tracker_name = "FAKE"
            mean_iou = 0.55
            mean_fps = 200.0

        entry = RadarEntry.from_benchmark_result(_FakeBR())
        assert entry.tracker_name == "FAKE"
        assert entry.mean_iou == pytest.approx(0.55)
        assert entry.fps == pytest.approx(200.0)

    def test_from_benchmark_result_proxies_survival_to_iou(self):
        """When survival_rate not provided, it defaults to mean_iou proxy."""
        class _FakeBR:
            tracker_name = "X"
            mean_iou = 0.6
            mean_fps = 50.0

        entry = RadarEntry.from_benchmark_result(_FakeBR())
        assert entry.survival_rate == pytest.approx(0.6)
        assert entry.ees == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# RadarChartBuilder — construction
# ---------------------------------------------------------------------------

class TestRadarChartBuilderConstruction:
    def test_empty_entries_raises(self):
        with pytest.raises(ValueError):
            RadarChartBuilder([])

    def test_single_entry_allowed(self):
        entry = RadarEntry("T", mean_iou=0.5, fps=100.0, survival_rate=0.8,
                           smoothness_score=0.9, ees=0.3)
        builder = RadarChartBuilder([entry])
        normed = builder.normalise()
        assert normed.shape == (1, _N_AXES)
        # Single tracker: all scores normalise to 1.0
        assert np.allclose(normed, 1.0)

    def test_stores_entries(self):
        entries = _three_entries()
        builder = RadarChartBuilder(entries)
        assert builder.entries is entries


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class TestNormalisation:
    def test_output_shape(self):
        builder = RadarChartBuilder(_three_entries())
        normed = builder.normalise()
        assert normed.shape == (3, _N_AXES)

    def test_values_in_unit_interval(self):
        builder = RadarChartBuilder(_three_entries())
        normed = builder.normalise()
        assert np.all(normed >= 0.0)
        assert np.all(normed <= 1.0 + 1e-9)

    def test_best_tracker_scores_1_on_each_axis(self):
        """The tracker with the highest raw value on each axis must score 1.0."""
        entries = _three_entries()
        builder = RadarChartBuilder(entries)
        normed = builder.normalise()
        for col in range(_N_AXES):
            assert normed[:, col].max() == pytest.approx(1.0, abs=1e-6), \
                f"Axis {_AXES[col]} max is {normed[:, col].max()}, expected 1.0"

    def test_speed_is_log_normalised(self):
        """Log-normalising speed means doubling FPS does not double score."""
        entries = [
            RadarEntry("slow", mean_iou=0.5, fps=10.0,  survival_rate=0.8,
                       smoothness_score=0.9, ees=0.3),
            RadarEntry("fast", mean_iou=0.5, fps=100.0, survival_rate=0.8,
                       smoothness_score=0.9, ees=0.3),
        ]
        builder = RadarChartBuilder(entries)
        normed = builder.normalise()
        # 100 FPS = 10× linear, but log1p(100)/log1p(10) ≠ 10
        slow_speed = normed[0, 1]
        fast_speed = normed[1, 1]
        assert fast_speed == pytest.approx(1.0, abs=1e-6)
        expected_slow = np.log1p(10.0) / np.log1p(100.0)
        assert slow_speed == pytest.approx(expected_slow, abs=1e-6)

    def test_all_zero_axis_does_not_crash(self):
        """An axis where every tracker has 0 should not produce NaN."""
        entries = [
            RadarEntry("A", mean_iou=0.0, fps=0.0, survival_rate=0.0,
                       smoothness_score=0.0, ees=0.0),
            RadarEntry("B", mean_iou=0.0, fps=0.0, survival_rate=0.0,
                       smoothness_score=0.0, ees=0.0),
        ]
        builder = RadarChartBuilder(entries)
        normed = builder.normalise()
        assert not np.any(np.isnan(normed))
        assert not np.any(np.isinf(normed))

    def test_normalise_idempotent(self):
        """Calling normalise() twice returns identical results."""
        builder = RadarChartBuilder(_three_entries())
        n1 = builder.normalise().copy()
        n2 = builder.normalise().copy()
        assert np.allclose(n1, n2)


# ---------------------------------------------------------------------------
# to_dict()
# ---------------------------------------------------------------------------

class TestToDict:
    def test_required_keys(self):
        builder = RadarChartBuilder(_three_entries())
        d = builder.to_dict()
        assert "axes" in d
        assert "trackers" in d
        assert "raw" in d

    def test_axes_are_correct(self):
        builder = RadarChartBuilder(_three_entries())
        assert builder.to_dict()["axes"] == list(_AXES)

    def test_tracker_count(self):
        entries = _three_entries()
        builder = RadarChartBuilder(entries)
        d = builder.to_dict()
        assert len(d["trackers"]) == len(entries)
        assert len(d["raw"]) == len(entries)

    def test_tracker_names_preserved(self):
        entries = _three_entries()
        builder = RadarChartBuilder(entries)
        names = [t["name"] for t in builder.to_dict()["trackers"]]
        assert names == [e.tracker_name for e in entries]

    def test_tracker_entries_have_all_axes(self):
        builder = RadarChartBuilder(_three_entries())
        for tracker_row in builder.to_dict()["trackers"]:
            for axis in _AXES:
                assert axis in tracker_row, f"Missing axis '{axis}' in {tracker_row}"

    def test_normed_values_in_unit_range(self):
        builder = RadarChartBuilder(_three_entries())
        for tracker_row in builder.to_dict()["trackers"]:
            for axis in _AXES:
                v = tracker_row[axis]
                assert 0.0 <= v <= 1.0 + 1e-6, \
                    f"{tracker_row['name']} axis {axis} = {v} out of [0,1]"


# ---------------------------------------------------------------------------
# to_markdown()
# ---------------------------------------------------------------------------

class TestToMarkdown:
    def test_returns_string(self):
        builder = RadarChartBuilder(_three_entries())
        md = builder.to_markdown()
        assert isinstance(md, str)

    def test_contains_tracker_names(self):
        entries = _three_entries()
        builder = RadarChartBuilder(entries)
        md = builder.to_markdown()
        for entry in entries:
            assert entry.tracker_name in md

    def test_contains_axis_names(self):
        builder = RadarChartBuilder(_three_entries())
        md = builder.to_markdown()
        for axis in _AXES:
            assert axis in md

    def test_has_header_separator_and_data_rows(self):
        entries = _three_entries()
        builder = RadarChartBuilder(entries)
        lines = [l for l in builder.to_markdown().splitlines() if l.strip()]
        # Header + separator + N data rows
        assert len(lines) == 2 + len(entries)

    def test_pipe_delimited(self):
        builder = RadarChartBuilder(_three_entries())
        for line in builder.to_markdown().splitlines():
            if line.strip():
                assert line.startswith("|") and line.endswith("|")


# ---------------------------------------------------------------------------
# plot() — file output (no display needed)
# ---------------------------------------------------------------------------

class TestPlot:
    def test_saves_png_file(self):
        pytest.importorskip("matplotlib")
        entries = _three_entries()
        builder = RadarChartBuilder(entries)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            builder.plot(output_path=path, title="Test Radar")
            assert os.path.getsize(path) > 0
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_no_matplotlib_import_error(self):
        """Importing the module itself must not require matplotlib."""
        import importlib
        import eovot.visualization.radar as module
        # If matplotlib is absent, the ImportError is deferred to plot(),
        # not raised at module import time.
        assert hasattr(module, "RadarChartBuilder")

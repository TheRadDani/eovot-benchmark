"""Unit tests for the trajectory visualisation module."""

from __future__ import annotations

import os

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_result_dict(n_seq: int = 6, rng_seed: int = 42) -> dict:
    rng = np.random.default_rng(rng_seed)
    sequences = []
    for i in range(n_seq):
        sequences.append(
            {
                "sequence_name": f"seq_{i:02d}",
                "mean_iou": float(rng.uniform(0.3, 0.9)),
                "fps": float(rng.uniform(50.0, 400.0)),
                "peak_memory_mb": float(rng.uniform(100.0, 500.0)),
                "mean_latency_ms": float(rng.uniform(2.0, 20.0)),
            }
        )
    return {
        "summary": {"tracker": "TestTracker"},
        "sequences": sequences,
    }


def _synthetic_ious(n: int = 60, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.4, 0.9, n)
    # Inject a failure region (frames 20–29).
    base[20:30] = rng.uniform(0.0, 0.08, 10)
    return base


# ---------------------------------------------------------------------------
# plot_iou_timeline
# ---------------------------------------------------------------------------

class TestPlotIouTimeline:
    def test_saves_image_to_disk(self, tmp_path):
        pytest.importorskip("matplotlib")
        from eovot.visualization.trajectory import plot_iou_timeline

        ious = _synthetic_ious()
        output = str(tmp_path / "timeline.png")
        plot_iou_timeline(
            ious,
            sequence_name="Basketball",
            tracker_name="MOSSE",
            output_path=output,
        )
        assert os.path.exists(output)
        assert os.path.getsize(output) > 0

    def test_all_zero_ious(self, tmp_path):
        pytest.importorskip("matplotlib")
        from eovot.visualization.trajectory import plot_iou_timeline

        ious = np.zeros(30)
        output = str(tmp_path / "zero.png")
        plot_iou_timeline(ious, output_path=output)
        assert os.path.exists(output)

    def test_all_high_ious(self, tmp_path):
        pytest.importorskip("matplotlib")
        from eovot.visualization.trajectory import plot_iou_timeline

        ious = np.ones(50) * 0.95
        output = str(tmp_path / "high.png")
        plot_iou_timeline(ious, output_path=output)
        assert os.path.exists(output)

    def test_single_frame(self, tmp_path):
        pytest.importorskip("matplotlib")
        from eovot.visualization.trajectory import plot_iou_timeline

        ious = np.array([0.7])
        output = str(tmp_path / "single.png")
        plot_iou_timeline(ious, output_path=output)
        assert os.path.exists(output)

    def test_custom_failure_threshold(self, tmp_path):
        pytest.importorskip("matplotlib")
        from eovot.visualization.trajectory import plot_iou_timeline

        ious = _synthetic_ious()
        output = str(tmp_path / "custom_threshold.png")
        plot_iou_timeline(ious, failure_threshold=0.3, output_path=output)
        assert os.path.exists(output)

    def test_custom_title(self, tmp_path):
        pytest.importorskip("matplotlib")
        from eovot.visualization.trajectory import plot_iou_timeline

        ious = _synthetic_ious(30)
        output = str(tmp_path / "titled.png")
        plot_iou_timeline(ious, title="My Custom Title", output_path=output)
        assert os.path.exists(output)

    def test_failure_region_at_end(self, tmp_path):
        """Failure shading extending to the last frame should not crash."""
        pytest.importorskip("matplotlib")
        from eovot.visualization.trajectory import plot_iou_timeline

        ious = np.concatenate([np.ones(40) * 0.8, np.zeros(20)])
        output = str(tmp_path / "tail_failure.png")
        plot_iou_timeline(ious, output_path=output)
        assert os.path.exists(output)


# ---------------------------------------------------------------------------
# plot_sequence_heatmap
# ---------------------------------------------------------------------------

class TestPlotSequenceHeatmap:
    def test_saves_image_to_disk(self, tmp_path):
        pytest.importorskip("matplotlib")
        from eovot.visualization.trajectory import plot_sequence_heatmap

        result = _make_result_dict(10)
        output = str(tmp_path / "heatmap.png")
        plot_sequence_heatmap(result, output_path=output)
        assert os.path.exists(output)
        assert os.path.getsize(output) > 0

    def test_empty_sequences_prints_warning(self, capsys):
        pytest.importorskip("matplotlib")
        from eovot.visualization.trajectory import plot_sequence_heatmap

        result = {"summary": {}, "sequences": []}
        plot_sequence_heatmap(result)
        out = capsys.readouterr().out
        assert "No sequence data" in out

    def test_max_sequences_limits_rows(self, tmp_path):
        pytest.importorskip("matplotlib")
        from eovot.visualization.trajectory import plot_sequence_heatmap

        result = _make_result_dict(30)
        output = str(tmp_path / "limited.png")
        plot_sequence_heatmap(result, max_sequences=5, output_path=output)
        assert os.path.exists(output)

    def test_custom_metrics(self, tmp_path):
        pytest.importorskip("matplotlib")
        from eovot.visualization.trajectory import plot_sequence_heatmap

        result = _make_result_dict(8)
        output = str(tmp_path / "custom_metrics.png")
        plot_sequence_heatmap(
            result,
            metrics=["mean_iou", "mean_latency_ms"],
            output_path=output,
        )
        assert os.path.exists(output)

    def test_custom_title(self, tmp_path):
        pytest.importorskip("matplotlib")
        from eovot.visualization.trajectory import plot_sequence_heatmap

        result = _make_result_dict(4)
        output = str(tmp_path / "titled.png")
        plot_sequence_heatmap(result, title="Custom Heatmap Title", output_path=output)
        assert os.path.exists(output)

    def test_single_sequence(self, tmp_path):
        pytest.importorskip("matplotlib")
        from eovot.visualization.trajectory import plot_sequence_heatmap

        result = _make_result_dict(1)
        output = str(tmp_path / "single_seq.png")
        plot_sequence_heatmap(result, output_path=output)
        assert os.path.exists(output)


# ---------------------------------------------------------------------------
# plot_multi_tracker_iou_timeline
# ---------------------------------------------------------------------------

class TestPlotMultiTrackerIouTimeline:
    def test_saves_image_to_disk(self, tmp_path):
        pytest.importorskip("matplotlib")
        from eovot.visualization.trajectory import plot_multi_tracker_iou_timeline

        tracker_ious = {
            "MOSSE": _synthetic_ious(80, seed=0),
            "KCF": _synthetic_ious(80, seed=1),
            "CSRT": _synthetic_ious(80, seed=2),
        }
        output = str(tmp_path / "compare.png")
        plot_multi_tracker_iou_timeline(
            tracker_ious,
            sequence_name="CarScale",
            output_path=output,
        )
        assert os.path.exists(output)
        assert os.path.getsize(output) > 0

    def test_empty_dict_prints_message(self, capsys):
        pytest.importorskip("matplotlib")
        from eovot.visualization.trajectory import plot_multi_tracker_iou_timeline

        plot_multi_tracker_iou_timeline({})
        out = capsys.readouterr().out
        assert "empty" in out

    def test_single_tracker(self, tmp_path):
        pytest.importorskip("matplotlib")
        from eovot.visualization.trajectory import plot_multi_tracker_iou_timeline

        output = str(tmp_path / "single.png")
        plot_multi_tracker_iou_timeline(
            {"MOSSE": _synthetic_ious(50)},
            output_path=output,
        )
        assert os.path.exists(output)

    def test_different_length_arrays(self, tmp_path):
        """Trackers with different numbers of frames should plot without error."""
        pytest.importorskip("matplotlib")
        from eovot.visualization.trajectory import plot_multi_tracker_iou_timeline

        output = str(tmp_path / "diff_len.png")
        plot_multi_tracker_iou_timeline(
            {
                "Short": _synthetic_ious(30),
                "Long": _synthetic_ious(100),
            },
            output_path=output,
        )
        assert os.path.exists(output)

    def test_all_zero_ious(self, tmp_path):
        pytest.importorskip("matplotlib")
        from eovot.visualization.trajectory import plot_multi_tracker_iou_timeline

        output = str(tmp_path / "zero.png")
        plot_multi_tracker_iou_timeline(
            {"Zero": np.zeros(40)},
            output_path=output,
        )
        assert os.path.exists(output)

    def test_custom_failure_threshold(self, tmp_path):
        pytest.importorskip("matplotlib")
        from eovot.visualization.trajectory import plot_multi_tracker_iou_timeline

        output = str(tmp_path / "threshold.png")
        plot_multi_tracker_iou_timeline(
            {"MOSSE": _synthetic_ious(60)},
            failure_threshold=0.25,
            output_path=output,
        )
        assert os.path.exists(output)

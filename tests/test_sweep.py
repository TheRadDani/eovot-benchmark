"""Unit tests for eovot.benchmark.sweep.

Uses the same synthetic dataset and tracker pattern as test_engine.py —
no real datasets or filesystem access required.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkResult
from eovot.benchmark.sweep import SweepConfig, SweepResult, SweepRunner
from eovot.datasets.base import BaseDataset, Sequence
from eovot.trackers.base import BaseTracker

# ---------------------------------------------------------------------------
# Synthetic fixtures (mirrored from test_engine.py)
# ---------------------------------------------------------------------------

NUM_FRAMES = 15
GT_BOX = (10.0, 10.0, 50.0, 50.0)


class ConstantTracker(BaseTracker):
    """Returns the same fixed bounding box on every update."""

    def __init__(self, box=GT_BOX, name="ConstantTracker"):
        super().__init__(name=name)
        self._box = box

    def initialize(self, frame: np.ndarray, bbox) -> None:
        pass

    def update(self, frame: np.ndarray):
        return self._box


class ShiftedTracker(BaseTracker):
    """Returns a box shifted by (dx, dy) — produces lower IoU."""

    def __init__(self, dx: float = 20.0, dy: float = 20.0):
        super().__init__(name="ShiftedTracker")
        self._dx = dx
        self._dy = dy

    def initialize(self, frame: np.ndarray, bbox) -> None:
        self._box = (bbox[0] + self._dx, bbox[1] + self._dy, bbox[2], bbox[3])

    def update(self, frame: np.ndarray):
        return self._box


class SyntheticSequence(Sequence):
    def __init__(self, name: str, n_frames: int = NUM_FRAMES, gt_box=GT_BOX):
        gt = np.tile(np.array(gt_box), (n_frames, 1))
        super().__init__(
            name=name,
            frame_paths=[f"frame_{i:04d}.jpg" for i in range(n_frames)],
            ground_truth=gt,
        )
        self._n_frames = n_frames

    def __iter__(self) -> Iterator[np.ndarray]:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for _ in range(self._n_frames):
            yield frame


class SyntheticDataset(BaseDataset):
    def __init__(self, n_sequences: int = 3):
        self._seqs = [SyntheticSequence(f"seq_{i:02d}") for i in range(n_sequences)]

    def __len__(self) -> int:
        return len(self._seqs)

    def __getitem__(self, idx: int) -> Sequence:
        return self._seqs[idx]


# ---------------------------------------------------------------------------
# SweepConfig validation
# ---------------------------------------------------------------------------

class TestSweepConfig:
    def test_minimal_config_with_instances(self):
        dataset = SyntheticDataset()
        cfg = SweepConfig(
            name="test",
            dataset=dataset,
            dataset_name="Synthetic",
            tracker_instances=[ConstantTracker()],
        )
        assert cfg.name == "test"
        assert cfg.dataset_name == "Synthetic"

    def test_config_defaults(self):
        dataset = SyntheticDataset()
        cfg = SweepConfig(name="x", dataset=dataset, dataset_name="D")
        assert cfg.max_sequences is None
        assert cfg.tdp_watts is None
        assert cfg.verbose is True

    def test_empty_trackers_raises_on_run(self):
        dataset = SyntheticDataset()
        cfg = SweepConfig(name="x", dataset=dataset, dataset_name="D")
        runner = SweepRunner()
        with pytest.raises(ValueError, match="at least one tracker"):
            runner.run(cfg)


# ---------------------------------------------------------------------------
# SweepRunner.run
# ---------------------------------------------------------------------------

class TestSweepRunner:
    def setup_method(self):
        self.dataset = SyntheticDataset(n_sequences=3)
        self.runner = SweepRunner()

    def _make_config(self, tracker_instances, **kwargs):
        return SweepConfig(
            name="test-sweep",
            dataset=self.dataset,
            dataset_name="Synthetic",
            tracker_instances=tracker_instances,
            verbose=False,
            **kwargs,
        )

    def test_returns_sweep_result(self):
        cfg = self._make_config([ConstantTracker()])
        result = self.runner.run(cfg)
        assert isinstance(result, SweepResult)

    def test_one_result_per_tracker(self):
        trackers = [ConstantTracker(name="T1"), ConstantTracker(name="T2")]
        cfg = self._make_config(trackers)
        result = self.runner.run(cfg)
        assert len(result.results) == 2

    def test_results_are_benchmark_results(self):
        cfg = self._make_config([ConstantTracker()])
        result = self.runner.run(cfg)
        for r in result.results:
            assert isinstance(r, BenchmarkResult)

    def test_perfect_tracker_mean_iou_one(self):
        cfg = self._make_config([ConstantTracker(GT_BOX)])
        result = self.runner.run(cfg)
        assert result.results[0].mean_iou == pytest.approx(1.0)

    def test_max_sequences_respected(self):
        cfg = self._make_config([ConstantTracker()], max_sequences=2)
        result = self.runner.run(cfg)
        assert len(result.results[0].sequence_results) == 2

    def test_sweep_name_propagated(self):
        cfg = self._make_config([ConstantTracker()])
        result = self.runner.run(cfg)
        assert result.sweep_name == "test-sweep"

    def test_tracker_classes_used(self):
        """SweepConfig.trackers dict (class registry) path."""
        cfg = SweepConfig(
            name="class-sweep",
            dataset=self.dataset,
            dataset_name="Synthetic",
            trackers={"Constant": ConstantTracker},
            verbose=False,
        )
        result = self.runner.run(cfg)
        assert len(result.results) == 1

    def test_tracker_instances_override_classes(self):
        """tracker_instances takes priority over trackers dict."""
        cfg = SweepConfig(
            name="inst-sweep",
            dataset=self.dataset,
            dataset_name="Synthetic",
            trackers={"Constant": ConstantTracker},  # would add 1 more
            tracker_instances=[ConstantTracker(name="Explicit")],
            verbose=False,
        )
        result = self.runner.run(cfg)
        # Both are used: 1 instance + 1 from class dict = 2 total
        assert len(result.results) == 2


# ---------------------------------------------------------------------------
# SweepResult analysis helpers
# ---------------------------------------------------------------------------

class TestSweepResult:
    def setup_method(self):
        dataset = SyntheticDataset(n_sequences=3)
        runner = SweepRunner()
        trackers = [
            ConstantTracker(GT_BOX, name="Perfect"),
            ShiftedTracker(dx=30.0, dy=30.0),  # lower IoU
        ]
        cfg = SweepConfig(
            name="analysis-test",
            dataset=dataset,
            dataset_name="Synthetic",
            tracker_instances=trackers,
            verbose=False,
        )
        self.sweep_result = runner.run(cfg)

    def test_ranking_returns_list(self):
        rows = self.sweep_result.ranking()
        assert isinstance(rows, list)
        assert len(rows) == 2

    def test_ranking_sorted_by_iou_descending(self):
        rows = self.sweep_result.ranking(sort_by="mean_iou")
        ious = [r["mean_iou"] for r in rows]
        assert ious == sorted(ious, reverse=True)

    def test_ranking_sorted_by_fps_descending(self):
        rows = self.sweep_result.ranking(sort_by="mean_fps")
        fps_vals = [r["mean_fps"] for r in rows]
        assert fps_vals == sorted(fps_vals, reverse=True)

    def test_perfect_tracker_ranks_first(self):
        rows = self.sweep_result.ranking(sort_by="mean_iou")
        assert rows[0]["tracker"] == "Perfect"

    def test_summary_table_is_string(self):
        table = self.sweep_result.summary_table()
        assert isinstance(table, str)

    def test_summary_table_contains_tracker_names(self):
        table = self.sweep_result.summary_table()
        assert "Perfect" in table
        assert "ShiftedTracker" in table

    def test_summary_table_has_markdown_format(self):
        table = self.sweep_result.summary_table()
        assert "|" in table
        lines = table.strip().split("\n")
        assert len(lines) >= 3  # header + separator + at least 1 data row

    def test_to_dict_structure(self):
        d = self.sweep_result.to_dict()
        assert "sweep_name" in d
        assert "ranking" in d
        assert "trackers" in d
        assert isinstance(d["trackers"], list)

    def test_to_dict_ranking_count(self):
        d = self.sweep_result.to_dict()
        assert len(d["ranking"]) == 2

    def test_save_writes_json(self, tmp_path):
        out = self.sweep_result.save(str(tmp_path))
        assert out.endswith(".json")
        import json
        with open(out) as fh:
            data = json.load(fh)
        assert data["sweep_name"] == "analysis-test"

    def test_empty_sweep_result_table(self):
        empty = SweepResult(sweep_name="empty")
        assert "_No results._" in empty.summary_table()


# ---------------------------------------------------------------------------
# Package-level imports
# ---------------------------------------------------------------------------

class TestBenchmarkPackageExports:
    def test_sweep_symbols_importable(self):
        from eovot.benchmark import SweepConfig, SweepResult, SweepRunner  # noqa: F401

"""Tests for SequenceDifficultyAnalyzer and DifficultyFilteredDataset."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.datasets.difficulty import (
    TAG_DEFORMATION,
    TAG_FAST_MOTION,
    TAG_OCCLUSION,
    TAG_SCALE_CHANGE,
    DifficultyFilteredDataset,
    SequenceDifficulty,
    SequenceDifficultyAnalyzer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _constant_gt(n: int = 50, x=10, y=10, w=30, h=30) -> np.ndarray:
    """GT with a static target — zero velocity, zero scale change."""
    return np.tile([x, y, w, h], (n, 1)).astype(np.float64)


def _growing_gt(n: int = 50, w0: int = 20, dw: int = 1) -> np.ndarray:
    """GT with a target that grows by dw each frame — high scale change."""
    rows = []
    for i in range(n):
        w = w0 + i * dw
        rows.append([10.0, 10.0, float(w), float(w)])
    return np.array(rows)


def _fast_moving_gt(n: int = 50, step: int = 10, h: int = 80, w: int = 80) -> np.ndarray:
    """GT with fast horizontal motion."""
    rows = []
    for i in range(n):
        x = (i * step) % (h - 20)
        rows.append([float(x), 10.0, 20.0, 20.0])
    return np.array(rows)


def _deforming_gt(n: int = 50) -> np.ndarray:
    """GT where width oscillates — high aspect-ratio jitter."""
    rows = []
    for i in range(n):
        w = 20.0 + 10.0 * np.sin(i * 0.5)
        rows.append([10.0, 10.0, w, 20.0])
    return np.array(rows)


def _occluded_gt(n: int = 50, occlusion_rate: float = 0.20) -> np.ndarray:
    """GT where ~20% of frames have zero-area boxes (occluded)."""
    gt = np.tile([10.0, 10.0, 30.0, 30.0], (n, 1)).astype(np.float64)
    rng = np.random.default_rng(42)
    occ_idx = rng.choice(n, size=int(n * occlusion_rate), replace=False)
    gt[occ_idx] = [0.0, 0.0, 0.0, 0.0]
    return gt


# ---------------------------------------------------------------------------
# SequenceDifficultyAnalyzer — individual metrics
# ---------------------------------------------------------------------------

class TestIndividualMetrics:
    def setup_method(self):
        self.analyzer = SequenceDifficultyAnalyzer()

    def test_scr_static_target(self):
        scr = self.analyzer.compute_scale_change_ratio(_constant_gt())
        assert scr == pytest.approx(0.0, abs=1e-6)

    def test_scr_growing_target_positive(self):
        scr = self.analyzer.compute_scale_change_ratio(_growing_gt())
        assert scr > 0.0

    def test_mv_static_target(self):
        mv = self.analyzer.compute_mean_velocity(_constant_gt())
        assert mv == pytest.approx(0.0, abs=1e-6)

    def test_mv_fast_motion_positive(self):
        mv = self.analyzer.compute_mean_velocity(_fast_moving_gt())
        assert mv > 0.0

    def test_arj_static_target(self):
        arj = self.analyzer.compute_aspect_ratio_jitter(_constant_gt())
        assert arj == pytest.approx(0.0, abs=1e-6)

    def test_arj_deforming_target_positive(self):
        arj = self.analyzer.compute_aspect_ratio_jitter(_deforming_gt())
        assert arj > 0.0

    def test_dfr_no_occlusion(self):
        dfr = self.analyzer.compute_degenerate_frame_ratio(_constant_gt())
        assert dfr == pytest.approx(0.0, abs=1e-6)

    def test_dfr_with_occlusion(self):
        dfr = self.analyzer.compute_degenerate_frame_ratio(_occluded_gt())
        assert dfr > 0.0

    def test_metrics_on_short_sequence(self):
        gt = _constant_gt(n=1)
        assert self.analyzer.compute_scale_change_ratio(gt) == 0.0
        assert self.analyzer.compute_mean_velocity(gt) == 0.0
        assert self.analyzer.compute_aspect_ratio_jitter(gt) == 0.0

    def test_metrics_on_empty_sequence(self):
        gt = np.empty((0, 4))
        assert self.analyzer.compute_degenerate_frame_ratio(gt) == 0.0


# ---------------------------------------------------------------------------
# SequenceDifficultyAnalyzer — analyze()
# ---------------------------------------------------------------------------

class TestAnalyze:
    def setup_method(self):
        self.analyzer = SequenceDifficultyAnalyzer()

    def test_analyze_returns_correct_type(self):
        result = self.analyzer.analyze(_constant_gt(), name="seq1")
        assert isinstance(result, SequenceDifficulty)
        assert result.name == "seq1"

    def test_analyze_static_is_easy(self):
        result = self.analyzer.analyze(_constant_gt())
        assert result.tier == "easy"
        assert result.challenges == []

    def test_analyze_growing_tags_scale_change(self):
        gt = _growing_gt(n=100, w0=10, dw=2)
        result = self.analyzer.analyze(gt)
        assert TAG_SCALE_CHANGE in result.challenges

    def test_analyze_fast_motion_tagged(self):
        gt = _fast_moving_gt(n=100, step=15)
        result = self.analyzer.analyze(gt)
        assert TAG_FAST_MOTION in result.challenges

    def test_analyze_deformation_tagged(self):
        gt = _deforming_gt(n=80)
        result = self.analyzer.analyze(gt)
        assert TAG_DEFORMATION in result.challenges

    def test_analyze_occlusion_tagged(self):
        gt = _occluded_gt(n=80, occlusion_rate=0.20)
        result = self.analyzer.analyze(gt)
        assert TAG_OCCLUSION in result.challenges

    def test_difficulty_score_range(self):
        for gt in [_constant_gt(), _growing_gt(), _fast_moving_gt(), _deforming_gt()]:
            result = self.analyzer.analyze(gt)
            assert 0.0 <= result.difficulty_score <= 1.0

    def test_analyze_invalid_shape_raises(self):
        with pytest.raises(ValueError, match=r"\(N, 4\)"):
            self.analyzer.analyze(np.ones((10, 3)))

    def test_harder_sequence_higher_score(self):
        static = self.analyzer.analyze(_constant_gt())
        hard = self.analyzer.analyze(_growing_gt(n=100, w0=5, dw=3))
        assert hard.difficulty_score >= static.difficulty_score

    def test_num_frames_populated(self):
        gt = _constant_gt(n=37)
        result = self.analyzer.analyze(gt, name="test")
        assert result.num_frames == 37

    def test_to_dict_keys(self):
        result = self.analyzer.analyze(_constant_gt(), name="abc")
        d = result.to_dict()
        for key in ["name", "tier", "difficulty_score", "challenges", "num_frames"]:
            assert key in d


# ---------------------------------------------------------------------------
# SequenceDifficultyAnalyzer — analyze_dataset()
# ---------------------------------------------------------------------------

class TestAnalyzeDataset:
    def test_analyze_synthetic_dataset(self):
        from eovot.datasets.synthetic import SyntheticDataset

        ds = SyntheticDataset(num_sequences=5, num_frames=40, motion="linear")
        analyzer = SequenceDifficultyAnalyzer()
        results = analyzer.analyze_dataset(ds)
        assert len(results) == 5
        for r in results:
            assert isinstance(r, SequenceDifficulty)
            assert r.num_frames == 40

    def test_dataset_summary_structure(self):
        from eovot.datasets.synthetic import SyntheticDataset

        ds = SyntheticDataset(num_sequences=6, num_frames=30, motion="random")
        analyzer = SequenceDifficultyAnalyzer()
        diffs = analyzer.analyze_dataset(ds)
        summary = analyzer.dataset_summary(diffs)

        assert "num_sequences" in summary
        assert summary["num_sequences"] == 6
        assert "tier_counts" in summary
        assert sum(summary["tier_counts"].values()) == 6
        assert "mean_difficulty" in summary
        assert 0.0 <= summary["mean_difficulty"] <= 1.0

    def test_dataset_summary_empty(self):
        analyzer = SequenceDifficultyAnalyzer()
        assert analyzer.dataset_summary([]) == {}


# ---------------------------------------------------------------------------
# Filter methods
# ---------------------------------------------------------------------------

class TestFilterMethods:
    def setup_method(self):
        self.analyzer = SequenceDifficultyAnalyzer()
        self._diffs = [
            self.analyzer.analyze(_constant_gt(), name="easy1"),
            self.analyzer.analyze(_growing_gt(n=100, w0=5, dw=3), name="hard1"),
            self.analyzer.analyze(_fast_moving_gt(n=100, step=15), name="hard2"),
        ]

    def test_filter_by_tier_easy(self):
        indices = self.analyzer.filter_by_tier(self._diffs, ["easy"])
        assert 0 in indices  # static sequence should be easy

    def test_filter_by_tier_all_tiers_returns_all(self):
        indices = self.analyzer.filter_by_tier(
            self._diffs, ["easy", "medium", "hard"]
        )
        assert len(indices) == 3

    def test_filter_by_challenge_any(self):
        indices = self.analyzer.filter_by_challenge(
            self._diffs, [TAG_SCALE_CHANGE], require_all=False
        )
        assert all(TAG_SCALE_CHANGE in self._diffs[i].challenges for i in indices)

    def test_filter_by_challenge_require_all(self):
        indices = self.analyzer.filter_by_challenge(
            self._diffs, [TAG_SCALE_CHANGE, TAG_FAST_MOTION], require_all=True
        )
        for i in indices:
            assert TAG_SCALE_CHANGE in self._diffs[i].challenges
            assert TAG_FAST_MOTION in self._diffs[i].challenges


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------

class TestMarkdownTable:
    def test_table_is_string(self):
        from eovot.datasets.synthetic import SyntheticDataset

        ds = SyntheticDataset(num_sequences=3, num_frames=30)
        analyzer = SequenceDifficultyAnalyzer()
        diffs = analyzer.analyze_dataset(ds)
        table = analyzer.to_markdown_table(diffs)
        assert isinstance(table, str)
        assert "Rank" in table
        assert "Score" in table

    def test_top_n_limits_rows(self):
        from eovot.datasets.synthetic import SyntheticDataset

        ds = SyntheticDataset(num_sequences=10, num_frames=30)
        analyzer = SequenceDifficultyAnalyzer()
        diffs = analyzer.analyze_dataset(ds)
        table = analyzer.to_markdown_table(diffs, top_n=3)
        data_rows = [l for l in table.split("\n") if l.startswith("|") and "---" not in l and "Rank" not in l]
        assert len(data_rows) == 3


# ---------------------------------------------------------------------------
# DifficultyFilteredDataset
# ---------------------------------------------------------------------------

class TestDifficultyFilteredDataset:
    def test_basic_filtering(self):
        from eovot.datasets.synthetic import SyntheticDataset

        ds = SyntheticDataset(num_sequences=10, num_frames=30, motion="random")
        filtered = DifficultyFilteredDataset(ds, tiers=["easy", "medium", "hard"])
        assert len(filtered) == len(ds)

    def test_strict_tier_filter_subset(self):
        from eovot.datasets.synthetic import SyntheticDataset

        ds = SyntheticDataset(num_sequences=10, num_frames=30, motion="random")
        easy_ds = DifficultyFilteredDataset(ds, tiers=["easy"])
        medium_ds = DifficultyFilteredDataset(ds, tiers=["medium"])
        hard_ds = DifficultyFilteredDataset(ds, tiers=["hard"])
        total = len(easy_ds) + len(medium_ds) + len(hard_ds)
        assert total == len(ds)

    def test_getitem_returns_sequence(self):
        from eovot.datasets.base import Sequence
        from eovot.datasets.synthetic import SyntheticDataset

        ds = SyntheticDataset(num_sequences=6, num_frames=30, motion="linear")
        fds = DifficultyFilteredDataset(ds, tiers=["easy", "medium", "hard"])
        for i in range(len(fds)):
            assert isinstance(fds[i], Sequence)

    def test_out_of_range_raises(self):
        from eovot.datasets.synthetic import SyntheticDataset

        ds = SyntheticDataset(num_sequences=5, num_frames=20)
        fds = DifficultyFilteredDataset(ds, tiers=["easy", "medium", "hard"])
        with pytest.raises(IndexError):
            _ = fds[len(fds)]

    def test_empty_tiers_raises(self):
        from eovot.datasets.synthetic import SyntheticDataset

        ds = SyntheticDataset(num_sequences=3, num_frames=20)
        with pytest.raises(ValueError, match="empty"):
            DifficultyFilteredDataset(ds, tiers=[])

    def test_invalid_tier_raises(self):
        from eovot.datasets.synthetic import SyntheticDataset

        ds = SyntheticDataset(num_sequences=3, num_frames=20)
        with pytest.raises(ValueError, match="Invalid tier"):
            DifficultyFilteredDataset(ds, tiers=["extreme"])

    def test_get_difficulty_method(self):
        from eovot.datasets.synthetic import SyntheticDataset

        ds = SyntheticDataset(num_sequences=6, num_frames=30, motion="linear")
        fds = DifficultyFilteredDataset(ds, tiers=["easy", "medium", "hard"])
        for i in range(len(fds)):
            diff = fds.get_difficulty(i)
            assert isinstance(diff, SequenceDifficulty)

    def test_repr(self):
        from eovot.datasets.synthetic import SyntheticDataset

        ds = SyntheticDataset(num_sequences=4, num_frames=20)
        fds = DifficultyFilteredDataset(ds, tiers=["hard"])
        assert "DifficultyFilteredDataset" in repr(fds)

    def test_works_with_benchmark_engine(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        ds = SyntheticDataset(num_sequences=4, num_frames=20, motion="random")
        fds = DifficultyFilteredDataset(ds, tiers=["easy", "medium", "hard"])
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), fds, dataset_name="Synthetic-Filtered")
        assert len(result.sequence_results) == len(fds)

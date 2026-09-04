"""Tests for eovot.datasets.occlusion and eovot.metrics.occlusion."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.datasets.occlusion import OcclusionSequence, OcclusionSyntheticDataset
from eovot.metrics.occlusion import OcclusionRobustnessAnalyzer, OcclusionRobustnessResult


# ===========================================================================
# OcclusionSyntheticDataset
# ===========================================================================


class TestOcclusionSyntheticDataset:

    def _default_ds(self, **kwargs) -> OcclusionSyntheticDataset:
        defaults = dict(
            num_sequences=3,
            num_frames=60,
            occlusion_interval=15,
            occlusion_duration=5,
            seed=0,
        )
        defaults.update(kwargs)
        return OcclusionSyntheticDataset(**defaults)

    # --- construction -------------------------------------------------------

    def test_default_construction(self):
        ds = OcclusionSyntheticDataset()
        assert len(ds) == 5

    def test_invalid_motion_raises(self):
        with pytest.raises(ValueError, match="Unknown motion"):
            OcclusionSyntheticDataset(motion="invalid")

    def test_interval_lte_duration_raises(self):
        with pytest.raises(ValueError):
            OcclusionSyntheticDataset(occlusion_interval=5, occlusion_duration=5)
        with pytest.raises(ValueError):
            OcclusionSyntheticDataset(occlusion_interval=3, occlusion_duration=5)

    def test_zero_duration_raises(self):
        with pytest.raises(ValueError):
            OcclusionSyntheticDataset(occlusion_duration=0)

    def test_repr(self):
        ds = self._default_ds()
        r = repr(ds)
        assert "OcclusionSyntheticDataset" in r
        assert "sequences=3" in r

    def test_index_out_of_range_raises(self):
        ds = self._default_ds()
        with pytest.raises(IndexError):
            ds[10]
        with pytest.raises(IndexError):
            ds[-1]

    # --- sequence properties ------------------------------------------------

    def test_sequence_length_matches(self):
        ds = self._default_ds(num_frames=60)
        seq = ds[0]
        assert len(seq) == 60

    def test_ground_truth_shape(self):
        ds = self._default_ds(num_frames=60)
        seq = ds[0]
        assert seq.ground_truth.shape == (60, 4)

    def test_occlusion_mask_shape_and_dtype(self):
        ds = self._default_ds(num_frames=60)
        seq = ds[0]
        assert seq.occlusion_mask.shape == (60,)
        assert seq.occlusion_mask.dtype == bool

    def test_occlusion_mask_has_expected_events(self):
        """With interval=15 and duration=5, expect onsets at 15, 30, 45."""
        ds = self._default_ds(num_frames=60, occlusion_interval=15, occlusion_duration=5)
        seq = ds[0]
        mask = seq.occlusion_mask
        # Check first event starts at frame 15
        assert mask[15] is np.bool_(True)
        assert mask[14] is np.bool_(False)
        # Check event ends at frame 20 (exclusive)
        assert mask[19] is np.bool_(True)
        assert mask[20] is np.bool_(False)

    def test_occlusion_mask_total_occluded_frames(self):
        """3 events × 5 frames each = 15 occluded frames in a 60-frame sequence."""
        ds = self._default_ds(num_frames=60, occlusion_interval=15, occlusion_duration=5)
        seq = ds[0]
        assert seq.occlusion_mask.sum() == 15

    def test_frames_are_bgr_uint8(self):
        ds = self._default_ds()
        seq = ds[0]
        frames = list(seq)
        assert len(frames) == len(seq)
        for f in frames:
            assert f.dtype == np.uint8
            assert f.ndim == 3
            assert f.shape[2] == 3

    def test_init_bbox_matches_first_gt(self):
        ds = self._default_ds()
        seq = ds[0]
        assert seq.init_bbox == tuple(seq.ground_truth[0])

    def test_different_seeds_produce_different_sequences(self):
        ds_a = self._default_ds(seed=0)
        ds_b = self._default_ds(seed=999)
        gt_a = ds_a[0].ground_truth
        gt_b = ds_b[0].ground_truth
        assert not np.allclose(gt_a, gt_b)

    def test_same_seed_is_reproducible(self):
        ds_a = self._default_ds(seed=7)
        ds_b = self._default_ds(seed=7)
        np.testing.assert_array_equal(ds_a[0].ground_truth, ds_b[0].ground_truth)
        frames_a = list(ds_a[0])
        frames_b = list(ds_b[0])
        for fa, fb in zip(frames_a, frames_b):
            np.testing.assert_array_equal(fa, fb)

    def test_caching_returns_same_object(self):
        ds = self._default_ds()
        seq_a = ds[0]
        seq_b = ds[0]
        assert seq_a is seq_b

    def test_random_motion(self):
        ds = OcclusionSyntheticDataset(
            num_sequences=2,
            num_frames=40,
            occlusion_interval=12,
            occlusion_duration=4,
            motion="random",
            seed=1,
        )
        seq = ds[0]
        assert len(seq) == 40
        assert seq.occlusion_mask.sum() > 0

    def test_occluded_frames_look_different_from_clear(self):
        """Occluded frames should differ from clear frames at the target location."""
        ds = self._default_ds(num_frames=60, occlusion_interval=15, occlusion_duration=5)
        seq = ds[0]
        frames = list(seq)
        mask = seq.occlusion_mask
        gt = seq.ground_truth

        # First clear frame
        clear_idx = int(np.where(~mask)[0][5])
        # First occluded frame
        occ_idx = int(np.where(mask)[0][0])

        x, y, w, h = [int(v) for v in gt[clear_idx]]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(320, x + w), min(240, y + h)

        clear_patch = frames[clear_idx][y1:y2, x1:x2].astype(float)
        occ_patch = frames[occ_idx][y1:y2, x1:x2].astype(float)
        # Patches must differ (occluder changes appearance)
        assert not np.allclose(clear_patch, occ_patch)

    def test_sequence_repr(self):
        ds = self._default_ds()
        r = repr(ds[0])
        assert "OcclusionSequence" in r

    def test_iter_full_dataset(self):
        ds = self._default_ds(num_sequences=3)
        seqs = list(ds)
        assert len(seqs) == 3
        for seq in seqs:
            assert isinstance(seq, OcclusionSequence)


# ===========================================================================
# OcclusionRobustnessAnalyzer
# ===========================================================================


class TestOcclusionRobustnessAnalyzer:

    def _analyzer(self, **kwargs) -> OcclusionRobustnessAnalyzer:
        return OcclusionRobustnessAnalyzer(**kwargs)

    # --- construction -------------------------------------------------------

    def test_default_construction(self):
        a = self._analyzer()
        assert a.context_frames == 10
        assert a.recovery_window == 15

    def test_invalid_context_frames_raises(self):
        with pytest.raises(ValueError):
            OcclusionRobustnessAnalyzer(context_frames=0)

    def test_invalid_recovery_window_raises(self):
        with pytest.raises(ValueError):
            OcclusionRobustnessAnalyzer(recovery_window=0)

    # --- _detect_events helper ---------------------------------------------

    def test_detect_events_single(self):
        mask = np.array([False] * 5 + [True] * 5 + [False] * 5, dtype=bool)
        events = OcclusionRobustnessAnalyzer._detect_events(mask)
        assert events == [(5, 10)]

    def test_detect_events_multiple(self):
        mask = np.zeros(30, dtype=bool)
        mask[5:10] = True
        mask[20:25] = True
        events = OcclusionRobustnessAnalyzer._detect_events(mask)
        assert events == [(5, 10), (20, 25)]

    def test_detect_events_trailing_occlusion(self):
        mask = np.array([False] * 5 + [True] * 5, dtype=bool)
        events = OcclusionRobustnessAnalyzer._detect_events(mask)
        assert events == [(5, 10)]

    def test_detect_events_no_occlusion(self):
        mask = np.zeros(20, dtype=bool)
        assert OcclusionRobustnessAnalyzer._detect_events(mask) == []

    def test_detect_events_all_occluded(self):
        mask = np.ones(10, dtype=bool)
        events = OcclusionRobustnessAnalyzer._detect_events(mask)
        assert events == [(0, 10)]

    # --- analyze: edge cases -----------------------------------------------

    def test_no_occlusion_returns_zeros(self):
        a = self._analyzer()
        ious = np.linspace(0.8, 0.9, 30)
        mask = np.zeros(30, dtype=bool)
        r = a.analyze(ious, mask, "T", "seq")
        assert r.num_occlusion_events == 0
        assert r.during_occlusion_iou == 0.0
        assert r.recovery_auc == 0.0

    def test_all_occluded(self):
        a = self._analyzer()
        ious = np.full(20, 0.5)
        mask = np.ones(20, dtype=bool)
        r = a.analyze(ious, mask, "T", "seq")
        assert r.num_occlusion_events == 1
        assert r.during_occlusion_iou == pytest.approx(0.5)
        assert r.pre_occlusion_iou == 0.0  # no pre-occlusion frames

    # --- analyze: correctness on synthetic data ----------------------------

    def test_pre_iou_higher_than_during_on_degraded_ious(self):
        """IoU should be lower during occlusion than before it."""
        a = self._analyzer(context_frames=5, recovery_window=5)
        # IoU 0.9 before occlusion, 0.2 during, 0.8 after
        n = 60
        ious = np.full(n, 0.9)
        mask = np.zeros(n, dtype=bool)
        mask[20:28] = True
        ious[20:28] = 0.2
        r = a.analyze(ious, mask, "T", "seq")
        assert r.pre_occlusion_iou > r.during_occlusion_iou

    def test_post_iou_recovered_after_clear(self):
        """Post-occlusion IoU should be high when tracker recovers."""
        a = self._analyzer(context_frames=5, recovery_window=8)
        n = 60
        ious = np.full(n, 0.85)
        mask = np.zeros(n, dtype=bool)
        mask[15:22] = True
        ious[15:22] = 0.1  # during occlusion drop
        r = a.analyze(ious, mask)
        assert r.post_occlusion_iou > 0.7

    def test_iou_drop_ratio_bounds(self):
        a = self._analyzer()
        ious = np.full(50, 0.8)
        mask = np.zeros(50, dtype=bool)
        mask[20:28] = True
        ious[20:28] = 0.2
        r = a.analyze(ious, mask)
        assert 0.0 <= r.iou_drop_ratio <= 1.0

    def test_recovery_auc_perfect_recovery(self):
        """When IoU returns to 1.0 immediately, recovery AUC should be near 1.0."""
        a = self._analyzer(context_frames=5, recovery_window=5)
        n = 60
        ious = np.ones(n)
        mask = np.zeros(n, dtype=bool)
        mask[20:25] = True
        ious[20:25] = 0.0
        r = a.analyze(ious, mask)
        assert r.recovery_auc > 0.8

    def test_result_str_repr(self):
        a = self._analyzer()
        ious = np.linspace(0.5, 0.9, 40)
        mask = np.zeros(40, dtype=bool)
        mask[15:20] = True
        r = a.analyze(ious, mask, "MOSSE", "seq_01")
        s = str(r)
        assert "MOSSE" in s
        assert "seq_01" in s

    def test_result_fields_populated(self):
        a = self._analyzer(context_frames=5, recovery_window=5)
        ious = np.full(50, 0.7)
        mask = np.zeros(50, dtype=bool)
        mask[15:23] = True
        ious[15:23] = 0.2
        r = a.analyze(ious, mask, "KCF", "testseq")
        assert isinstance(r, OcclusionRobustnessResult)
        assert r.tracker_name == "KCF"
        assert r.sequence_name == "testseq"
        assert r.num_occlusion_events == 1
        assert isinstance(r.pre_occlusion_iou, float)
        assert isinstance(r.recovery_auc, float)

    # --- to_markdown_table -------------------------------------------------

    def test_markdown_table_output(self):
        a = self._analyzer()
        results = []
        for i in range(3):
            ious = np.linspace(0.5, 0.9, 40)
            mask = np.zeros(40, dtype=bool)
            mask[15:20] = True
            results.append(a.analyze(ious, mask, "T", f"seq_{i}"))
        md = a.to_markdown_table(results)
        assert "| Sequence |" in md
        assert "seq_0" in md
        assert "seq_2" in md

    # --- analyze_benchmark integration ------------------------------------

    def test_analyze_benchmark_integration(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.trackers.mosse import MOSSETracker

        ds = OcclusionSyntheticDataset(
            num_sequences=2,
            num_frames=60,
            occlusion_interval=15,
            occlusion_duration=5,
            seed=0,
        )
        engine = BenchmarkEngine(verbose=False)
        bench_result = engine.run(MOSSETracker(), ds, dataset_name="OcclusionSynth")

        analyzer = OcclusionRobustnessAnalyzer(context_frames=5, recovery_window=8)
        agg_result = analyzer.analyze_benchmark(bench_result, ds, tracker_name="MOSSE")

        assert "aggregate" in agg_result
        assert "per_sequence" in agg_result
        agg = agg_result["aggregate"]
        assert agg["num_sequences"] == 2
        assert agg["total_occlusion_events"] > 0
        assert "mean_pre_occlusion_iou" in agg
        assert "mean_during_occlusion_iou" in agg
        assert "mean_post_occlusion_iou" in agg
        assert "mean_recovery_auc" in agg
        assert "mean_iou_drop_ratio" in agg

    def test_analyze_benchmark_empty_returns_empty(self):
        from eovot.benchmark.engine import BenchmarkResult

        ds = OcclusionSyntheticDataset(num_sequences=2, num_frames=30,
                                       occlusion_interval=10, occlusion_duration=3)
        empty_result = BenchmarkResult(tracker_name="T", dataset_name="D")
        analyzer = OcclusionRobustnessAnalyzer()
        out = analyzer.analyze_benchmark(empty_result, ds, tracker_name="T")
        assert out["per_sequence"] == {}
        assert out["aggregate"] == {}

"""Tests for eovot.metrics.attributes — per-attribute performance analysis."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.metrics.attributes import (
    AttributeAnalyzer,
    AttributeAnalysis,
    AttributeResult,
    LASOT_ATTRIBUTES,
    OTB_ATTRIBUTES,
    AUTO_DERIVABLE_ATTRIBUTES,
    derive_fast_motion_mask,
    derive_scale_variation_mask,
    derive_low_resolution_mask,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _perfect_boxes(n: int = 50, x: float = 10.0, y: float = 10.0,
                   w: float = 40.0, h: float = 30.0) -> np.ndarray:
    """Return *n* identical ground-truth boxes (tracker sees perfect IoU=1)."""
    return np.tile([x, y, w, h], (n, 1)).astype(np.float32)


def _noisy_boxes(gt: np.ndarray, noise_std: float = 3.0,
                 seed: int = 0) -> np.ndarray:
    """Add Gaussian noise to boxes; IoU stays high but < 1."""
    rng = np.random.default_rng(seed)
    return (gt + rng.normal(0, noise_std, gt.shape)).astype(np.float32)


# ── Mask derivation functions ─────────────────────────────────────────────────


class TestDeriveFastMotionMask:
    def test_stationary_target_no_fast_motion(self):
        gt = _perfect_boxes(20)
        mask = derive_fast_motion_mask(gt, velocity_threshold=20.0)
        assert mask.shape == (20,)
        assert not mask.any(), "Stationary target should never be fast-motion"

    def test_fast_target_mostly_flagged(self):
        # Target moves 50 px per frame
        gt = np.column_stack([
            np.arange(100) * 50.0,
            np.zeros(100),
            np.full(100, 40.0),
            np.full(100, 30.0),
        ])
        mask = derive_fast_motion_mask(gt, velocity_threshold=20.0)
        # Frame 0 is always False; frames 1..99 should all be True
        assert not mask[0]
        assert mask[1:].all()

    def test_first_frame_always_false(self):
        gt = np.random.rand(30, 4) * 100
        mask = derive_fast_motion_mask(gt)
        assert not mask[0], "First frame has no previous frame; must be False"

    def test_output_dtype_is_bool(self):
        gt = _perfect_boxes(10)
        assert derive_fast_motion_mask(gt).dtype == bool


class TestDeriveScaleVariationMask:
    def test_constant_size_no_variation(self):
        gt = _perfect_boxes(30)
        mask = derive_scale_variation_mask(gt, ratio_threshold=0.25)
        assert not mask.any()

    def test_doubling_area_flagged(self):
        gt = np.array([[0, 0, 40, 30]] * 30, dtype=np.float32)
        # Double width and height starting at frame 10 → area ×4
        gt[10:, 2] = 80.0
        gt[10:, 3] = 60.0
        mask = derive_scale_variation_mask(gt, ratio_threshold=0.25)
        assert mask[10:].all(), "4× area should be scale-variation"
        assert not mask[:10].any()

    def test_output_dtype_is_bool(self):
        gt = _perfect_boxes(10)
        assert derive_scale_variation_mask(gt).dtype == bool


class TestDeriveLowResolutionMask:
    def test_large_boxes_not_low_res(self):
        gt = _perfect_boxes(20, w=100.0, h=100.0)  # area = 10,000 px²
        mask = derive_low_resolution_mask(gt, area_threshold=400.0)
        assert not mask.any()

    def test_tiny_boxes_all_low_res(self):
        gt = _perfect_boxes(20, w=10.0, h=10.0)  # area = 100 px²
        mask = derive_low_resolution_mask(gt, area_threshold=400.0)
        assert mask.all()

    def test_output_dtype_is_bool(self):
        gt = _perfect_boxes(10)
        assert derive_low_resolution_mask(gt).dtype == bool


# ── AttributeAnalyzer: single-sequence ───────────────────────────────────────


class TestAnalyzeSequence:
    def setup_method(self):
        self.analyzer = AttributeAnalyzer()

    def _run(self, n: int = 50, noise: float = 2.0):
        gt = _perfect_boxes(n)
        pred = _noisy_boxes(gt, noise_std=noise)
        masks = {
            "occlusion": np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(bool),
            "fast_motion": np.zeros(n, dtype=bool),
        }
        return self.analyzer.analyze_sequence(pred, gt, masks), pred, gt

    def test_returns_attribute_analysis(self):
        result, _, _ = self._run()
        assert isinstance(result, AttributeAnalysis)

    def test_per_attribute_keys_match_masks(self):
        result, _, _ = self._run()
        # "fast_motion" has 0 True frames → should be absent (< 2 frames)
        assert "occlusion" in result.per_attribute
        assert "fast_motion" not in result.per_attribute

    def test_overall_metrics_populated(self):
        result, _, _ = self._run()
        assert result.overall is not None
        assert 0.0 <= result.overall.mean_iou <= 1.0

    def test_attribute_result_fields(self):
        result, _, _ = self._run()
        attr = result.per_attribute["occlusion"]
        assert isinstance(attr, AttributeResult)
        assert attr.attribute == "occlusion"
        assert attr.num_frames == 25  # half of 50 frames
        assert 0.0 <= attr.mean_iou <= 1.0
        assert 0.0 <= attr.success_auc <= 1.0
        assert 0.0 <= attr.precision_auc <= 1.0

    def test_perfect_predictions_iou_one(self):
        n = 40
        gt = _perfect_boxes(n)
        pred = gt.copy()
        masks = {"all_frames": np.ones(n, dtype=bool)}
        result = self.analyzer.analyze_sequence(pred, gt, masks)
        assert result.per_attribute["all_frames"].mean_iou == pytest.approx(1.0)

    def test_empty_mask_attribute_skipped(self):
        n = 30
        gt = _perfect_boxes(n)
        pred = gt.copy()
        masks = {"empty_attr": np.zeros(n, dtype=bool)}
        result = self.analyzer.analyze_sequence(pred, gt, masks)
        assert "empty_attr" not in result.per_attribute

    def test_mask_longer_than_frames_is_clipped(self):
        n = 20
        gt = _perfect_boxes(n)
        pred = gt.copy()
        masks = {"long_mask": np.ones(n + 10, dtype=bool)}
        # Should not raise
        result = self.analyzer.analyze_sequence(pred, gt, masks)
        assert "long_mask" in result.per_attribute


# ── AttributeAnalyzer: benchmark (multi-sequence) ────────────────────────────


class TestAnalyzeBenchmark:
    def setup_method(self):
        self.analyzer = AttributeAnalyzer()

    def _make_seq(self, n: int = 30):
        gt = _perfect_boxes(n)
        pred = _noisy_boxes(gt, noise_std=1.0)
        masks = {
            "occlusion": (np.arange(n) % 2 == 0),
            "fast_motion": np.zeros(n, dtype=bool),
        }
        return pred, gt, masks

    def test_returns_attribute_analysis(self):
        preds, gts, masks = zip(*[self._make_seq() for _ in range(3)])
        result = self.analyzer.analyze_benchmark(
            list(preds), list(gts), list(masks)
        )
        assert isinstance(result, AttributeAnalysis)

    def test_pooled_frame_count_correct(self):
        n = 30
        preds, gts, masks = zip(*[self._make_seq(n) for _ in range(4)])
        result = self.analyzer.analyze_benchmark(
            list(preds), list(gts), list(masks)
        )
        occ = result.per_attribute.get("occlusion")
        assert occ is not None
        # Every other frame is occlusion → 15 per sequence × 4 = 60
        assert occ.num_frames == 60

    def test_overall_always_present(self):
        preds, gts, masks = zip(*[self._make_seq() for _ in range(2)])
        result = self.analyzer.analyze_benchmark(
            list(preds), list(gts), list(masks)
        )
        assert result.overall is not None


# ── Auto-derive masks ─────────────────────────────────────────────────────────


class TestAutoDeriveMasks:
    def setup_method(self):
        self.analyzer = AttributeAnalyzer()

    def test_returns_three_masks(self):
        gt = _perfect_boxes(50)
        masks = self.analyzer.auto_derive_masks(gt)
        assert set(masks.keys()) == {"fast_motion", "scale_variation", "low_resolution"}

    def test_mask_lengths_match_gt(self):
        gt = _perfect_boxes(40)
        masks = self.analyzer.auto_derive_masks(gt)
        for name, mask in masks.items():
            assert len(mask) == 40, f"{name} mask length mismatch"

    def test_all_bool_arrays(self):
        gt = _perfect_boxes(20)
        masks = self.analyzer.auto_derive_masks(gt)
        for mask in masks.values():
            assert mask.dtype == bool


# ── OTB attribute extraction ──────────────────────────────────────────────────


class TestExtractOtbAttributes:
    def setup_method(self):
        self.analyzer = AttributeAnalyzer()

    def test_tags_become_all_true_masks(self):
        n = 50
        masks = self.analyzer.extract_otb_attributes(["occlusion", "fast_motion"], n)
        assert set(masks.keys()) == {"occlusion", "fast_motion"}
        for mask in masks.values():
            assert mask.all()
            assert len(mask) == n

    def test_tag_normalisation(self):
        masks = self.analyzer.extract_otb_attributes(["Background Clutter"], 10)
        assert "background_clutter" in masks

    def test_empty_tags_returns_empty_dict(self):
        masks = self.analyzer.extract_otb_attributes([], 20)
        assert masks == {}


# ── AttributeAnalysis formatting ─────────────────────────────────────────────


class TestAttributeAnalysisFormatting:
    def _build_analysis(self):
        analyzer = AttributeAnalyzer()
        n = 60
        gt = _perfect_boxes(n)
        pred = _noisy_boxes(gt, noise_std=3.0)
        masks = {
            "occlusion": np.concatenate([np.ones(30), np.zeros(30)]).astype(bool),
            "fast_motion": np.concatenate([np.zeros(30), np.ones(30)]).astype(bool),
        }
        return analyzer.analyze_sequence(pred, gt, masks)

    def test_summary_table_is_string(self):
        analysis = self._build_analysis()
        table = analysis.summary_table()
        assert isinstance(table, str)
        assert "mIoU" in table

    def test_worst_attributes_returns_list(self):
        analysis = self._build_analysis()
        worst = analysis.worst_attributes(n=1)
        assert len(worst) == 1
        assert isinstance(worst[0], AttributeResult)

    def test_best_attributes_sorted_descending(self):
        analysis = self._build_analysis()
        best = analysis.best_attributes(n=2)
        assert len(best) <= 2
        if len(best) == 2:
            assert best[0].mean_iou >= best[1].mean_iou

    def test_to_dict_serialisable(self):
        analysis = self._build_analysis()
        d = analysis.to_dict()
        import json
        json.dumps(d)  # should not raise

    def test_empty_analysis_summary_table(self):
        analysis = AttributeAnalysis()
        assert "No attribute data" in analysis.summary_table()


# ── Vocabulary constants ──────────────────────────────────────────────────────


class TestVocabularyConstants:
    def test_lasot_attributes_not_empty(self):
        assert len(LASOT_ATTRIBUTES) > 0

    def test_otb_attributes_not_empty(self):
        assert len(OTB_ATTRIBUTES) > 0

    def test_auto_derivable_subset(self):
        for attr in AUTO_DERIVABLE_ATTRIBUTES:
            assert isinstance(attr, str)

    def test_no_duplicates_in_lasot(self):
        assert len(LASOT_ATTRIBUTES) == len(set(LASOT_ATTRIBUTES))

    def test_no_duplicates_in_otb(self):
        assert len(OTB_ATTRIBUTES) == len(set(OTB_ATTRIBUTES))

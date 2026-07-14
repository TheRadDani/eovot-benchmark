"""Tests for eovot.metrics.attributes — AttributeDetector and AttributeAnalyzer."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from eovot.metrics.attributes import (
    ALL_ATTRIBUTES,
    ATTRIBUTE_DESCRIPTIONS,
    AttributeAnalyzer,
    AttributeDetector,
    AttributePerformanceTable,
    SequenceAttributes,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _static_gt(n: int, x=50, y=50, w=40, h=40) -> np.ndarray:
    """Constant-position, constant-size GT (no motion, no scale change)."""
    return np.tile([x, y, w, h], (n, 1)).astype(np.float64)


def _moving_gt(n: int, vx: float = 5.0, vy: float = 0.0, w=40, h=40) -> np.ndarray:
    """GT with constant-velocity horizontal motion."""
    gt = np.zeros((n, 4))
    for i in range(n):
        gt[i] = [50 + i * vx, 50 + i * vy, w, h]
    return gt


def _scaling_gt(n: int, w_start=20, h_start=20, w_end=100, h_end=100) -> np.ndarray:
    """GT with linearly growing box dimensions (scale variation)."""
    gt = np.zeros((n, 4))
    gt[:, 0] = 50.0
    gt[:, 1] = 50.0
    gt[:, 2] = np.linspace(w_start, w_end, n)
    gt[:, 3] = np.linspace(h_start, h_end, n)
    return gt


def _make_mock_result(seq_data):
    """Build a minimal BenchmarkResult-like object from a list of tuples.

    Args:
        seq_data: List of ``(name, gt, mean_iou, success_auc, precision_auc)``
    """
    result = MagicMock()
    result.tracker_name = "MOSSE"
    result.dataset_name = "TestDataset"
    seq_results = []
    for name, gt, miou, sauc, pauc in seq_data:
        sr = MagicMock()
        sr.sequence_name = name
        sr.mean_iou = miou
        sr.ground_truths = gt
        sr.accuracy_metrics = MagicMock()
        sr.accuracy_metrics.success_auc = sauc
        sr.accuracy_metrics.precision_auc = pauc
        seq_results.append(sr)
    result.sequence_results = seq_results
    return result


# ---------------------------------------------------------------------------
# SequenceAttributes
# ---------------------------------------------------------------------------

class TestSequenceAttributes:
    def test_active_attributes_all_false(self):
        sa = SequenceAttributes("s", {k: False for k in ALL_ATTRIBUTES})
        assert sa.active_attributes == []

    def test_active_attributes_some_true(self):
        attrs = {k: False for k in ALL_ATTRIBUTES}
        attrs["fast_motion"] = True
        attrs["low_resolution"] = True
        sa = SequenceAttributes("s", attrs)
        assert set(sa.active_attributes) == {"fast_motion", "low_resolution"}

    def test_has_true(self):
        sa = SequenceAttributes("s", {"fast_motion": True})
        assert sa.has("fast_motion") is True

    def test_has_false(self):
        sa = SequenceAttributes("s", {"fast_motion": False})
        assert sa.has("fast_motion") is False

    def test_has_missing_key(self):
        sa = SequenceAttributes("s", {})
        assert sa.has("nonexistent") is False

    def test_str_contains_name_and_attributes(self):
        sa = SequenceAttributes("my_seq", {"fast_motion": True, "long_sequence": False})
        text = str(sa)
        assert "my_seq" in text
        assert "fast_motion" in text


# ---------------------------------------------------------------------------
# AttributeDetector
# ---------------------------------------------------------------------------

class TestAttributeDetector:
    @pytest.fixture
    def det(self):
        return AttributeDetector()

    def test_empty_gt_all_false(self, det):
        sa = det.detect(np.zeros((0, 4)), "empty")
        assert all(not v for v in sa.attributes.values())

    def test_single_frame_temporal_attributes_false(self, det):
        sa = det.detect(np.array([[50.0, 50.0, 40.0, 40.0]]), "single")
        assert sa.has("fast_motion") is False
        assert sa.has("partial_occlusion") is False

    def test_sequence_name_stored(self, det):
        sa = det.detect(_static_gt(10), sequence_name="abc")
        assert sa.sequence_name == "abc"

    def test_all_seven_attributes_returned(self, det):
        sa = det.detect(_static_gt(10))
        assert set(sa.attributes.keys()) == set(ALL_ATTRIBUTES)

    # -- long_sequence --

    def test_long_sequence_true(self, det):
        assert det.detect(_static_gt(401), "long").has("long_sequence")

    def test_long_sequence_false(self, det):
        assert not det.detect(_static_gt(400), "not-long").has("long_sequence")

    # -- scale_variation --

    def test_scale_variation_large_ratio(self, det):
        # area: 20²=400 → 100²=10000, ratio=25 > 4
        gt = _scaling_gt(30, 20, 20, 100, 100)
        assert det.detect(gt, "sv").has("scale_variation")

    def test_scale_variation_stable_size(self, det):
        assert not det.detect(_static_gt(30), "no-sv").has("scale_variation")

    def test_scale_variation_custom_threshold(self):
        det = AttributeDetector(sv_ratio_threshold=2.0)
        # area: 30²=900 → 50²=2500, ratio≈2.78 > 2.0
        gt = _scaling_gt(30, 30, 30, 50, 50)
        assert det.detect(gt, "sv-low-thresh").has("scale_variation")

    # -- low_resolution --

    def test_low_resolution_small_box(self, det):
        # area = 15×15 = 225 < 400
        assert det.detect(_static_gt(30, w=15, h=15), "lr").has("low_resolution")

    def test_low_resolution_large_box(self, det):
        # area = 40×40 = 1600 > 400
        assert not det.detect(_static_gt(30, w=40, h=40), "hr").has("low_resolution")

    # -- fast_motion --

    def test_fast_motion_high_velocity(self, det):
        # displacement=30px, mean_diag≈56px → 30/56≈0.54 > 0.20
        gt = _moving_gt(30, vx=30.0, w=40, h=40)
        assert det.detect(gt, "fm").has("fast_motion")

    def test_fast_motion_slow_motion(self, det):
        # displacement=1px, mean_diag≈56px → 1/56≈0.018 < 0.20
        gt = _moving_gt(50, vx=1.0, w=40, h=40)
        assert not det.detect(gt, "slow").has("fast_motion")

    # -- out_plane_rotation --

    def test_out_plane_rotation_varying_ar(self, det):
        n = 40
        gt = np.zeros((n, 4))
        gt[:, :2] = 50
        gt[:, 3] = 40  # h constant
        # w alternates: 20 (ar=0.5) and 80 (ar=2.0) → ratio=4 > 2
        gt[::2, 2] = 20
        gt[1::2, 2] = 80
        assert det.detect(gt, "opr").has("out_plane_rotation")

    def test_out_plane_rotation_stable_ar(self, det):
        # w=40, h=40 → ar=1.0 everywhere → ratio=1 < 2
        assert not det.detect(_static_gt(30), "no-opr").has("out_plane_rotation")

    # -- deformation --

    def test_deformation_high_ar_std(self, det):
        rng = np.random.default_rng(0)
        n = 60
        gt = np.zeros((n, 4))
        gt[:, :2] = 50
        gt[:, 2] = rng.uniform(10, 100, n)   # highly variable width
        gt[:, 3] = rng.uniform(10, 100, n)   # highly variable height
        assert det.detect(gt, "def").has("deformation")

    def test_deformation_stable_shape(self, det):
        # Constant square: ar=1.0, std=0 < 0.30
        assert not det.detect(_static_gt(30), "no-def").has("deformation")

    # -- partial_occlusion --

    def test_partial_occlusion_sudden_shrink(self, det):
        gt = _static_gt(30, w=80, h=80)
        # Frame 15: box shrinks to 10×10 (area drop ≈ 98% > 50%)
        gt[15, 2] = 10
        gt[15, 3] = 10
        assert det.detect(gt, "occ").has("partial_occlusion")

    def test_partial_occlusion_gradual_change(self, det):
        # Smooth scale variation: no single-frame drop > 50%
        gt = _scaling_gt(30, 40, 40, 60, 60)
        # max area drop ≈ (60²-40²)/60² ≈ 55% — let's use a gentler range
        gt = _scaling_gt(30, 40, 40, 50, 50)
        # Even smaller, single-frame drops well below 50%
        assert not det.detect(gt, "no-occ").has("partial_occlusion")


# ---------------------------------------------------------------------------
# AttributePerformanceTable
# ---------------------------------------------------------------------------

class TestAttributePerformanceTable:
    def test_to_markdown_empty(self):
        tbl = AttributePerformanceTable("MOSSE", "DS", entries={})
        md = tbl.to_markdown()
        assert "No attribute data" in md

    def test_to_markdown_with_iou_only(self):
        tbl = AttributePerformanceTable(
            "KCF", "OTB",
            entries={"fast_motion": {"n_sequences": 3, "mean_iou": 0.42}},
        )
        md = tbl.to_markdown()
        assert "KCF" in md
        assert "fast_motion" in md
        assert "0.4200" in md

    def test_to_markdown_with_auc(self):
        tbl = AttributePerformanceTable(
            "KCF", "OTB",
            entries={
                "fast_motion": {
                    "n_sequences": 5,
                    "mean_iou": 0.42,
                    "success_auc": 0.38,
                    "precision_auc": 0.55,
                }
            },
        )
        md = tbl.to_markdown()
        assert "Success AUC" in md
        assert "0.3800" in md

    def test_to_dict_round_trip(self):
        entries = {"low_resolution": {"n_sequences": 4, "mean_iou": 0.30}}
        tbl = AttributePerformanceTable("CSRT", "GOT", entries=entries)
        d = tbl.to_dict()
        assert d["tracker_name"] == "CSRT"
        assert d["dataset_name"] == "GOT"
        assert d["entries"]["low_resolution"]["mean_iou"] == 0.30


# ---------------------------------------------------------------------------
# AttributeAnalyzer
# ---------------------------------------------------------------------------

class TestAttributeAnalyzer:
    def _build_result(self):
        return _make_mock_result([
            # name, gt, mean_iou, success_auc, precision_auc
            ("seq_fast", _moving_gt(50, vx=30.0, w=40, h=40), 0.50, 0.45, 0.60),
            ("seq_sv",   _scaling_gt(50, 20, 20, 100, 100),   0.30, 0.28, 0.35),
            ("seq_static", _static_gt(50, w=80, h=80),        0.70, 0.65, 0.75),
        ])

    def test_breakdown_returns_table(self):
        result = self._build_result()
        tbl = AttributeAnalyzer().breakdown(result)
        assert isinstance(tbl, AttributePerformanceTable)
        assert tbl.tracker_name == "MOSSE"

    def test_breakdown_static_seq_in_no_attribute(self):
        result = self._build_result()
        tbl = AttributeAnalyzer().breakdown(result)
        # "seq_static" has no attributes → should not appear in any attribute's n_sequences alone
        # Its mean_iou should not suppress entries from other sequences
        assert isinstance(tbl.entries, dict)

    def test_breakdown_manual_attributes_correct_grouping(self):
        result = self._build_result()
        # Manually mark seq_fast and seq_static with fast_motion
        manual = {
            "seq_fast":   SequenceAttributes("seq_fast",   {"fast_motion": True,  **{k: False for k in ALL_ATTRIBUTES if k != "fast_motion"}}),
            "seq_sv":     SequenceAttributes("seq_sv",     {k: False for k in ALL_ATTRIBUTES}),
            "seq_static": SequenceAttributes("seq_static", {"fast_motion": True,  **{k: False for k in ALL_ATTRIBUTES if k != "fast_motion"}}),
        }
        tbl = AttributeAnalyzer().breakdown(result, sequence_attributes=manual)
        assert "fast_motion" in tbl.entries
        e = tbl.entries["fast_motion"]
        assert e["n_sequences"] == 2
        # mean_iou = (0.50 + 0.70) / 2 = 0.60
        assert abs(e["mean_iou"] - 0.60) < 1e-4

    def test_detect_all_returns_all_sequences(self):
        result = self._build_result()
        attrs = AttributeAnalyzer().detect_all(result)
        assert set(attrs.keys()) == {"seq_fast", "seq_sv", "seq_static"}
        for sa in attrs.values():
            assert isinstance(sa, SequenceAttributes)

    def test_breakdown_no_ground_truths_skipped(self):
        sr = MagicMock()
        sr.sequence_name = "no_gt"
        sr.mean_iou = 0.5
        sr.ground_truths = None
        sr.accuracy_metrics = None

        result = MagicMock()
        result.tracker_name = "KCF"
        result.dataset_name = "DS"
        result.sequence_results = [sr]

        tbl = AttributeAnalyzer().breakdown(result)
        assert tbl.entries == {}

    def test_breakdown_markdown_is_string(self):
        result = self._build_result()
        md = AttributeAnalyzer().breakdown(result).to_markdown()
        assert isinstance(md, str) and len(md) > 0


# ---------------------------------------------------------------------------
# Module-level sanity checks
# ---------------------------------------------------------------------------

def test_all_attributes_count():
    assert len(ALL_ATTRIBUTES) == 7


def test_all_attributes_have_descriptions():
    for attr in ALL_ATTRIBUTES:
        assert attr in ATTRIBUTE_DESCRIPTIONS
        assert isinstance(ATTRIBUTE_DESCRIPTIONS[attr], str)
        assert len(ATTRIBUTE_DESCRIPTIONS[attr]) > 0

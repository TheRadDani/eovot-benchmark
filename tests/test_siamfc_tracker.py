"""Unit tests for eovot.trackers.siamfc and eovot.datasets.otb."""

import numpy as np
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# SiamFC tracker tests
# ──────────────────────────────────────────────────────────────────────────────


def _require_torch():
    """Skip the test if PyTorch is not installed."""
    pytest.importorskip("torch", reason="PyTorch not installed — skipping SiamFC tests")


class TestSiamFCTracker:
    """Tests for SiamFCTracker that run only when PyTorch is available."""

    def setup_method(self):
        _require_torch()
        from eovot.trackers.siamfc import SiamFCTracker
        self.TrackerClass = SiamFCTracker

    def _make_frame(self, h=240, w=320):
        rng = np.random.default_rng(0)
        return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)

    def test_instantiation_cpu(self):
        tracker = self.TrackerClass(device="cpu")
        assert tracker.name == "SiamFC"

    def test_initialize_sets_state(self):
        tracker = self.TrackerClass(device="cpu")
        frame = self._make_frame()
        tracker.initialize(frame, (50.0, 50.0, 40.0, 30.0))
        assert tracker._pos is not None
        assert tracker._z_feat is not None
        assert tracker._target_sz == pytest.approx((40.0, 30.0))

    def test_update_returns_bbox_tuple(self):
        tracker = self.TrackerClass(device="cpu")
        frame = self._make_frame()
        tracker.initialize(frame, (50.0, 50.0, 40.0, 30.0))
        bbox = tracker.update(frame)
        assert len(bbox) == 4
        # Width and height should be unchanged (no scale adaptation in this impl).
        assert bbox[2] == pytest.approx(40.0)
        assert bbox[3] == pytest.approx(30.0)

    def test_update_without_initialize_raises(self):
        tracker = self.TrackerClass(device="cpu")
        frame = self._make_frame()
        with pytest.raises(RuntimeError):
            tracker.update(frame)

    def test_reset_clears_state(self):
        tracker = self.TrackerClass(device="cpu")
        frame = self._make_frame()
        tracker.initialize(frame, (50.0, 50.0, 40.0, 30.0))
        tracker.reset()
        assert tracker._z_feat is None
        assert tracker._pos is None

    def test_re_initialize_after_reset(self):
        tracker = self.TrackerClass(device="cpu")
        frame = self._make_frame()
        tracker.initialize(frame, (50.0, 50.0, 40.0, 30.0))
        tracker.reset()
        tracker.initialize(frame, (10.0, 10.0, 20.0, 20.0))
        assert tracker._pos == pytest.approx((20.0, 20.0))

    def test_consecutive_updates(self):
        tracker = self.TrackerClass(device="cpu")
        rng = np.random.default_rng(1)
        frames = [rng.integers(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(5)]
        tracker.initialize(frames[0], (80.0, 60.0, 40.0, 40.0))
        for frame in frames[1:]:
            bbox = tracker.update(frame)
            assert len(bbox) == 4
            # The tracker should not wander infinitely far from the initial position.
            cx = bbox[0] + bbox[2] / 2
            cy = bbox[1] + bbox[3] / 2
            assert -50 < cx < 400
            assert -50 < cy < 310

    def test_bbox_at_image_boundary(self):
        """Target at image edge should not crash (border padding must handle it)."""
        tracker = self.TrackerClass(device="cpu")
        frame = self._make_frame(h=240, w=320)
        # Top-left corner — crop extends outside image bounds.
        tracker.initialize(frame, (0.0, 0.0, 20.0, 20.0))
        bbox = tracker.update(frame)
        assert len(bbox) == 4

    def test_crop_side_computed_from_context(self):
        """_z_crop_side should be > 0 and encode context padding."""
        import math
        tracker = self.TrackerClass(device="cpu", context_amount=0.5)
        frame = self._make_frame()
        w, h = 40.0, 30.0
        tracker.initialize(frame, (50.0, 50.0, w, h))
        ctx = 0.5 * (w + h)
        expected_side = math.sqrt((w + ctx) * (h + ctx))
        assert tracker._z_crop_side == pytest.approx(expected_side, rel=1e-5)

    def test_backbone_output_shape(self):
        """Backbone should produce 6×6×256 for a 127×127 input."""
        import torch
        from eovot.trackers.siamfc import _SiamFCBackbone
        net = _SiamFCBackbone().eval()
        x = torch.zeros(1, 3, 127, 127)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (1, 256, 6, 6)

    def test_backbone_search_output_shape(self):
        """Backbone should produce 22×22×256 for a 255×255 input."""
        import torch
        from eovot.trackers.siamfc import _SiamFCBackbone
        net = _SiamFCBackbone().eval()
        x = torch.zeros(1, 3, 255, 255)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (1, 256, 22, 22)

    def test_cross_correlation_score_shape(self):
        """Cross-correlation of 22×22 and 6×6 feature maps → 17×17 score."""
        import torch
        import torch.nn.functional as F
        from eovot.trackers.siamfc import _SiamFCBackbone
        net = _SiamFCBackbone().eval()
        z = torch.zeros(1, 3, 127, 127)
        x = torch.zeros(1, 3, 255, 255)
        with torch.no_grad():
            z_feat = net(z)
            x_feat = net(x)
            score = F.conv2d(x_feat, z_feat)
        assert score.shape == (1, 1, 17, 17)

    def test_not_available_without_torch(self, monkeypatch):
        """ImportError should be raised when torch is absent."""
        import eovot.trackers.siamfc as _m
        monkeypatch.setattr(_m, "_TORCH_AVAILABLE", False)
        with pytest.raises(ImportError, match="PyTorch"):
            _m.SiamFCTracker()


# ──────────────────────────────────────────────────────────────────────────────
# OTB attribute dataset tests (no filesystem access needed)
# ──────────────────────────────────────────────────────────────────────────────


class TestOTBAttribute:
    """Tests for OTBAttribute enum and OTB100_ATTRIBUTES map."""

    def test_all_11_attributes_defined(self):
        from eovot.datasets.otb import OTBAttribute
        assert len(OTBAttribute) == 11

    def test_attribute_names(self):
        from eovot.datasets.otb import OTBAttribute
        names = {a.name for a in OTBAttribute}
        expected = {"IV", "SV", "OCC", "DEF", "MB", "FM", "IPR", "OPR", "OV", "BC", "LR"}
        assert names == expected

    def test_otb100_map_covers_100_sequences(self):
        from eovot.datasets.otb import OTB100_ATTRIBUTES
        assert len(OTB100_ATTRIBUTES) == 100

    def test_all_attribute_sets_non_empty(self):
        from eovot.datasets.otb import OTB100_ATTRIBUTES
        for name, attrs in OTB100_ATTRIBUTES.items():
            assert len(attrs) > 0, f"Sequence {name!r} has no attributes"

    def test_all_attribute_values_valid(self):
        from eovot.datasets.otb import OTB100_ATTRIBUTES, OTBAttribute
        valid = set(OTBAttribute)
        for name, attrs in OTB100_ATTRIBUTES.items():
            for attr in attrs:
                assert attr in valid, f"{name}: invalid attribute {attr}"

    def test_known_sequence_attributes(self):
        from eovot.datasets.otb import OTB100_ATTRIBUTES, OTBAttribute
        # Basketball: IV, OCC, DEF, OPR, BC  (from the original paper)
        bball = OTB100_ATTRIBUTES["Basketball"]
        assert OTBAttribute.IV in bball
        assert OTBAttribute.OCC in bball
        assert OTBAttribute.DEF in bball
        assert OTBAttribute.SV not in bball

    def test_str_representation(self):
        from eovot.datasets.otb import OTBAttribute
        assert str(OTBAttribute.OCC) == "OCC"
        assert str(OTBAttribute.FM) == "FM"


class TestOTBAttributeDatasetFiltering:
    """Tests for filtering logic — no filesystem access required."""

    def _make_dataset(self, tmpdir, attributes=None):
        """Create a minimal OTBAttributeDataset with synthetic sequences."""
        import os
        # Build two synthetic sequences with the OTB directory layout.
        for seq_name in ("Basketball", "Deer", "Dancer"):
            seq_dir = os.path.join(str(tmpdir), seq_name)
            img_dir = os.path.join(seq_dir, "img")
            os.makedirs(img_dir)
            # Write a single placeholder frame (1×1 white pixel).
            import cv2
            cv2.imwrite(os.path.join(img_dir, "0001.jpg"), np.ones((10, 10, 3), dtype=np.uint8) * 255)
            with open(os.path.join(seq_dir, "groundtruth_rect.txt"), "w") as f:
                f.write("0,0,5,5\n")

        from eovot.datasets.otb import OTBAttributeDataset
        return OTBAttributeDataset(str(tmpdir), attributes=attributes)

    def test_no_filter_returns_all_sequences(self, tmp_path):
        ds = self._make_dataset(tmp_path)
        assert len(ds) == 3

    def test_filter_occ_excludes_deer(self, tmp_path):
        # Basketball has OCC; Deer does NOT have OCC; Dancer does NOT.
        from eovot.datasets.otb import OTBAttribute
        ds = self._make_dataset(tmp_path, attributes=[OTBAttribute.OCC])
        names = [ds._entries[i][0] for i in range(len(ds))]
        assert "Basketball" in names
        assert "Deer" not in names

    def test_filter_mb_includes_deer(self, tmp_path):
        # Deer has MB (Motion Blur) in OTB100_ATTRIBUTES.
        from eovot.datasets.otb import OTBAttribute
        ds = self._make_dataset(tmp_path, attributes=[OTBAttribute.MB])
        names = [ds._entries[i][0] for i in range(len(ds))]
        assert "Deer" in names

    def test_multi_attribute_filter(self, tmp_path):
        # Dancer has {SV, IPR, OPR} — requesting SV+IV should exclude it.
        from eovot.datasets.otb import OTBAttribute
        ds = self._make_dataset(tmp_path, attributes=[OTBAttribute.SV, OTBAttribute.IV])
        names = [ds._entries[i][0] for i in range(len(ds))]
        # Basketball has IV but not SV; Dancer has SV but not IV; Deer has neither.
        assert len(names) == 0

    def test_get_attributes_known_sequence(self, tmp_path):
        ds = self._make_dataset(tmp_path)
        from eovot.datasets.otb import OTBAttribute
        attrs = ds.get_attributes("Basketball")
        assert OTBAttribute.IV in attrs
        assert OTBAttribute.OCC in attrs

    def test_get_attributes_unknown_sequence(self, tmp_path):
        ds = self._make_dataset(tmp_path)
        attrs = ds.get_attributes("NonExistentSeq")
        assert attrs == frozenset()

    def test_attribute_counts_dict_has_all_keys(self, tmp_path):
        ds = self._make_dataset(tmp_path)
        counts = ds.attribute_counts()
        from eovot.datasets.otb import OTBAttribute
        for attr in OTBAttribute:
            assert attr.name in counts

    def test_attribute_summary_returns_string(self, tmp_path):
        ds = self._make_dataset(tmp_path)
        summary = ds.attribute_summary()
        assert isinstance(summary, str)
        assert "OCC" in summary
        assert "IV" in summary

    def test_sequences_with_attribute(self, tmp_path):
        ds = self._make_dataset(tmp_path)
        from eovot.datasets.otb import OTBAttribute
        occ_seqs = ds.sequences_with_attribute(OTBAttribute.OCC)
        assert "Basketball" in occ_seqs
        assert "Deer" not in occ_seqs

"""Tests for the SiamFC-Lite tracker (eovot/trackers/siamfc_lite.py).

All tests are automatically skipped when PyTorch is not installed.
The suite covers: interface conformance, forward-pass geometry,
bbox invariants, error handling, and cross-correlation correctness.
"""

from __future__ import annotations

import numpy as np
import pytest

# Determine whether torch is present before importing the tracker
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

pytestmark = pytest.mark.skipif(
    not _HAS_TORCH,
    reason="PyTorch not installed — SiamFC-Lite tests skipped",
)

from eovot.trackers.siamfc_lite import SiamFCLiteTracker, _SiamFCBackbone  # noqa: E402
from eovot.trackers.base import BaseTracker                                  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame(h: int = 240, w: int = 320, seed: int = 42) -> np.ndarray:
    """Return a deterministic random BGR frame."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


BBOX = (100.0, 80.0, 60.0, 50.0)  # (x, y, w, h) — well inside a 240×320 frame


# ---------------------------------------------------------------------------
# Backbone unit tests
# ---------------------------------------------------------------------------

class TestSiamFCBackbone:
    """Tests for the _SiamFCBackbone CNN module."""

    def setup_method(self):
        self.backbone = _SiamFCBackbone().eval()

    def test_template_output_shape(self):
        """127×127 input → (1, 64, 13, 13) feature map."""
        x = torch.zeros(1, 3, 127, 127)
        out = self.backbone(x)
        assert out.shape == (1, 64, 13, 13), f"Unexpected shape: {out.shape}"

    def test_search_output_shape(self):
        """255×255 input → (1, 64, 29, 29) feature map."""
        x = torch.zeros(1, 3, 255, 255)
        out = self.backbone(x)
        assert out.shape == (1, 64, 29, 29), f"Unexpected shape: {out.shape}"

    def test_output_is_l2_normalised(self):
        """Output should have unit L2 norm along the channel dimension."""
        import torch.nn.functional as F
        x = torch.randn(2, 3, 127, 127)
        out = self.backbone(x)
        norms = out.norm(p=2, dim=1)  # (B, H', W')
        # All norms should be ≈ 1.0 (allow float32 rounding)
        assert norms.min().item() > 0.9, "Some feature norms are too small"
        assert norms.max().item() < 1.1, "Some feature norms are too large"

    def test_eval_mode_is_deterministic(self):
        """Same input → same output in eval mode."""
        x = torch.randn(1, 3, 127, 127)
        out1 = self.backbone(x)
        out2 = self.backbone(x)
        assert torch.allclose(out1, out2), "Backbone is non-deterministic in eval mode"


# ---------------------------------------------------------------------------
# SiamFCLiteTracker interface tests
# ---------------------------------------------------------------------------

class TestSiamFCLiteInterface:
    """Interface conformance: BaseTracker contract."""

    def test_is_base_tracker(self):
        assert isinstance(SiamFCLiteTracker(), BaseTracker)

    def test_name(self):
        assert SiamFCLiteTracker().name == "SiamFC-Lite"

    def test_repr(self):
        r = repr(SiamFCLiteTracker())
        assert "SiamFCLiteTracker" in r


# ---------------------------------------------------------------------------
# Initialisation and update tests
# ---------------------------------------------------------------------------

class TestSiamFCLiteTracker:
    """Functional tests for the full tracking pipeline."""

    def setup_method(self):
        self.tracker = SiamFCLiteTracker(device="cpu")
        self.frame = _frame()

    def test_initialize_does_not_raise(self):
        self.tracker.initialize(self.frame, BBOX)

    def test_update_returns_4_tuple(self):
        self.tracker.initialize(self.frame, BBOX)
        bbox = self.tracker.update(self.frame)
        assert len(bbox) == 4, "update() must return a 4-tuple (x, y, w, h)"

    def test_update_preserves_target_size(self):
        """SiamFC-Lite does not scale the bounding box — w and h must be unchanged."""
        _, _, w_init, h_init = BBOX
        self.tracker.initialize(self.frame, BBOX)
        for _ in range(5):
            x, y, w, h = self.tracker.update(self.frame)
            assert abs(w - w_init) < 1e-6, "Width must not change between updates"
            assert abs(h - h_init) < 1e-6, "Height must not change between updates"

    def test_update_width_height_positive(self):
        self.tracker.initialize(self.frame, BBOX)
        x, y, w, h = self.tracker.update(self.frame)
        assert w > 0, "Predicted width must be positive"
        assert h > 0, "Predicted height must be positive"

    def test_update_without_init_raises(self):
        tracker = SiamFCLiteTracker(device="cpu")
        with pytest.raises(RuntimeError, match="initialised"):
            tracker.update(self.frame)

    def test_multiple_sequential_updates(self):
        """Tracker must remain stable across many consecutive frames."""
        frames = [_frame(seed=i) for i in range(10)]
        self.tracker.initialize(frames[0], BBOX)
        for frame in frames[1:]:
            bbox = self.tracker.update(frame)
            assert len(bbox) == 4

    def test_reinitialize_on_new_sequence(self):
        """Calling initialize() a second time must reset per-sequence state."""
        self.tracker.initialize(self.frame, BBOX)
        self.tracker.update(self.frame)
        # Re-initialize with a different bbox
        new_bbox = (10.0, 10.0, 30.0, 30.0)
        self.tracker.initialize(self.frame, new_bbox)
        x, y, w, h = self.tracker.update(self.frame)
        # Width/height should match the new bbox after re-init
        assert abs(w - 30.0) < 1e-6
        assert abs(h - 30.0) < 1e-6

    def test_template_feature_shape_after_init(self):
        """Template features must have the expected backbone output shape."""
        self.tracker.initialize(self.frame, BBOX)
        feat = self.tracker._template_feat
        assert feat is not None
        assert feat.shape == (1, 64, 13, 13), f"Unexpected template shape: {feat.shape}"

    def test_small_frame_does_not_raise(self):
        """Out-of-bounds patches must be padded without error."""
        small_frame = _frame(h=60, w=80)
        bbox_oob = (5.0, 5.0, 50.0, 40.0)   # bbox nearly fills the small frame
        self.tracker.initialize(small_frame, bbox_oob)
        bbox = self.tracker.update(small_frame)
        assert len(bbox) == 4

    def test_target_at_frame_edge(self):
        """Target placed at the top-left corner must be handled gracefully."""
        self.tracker.initialize(self.frame, (0.0, 0.0, 40.0, 40.0))
        bbox = self.tracker.update(self.frame)
        assert len(bbox) == 4


# ---------------------------------------------------------------------------
# Cross-correlation tests
# ---------------------------------------------------------------------------

class TestCrossCorrelate:
    """Unit tests for the static _cross_correlate helper."""

    def test_output_shape(self):
        """Response map shape = (1, 1, hx-hz+1, wx-wz+1).

        Backbone output sizes: template 127×127 → 13×13, search 255×255 → 29×29.
        Cross-correlation response: (29 - 13 + 1) = 17 → 17×17.
        """
        z = torch.zeros(1, 64, 13, 13)
        x = torch.zeros(1, 64, 29, 29)
        resp = SiamFCLiteTracker._cross_correlate(z, x)
        assert resp.shape == (1, 1, 17, 17), f"Unexpected response shape: {resp.shape}"

    def test_perfect_match_produces_high_score(self):
        """When z_feat == centre patch of x_feat, the response peak must be central."""
        # Template: 13×13.  Search: 29×29.  Offset = (29-13) // 2 = 8.
        z = torch.randn(1, 64, 13, 13)
        x = torch.zeros(1, 64, 29, 29)
        x[:, :, 8:21, 8:21] = z
        resp = SiamFCLiteTracker._cross_correlate(z, x)
        resp_np = resp.squeeze().numpy()
        peak_y, peak_x = np.unravel_index(resp_np.argmax(), resp_np.shape)
        # Peak should be at or near (8, 8) — the centre of the 17×17 response map
        assert abs(peak_y - 8) <= 1, f"Peak y={peak_y} not near centre"
        assert abs(peak_x - 8) <= 1, f"Peak x={peak_x} not near centre"


# ---------------------------------------------------------------------------
# ImportError guard test
# ---------------------------------------------------------------------------

class TestImportGuard:
    """Verify that a clear ImportError is raised without PyTorch."""

    def test_no_torch_raises_import_error(self, monkeypatch):
        import eovot.trackers.siamfc_lite as mod
        original = mod._HAS_TORCH
        monkeypatch.setattr(mod, "_HAS_TORCH", False)
        with pytest.raises(ImportError, match="PyTorch"):
            SiamFCLiteTracker(device="cpu")
        monkeypatch.setattr(mod, "_HAS_TORCH", original)

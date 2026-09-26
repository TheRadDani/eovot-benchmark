"""Tests for eovot.profiling.model_complexity.

Covers parameter counting, FLOPs estimation, edge-tier classification,
file profiling, and comparison table generation.  PyTorch-dependent tests
are skipped automatically when torch is not installed.
"""

from __future__ import annotations

import pytest

# All tests that require torch are grouped under this marker.
torch_available = pytest.mark.skipif(
    False, reason="torch available"  # re-evaluated per test
)

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

skip_no_torch = pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed")

from eovot.profiling.model_complexity import (
    ModelComplexityAnalyzer,
    ModelComplexityResult,
    _classify_edge_tier,
)


# ---------------------------------------------------------------------------
# Tier classification (no torch needed)
# ---------------------------------------------------------------------------

class TestClassifyEdgeTier:
    def test_ultralight(self):
        assert _classify_edge_tier(100_000) == "Ultralight"

    def test_light_lower_bound(self):
        assert _classify_edge_tier(500_000) == "Light"

    def test_light_upper_bound(self):
        assert _classify_edge_tier(4_999_999) == "Light"

    def test_medium(self):
        assert _classify_edge_tier(10_000_000) == "Medium"

    def test_heavy(self):
        assert _classify_edge_tier(50_000_000) == "Heavy"

    def test_boundary_medium_heavy(self):
        assert _classify_edge_tier(25_000_000) == "Heavy"


# ---------------------------------------------------------------------------
# ModelComplexityResult helpers (no torch needed)
# ---------------------------------------------------------------------------

class TestModelComplexityResult:
    def _make_result(self, total_params: int = 2_500_000, mflops: float = 100.0) -> ModelComplexityResult:
        return ModelComplexityResult(
            model_name="TestModel",
            total_params=total_params,
            trainable_params=total_params,
            frozen_params=0,
            estimated_mflops=mflops,
            model_size_mb=10.0,
            input_shape=(1, 3, 128, 128),
            edge_tier=_classify_edge_tier(total_params),
        )

    def test_total_params_m(self):
        r = self._make_result(total_params=2_500_000)
        assert r.total_params_m == pytest.approx(2.5)

    def test_trainable_params_m(self):
        r = self._make_result(total_params=1_000_000)
        assert r.trainable_params_m == pytest.approx(1.0)

    def test_str_contains_model_name(self):
        r = self._make_result()
        assert "TestModel" in str(r)

    def test_str_contains_tier(self):
        r = self._make_result(total_params=10_000_000)
        assert "Medium" in str(r)

    def test_to_dict_keys(self):
        r = self._make_result()
        d = r.to_dict()
        required = {"model_name", "total_params", "trainable_params", "frozen_params",
                    "estimated_mflops", "model_size_mb", "input_shape", "edge_tier"}
        assert required.issubset(d.keys())

    def test_to_dict_none_mflops(self):
        r = ModelComplexityResult(
            model_name="X", total_params=0, trainable_params=0, frozen_params=0,
            estimated_mflops=None, model_size_mb=None, input_shape=None, edge_tier="Ultralight",
        )
        d = r.to_dict()
        assert d["estimated_mflops"] is None
        assert d["model_size_mb"] is None
        assert d["input_shape"] is None


# ---------------------------------------------------------------------------
# Profile module (torch required)
# ---------------------------------------------------------------------------

@skip_no_torch
class TestProfileModule:
    def _simple_cnn(self) -> "nn.Module":
        return nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 4),
        )

    def test_param_count_correct(self):
        model = self._simple_cnn()
        expected = sum(p.numel() for p in model.parameters())
        analyzer = ModelComplexityAnalyzer()
        result = analyzer.profile_module(model, input_shape=(1, 3, 32, 32))
        assert result.total_params == expected

    def test_all_trainable_by_default(self):
        model = self._simple_cnn()
        analyzer = ModelComplexityAnalyzer()
        result = analyzer.profile_module(model, input_shape=(1, 3, 32, 32))
        assert result.frozen_params == 0
        assert result.trainable_params == result.total_params

    def test_frozen_params_counted(self):
        model = self._simple_cnn()
        for p in model[0].parameters():
            p.requires_grad_(False)
        frozen_expected = sum(p.numel() for p in model[0].parameters())
        analyzer = ModelComplexityAnalyzer()
        result = analyzer.profile_module(model, input_shape=(1, 3, 32, 32))
        assert result.frozen_params == frozen_expected

    def test_mflops_positive(self):
        model = self._simple_cnn()
        analyzer = ModelComplexityAnalyzer()
        result = analyzer.profile_module(model, input_shape=(1, 3, 32, 32))
        assert result.estimated_mflops is not None
        assert result.estimated_mflops > 0

    def test_larger_input_gives_more_flops(self):
        model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1))
        analyzer = ModelComplexityAnalyzer()
        r_small = analyzer.profile_module(model, input_shape=(1, 3, 16, 16))
        r_large = analyzer.profile_module(model, input_shape=(1, 3, 64, 64))
        assert r_large.estimated_mflops > r_small.estimated_mflops

    def test_layer_flops_populated(self):
        model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1))
        analyzer = ModelComplexityAnalyzer()
        result = analyzer.profile_module(model, input_shape=(1, 3, 16, 16))
        assert len(result.layer_flops) > 0

    def test_linear_layer_flops(self):
        model = nn.Linear(128, 64)
        analyzer = ModelComplexityAnalyzer()
        result = analyzer.profile_module(model, input_shape=(1, 128))
        # Expected FLOPs = 2 × 128 × 64 = 16384
        assert result.estimated_mflops == pytest.approx(16384 / 1e6, rel=1e-3)

    def test_model_name_default(self):
        model = nn.Linear(8, 4)
        analyzer = ModelComplexityAnalyzer()
        result = analyzer.profile_module(model, input_shape=(1, 8))
        assert result.model_name == "Linear"

    def test_model_name_custom(self):
        model = nn.Linear(8, 4)
        analyzer = ModelComplexityAnalyzer()
        result = analyzer.profile_module(model, input_shape=(1, 8), model_name="MyTracker")
        assert result.model_name == "MyTracker"

    def test_edge_tier_ultralight_small_model(self):
        model = nn.Linear(4, 2)  # 10 params
        analyzer = ModelComplexityAnalyzer()
        result = analyzer.profile_module(model, input_shape=(1, 4))
        assert result.edge_tier == "Ultralight"

    def test_input_shape_stored(self):
        model = nn.Linear(8, 4)
        analyzer = ModelComplexityAnalyzer()
        result = analyzer.profile_module(model, input_shape=(1, 8))
        assert result.input_shape == (1, 8)

    def test_model_size_mb_none_for_module(self):
        model = nn.Linear(8, 4)
        analyzer = ModelComplexityAnalyzer()
        result = analyzer.profile_module(model, input_shape=(1, 8))
        assert result.model_size_mb is None

    def test_conv_transpose_counted(self):
        model = nn.ConvTranspose2d(16, 8, kernel_size=3, padding=1)
        analyzer = ModelComplexityAnalyzer()
        result = analyzer.profile_module(model, input_shape=(1, 16, 8, 8))
        assert result.estimated_mflops is not None and result.estimated_mflops > 0


# ---------------------------------------------------------------------------
# Profile file (torch required, uses tmp file)
# ---------------------------------------------------------------------------

@skip_no_torch
class TestProfileFile:
    def test_file_not_found(self, tmp_path):
        analyzer = ModelComplexityAnalyzer()
        with pytest.raises(FileNotFoundError):
            analyzer.profile_file(str(tmp_path / "nonexistent.pt"))

    def test_size_measured(self, tmp_path):
        model = nn.Linear(32, 16)
        path = tmp_path / "model.pt"
        torch.save(model, str(path))
        analyzer = ModelComplexityAnalyzer()
        result = analyzer.profile_file(str(path))
        assert result.model_size_mb is not None
        assert result.model_size_mb > 0

    def test_params_from_full_module(self, tmp_path):
        model = nn.Linear(32, 16)
        expected = sum(p.numel() for p in model.parameters())
        path = tmp_path / "model.pt"
        torch.save(model, str(path))
        analyzer = ModelComplexityAnalyzer()
        result = analyzer.profile_file(str(path))
        assert result.total_params == expected

    def test_flops_with_input_shape(self, tmp_path):
        model = nn.Linear(32, 16)
        path = tmp_path / "model.pt"
        torch.save(model, str(path))
        analyzer = ModelComplexityAnalyzer()
        result = analyzer.profile_file(str(path), input_shape=(1, 32))
        assert result.estimated_mflops is not None and result.estimated_mflops > 0

    def test_state_dict_file(self, tmp_path):
        model = nn.Linear(32, 16)
        expected = sum(p.numel() for p in model.parameters())
        path = tmp_path / "state.pt"
        torch.save(model.state_dict(), str(path))
        analyzer = ModelComplexityAnalyzer()
        result = analyzer.profile_file(str(path))
        assert result.total_params == expected

    def test_model_name_from_stem(self, tmp_path):
        model = nn.Linear(8, 4)
        path = tmp_path / "my_tracker.pt"
        torch.save(model, str(path))
        analyzer = ModelComplexityAnalyzer()
        result = analyzer.profile_file(str(path))
        assert result.model_name == "my_tracker"


# ---------------------------------------------------------------------------
# Compare table
# ---------------------------------------------------------------------------

class TestCompareTable:
    def _make_results(self) -> list:
        return [
            ModelComplexityResult(
                model_name="A", total_params=100_000, trainable_params=100_000,
                frozen_params=0, estimated_mflops=50.0, model_size_mb=1.0,
                input_shape=(1, 3, 64, 64), edge_tier="Ultralight",
            ),
            ModelComplexityResult(
                model_name="B", total_params=10_000_000, trainable_params=8_000_000,
                frozen_params=2_000_000, estimated_mflops=500.0, model_size_mb=40.0,
                input_shape=(1, 3, 64, 64), edge_tier="Medium",
            ),
        ]

    def test_markdown_contains_headers(self):
        analyzer = ModelComplexityAnalyzer()
        table = analyzer.compare(self._make_results())
        assert "Model" in table
        assert "Params" in table
        assert "MFLOPs" in table
        assert "Tier" in table

    def test_markdown_contains_model_names(self):
        analyzer = ModelComplexityAnalyzer()
        table = analyzer.compare(self._make_results())
        assert "| A |" in table
        assert "| B |" in table

    def test_empty_list(self):
        analyzer = ModelComplexityAnalyzer()
        table = analyzer.compare([])
        assert "Model" in table  # header still present


# ---------------------------------------------------------------------------
# No-torch error path
# ---------------------------------------------------------------------------

class TestNoTorchError:
    def test_import_error_without_torch(self, monkeypatch):
        import eovot.profiling.model_complexity as mc
        monkeypatch.setattr(mc, "_TORCH_AVAILABLE", False)
        analyzer = ModelComplexityAnalyzer()
        # profile_file with a non-existent path raises FileNotFoundError first
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"dummy")
            path = f.name
        try:
            result = analyzer.profile_file(path)
            # When torch unavailable, params should be 0 and mflops None
            assert result.total_params == 0
            assert result.estimated_mflops is None
        finally:
            os.unlink(path)

"""Tests for eovot.optimization — quantization, pruning, and ONNX export.

All tests are designed to run without optional backends (torch, onnxruntime)
by verifying graceful ImportError handling and validating the pure-Python /
pure-NumPy logic that does not depend on those libraries.
"""

from __future__ import annotations

import math
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _ort_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


# ===========================================================================
# QuantizationConfig
# ===========================================================================

class TestQuantizationConfig:
    def test_default_values(self):
        from eovot.optimization.quantization import QuantizationConfig
        cfg = QuantizationConfig()
        assert cfg.mode == "int8"
        assert cfg.per_channel is True
        assert cfg.activation_type == "asymmetric"
        assert cfg.calibration_frames == 100

    def test_fp16_mode(self):
        from eovot.optimization.quantization import QuantizationConfig
        cfg = QuantizationConfig(mode="fp16")
        assert cfg.mode == "fp16"

    def test_invalid_mode_raises(self):
        from eovot.optimization.quantization import QuantizationConfig
        with pytest.raises(ValueError, match="mode"):
            QuantizationConfig(mode="bf16")

    def test_invalid_activation_type_raises(self):
        from eovot.optimization.quantization import QuantizationConfig
        with pytest.raises(ValueError, match="activation_type"):
            QuantizationConfig(activation_type="unknown")


# ===========================================================================
# ModelQuantizer — dependency checks
# ===========================================================================

class TestModelQuantizer:
    def test_instantiation_without_ort(self):
        from eovot.optimization.quantization import ModelQuantizer
        q = ModelQuantizer()
        # Should instantiate cleanly regardless of whether ORT is installed
        assert hasattr(q, "_ort_available")
        assert hasattr(q, "_quant_available")

    def test_quantize_dynamic_missing_file(self, tmp_path):
        from eovot.optimization.quantization import ModelQuantizer
        q = ModelQuantizer()
        if not q._quant_available:
            pytest.skip("onnxruntime.quantization not installed")
        with pytest.raises(FileNotFoundError):
            q.quantize_dynamic(str(tmp_path / "nonexistent.onnx"), str(tmp_path / "out.onnx"))

    def test_benchmark_inference_missing_file(self, tmp_path):
        from eovot.optimization.quantization import ModelQuantizer
        q = ModelQuantizer()
        if not q._ort_available:
            pytest.skip("onnxruntime not installed")
        with pytest.raises(FileNotFoundError):
            q.benchmark_inference(str(tmp_path / "nonexistent.onnx"), (1, 3, 128, 128))

    def test_quantize_dynamic_raises_without_ort(self, tmp_path, monkeypatch):
        from eovot.optimization import quantization as qmod
        monkeypatch.setattr(qmod, "_ModelQuantizer__probe_quant_tools", lambda: False, raising=False)
        from eovot.optimization.quantization import ModelQuantizer
        q = ModelQuantizer()
        q._quant_available = False
        with pytest.raises(ImportError, match="onnxruntime"):
            q.quantize_dynamic("dummy.onnx", "out.onnx")

    def test_benchmark_raises_without_ort(self):
        from eovot.optimization.quantization import ModelQuantizer
        q = ModelQuantizer()
        q._ort_available = False
        with pytest.raises(ImportError, match="onnxruntime"):
            q.benchmark_inference("dummy.onnx", (1, 3, 128, 128))


# ===========================================================================
# InferenceStats
# ===========================================================================

class TestInferenceStats:
    def test_to_dict_keys(self):
        from eovot.optimization.quantization import InferenceStats
        stats = InferenceStats(
            model_path="a.onnx",
            mean_ms=5.0,
            std_ms=0.3,
            p95_ms=6.5,
            fps=200.0,
            n_runs=100,
            input_shape=(1, 3, 128, 128),
        )
        d = stats.to_dict()
        assert "mean_latency_ms" in d
        assert "fps" in d
        assert d["fps"] == 200.0


# ===========================================================================
# QuantizationBenchmark
# ===========================================================================

class TestQuantizationBenchmark:
    def _make_stats(self, mean_ms, p95_ms, fps):
        from eovot.optimization.quantization import InferenceStats
        return InferenceStats(
            model_path="x.onnx",
            mean_ms=mean_ms,
            std_ms=0.1,
            p95_ms=p95_ms,
            fps=fps,
            n_runs=50,
            input_shape=(1, 3, 64, 64),
        )

    def test_compare_speedup(self):
        from eovot.optimization.quantization import QuantizationBenchmark
        orig = self._make_stats(mean_ms=10.0, p95_ms=12.0, fps=100.0)
        quant = self._make_stats(mean_ms=4.0, p95_ms=5.0, fps=250.0)
        result = QuantizationBenchmark.compare(orig, quant)
        assert result["latency_speedup"] == pytest.approx(2.5, rel=1e-3)
        assert result["fps_gain_pct"] == pytest.approx(150.0, rel=1e-3)

    def test_compare_no_speedup(self):
        from eovot.optimization.quantization import QuantizationBenchmark
        stats = self._make_stats(mean_ms=8.0, p95_ms=10.0, fps=125.0)
        result = QuantizationBenchmark.compare(stats, stats)
        assert result["latency_speedup"] == pytest.approx(1.0, rel=1e-3)
        assert result["fps_gain_pct"] == pytest.approx(0.0, abs=1e-3)

    def test_format_report_contains_table(self):
        from eovot.optimization.quantization import QuantizationBenchmark
        orig = self._make_stats(10.0, 12.0, 100.0)
        quant = self._make_stats(4.0, 5.0, 250.0)
        report = QuantizationBenchmark.format_report(orig, quant)
        assert "## Quantization Report" in report
        assert "Mean latency" in report
        assert "FPS" in report

    def test_format_report_with_meta(self):
        from eovot.optimization.quantization import QuantizationBenchmark
        orig = self._make_stats(10.0, 12.0, 100.0)
        quant = self._make_stats(4.0, 5.0, 250.0)
        meta = {"original_size_mb": 4.2, "quantized_size_mb": 1.1, "compression_ratio": 3.82}
        report = QuantizationBenchmark.format_report(orig, quant, quantization_meta=meta)
        assert "compression" in report.lower()


# ===========================================================================
# PruningConfig
# ===========================================================================

class TestPruningConfig:
    def test_default_values(self):
        from eovot.optimization.pruning import PruningConfig
        cfg = PruningConfig()
        assert cfg.sparsity == 0.3
        assert cfg.min_channels == 8
        assert cfg.scope == "all"

    def test_invalid_sparsity_too_high(self):
        from eovot.optimization.pruning import PruningConfig
        with pytest.raises(ValueError, match="sparsity"):
            PruningConfig(sparsity=1.0)

    def test_invalid_sparsity_negative(self):
        from eovot.optimization.pruning import PruningConfig
        with pytest.raises(ValueError, match="sparsity"):
            PruningConfig(sparsity=-0.1)

    def test_invalid_min_channels(self):
        from eovot.optimization.pruning import PruningConfig
        with pytest.raises(ValueError, match="min_channels"):
            PruningConfig(min_channels=0)

    def test_invalid_scope(self):
        from eovot.optimization.pruning import PruningConfig
        with pytest.raises(ValueError, match="scope"):
            PruningConfig(scope="depthwise_only")


# ===========================================================================
# PruningStats
# ===========================================================================

class TestPruningStats:
    def test_str_representation(self):
        from eovot.optimization.pruning import PruningStats
        stats = PruningStats(
            layers_pruned=3,
            channels_removed=48,
            total_params_before=100_000,
            total_params_after=71_600,
            size_reduction_pct=28.4,
        )
        s = str(stats)
        assert "layers_pruned=3" in s
        assert "channels_removed=48" in s

    def test_to_dict_keys(self):
        from eovot.optimization.pruning import PruningStats
        stats = PruningStats(
            layers_pruned=2,
            channels_removed=16,
            total_params_before=50_000,
            total_params_after=40_000,
            size_reduction_pct=20.0,
        )
        d = stats.to_dict()
        assert "layers_pruned" in d
        assert "size_reduction_pct" in d
        assert isinstance(d["layers"], list)


# ===========================================================================
# PruningAnalyzer — requires torch
# ===========================================================================

@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
class TestPruningAnalyzer:
    def _small_cnn(self):
        import torch.nn as nn
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def test_analyse_returns_stats(self):
        from eovot.optimization.pruning import PruningAnalyzer, PruningConfig
        model = self._small_cnn()
        analyser = PruningAnalyzer(PruningConfig(sparsity=0.3))
        stats = analyser.analyse(model)
        assert stats.layers_pruned >= 1
        assert stats.total_params_before > 0
        assert 0.0 <= stats.size_reduction_pct <= 100.0

    def test_analyse_respects_min_channels(self):
        from eovot.optimization.pruning import PruningAnalyzer, PruningConfig
        model = self._small_cnn()
        # With high sparsity, min_channels=8 should be the floor
        analyser = PruningAnalyzer(PruningConfig(sparsity=0.9, min_channels=8))
        stats = analyser.analyse(model)
        for detail in stats.layer_details:
            assert detail.kept_channels >= 8

    def test_no_op_on_linear_model(self):
        import torch.nn as nn
        from eovot.optimization.pruning import PruningAnalyzer, PruningConfig
        # A model with no Conv2d layers should have 0 prunable layers
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        analyser = PruningAnalyzer(PruningConfig())
        stats = analyser.analyse(model)
        assert stats.layers_pruned == 0
        assert stats.channels_removed == 0
        assert stats.size_reduction_pct == 0.0


# ===========================================================================
# MagnitudePruner — requires torch
# ===========================================================================

@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
class TestMagnitudePruner:
    def _small_cnn(self):
        import torch.nn as nn
        return nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
        )

    def test_prune_returns_stats(self):
        from eovot.optimization.pruning import MagnitudePruner, PruningConfig
        model = self._small_cnn()
        pruner = MagnitudePruner(PruningConfig(sparsity=0.25, min_channels=4))
        stats = pruner.prune(model)
        assert stats.layers_pruned >= 1

    def test_weight_mask_registered(self):
        import torch.nn as nn
        from eovot.optimization.pruning import MagnitudePruner, PruningConfig
        model = self._small_cnn()
        MagnitudePruner(PruningConfig(sparsity=0.25, min_channels=4)).prune(model)
        conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        for conv in conv_layers:
            assert hasattr(conv, "weight_mask"), "weight_mask buffer missing after pruning"

    def test_actual_weight_sparsity_nonzero(self):
        from eovot.optimization.pruning import MagnitudePruner, PruningConfig
        model = self._small_cnn()
        pruner = MagnitudePruner(PruningConfig(sparsity=0.25, min_channels=4))
        pruner.prune(model)
        sparsity_map = pruner.get_sparsity(model)
        assert any(v > 0.0 for v in sparsity_map.values()), (
            "Expected some zero weights after pruning"
        )

    def test_get_sparsity_keys_match_layers(self):
        import torch.nn as nn
        from eovot.optimization.pruning import MagnitudePruner, PruningConfig
        model = self._small_cnn()
        pruner = MagnitudePruner(PruningConfig(sparsity=0.0))
        pruner.prune(model)
        sparsity = pruner.get_sparsity(model)
        conv_names = [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
        for name in conv_names:
            assert name in sparsity


# ===========================================================================
# ONNXExportConfig
# ===========================================================================

class TestONNXExportConfig:
    def test_default_values(self):
        from eovot.optimization.onnx_export import ONNXExportConfig
        cfg = ONNXExportConfig()
        assert cfg.input_shape == (1, 3, 128, 128)
        assert cfg.opset == 17
        assert cfg.validate is True
        assert cfg.simplify is False

    def test_invalid_opset_raises(self):
        from eovot.optimization.onnx_export import ONNXExportConfig
        with pytest.raises(ValueError, match="opset"):
            ONNXExportConfig(opset=9)

    def test_invalid_input_shape_raises(self):
        from eovot.optimization.onnx_export import ONNXExportConfig
        with pytest.raises(ValueError, match="input_shape"):
            ONNXExportConfig(input_shape=(128,))


# ===========================================================================
# ONNXExporter — dependency checks + torch-required export tests
# ===========================================================================

class TestONNXExporterDependencyChecks:
    def test_instantiation_without_backends(self):
        from eovot.optimization.onnx_export import ONNXExporter
        exp = ONNXExporter()
        assert hasattr(exp, "_torch_available")
        assert hasattr(exp, "_onnx_available")
        assert hasattr(exp, "_ort_available")

    def test_export_raises_without_torch(self, tmp_path):
        from eovot.optimization.onnx_export import ONNXExporter
        exp = ONNXExporter()
        exp._torch_available = False
        with pytest.raises(ImportError, match="torch"):
            exp.export(object(), str(tmp_path / "out.onnx"))


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
class TestONNXExporterWithTorch:
    def _simple_model(self):
        import torch.nn as nn
        model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
        model.eval()
        return model

    def test_export_creates_file(self, tmp_path):
        from eovot.optimization.onnx_export import ONNXExporter, ONNXExportConfig
        cfg = ONNXExportConfig(input_shape=(1, 3, 16, 16), opset=12, validate=False)
        exp = ONNXExporter(cfg)
        report = exp.export(self._simple_model(), str(tmp_path / "model.onnx"))
        assert (tmp_path / "model.onnx").exists()
        assert report.file_size_mb > 0

    def test_export_report_fields(self, tmp_path):
        from eovot.optimization.onnx_export import ONNXExporter, ONNXExportConfig
        cfg = ONNXExportConfig(input_shape=(1, 3, 16, 16), opset=12, validate=False)
        exp = ONNXExporter(cfg)
        report = exp.export(self._simple_model(), str(tmp_path / "model.onnx"))
        d = report.to_dict()
        for key in ("output_path", "export_time_s", "file_size_mb", "num_nodes"):
            assert key in d, f"Missing key: {key}"
        assert d["export_time_s"] > 0

    def test_export_str_representation(self, tmp_path):
        from eovot.optimization.onnx_export import ONNXExporter, ONNXExportConfig
        cfg = ONNXExportConfig(input_shape=(1, 3, 16, 16), opset=12, validate=False)
        report = ONNXExporter(cfg).export(self._simple_model(), str(tmp_path / "m.onnx"))
        s = str(report)
        assert "ExportReport" in s
        assert "opset=" in s

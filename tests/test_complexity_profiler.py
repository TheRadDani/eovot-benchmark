"""Tests for eovot.profiling.complexity."""

from __future__ import annotations

import pytest

from eovot.profiling.complexity import ComplexityAnalyzer, ComplexityProfile


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def analyzer() -> ComplexityAnalyzer:
    return ComplexityAnalyzer()


@pytest.fixture()
def mosse_profile(analyzer: ComplexityAnalyzer) -> ComplexityProfile:
    return analyzer.estimate_classical(
        name="MOSSE", frame_h=240, frame_w=320,
        filter_size=64, algorithm="mosse",
    )


# ---------------------------------------------------------------------------
# ComplexityProfile properties
# ---------------------------------------------------------------------------

class TestComplexityProfile:
    def test_flops_equals_2x_macs(self, mosse_profile: ComplexityProfile) -> None:
        assert mosse_profile.flops == 2 * mosse_profile.macs

    def test_gflops_consistent(self, mosse_profile: ComplexityProfile) -> None:
        assert abs(mosse_profile.gflops - mosse_profile.flops / 1e9) < 1e-12

    def test_mflops_consistent(self, mosse_profile: ComplexityProfile) -> None:
        assert abs(mosse_profile.mflops - mosse_profile.flops / 1e6) < 1e-9

    def test_param_millions_zero_for_classical(self, mosse_profile: ComplexityProfile) -> None:
        assert mosse_profile.param_millions == 0.0

    def test_is_classical_flag(self, mosse_profile: ComplexityProfile) -> None:
        assert mosse_profile.is_classical is True

    def test_str_contains_name(self, mosse_profile: ComplexityProfile) -> None:
        assert "MOSSE" in str(mosse_profile)

    def test_str_contains_source(self, mosse_profile: ComplexityProfile) -> None:
        assert "classical" in str(mosse_profile)

    def test_str_contains_flops(self, mosse_profile: ComplexityProfile) -> None:
        assert "FLOPs" in str(mosse_profile)


# ---------------------------------------------------------------------------
# Classical tracker estimates
# ---------------------------------------------------------------------------

class TestClassicalEstimates:
    @pytest.mark.parametrize("algorithm", [
        "mosse", "kcf", "csrt", "camshift", "medianflow", "mil",
    ])
    def test_positive_flops_all_algorithms(
        self, analyzer: ComplexityAnalyzer, algorithm: str
    ) -> None:
        profile = analyzer.estimate_classical(
            name=algorithm.upper(), frame_h=240, frame_w=320,
            filter_size=64, algorithm=algorithm,
        )
        assert profile.flops > 0, f"Expected positive FLOPs for {algorithm}"
        assert profile.macs > 0, f"Expected positive MACs for {algorithm}"

    @pytest.mark.parametrize("algorithm", [
        "mosse", "kcf", "csrt", "camshift", "medianflow", "mil",
    ])
    def test_positive_memory_all_algorithms(
        self, analyzer: ComplexityAnalyzer, algorithm: str
    ) -> None:
        profile = analyzer.estimate_classical(
            name=algorithm.upper(), frame_h=240, frame_w=320,
            filter_size=64, algorithm=algorithm,
        )
        assert profile.memory_read_bytes > 0

    def test_unknown_algorithm_raises(self, analyzer: ComplexityAnalyzer) -> None:
        with pytest.raises(ValueError, match="Unknown algorithm"):
            analyzer.estimate_classical(
                name="X", frame_h=240, frame_w=320,
                filter_size=64, algorithm="nonexistent_tracker",
            )

    def test_kcf_more_flops_than_mosse(
        self, analyzer: ComplexityAnalyzer
    ) -> None:
        mosse = analyzer.estimate_classical(
            name="MOSSE", frame_h=240, frame_w=320, filter_size=64, algorithm="mosse"
        )
        kcf = analyzer.estimate_classical(
            name="KCF", frame_h=240, frame_w=320, filter_size=64, algorithm="kcf"
        )
        assert kcf.flops > mosse.flops, "KCF should have more FLOPs than MOSSE"

    def test_csrt_most_flops_among_filters(
        self, analyzer: ComplexityAnalyzer
    ) -> None:
        mosse = analyzer.estimate_classical(
            name="MOSSE", frame_h=240, frame_w=320, filter_size=64, algorithm="mosse"
        )
        csrt = analyzer.estimate_classical(
            name="CSRT", frame_h=240, frame_w=320, filter_size=64, algorithm="csrt"
        )
        assert csrt.flops > mosse.flops, "CSRT should be costlier than MOSSE"

    def test_filter_size_scaling_mosse(
        self, analyzer: ComplexityAnalyzer
    ) -> None:
        small = analyzer.estimate_classical(
            name="MOSSE-S", frame_h=240, frame_w=320, filter_size=32, algorithm="mosse"
        )
        large = analyzer.estimate_classical(
            name="MOSSE-L", frame_h=240, frame_w=320, filter_size=128, algorithm="mosse"
        )
        assert large.flops > small.flops

    def test_case_insensitive_algorithm(
        self, analyzer: ComplexityAnalyzer
    ) -> None:
        p1 = analyzer.estimate_classical(
            name="A", frame_h=240, frame_w=320, filter_size=64, algorithm="MOSSE"
        )
        p2 = analyzer.estimate_classical(
            name="B", frame_h=240, frame_w=320, filter_size=64, algorithm="mosse"
        )
        assert p1.flops == p2.flops

    def test_camshift_depends_on_frame_size(
        self, analyzer: ComplexityAnalyzer
    ) -> None:
        small_frame = analyzer.estimate_classical(
            name="CS-S", frame_h=120, frame_w=160, filter_size=64, algorithm="camshift"
        )
        large_frame = analyzer.estimate_classical(
            name="CS-L", frame_h=480, frame_w=640, filter_size=64, algorithm="camshift"
        )
        assert large_frame.flops > small_frame.flops

    def test_classical_params_always_zero(
        self, analyzer: ComplexityAnalyzer
    ) -> None:
        for algo in ["mosse", "kcf", "csrt", "camshift", "medianflow", "mil"]:
            p = analyzer.estimate_classical(
                name=algo, frame_h=240, frame_w=320, filter_size=64, algorithm=algo
            )
            assert p.params == 0, f"{algo}: expected params=0, got {p.params}"


# ---------------------------------------------------------------------------
# Fleet estimation and Markdown table
# ---------------------------------------------------------------------------

class TestFleetAndTable:
    def test_fleet_length_matches_input(
        self, analyzer: ComplexityAnalyzer
    ) -> None:
        configs = [
            ("MOSSE", "mosse", 240, 320, 64),
            ("KCF",   "kcf",   240, 320, 64),
            ("CSRT",  "csrt",  240, 320, 64),
        ]
        profiles = analyzer.estimate_tracker_fleet(configs)
        assert len(profiles) == 3

    def test_fleet_order_preserved(
        self, analyzer: ComplexityAnalyzer
    ) -> None:
        configs = [
            ("MOSSE", "mosse", 240, 320, 64),
            ("KCF",   "kcf",   240, 320, 64),
        ]
        profiles = analyzer.estimate_tracker_fleet(configs)
        assert profiles[0].name == "MOSSE"
        assert profiles[1].name == "KCF"

    def test_fleet_all_positive_flops(
        self, analyzer: ComplexityAnalyzer
    ) -> None:
        configs = [
            ("MOSSE",      "mosse",     240, 320, 64),
            ("KCF",        "kcf",       240, 320, 64),
            ("CSRT",       "csrt",      240, 320, 64),
            ("CamShift",   "camshift",  240, 320, 64),
            ("MedianFlow", "medianflow",240, 320, 64),
            ("MIL",        "mil",       240, 320, 64),
        ]
        profiles = analyzer.estimate_tracker_fleet(configs)
        for p in profiles:
            assert p.flops > 0

    def test_markdown_table_contains_header(
        self, analyzer: ComplexityAnalyzer
    ) -> None:
        configs = [("MOSSE", "mosse", 240, 320, 64)]
        profiles = analyzer.estimate_tracker_fleet(configs)
        table = ComplexityAnalyzer.to_markdown_table(profiles)
        assert "MFLOPs" in table
        assert "MMACs" in table

    def test_markdown_table_contains_tracker_names(
        self, analyzer: ComplexityAnalyzer
    ) -> None:
        configs = [
            ("MOSSE", "mosse", 240, 320, 64),
            ("KCF",   "kcf",   240, 320, 64),
        ]
        profiles = analyzer.estimate_tracker_fleet(configs)
        table = ComplexityAnalyzer.to_markdown_table(profiles)
        assert "MOSSE" in table
        assert "KCF" in table

    def test_markdown_table_has_separator_line(
        self, analyzer: ComplexityAnalyzer
    ) -> None:
        configs = [("MOSSE", "mosse", 240, 320, 64)]
        profiles = analyzer.estimate_tracker_fleet(configs)
        table = ComplexityAnalyzer.to_markdown_table(profiles)
        lines = table.split("\n")
        assert any("---" in line for line in lines)


# ---------------------------------------------------------------------------
# PyTorch model profiling (skipped if torch unavailable)
# ---------------------------------------------------------------------------

class TestTorchModelProfiling:
    def test_basic_conv_model(self, analyzer: ComplexityAnalyzer) -> None:
        torch = pytest.importorskip("torch")
        nn = torch.nn

        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
        )
        dummy = torch.zeros(1, 3, 64, 64)
        profile = analyzer.profile_torch_model(model, dummy, name="TestConvNet")

        assert not profile.is_classical
        assert profile.macs > 0
        assert profile.params > 0
        assert profile.flops == 2 * profile.macs
        assert profile.memory_read_bytes == profile.params * 4

    def test_linear_model(self, analyzer: ComplexityAnalyzer) -> None:
        torch = pytest.importorskip("torch")
        nn = torch.nn

        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        dummy = torch.zeros(1, 64)
        profile = analyzer.profile_torch_model(model, dummy, name="TestMLP")

        assert profile.macs > 0
        assert profile.params > 0

    def test_model_restored_to_eval_mode(self, analyzer: ComplexityAnalyzer) -> None:
        torch = pytest.importorskip("torch")
        nn = torch.nn

        model = nn.Sequential(nn.Conv2d(3, 8, 3))
        model.eval()
        dummy = torch.zeros(1, 3, 32, 32)
        analyzer.profile_torch_model(model, dummy)
        assert not model.training

    def test_model_restored_to_train_mode(self, analyzer: ComplexityAnalyzer) -> None:
        torch = pytest.importorskip("torch")
        nn = torch.nn

        model = nn.Sequential(nn.Conv2d(3, 8, 3))
        model.train()
        dummy = torch.zeros(1, 3, 32, 32)
        analyzer.profile_torch_model(model, dummy)
        assert model.training

    def test_larger_model_more_macs(self, analyzer: ComplexityAnalyzer) -> None:
        torch = pytest.importorskip("torch")
        nn = torch.nn

        small = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1))
        large = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1))
        dummy = torch.zeros(1, 3, 64, 64)

        p_small = analyzer.profile_torch_model(small, dummy, name="small")
        p_large = analyzer.profile_torch_model(large, dummy, name="large")
        assert p_large.macs > p_small.macs

    def test_profile_name_stored(self, analyzer: ComplexityAnalyzer) -> None:
        torch = pytest.importorskip("torch")
        nn = torch.nn

        model = nn.Sequential(nn.Conv2d(1, 1, 1))
        dummy = torch.zeros(1, 1, 8, 8)
        profile = analyzer.profile_torch_model(model, dummy, name="MySiamNet")
        assert profile.name == "MySiamNet"

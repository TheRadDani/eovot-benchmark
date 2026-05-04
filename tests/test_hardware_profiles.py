"""Unit tests for hardware profiles and deployment feasibility analysis."""

import math
import os
import tempfile

import pytest
import yaml

from eovot.profiling.hardware_profiles import (
    BUILTIN_PROFILES,
    HardwareProfile,
    get_profile,
    load_profile_from_yaml,
)
from eovot.profiling.deployment_report import (
    compare_trackers_on_hardware,
    evaluate_deployment,
)


# ---------------------------------------------------------------------------
# HardwareProfile — unit tests
# ---------------------------------------------------------------------------


class TestHardwareProfile:
    def test_builtin_profiles_are_complete(self):
        expected = {"raspberry_pi4", "jetson_nano", "intel_nuc", "desktop_cpu", "smartphone"}
        assert set(BUILTIN_PROFILES.keys()) == expected

    def test_get_profile_returns_correct_instance(self):
        pi = get_profile("raspberry_pi4")
        assert isinstance(pi, HardwareProfile)
        assert pi.cpu_cores == 4
        assert pi.target_fps == 15.0

    def test_get_profile_raises_for_unknown_key(self):
        with pytest.raises(KeyError, match="Unknown hardware profile"):
            get_profile("nonexistent_device")

    def test_is_tracker_suitable_true(self):
        pi = get_profile("raspberry_pi4")
        # fps=20 > 15, latency=50 < 66.7, memory=200 < 512
        assert pi.is_tracker_suitable(fps=20.0, latency_ms=50.0, memory_mb=200.0) is True

    def test_is_tracker_suitable_false_fps(self):
        pi = get_profile("raspberry_pi4")
        assert pi.is_tracker_suitable(fps=10.0, latency_ms=50.0, memory_mb=200.0) is False

    def test_is_tracker_suitable_false_latency(self):
        pi = get_profile("raspberry_pi4")
        assert pi.is_tracker_suitable(fps=20.0, latency_ms=100.0, memory_mb=200.0) is False

    def test_is_tracker_suitable_false_memory(self):
        pi = get_profile("raspberry_pi4")
        assert pi.is_tracker_suitable(fps=20.0, latency_ms=50.0, memory_mb=600.0) is False

    def test_deployment_score_perfect(self):
        pi = get_profile("raspberry_pi4")
        # All constraints are exactly met or exceeded
        score = pi.deployment_score(fps=pi.target_fps, latency_ms=pi.max_latency_ms, memory_mb=pi.max_memory_mb)
        assert math.isclose(score, 1.0, abs_tol=1e-5)

    def test_deployment_score_zero_fps(self):
        pi = get_profile("raspberry_pi4")
        score = pi.deployment_score(fps=0.0, latency_ms=50.0, memory_mb=200.0)
        assert score == pytest.approx(0.0, abs=1e-5)

    def test_deployment_score_clamped_above_one(self):
        pi = get_profile("raspberry_pi4")
        # fps far exceeds target, etc — score must still be <= 1
        score = pi.deployment_score(fps=10_000.0, latency_ms=0.001, memory_mb=1.0)
        assert score <= 1.0 + 1e-9

    def test_deployment_score_partial(self):
        pi = get_profile("raspberry_pi4")
        # fps exactly meets target (score=1), latency and memory exceed limits (score < 1)
        score = pi.deployment_score(
            fps=pi.target_fps,
            latency_ms=pi.max_latency_ms * 2,  # 2x over limit
            memory_mb=pi.max_memory_mb * 2,
        )
        # fps_score=1, lat_score=0.5, mem_score=0.5 -> mean=0.666...
        assert 0.0 < score < 1.0

    def test_jetson_nano_has_gpu(self):
        jetson = get_profile("jetson_nano")
        assert jetson.has_gpu is True
        assert jetson.gpu_tdp_watts > 0.0

    def test_all_profiles_have_positive_target_fps(self):
        for key, profile in BUILTIN_PROFILES.items():
            assert profile.target_fps > 0, f"{key} has non-positive target_fps"


# ---------------------------------------------------------------------------
# load_profile_from_yaml
# ---------------------------------------------------------------------------


class TestLoadProfileFromYaml:
    def test_load_valid_yaml(self, tmp_path):
        data = {
            "name": "Test Device",
            "cpu_tdp_watts": 8.0,
            "ram_mb": 2048,
            "cpu_cores": 4,
            "cpu_freq_ghz": 1.8,
        }
        yaml_file = tmp_path / "test_device.yaml"
        yaml_file.write_text(yaml.dump(data), encoding="utf-8")

        profile = load_profile_from_yaml(str(yaml_file))
        assert profile.name == "Test Device"
        assert profile.cpu_tdp_watts == 8.0
        assert profile.ram_mb == 2048

    def test_load_yaml_ignores_unknown_keys(self, tmp_path):
        data = {
            "name": "Future Device",
            "cpu_tdp_watts": 5.0,
            "ram_mb": 1024,
            "cpu_cores": 2,
            "cpu_freq_ghz": 1.0,
            "unknown_future_field": "ignored",
        }
        yaml_file = tmp_path / "future.yaml"
        yaml_file.write_text(yaml.dump(data), encoding="utf-8")
        # Should not raise
        profile = load_profile_from_yaml(str(yaml_file))
        assert profile.name == "Future Device"

    def test_load_yaml_raises_for_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_profile_from_yaml("/nonexistent/path/device.yaml")


# ---------------------------------------------------------------------------
# evaluate_deployment
# ---------------------------------------------------------------------------


class TestEvaluateDeployment:
    """Tests for the single-tracker deployment feasibility report."""

    FAST_TRACKER = {"fps": 200.0, "mean_latency_ms": 5.0, "peak_memory_mb": 40.0}
    SLOW_TRACKER = {"fps": 5.0, "mean_latency_ms": 200.0, "peak_memory_mb": 800.0}

    def test_returns_dataframe_with_expected_columns(self):
        import pandas as pd
        df = evaluate_deployment(self.FAST_TRACKER)
        assert isinstance(df, pd.DataFrame)
        for col in ("profile", "suitable", "deployment_score", "fps_ok", "latency_ok", "memory_ok"):
            assert col in df.columns

    def test_fast_tracker_suitable_on_all_profiles(self):
        df = evaluate_deployment(self.FAST_TRACKER)
        assert df["suitable"].all(), "Fast tracker should be suitable on every profile"

    def test_slow_tracker_not_suitable_on_any_profile(self):
        df = evaluate_deployment(self.SLOW_TRACKER)
        assert not df["suitable"].any(), "Slow tracker should not be suitable on any profile"

    def test_sorted_by_deployment_score_descending(self):
        df = evaluate_deployment(self.FAST_TRACKER)
        scores = df["deployment_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_custom_profiles_list(self):
        from eovot.profiling.hardware_profiles import RASPBERRY_PI_4, JETSON_NANO
        df = evaluate_deployment(self.FAST_TRACKER, profiles=[RASPBERRY_PI_4, JETSON_NANO])
        assert len(df) == 2

    def test_defaults_to_all_builtin_profiles(self):
        df = evaluate_deployment(self.FAST_TRACKER)
        assert len(df) == len(BUILTIN_PROFILES)


# ---------------------------------------------------------------------------
# compare_trackers_on_hardware
# ---------------------------------------------------------------------------


class TestCompareTrackersOnHardware:
    TRACKER_RESULTS = {
        "MOSSE": {"fps": 520.0, "mean_latency_ms": 1.9, "peak_memory_mb": 42.0},
        "KCF": {"fps": 280.0, "mean_latency_ms": 3.6, "peak_memory_mb": 68.0},
        "CSRT": {"fps": 45.0, "mean_latency_ms": 22.2, "peak_memory_mb": 130.0},
    }

    def test_returns_dataframe_with_trackers_as_index(self):
        import pandas as pd
        df = compare_trackers_on_hardware(self.TRACKER_RESULTS)
        assert isinstance(df, pd.DataFrame)
        assert set(df.index) == {"MOSSE", "KCF", "CSRT"}

    def test_columns_match_profile_names(self):
        from eovot.profiling.hardware_profiles import RASPBERRY_PI_4, JETSON_NANO
        df = compare_trackers_on_hardware(
            self.TRACKER_RESULTS,
            profiles=[RASPBERRY_PI_4, JETSON_NANO],
        )
        assert RASPBERRY_PI_4.name in df.columns
        assert JETSON_NANO.name in df.columns

    def test_mosse_scores_higher_than_csrt_on_all_profiles(self):
        df = compare_trackers_on_hardware(self.TRACKER_RESULTS)
        for col in df.columns:
            assert df.loc["MOSSE", col] >= df.loc["CSRT", col], (
                f"MOSSE should score >= CSRT on {col}"
            )

    def test_scores_in_unit_interval(self):
        df = compare_trackers_on_hardware(self.TRACKER_RESULTS)
        assert (df.values >= 0.0).all()
        assert (df.values <= 1.0 + 1e-9).all()

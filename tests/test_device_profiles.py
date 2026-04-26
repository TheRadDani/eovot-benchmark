"""Tests for eovot.profiling.device_profiles and eovot.metrics.deployment_score."""

import pytest

from eovot.profiling.device_profiles import (
    DEVICE_PROFILES,
    DeviceProfile,
    get_device,
    list_devices,
)
from eovot.metrics.deployment_score import (
    DeploymentScore,
    DeploymentScorer,
    score_all_devices,
)


# ---------------------------------------------------------------------------
# DeviceProfile registry tests
# ---------------------------------------------------------------------------

class TestDeviceProfileRegistry:
    def test_all_expected_devices_present(self):
        expected = {
            "raspberry_pi_4",
            "jetson_nano",
            "jetson_orin_nano",
            "coral_usb",
            "intel_nuc_i5",
            "desktop_cpu",
        }
        assert expected.issubset(set(DEVICE_PROFILES.keys()))

    def test_list_devices_sorted(self):
        keys = list_devices()
        assert keys == sorted(keys)
        assert len(keys) >= 6

    def test_get_device_known(self):
        profile = get_device("raspberry_pi_4")
        assert isinstance(profile, DeviceProfile)
        assert profile.key == "raspberry_pi_4"
        assert profile.tdp_watts == pytest.approx(5.1)
        assert profile.memory_limit_mb == 4096
        assert profile.target_fps == pytest.approx(15.0)
        assert profile.cpu_cores == 4
        assert profile.has_gpu is False
        assert profile.has_npu is False

    def test_get_device_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown device"):
            get_device("nonexistent_device_xyz")

    def test_jetson_nano_has_gpu(self):
        profile = get_device("jetson_nano")
        assert profile.has_gpu is True

    def test_coral_usb_has_npu(self):
        profile = get_device("coral_usb")
        assert profile.has_npu is True

    def test_desktop_cpu_highest_tdp(self):
        desktop = get_device("desktop_cpu")
        all_tdps = [p.tdp_watts for p in DEVICE_PROFILES.values()]
        assert desktop.tdp_watts == max(all_tdps)

    def test_to_dict_contains_all_fields(self):
        d = get_device("jetson_nano").to_dict()
        required = {
            "key", "display_name", "tdp_watts", "memory_limit_mb",
            "target_fps", "cpu_cores", "has_gpu", "has_npu", "notes",
        }
        assert required.issubset(d.keys())


class TestDeviceProfileMethods:
    def test_fits_memory_within_limit(self):
        profile = get_device("raspberry_pi_4")
        assert profile.fits_memory(100.0) is True

    def test_fits_memory_at_limit(self):
        profile = get_device("raspberry_pi_4")
        assert profile.fits_memory(float(profile.memory_limit_mb)) is True

    def test_fits_memory_exceeds_limit(self):
        profile = get_device("raspberry_pi_4")
        assert profile.fits_memory(float(profile.memory_limit_mb) + 1.0) is False

    def test_meets_fps_above_target(self):
        profile = get_device("raspberry_pi_4")  # target 15.0
        assert profile.meets_fps(30.0) is True

    def test_meets_fps_at_target(self):
        profile = get_device("raspberry_pi_4")
        assert profile.meets_fps(15.0) is True

    def test_meets_fps_below_target(self):
        profile = get_device("raspberry_pi_4")
        assert profile.meets_fps(5.0) is False


# ---------------------------------------------------------------------------
# DeploymentScorer tests
# ---------------------------------------------------------------------------

class TestDeploymentScorer:
    def setup_method(self):
        self.device = get_device("raspberry_pi_4")
        self.scorer = DeploymentScorer(device=self.device)

    def test_score_returns_deployment_score_instance(self):
        result = self.scorer.score(mean_iou=0.6, fps=20.0, peak_memory_mb=100.0)
        assert isinstance(result, DeploymentScore)

    def test_score_accuracy_clipped_to_one(self):
        result = self.scorer.score(mean_iou=1.5, fps=20.0, peak_memory_mb=100.0)
        assert result.accuracy_score == pytest.approx(1.0)

    def test_score_accuracy_clipped_to_zero(self):
        result = self.scorer.score(mean_iou=-0.1, fps=20.0, peak_memory_mb=100.0)
        assert result.accuracy_score == pytest.approx(0.0)

    def test_fps_score_above_target_clips_to_one(self):
        # target_fps=15.0; pass fps=100.0 → fps_score=1.0
        result = self.scorer.score(mean_iou=0.5, fps=100.0, peak_memory_mb=100.0)
        assert result.fps_score == pytest.approx(1.0)

    def test_fps_score_below_target(self):
        # fps=7.5, target=15.0 → fps_score=0.5
        result = self.scorer.score(mean_iou=0.5, fps=7.5, peak_memory_mb=100.0)
        assert result.fps_score == pytest.approx(0.5)

    def test_memory_score_zero_when_exceeds_limit(self):
        result = self.scorer.score(
            mean_iou=0.5, fps=20.0, peak_memory_mb=float(self.device.memory_limit_mb) + 1
        )
        assert result.memory_score == pytest.approx(0.0)

    def test_total_score_in_unit_interval(self):
        result = self.scorer.score(mean_iou=0.5, fps=20.0, peak_memory_mb=100.0)
        assert 0.0 <= result.total_score <= 1.0

    def test_total_score_computation(self):
        # w_acc=0.5, w_fps=0.3, w_mem=0.2 (defaults)
        # acc=0.6, fps_score=min(1, 20/15)=1.0, mem_score=1-(100/4096)≈0.9756
        result = self.scorer.score(mean_iou=0.6, fps=20.0, peak_memory_mb=100.0)
        expected = 0.5 * 0.6 + 0.3 * 1.0 + 0.2 * (1.0 - 100.0 / 4096.0)
        assert result.total_score == pytest.approx(expected, abs=1e-3)

    def test_deployable_true_when_both_constraints_met(self):
        # fps=20 >= 15, mem=100 <= 4096
        result = self.scorer.score(mean_iou=0.5, fps=20.0, peak_memory_mb=100.0)
        assert result.deployable is True

    def test_not_deployable_when_fps_too_low(self):
        result = self.scorer.score(mean_iou=0.8, fps=5.0, peak_memory_mb=100.0)
        assert result.deployable is False

    def test_not_deployable_when_memory_too_high(self):
        result = self.scorer.score(mean_iou=0.8, fps=20.0, peak_memory_mb=5000.0)
        assert result.deployable is False

    def test_tracker_name_embedded(self):
        result = self.scorer.score(
            mean_iou=0.5, fps=20.0, peak_memory_mb=100.0, tracker_name="MOSSE"
        )
        assert result.tracker_name == "MOSSE"
        assert result.device_key == "raspberry_pi_4"

    def test_str_contains_device_key(self):
        result = self.scorer.score(mean_iou=0.5, fps=20.0, peak_memory_mb=100.0)
        assert "raspberry_pi_4" in str(result)

    def test_to_dict_keys(self):
        result = self.scorer.score(mean_iou=0.5, fps=20.0, peak_memory_mb=100.0)
        d = result.to_dict()
        required = {
            "device_key", "tracker_name", "total_score", "accuracy_score",
            "fps_score", "memory_score", "deployable", "fps_achieved",
            "peak_memory_mb",
        }
        assert required.issubset(d.keys())

    def test_invalid_weights_raises(self):
        with pytest.raises(ValueError, match="Weights must sum"):
            DeploymentScorer(device=self.device, weight_acc=0.6, weight_fps=0.3, weight_mem=0.2)

    def test_for_device_key_constructor(self):
        scorer = DeploymentScorer.for_device_key("jetson_nano")
        assert scorer.device.key == "jetson_nano"


# ---------------------------------------------------------------------------
# score_all_devices tests
# ---------------------------------------------------------------------------

class TestScoreAllDevices:
    def test_returns_all_registered_devices(self):
        scores = score_all_devices(mean_iou=0.5, fps=30.0, peak_memory_mb=200.0)
        assert set(scores.keys()) == set(DEVICE_PROFILES.keys())

    def test_all_total_scores_in_unit_interval(self):
        scores = score_all_devices(mean_iou=0.5, fps=30.0, peak_memory_mb=200.0)
        for key, s in scores.items():
            assert 0.0 <= s.total_score <= 1.0, f"Out of range for {key}"

    def test_desktop_cpu_most_likely_deployable(self):
        # Very fast, unconstrained — desktop should always be deployable
        scores = score_all_devices(mean_iou=0.5, fps=200.0, peak_memory_mb=100.0)
        assert scores["desktop_cpu"].deployable is True

    def test_coral_not_deployable_with_high_memory(self):
        # coral_usb has only 256 MB — 512 MB should fail
        scores = score_all_devices(mean_iou=0.5, fps=30.0, peak_memory_mb=512.0)
        assert scores["coral_usb"].deployable is False

    def test_tracker_name_propagated(self):
        scores = score_all_devices(
            mean_iou=0.5, fps=30.0, peak_memory_mb=100.0, tracker_name="KCF"
        )
        for s in scores.values():
            assert s.tracker_name == "KCF"

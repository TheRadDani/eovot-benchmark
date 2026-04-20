"""Tests for eovot.profiling.device_profiles."""

import pytest

from eovot.profiling.device_profiles import (
    DEVICE_PROFILES,
    DeviceProfile,
    get_profile,
    list_profiles,
)


class TestDeviceProfile:
    def test_valid_profile_creation(self):
        p = DeviceProfile(
            key="test_device",
            name="Test Device",
            tdp_watts=10.0,
            ram_gb=4.0,
            cpu_cores=4,
        )
        assert p.tdp_watts == 10.0
        assert p.ram_gb == 4.0
        assert p.cpu_cores == 4
        assert not p.has_gpu
        assert p.gpu_vram_gb is None

    def test_invalid_tdp_raises(self):
        with pytest.raises(ValueError, match="tdp_watts"):
            DeviceProfile(key="x", name="x", tdp_watts=0.0, ram_gb=4.0, cpu_cores=2)

    def test_invalid_ram_raises(self):
        with pytest.raises(ValueError, match="ram_gb"):
            DeviceProfile(key="x", name="x", tdp_watts=10.0, ram_gb=-1.0, cpu_cores=2)

    def test_invalid_cores_raises(self):
        with pytest.raises(ValueError, match="cpu_cores"):
            DeviceProfile(key="x", name="x", tdp_watts=10.0, ram_gb=4.0, cpu_cores=0)

    def test_to_dict_keys(self):
        p = DeviceProfile(
            key="rpi4",
            name="RPi 4",
            tdp_watts=6.4,
            ram_gb=4.0,
            cpu_cores=4,
            has_gpu=False,
        )
        d = p.to_dict()
        assert set(d.keys()) == {
            "key", "name", "tdp_watts", "ram_gb",
            "cpu_cores", "has_gpu", "gpu_vram_gb", "description",
        }
        assert d["tdp_watts"] == 6.4

    def test_str_representation(self):
        p = DeviceProfile(key="x", name="MyDevice", tdp_watts=15.0, ram_gb=8.0, cpu_cores=4)
        s = str(p)
        assert "MyDevice" in s
        assert "15.0" in s

    def test_gpu_profile_has_vram(self):
        p = DeviceProfile(
            key="gpu_box",
            name="GPU Box",
            tdp_watts=250.0,
            ram_gb=64.0,
            cpu_cores=16,
            has_gpu=True,
            gpu_vram_gb=24.0,
        )
        assert p.has_gpu
        assert p.gpu_vram_gb == 24.0
        assert "GPU" in str(p)


class TestBuiltinProfiles:
    def test_all_required_profiles_exist(self):
        required = [
            "raspberry_pi_4",
            "jetson_nano",
            "jetson_xavier_nx",
            "coral_dev_board",
            "intel_nuc_i5",
            "laptop_cpu",
            "desktop_gpu_workstation",
        ]
        for key in required:
            assert key in DEVICE_PROFILES, f"Missing profile: {key}"

    def test_all_profiles_have_positive_tdp(self):
        for key, profile in DEVICE_PROFILES.items():
            assert profile.tdp_watts > 0, f"{key}: tdp_watts must be positive"

    def test_all_profiles_have_positive_ram(self):
        for key, profile in DEVICE_PROFILES.items():
            assert profile.ram_gb > 0, f"{key}: ram_gb must be positive"

    def test_all_profiles_have_at_least_one_core(self):
        for key, profile in DEVICE_PROFILES.items():
            assert profile.cpu_cores >= 1, f"{key}: cpu_cores must be >= 1"

    def test_gpu_profiles_have_vram(self):
        for key, profile in DEVICE_PROFILES.items():
            if profile.has_gpu:
                assert profile.gpu_vram_gb is not None and profile.gpu_vram_gb > 0, (
                    f"{key}: GPU profile must have gpu_vram_gb > 0"
                )

    def test_raspberry_pi_4_specs(self):
        p = DEVICE_PROFILES["raspberry_pi_4"]
        assert p.tdp_watts == pytest.approx(6.4)
        assert p.ram_gb == 4.0
        assert p.cpu_cores == 4
        assert not p.has_gpu

    def test_jetson_nano_has_gpu(self):
        p = DEVICE_PROFILES["jetson_nano"]
        assert p.has_gpu
        assert p.gpu_vram_gb == pytest.approx(4.0)

    def test_desktop_workstation_is_highest_tdp(self):
        tdps = {k: v.tdp_watts for k, v in DEVICE_PROFILES.items()}
        assert max(tdps, key=tdps.get) == "desktop_gpu_workstation"

    def test_coral_is_lowest_tdp(self):
        tdps = {k: v.tdp_watts for k, v in DEVICE_PROFILES.items()}
        assert min(tdps, key=tdps.get) == "coral_dev_board"


class TestGetProfile:
    def test_get_valid_profile(self):
        p = get_profile("jetson_nano")
        assert isinstance(p, DeviceProfile)
        assert p.key == "jetson_nano"

    def test_get_invalid_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown device profile"):
            get_profile("nonexistent_device")

    def test_get_profile_error_lists_available(self):
        with pytest.raises(ValueError) as exc_info:
            get_profile("bad_key")
        assert "raspberry_pi_4" in str(exc_info.value)

    def test_get_all_profiles_by_key(self):
        for key in DEVICE_PROFILES:
            p = get_profile(key)
            assert p.key == key


class TestListProfiles:
    def test_returns_dict(self):
        profiles = list_profiles()
        assert isinstance(profiles, dict)

    def test_returns_copy(self):
        p1 = list_profiles()
        p2 = list_profiles()
        assert p1 is not p2
        p1["injected"] = None
        assert "injected" not in DEVICE_PROFILES

    def test_contains_all_profiles(self):
        profiles = list_profiles()
        assert set(profiles.keys()) == set(DEVICE_PROFILES.keys())

"""Unit tests for eovot.profiling.hardware_profiles."""

from __future__ import annotations

import pytest

from eovot.profiling.hardware_profiles import (
    HARDWARE_PROFILES,
    HardwareProfile,
    get_profile,
    list_profiles,
)


class TestHardwareProfile:
    def test_is_frozen_dataclass(self):
        profile = get_profile("rpi4")
        with pytest.raises((TypeError, AttributeError)):
            profile.cpu_tdp_w = 999.0  # type: ignore[misc]

    def test_fields_have_correct_types(self):
        p = get_profile("jetson_nano")
        assert isinstance(p.name, str)
        assert isinstance(p.cpu_tdp_w, float)
        assert isinstance(p.total_tdp_w, float)
        assert isinstance(p.ram_gb, float)
        assert isinstance(p.category, str)

    def test_gpu_tdp_none_for_cpu_only_device(self):
        p = get_profile("rpi4")
        assert p.gpu_tdp_w is None

    def test_gpu_tdp_present_for_jetson(self):
        p = get_profile("jetson_nano")
        assert p.gpu_tdp_w is not None
        assert p.gpu_tdp_w > 0


class TestGetProfile:
    def test_known_profiles_return_correctly(self):
        for key in ("rpi4", "rpi5", "jetson_nano", "jetson_nx", "jetson_agx",
                    "apple_m1", "apple_m2", "intel_core_u15w"):
            p = get_profile(key)
            assert isinstance(p, HardwareProfile)

    def test_case_insensitive(self):
        p1 = get_profile("RPI4")
        p2 = get_profile("rpi4")
        assert p1 == p2

    def test_unknown_key_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown hardware profile"):
            get_profile("nonexistent_device_xyz")

    def test_error_message_lists_available_profiles(self):
        with pytest.raises(KeyError) as exc_info:
            get_profile("bad_key")
        assert "rpi4" in str(exc_info.value)

    def test_rpi4_total_tdp(self):
        p = get_profile("rpi4")
        assert p.total_tdp_w == pytest.approx(6.0)

    def test_jetson_nano_total_tdp(self):
        p = get_profile("jetson_nano")
        assert p.total_tdp_w == pytest.approx(10.0)

    def test_category_is_edge_for_jetson(self):
        p = get_profile("jetson_nx")
        assert p.category == "edge"

    def test_category_is_laptop_for_m1(self):
        p = get_profile("apple_m1")
        assert p.category == "laptop"


class TestListProfiles:
    def test_returns_dict(self):
        result = list_profiles()
        assert isinstance(result, dict)

    def test_all_keys_present(self):
        result = list_profiles()
        for key in HARDWARE_PROFILES:
            assert key in result

    def test_values_are_strings(self):
        for name in list_profiles().values():
            assert isinstance(name, str)
            assert len(name) > 0

    def test_count_matches_registry(self):
        assert len(list_profiles()) == len(HARDWARE_PROFILES)

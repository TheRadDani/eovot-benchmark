"""Unit tests for eovot.hardware.detector.

Uses monkeypatching to exercise every detection branch without requiring
actual edge-device hardware or filesystem side effects.
"""

from __future__ import annotations

import pytest

from eovot.hardware.detector import (
    HardwarePlatform,
    _PRESETS,
    detect_platform,
    get_recommended_tdp,
    list_known_platforms,
)


# ---------------------------------------------------------------------------
# HardwarePlatform dataclass
# ---------------------------------------------------------------------------

class TestHardwarePlatform:
    def test_is_frozen(self):
        hp = HardwarePlatform(
            name="Test", arch="x86_64", tdp_watts=10.0, description="Test"
        )
        with pytest.raises((AttributeError, TypeError)):
            hp.tdp_watts = 99.0  # type: ignore[misc]

    def test_str_contains_name_and_tdp(self):
        hp = HardwarePlatform(
            name="My Device", arch="arm", tdp_watts=7.5, description="desc", detected=True
        )
        s = str(hp)
        assert "My Device" in s
        assert "7.5" in s

    def test_str_shows_detected_tag(self):
        hp = HardwarePlatform(name="X", arch="x86_64", tdp_watts=10.0, description="", detected=True)
        assert "detected" in str(hp)

    def test_str_shows_assumed_tag(self):
        hp = HardwarePlatform(name="X", arch="x86_64", tdp_watts=10.0, description="", detected=False)
        assert "assumed" in str(hp)


# ---------------------------------------------------------------------------
# _PRESETS integrity
# ---------------------------------------------------------------------------

class TestPresets:
    def test_required_keys_present(self):
        required = {"name", "arch", "tdp_watts", "description"}
        for key, preset in _PRESETS.items():
            assert required <= set(preset), f"Preset {key!r} missing keys"

    def test_tdp_watts_are_positive(self):
        for key, preset in _PRESETS.items():
            assert preset["tdp_watts"] > 0, f"Preset {key!r} has non-positive TDP"

    def test_laptop_tdp_lower_than_desktop(self):
        assert _PRESETS["laptop"]["tdp_watts"] < _PRESETS["desktop"]["tdp_watts"]

    def test_raspberry_pi_lower_than_laptop(self):
        assert _PRESETS["raspberry_pi"]["tdp_watts"] < _PRESETS["laptop"]["tdp_watts"]


# ---------------------------------------------------------------------------
# detect_platform — monkeypatched probes
# ---------------------------------------------------------------------------

class TestDetectPlatform:
    """Test each detection branch by patching internal helpers."""

    def test_raspberry_pi_detected_via_model(self, monkeypatch):
        monkeypatch.setattr(
            "eovot.hardware.detector._probe_device_tree_model",
            lambda: "raspberry pi 4 model b rev 1.4",
        )
        result = detect_platform()
        assert result.name == "Raspberry Pi 4"
        assert result.detected is True
        assert result.tdp_watts == pytest.approx(6.0)

    def test_jetson_nano_detected_via_model(self, monkeypatch):
        monkeypatch.setattr(
            "eovot.hardware.detector._probe_device_tree_model",
            lambda: "nvidia jetson nano developer kit",
        )
        result = detect_platform()
        assert "Jetson" in result.name
        assert result.detected is True

    def test_jetson_orin_detected(self, monkeypatch):
        monkeypatch.setattr(
            "eovot.hardware.detector._probe_device_tree_model",
            lambda: "nvidia jetson orin nx 16gb",
        )
        result = detect_platform()
        assert "Orin" in result.name

    def test_jetson_detected_via_compatible(self, monkeypatch):
        monkeypatch.setattr(
            "eovot.hardware.detector._probe_device_tree_model",
            lambda: "",
        )
        monkeypatch.setattr(
            "eovot.hardware.detector._probe_device_tree_compatible",
            lambda: "nvidia,tegra210",
        )
        result = detect_platform()
        assert "Jetson" in result.name

    def test_laptop_detected_via_battery(self, monkeypatch):
        monkeypatch.setattr("eovot.hardware.detector._probe_device_tree_model", lambda: "")
        monkeypatch.setattr("eovot.hardware.detector._probe_device_tree_compatible", lambda: "")
        monkeypatch.setattr("eovot.hardware.detector._has_battery", lambda: True)
        import platform as _plat
        monkeypatch.setattr(_plat, "machine", lambda: "x86_64")
        result = detect_platform()
        assert result.name == "Laptop CPU"
        assert result.tdp_watts == pytest.approx(15.0)

    def test_desktop_detected_via_dmi(self, monkeypatch):
        monkeypatch.setattr("eovot.hardware.detector._probe_device_tree_model", lambda: "")
        monkeypatch.setattr("eovot.hardware.detector._probe_device_tree_compatible", lambda: "")
        monkeypatch.setattr("eovot.hardware.detector._has_battery", lambda: False)
        monkeypatch.setattr("eovot.hardware.detector._dmi_chassis_type", lambda: "3")
        import platform as _plat
        monkeypatch.setattr(_plat, "machine", lambda: "x86_64")
        result = detect_platform()
        assert result.name == "Desktop CPU"

    def test_laptop_detected_via_dmi_chassis(self, monkeypatch):
        monkeypatch.setattr("eovot.hardware.detector._probe_device_tree_model", lambda: "")
        monkeypatch.setattr("eovot.hardware.detector._probe_device_tree_compatible", lambda: "")
        monkeypatch.setattr("eovot.hardware.detector._has_battery", lambda: False)
        monkeypatch.setattr("eovot.hardware.detector._dmi_chassis_type", lambda: "9")
        import platform as _plat
        monkeypatch.setattr(_plat, "machine", lambda: "x86_64")
        result = detect_platform()
        assert result.name == "Laptop CPU"

    def test_fallback_returns_laptop(self, monkeypatch):
        """When all probes fail, the fallback should be Laptop (safe default)."""
        monkeypatch.setattr("eovot.hardware.detector._probe_device_tree_model", lambda: "")
        monkeypatch.setattr("eovot.hardware.detector._probe_device_tree_compatible", lambda: "")
        monkeypatch.setattr("eovot.hardware.detector._has_battery", lambda: False)
        monkeypatch.setattr("eovot.hardware.detector._dmi_chassis_type", lambda: "")
        import platform as _plat
        monkeypatch.setattr(_plat, "machine", lambda: "x86_64")
        result = detect_platform()
        assert result.name == "Laptop CPU"
        assert result.detected is False

    def test_arm_without_device_tree_returns_edge(self, monkeypatch):
        monkeypatch.setattr("eovot.hardware.detector._probe_device_tree_model", lambda: "")
        monkeypatch.setattr("eovot.hardware.detector._probe_device_tree_compatible", lambda: "")
        import platform as _plat
        monkeypatch.setattr(_plat, "machine", lambda: "aarch64")
        result = detect_platform()
        assert result.arch == "aarch64"
        assert result.tdp_watts <= 15.0  # edge or RPi, not laptop/desktop


# ---------------------------------------------------------------------------
# get_recommended_tdp
# ---------------------------------------------------------------------------

class TestGetRecommendedTDP:
    def test_returns_float(self):
        tdp = get_recommended_tdp()
        assert isinstance(tdp, float)
        assert tdp > 0.0

    def test_known_override_raspberry_pi(self):
        tdp = get_recommended_tdp("raspberry_pi")
        assert tdp == pytest.approx(6.0)

    def test_known_override_laptop(self):
        tdp = get_recommended_tdp("laptop")
        assert tdp == pytest.approx(15.0)

    def test_known_override_desktop(self):
        tdp = get_recommended_tdp("desktop")
        assert tdp == pytest.approx(65.0)

    def test_invalid_override_raises(self):
        with pytest.raises(ValueError, match="Unknown platform preset"):
            get_recommended_tdp("flying_saucer")

    def test_all_preset_keys_accepted(self):
        for key in _PRESETS:
            tdp = get_recommended_tdp(key)
            assert tdp > 0.0


# ---------------------------------------------------------------------------
# list_known_platforms
# ---------------------------------------------------------------------------

class TestListKnownPlatforms:
    def test_returns_list(self):
        platforms = list_known_platforms()
        assert isinstance(platforms, list)
        assert len(platforms) > 0

    def test_all_are_hardware_platform_instances(self):
        for p in list_known_platforms():
            assert isinstance(p, HardwarePlatform)

    def test_count_matches_presets(self):
        assert len(list_known_platforms()) == len(_PRESETS)


# ---------------------------------------------------------------------------
# Package-level imports
# ---------------------------------------------------------------------------

class TestHardwarePackageImports:
    def test_imports_from_package(self):
        from eovot.hardware import (  # noqa: F401
            HardwarePlatform,
            detect_platform,
            get_recommended_tdp,
        )

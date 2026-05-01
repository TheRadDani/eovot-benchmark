"""Unit tests for eovot.profiling.device_profiles."""

import pytest

from eovot.profiling.device_profiles import (
    DeviceProfile,
    get_profile,
    list_profiles,
    register_profile,
    _REGISTRY,
)


# ---------------------------------------------------------------------------
# DeviceProfile construction
# ---------------------------------------------------------------------------

class TestDeviceProfile:
    def test_valid_construction(self):
        p = DeviceProfile(
            name="test-device",
            display_name="Test Device",
            device_class="edge-cpu",
            tdp_watts=5.0,
            ram_gb=2.0,
            cpu_cores=4,
            description="A test device",
        )
        assert p.tdp_watts == 5.0
        assert p.device_class == "edge-cpu"
        assert p.has_gpu is False

    def test_invalid_device_class_raises(self):
        with pytest.raises(ValueError, match="device_class"):
            DeviceProfile(
                name="bad",
                display_name="Bad",
                device_class="quantum-computer",
                tdp_watts=5.0,
            )

    def test_non_positive_tdp_raises(self):
        with pytest.raises(ValueError, match="tdp_watts"):
            DeviceProfile(
                name="zero-tdp",
                display_name="Zero TDP",
                device_class="laptop",
                tdp_watts=0.0,
            )

    def test_to_dict_keys(self):
        p = DeviceProfile(
            name="x",
            display_name="X",
            device_class="server",
            tdp_watts=200.0,
        )
        d = p.to_dict()
        for key in ("name", "display_name", "device_class", "tdp_watts",
                    "peak_power_w", "ram_gb", "cpu_cores", "has_gpu",
                    "typical_fps_target", "description"):
            assert key in d

    def test_str_representation(self):
        p = DeviceProfile(
            name="rpi4-copy",
            display_name="RPi4 copy",
            device_class="edge-cpu",
            tdp_watts=6.0,
            has_gpu=False,
        )
        s = str(p)
        assert "rpi4-copy" in s
        assert "6.0" in s


# ---------------------------------------------------------------------------
# Built-in registry
# ---------------------------------------------------------------------------

class TestBuiltinRegistry:
    def test_rpi4_exists(self):
        p = get_profile("rpi4")
        assert p.tdp_watts == 6.0
        assert p.device_class == "edge-cpu"
        assert p.has_gpu is False

    def test_jetson_nano_exists(self):
        p = get_profile("jetson-nano")
        assert p.tdp_watts == 10.0
        assert p.has_gpu is True
        assert p.device_class == "edge-gpu"

    def test_jetson_xavier_nx_exists(self):
        p = get_profile("jetson-xavier-nx")
        assert p.tdp_watts == 15.0
        assert p.device_class == "edge-gpu"

    def test_laptop_low_exists(self):
        p = get_profile("laptop-low")
        assert p.device_class == "laptop"
        assert p.tdp_watts == 15.0

    def test_desktop_cpu_exists(self):
        p = get_profile("desktop-cpu")
        assert p.device_class == "workstation"
        assert p.tdp_watts >= 60.0

    def test_server_cpu_exists(self):
        p = get_profile("server-cpu")
        assert p.device_class == "server"

    def test_case_insensitive_lookup(self):
        p1 = get_profile("RPI4")
        p2 = get_profile("rpi4")
        assert p1.name == p2.name

    def test_unknown_profile_raises_key_error(self):
        with pytest.raises(KeyError, match="not-a-real-device"):
            get_profile("not-a-real-device")

    def test_error_message_lists_available(self):
        try:
            get_profile("ghost-device")
        except KeyError as exc:
            assert "rpi4" in str(exc)

    def test_all_built_in_profiles_are_valid(self):
        for name, profile in list_profiles():
            assert profile.tdp_watts > 0
            assert profile.name == name
            assert profile.display_name
            assert profile.device_class in {
                "edge-cpu", "edge-gpu", "laptop", "workstation", "server"
            }

    def test_list_profiles_returns_sorted(self):
        names = [n for n, _ in list_profiles()]
        assert names == sorted(names)

    def test_list_profiles_non_empty(self):
        assert len(list_profiles()) >= 10


# ---------------------------------------------------------------------------
# Custom profile registration
# ---------------------------------------------------------------------------

class TestRegisterProfile:
    def test_register_and_retrieve(self):
        custom = DeviceProfile(
            name="_test-custom-board",
            display_name="Custom Board",
            device_class="edge-cpu",
            tdp_watts=3.5,
            description="Ephemeral test device",
        )
        register_profile(custom)
        retrieved = get_profile("_test-custom-board")
        assert retrieved.tdp_watts == 3.5
        # Cleanup so other tests are not affected
        del _REGISTRY["_test-custom-board"]

    def test_duplicate_registration_raises(self):
        dup = DeviceProfile(
            name="rpi4",
            display_name="Duplicate RPi4",
            device_class="edge-cpu",
            tdp_watts=6.0,
        )
        with pytest.raises(ValueError, match="already registered"):
            register_profile(dup)

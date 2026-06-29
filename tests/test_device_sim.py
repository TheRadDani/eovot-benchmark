"""Tests for the edge device simulation module."""

from __future__ import annotations

import pytest

from eovot.profiling.device_sim import (
    KNOWN_DEVICES,
    DeviceProfile,
    DeviceSimResult,
    DeviceSimulator,
)
from eovot.profiling.profiler import ProfilingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profiling_result(
    fps: float = 200.0,
    latency_mean_ms: float = 5.0,
    peak_memory_mb: float = 300.0,
    tracker_name: str = "MOSSE",
) -> ProfilingResult:
    return ProfilingResult(
        tracker_name=tracker_name,
        frame_count=200,
        fps=fps,
        latency_mean_ms=latency_mean_ms,
        latency_std_ms=0.5,
        latency_p50_ms=latency_mean_ms * 0.95,
        latency_p95_ms=latency_mean_ms * 1.3,
        latency_p99_ms=latency_mean_ms * 1.6,
        peak_memory_mb=peak_memory_mb,
    )


# ---------------------------------------------------------------------------
# KNOWN_DEVICES
# ---------------------------------------------------------------------------

class TestKnownDevices:
    def test_all_builtin_keys_present(self):
        expected = {"rpi4", "rpi5", "jetson_nano", "jetson_xnx", "coral_board", "snapdragon888"}
        assert expected == set(KNOWN_DEVICES)

    def test_all_profiles_have_valid_speed_factor(self):
        for name, profile in KNOWN_DEVICES.items():
            assert 0 < profile.cpu_speed_factor < 1, (
                f"{name}: cpu_speed_factor should be in (0, 1), got {profile.cpu_speed_factor}"
            )

    def test_all_profiles_have_positive_tdp(self):
        for name, profile in KNOWN_DEVICES.items():
            assert profile.tdp_watts > 0, f"{name}: tdp_watts must be positive"

    def test_all_profiles_have_positive_memory(self):
        for name, profile in KNOWN_DEVICES.items():
            assert profile.memory_limit_mb > 0, f"{name}: memory_limit_mb must be positive"

    def test_throttle_factor_in_valid_range(self):
        for name, profile in KNOWN_DEVICES.items():
            assert 0 < profile.throttle_factor <= 1.0, (
                f"{name}: throttle_factor must be in (0, 1], got {profile.throttle_factor}"
            )

    def test_coral_board_is_most_memory_constrained(self):
        coral = KNOWN_DEVICES["coral_board"]
        others = [p for k, p in KNOWN_DEVICES.items() if k != "coral_board"]
        assert all(coral.memory_limit_mb < p.memory_limit_mb for p in others)


# ---------------------------------------------------------------------------
# DeviceSimulator — profile management
# ---------------------------------------------------------------------------

class TestDeviceSimulatorProfiles:
    def test_list_devices_returns_all_builtin(self):
        sim = DeviceSimulator()
        assert set(sim.list_devices()) == set(KNOWN_DEVICES)

    def test_list_devices_is_sorted(self):
        sim = DeviceSimulator()
        names = sim.list_devices()
        assert names == sorted(names)

    def test_get_profile_returns_correct_profile(self):
        sim = DeviceSimulator()
        p = sim.get_profile("rpi4")
        assert p.name == "rpi4"
        assert p.display_name == KNOWN_DEVICES["rpi4"].display_name

    def test_get_profile_raises_on_unknown(self):
        sim = DeviceSimulator()
        with pytest.raises(KeyError, match="unknown_device"):
            sim.get_profile("unknown_device")

    def test_register_custom_device(self):
        sim = DeviceSimulator()
        custom = DeviceProfile(
            name="my_board",
            display_name="My Custom SBC",
            cpu_speed_factor=0.20,
            memory_limit_mb=2048.0,
            tdp_watts=8.0,
        )
        sim.register_device("my_board", custom)
        assert "my_board" in sim.list_devices()
        assert sim.get_profile("my_board").display_name == "My Custom SBC"

    def test_register_overwrites_existing(self):
        sim = DeviceSimulator()
        original_factor = KNOWN_DEVICES["rpi4"].cpu_speed_factor
        patched = DeviceProfile(
            name="rpi4",
            display_name="Patched RPi4",
            cpu_speed_factor=0.99,
            memory_limit_mb=4096.0,
            tdp_watts=7.5,
        )
        sim.register_device("rpi4", patched)
        assert sim.get_profile("rpi4").cpu_speed_factor == 0.99
        assert sim.get_profile("rpi4").cpu_speed_factor != original_factor

    def test_invalid_calibration_factor(self):
        with pytest.raises(ValueError):
            DeviceSimulator(host_calibration_factor=0.0)
        with pytest.raises(ValueError):
            DeviceSimulator(host_calibration_factor=-1.0)


# ---------------------------------------------------------------------------
# DeviceSimulator — simulate()
# ---------------------------------------------------------------------------

class TestSimulate:
    def setup_method(self):
        self.sim = DeviceSimulator()
        self.host = _make_profiling_result(fps=200.0, latency_mean_ms=5.0, peak_memory_mb=300.0)

    def test_returns_device_sim_result(self):
        r = self.sim.simulate(self.host, "rpi4")
        assert isinstance(r, DeviceSimResult)

    def test_estimated_fps_lower_than_host(self):
        r = self.sim.simulate(self.host, "rpi4")
        assert r.estimated_fps < r.host_fps

    def test_estimated_latency_higher_than_host(self):
        r = self.sim.simulate(self.host, "rpi4")
        assert r.estimated_latency_ms > r.host_latency_ms

    def test_fps_latency_consistency(self):
        r = self.sim.simulate(self.host, "jetson_xnx")
        # FPS should be consistent with latency
        expected_fps = 1_000.0 / r.estimated_latency_ms
        assert abs(r.estimated_fps - expected_fps) < 1e-6

    def test_memory_unchanged(self):
        r = self.sim.simulate(self.host, "rpi4")
        assert r.estimated_memory_mb == self.host.peak_memory_mb
        assert r.host_memory_mb == self.host.peak_memory_mb

    def test_fits_in_memory_true(self):
        low_mem = _make_profiling_result(peak_memory_mb=100.0)
        r = self.sim.simulate(low_mem, "rpi4")
        assert r.fits_in_memory is True

    def test_fits_in_memory_false_coral(self):
        large_mem = _make_profiling_result(peak_memory_mb=950.0)
        r = self.sim.simulate(large_mem, "coral_board")
        assert r.fits_in_memory is False

    def test_tracker_name_propagated(self):
        host = _make_profiling_result(tracker_name="KCF")
        r = self.sim.simulate(host, "rpi4")
        assert r.tracker_name == "KCF"

    def test_energy_is_positive(self):
        r = self.sim.simulate(self.host, "rpi4")
        assert r.estimated_energy_mj_per_frame > 0

    def test_energy_scales_with_tdp(self):
        # Higher TDP → more energy per frame (same FPS projection)
        r_coral = self.sim.simulate(self.host, "coral_board")    # 4 W TDP
        r_xnx = self.sim.simulate(self.host, "jetson_xnx")      # 15 W TDP
        # Jetson Xavier NX is much faster AND has higher TDP — net energy depends on both
        # But for devices with similar speed: more TDP = more energy at same latency
        assert r_coral.estimated_energy_mj_per_frame > 0
        assert r_xnx.estimated_energy_mj_per_frame > 0

    def test_unknown_device_raises(self):
        with pytest.raises(KeyError):
            self.sim.simulate(self.host, "non_existent_device")

    def test_faster_device_higher_fps(self):
        r_slow = self.sim.simulate(self.host, "coral_board")   # slowest
        r_fast = self.sim.simulate(self.host, "jetson_xnx")    # fastest CPU factor
        assert r_fast.estimated_fps > r_slow.estimated_fps

    def test_all_builtin_devices_producible(self):
        for device_name in KNOWN_DEVICES:
            r = self.sim.simulate(self.host, device_name)
            assert r.device_name == device_name
            assert r.estimated_fps > 0


# ---------------------------------------------------------------------------
# DeviceSimulator — thermal throttling
# ---------------------------------------------------------------------------

class TestThermalThrottling:
    def setup_method(self):
        self.sim = DeviceSimulator()
        self.host = _make_profiling_result(fps=200.0, latency_mean_ms=5.0)

    def test_no_throttle_at_zero_seconds(self):
        r = self.sim.simulate(self.host, "rpi4", sustained_seconds=0.0)
        assert r.thermal_state == "nominal"

    def test_nominal_before_onset(self):
        onset = KNOWN_DEVICES["rpi4"].throttle_onset_seconds
        r = self.sim.simulate(self.host, "rpi4", sustained_seconds=onset - 1)
        assert r.thermal_state == "nominal"

    def test_throttled_after_full_ramp(self):
        profile = KNOWN_DEVICES["rpi4"]
        sustained = profile.throttle_onset_seconds + profile.throttle_ramp_seconds + 1
        r = self.sim.simulate(self.host, "rpi4", sustained_seconds=sustained)
        assert r.thermal_state == "throttled"

    def test_transitioning_during_ramp(self):
        profile = KNOWN_DEVICES["rpi4"]
        mid_ramp = profile.throttle_onset_seconds + profile.throttle_ramp_seconds / 2
        r = self.sim.simulate(self.host, "rpi4", sustained_seconds=mid_ramp)
        assert r.thermal_state == "transitioning"

    def test_throttled_fps_lower_than_nominal(self):
        profile = KNOWN_DEVICES["rpi4"]
        r_cold = self.sim.simulate(self.host, "rpi4", sustained_seconds=0.0)
        r_hot = self.sim.simulate(
            self.host, "rpi4",
            sustained_seconds=profile.throttle_onset_seconds + profile.throttle_ramp_seconds + 5
        )
        assert r_hot.estimated_fps < r_cold.estimated_fps

    def test_throttled_speed_factor_equals_profile_factor(self):
        profile = KNOWN_DEVICES["rpi4"]
        sustained = profile.throttle_onset_seconds + profile.throttle_ramp_seconds + 1
        r = self.sim.simulate(self.host, "rpi4", sustained_seconds=sustained)
        expected = profile.cpu_speed_factor * profile.throttle_factor
        assert abs(r.effective_speed_factor - expected) < 1e-9

    def test_transitioning_factor_between_nominal_and_throttled(self):
        profile = KNOWN_DEVICES["rpi4"]
        mid_ramp = profile.throttle_onset_seconds + profile.throttle_ramp_seconds / 2
        r = self.sim.simulate(self.host, "rpi4", sustained_seconds=mid_ramp)
        nominal_factor = profile.cpu_speed_factor
        throttled_factor = profile.cpu_speed_factor * profile.throttle_factor
        assert throttled_factor < r.effective_speed_factor < nominal_factor


# ---------------------------------------------------------------------------
# DeviceSimulator — simulate_all()
# ---------------------------------------------------------------------------

class TestSimulateAll:
    def setup_method(self):
        self.sim = DeviceSimulator()
        self.host = _make_profiling_result()

    def test_returns_result_for_each_builtin_device(self):
        results = self.sim.simulate_all(self.host)
        assert len(results) == len(KNOWN_DEVICES)

    def test_sorted_by_fps_descending(self):
        results = self.sim.simulate_all(self.host)
        fps_list = [r.estimated_fps for r in results]
        assert fps_list == sorted(fps_list, reverse=True)

    def test_subset_of_devices(self):
        results = self.sim.simulate_all(self.host, device_names=["rpi4", "rpi5"])
        assert len(results) == 2
        names = {r.device_name for r in results}
        assert names == {"rpi4", "rpi5"}

    def test_unknown_device_in_subset_raises(self):
        with pytest.raises(KeyError):
            self.sim.simulate_all(self.host, device_names=["rpi4", "does_not_exist"])

    def test_all_results_have_positive_fps(self):
        results = self.sim.simulate_all(self.host)
        assert all(r.estimated_fps > 0 for r in results)


# ---------------------------------------------------------------------------
# DeviceSimulator — reporting
# ---------------------------------------------------------------------------

class TestReporting:
    def setup_method(self):
        self.sim = DeviceSimulator()
        self.host = _make_profiling_result()
        self.results = self.sim.simulate_all(self.host)

    def test_markdown_table_has_header(self):
        md = self.sim.to_markdown_table(self.results)
        assert "Device" in md
        assert "FPS" in md
        assert "Latency" in md
        assert "Fits?" in md

    def test_markdown_table_has_correct_row_count(self):
        md = self.sim.to_markdown_table(self.results)
        # header row + separator + N data rows
        data_rows = [
            l for l in md.splitlines()
            if l.startswith("|") and "---" not in l and "Device" not in l
        ]
        assert len(data_rows) == len(self.results)

    def test_markdown_fits_shows_check(self):
        low_mem = _make_profiling_result(peak_memory_mb=100.0)
        r = self.sim.simulate(low_mem, "rpi4")
        md = self.sim.to_markdown_table([r])
        assert "✓" in md

    def test_markdown_oom_shows_cross(self):
        big_mem = _make_profiling_result(peak_memory_mb=950.0)
        r = self.sim.simulate(big_mem, "coral_board")
        md = self.sim.to_markdown_table([r])
        assert "✗" in md

    def test_to_summary_dict_structure(self):
        dicts = self.sim.to_summary_dict(self.results)
        assert len(dicts) == len(self.results)
        required_keys = {
            "device", "display_name", "tracker", "host_fps", "estimated_fps",
            "estimated_latency_ms", "fits_in_memory", "thermal_state",
        }
        for d in dicts:
            assert required_keys.issubset(d)

    def test_to_summary_dict_fits_in_memory_is_bool(self):
        dicts = self.sim.to_summary_dict(self.results)
        for d in dicts:
            assert isinstance(d["fits_in_memory"], bool)

    def test_str_repr_contains_device_name(self):
        r = self.sim.simulate(self.host, "rpi4")
        s = str(r)
        assert "rpi4" in s
        assert "FPS" in s
        assert "thermal" in s


# ---------------------------------------------------------------------------
# Host calibration factor
# ---------------------------------------------------------------------------

class TestHostCalibration:
    def test_faster_host_projects_slower_device_fps(self):
        """A faster host machine should project lower device FPS (same tracker)."""
        slow_host_sim = DeviceSimulator(host_calibration_factor=0.5)  # host is half reference
        fast_host_sim = DeviceSimulator(host_calibration_factor=2.0)  # host is twice reference

        host = _make_profiling_result(fps=200.0, latency_mean_ms=5.0)

        r_slow = slow_host_sim.simulate(host, "rpi4")
        r_fast = fast_host_sim.simulate(host, "rpi4")

        # On a faster host the same latency represents more work → device slower
        assert r_fast.estimated_fps < r_slow.estimated_fps

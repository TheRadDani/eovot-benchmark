"""Unit tests for eovot.profiling.hardware_profiles."""

import pytest

from eovot.profiling.hardware_profiles import (
    HardwareProfile,
    PROFILES,
    get_profile,
    JETSON_NANO,
    RASPBERRY_PI_4,
    RASPBERRY_PI_ZERO_2W,
    LAPTOP_CPU,
    DESKTOP_GPU,
)


# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------

def test_profiles_registry_not_empty():
    assert len(PROFILES) > 0


def test_all_expected_slugs_present():
    expected = {"jetson-nano", "rpi4", "rpi-zero-2w", "intel-ncs2", "coral-tpu", "laptop", "desktop"}
    assert expected.issubset(set(PROFILES))


def test_all_profiles_are_hardware_profile_instances():
    for slug, p in PROFILES.items():
        assert isinstance(p, HardwareProfile), f"{slug} is not a HardwareProfile"


# ---------------------------------------------------------------------------
# get_profile
# ---------------------------------------------------------------------------

def test_get_profile_known_slug():
    p = get_profile("jetson-nano")
    assert p is JETSON_NANO


def test_get_profile_unknown_slug_raises_key_error():
    with pytest.raises(KeyError, match="Unknown hardware profile"):
        get_profile("nonexistent-device")


def test_get_profile_error_message_lists_available():
    with pytest.raises(KeyError) as exc_info:
        get_profile("invalid")
    assert "jetson-nano" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Profile field constraints
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("slug", list(PROFILES))
def test_tdp_positive(slug):
    assert PROFILES[slug].tdp_watts > 0


@pytest.mark.parametrize("slug", list(PROFILES))
def test_memory_limit_positive(slug):
    assert PROFILES[slug].memory_limit_mb > 0


@pytest.mark.parametrize("slug", list(PROFILES))
def test_target_fps_positive(slug):
    assert PROFILES[slug].target_fps > 0


@pytest.mark.parametrize("slug", list(PROFILES))
def test_cpu_cores_positive(slug):
    assert PROFILES[slug].cpu_cores >= 1


@pytest.mark.parametrize("slug", list(PROFILES))
def test_name_non_empty(slug):
    assert PROFILES[slug].name.strip() != ""


# ---------------------------------------------------------------------------
# Ordering sanity: resource-constrained devices have stricter specs
# ---------------------------------------------------------------------------

def test_rpi_zero_has_less_memory_than_rpi4():
    assert RASPBERRY_PI_ZERO_2W.memory_limit_mb < RASPBERRY_PI_4.memory_limit_mb


def test_laptop_has_more_memory_than_jetson_nano():
    assert LAPTOP_CPU.memory_limit_mb > JETSON_NANO.memory_limit_mb


def test_desktop_has_higher_tdp_than_laptop():
    assert DESKTOP_GPU.tdp_watts > LAPTOP_CPU.tdp_watts


def test_rpi_zero_has_lower_fps_target_than_rpi4():
    assert RASPBERRY_PI_ZERO_2W.target_fps < RASPBERRY_PI_4.target_fps


# ---------------------------------------------------------------------------
# Immutability (frozen dataclass)
# ---------------------------------------------------------------------------

def test_profiles_are_immutable():
    p = get_profile("rpi4")
    with pytest.raises((AttributeError, TypeError)):
        p.target_fps = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# __str__ representation
# ---------------------------------------------------------------------------

def test_str_contains_name():
    p = get_profile("laptop")
    s = str(p)
    assert "Laptop" in s


def test_str_contains_fps():
    p = get_profile("jetson-nano")
    s = str(p)
    assert "30" in s

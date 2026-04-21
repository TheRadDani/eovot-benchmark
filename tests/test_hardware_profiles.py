"""Tests for HardwareProfile and PROFILES registry."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from eovot.profiling.hardware_profiles import HardwareProfile, PROFILES


# ---------------------------------------------------------------------------
# HardwareProfile construction
# ---------------------------------------------------------------------------

def test_profile_creation():
    p = HardwareProfile(name="Test", tdp_watts=10.0, memory_limit_mb=2048.0, target_fps=25.0)
    assert p.name == "Test"
    assert p.tdp_watts == 10.0
    assert p.memory_limit_mb == 2048.0
    assert p.target_fps == 25.0
    assert p.description == ""


def test_profile_with_description():
    p = HardwareProfile(
        name="Custom", tdp_watts=5.0, memory_limit_mb=1024.0, target_fps=15.0,
        description="A custom device",
    )
    assert p.description == "A custom device"


def test_profile_validation_tdp_zero():
    with pytest.raises(ValueError, match="tdp_watts"):
        HardwareProfile(name="Bad", tdp_watts=0.0, memory_limit_mb=1024.0, target_fps=10.0)


def test_profile_validation_tdp_negative():
    with pytest.raises(ValueError, match="tdp_watts"):
        HardwareProfile(name="Bad", tdp_watts=-5.0, memory_limit_mb=1024.0, target_fps=10.0)


def test_profile_validation_memory_zero():
    with pytest.raises(ValueError, match="memory_limit_mb"):
        HardwareProfile(name="Bad", tdp_watts=10.0, memory_limit_mb=0.0, target_fps=10.0)


def test_profile_validation_fps_zero():
    with pytest.raises(ValueError, match="target_fps"):
        HardwareProfile(name="Bad", tdp_watts=10.0, memory_limit_mb=1024.0, target_fps=0.0)


# ---------------------------------------------------------------------------
# PROFILES registry
# ---------------------------------------------------------------------------

def test_profiles_registry_keys():
    expected = {"raspberry_pi4", "jetson_nano", "jetson_orin_nano", "x86_laptop", "x86_desktop"}
    assert expected.issubset(set(PROFILES.keys()))


def test_raspberry_pi4_values():
    p = PROFILES["raspberry_pi4"]
    assert p.tdp_watts == pytest.approx(6.0)
    assert p.memory_limit_mb == pytest.approx(4096.0)
    assert p.target_fps == pytest.approx(10.0)


def test_jetson_nano_values():
    p = PROFILES["jetson_nano"]
    assert p.tdp_watts == pytest.approx(10.0)
    assert p.target_fps == pytest.approx(20.0)


def test_all_profiles_valid():
    for key, profile in PROFILES.items():
        assert profile.tdp_watts > 0, f"{key}: tdp_watts must be positive"
        assert profile.memory_limit_mb > 0, f"{key}: memory_limit_mb must be positive"
        assert profile.target_fps > 0, f"{key}: target_fps must be positive"


# ---------------------------------------------------------------------------
# Serialisation round-trips
# ---------------------------------------------------------------------------

def test_to_dict_round_trip():
    p = PROFILES["jetson_nano"]
    d = p.to_dict()
    assert d["name"] == p.name
    assert d["tdp_watts"] == p.tdp_watts
    assert d["memory_limit_mb"] == p.memory_limit_mb
    assert d["target_fps"] == p.target_fps
    assert "description" in d


def test_from_dict():
    d = {
        "name": "Custom Device",
        "tdp_watts": 7.5,
        "memory_limit_mb": 3000.0,
        "target_fps": 15.0,
        "description": "A custom edge device",
    }
    p = HardwareProfile.from_dict(d)
    assert p.name == "Custom Device"
    assert p.tdp_watts == pytest.approx(7.5)
    assert p.description == "A custom edge device"


def test_from_dict_without_description():
    d = {"name": "Min", "tdp_watts": 5.0, "memory_limit_mb": 512.0, "target_fps": 5.0}
    p = HardwareProfile.from_dict(d)
    assert p.description == ""


def test_from_yaml():
    data = {
        "name": "YAML Device",
        "tdp_watts": 12.0,
        "memory_limit_mb": 8192.0,
        "target_fps": 25.0,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
        yaml.dump(data, fh)
        path = fh.name

    p = HardwareProfile.from_yaml(path)
    assert p.name == "YAML Device"
    assert p.tdp_watts == pytest.approx(12.0)
    assert p.target_fps == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# String representation
# ---------------------------------------------------------------------------

def test_str_contains_name():
    p = PROFILES["raspberry_pi4"]
    assert "Raspberry Pi 4" in str(p)


def test_str_contains_tdp():
    p = PROFILES["jetson_nano"]
    s = str(p)
    assert "10.0" in s


def test_repr_is_str():
    p = PROFILES["x86_laptop"]
    assert isinstance(str(p), str)

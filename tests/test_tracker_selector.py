"""Unit tests for the hardware-aware tracker selection module."""

from __future__ import annotations

import pytest

from eovot.selection.hardware_profile import DeviceClass, HardwareProfile
from eovot.selection.tracker_selector import (
    TrackerConstraints,
    TrackerRecommendation,
    TrackerSelector,
)


# ---------------------------------------------------------------------------
# HardwareProfile tests
# ---------------------------------------------------------------------------

class TestDeviceClass:
    def test_typical_tdp_increases_with_class(self):
        assert (
            DeviceClass.MICROCONTROLLER.typical_tdp_watts
            < DeviceClass.EMBEDDED.typical_tdp_watts
            < DeviceClass.LAPTOP.typical_tdp_watts
            < DeviceClass.DESKTOP.typical_tdp_watts
        )

    def test_performance_factor_ordering(self):
        assert (
            DeviceClass.MICROCONTROLLER.performance_factor
            < DeviceClass.EMBEDDED.performance_factor
            < DeviceClass.LAPTOP.performance_factor
            < DeviceClass.DESKTOP.performance_factor
        )


class TestHardwareProfile:
    def test_detect_returns_profile(self):
        profile = HardwareProfile.detect()
        assert profile.cpu_count >= 1
        assert profile.total_ram_gb > 0
        assert profile.available_ram_gb >= 0
        assert profile.tdp_watts > 0
        assert profile.platform_tag != ""

    def test_detect_tdp_override(self):
        profile = HardwareProfile.detect(tdp_watts=42.0)
        assert profile.tdp_watts == 42.0

    def test_classify_microcontroller(self):
        cls = HardwareProfile._classify(cpu_count=2, total_ram_gb=2.0)
        assert cls == DeviceClass.MICROCONTROLLER

    def test_classify_embedded(self):
        # 4 CPUs, 6 GB RAM: exceeds MICROCONTROLLER (≤4 GB) → EMBEDDED
        cls = HardwareProfile._classify(cpu_count=4, total_ram_gb=6.0)
        assert cls == DeviceClass.EMBEDDED

    def test_classify_laptop(self):
        cls = HardwareProfile._classify(cpu_count=8, total_ram_gb=16.0)
        assert cls == DeviceClass.LAPTOP

    def test_classify_desktop(self):
        cls = HardwareProfile._classify(cpu_count=16, total_ram_gb=32.0)
        assert cls == DeviceClass.DESKTOP

    def test_from_dict_round_trip(self):
        profile = HardwareProfile.detect()
        d = profile.to_dict()
        restored = HardwareProfile.from_dict(d)
        assert restored.cpu_count == profile.cpu_count
        assert restored.device_class == profile.device_class
        assert restored.tdp_watts == profile.tdp_watts

    def test_str_representation(self):
        profile = HardwareProfile.detect()
        s = str(profile)
        assert "HardwareProfile" in s
        assert "tdp=" in s


# ---------------------------------------------------------------------------
# TrackerConstraints tests
# ---------------------------------------------------------------------------

class TestTrackerConstraints:
    def test_defaults(self):
        c = TrackerConstraints()
        assert c.target_fps == 1.0
        assert c.max_memory_mb is None
        assert c.accuracy_weight == 0.5

    def test_invalid_accuracy_weight(self):
        with pytest.raises(ValueError):
            TrackerConstraints(accuracy_weight=1.5)
        with pytest.raises(ValueError):
            TrackerConstraints(accuracy_weight=-0.1)

    def test_invalid_fps(self):
        with pytest.raises(ValueError):
            TrackerConstraints(target_fps=-5.0)


# ---------------------------------------------------------------------------
# TrackerSelector tests
# ---------------------------------------------------------------------------

@pytest.fixture
def laptop_profile() -> HardwareProfile:
    return HardwareProfile.from_dict({
        "cpu_count": 12,
        "total_ram_gb": 16.0,
        "available_ram_gb": 12.0,
        "device_class": "laptop",
        "tdp_watts": 15.0,
        "cpu_freq_mhz": 2600.0,
        "platform_tag": "Linux-x86_64",
    })


@pytest.fixture
def embedded_profile() -> HardwareProfile:
    return HardwareProfile.from_dict({
        "cpu_count": 4,
        "total_ram_gb": 4.0,
        "available_ram_gb": 3.0,
        "device_class": "embedded",
        "tdp_watts": 6.0,
        "cpu_freq_mhz": 1500.0,
        "platform_tag": "Linux-aarch64",
    })


class TestTrackerSelector:
    def test_rank_returns_list(self, laptop_profile):
        selector = TrackerSelector()
        recs = selector.rank(laptop_profile)
        assert isinstance(recs, list)
        assert len(recs) > 0

    def test_rank_sorted_best_first(self, laptop_profile):
        selector = TrackerSelector()
        recs = selector.rank(laptop_profile)
        scores = [r.score for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_rank_assigns_consecutive_ranks(self, laptop_profile):
        selector = TrackerSelector()
        recs = selector.rank(laptop_profile)
        assert [r.rank for r in recs] == list(range(1, len(recs) + 1))

    def test_fps_filter_excludes_slow_trackers(self, embedded_profile):
        selector = TrackerSelector()
        # On embedded (factor ≈ 0.35), CSRT baseline=45 → est ≈ 15.75 FPS.
        # Setting target to 20 FPS should exclude CSRT and MIL.
        constraints = TrackerConstraints(target_fps=20.0)
        recs = selector.rank(embedded_profile, constraints)
        for r in recs:
            assert r.estimated_fps >= 20.0

    def test_memory_filter(self, laptop_profile):
        selector = TrackerSelector()
        constraints = TrackerConstraints(max_memory_mb=50.0)
        recs = selector.rank(laptop_profile, constraints)
        for r in recs:
            assert r.peak_memory_mb <= 50.0

    def test_accuracy_weight_one_picks_most_accurate(self, laptop_profile):
        selector = TrackerSelector()
        constraints = TrackerConstraints(accuracy_weight=1.0)
        recs = selector.rank(laptop_profile, constraints)
        assert recs[0].tracker_name == "CSRT"

    def test_accuracy_weight_zero_picks_fastest(self, laptop_profile):
        selector = TrackerSelector()
        constraints = TrackerConstraints(accuracy_weight=0.0)
        recs = selector.rank(laptop_profile, constraints)
        assert recs[0].tracker_name == "MOSSE"

    def test_empty_result_when_no_tracker_qualifies(self, embedded_profile):
        selector = TrackerSelector()
        # No tracker can do 10000 FPS on any device.
        constraints = TrackerConstraints(target_fps=10_000.0)
        recs = selector.rank(embedded_profile, constraints)
        assert recs == []

    def test_recommend_returns_best(self, laptop_profile):
        selector = TrackerSelector()
        rec = selector.recommend(laptop_profile, accuracy_weight=1.0)
        assert rec is not None
        assert rec.rank == 1

    def test_recommend_returns_none_when_no_match(self, embedded_profile):
        selector = TrackerSelector()
        rec = selector.recommend(embedded_profile, target_fps=10_000.0)
        assert rec is None

    def test_summary_table_is_markdown(self, laptop_profile):
        selector = TrackerSelector()
        table = selector.summary_table(laptop_profile)
        assert "| Rank |" in table
        assert "| Tracker |" in table

    def test_recommendation_str(self, laptop_profile):
        selector = TrackerSelector()
        recs = selector.rank(laptop_profile)
        s = str(recs[0])
        assert "#" in s
        assert "FPS" in s
        assert "AUC" in s

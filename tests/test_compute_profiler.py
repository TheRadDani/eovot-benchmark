"""Tests for eovot.profiling.compute — FLOPs estimation and compute profiling."""

from __future__ import annotations

import math
import pytest

from eovot.profiling.compute import (
    ComputeProfile,
    ComputeProfiler,
    correlation_filter_flops,
    kcf_flops,
    mosse_flops,
    siamese_tracker_flops,
    _fft2d_flops,
)


# ── _fft2d_flops primitive ────────────────────────────────────────────────────


class TestFft2dFlops:
    def test_zero_dimensions_return_zero(self):
        assert _fft2d_flops(0, 64) == 0.0
        assert _fft2d_flops(64, 0) == 0.0

    def test_positive_output(self):
        val = _fft2d_flops(64, 64)
        assert val > 0.0

    def test_larger_patch_more_flops(self):
        small = _fft2d_flops(32, 32)
        large = _fft2d_flops(64, 64)
        assert large > small

    def test_square_patch_symmetry(self):
        assert _fft2d_flops(64, 32) == pytest.approx(_fft2d_flops(32, 64))

    def test_known_value(self):
        # For H=W=2: row FFTs = 2 * 5 * 2 * 1 = 20; col FFTs = 2 * 5 * 2 * 1 = 20 → 40
        assert _fft2d_flops(2, 2) == pytest.approx(40.0)


# ── mosse_flops ───────────────────────────────────────────────────────────────


class TestMosseFlops:
    def test_returns_compute_profile(self):
        profile = mosse_flops()
        assert isinstance(profile, ComputeProfile)

    def test_tracker_name(self):
        assert mosse_flops().tracker_name == "mosse"

    def test_patch_size_stored(self):
        profile = mosse_flops(patch_h=32, patch_w=48)
        assert profile.patch_size == (32, 48)

    def test_positive_flops(self):
        assert mosse_flops().flops_per_frame > 0.0

    def test_larger_patch_more_flops(self):
        small = mosse_flops(32, 32).flops_per_frame
        large = mosse_flops(128, 128).flops_per_frame
        assert large > small

    def test_mega_flops_property(self):
        profile = mosse_flops()
        assert profile.mega_flops == pytest.approx(profile.flops_per_frame / 1e6)

    def test_compute_note_not_empty(self):
        assert len(mosse_flops().compute_note) > 0

    def test_params_count_none(self):
        assert mosse_flops().params_count is None


# ── kcf_flops ─────────────────────────────────────────────────────────────────


class TestKcfFlops:
    def test_returns_compute_profile(self):
        assert isinstance(kcf_flops(), ComputeProfile)

    def test_tracker_name(self):
        assert kcf_flops().tracker_name == "kcf"

    def test_kcf_more_flops_than_mosse_same_patch(self):
        h, w = 64, 64
        assert kcf_flops(h, w).flops_per_frame > mosse_flops(h, w).flops_per_frame

    def test_more_features_more_flops(self):
        single = kcf_flops(64, 64, num_features=1).flops_per_frame
        multi = kcf_flops(64, 64, num_features=8).flops_per_frame
        assert multi > single

    def test_patch_size_stored(self):
        profile = kcf_flops(patch_h=48, patch_w=48)
        assert profile.patch_size == (48, 48)


# ── correlation_filter_flops ──────────────────────────────────────────────────


class TestCorrelationFilterFlops:
    def test_returns_compute_profile(self):
        profile = correlation_filter_flops("custom", 64, 64)
        assert isinstance(profile, ComputeProfile)

    def test_tracker_name_preserved(self):
        profile = correlation_filter_flops("my_tracker", 64, 64)
        assert profile.tracker_name == "my_tracker"

    def test_more_fft_passes_more_flops(self):
        two = correlation_filter_flops("t", 64, 64, num_fft_passes=2).flops_per_frame
        six = correlation_filter_flops("t", 64, 64, num_fft_passes=6).flops_per_frame
        assert six > two

    def test_more_channels_more_flops(self):
        one = correlation_filter_flops("t", 64, 64, feature_channels=1).flops_per_frame
        eight = correlation_filter_flops("t", 64, 64, feature_channels=8).flops_per_frame
        assert eight > one

    def test_positive_flops(self):
        assert correlation_filter_flops("t", 32, 32).flops_per_frame > 0.0


# ── siamese_tracker_flops ─────────────────────────────────────────────────────


class TestSiameseTrackerFlops:
    def test_returns_compute_profile(self):
        profile = siamese_tracker_flops("siamrpn", backbone_flops=1e8)
        assert isinstance(profile, ComputeProfile)

    def test_tracker_name_preserved(self):
        profile = siamese_tracker_flops("siamrpn", backbone_flops=1e8)
        assert profile.tracker_name == "siamrpn"

    def test_patch_size_is_siam_standard(self):
        profile = siamese_tracker_flops("siamrpn", backbone_flops=1e8)
        assert profile.patch_size == (127, 127)

    def test_larger_search_area_more_flops(self):
        small = siamese_tracker_flops("t", 1e8, search_area_factor=2.0).flops_per_frame
        large = siamese_tracker_flops("t", 1e8, search_area_factor=6.0).flops_per_frame
        assert large > small

    def test_more_anchors_more_flops(self):
        five = siamese_tracker_flops("t", 1e8, num_anchors=5).flops_per_frame
        fifteen = siamese_tracker_flops("t", 1e8, num_anchors=15).flops_per_frame
        assert fifteen > five

    def test_positive_flops(self):
        assert siamese_tracker_flops("t", 1e8).flops_per_frame > 0.0


# ── ComputeProfile dataclass ──────────────────────────────────────────────────


class TestComputeProfile:
    def _make(self, flops: float = 1e6) -> ComputeProfile:
        return ComputeProfile(
            tracker_name="test",
            patch_size=(64, 64),
            flops_per_frame=flops,
        )

    def test_mega_flops(self):
        p = self._make(5e6)
        assert p.mega_flops == pytest.approx(5.0)

    def test_giga_flops(self):
        p = self._make(2e9)
        assert p.giga_flops == pytest.approx(2.0)

    def test_throughput_gflops_per_sec(self):
        p = self._make(1e9)  # 1 GFLOPs per frame
        # At 10 FPS → 10 GFLOPs/s
        assert p.throughput_gflops_per_sec(10.0) == pytest.approx(10.0)

    def test_flops_per_pixel(self):
        p = self._make(64 * 64)  # 4096 FLOPs for a 64×64 patch → 1 per pixel
        assert p.flops_per_pixel() == pytest.approx(1.0)

    def test_to_dict_has_required_keys(self):
        d = self._make().to_dict()
        for key in ("tracker_name", "patch_size", "flops_per_frame",
                    "mega_flops_per_frame", "giga_flops_per_frame"):
            assert key in d

    def test_to_dict_values_correct(self):
        p = self._make(2e6)
        d = p.to_dict()
        assert d["tracker_name"] == "test"
        assert d["patch_size"] == [64, 64]
        assert d["mega_flops_per_frame"] == pytest.approx(2.0)

    def test_str_representation(self):
        p = self._make()
        s = str(p)
        assert "test" in s
        assert "MFLOPs" in s


# ── ComputeProfiler registry ──────────────────────────────────────────────────


class TestComputeProfiler:
    def setup_method(self):
        self.profiler = ComputeProfiler()

    # ── Built-in trackers ─────────────────────────────────────────────────────

    def test_profile_mosse(self):
        p = self.profiler.profile("mosse")
        assert p.tracker_name == "mosse"
        assert p.flops_per_frame > 0.0

    def test_profile_kcf(self):
        p = self.profiler.profile("kcf")
        assert p.tracker_name == "kcf"

    def test_profile_csrt(self):
        p = self.profiler.profile("csrt")
        assert p.tracker_name == "csrt"

    def test_profile_medianflow(self):
        p = self.profiler.profile("medianflow")
        assert p.tracker_name == "medianflow"

    def test_profile_mil(self):
        p = self.profiler.profile("mil")
        assert p.tracker_name == "mil"

    def test_case_insensitive_lookup(self):
        p1 = self.profiler.profile("MOSSE")
        p2 = self.profiler.profile("mosse")
        assert p1.flops_per_frame == p2.flops_per_frame

    def test_unknown_tracker_returns_generic_estimate(self):
        p = self.profiler.profile("unknown_tracker_xyz", patch_size=(64, 64))
        assert p.tracker_name == "unknown_tracker_xyz"
        assert p.flops_per_frame > 0.0

    # ── Custom registration ───────────────────────────────────────────────────

    def test_register_custom_profile(self):
        custom = ComputeProfile("custom_dl", (224, 224), flops_per_frame=50e9)
        self.profiler.register("custom_dl", custom)
        retrieved = self.profiler.profile("custom_dl")
        assert retrieved.flops_per_frame == pytest.approx(50e9)

    def test_register_overrides_builtin(self):
        override = ComputeProfile("mosse", (64, 64), flops_per_frame=1.0)
        self.profiler.register("mosse", override)
        p = self.profiler.profile("mosse")
        assert p.flops_per_frame == pytest.approx(1.0)

    # ── patch_size parameter ──────────────────────────────────────────────────

    def test_larger_patch_more_flops_for_mosse(self):
        small = self.profiler.profile("mosse", patch_size=(32, 32))
        large = self.profiler.profile("mosse", patch_size=(128, 128))
        assert large.flops_per_frame > small.flops_per_frame

    def test_patch_size_stored_correctly(self):
        p = self.profiler.profile("kcf", patch_size=(48, 96))
        assert p.patch_size == (48, 96)

    # ── all_builtin_profiles ──────────────────────────────────────────────────

    def test_all_builtin_profiles_count(self):
        profiles = self.profiler.all_builtin_profiles()
        assert len(profiles) == 5  # mosse, kcf, csrt, medianflow, mil

    def test_all_builtin_profiles_sorted_ascending(self):
        profiles = self.profiler.all_builtin_profiles()
        flops = [p.flops_per_frame for p in profiles]
        assert flops == sorted(flops)

    # ── comparison_table ──────────────────────────────────────────────────────

    def test_comparison_table_returns_string(self):
        table = self.profiler.comparison_table()
        assert isinstance(table, str)
        assert "MFLOPs" in table

    def test_comparison_table_contains_all_builtins(self):
        table = self.profiler.comparison_table()
        for name in ("mosse", "kcf", "csrt", "medianflow", "mil"):
            assert name in table

    def test_comparison_table_with_fps_column(self):
        table = self.profiler.comparison_table(show_fps_at=200.0)
        assert "GFLOPs/s" in table

    def test_comparison_table_custom_subset(self):
        table = self.profiler.comparison_table(["mosse", "kcf"])
        assert "mosse" in table
        assert "kcf" in table
        assert "csrt" not in table

    # ── efficiency_rank ───────────────────────────────────────────────────────

    def test_efficiency_rank_returns_list(self):
        fps_map = {"mosse": 500.0, "kcf": 200.0}
        ranks = self.profiler.efficiency_rank(fps_map)
        assert isinstance(ranks, list)
        assert len(ranks) == 2

    def test_efficiency_rank_format(self):
        fps_map = {"mosse": 300.0}
        ranks = self.profiler.efficiency_rank(fps_map)
        name, fps, mf, eff = ranks[0]
        assert name == "mosse"
        assert fps == pytest.approx(300.0)
        assert mf > 0.0
        assert eff > 0.0

    def test_efficiency_rank_sorted_descending(self):
        # MOSSE has fewer FLOPs, so at equal FPS it has higher fps/MFLOPs
        fps_map = {"mosse": 300.0, "csrt": 300.0}
        ranks = self.profiler.efficiency_rank(fps_map)
        efficiencies = [r[3] for r in ranks]
        assert efficiencies == sorted(efficiencies, reverse=True)

    def test_efficiency_rank_empty_map(self):
        ranks = self.profiler.efficiency_rank({})
        assert ranks == []


# ── Ordering invariants across trackers ───────────────────────────────────────


class TestTrackerOrdering:
    """Sanity-check that analytical complexity ordering matches intuition."""

    def test_mosse_fewer_flops_than_kcf(self):
        h, w = 64, 64
        assert mosse_flops(h, w).flops_per_frame < kcf_flops(h, w).flops_per_frame

    def test_kcf_fewer_flops_than_csrt(self):
        profiler = ComputeProfiler()
        kcf = profiler.profile("kcf", patch_size=(64, 64)).flops_per_frame
        csrt = profiler.profile("csrt", patch_size=(64, 64)).flops_per_frame
        assert kcf < csrt

    def test_medianflow_fewer_flops_than_kcf(self):
        profiler = ComputeProfiler()
        mf = profiler.profile("medianflow", patch_size=(64, 64)).flops_per_frame
        kcf = profiler.profile("kcf", patch_size=(64, 64)).flops_per_frame
        assert mf < kcf

"""Tests for eovot.profiling.complexity."""

import math

import pytest

from eovot.profiling.complexity import (
    SUPPORTED_TRACKERS,
    ComplexityReport,
    TrackerComplexityAnalyzer,
    analyze_tracker_complexity,
)


# ---------------------------------------------------------------------------
# TrackerComplexityAnalyzer construction
# ---------------------------------------------------------------------------

class TestAnalyzerConstruction:
    def test_default_params(self):
        a = TrackerComplexityAnalyzer()
        assert a.patch_size == 64
        assert a.search_scale == 2.0
        assert a.float_bytes == 4

    def test_custom_patch_size(self):
        a = TrackerComplexityAnalyzer(patch_size=128)
        assert a.patch_size == 128

    def test_invalid_patch_size_raises(self):
        with pytest.raises(ValueError, match="patch_size"):
            TrackerComplexityAnalyzer(patch_size=0)

    def test_invalid_search_scale_raises(self):
        with pytest.raises(ValueError, match="search_scale"):
            TrackerComplexityAnalyzer(search_scale=-1.0)


# ---------------------------------------------------------------------------
# analyze() per tracker
# ---------------------------------------------------------------------------

class TestAnalyzeReturn:
    @pytest.fixture(autouse=True)
    def analyzer(self):
        self.analyzer = TrackerComplexityAnalyzer(patch_size=64)

    def test_returns_complexity_report(self):
        r = self.analyzer.analyze("MOSSE")
        assert isinstance(r, ComplexityReport)

    @pytest.mark.parametrize("name", SUPPORTED_TRACKERS)
    def test_all_trackers_return_report(self, name):
        r = self.analyzer.analyze(name)
        assert isinstance(r, ComplexityReport)
        assert r.tracker_name == name

    def test_unknown_tracker_raises_key_error(self):
        with pytest.raises(KeyError, match="No complexity model"):
            self.analyzer.analyze("SiamRPN")

    @pytest.mark.parametrize("name", SUPPORTED_TRACKERS)
    def test_param_count_positive(self, name):
        r = self.analyzer.analyze(name)
        assert r.param_count > 0

    @pytest.mark.parametrize("name", SUPPORTED_TRACKERS)
    def test_flops_per_frame_positive(self, name):
        r = self.analyzer.analyze(name)
        assert r.flops_per_frame > 0

    @pytest.mark.parametrize("name", SUPPORTED_TRACKERS)
    def test_model_size_positive(self, name):
        r = self.analyzer.analyze(name)
        assert r.model_size_mb > 0.0

    @pytest.mark.parametrize("name", SUPPORTED_TRACKERS)
    def test_patch_size_recorded(self, name):
        r = self.analyzer.analyze(name)
        assert r.patch_size == 64

    @pytest.mark.parametrize("name", SUPPORTED_TRACKERS)
    def test_mflops_property(self, name):
        r = self.analyzer.analyze(name)
        assert r.mflops == pytest.approx(r.flops_per_frame / 1_000_000.0)


# ---------------------------------------------------------------------------
# Analytical correctness checks
# ---------------------------------------------------------------------------

class TestAnalyticalValues:
    def setup_method(self):
        self.p = 64
        self.a = TrackerComplexityAnalyzer(patch_size=self.p)

    def _fft_flops(self, n):
        return int(5 * n * math.log2(n))

    def test_mosse_flops_formula(self):
        n = self.p ** 2
        expected = 3 * self._fft_flops(n) + n
        r = self.a.analyze("MOSSE")
        assert r.flops_per_frame == expected

    def test_mosse_params_four_real_arrays(self):
        n = self.p ** 2
        r = self.a.analyze("MOSSE")
        assert r.param_count == 4 * n

    def test_kcf_more_flops_than_mosse(self):
        mosse = self.a.analyze("MOSSE")
        kcf = self.a.analyze("KCF")
        assert kcf.flops_per_frame > mosse.flops_per_frame

    def test_csrt_more_flops_than_kcf(self):
        kcf = self.a.analyze("KCF")
        csrt = self.a.analyze("CSRT")
        assert csrt.flops_per_frame > kcf.flops_per_frame

    def test_csrt_more_params_than_kcf(self):
        kcf = self.a.analyze("KCF")
        csrt = self.a.analyze("CSRT")
        assert csrt.param_count > kcf.param_count

    def test_mosse_smallest_model_size(self):
        reports = self.a.analyze_all()
        mosse_size = reports["MOSSE"].model_size_mb
        for name, r in reports.items():
            if name in ("KCF", "CSRT"):
                assert r.model_size_mb > mosse_size, (
                    f"{name} should have larger model size than MOSSE"
                )

    def test_model_size_consistent_with_params(self):
        for name in SUPPORTED_TRACKERS:
            r = self.a.analyze(name)
            expected_mb = r.param_count * 4 / (1024 ** 2)
            assert r.model_size_mb == pytest.approx(expected_mb, rel=1e-5)


# ---------------------------------------------------------------------------
# Patch-size scaling
# ---------------------------------------------------------------------------

class TestPatchSizeScaling:
    def test_larger_patch_increases_flops(self):
        a32 = TrackerComplexityAnalyzer(patch_size=32)
        a128 = TrackerComplexityAnalyzer(patch_size=128)
        for name in ("MOSSE", "KCF", "CSRT"):
            assert a128.analyze(name).flops_per_frame > a32.analyze(name).flops_per_frame

    def test_larger_patch_increases_model_size(self):
        a32 = TrackerComplexityAnalyzer(patch_size=32)
        a128 = TrackerComplexityAnalyzer(patch_size=128)
        for name in ("MOSSE", "KCF", "CSRT"):
            assert a128.analyze(name).model_size_mb > a32.analyze(name).model_size_mb

    def test_median_flow_patch_size_independent_in_params(self):
        # MedianFlow params = num_points × 2 (point coords), patch-size agnostic
        a32 = TrackerComplexityAnalyzer(patch_size=32)
        a128 = TrackerComplexityAnalyzer(patch_size=128)
        assert a32.analyze("MedianFlow").param_count == a128.analyze("MedianFlow").param_count


# ---------------------------------------------------------------------------
# analyze_all()
# ---------------------------------------------------------------------------

class TestAnalyzeAll:
    def test_returns_all_trackers(self):
        a = TrackerComplexityAnalyzer()
        reports = a.analyze_all()
        assert set(reports.keys()) == set(SUPPORTED_TRACKERS)

    def test_order_matches_supported_trackers(self):
        a = TrackerComplexityAnalyzer()
        assert list(a.analyze_all().keys()) == SUPPORTED_TRACKERS


# ---------------------------------------------------------------------------
# ComplexityReport serialisation
# ---------------------------------------------------------------------------

class TestComplexityReportSerialization:
    def setup_method(self):
        self.r = TrackerComplexityAnalyzer().analyze("KCF")

    def test_to_dict_has_required_keys(self):
        d = self.r.to_dict()
        required = {
            "tracker_name", "patch_size", "param_count",
            "flops_per_frame", "mflops_per_frame", "model_size_mb", "notes",
        }
        assert required.issubset(d.keys())

    def test_str_contains_tracker_name(self):
        assert "KCF" in str(self.r)

    def test_str_contains_mflops(self):
        assert "MFLOPs" in str(self.r)

    def test_mflops_in_dict_matches_property(self):
        d = self.r.to_dict()
        assert d["mflops_per_frame"] == pytest.approx(self.r.mflops, rel=1e-4)


# ---------------------------------------------------------------------------
# analyze_tracker_complexity() convenience function
# ---------------------------------------------------------------------------

class TestConvenienceFunction:
    def test_returns_report(self):
        r = analyze_tracker_complexity("MOSSE")
        assert isinstance(r, ComplexityReport)
        assert r.tracker_name == "MOSSE"

    def test_patch_size_forwarded(self):
        r = analyze_tracker_complexity("KCF", patch_size=32)
        assert r.patch_size == 32

    def test_result_matches_class_method(self):
        r_fn = analyze_tracker_complexity("CSRT", patch_size=48)
        r_cls = TrackerComplexityAnalyzer(patch_size=48).analyze("CSRT")
        assert r_fn.flops_per_frame == r_cls.flops_per_frame
        assert r_fn.param_count == r_cls.param_count


# ---------------------------------------------------------------------------
# compare_table() string output
# ---------------------------------------------------------------------------

class TestCompareTable:
    def test_table_contains_all_tracker_names(self):
        table = TrackerComplexityAnalyzer().compare_table()
        for name in SUPPORTED_TRACKERS:
            assert name in table

    def test_table_contains_header_columns(self):
        table = TrackerComplexityAnalyzer().compare_table()
        assert "Tracker" in table
        assert "Params" in table
        assert "MFLOPs" in table
        assert "Size" in table

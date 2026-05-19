"""Unit and integration tests for eovot.metrics.temporal."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.metrics.temporal import TemporalConsistencyAnalyzer, TemporalConsistencyResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analyzer():
    return TemporalConsistencyAnalyzer()


def _constant_preds(n: int, box=(10.0, 10.0, 50.0, 50.0)) -> np.ndarray:
    """N identical bounding boxes — perfectly smooth, zero jitter."""
    return np.tile(np.array(box, dtype=np.float64), (n, 1))


def _linear_preds(n: int, step: float = 2.0) -> np.ndarray:
    """N boxes moving right at constant velocity — smooth, non-zero velocity."""
    preds = np.empty((n, 4), dtype=np.float64)
    for i in range(n):
        preds[i] = [i * step, 10.0, 50.0, 50.0]
    return preds


def _jittery_preds(n: int, noise: float = 15.0, seed: int = 0) -> np.ndarray:
    """N boxes with large random position noise — high jitter."""
    rng = np.random.default_rng(seed)
    base = np.tile([50.0, 50.0, 40.0, 40.0], (n, 1)).astype(np.float64)
    base[:, :2] += rng.normal(0, noise, size=(n, 2))
    return base


# ---------------------------------------------------------------------------
# TemporalConsistencyAnalyzer — constructor
# ---------------------------------------------------------------------------

class TestAnalyzerInit:
    def test_default_threshold(self):
        a = TemporalConsistencyAnalyzer()
        assert a.outlier_threshold == 3.0

    def test_custom_threshold(self):
        a = TemporalConsistencyAnalyzer(outlier_threshold=2.0)
        assert a.outlier_threshold == 2.0

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            TemporalConsistencyAnalyzer(outlier_threshold=0.0)

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            TemporalConsistencyAnalyzer(outlier_threshold=-1.0)


# ---------------------------------------------------------------------------
# compute_position_jitter
# ---------------------------------------------------------------------------

class TestPositionJitter:
    def setup_method(self):
        self.analyzer = TemporalConsistencyAnalyzer()

    def test_too_few_frames_returns_zero(self):
        preds = _constant_preds(1)
        jitter, vel = self.analyzer.compute_position_jitter(preds)
        assert jitter == pytest.approx(0.0)
        assert vel == pytest.approx(0.0)

    def test_constant_predictions_zero_jitter(self):
        preds = _constant_preds(30)
        jitter, vel = self.analyzer.compute_position_jitter(preds)
        assert jitter == pytest.approx(0.0, abs=1e-9)
        assert vel == pytest.approx(0.0, abs=1e-9)

    def test_linear_motion_low_jitter(self):
        preds = _linear_preds(50, step=2.0)
        jitter, vel = self.analyzer.compute_position_jitter(preds)
        # Perfectly constant velocity → std of speeds = 0 → jitter = 0
        assert jitter == pytest.approx(0.0, abs=1e-9)
        assert vel == pytest.approx(2.0, rel=1e-3)

    def test_jittery_predictions_high_jitter(self):
        preds = _jittery_preds(50, noise=20.0)
        jitter, vel = self.analyzer.compute_position_jitter(preds)
        assert jitter > 0.05  # significantly above zero

    def test_higher_noise_higher_jitter(self):
        preds_low = _jittery_preds(50, noise=2.0, seed=1)
        preds_high = _jittery_preds(50, noise=20.0, seed=1)
        j_low, _ = self.analyzer.compute_position_jitter(preds_low)
        j_high, _ = self.analyzer.compute_position_jitter(preds_high)
        assert j_high > j_low

    def test_velocity_scales_with_step(self):
        for step in [1.0, 3.0, 7.0]:
            preds = _linear_preds(30, step=step)
            _, vel = self.analyzer.compute_position_jitter(preds)
            assert vel == pytest.approx(step, rel=1e-3)

    def test_non_negative_jitter(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            preds = rng.uniform(0, 100, (40, 4))
            preds[:, 2:] = np.abs(preds[:, 2:]) + 5
            jitter, _ = self.analyzer.compute_position_jitter(preds)
            assert jitter >= 0.0


# ---------------------------------------------------------------------------
# compute_scale_jitter
# ---------------------------------------------------------------------------

class TestScaleJitter:
    def setup_method(self):
        self.analyzer = TemporalConsistencyAnalyzer()

    def test_too_few_frames_returns_zero(self):
        assert self.analyzer.compute_scale_jitter(_constant_preds(1)) == pytest.approx(0.0)

    def test_constant_area_zero_jitter(self):
        preds = _constant_preds(30)
        assert self.analyzer.compute_scale_jitter(preds) == pytest.approx(0.0, abs=1e-9)

    def test_linear_motion_constant_size_zero_jitter(self):
        preds = _linear_preds(30)
        # All boxes have identical w=50, h=50 → constant area → jitter = 0
        assert self.analyzer.compute_scale_jitter(preds) == pytest.approx(0.0, abs=1e-9)

    def test_alternating_scale_high_jitter(self):
        n = 40
        preds = np.zeros((n, 4), dtype=np.float64)
        preds[:, 2:] = np.where(
            np.arange(n)[:, None] % 2 == 0,
            [[20.0, 20.0]],
            [[60.0, 60.0]],
        )
        jitter = self.analyzer.compute_scale_jitter(preds)
        assert jitter > 0.5  # ratios alternate between 9 and 1/9

    def test_non_negative(self):
        rng = np.random.default_rng(7)
        for _ in range(10):
            preds = rng.uniform(5, 100, (30, 4))
            assert self.analyzer.compute_scale_jitter(preds) >= 0.0


# ---------------------------------------------------------------------------
# compute_velocity_outlier_ratio
# ---------------------------------------------------------------------------

class TestVelocityOutlierRatio:
    def setup_method(self):
        self.analyzer = TemporalConsistencyAnalyzer()

    def test_too_few_frames_returns_zero(self):
        assert self.analyzer.compute_velocity_outlier_ratio(_constant_preds(2)) == pytest.approx(0.0)

    def test_constant_predictions_zero_vor(self):
        preds = _constant_preds(50)
        assert self.analyzer.compute_velocity_outlier_ratio(preds) == pytest.approx(0.0)

    def test_linear_motion_zero_vor(self):
        preds = _linear_preds(50)
        assert self.analyzer.compute_velocity_outlier_ratio(preds) == pytest.approx(0.0)

    def test_vor_bounded_in_01(self):
        rng = np.random.default_rng(13)
        for _ in range(10):
            preds = rng.uniform(0, 200, (50, 4))
            preds[:, 2:] = np.abs(preds[:, 2:]) + 5
            vor = self.analyzer.compute_velocity_outlier_ratio(preds)
            assert 0.0 <= vor <= 1.0

    def test_sudden_jump_creates_outlier(self):
        """Insert a large single-frame displacement; VOR should be > 0."""
        preds = _linear_preds(60, step=1.0)
        # Inject a big displacement jump at frame 30
        preds[30, 0] += 200.0
        vor = self.analyzer.compute_velocity_outlier_ratio(preds)
        assert vor > 0.0


# ---------------------------------------------------------------------------
# analyze — single sequence
# ---------------------------------------------------------------------------

class TestAnalyzeSingleSequence:
    def setup_method(self):
        self.analyzer = TemporalConsistencyAnalyzer()

    def test_returns_temporal_result(self):
        preds = _linear_preds(30)
        result = self.analyzer.analyze(preds, "T", "S")
        assert isinstance(result, TemporalConsistencyResult)

    def test_tracker_name_propagated(self):
        result = self.analyzer.analyze(_linear_preds(20), tracker_name="MOSSE")
        assert result.tracker_name == "MOSSE"

    def test_sequence_name_propagated(self):
        result = self.analyzer.analyze(_linear_preds(20), sequence_name="car1")
        assert result.sequence_name == "car1"

    def test_num_frames_correct(self):
        result = self.analyzer.analyze(_linear_preds(35))
        assert result.num_frames == 35

    def test_perfect_smooth_tracker_high_score(self):
        """Constant-velocity predictions → smoothness near 1."""
        preds = _linear_preds(50, step=1.0)
        result = self.analyzer.analyze(preds)
        assert result.smoothness_score > 0.9

    def test_jittery_tracker_lower_score(self):
        smooth = self.analyzer.analyze(_linear_preds(50, step=1.0))
        noisy = self.analyzer.analyze(_jittery_preds(50, noise=25.0))
        assert smooth.smoothness_score > noisy.smoothness_score

    def test_smoothness_score_in_range(self):
        for preds in [_constant_preds(30), _linear_preds(30), _jittery_preds(30)]:
            r = self.analyzer.analyze(preds)
            assert 0.0 < r.smoothness_score <= 1.0

    def test_single_frame_returns_default(self):
        preds = _constant_preds(1)
        result = self.analyzer.analyze(preds)
        assert result.smoothness_score == pytest.approx(1.0)
        assert result.position_jitter == pytest.approx(0.0)

    def test_to_dict_keys(self):
        result = self.analyzer.analyze(_linear_preds(20))
        d = result.to_dict()
        for key in ("tracker_name", "sequence_name", "position_jitter",
                    "scale_jitter", "velocity_outlier_ratio",
                    "smoothness_score", "mean_velocity_px", "num_frames"):
            assert key in d

    def test_smoothness_formula(self):
        """smoothness_score should equal 1/(1 + pos_jitter + scale_jitter + VOR)."""
        preds = _jittery_preds(40, noise=5.0)
        r = self.analyzer.analyze(preds)
        expected = 1.0 / (1.0 + r.position_jitter + r.scale_jitter + r.velocity_outlier_ratio)
        assert r.smoothness_score == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# analyze_benchmark — multi-sequence aggregation
# ---------------------------------------------------------------------------

class TestAnalyzeBenchmark:
    def setup_method(self):
        self.analyzer = TemporalConsistencyAnalyzer()

    def test_empty_input_returns_empty_dicts(self):
        result = self.analyzer.analyze_benchmark({})
        assert result["per_sequence"] == {}
        assert result["aggregate"] == {}

    def test_returns_per_sequence_results(self):
        seq_preds = {
            "seq0": _linear_preds(30),
            "seq1": _jittery_preds(30),
        }
        result = self.analyzer.analyze_benchmark(seq_preds, "MOSSE")
        assert "seq0" in result["per_sequence"]
        assert "seq1" in result["per_sequence"]

    def test_aggregate_num_sequences(self):
        seq_preds = {f"seq{i}": _linear_preds(20) for i in range(5)}
        agg = self.analyzer.analyze_benchmark(seq_preds)["aggregate"]
        assert agg["num_sequences"] == 5

    def test_aggregate_keys_present(self):
        seq_preds = {"s1": _linear_preds(20), "s2": _constant_preds(20)}
        agg = self.analyzer.analyze_benchmark(seq_preds, "KCF")["aggregate"]
        for key in ("tracker_name", "num_sequences", "mean_position_jitter",
                    "mean_scale_jitter", "mean_velocity_outlier_ratio",
                    "mean_smoothness_score", "mean_velocity_px"):
            assert key in agg, f"Missing key: {key}"

    def test_aggregate_tracker_name(self):
        seq_preds = {"s": _linear_preds(15)}
        agg = self.analyzer.analyze_benchmark(seq_preds, "CSRT")["aggregate"]
        assert agg["tracker_name"] == "CSRT"

    def test_aggregate_smoothness_in_range(self):
        seq_preds = {f"s{i}": _jittery_preds(30, seed=i) for i in range(4)}
        agg = self.analyzer.analyze_benchmark(seq_preds)["aggregate"]
        assert 0.0 < agg["mean_smoothness_score"] <= 1.0

    def test_smooth_tracker_higher_aggregate_score(self):
        smooth_preds = {f"s{i}": _linear_preds(30, step=1.0) for i in range(3)}
        noisy_preds = {f"s{i}": _jittery_preds(30, noise=20.0, seed=i) for i in range(3)}
        smooth_agg = self.analyzer.analyze_benchmark(smooth_preds)["aggregate"]
        noisy_agg = self.analyzer.analyze_benchmark(noisy_preds)["aggregate"]
        assert smooth_agg["mean_smoothness_score"] > noisy_agg["mean_smoothness_score"]


# ---------------------------------------------------------------------------
# Integration with BenchmarkEngine
# ---------------------------------------------------------------------------

class TestIntegrationWithBenchmarkEngine:
    def test_predictions_from_engine_are_compatible(self):
        """TemporalConsistencyAnalyzer works on predictions stored by BenchmarkEngine."""
        import numpy as np
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.base import BaseDataset, Sequence
        from eovot.trackers.base import BaseTracker
        from typing import Iterator

        class FixedTracker(BaseTracker):
            def __init__(self):
                super().__init__("Fixed")
            def initialize(self, frame, bbox): pass
            def update(self, frame):
                return (10.0, 10.0, 40.0, 40.0)

        class TinySeq(Sequence):
            def __init__(self):
                gt = np.tile([10.0, 10.0, 40.0, 40.0], (20, 1))
                super().__init__("s", ["f"] * 20, gt)
            def __iter__(self) -> Iterator[np.ndarray]:
                for _ in range(20):
                    yield np.zeros((120, 160, 3), dtype=np.uint8)

        class TinyDataset(BaseDataset):
            def __len__(self): return 1
            def __getitem__(self, idx): return TinySeq()

        engine = BenchmarkEngine(verbose=False)
        result = engine.run(FixedTracker(), TinyDataset(), "tiny")
        analyzer = TemporalConsistencyAnalyzer()

        for sr in result.sequence_results:
            assert sr.predictions is not None
            tc_result = analyzer.analyze(sr.predictions, "Fixed", sr.sequence_name)
            assert isinstance(tc_result, TemporalConsistencyResult)
            # A fixed-box tracker has zero position jitter
            assert tc_result.position_jitter == pytest.approx(0.0, abs=1e-9)
            assert tc_result.smoothness_score == pytest.approx(1.0, rel=1e-6)

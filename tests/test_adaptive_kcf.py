"""Tests for AdaptiveKCFTracker and ConfidenceAnalyzer."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.adaptive_kcf import AdaptiveKCFTracker
from eovot.metrics.confidence import ConfidenceAnalyzer, ConfidenceResult


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _solid_frame(
    h: int = 120,
    w: int = 160,
    seed: int = 42,
) -> np.ndarray:
    """Reproducible random BGR frame."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _linear_sequence(n: int = 30, seed: int = 7):
    """Frames with a brightly-coloured rectangle moving linearly."""
    rng = np.random.default_rng(seed)
    H, W = 120, 160
    bw, bh = 30, 30
    colour = (220, 80, 60)
    bg = rng.integers(20, 80, (H, W, 3), dtype=np.uint8)
    frames, gts = [], []
    cx, cy = 60.0, 60.0
    vx, vy = 1.5, 1.0
    for _ in range(n):
        frame = bg.copy()
        x1 = max(0, int(cx - bw / 2))
        y1 = max(0, int(cy - bh / 2))
        x2 = min(W, x1 + bw)
        y2 = min(H, y1 + bh)
        frame[y1:y2, x1:x2] = colour
        frames.append(frame)
        gts.append((float(x1), float(y1), float(bw), float(bh)))
        cx = float(np.clip(cx + vx, bw / 2, W - bw / 2))
        cy = float(np.clip(cy + vy, bh / 2, H - bh / 2))
        if cx <= bw / 2 or cx >= W - bw / 2:
            vx = -vx
        if cy <= bh / 2 or cy >= H - bh / 2:
            vy = -vy
    return frames, gts


# ---------------------------------------------------------------------------
# AdaptiveKCFTracker
# ---------------------------------------------------------------------------

class TestAdaptiveKCFTracker:

    def test_default_attributes(self):
        t = AdaptiveKCFTracker()
        assert t.name == "AdaptiveKCF"
        assert t.min_psr == 7.0
        assert t.psr_window == 2

    def test_custom_attributes(self):
        t = AdaptiveKCFTracker(min_psr=5.0, psr_window=3)
        assert t.min_psr == 5.0
        assert t.psr_window == 3

    def test_initialize_and_update_returns_bbox(self):
        t = AdaptiveKCFTracker()
        frames, gts = _linear_sequence(20)
        t.initialize(frames[0], gts[0])
        for frame in frames[1:]:
            bbox = t.update(frame)
            assert len(bbox) == 4
            _, _, w, h = bbox
            assert w > 0 and h > 0

    def test_psr_history_grows_with_updates(self):
        t = AdaptiveKCFTracker()
        frames, gts = _linear_sequence(15)
        t.initialize(frames[0], gts[0])
        for frame in frames[1:]:
            t.update(frame)
        assert len(t.psr_history) == len(frames) - 1

    def test_psr_values_non_negative(self):
        t = AdaptiveKCFTracker()
        frames, gts = _linear_sequence(10)
        t.initialize(frames[0], gts[0])
        for frame in frames[1:]:
            t.update(frame)
        assert all(p >= 0.0 for p in t.psr_history)

    def test_last_psr_set_after_update(self):
        t = AdaptiveKCFTracker()
        frames, gts = _linear_sequence(5)
        t.initialize(frames[0], gts[0])
        t.update(frames[1])
        assert t.last_psr >= 0.0

    def test_psr_history_is_copy(self):
        """Mutating the returned list must not affect internal state."""
        t = AdaptiveKCFTracker()
        frames, gts = _linear_sequence(5)
        t.initialize(frames[0], gts[0])
        t.update(frames[1])
        history = t.psr_history
        history.append(999.0)
        assert 999.0 not in t.psr_history

    def test_freezes_when_min_psr_unreachably_high(self):
        """Setting min_psr=∞ must always freeze; consecutive updates return identical boxes."""
        t = AdaptiveKCFTracker(min_psr=1e9)
        frames, gts = _linear_sequence(5)
        t.initialize(frames[0], gts[0])
        first = t.update(frames[1])
        second = t.update(frames[2])
        assert first == second
        assert t.last_confident is False

    def test_confident_on_easy_sequence(self):
        """On a clean, easy sequence the tracker should be confident most frames."""
        t = AdaptiveKCFTracker(min_psr=4.0)
        frames, gts = _linear_sequence(20)
        t.initialize(frames[0], gts[0])
        confident_count = sum(
            1 for frame in frames[1:] if (t.update(frame), t.last_confident)[1]
        )
        assert confident_count >= len(frames) // 2

    def test_reset_clears_history(self):
        t = AdaptiveKCFTracker()
        frames, gts = _linear_sequence(5)
        t.initialize(frames[0], gts[0])
        t.update(frames[1])
        t.reset()
        assert t.psr_history == []
        assert t.last_psr == 0.0
        assert t.last_confident is True

    def test_raises_before_initialize(self):
        t = AdaptiveKCFTracker()
        with pytest.raises(RuntimeError):
            t.update(_solid_frame())

    def test_registered_in_tracker_registry(self):
        from eovot.trackers.registry import TRACKER_REGISTRY, build_tracker
        assert "AdaptiveKCF" in TRACKER_REGISTRY
        t = build_tracker("AdaptiveKCF")
        assert t.name == "AdaptiveKCF"

    def test_in_trackers_package(self):
        from eovot.trackers import AdaptiveKCFTracker as T  # noqa: F401

    def test_benchmark_engine_compatible(self):
        """AdaptiveKCF must run through BenchmarkEngine without error."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset

        dataset = SyntheticDataset(num_sequences=2, num_frames=20, motion="linear")
        engine = BenchmarkEngine(verbose=False)
        tracker = AdaptiveKCFTracker(min_psr=4.0)
        result = engine.run(tracker, dataset, dataset_name="Synthetic")
        assert result.tracker_name == "AdaptiveKCF"
        assert len(result.sequence_results) == 2


# ---------------------------------------------------------------------------
# ConfidenceAnalyzer
# ---------------------------------------------------------------------------

class TestConfidenceAnalyzer:

    def _correlated_data(self, n: int = 60, seed: int = 0):
        """Confidence and IoU with strong positive Pearson correlation."""
        rng = np.random.default_rng(seed)
        conf = rng.uniform(2.0, 20.0, n)
        ious = np.clip(conf / 20.0 + rng.normal(0, 0.05, n), 0.0, 1.0)
        return conf.tolist(), ious.tolist()

    def test_pearson_positive_on_correlated_data(self):
        analyzer = ConfidenceAnalyzer()
        conf, ious = self._correlated_data(100)
        result = analyzer.analyze(conf, ious)
        assert result.pearson_r > 0.5

    def test_auroc_in_valid_range(self):
        analyzer = ConfidenceAnalyzer()
        conf, ious = self._correlated_data(100)
        result = analyzer.analyze(conf, ious)
        assert 0.0 <= result.auroc <= 1.0

    def test_best_f1_in_valid_range(self):
        analyzer = ConfidenceAnalyzer()
        conf, ious = self._correlated_data(100)
        result = analyzer.analyze(conf, ious)
        assert 0.0 <= result.best_f1 <= 1.0

    def test_n_frames_matches_input(self):
        analyzer = ConfidenceAnalyzer()
        conf, ious = self._correlated_data(50)
        result = analyzer.analyze(conf, ious)
        assert result.n_frames == 50

    def test_truncates_to_shorter_array(self):
        analyzer = ConfidenceAnalyzer()
        conf = [5.0] * 10
        ious = [0.7] * 7
        result = analyzer.analyze(conf, ious)
        assert result.n_frames == 7

    def test_raises_on_fewer_than_two_frames(self):
        analyzer = ConfidenceAnalyzer()
        with pytest.raises(ValueError):
            analyzer.analyze([5.0], [0.8])

    def test_result_str_contains_key_fields(self):
        analyzer = ConfidenceAnalyzer()
        conf, ious = self._correlated_data(20)
        result = analyzer.analyze(conf, ious, tracker_name="KCF", sequence_name="seq1")
        s = str(result)
        assert "r=" in s and "AUROC=" in s and "best_thr=" in s

    def test_to_dict_keys(self):
        analyzer = ConfidenceAnalyzer()
        conf, ious = self._correlated_data(20)
        result = analyzer.analyze(conf, ious)
        d = result.to_dict()
        for key in ("pearson_r", "auroc", "best_threshold", "best_f1", "n_frames"):
            assert key in d

    def test_analyze_benchmark_aggregate(self):
        analyzer = ConfidenceAnalyzer()
        conf_by_seq = {
            "s1": [5.0, 8.0, 3.0, 10.0, 12.0],
            "s2": [7.0, 9.0, 6.0, 11.0, 4.0],
        }
        iou_by_seq = {
            "s1": [0.05, 0.7, 0.04, 0.8, 0.9],
            "s2": [0.6, 0.8, 0.5, 0.9, 0.1],
        }
        out = analyzer.analyze_benchmark(conf_by_seq, iou_by_seq, tracker_name="T")
        agg = out["aggregate"]
        assert agg["num_sequences"] == 2
        assert "mean_auroc" in agg
        assert "median_best_threshold" in agg

    def test_analyze_benchmark_ignores_missing_iou_sequences(self):
        analyzer = ConfidenceAnalyzer()
        conf_by_seq = {"s1": [5.0, 8.0, 3.0], "s2": [7.0, 9.0, 6.0]}
        iou_by_seq = {"s1": [0.05, 0.7, 0.04]}  # s2 missing
        out = analyzer.analyze_benchmark(conf_by_seq, iou_by_seq)
        assert out["aggregate"]["num_sequences"] == 1

    def test_invalid_failure_threshold_raises(self):
        with pytest.raises(ValueError):
            ConfidenceAnalyzer(failure_iou_threshold=1.5)

    def test_auroc_chance_when_no_failures(self):
        analyzer = ConfidenceAnalyzer()
        conf = [10.0] * 20
        ious = [0.9] * 20  # no failures
        result = analyzer.analyze(conf, ious)
        assert result.auroc == 0.5
        assert result.n_failures == 0

    def test_end_to_end_with_adaptive_kcf(self):
        """PSR history from AdaptiveKCFTracker feeds directly into ConfidenceAnalyzer."""
        from eovot.benchmark.engine import BenchmarkEngine, MetricsEngine
        from eovot.datasets.synthetic import SyntheticDataset

        dataset = SyntheticDataset(num_sequences=1, num_frames=30, motion="linear")
        tracker = AdaptiveKCFTracker(min_psr=3.0)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(tracker, dataset, dataset_name="Synth")

        ious = result.sequence_results[0].ious.tolist()
        psr = tracker.psr_history

        if len(psr) >= 2 and len(ious) >= 2:
            analyzer = ConfidenceAnalyzer()
            cr = analyzer.analyze(psr, ious[1:])  # skip init frame
            assert isinstance(cr, ConfidenceResult)
            assert cr.n_frames >= 1

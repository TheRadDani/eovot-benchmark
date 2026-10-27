"""Tests for the GOT-10k evaluation protocol (AO, SR0.50, SR0.75).

Covers:
- Per-sequence evaluation with perfect, shifted, and random predictions
- Dataset-level aggregation (sequence-equal weighting)
- BenchmarkResult.got10k_metrics() integration
- Edge cases: empty arrays, single frame, all-zero preds
- Markdown and LaTeX report generation
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pytest

from eovot.metrics.got10k_protocol import (
    GOT10kEvaluator,
    GOT10kSequenceResult,
    GOT10kSummary,
)
from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.base import BaseDataset, Sequence
from eovot.trackers.base import BaseTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BOX = (10.0, 10.0, 50.0, 50.0)
_SHIFTED = (40.0, 40.0, 50.0, 50.0)
_N = 40


def _make_boxes(box: tuple, n: int) -> np.ndarray:
    return np.tile(np.array(box, dtype=np.float64), (n, 1))


class ConstantTracker(BaseTracker):
    def __init__(self, box=_BOX):
        super().__init__(name="Const")
        self._box = box

    def initialize(self, frame, bbox):
        pass

    def update(self, frame):
        return self._box


class TinySeq(Sequence):
    def __init__(self, name="s0", box=_BOX, n=_N):
        gt = np.tile(np.array(box), (n, 1))
        super().__init__(name, [f"f{i}" for i in range(n)], gt)
        self._n = n

    def __iter__(self) -> Iterator[np.ndarray]:
        for _ in range(self._n):
            yield np.zeros((120, 160, 3), dtype=np.uint8)


class TinyDataset(BaseDataset):
    def __init__(self, n_seqs=3, box=_BOX, n=_N):
        self._seqs = [TinySeq(f"s{i}", box, n) for i in range(n_seqs)]

    def __len__(self):
        return len(self._seqs)

    def __getitem__(self, idx):
        return self._seqs[idx]


# ---------------------------------------------------------------------------
# GOT10kEvaluator unit tests
# ---------------------------------------------------------------------------

class TestGOT10kEvaluatorPerSequence:
    def setup_method(self):
        self.ev = GOT10kEvaluator()

    def test_perfect_tracker_ao_is_one(self):
        preds = _make_boxes(_BOX, _N)
        gts = _make_boxes(_BOX, _N)
        result = self.ev.evaluate_sequence(preds, gts, "perfect")
        assert abs(result.ao - 1.0) < 1e-6
        assert abs(result.sr50 - 1.0) < 1e-6
        assert abs(result.sr75 - 1.0) < 1e-6

    def test_perfect_tracker_skips_init_frame(self):
        """When preds[0] == gts[0] exactly, evaluator skips that frame."""
        preds = _make_boxes(_BOX, _N)
        gts = _make_boxes(_BOX, _N)
        result = self.ev.evaluate_sequence(preds, gts)
        assert result.num_frames == _N - 1

    def test_shifted_tracker_lower_ao(self):
        preds = _make_boxes(_SHIFTED, _N)
        gts = _make_boxes(_BOX, _N)
        result = self.ev.evaluate_sequence(preds, gts)
        assert result.ao < 1.0
        assert result.ao >= 0.0

    def test_zero_predictions_ao_zero(self):
        preds = np.zeros((_N, 4))
        gts = _make_boxes(_BOX, _N)
        result = self.ev.evaluate_sequence(preds, gts)
        assert result.ao == 0.0
        assert result.sr50 == 0.0
        assert result.sr75 == 0.0

    def test_empty_arrays_returns_zeros(self):
        result = self.ev.evaluate_sequence(
            np.zeros((0, 4)), np.zeros((0, 4))
        )
        assert result.ao == 0.0
        assert result.num_frames == 0

    def test_sr50_and_sr75_ordering(self):
        """SR0.75 ≤ SR0.50 always (stricter threshold)."""
        rng = np.random.default_rng(0)
        preds = _make_boxes(_BOX, _N) + rng.uniform(-5, 5, (_N, 4))
        gts = _make_boxes(_BOX, _N)
        result = self.ev.evaluate_sequence(preds, gts)
        assert result.sr75 <= result.sr50

    def test_mismatched_lengths_uses_min(self):
        preds = _make_boxes(_BOX, _N)
        gts = _make_boxes(_BOX, _N // 2)
        result = self.ev.evaluate_sequence(preds, gts)
        assert result.num_frames <= _N // 2


class TestGOT10kEvaluatorAggregate:
    def setup_method(self):
        self.ev = GOT10kEvaluator()

    def test_empty_list_returns_zeros(self):
        summary = self.ev.aggregate([])
        assert summary.ao == 0.0
        assert summary.num_sequences == 0

    def test_sequence_equal_weighting(self):
        """Two sequences of different lengths should contribute equally to AO."""
        seq_short = GOT10kSequenceResult("s1", ao=0.8, sr50=0.9, sr75=0.6, num_frames=10)
        seq_long = GOT10kSequenceResult("s2", ao=0.2, sr50=0.3, sr75=0.1, num_frames=100)
        summary = self.ev.aggregate([seq_short, seq_long])
        # Sequence-equal average: (0.8 + 0.2) / 2 = 0.5
        assert abs(summary.ao - 0.5) < 1e-6
        assert summary.num_sequences == 2

    def test_single_sequence_passthrough(self):
        seq = GOT10kSequenceResult("s1", ao=0.42, sr50=0.55, sr75=0.30, num_frames=20)
        summary = self.ev.aggregate([seq])
        assert abs(summary.ao - 0.42) < 1e-6
        assert abs(summary.sr50 - 0.55) < 1e-6
        assert abs(summary.sr75 - 0.30) < 1e-6

    def test_per_sequence_dict_populated(self):
        seq = GOT10kSequenceResult("car1", ao=0.5, sr50=0.6, sr75=0.3, num_frames=50)
        summary = self.ev.aggregate([seq])
        assert "car1" in summary.per_sequence

    def test_to_dict_keys(self):
        seq = GOT10kSequenceResult("s1", ao=0.5, sr50=0.6, sr75=0.3, num_frames=20)
        d = self.ev.aggregate([seq]).to_dict()
        assert "ao" in d and "sr50" in d and "sr75" in d and "num_sequences" in d


# ---------------------------------------------------------------------------
# BenchmarkResult integration
# ---------------------------------------------------------------------------

class TestBenchmarkResultGOT10k:
    def setup_method(self):
        self.engine = BenchmarkEngine(verbose=False)

    def test_got10k_metrics_returns_summary(self):
        result = self.engine.run(ConstantTracker(_BOX), TinyDataset(box=_BOX), "T")
        g = result.got10k_metrics()
        assert isinstance(g, GOT10kSummary)
        assert g.num_sequences == 3

    def test_perfect_tracker_ao_near_one(self):
        result = self.engine.run(ConstantTracker(_BOX), TinyDataset(box=_BOX), "T")
        g = result.got10k_metrics()
        assert g.ao > 0.95

    def test_perfect_tracker_sr50_sr75_near_one(self):
        result = self.engine.run(ConstantTracker(_BOX), TinyDataset(box=_BOX), "T")
        g = result.got10k_metrics()
        assert g.sr50 > 0.95
        assert g.sr75 > 0.95

    def test_shifted_tracker_lower_ao(self):
        perfect = self.engine.run(ConstantTracker(_BOX), TinyDataset(box=_BOX), "T")
        shifted = self.engine.run(ConstantTracker(_SHIFTED), TinyDataset(box=_BOX), "T")
        assert perfect.got10k_metrics().ao > shifted.got10k_metrics().ao

    def test_summary_includes_got10k_fields(self):
        result = self.engine.run(ConstantTracker(_BOX), TinyDataset(box=_BOX), "T")
        s = result.summary()
        assert "got10k_ao" in s
        assert "got10k_sr50" in s
        assert "got10k_sr75" in s

    def test_sr75_leq_sr50_always(self):
        result = self.engine.run(ConstantTracker(_SHIFTED), TinyDataset(box=_BOX), "T")
        g = result.got10k_metrics()
        assert g.sr75 <= g.sr50


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

class TestGOT10kReporting:
    def setup_method(self):
        self.ev = GOT10kEvaluator()
        self.summaries = [
            ("MOSSE", GOT10kSummary(ao=0.42, sr50=0.55, sr75=0.28, num_sequences=10)),
            ("KCF", GOT10kSummary(ao=0.51, sr50=0.64, sr75=0.35, num_sequences=10)),
        ]

    def test_markdown_contains_header(self):
        md = self.ev.to_markdown(self.summaries)
        assert "AO" in md and "SR0.50" in md and "SR0.75" in md

    def test_markdown_sorted_by_ao(self):
        md = self.ev.to_markdown(self.summaries)
        kcf_pos = md.index("KCF")
        mosse_pos = md.index("MOSSE")
        assert kcf_pos < mosse_pos  # KCF has higher AO → appears first

    def test_latex_contains_tabular(self):
        tex = self.ev.to_latex(self.summaries)
        assert r"\begin{tabular}" in tex
        assert r"\end{tabular}" in tex

    def test_latex_bolds_best_value(self):
        tex = self.ev.to_latex(self.summaries)
        assert r"\textbf" in tex

    def test_markdown_empty_list(self):
        md = self.ev.to_markdown([])
        assert "AO" in md

"""Systematic prediction bias analysis for visual object trackers.

Standard benchmarks report aggregate accuracy (mean IoU, AUC) but do not
expose *directional* errors — whether a tracker consistently over-sizes boxes,
drifts left, or loses the target over time.  These systematic biases are
actionable: they can be corrected by post-processing calibration or guide
tracker hyperparameter tuning.

This module measures five bias dimensions from (prediction, ground-truth) pairs:

Size Bias
~~~~~~~~~
mean(pred_area / gt_area).  Values > 1 mean the tracker over-sizes targets
(common in correlation-filter trackers that inflate the search window);
values < 1 mean under-sizing.  An unbiased tracker scores exactly 1.0.

Dimension Bias (width / height)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mean(pred_w / gt_w) and mean(pred_h / gt_h) independently.  Lets you detect
aspect-ratio distortion that averages out in area-based measures — e.g. a
tracker that widens but not tallens its predictions.

Positional Offset Bias
~~~~~~~~~~~~~~~~~~~~~~
Mean *signed* centre offset in x and y, each normalised by the ground-truth
diagonal for scale invariance.  Positive x means the tracker predicts to
the right of the target; negative y means predictions sit above.  This
captures systematic lag behind moving targets or fixed camera misalignment.

Aspect Ratio Deviation
~~~~~~~~~~~~~~~~~~~~~~
Mean absolute fractional deviation of predicted AR from GT AR:
mean |pred_ar - gt_ar| / gt_ar.  Detects trackers that do not adapt their
box shape to non-rigid or rotating targets.

Temporal Drift Slope
~~~~~~~~~~~~~~~~~~~~~
Slope of a least-squares linear regression of IoU versus normalised frame
index.  A negative slope means tracking accuracy deteriorates over the
sequence (the tracker drifts); a near-zero slope means stable tracking.

Typical usage::

    from eovot.metrics.bias import BiasAnalyzer

    analyzer = BiasAnalyzer()

    # Per-sequence analysis — use stored predictions from the benchmark engine
    result = analyzer.analyze(
        seq_result.predictions,
        seq_result.ground_truths,
        tracker_name="KCF",
        sequence_name="car1",
    )
    print(result)
    # PredictionBiasResult[KCF on car1] size_bias=1.023 pos_offset=(+0.01234, -0.00891) ...

    # Aggregate over a full benchmark run
    report = analyzer.analyze_benchmark(benchmark_result)
    print(report["aggregate"])
    print(analyzer.to_markdown_table(report["aggregate"]))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult


@dataclass
class PredictionBiasResult:
    """Per-sequence systematic bias summary.

    All ratio-based fields (size_bias, width_bias, height_bias) are centred
    around 1.0 (1.0 = unbiased); signed offset fields are centred around
    0.0 (0.0 = no positional drift); aspect_ratio_deviation >= 0.

    Attributes:
        tracker_name: Human-readable tracker identifier.
        sequence_name: Sequence identifier.
        num_frames: Number of frames analysed.
        size_bias: mean(pred_area / gt_area).  >1 = over-sizing.
        width_bias: mean(pred_w / gt_w).
        height_bias: mean(pred_h / gt_h).
        x_offset_bias: Mean signed horizontal centre offset, normalised by GT diagonal.
        y_offset_bias: Mean signed vertical centre offset, normalised by GT diagonal.
        aspect_ratio_deviation: mean |pred_ar - gt_ar| / gt_ar.
        temporal_drift_slope: Slope of IoU vs. normalised time
            (negative = drifting away, positive = improving).
    """

    tracker_name: str
    sequence_name: str
    num_frames: int
    size_bias: float
    width_bias: float
    height_bias: float
    x_offset_bias: float
    y_offset_bias: float
    aspect_ratio_deviation: float
    temporal_drift_slope: float

    def __str__(self) -> str:
        return (
            f"PredictionBiasResult[{self.tracker_name} on {self.sequence_name}] "
            f"size_bias={self.size_bias:.3f}  "
            f"pos_offset=({self.x_offset_bias:+.4f}, {self.y_offset_bias:+.4f})  "
            f"ar_dev={self.aspect_ratio_deviation:.4f}  "
            f"drift_slope={self.temporal_drift_slope:+.5f}"
        )

    def to_dict(self) -> Dict:
        """Serialise to a plain dict for JSON export."""
        return {
            "tracker_name": self.tracker_name,
            "sequence_name": self.sequence_name,
            "num_frames": self.num_frames,
            "size_bias": round(self.size_bias, 4),
            "width_bias": round(self.width_bias, 4),
            "height_bias": round(self.height_bias, 4),
            "x_offset_bias": round(self.x_offset_bias, 5),
            "y_offset_bias": round(self.y_offset_bias, 5),
            "aspect_ratio_deviation": round(self.aspect_ratio_deviation, 4),
            "temporal_drift_slope": round(self.temporal_drift_slope, 6),
        }


class BiasAnalyzer:
    """Measure systematic prediction errors in bounding-box streams.

    Compares predicted boxes to ground-truth boxes and quantifies size,
    shape, position, and temporal biases that aggregate accuracy metrics
    (mean IoU, AUC) cannot reveal.

    All inputs use the ``(x, y, w, h)`` convention throughout EOVOT.

    Example::

        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.metrics.bias import BiasAnalyzer
        from eovot.trackers.kcf import KCFTracker
        from eovot.datasets.synthetic import SyntheticDataset

        engine = BenchmarkEngine(verbose=False)
        result = engine.run(KCFTracker(), SyntheticDataset(num_sequences=5))

        analyzer = BiasAnalyzer()
        report = analyzer.analyze_benchmark(result)
        print(analyzer.to_markdown_table(report["aggregate"]))
    """

    _EPS: float = 1e-6  # guard against near-zero denominators

    def analyze(
        self,
        predictions: np.ndarray,
        ground_truths: np.ndarray,
        tracker_name: str = "",
        sequence_name: str = "",
    ) -> PredictionBiasResult:
        """Run full bias analysis on one (predictions, ground-truths) pair.

        Args:
            predictions:   ``(N, 4)`` array of predicted boxes ``(x, y, w, h)``.
            ground_truths: ``(N, 4)`` array of ground-truth boxes ``(x, y, w, h)``.
            tracker_name:  Identifier stored in the result.
            sequence_name: Identifier stored in the result.

        Returns:
            :class:`PredictionBiasResult` with all bias measures populated.
            Returns neutral values (size_bias=1.0, offsets=0.0, …) when the
            input arrays are empty.
        """
        preds = np.asarray(predictions, dtype=np.float64)
        gts = np.asarray(ground_truths, dtype=np.float64)
        n = min(len(preds), len(gts))
        if n == 0:
            return self._neutral_result(tracker_name, sequence_name)
        preds, gts = preds[:n], gts[:n]

        size_bias = self._size_bias(preds, gts)
        width_bias = self._dim_bias(preds[:, 2], gts[:, 2])
        height_bias = self._dim_bias(preds[:, 3], gts[:, 3])
        x_off, y_off = self._position_offset_bias(preds, gts)
        ar_dev = self._aspect_ratio_deviation(preds, gts)
        drift = self._temporal_drift_slope(preds, gts)

        return PredictionBiasResult(
            tracker_name=tracker_name,
            sequence_name=sequence_name,
            num_frames=n,
            size_bias=size_bias,
            width_bias=width_bias,
            height_bias=height_bias,
            x_offset_bias=x_off,
            y_offset_bias=y_off,
            aspect_ratio_deviation=ar_dev,
            temporal_drift_slope=drift,
        )

    def analyze_benchmark(self, result: "BenchmarkResult") -> Dict:
        """Aggregate bias analysis across all sequences in a benchmark run.

        Sequences without stored ``predictions`` / ``ground_truths`` (e.g.
        results loaded from disk via :meth:`~eovot.benchmark.engine.BenchmarkResult.load`)
        are silently skipped.

        Args:
            result: A :class:`~eovot.benchmark.engine.BenchmarkResult` from
                :meth:`~eovot.benchmark.engine.BenchmarkEngine.run`.

        Returns:
            Dict with two keys:

            * ``"per_sequence"`` — ``{seq_name: PredictionBiasResult}``
            * ``"aggregate"`` — mean of each bias dimension across all sequences,
              plus ``tracker_name`` and ``num_sequences``.  Empty dict when no
              sequences have stored predictions.
        """
        per_seq: Dict[str, PredictionBiasResult] = {}
        for sr in result.sequence_results:
            if sr.predictions is None or sr.ground_truths is None:
                continue
            per_seq[sr.sequence_name] = self.analyze(
                sr.predictions,
                sr.ground_truths,
                tracker_name=result.tracker_name,
                sequence_name=sr.sequence_name,
            )

        if not per_seq:
            return {"per_sequence": {}, "aggregate": {}}

        vals = list(per_seq.values())
        agg: Dict = {
            "tracker_name": result.tracker_name,
            "num_sequences": len(vals),
            "mean_size_bias": round(float(np.mean([v.size_bias for v in vals])), 4),
            "mean_width_bias": round(float(np.mean([v.width_bias for v in vals])), 4),
            "mean_height_bias": round(float(np.mean([v.height_bias for v in vals])), 4),
            "mean_x_offset_bias": round(float(np.mean([v.x_offset_bias for v in vals])), 5),
            "mean_y_offset_bias": round(float(np.mean([v.y_offset_bias for v in vals])), 5),
            "mean_aspect_ratio_deviation": round(
                float(np.mean([v.aspect_ratio_deviation for v in vals])), 4
            ),
            "mean_temporal_drift_slope": round(
                float(np.mean([v.temporal_drift_slope for v in vals])), 6
            ),
        }
        return {"per_sequence": per_seq, "aggregate": agg}

    def to_markdown_table(self, aggregate: Dict) -> str:
        """Format an aggregate bias report as a Markdown table.

        Args:
            aggregate: The ``"aggregate"`` value from :meth:`analyze_benchmark`.

        Returns:
            A three-line Markdown table (header, separator, data row).
            Returns an italicised placeholder when *aggregate* is empty.
        """
        if not aggregate:
            return "_No bias data available._"
        header = (
            "| Tracker | Seqs | Size Bias | W Bias | H Bias "
            "| X Offset | Y Offset | AR Dev | Drift Slope |"
        )
        sep = (
            "|---------|-----:|----------:|-------:|-------:"
            "|---------:|---------:|-------:|------------:|"
        )
        row = (
            f"| {aggregate.get('tracker_name', '')} "
            f"| {aggregate.get('num_sequences', 0)} "
            f"| {aggregate.get('mean_size_bias', 0):.4f} "
            f"| {aggregate.get('mean_width_bias', 0):.4f} "
            f"| {aggregate.get('mean_height_bias', 0):.4f} "
            f"| {aggregate.get('mean_x_offset_bias', 0):+.5f} "
            f"| {aggregate.get('mean_y_offset_bias', 0):+.5f} "
            f"| {aggregate.get('mean_aspect_ratio_deviation', 0):.4f} "
            f"| {aggregate.get('mean_temporal_drift_slope', 0):+.6f} |"
        )
        return "\n".join([header, sep, row])

    # ------------------------------------------------------------------
    # Private metric helpers
    # ------------------------------------------------------------------

    def _size_bias(self, preds: np.ndarray, gts: np.ndarray) -> float:
        pred_areas = preds[:, 2] * preds[:, 3]
        gt_areas = gts[:, 2] * gts[:, 3]
        safe_gt = np.where(gt_areas > self._EPS, gt_areas, self._EPS)
        return float(np.mean(pred_areas / safe_gt))

    def _dim_bias(self, pred_dim: np.ndarray, gt_dim: np.ndarray) -> float:
        safe_gt = np.where(gt_dim > self._EPS, gt_dim, self._EPS)
        return float(np.mean(pred_dim / safe_gt))

    def _position_offset_bias(
        self, preds: np.ndarray, gts: np.ndarray
    ) -> Tuple[float, float]:
        pred_cx = preds[:, 0] + preds[:, 2] / 2.0
        pred_cy = preds[:, 1] + preds[:, 3] / 2.0
        gt_cx = gts[:, 0] + gts[:, 2] / 2.0
        gt_cy = gts[:, 1] + gts[:, 3] / 2.0
        gt_diag = np.sqrt(gts[:, 2] ** 2 + gts[:, 3] ** 2)
        norm = np.where(gt_diag > self._EPS, gt_diag, 1.0)
        x_off = float(np.mean((pred_cx - gt_cx) / norm))
        y_off = float(np.mean((pred_cy - gt_cy) / norm))
        return x_off, y_off

    def _aspect_ratio_deviation(self, preds: np.ndarray, gts: np.ndarray) -> float:
        pred_ar = preds[:, 2] / np.where(preds[:, 3] > self._EPS, preds[:, 3], 1.0)
        gt_ar = gts[:, 2] / np.where(gts[:, 3] > self._EPS, gts[:, 3], 1.0)
        safe_gt_ar = np.where(gt_ar > self._EPS, gt_ar, 1.0)
        return float(np.mean(np.abs(pred_ar - gt_ar) / safe_gt_ar))

    def _temporal_drift_slope(self, preds: np.ndarray, gts: np.ndarray) -> float:
        if len(preds) < 2:
            return 0.0
        from ..metrics.accuracy import MetricsEngine
        ious = MetricsEngine().batch_iou(preds, gts)
        t = np.linspace(0.0, 1.0, len(ious))
        t_mean = float(t.mean())
        iou_mean = float(ious.mean())
        cov = float(((t - t_mean) * (ious - iou_mean)).mean())
        var = float(((t - t_mean) ** 2).mean())
        return cov / var if var > 1e-12 else 0.0

    def _neutral_result(self, tracker_name: str, sequence_name: str) -> PredictionBiasResult:
        return PredictionBiasResult(
            tracker_name=tracker_name,
            sequence_name=sequence_name,
            num_frames=0,
            size_bias=1.0,
            width_bias=1.0,
            height_bias=1.0,
            x_offset_bias=0.0,
            y_offset_bias=0.0,
            aspect_ratio_deviation=0.0,
            temporal_drift_slope=0.0,
        )

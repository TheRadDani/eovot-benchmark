"""Sequence attribute analysis for VOT benchmarks.

VOT challenge attributes label sequences with the tracking challenges they
present (fast motion, occlusion, scale variation, etc.).  Breaking down
accuracy and efficiency metrics per attribute reveals *where* a tracker
excels or struggles — information that mean IoU alone cannot convey.
This is a standard component of publication-quality tracker comparisons
(OTB, VOT, GOT-10k, LaSOT all report per-attribute performance).

Components
----------
STANDARD_ATTRIBUTES
    Canonical set of 10 challenge categories used across OTB, VOT,
    GOT-10k, and LaSOT, with textual definitions.

auto_annotate_from_gt
    Automatically infer challenge attributes from GT box trajectories.
    Requires only the ground-truth array and frame dimensions; no
    manual labelling needed.

AttributeAnnotations
    Type alias for ``{sequence_name: set_of_attribute_strings}``.

AttributeResult
    Per-attribute metric breakdown for one tracker.

AttributeAnalyzer
    Aggregates BenchmarkResult metrics by attribute, generates ranked
    comparison tables, and supports multi-tracker breakdowns.

Typical usage::

    from eovot.metrics.attributes import AttributeAnalyzer, auto_annotate_from_gt

    # Build annotations automatically from GT boxes stored in BenchmarkResult
    annotations = {}
    for sr in benchmark_result.sequence_results:
        if sr.ground_truths is not None:
            annotations[sr.sequence_name] = auto_annotate_from_gt(
                sr.ground_truths, frame_size=(640, 480)
            )

    analyzer = AttributeAnalyzer()
    breakdown = analyzer.analyze(benchmark_result, annotations)
    print(analyzer.to_markdown_table(breakdown))

    # Multi-tracker comparison: one row per attribute per tracker
    breakdown_all = analyzer.compare_trackers(
        [mosse_result, kcf_result, csrt_result], annotations
    )
    print(analyzer.to_comparison_table(breakdown_all))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult

# Type alias: maps sequence name → set of attribute labels present.
AttributeAnnotations = Dict[str, Set[str]]


# ---------------------------------------------------------------------------
# Standard attribute taxonomy
# ---------------------------------------------------------------------------

STANDARD_ATTRIBUTES: Dict[str, str] = {
    "fast_motion": (
        "Target moves more than 20 px per frame on average — "
        "correlation-filter trackers struggle as the target leaves the search window."
    ),
    "scale_variation": (
        "Ratio of maximum to minimum GT bounding-box area exceeds 4× over the sequence — "
        "fixed-scale trackers fail to maintain tight coverage."
    ),
    "occlusion": (
        "Heuristic: mean IoU of GT with its own neighbours drops sharply (proxy for disappearance). "
        "Indicates periods where the target is fully or partially hidden."
    ),
    "illumination_change": (
        "Significant variation in mean frame brightness across the sequence — "
        "colour-based and gradient-based trackers are sensitive to this."
    ),
    "aspect_ratio_change": (
        "Standard deviation of the GT width/height ratio exceeds 0.3 — "
        "indicates a deformable or rotating target."
    ),
    "out_of_view": (
        "GT box centre exits the frame boundaries at least once — "
        "the target partially or fully leaves the camera field of view."
    ),
    "low_resolution": (
        "Minimum GT bounding-box area is smaller than 400 px² — "
        "appearance models degrade on very small targets."
    ),
    "background_clutter": (
        "High spatial gradient variance in the search region — "
        "distractors with similar appearance to the target are present."
    ),
    "motion_blur": (
        "Proxy: very high speed (> 40 px/frame) combined with small target size — "
        "the target appears blurred due to fast motion."
    ),
    "deformation": (
        "Non-rigid target: aspect ratio changes AND large variation in box diagonal — "
        "animals, humans, flags that change shape non-rigidly."
    ),
}


# ---------------------------------------------------------------------------
# Auto-annotation from GT boxes
# ---------------------------------------------------------------------------

def auto_annotate_from_gt(
    gt_boxes: np.ndarray,
    frame_size: Tuple[int, int] = (640, 480),
    *,
    fast_motion_px_per_frame: float = 20.0,
    scale_ratio_threshold: float = 4.0,
    aspect_ratio_std_threshold: float = 0.30,
    low_res_area_threshold: int = 400,
    motion_blur_speed_px: float = 40.0,
    out_of_view_margin: float = 0.0,
) -> Set[str]:
    """Infer challenge attributes from a GT box trajectory.

    Uses only the ground-truth bounding-box array — no frame pixel data
    required.  Attributes are detected via interpretable geometric tests so
    researchers can audit and tune each threshold.

    Args:
        gt_boxes: ``(N, 4)`` array of GT boxes ``(x, y, w, h)``.
        frame_size: ``(width, height)`` of the video frames in pixels.
            Used for out-of-view detection.  Default ``(640, 480)``.
        fast_motion_px_per_frame: Mean per-frame displacement (px) above
            which ``"fast_motion"`` is flagged.  Default ``20.0``.
        scale_ratio_threshold: max_area / min_area ratio above which
            ``"scale_variation"`` is flagged.  Default ``4.0``.
        aspect_ratio_std_threshold: Standard deviation of the ``w/h`` ratio
            above which ``"aspect_ratio_change"`` and ``"deformation"``
            are flagged.  Default ``0.30``.
        low_res_area_threshold: Minimum GT area (px²) below which
            ``"low_resolution"`` is flagged.  Default ``400``.
        motion_blur_speed_px: Speed threshold (px/frame) used together with
            small target size to flag ``"motion_blur"``.  Default ``40.0``.
        out_of_view_margin: Fraction of frame dimensions allowed outside
            the boundary before flagging ``"out_of_view"``.  ``0.0`` means
            any centre leaving the frame counts.  Default ``0.0``.

    Returns:
        Set of attribute strings from :data:`STANDARD_ATTRIBUTES` that
        apply to this sequence.  May be empty for a smooth, well-lit,
        medium-speed sequence.
    """
    gt = np.asarray(gt_boxes, dtype=np.float64)
    if len(gt) < 2:
        return set()

    attrs: Set[str] = set()
    fw, fh = float(frame_size[0]), float(frame_size[1])

    # --- Geometry helpers ---
    centers = gt[:, :2] + gt[:, 2:] / 2.0          # (N, 2)
    areas = gt[:, 2] * gt[:, 3]                      # (N,)
    diagonals = np.sqrt(gt[:, 2] ** 2 + gt[:, 3] ** 2)  # (N,)
    aspect = np.where(gt[:, 3] > 1e-6, gt[:, 2] / gt[:, 3], 1.0)  # w/h  (N,)

    # --- Frame-to-frame displacements ---
    displacements = np.diff(centers, axis=0)          # (N-1, 2)
    speeds = np.linalg.norm(displacements, axis=1)    # (N-1,) px/frame
    mean_speed = float(speeds.mean()) if len(speeds) > 0 else 0.0

    # --- Fast motion ---
    if mean_speed > fast_motion_px_per_frame:
        attrs.add("fast_motion")

    # --- Scale variation ---
    valid_areas = areas[areas > 1e-6]
    if len(valid_areas) >= 2:
        scale_ratio = float(valid_areas.max() / valid_areas.min())
        if scale_ratio > scale_ratio_threshold:
            attrs.add("scale_variation")

    # --- Aspect ratio change / deformation ---
    aspect_std = float(aspect.std())
    diag_std = float(diagonals.std())
    if aspect_std > aspect_ratio_std_threshold:
        attrs.add("aspect_ratio_change")
        if diag_std > 10.0:
            attrs.add("deformation")

    # --- Out of view ---
    x_margin = fw * out_of_view_margin
    y_margin = fh * out_of_view_margin
    cx, cy = centers[:, 0], centers[:, 1]
    if np.any(cx < -x_margin) or np.any(cx > fw + x_margin) or \
       np.any(cy < -y_margin) or np.any(cy > fh + y_margin):
        attrs.add("out_of_view")

    # --- Low resolution ---
    if float(valid_areas.min()) < low_res_area_threshold if len(valid_areas) > 0 else False:
        attrs.add("low_resolution")

    # --- Motion blur (proxy: high speed + small target) ---
    mean_area = float(valid_areas.mean()) if len(valid_areas) > 0 else 0.0
    if mean_speed > motion_blur_speed_px and mean_area < 1600.0:
        attrs.add("motion_blur")

    return attrs


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AttributeResult:
    """Per-attribute metric breakdown for one tracker.

    Attributes:
        attribute: Challenge category name (one of :data:`STANDARD_ATTRIBUTES`).
        tracker_name: Human-readable tracker identifier.
        dataset_name: Dataset on which the tracker was evaluated.
        num_sequences: Number of sequences annotated with this attribute.
        mean_iou: Mean IoU across all frames in annotated sequences.
        mean_success_auc: Mean success-curve AUC, or ``None`` if not computed.
        mean_fps: Mean frames-per-second on annotated sequences.
        sequences: Names of sequences contributing to this result.
    """

    attribute: str
    tracker_name: str
    dataset_name: str
    num_sequences: int
    mean_iou: float
    mean_success_auc: Optional[float]
    mean_fps: float
    sequences: List[str] = field(default_factory=list, repr=False)

    def __str__(self) -> str:
        sauc = f"  AUC={self.mean_success_auc:.4f}" if self.mean_success_auc is not None else ""
        return (
            f"AttributeResult[{self.attribute}] "
            f"tracker={self.tracker_name}  "
            f"n={self.num_sequences}  "
            f"mIoU={self.mean_iou:.4f}"
            f"{sauc}  "
            f"FPS={self.mean_fps:.1f}"
        )

    def to_dict(self) -> Dict:
        d: Dict = {
            "attribute": self.attribute,
            "tracker": self.tracker_name,
            "dataset": self.dataset_name,
            "num_sequences": self.num_sequences,
            "mean_iou": round(self.mean_iou, 4),
            "mean_fps": round(self.mean_fps, 1),
        }
        if self.mean_success_auc is not None:
            d["mean_success_auc"] = round(self.mean_success_auc, 4)
        return d


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class AttributeAnalyzer:
    """Break down BenchmarkResult metrics per challenge attribute.

    Args:
        attributes: Collection of attribute names to analyse.  Defaults to
            all keys in :data:`STANDARD_ATTRIBUTES`.  Pass a subset to
            focus on specific challenges.

    Example::

        analyzer = AttributeAnalyzer()

        annotations = {}
        for sr in benchmark_result.sequence_results:
            if sr.ground_truths is not None:
                annotations[sr.sequence_name] = auto_annotate_from_gt(
                    sr.ground_truths, frame_size=(640, 480)
                )

        breakdown = analyzer.analyze(benchmark_result, annotations)
        print(analyzer.to_markdown_table(breakdown))
    """

    def __init__(
        self,
        attributes: Optional[List[str]] = None,
    ) -> None:
        self.attributes: List[str] = (
            list(STANDARD_ATTRIBUTES.keys()) if attributes is None else list(attributes)
        )
        for attr in self.attributes:
            if attr not in STANDARD_ATTRIBUTES:
                raise ValueError(
                    f"Unknown attribute '{attr}'. "
                    f"Available: {list(STANDARD_ATTRIBUTES)}"
                )

    def analyze(
        self,
        result: "BenchmarkResult",
        annotations: AttributeAnnotations,
    ) -> List[AttributeResult]:
        """Compute per-attribute metric breakdown for one tracker's result.

        Sequences not present in *annotations* are silently skipped.

        Args:
            result:      Full benchmark result for one tracker.
            annotations: Mapping ``{sequence_name: set_of_attributes}``.

        Returns:
            One :class:`AttributeResult` per attribute that appears in at
            least one annotated sequence, sorted by attribute name.
        """
        attr_seqs: Dict[str, List] = {a: [] for a in self.attributes}

        for sr in result.sequence_results:
            seq_attrs = annotations.get(sr.sequence_name, set())
            for attr in self.attributes:
                if attr in seq_attrs:
                    attr_seqs[attr].append(sr)

        results: List[AttributeResult] = []
        for attr in self.attributes:
            seqs = attr_seqs[attr]
            if not seqs:
                continue

            ious_all = np.concatenate([s.ious for s in seqs]) if seqs else np.array([])
            mean_iou = float(ious_all.mean()) if len(ious_all) > 0 else 0.0

            aucs = [s.accuracy_metrics.success_auc for s in seqs
                    if s.accuracy_metrics is not None]
            mean_auc = float(np.mean(aucs)) if aucs else None

            mean_fps = float(np.mean([s.profiling.fps for s in seqs]))

            results.append(AttributeResult(
                attribute=attr,
                tracker_name=result.tracker_name,
                dataset_name=result.dataset_name,
                num_sequences=len(seqs),
                mean_iou=mean_iou,
                mean_success_auc=mean_auc,
                mean_fps=mean_fps,
                sequences=[s.sequence_name for s in seqs],
            ))

        results.sort(key=lambda r: r.attribute)
        return results

    def compare_trackers(
        self,
        results: List["BenchmarkResult"],
        annotations: AttributeAnnotations,
    ) -> Dict[str, List[AttributeResult]]:
        """Run per-attribute analysis for multiple trackers.

        Args:
            results:     One :class:`~eovot.benchmark.engine.BenchmarkResult`
                         per tracker.
            annotations: Shared sequence-to-attributes mapping.

        Returns:
            Dict mapping tracker name → list of :class:`AttributeResult`.
        """
        return {r.tracker_name: self.analyze(r, annotations) for r in results}

    def to_markdown_table(
        self,
        breakdown: List[AttributeResult],
        *,
        show_fps: bool = True,
    ) -> str:
        """Format a single-tracker attribute breakdown as a Markdown table.

        Args:
            breakdown: Output of :meth:`analyze`.
            show_fps:  Include the mean FPS column.  Default ``True``.

        Returns:
            Multi-line Markdown string ready for embedding in reports or READMEs.

        Example output::

            | Attribute           | # Seqs | mIoU   | Success AUC | FPS   |
            |---------------------|-------:|-------:|------------:|------:|
            | fast_motion         |      8 | 0.3421 |      0.3817 |  89.4 |
            | scale_variation     |      5 | 0.4102 |      0.4560 | 102.1 |
        """
        if not breakdown:
            return "*No attribute results to display.*\n"

        has_auc = any(r.mean_success_auc is not None for r in breakdown)
        tracker = breakdown[0].tracker_name
        dataset = breakdown[0].dataset_name

        header_parts = ["| Attribute", "# Seqs", "mIoU"]
        sep_parts = ["|---------|", "------:|", "------:|"]
        if has_auc:
            header_parts.append("Success AUC")
            sep_parts.append("-----------:|")
        if show_fps:
            header_parts.append("FPS |")
            sep_parts.append("-----:|")
        else:
            header_parts[-1] += " |"

        lines = [
            f"**Tracker:** {tracker} | **Dataset:** {dataset}\n",
            " | ".join(header_parts),
            " ".join(sep_parts),
        ]

        for r in breakdown:
            row = f"| {r.attribute:<21s} | {r.num_sequences:>6d} | {r.mean_iou:.4f}"
            if has_auc:
                auc_str = f"{r.mean_success_auc:.4f}" if r.mean_success_auc is not None else "  —   "
                row += f" | {auc_str:>11s}"
            if show_fps:
                row += f" | {r.mean_fps:>5.1f} |"
            else:
                row += " |"
            lines.append(row)

        return "\n".join(lines)

    def to_comparison_table(
        self,
        multi_breakdown: Dict[str, List[AttributeResult]],
        *,
        metric: str = "mean_iou",
    ) -> str:
        """Multi-tracker per-attribute comparison table (one column per tracker).

        Args:
            multi_breakdown: Output of :meth:`compare_trackers`.
            metric:          Which metric to display: ``"mean_iou"`` or
                             ``"mean_success_auc"``.  Default ``"mean_iou"``.

        Returns:
            Markdown table with one row per attribute and one column per tracker.

        Example output (metric='mean_iou')::

            | Attribute       | MOSSE  | KCF    | CSRT   |
            |-----------------|-------:|-------:|-------:|
            | fast_motion     | 0.3421 | 0.4102 | 0.5223 |
            | scale_variation | 0.2987 | 0.3410 | 0.4881 |
        """
        if not multi_breakdown:
            return "*No results to display.*\n"

        trackers = list(multi_breakdown.keys())

        # Build lookup: tracker → {attribute → result}
        lookup: Dict[str, Dict[str, AttributeResult]] = {}
        all_attrs: Set[str] = set()
        for tracker, breakdown in multi_breakdown.items():
            lookup[tracker] = {r.attribute: r for r in breakdown}
            all_attrs.update(lookup[tracker].keys())

        sorted_attrs = sorted(all_attrs)
        tracker_cols = " | ".join(f"{t:<7s}" for t in trackers)
        sep_cols = " | ".join("-------:" for _ in trackers)

        lines = [
            f"| Attribute             | {tracker_cols} |",
            f"|---|{sep_cols} |",
        ]

        for attr in sorted_attrs:
            row = f"| {attr:<21s} |"
            for tracker in trackers:
                res = lookup[tracker].get(attr)
                if res is None:
                    row += "    —   |"
                elif metric == "mean_success_auc":
                    val = res.mean_success_auc
                    row += f" {'—' if val is None else f'{val:.4f}':>7s} |"
                else:
                    row += f" {res.mean_iou:>7.4f} |"
            lines.append(row)

        return "\n".join(lines)

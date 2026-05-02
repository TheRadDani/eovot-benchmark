"""Per-attribute performance breakdown for benchmark results.

Given a set of :class:`~eovot.benchmark.engine.BenchmarkResult` objects and
a dict of :class:`~eovot.analysis.sequence_attributes.SequenceAttributes`,
this module slices tracker performance by difficulty attribute and produces
a research-grade breakdown table showing where each tracker succeeds or fails.

Example::

    from eovot.analysis.sequence_attributes import tag_sequences
    from eovot.analysis.attribute_report import generate_attribute_report

    tagged  = tag_sequences(dataset)
    report  = generate_attribute_report({"MOSSE": result_mosse}, tagged)
    print(report.markdown_table())
    report.save_json("results/attribute_breakdown.json")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..benchmark.engine import BenchmarkResult
from .sequence_attributes import SequenceAttributes

_ALL_FLAGS = ["SV", "ARC", "FM", "LR", "PO", "DEF"]

_FLAG_DESCRIPTIONS = {
    "SV":  "Scale Variation",
    "ARC": "Aspect Ratio Change",
    "FM":  "Fast Motion",
    "LR":  "Low Resolution",
    "PO":  "Partial Occlusion",
    "DEF": "Deformation",
}


@dataclass
class TrackerAttributeSlice:
    """Mean IoU for one tracker on one attribute subset of sequences.

    Attributes:
        tracker_name:   Tracker identifier.
        attribute:      Short attribute code (e.g. ``"FM"``).
        num_sequences:  Number of sequences in this attribute subset.
        mean_iou:       Mean IoU across all sequences that have the attribute.
        mean_fps:       Mean FPS across those sequences.
    """

    tracker_name: str
    attribute: str
    num_sequences: int
    mean_iou: float
    mean_fps: float

    def to_dict(self) -> Dict:
        return {
            "tracker": self.tracker_name,
            "attribute": self.attribute,
            "num_sequences": self.num_sequences,
            "mean_iou": round(self.mean_iou, 4),
            "mean_fps": round(self.mean_fps, 2),
        }


@dataclass
class AttributeReport:
    """Full per-attribute breakdown for one or more trackers.

    Built by :func:`generate_attribute_report`.

    Attributes:
        tracker_names:    Ordered list of tracker identifiers.
        attribute_slices: Nested dict ``{attr_code: {tracker_name: slice}}``.
        overall:          Overall (all-sequence) slice per tracker.
    """

    tracker_names: List[str]
    attribute_slices: Dict[str, Dict[str, TrackerAttributeSlice]] = field(default_factory=dict)
    overall: Dict[str, TrackerAttributeSlice] = field(default_factory=dict)

    def markdown_table(self) -> str:
        """Render a Markdown comparison table (attributes × trackers).

        Returns:
            Multi-line Markdown string suitable for GitHub PR descriptions or
            Jupyter notebooks.
        """
        cols = self.tracker_names
        header = "| Attribute | Sequences | " + " | ".join(f"{t} mIoU" for t in cols) + " |"
        sep = "|" + "---|" * (len(cols) + 2)
        rows = [header, sep]

        # Overall row first
        overall_cells = []
        for t in cols:
            sl = self.overall.get(t)
            overall_cells.append(f"{sl.mean_iou:.4f}" if sl else "—")
        rows.append(
            f"| **Overall** | {next(iter(self.overall.values())).num_sequences if self.overall else 0} | "
            + " | ".join(overall_cells)
            + " |"
        )

        # Per-attribute rows
        for flag in _ALL_FLAGS:
            slices = self.attribute_slices.get(flag, {})
            num_seqs = next(iter(slices.values())).num_sequences if slices else 0
            cells = []
            for t in cols:
                sl = slices.get(t)
                if sl is None or sl.num_sequences == 0:
                    cells.append("—")
                else:
                    cells.append(f"{sl.mean_iou:.4f}")
            desc = _FLAG_DESCRIPTIONS.get(flag, flag)
            rows.append(f"| {desc} ({flag}) | {num_seqs} | " + " | ".join(cells) + " |")

        return "\n".join(rows)

    def to_dict(self) -> Dict:
        """Serialise to a plain dict for JSON export."""
        return {
            "trackers": self.tracker_names,
            "overall": {t: sl.to_dict() for t, sl in self.overall.items()},
            "per_attribute": {
                flag: {t: sl.to_dict() for t, sl in tracker_map.items()}
                for flag, tracker_map in self.attribute_slices.items()
            },
        }

    def save_json(self, path: str) -> None:
        """Write the report to a JSON file.

        Args:
            path: Output file path (parent directories are created if needed).
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    def print_summary(self) -> None:
        """Print the Markdown table to stdout."""
        print(self.markdown_table())


def generate_attribute_report(
    results: Dict[str, BenchmarkResult],
    sequence_attributes: Dict[str, SequenceAttributes],
) -> AttributeReport:
    """Build a per-attribute performance breakdown across multiple trackers.

    For each difficulty attribute, selects the subset of sequences that have
    that attribute flagged, then computes mean IoU and mean FPS for every
    tracker on that subset.

    Args:
        results: Dict mapping tracker name → :class:`~eovot.benchmark.engine.BenchmarkResult`.
            All results must share the same set of evaluated sequences for
            the comparison to be valid.
        sequence_attributes: Dict mapping sequence name → :class:`SequenceAttributes`.
            Typically produced by :func:`~eovot.analysis.sequence_attributes.tag_sequences`.

    Returns:
        :class:`AttributeReport` ready for printing or JSON export.

    Example::

        report = generate_attribute_report(
            {"MOSSE": result_mosse, "KCF": result_kcf},
            tag_sequences(dataset),
        )
        print(report.markdown_table())
    """
    tracker_names = list(results.keys())
    report = AttributeReport(tracker_names=tracker_names)

    # Build overall slice
    for tracker_name, bench_result in results.items():
        if not bench_result.sequence_results:
            continue
        all_ious = np.concatenate([sr.ious for sr in bench_result.sequence_results])
        all_fps = [sr.profiling.fps for sr in bench_result.sequence_results]
        report.overall[tracker_name] = TrackerAttributeSlice(
            tracker_name=tracker_name,
            attribute="ALL",
            num_sequences=len(bench_result.sequence_results),
            mean_iou=float(all_ious.mean()) if len(all_ious) else 0.0,
            mean_fps=float(np.mean(all_fps)) if all_fps else 0.0,
        )

    # Per-attribute slices
    for flag_code in _ALL_FLAGS:
        flag_attr = _flag_code_to_attr(flag_code)
        flag_slices: Dict[str, TrackerAttributeSlice] = {}

        # Find sequences with this flag present
        flagged_sequences = {
            name for name, attrs in sequence_attributes.items()
            if getattr(attrs.flags, flag_attr)
        }

        for tracker_name, bench_result in results.items():
            matching = [
                sr for sr in bench_result.sequence_results
                if sr.sequence_name in flagged_sequences
            ]
            if not matching:
                flag_slices[tracker_name] = TrackerAttributeSlice(
                    tracker_name=tracker_name,
                    attribute=flag_code,
                    num_sequences=0,
                    mean_iou=0.0,
                    mean_fps=0.0,
                )
                continue

            ious = np.concatenate([sr.ious for sr in matching])
            fps_vals = [sr.profiling.fps for sr in matching]
            flag_slices[tracker_name] = TrackerAttributeSlice(
                tracker_name=tracker_name,
                attribute=flag_code,
                num_sequences=len(matching),
                mean_iou=float(ious.mean()) if len(ious) else 0.0,
                mean_fps=float(np.mean(fps_vals)),
            )

        report.attribute_slices[flag_code] = flag_slices

    return report


def _flag_code_to_attr(code: str) -> str:
    """Map short flag code to :class:`~eovot.analysis.sequence_attributes.AttributeFlags` field name."""
    _map = {
        "SV": "scale_variation",
        "ARC": "aspect_ratio_change",
        "FM": "fast_motion",
        "LR": "low_resolution",
        "PO": "partial_occlusion",
        "DEF": "deformation",
    }
    if code not in _map:
        raise KeyError(f"Unknown attribute code {code!r}. Valid codes: {list(_map)}")
    return _map[code]

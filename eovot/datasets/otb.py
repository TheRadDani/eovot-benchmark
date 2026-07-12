"""Attribute-aware OTB dataset loader for EOVOT.

Extends the simple :class:`~eovot.datasets.base.OTBDataset` with per-sequence
challenge-attribute metadata, OTB50/OTB100 subset filtering, and attribute-
based sequence selection — enabling the attribute-stratified evaluation that
every VOT paper reports on OTB.

OTB defines 11 challenge attributes:
  IV  — Illumination Variation
  SV  — Scale Variation
  OCC — Occlusion
  DEF — Deformation
  MB  — Motion Blur
  FM  — Fast Motion
  IPR — In-Plane Rotation
  OPR — Out-of-Plane Rotation
  OV  — Out of View
  BC  — Background Clutter
  LR  — Low Resolution

Usage::

    from eovot.datasets.otb import AttributeAwareOTBDataset

    ds = AttributeAwareOTBDataset("/data/OTB100")
    print(f"Loaded {len(ds)} sequences")
    print(ds.attribute_distribution())

    # Only evaluate on occlusion + fast-motion sequences
    hard = AttributeAwareOTBDataset(
        "/data/OTB100",
        attributes=["occlusion", "fast_motion"],
    )

    # Attribute breakdown from evaluation results
    for attr in AttributeAwareOTBDataset.ATTRIBUTES:
        seqs = ds.sequences_by_attribute(attr)
        print(f"{attr}: {len(seqs)} sequences")
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Iterator, List, Optional, Set

import numpy as np

from .base import BaseDataset, Sequence

# ── Challenge attribute constants ─────────────────────────────────────────────

#: All 11 OTB challenge attribute names used by :class:`AttributeAwareOTBDataset`.
OTB_ATTRIBUTES: List[str] = [
    "illumination_variation",  # IV
    "scale_variation",         # SV
    "occlusion",               # OCC
    "deformation",             # DEF
    "motion_blur",             # MB
    "fast_motion",             # FM
    "in_plane_rotation",       # IPR
    "out_of_plane_rotation",   # OPR
    "out_of_view",             # OV
    "background_clutter",      # BC
    "low_resolution",          # LR
]

# ── OTB-50 membership (Wu et al. 2013, ~51 sequences) ────────────────────────

OTB50_SEQUENCES: Set[str] = {
    "Basketball", "Biker", "Bird1", "BlurBody", "BlurCar2", "BlurFace",
    "BlurOwl", "Bolt", "Box", "Car1", "Car4", "CarDark", "CarScale",
    "ClifBar", "Couple", "Crowds", "David", "Deer", "Diving", "DragonBaby",
    "Dudek", "Football", "Freeman4", "Girl", "Human3", "Human4-2",
    "Human6", "Human7", "Human8", "Human9", "Ironman", "Jump", "Jumping",
    "KiteSurf", "Lemming", "Man", "MiniBike", "Matrix", "Mhyang",
    "MotorRolling", "Panda", "RedTeam", "Shaking", "Singer2", "Skating1",
    "Skating2-1", "Skiing", "Soccer", "Surfer", "Sylvester", "Tiger2",
    "Trellis", "Walking", "Walking2", "Woman",
}

# ── Per-sequence attribute annotations (OTB-100 official) ────────────────────

#: Maps each OTB sequence name to its list of challenge attributes.
SEQUENCE_ATTRIBUTES: Dict[str, List[str]] = {
    "Basketball":    ["illumination_variation", "occlusion", "in_plane_rotation", "out_of_plane_rotation"],
    "Biker":         ["scale_variation", "occlusion", "out_of_plane_rotation", "out_of_view", "fast_motion"],
    "Bird1":         ["deformation", "fast_motion", "out_of_view"],
    "Bird2":         ["occlusion", "in_plane_rotation", "out_of_plane_rotation", "fast_motion"],
    "BlurBody":      ["scale_variation", "deformation", "motion_blur", "fast_motion"],
    "BlurCar1":      ["scale_variation", "motion_blur", "fast_motion"],
    "BlurCar2":      ["scale_variation", "motion_blur", "fast_motion"],
    "BlurCar3":      ["scale_variation", "motion_blur", "fast_motion"],
    "BlurCar4":      ["scale_variation", "motion_blur", "fast_motion"],
    "BlurFace":      ["motion_blur", "fast_motion"],
    "BlurOwl":       ["scale_variation", "in_plane_rotation", "out_of_plane_rotation", "motion_blur", "fast_motion"],
    "Board":         ["illumination_variation", "scale_variation", "occlusion", "out_of_view", "motion_blur", "fast_motion"],
    "Bolt":          ["occlusion", "deformation", "fast_motion"],
    "Bolt2":         ["deformation", "fast_motion"],
    "Box":           ["illumination_variation", "scale_variation", "occlusion", "out_of_plane_rotation", "out_of_view", "motion_blur", "fast_motion", "background_clutter"],
    "Car1":          ["illumination_variation", "scale_variation", "motion_blur", "fast_motion", "background_clutter", "low_resolution"],
    "Car2":          ["illumination_variation", "scale_variation", "fast_motion", "background_clutter"],
    "Car4":          ["illumination_variation", "scale_variation"],
    "Car24":         ["illumination_variation", "scale_variation"],
    "CarDark":       ["illumination_variation", "background_clutter", "low_resolution"],
    "CarScale":      ["scale_variation", "occlusion", "out_of_plane_rotation"],
    "ClifBar":       ["scale_variation", "occlusion", "out_of_plane_rotation", "out_of_view", "motion_blur", "fast_motion", "background_clutter"],
    "Coke":          ["illumination_variation", "occlusion", "fast_motion", "background_clutter"],
    "Couple":        ["scale_variation", "deformation", "fast_motion", "background_clutter"],
    "Coupon":        ["occlusion", "background_clutter"],
    "Crossing":      ["scale_variation", "deformation", "fast_motion", "background_clutter"],
    "Crowds":        ["illumination_variation", "deformation", "background_clutter"],
    "David":         ["illumination_variation", "scale_variation", "occlusion", "deformation", "out_of_plane_rotation"],
    "David2":        ["in_plane_rotation", "out_of_plane_rotation"],
    "David3":        ["occlusion", "deformation", "out_of_plane_rotation", "background_clutter"],
    "Deer":          ["motion_blur", "fast_motion", "background_clutter", "low_resolution"],
    "Diving":        ["scale_variation", "deformation", "in_plane_rotation", "out_of_plane_rotation"],
    "Dog":           ["scale_variation", "out_of_plane_rotation", "deformation"],
    "Dog1":          ["scale_variation", "out_of_plane_rotation"],
    "Doll":          ["scale_variation", "in_plane_rotation", "out_of_plane_rotation"],
    "DragonBaby":    ["scale_variation", "occlusion", "out_of_plane_rotation", "out_of_view", "motion_blur", "fast_motion"],
    "Dudek":         ["scale_variation", "occlusion", "deformation", "in_plane_rotation", "out_of_plane_rotation", "out_of_view", "background_clutter"],
    "FaceOcc1":      ["occlusion"],
    "FaceOcc2":      ["illumination_variation", "occlusion", "in_plane_rotation"],
    "Fish":          ["illumination_variation"],
    "FleetFace":     ["scale_variation", "deformation", "in_plane_rotation", "out_of_plane_rotation", "motion_blur", "fast_motion"],
    "Football":      ["illumination_variation", "occlusion", "out_of_plane_rotation", "background_clutter"],
    "Football1":     ["illumination_variation", "in_plane_rotation", "out_of_plane_rotation", "background_clutter"],
    "Freeman1":      ["scale_variation", "out_of_plane_rotation"],
    "Freeman3":      ["scale_variation", "out_of_plane_rotation"],
    "Freeman4":      ["scale_variation", "occlusion", "out_of_plane_rotation"],
    "Girl":          ["scale_variation", "occlusion", "in_plane_rotation", "out_of_plane_rotation"],
    "Girl2":         ["scale_variation", "occlusion", "deformation", "in_plane_rotation", "out_of_plane_rotation"],
    "Gym":           ["scale_variation", "deformation", "in_plane_rotation", "out_of_plane_rotation"],
    "Human2":        ["scale_variation"],
    "Human3":        ["scale_variation", "occlusion", "deformation", "out_of_plane_rotation", "out_of_view"],
    "Human4-2":      ["scale_variation", "occlusion", "deformation", "out_of_plane_rotation"],
    "Human5":        ["scale_variation", "occlusion", "deformation", "out_of_plane_rotation"],
    "Human6":        ["scale_variation", "occlusion", "deformation", "out_of_plane_rotation", "out_of_view"],
    "Human7":        ["scale_variation", "occlusion", "deformation", "out_of_plane_rotation", "out_of_view"],
    "Human8":        ["scale_variation", "occlusion", "deformation", "out_of_plane_rotation"],
    "Human9":        ["scale_variation", "occlusion", "deformation", "out_of_plane_rotation"],
    "Ironman":       ["scale_variation", "occlusion", "in_plane_rotation", "out_of_plane_rotation", "out_of_view", "motion_blur", "fast_motion", "background_clutter"],
    "Jogging-1":     ["occlusion", "deformation", "out_of_plane_rotation"],
    "Jogging-2":     ["occlusion", "deformation", "out_of_plane_rotation"],
    "Jump":          ["scale_variation", "in_plane_rotation", "out_of_plane_rotation", "motion_blur", "fast_motion"],
    "Jumping":       ["motion_blur", "fast_motion"],
    "KiteSurf":      ["illumination_variation", "occlusion", "in_plane_rotation", "out_of_plane_rotation"],
    "Lemming":       ["scale_variation", "occlusion", "out_of_plane_rotation", "out_of_view", "fast_motion"],
    "Liquor":        ["scale_variation", "occlusion", "out_of_view", "motion_blur", "fast_motion", "background_clutter"],
    "Man":           ["illumination_variation"],
    "Matrix":        ["illumination_variation", "scale_variation", "occlusion", "out_of_plane_rotation", "motion_blur", "fast_motion", "background_clutter"],
    "Mhyang":        ["illumination_variation", "deformation", "background_clutter"],
    "MiniBike":      ["scale_variation", "occlusion", "out_of_view", "fast_motion", "background_clutter"],
    "Monkey":        ["scale_variation", "occlusion", "out_of_plane_rotation", "deformation"],
    "MotorRolling":  ["illumination_variation", "scale_variation", "motion_blur", "fast_motion", "background_clutter"],
    "MountainBike":  ["in_plane_rotation", "out_of_plane_rotation", "background_clutter"],
    "Panda":         ["scale_variation", "occlusion", "deformation", "out_of_view", "low_resolution"],
    "RedTeam":       ["scale_variation", "occlusion", "out_of_view", "low_resolution"],
    "Rubik":         ["scale_variation", "in_plane_rotation", "out_of_plane_rotation"],
    "Shaking":       ["illumination_variation", "scale_variation", "in_plane_rotation", "out_of_plane_rotation", "background_clutter"],
    "Singer1":       ["illumination_variation", "scale_variation", "occlusion", "out_of_plane_rotation"],
    "Singer2":       ["illumination_variation", "in_plane_rotation", "out_of_plane_rotation"],
    "Skater":        ["scale_variation", "deformation", "in_plane_rotation", "out_of_plane_rotation"],
    "Skater2":       ["scale_variation", "deformation", "in_plane_rotation", "out_of_plane_rotation", "fast_motion"],
    "Skating1":      ["illumination_variation", "occlusion", "deformation"],
    "Skating2-1":    ["scale_variation", "occlusion", "deformation", "fast_motion"],
    "Skating2-2":    ["scale_variation", "occlusion", "deformation", "fast_motion"],
    "Skiing":        ["illumination_variation", "scale_variation", "deformation", "in_plane_rotation", "out_of_plane_rotation"],
    "Soccer":        ["illumination_variation", "scale_variation", "occlusion", "in_plane_rotation", "out_of_plane_rotation", "motion_blur", "fast_motion", "background_clutter"],
    "Subway":        ["occlusion", "deformation", "out_of_plane_rotation"],
    "Surfer":        ["scale_variation", "in_plane_rotation", "out_of_plane_rotation", "out_of_view", "fast_motion"],
    "Suv":           ["scale_variation", "occlusion", "out_of_view"],
    "Sylvester":     ["illumination_variation", "in_plane_rotation", "out_of_plane_rotation"],
    "Tiger1":        ["occlusion", "in_plane_rotation", "out_of_plane_rotation", "motion_blur", "fast_motion", "background_clutter"],
    "Tiger2":        ["scale_variation", "occlusion", "in_plane_rotation", "out_of_plane_rotation", "out_of_view", "motion_blur", "fast_motion", "background_clutter"],
    "Toy":           ["scale_variation", "in_plane_rotation", "out_of_plane_rotation"],
    "Trans":         ["illumination_variation", "scale_variation", "deformation"],
    "Trellis":       ["illumination_variation", "in_plane_rotation", "out_of_plane_rotation", "background_clutter"],
    "Twinnings":     ["scale_variation", "out_of_plane_rotation"],
    "Vase":          ["scale_variation", "in_plane_rotation", "out_of_plane_rotation"],
    "Walking":       ["scale_variation", "occlusion", "deformation"],
    "Walking2":      ["scale_variation", "occlusion"],
    "Woman":         ["scale_variation", "occlusion", "deformation", "out_of_plane_rotation", "fast_motion"],
}

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def get_sequence_attributes(name: str) -> List[str]:
    """Return the challenge attributes for an OTB sequence by name.

    Args:
        name: Sequence directory name (e.g. ``"Basketball"``).

    Returns:
        List of attribute strings, empty if the name is unknown.
    """
    return list(SEQUENCE_ATTRIBUTES.get(name, []))


class OTBTaggedSequence(Sequence):
    """A :class:`~eovot.datasets.base.Sequence` with OTB challenge attributes.

    Created by :class:`AttributeAwareOTBDataset`; not intended for direct
    construction outside tests.
    """

    def __init__(
        self,
        name: str,
        frame_paths: List[str],
        ground_truth: np.ndarray,
        attributes: Optional[List[str]] = None,
    ) -> None:
        super().__init__(name=name, frame_paths=frame_paths, ground_truth=ground_truth)
        self._attributes: List[str] = list(attributes or [])

    @property
    def attributes(self) -> List[str]:
        """Challenge attributes associated with this sequence."""
        return list(self._attributes)

    def has_attribute(self, attr: str) -> bool:
        """Return whether this sequence is tagged with *attr*."""
        return attr in self._attributes

    def __repr__(self) -> str:
        return (
            f"OTBTaggedSequence(name={self.name!r}, frames={len(self)}, "
            f"attributes={self._attributes})"
        )


class AttributeAwareOTBDataset(BaseDataset):
    """OTB-50 / OTB-100 dataset loader with challenge-attribute support.

    Extends the basic :class:`~eovot.datasets.base.OTBDataset` with:

    * **Subset filtering** — restrict to the 55-sequence OTB-50 set or the
      full 100-sequence OTB-100 set via the *subset* argument.
    * **Per-sequence attribute metadata** — each sequence exposes
      ``sequence.attributes`` (a list of challenge attribute strings).
    * **Attribute-based filtering** — pass ``attributes=[...]`` to load only
      sequences that have *at least one* of the listed attributes.
    * **Analytical helpers** — :meth:`sequences_by_attribute` and
      :meth:`attribute_distribution` for building per-attribute leaderboard rows.

    Args:
        root:          Path to the OTB dataset root containing one subdirectory
                       per sequence (e.g. ``/data/OTB100``).
        subset:        ``"OTB50"`` restricts loading to the original 55-sequence
                       subset; ``"OTB100"`` (default) loads all available sequences.
        attributes:    Optional list of challenge attributes.  Only sequences
                       tagged with **any** listed attribute are loaded.  Pass
                       ``None`` (default) to load all sequences in the subset.
        max_sequences: Maximum number of sequences to load.

    Example::

        from eovot.datasets.otb import AttributeAwareOTBDataset

        # Full OTB-100
        ds = AttributeAwareOTBDataset("/data/OTB100")

        # OTB-50 sequences that involve occlusion
        occ_50 = AttributeAwareOTBDataset(
            "/data/OTB100", subset="OTB50", attributes=["occlusion"]
        )

        # Attribute breakdown
        for attr, count in ds.attribute_distribution().items():
            print(f"{attr:30s}: {count} sequences")
    """

    #: The 11 OTB challenge attribute names.
    ATTRIBUTES = OTB_ATTRIBUTES

    def __init__(
        self,
        root: str,
        subset: str = "OTB100",
        attributes: Optional[Iterable[str]] = None,
        max_sequences: Optional[int] = None,
    ) -> None:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"OTB root not found: {root}")
        if subset not in ("OTB50", "OTB100"):
            raise ValueError(f"subset must be 'OTB50' or 'OTB100', got {subset!r}")

        self.root = root
        self._subset = subset
        self._attr_filter: Optional[Set[str]] = (
            set(attributes) if attributes is not None else None
        )
        self._sequences: List[OTBTaggedSequence] = []
        self._load(max_sequences)

    # ── public properties ──────────────────────────────────────────────────────

    @property
    def subset(self) -> str:
        """``"OTB50"`` or ``"OTB100"``."""
        return self._subset

    @property
    def name(self) -> str:
        return self._subset

    # ── loading ───────────────────────────────────────────────────────────────

    def _load(self, max_sequences: Optional[int]) -> None:
        loaded = 0
        for seq_name in sorted(os.listdir(self.root)):
            if max_sequences is not None and loaded >= max_sequences:
                break

            seq_dir = os.path.join(self.root, seq_name)
            if not os.path.isdir(seq_dir):
                continue

            if self._subset == "OTB50" and seq_name not in OTB50_SEQUENCES:
                continue

            gt_path = os.path.join(seq_dir, "groundtruth_rect.txt")
            if not os.path.isfile(gt_path):
                continue

            img_dir = os.path.join(seq_dir, "img")
            if not os.path.isdir(img_dir):
                img_dir = seq_dir

            frame_paths = sorted(
                os.path.join(img_dir, f)
                for f in os.listdir(img_dir)
                if os.path.splitext(f)[1].lower() in _IMG_EXTS
            )
            if not frame_paths:
                continue

            gt = self._parse_gt(gt_path)
            if len(gt) == 0:
                continue

            n = min(len(frame_paths), len(gt))
            frame_paths = frame_paths[:n]
            gt = gt[:n]

            attrs = SEQUENCE_ATTRIBUTES.get(seq_name, [])

            if self._attr_filter is not None:
                if not self._attr_filter.intersection(attrs):
                    continue

            self._sequences.append(
                OTBTaggedSequence(
                    name=seq_name,
                    frame_paths=frame_paths,
                    ground_truth=gt,
                    attributes=attrs,
                )
            )
            loaded += 1

    @staticmethod
    def _parse_gt(path: str) -> np.ndarray:
        rows: List[List[float]] = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) < 4:
                    continue
                try:
                    rows.append([float(parts[i]) for i in range(4)])
                except ValueError:
                    continue
        return np.array(rows, dtype=np.float64)

    # ── BaseDataset interface ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> OTBTaggedSequence:
        return self._sequences[idx]

    def __iter__(self) -> Iterator[OTBTaggedSequence]:
        return iter(self._sequences)

    # ── attribute helpers ─────────────────────────────────────────────────────

    def sequences_by_attribute(self, attribute: str) -> List[OTBTaggedSequence]:
        """Return sequences tagged with *attribute*.

        Args:
            attribute: One of :data:`OTB_ATTRIBUTES`.

        Returns:
            List of :class:`OTBTaggedSequence` that have this attribute.

        Raises:
            ValueError: If *attribute* is not a recognised OTB attribute.
        """
        if attribute not in OTB_ATTRIBUTES:
            raise ValueError(
                f"Unknown attribute '{attribute}'. "
                f"Valid choices: {OTB_ATTRIBUTES}"
            )
        return [s for s in self._sequences if s.has_attribute(attribute)]

    def attribute_distribution(self) -> Dict[str, int]:
        """Count sequences per challenge attribute in the loaded dataset.

        Returns:
            Ordered dict mapping attribute name to sequence count, sorted by
            descending count.
        """
        counts: Dict[str, int] = {a: 0 for a in OTB_ATTRIBUTES}
        for seq in self._sequences:
            for attr in seq.attributes:
                if attr in counts:
                    counts[attr] += 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def attribute_breakdown_summary(self) -> str:
        """Return a Markdown table of attribute distribution.

        Returns:
            Markdown string with a header and one row per attribute.
        """
        n = len(self._sequences) or 1
        dist = self.attribute_distribution()
        lines = [
            f"## {self._subset} Attribute Distribution ({n} sequences)\n",
            "| Attribute | Sequences | % of dataset |",
            "| --- | --- | --- |",
        ]
        for attr, count in dist.items():
            lines.append(f"| {attr} | {count} | {100 * count / n:.1f}% |")
        return "\n".join(lines) + "\n"

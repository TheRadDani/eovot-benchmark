"""OTB-100 dataset loader with per-sequence challenge-attribute filtering.

Extends the base :class:`~eovot.datasets.base.OTBDataset` with a built-in
attribute map for the 100 canonical OTB-100 sequences.  Researchers can
filter to sequences that exhibit specific visual challenges (occlusion, fast
motion, scale variation, etc.) to produce attribute-stratified benchmark
results — a standard analysis in visual tracking papers.

OTB-100 Challenge Attributes
-----------------------------
=====  =========================
Code   Attribute
=====  =========================
IV     Illumination Variation
SV     Scale Variation
OCC    Occlusion
DEF    Deformation
MB     Motion Blur
FM     Fast Motion
IPR    In-Plane Rotation
OPR    Out-of-Plane Rotation
OV     Out-of-View
BC     Background Clutter
LR     Low Resolution
=====  =========================

Reference
---------
Wu et al., "Object Tracking Benchmark." IEEE Transactions on Pattern
Analysis and Machine Intelligence (TPAMI), 2015.

Usage
-----
::

    from eovot.datasets.otb import OTBAttributeDataset, OTBAttribute

    # All sequences
    dataset = OTBAttributeDataset("/data/OTB100")

    # Only sequences with occlusion AND scale variation
    occ_sv = OTBAttributeDataset("/data/OTB100",
                                  attributes=[OTBAttribute.OCC, OTBAttribute.SV])

    # Get attribute summary table
    print(dataset.attribute_summary())
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Set

from .base import OTBDataset, Sequence


# ──────────────────────────────────────────────────────────────────────────────
# Attribute enum
# ──────────────────────────────────────────────────────────────────────────────


class OTBAttribute(Enum):
    """OTB-100 visual challenge attributes."""

    IV = auto()   # Illumination Variation
    SV = auto()   # Scale Variation
    OCC = auto()  # Occlusion
    DEF = auto()  # Deformation
    MB = auto()   # Motion Blur
    FM = auto()   # Fast Motion
    IPR = auto()  # In-Plane Rotation
    OPR = auto()  # Out-of-Plane Rotation
    OV = auto()   # Out-of-View
    BC = auto()   # Background Clutter
    LR = auto()   # Low Resolution

    def __str__(self) -> str:
        return self.name


# Short-hand aliases for convenience when building the attribute map below.
_A = OTBAttribute
IV, SV, OCC, DEF, MB, FM, IPR, OPR, OV, BC, LR = (
    _A.IV, _A.SV, _A.OCC, _A.DEF, _A.MB, _A.FM,
    _A.IPR, _A.OPR, _A.OV, _A.BC, _A.LR,
)


# ──────────────────────────────────────────────────────────────────────────────
# Built-in attribute map (OTB-100, Wu et al. TPAMI 2015, Table 1)
# ──────────────────────────────────────────────────────────────────────────────

#: Mapping from sequence directory name → frozenset of challenge attributes.
#: Covers all 100 canonical OTB-100 sequences.
OTB100_ATTRIBUTES: Dict[str, FrozenSet[OTBAttribute]] = {
    "Basketball":     frozenset({IV, OCC, DEF, OPR, BC}),
    "Biker":          frozenset({SV, OCC, MB, FM, OPR, OV, BC, LR}),
    "Bird1":          frozenset({DEF, FM, IPR}),
    "Bird2":          frozenset({SV, OCC, FM, IPR, OPR}),
    "BlurBody":       frozenset({SV, DEF, MB, FM, IPR, OPR}),
    "BlurCar1":       frozenset({SV, MB, FM}),
    "BlurCar2":       frozenset({SV, MB, FM}),
    "BlurCar3":       frozenset({SV, MB, FM}),
    "BlurCar4":       frozenset({SV, MB, FM}),
    "BlurFace":       frozenset({MB, FM, IPR}),
    "BlurOwl":        frozenset({SV, MB, FM, IPR, OPR}),
    "Board":          frozenset({SV, MB, FM, IPR, OV, BC, LR}),
    "Bolt":           frozenset({OCC, DEF, FM, IPR}),
    "Bolt2":          frozenset({DEF, FM}),
    "Box":            frozenset({IV, SV, OCC, MB, IPR, OPR, BC}),
    "Boy":            frozenset({SV, MB, FM, IPR}),
    "Car1":           frozenset({SV, MB, FM, BC, LR}),
    "Car2":           frozenset({SV, MB, FM, BC}),
    "Car24":          frozenset({IV, SV, BC}),
    "Car4":           frozenset({SV, MB, FM, BC}),
    "CarDark":        frozenset({IV, SV, BC}),
    "CarScale":       frozenset({SV, OCC, FM, IPR, OPR}),
    "ClifBar":        frozenset({SV, OCC, MB, FM, IPR, BC}),
    "Coke":           frozenset({IV, OCC, FM, IPR, BC}),
    "Couple":         frozenset({SV, FM, OPR, BC}),
    "Coupon":         frozenset({OCC, BC}),
    "Crossing":       frozenset({SV, OCC, FM, IPR, OPR, BC}),
    "Crowds":         frozenset({SV, OCC, IPR, OPR, BC}),
    "Dancer":         frozenset({SV, IPR, OPR}),
    "Dancer2":        frozenset({IV, SV}),
    "David":          frozenset({IV, SV, OCC, DEF, MB, IPR, OPR}),
    "David2":         frozenset({IPR, OPR}),
    "David3":         frozenset({OCC, DEF, IPR, OPR, BC}),
    "Deer":           frozenset({MB, FM, BC, LR}),
    "Diving":         frozenset({SV, DEF, IPR, OPR}),
    "Dog":            frozenset({SV, OCC, DEF, IPR, OPR}),
    "Dog1":           frozenset({SV, MB, FM, IPR, OPR}),
    "Doll":           frozenset({SV, OCC, MB, IPR, OPR}),
    "DragonBaby":     frozenset({SV, OCC, MB, FM, IPR, OPR}),
    "Dudek":          frozenset({SV, OCC, DEF, FM, IPR, BC}),
    "FaceOcc1":       frozenset({OCC}),
    "FaceOcc2":       frozenset({IV, OCC, IPR, OPR}),
    "Fish":           frozenset({IV, SV, MB, FM}),
    "FleetFace":      frozenset({SV, DEF, MB, FM, IPR, OPR}),
    "Football":       frozenset({OCC, IPR, OPR, BC}),
    "Football1":      frozenset({SV, OCC, IPR, OPR, BC}),
    "Freeman1":       frozenset({SV, OPR}),
    "Freeman3":       frozenset({SV, OCC, IPR, OPR}),
    "Freeman4":       frozenset({SV, OCC, IPR, OPR}),
    "Girl":           frozenset({SV, OCC, IPR, OPR}),
    "Girl2":          frozenset({SV, OCC, DEF, MB, FM}),
    "Gym":            frozenset({SV, DEF, IPR, OPR}),
    "Hand":           frozenset({DEF, MB, FM, IPR, OPR}),
    "Ironman":        frozenset({IV, SV, OCC, MB, FM, IPR, OPR, BC, LR}),
    "Jogging-1":      frozenset({OCC, DEF, OPR}),
    "Jogging-2":      frozenset({OCC, DEF, OPR}),
    "Jump":           frozenset({SV, MB, FM, IPR, OPR, BC}),
    "KiteSurf":       frozenset({IV, SV, OCC, OPR, BC}),
    "Lemming":        frozenset({IV, SV, OCC, FM, IPR, OV, BC}),
    "Liquor":         frozenset({IV, SV, OCC, MB, FM, OV, BC, LR}),
    "Man":            frozenset({IV, BC}),
    "Matrix":         frozenset({IV, SV, OCC, MB, FM, IPR, OPR, BC}),
    "Mhyang":         frozenset({IV, DEF, BC}),
    "MotorRolling":   frozenset({IV, SV, MB, FM, IPR, BC}),
    "MountainBike":   frozenset({SV, OCC, IPR, OPR, BC}),
    "Panda":          frozenset({IV, SV, OCC, DEF, OPR, BC, LR}),
    "RedTeam":        frozenset({SV, OCC, IPR, OPR, BC, LR}),
    "Rubik":          frozenset({SV, OCC, IPR, OPR}),
    "Shaking":        frozenset({IV, SV, IPR, OPR, BC}),
    "Singer1":        frozenset({IV, SV, OCC, IPR, OPR, BC}),
    "Singer2":        frozenset({IV, SV, IPR, OPR, BC}),
    "Skater":         frozenset({SV, OCC, DEF, FM, IPR, OPR}),
    "Skater2":        frozenset({SV, OCC, DEF, FM, IPR, OPR}),
    "Skating1":       frozenset({IV, SV, OCC, DEF, IPR, OPR, BC}),
    "Skating2-1":     frozenset({SV, OCC, DEF, FM, IPR, OPR}),
    "Skating2-2":     frozenset({SV, OCC, DEF, FM, IPR, OPR}),
    "Skiing":         frozenset({IV, SV, DEF, IPR, OPR}),
    "Skiing2":        frozenset({SV, OCC, FM, IPR, OPR, BC, LR}),
    "Skiing3":        frozenset({IV, OCC, DEF, FM, IPR, OPR}),
    "Slip":           frozenset({OCC, DEF, OPR, BC}),
    "Smoke":          frozenset({IV, SV, MB, FM, BC}),
    "Snake":          frozenset({IV, SV, OCC, DEF, IPR, OPR, BC}),
    "Soccer":         frozenset({IV, SV, OCC, MB, FM, IPR, OPR, BC}),
    "Subway":         frozenset({OCC, DEF, OPR, BC}),
    "Suitcase":       frozenset({IV, SV, OCC, MB, FM, OPR, BC, LR}),
    "Sunshade":       frozenset({IV, SV, OCC, DEF, BC}),
    "Surf":           frozenset({IV, SV, OCC, OPR, BC}),
    "Surfer":         frozenset({SV, OCC, FM, IPR, OPR, BC}),
    "Suv":            frozenset({OCC, FM, BC}),
    "Sylvester":      frozenset({IV, IPR, OPR}),
    "Tiger1":         frozenset({IV, OCC, MB, FM, IPR, OPR}),
    "Tiger2":         frozenset({IV, SV, OCC, MB, FM, IPR, OPR}),
    "Toy":            frozenset({SV, OCC, MB, FM, IPR, OV}),
    "Trans":          frozenset({IV, SV, OCC, DEF, FM, IPR, OPR, BC}),
    "Trellis":        frozenset({IV, SV, MB, FM, IPR, OPR, BC}),
    "Tunnel":         frozenset({IV, SV, MB, FM, BC}),
    "Vase":           frozenset({SV, DEF, IPR, OPR}),
    "Walking":        frozenset({IV, SV, OCC, OPR, BC}),
    "Walking2":       frozenset({SV, OCC, FM, BC, LR}),
    "Woman":          frozenset({IV, SV, OCC, DEF, OPR, BC}),
}


# ──────────────────────────────────────────────────────────────────────────────
# Dataset class
# ──────────────────────────────────────────────────────────────────────────────


class OTBAttributeDataset(OTBDataset):
    """OTB-style dataset loader with per-sequence attribute filtering.

    Wraps :class:`~eovot.datasets.base.OTBDataset` and adds the ability to
    restrict the sequence list to those that exhibit one or more specified
    visual challenge attributes.  This enables attribute-conditioned
    benchmarking — a standard analysis in tracking papers that reveals
    *where* a tracker succeeds or fails.

    Args:
        root: Path to the dataset root directory (same layout as
            :class:`~eovot.datasets.base.OTBDataset`).
        attributes: Optional collection of :class:`OTBAttribute` values.
            When provided, only sequences that contain **all** listed
            attributes are included.  Pass a single-element list to filter
            by one attribute.  If ``None`` or empty, all discovered
            sequences are included.
        attribute_map: Custom mapping from sequence name → attribute set,
            overriding :data:`OTB100_ATTRIBUTES`.  Useful for other
            OTB-style datasets or extended annotations.

    Example::

        from eovot.datasets.otb import OTBAttributeDataset, OTBAttribute

        # Sequences with fast motion only
        fm_dataset = OTBAttributeDataset(
            "/data/OTB100",
            attributes=[OTBAttribute.FM],
        )
        print(f"{len(fm_dataset)} fast-motion sequences")

        # Sequences with BOTH occlusion and scale variation
        hard = OTBAttributeDataset(
            "/data/OTB100",
            attributes=[OTBAttribute.OCC, OTBAttribute.SV],
        )
    """

    def __init__(
        self,
        root: str,
        attributes: Optional[List[OTBAttribute]] = None,
        attribute_map: Optional[Dict[str, FrozenSet[OTBAttribute]]] = None,
    ) -> None:
        super().__init__(root)
        self._attr_map: Dict[str, FrozenSet[OTBAttribute]] = (
            attribute_map if attribute_map is not None else OTB100_ATTRIBUTES
        )
        self._filter_attrs: FrozenSet[OTBAttribute] = (
            frozenset(attributes) if attributes else frozenset()
        )

        # Apply attribute filter to the entries discovered by OTBDataset.
        if self._filter_attrs:
            self._entries = [
                entry for entry in self._entries
                if self._filter_attrs.issubset(
                    self._attr_map.get(entry[0], frozenset())
                )
            ]

    def get_attributes(self, sequence_name: str) -> FrozenSet[OTBAttribute]:
        """Return the challenge attributes for a named sequence.

        Args:
            sequence_name: Directory name of the sequence.

        Returns:
            Frozenset of :class:`OTBAttribute` values, or an empty
            frozenset if the sequence is not in the attribute map.
        """
        return self._attr_map.get(sequence_name, frozenset())

    def sequences_with_attribute(self, attr: OTBAttribute) -> List[str]:
        """Return names of all loaded sequences that have *attr*.

        Args:
            attr: The challenge attribute to query.

        Returns:
            Sorted list of sequence names present in this dataset that
            include *attr* in their annotation.
        """
        return sorted(
            name for name, _ in self._entries
            if attr in self._attr_map.get(name, frozenset())
        )

    def attribute_counts(self) -> Dict[str, int]:
        """Count loaded sequences per attribute.

        Returns:
            Dict mapping attribute name string (e.g. ``"OCC"``) to the
            number of loaded sequences that have that attribute.
        """
        counts: Dict[str, int] = {a.name: 0 for a in OTBAttribute}
        for name, _ in self._entries:
            for attr in self._attr_map.get(name, frozenset()):
                counts[attr.name] += 1
        return counts

    def attribute_summary(self) -> str:
        """Return a formatted table of attribute counts for this dataset.

        Useful for sanity-checking the filter configuration at the start
        of a benchmark run.

        Returns:
            Multi-line string ready to print to stdout.
        """
        counts = self.attribute_counts()
        lines = [
            f"OTBAttributeDataset  root={self.root}",
            f"Sequences loaded: {len(self)}  "
            f"(filter: {{{', '.join(str(a) for a in sorted(self._filter_attrs, key=lambda x: x.name))}}})"
            if self._filter_attrs else f"Sequences loaded: {len(self)}  (no filter)",
            "",
            f"{'Attribute':<6}  {'Code':<5}  {'Count':>5}",
            "-" * 24,
        ]
        descriptions = {
            "IV": "Illumination Variation",
            "SV": "Scale Variation",
            "OCC": "Occlusion",
            "DEF": "Deformation",
            "MB": "Motion Blur",
            "FM": "Fast Motion",
            "IPR": "In-Plane Rotation",
            "OPR": "Out-of-Plane Rotation",
            "OV": "Out-of-View",
            "BC": "Background Clutter",
            "LR": "Low Resolution",
        }
        for code, desc in descriptions.items():
            lines.append(f"{code:<6}  {desc:<22}  {counts.get(code, 0):>5}")
        return "\n".join(lines)

    def __getitem__(self, idx: int) -> Sequence:
        """Return the *idx*-th sequence (post-filter)."""
        return super().__getitem__(idx)

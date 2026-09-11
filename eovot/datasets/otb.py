"""OTB-100 / OTB-50 dataset loaders with canonical sequence attributes.

Builds on the base :class:`~eovot.datasets.base.OTBDataset` to add:

* Canonical OTB-50 and OTB-100 sequence lists (Wu et al., CVPR 2013;
  TPAMI 2015).
* Per-sequence attribute annotations (IV, SV, OCC, DEF, MB, FM, IPR, OPR,
  OV, BC, LR) enabling attribute-stratified evaluation via
  :mod:`eovot.metrics.attributes`.
* :meth:`OTB100Dataset.filter_by_attribute` for convenient sub-selection.
* :class:`AttributedSequence` that carries the attribute set alongside
  frames and ground truth.

Expected directory layout (identical to the standard OTB release)::

    <root>/
      Basketball/
        img/0001.jpg ...
        groundtruth_rect.txt
      Biker/
        ...

Example::

    from eovot.datasets.otb import OTB100Dataset, OTB50Dataset

    ds = OTB100Dataset("/data/OTB100")
    fm_seqs = ds.filter_by_attribute("FM")   # fast-motion sequences only
    for seq in fm_seqs:
        print(seq.name, seq.attributes)
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List

import numpy as np

from .base import OTBDataset, Sequence

# ---------------------------------------------------------------------------
# Attribute constants (standard OTB shortcodes)
# ---------------------------------------------------------------------------

IV  = "IV"   # Illumination Variation
SV  = "SV"   # Scale Variation
OCC = "OCC"  # Occlusion
DEF = "DEF"  # Deformation
MB  = "MB"   # Motion Blur
FM  = "FM"   # Fast Motion
IPR = "IPR"  # In-Plane Rotation
OPR = "OPR"  # Out-of-Plane Rotation
OV  = "OV"   # Out-of-View
BC  = "BC"   # Background Clutter
LR  = "LR"   # Low Resolution

ALL_ATTRIBUTES: FrozenSet[str] = frozenset({IV, SV, OCC, DEF, MB, FM, IPR, OPR, OV, BC, LR})

ATTRIBUTE_NAMES: Dict[str, str] = {
    IV:  "Illumination Variation",
    SV:  "Scale Variation",
    OCC: "Occlusion",
    DEF: "Deformation",
    MB:  "Motion Blur",
    FM:  "Fast Motion",
    IPR: "In-Plane Rotation",
    OPR: "Out-of-Plane Rotation",
    OV:  "Out-of-View",
    BC:  "Background Clutter",
    LR:  "Low Resolution",
}

# ---------------------------------------------------------------------------
# Per-sequence attribute table (OTB-100)
# Source: Wu et al., "Object Tracking Benchmark", TPAMI 2015, Table 1.
# ---------------------------------------------------------------------------

_OTB100_ATTRIBUTES: Dict[str, FrozenSet[str]] = {
    "Basketball":   frozenset({IV, OCC, DEF, OPR, BC}),
    "Biker":        frozenset({SV, OCC, MB, FM, OPR, OV}),
    "Bird1":        frozenset({DEF, FM, OV}),
    "Bird2":        frozenset({SV, OCC, DEF, FM, IPR, OPR}),
    "BlurBody":     frozenset({SV, DEF, MB, FM, IPR, OPR}),
    "BlurCar1":     frozenset({SV, MB, FM}),
    "BlurCar2":     frozenset({SV, MB, FM}),
    "BlurCar3":     frozenset({SV, MB, FM}),
    "BlurCar4":     frozenset({SV, MB, FM}),
    "BlurFace":     frozenset({MB, FM, IPR}),
    "BlurOwl":      frozenset({SV, DEF, MB, FM, IPR, OPR}),
    "Board":        frozenset({SV, OCC, MB, FM, OPR, OV, BC}),
    "Bolt":         frozenset({OCC, DEF, IPR, OPR}),
    "Bolt2":        frozenset({DEF, BC}),
    "Box":          frozenset({IV, SV, OCC, MB, FM, IPR, OPR, OV, BC}),
    "Boy":          frozenset({SV, MB, FM, IPR, OPR}),
    "Car1":         frozenset({IV, SV, MB, FM, BC}),
    "Car2":         frozenset({IV, SV, MB, FM}),
    "Car4":         frozenset({IV, SV, MB, FM}),
    "Car24":        frozenset({IV, SV, MB, FM}),
    "CarDark":      frozenset({IV, BC}),
    "CarScale":     frozenset({SV, OCC, FM, IPR, OPR}),
    "ClifBar":      frozenset({SV, OCC, MB, FM, IPR, OV, BC}),
    "Coke":         frozenset({IV, OCC, FM, IPR, OPR, BC}),
    "Couple":       frozenset({SV, FM, OPR, BC}),
    "Coupon":       frozenset({OCC, BC}),
    "Crossing":     frozenset({SV, FM, IPR, OPR, BC}),
    "Cup":          frozenset({IV, OPR}),
    "David":        frozenset({IV, SV, OCC, DEF, MB, IPR, OPR}),
    "David2":       frozenset({IPR, OPR}),
    "David3":       frozenset({OCC, DEF, IPR, OPR, BC}),
    "Deer":         frozenset({MB, FM, BC}),
    "Diving":       frozenset({SV, DEF, OPR}),
    "Dog":          frozenset({SV, DEF, OPR}),
    "Dog1":         frozenset({SV, FM, OPR}),
    "Doll":         frozenset({SV, OCC, IPR, OPR, BC}),
    "DragonBaby":   frozenset({SV, OCC, MB, FM, IPR, OPR, OV}),
    "Dudek":        frozenset({SV, OCC, DEF, FM, IPR, OPR, BC}),
    "FaceOcc1":     frozenset({OCC}),
    "FaceOcc2":     frozenset({IV, OCC, DEF, IPR, OPR}),
    "Fish":         frozenset({IV, SV, DEF}),
    "FleetFace":    frozenset({SV, DEF, MB, FM, IPR, OPR}),
    "Football":     frozenset({OCC, IPR, OPR, BC}),
    "Football1":    frozenset({OCC, IPR, OPR, BC}),
    "Freeman1":     frozenset({SV, OPR}),
    "Freeman3":     frozenset({SV, OCC, OPR}),
    "Freeman4":     frozenset({SV, OCC, IPR, OPR}),
    "Girl":         frozenset({SV, OCC, IPR, OPR}),
    "Girl2":        frozenset({SV, OCC, DEF, IPR, OPR}),
    "Gym":          frozenset({SV, DEF, IPR, OPR}),
    "Human2":       frozenset({IV, SV, OCC, MB, FM, OPR, BC}),
    "Human3":       frozenset({OCC, DEF, FM, OPR, BC}),
    "Human4":       frozenset({IV, SV, OCC, DEF, FM, OPR}),
    "Human4_2":     frozenset({IV, SV, OCC, DEF, FM, OPR}),
    "Human5":       frozenset({IV, OCC, DEF, FM, OPR}),
    "Human6":       frozenset({IV, OCC, DEF, FM, OPR, OV}),
    "Human7":       frozenset({IV, SV, OCC, DEF, FM, OPR, OV}),
    "Human8":       frozenset({IV, OCC, DEF, FM, OPR}),
    "Human9":       frozenset({IV, OCC, DEF, FM, OPR}),
    "Ironman":      frozenset({IV, SV, OCC, MB, FM, IPR, OPR, OV, BC}),
    "Jogging":      frozenset({OCC, DEF, OPR}),
    "Jogging_1":    frozenset({OCC, DEF, OPR}),
    "Jogging_2":    frozenset({OCC, DEF, OPR}),
    "Jump":         frozenset({SV, MB, FM, IPR, OPR}),
    "Jumping":      frozenset({MB, FM}),
    "KiteSurf":     frozenset({IV, SV, OCC, DEF, OPR}),
    "Lemming":      frozenset({IV, SV, OCC, FM, OPR, OV}),
    "Liquor":       frozenset({IV, SV, OCC, MB, FM, OPR, BC}),
    "Man":          frozenset({IV, FM}),
    "Matrix":       frozenset({IV, SV, OCC, MB, FM, IPR, OPR, BC}),
    "Mhyang":       frozenset({IV, DEF, OPR, BC}),
    "MotorRolling": frozenset({IV, SV, MB, FM, IPR, BC}),
    "MountainBike": frozenset({IPR, OPR, BC}),
    "Panda":        frozenset({SV, OCC, DEF, OPR}),
    "RedTeam":      frozenset({SV, OCC, OPR}),
    "Rubik":        frozenset({IV, SV, IPR, OPR}),
    "Shaking":      frozenset({IV, SV, IPR, OPR, BC}),
    "Singer1":      frozenset({IV, SV, OCC, IPR, OPR}),
    "Singer2":      frozenset({IV, IPR, OPR}),
    "Skater":       frozenset({SV, DEF, IPR, OPR}),
    "Skater2":      frozenset({SV, DEF, FM, IPR, OPR}),
    "Skating1":     frozenset({IV, SV, OCC, DEF, BC}),
    "Skating2":     frozenset({SV, DEF, FM, IPR, OPR}),
    "Skating2_1":   frozenset({SV, DEF, FM, IPR, OPR}),
    "Skating2_2":   frozenset({SV, DEF, FM, IPR, OPR}),
    "Skiing":       frozenset({IV, OCC, DEF, IPR, OPR}),
    "Soccer":       frozenset({IV, SV, OCC, MB, FM, IPR, OPR, BC}),
    "Subway":       frozenset({OCC, DEF, BC}),
    "Surfer":       frozenset({IV, SV, OPR, BC, LR}),
    "Suv":          frozenset({OCC, FM, OV, BC}),
    "Sylvester":    frozenset({IV, IPR, OPR}),
    "Tiger1":       frozenset({IV, OCC, DEF, MB, FM, IPR, OPR}),
    "Tiger2":       frozenset({IV, SV, OCC, DEF, MB, FM, IPR, OPR}),
    "Toy":          frozenset({SV, IPR, OPR, BC}),
    "Trans":        frozenset({IV, SV, OCC, BC}),
    "Trellis":      frozenset({IV, SV, IPR, OPR, BC}),
    "Twinnings":    frozenset({SV, OPR}),
    "Vase":         frozenset({SV, IPR, OPR}),
    "Walking":      frozenset({SV, OCC, DEF}),
    "Walking2":     frozenset({SV, OCC, LR}),
    "Woman":        frozenset({IV, SV, OCC, DEF, MB, FM, OPR}),
}

# Canonical OTB-50 sequence list (Wu et al., CVPR 2013)
_OTB50_NAMES: FrozenSet[str] = frozenset({
    "Basketball", "Bolt", "Boy", "Car4", "CarDark", "CarScale", "ClifBar",
    "Couple", "Crossing", "David", "David2", "David3", "Deer", "Dog1",
    "Doll", "Dudek", "FaceOcc1", "FaceOcc2", "Fish", "FleetFace",
    "Football", "Freeman1", "Freeman3", "Freeman4", "Girl", "Ironman",
    "Jogging_1", "Jogging_2", "Jump", "Jumping", "Lemming", "Liquor",
    "Matrix", "Mhyang", "MotorRolling", "MountainBike", "Shaking",
    "Singer1", "Singer2", "Skating1", "Soccer", "Surfer", "Suv",
    "Sylvester", "Tiger1", "Tiger2", "Trellis", "Walking", "Walking2",
    "Woman",
})


class AttributedSequence(Sequence):
    """A :class:`~eovot.datasets.base.Sequence` with OTB attribute labels.

    Args:
        name:         Sequence name (directory basename).
        frame_paths:  Ordered list of frame file paths.
        ground_truth: Ground-truth bounding boxes, shape ``(N, 4)``.
        attributes:   Frozen set of OTB attribute codes present in this
                      sequence (e.g. ``frozenset({"IV", "OCC", "FM"}))``).
    """

    def __init__(
        self,
        name: str,
        frame_paths: List[str],
        ground_truth,
        attributes: FrozenSet[str],
    ) -> None:
        super().__init__(name, frame_paths, ground_truth)
        self.attributes: FrozenSet[str] = attributes

    def has_attribute(self, attr: str) -> bool:
        """Return ``True`` if this sequence carries the given attribute."""
        return attr in self.attributes

    def __repr__(self) -> str:
        attrs = sorted(self.attributes)
        return f"AttributedSequence(name={self.name!r}, attributes={attrs})"


class OTB100Dataset(OTBDataset):
    """OTB-100 dataset with per-sequence attribute annotations.

    Wraps :class:`~eovot.datasets.base.OTBDataset` and attaches the
    canonical challenge-attribute labels to every sequence, enabling
    attribute-stratified evaluation with :mod:`eovot.metrics.attributes`.

    Args:
        root: Path to the OTB dataset root.  Each sub-directory should
              contain an ``img/`` folder and ``groundtruth_rect.txt``.

    Example::

        ds = OTB100Dataset("/data/OTB100")
        occ_seqs = ds.filter_by_attribute("OCC")
        print(f"{len(occ_seqs)} sequences contain occlusion")
    """

    def __getitem__(self, idx: int) -> AttributedSequence:
        seq = super().__getitem__(idx)
        attrs = _OTB100_ATTRIBUTES.get(seq.name, frozenset())
        return AttributedSequence(
            name=seq.name,
            frame_paths=seq._frame_paths,
            ground_truth=seq.ground_truth,
            attributes=attrs,
        )

    def filter_by_attribute(self, attribute: str) -> List[AttributedSequence]:
        """Return all sequences that carry the given attribute label.

        Args:
            attribute: One of the 11 OTB attribute codes: ``IV``, ``SV``,
                       ``OCC``, ``DEF``, ``MB``, ``FM``, ``IPR``, ``OPR``,
                       ``OV``, ``BC``, ``LR``.

        Returns:
            List of :class:`AttributedSequence` sorted alphabetically by
            sequence name.

        Raises:
            ValueError: If *attribute* is not a recognised code.
        """
        if attribute not in ALL_ATTRIBUTES:
            raise ValueError(
                f"Unknown attribute {attribute!r}. "
                f"Valid codes: {sorted(ALL_ATTRIBUTES)}"
            )
        result = []
        for i in range(len(self)):
            name = self._entries[i][0]
            if attribute in _OTB100_ATTRIBUTES.get(name, frozenset()):
                result.append(self[i])
        return result

    def attribute_summary(self) -> Dict[str, int]:
        """Count sequences per attribute across the loaded dataset.

        Returns:
            Dict mapping each attribute code to the number of sequences
            in the dataset root that carry that attribute.
        """
        counts: Dict[str, int] = {a: 0 for a in ALL_ATTRIBUTES}
        for i in range(len(self)):
            name = self._entries[i][0]
            for attr in _OTB100_ATTRIBUTES.get(name, frozenset()):
                counts[attr] += 1
        return counts


class OTB50Dataset(OTB100Dataset):
    """OTB-50 dataset — the canonical 50-sequence subset of OTB-100.

    Filters the parent :class:`OTB100Dataset` to include only the 50
    sequences from Wu et al., CVPR 2013.  The root directory may contain
    all 100 OTB sequences; this class exposes only the OTB-50 subset.

    Args:
        root: Path to the OTB dataset root directory.
    """

    def _discover(self):  # type: ignore[override]
        all_entries = super()._discover()
        return [e for e in all_entries if e[0] in _OTB50_NAMES]

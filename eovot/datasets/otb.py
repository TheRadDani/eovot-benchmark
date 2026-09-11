"""OTB-100 and OTB-50 dataset loaders with canonical sequence attributes.

Attribute annotations sourced from:
  Wu et al., "Object Tracking Benchmark", TPAMI 2015
  Wu et al., "Online Object Tracking: A Benchmark", CVPR 2013
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List

from .base import OTBDataset, Sequence

# ---------------------------------------------------------------------------
# Attribute constants
# ---------------------------------------------------------------------------
IV = "IV"    # Illumination Variation
SV = "SV"    # Scale Variation
OCC = "OCC"  # Occlusion
DEF = "DEF"  # Deformation
MB = "MB"    # Motion Blur
FM = "FM"    # Fast Motion
IPR = "IPR"  # In-Plane Rotation
OPR = "OPR"  # Out-of-Plane Rotation
OV = "OV"    # Out-of-View
BC = "BC"    # Background Clutter
LR = "LR"    # Low Resolution

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
# Canonical OTB-100 sequence attributes (Wu et al., TPAMI 2015)
# ---------------------------------------------------------------------------
_OTB100_ATTRIBUTES: Dict[str, FrozenSet[str]] = {
    "Basketball":   frozenset({IV, OCC, DEF, OPR, BC}),
    "Biker":        frozenset({SV, OCC, MB, FM, OPR, OV}),
    "Bird1":        frozenset({FM, OV}),
    "Bird2":        frozenset({MB, FM, OV}),
    "BlurBody":     frozenset({DEF, MB, FM, OPR, SV}),
    "BlurCar1":     frozenset({MB, FM, SV}),
    "BlurCar2":     frozenset({MB, FM, SV}),
    "BlurCar3":     frozenset({MB, FM, SV}),
    "BlurCar4":     frozenset({MB, FM, SV}),
    "BlurFace":     frozenset({MB, FM, OPR}),
    "BlurOwl":      frozenset({MB, FM, OPR, SV}),
    "Bolt":         frozenset({OCC, DEF, FM, IPR, OPR}),
    "Bolt2":        frozenset({OCC, DEF, FM, OPR}),
    "Box":          frozenset({IV, SV, OCC, MB, FM, IPR, OPR, OV, BC}),
    "Boy":          frozenset({SV, FM, IPR, OPR}),
    "Car1":         frozenset({IV, SV, MB, FM, BC, LR}),
    "Car2":         frozenset({SV, FM, BC}),
    "Car4":         frozenset({IV, SV, OCC, BC}),
    "CarDark":      frozenset({IV, BC}),
    "CarScale":     frozenset({SV, OCC, FM, OPR}),
    "ClifBar":      frozenset({SV, OCC, FM, IPR, OPR, OV}),
    "Coke":         frozenset({IV, OCC, FM, IPR, OPR, BC}),
    "Couple":       frozenset({SV, FM, OPR}),
    "Coupon":       frozenset({OCC, BC}),
    "Crossing":     frozenset({SV, IPR, OPR, BC}),
    "Crowds":       frozenset({OPR, BC}),
    "David":        frozenset({IV, SV, OCC, DEF, MB, FM, IPR, OPR}),
    "David2":       frozenset({IPR, OPR}),
    "David3":       frozenset({OCC, DEF, OPR, BC}),
    "Deer":         frozenset({MB, FM, BC, LR}),
    "Diving":       frozenset({SV, DEF, IPR, OPR}),
    "Dog":          frozenset({SV, OCC, IPR, OPR}),
    "Dog1":         frozenset({SV, FM, OPR}),
    "Doll":         frozenset({SV, OCC, IPR, OPR}),
    "DragonBaby":   frozenset({SV, OCC, MB, FM, IPR, OPR}),
    "Dudek":        frozenset({SV, OCC, DEF, FM, IPR, OPR, BC}),
    "FaceOcc1":     frozenset({OCC}),
    "FaceOcc2":     frozenset({IV, OCC}),
    "Fish":         frozenset({IV, SV}),
    "FleetFace":    frozenset({SV, DEF, MB, FM, IPR, OPR}),
    "Football":     frozenset({OCC, OPR, BC}),
    "Football1":    frozenset({IV, SV, OCC, OPR, BC}),
    "Freeman1":     frozenset({SV, OCC, OPR}),
    "Freeman2":     frozenset({SV, OPR}),
    "Freeman3":     frozenset({SV, OCC, OPR}),
    "Freeman4":     frozenset({SV, OCC, OPR}),
    "Girl":         frozenset({SV, OCC, IPR, OPR}),
    "Girl2":        frozenset({SV, OCC, IPR, OPR}),
    "Gym":          frozenset({DEF, IPR, OPR}),
    "Human2":       frozenset({IV, SV, OCC, DEF, FM, OPR}),
    "Human3":       frozenset({OCC, DEF, OPR, BC}),
    "Human4":       frozenset({OCC, DEF, IPR, OPR, BC}),
    "Human5":       frozenset({OCC, DEF, OPR}),
    "Human6":       frozenset({OCC, DEF, OPR, BC}),
    "Human7":       frozenset({OCC, DEF, OPR, BC}),
    "Human8":       frozenset({OCC, DEF, OPR}),
    "Human9":       frozenset({OCC, IPR, OPR, BC}),
    "Ironman":      frozenset({IV, SV, OCC, MB, FM, IPR, OPR, OV, BC}),
    "Jogging-1":    frozenset({OCC, DEF, OPR}),
    "Jogging-2":    frozenset({OCC, DEF, OPR}),
    "Jump":         frozenset({SV, DEF, MB, FM, OPR}),
    "Jumping":      frozenset({MB, FM}),
    "KiteSurf":     frozenset({IV, SV, OCC, OPR}),
    "Lemming":      frozenset({SV, OCC, FM, OPR, OV}),
    "Liquor":       frozenset({IV, SV, OCC, MB, FM, OPR, OV, BC}),
    "Matrix":       frozenset({IV, SV, FM, IPR, OPR, OV, BC}),
    "Mhyang":       frozenset({IV, OCC, DEF, BC}),
    "MotorRolling": frozenset({IV, SV, MB, FM, BC}),
    "MountainBike": frozenset({OPR, BC}),
    "Panda":        frozenset({SV, OCC, DEF, OPR, BC}),
    "RedTeam":      frozenset({SV, OPR, BC}),
    "Rubik":        frozenset({SV, IPR, OPR}),
    "Shaking":      frozenset({IV, SV, IPR, OPR, BC}),
    "Singer1":      frozenset({IV, SV, OCC, OPR}),
    "Singer2":      frozenset({IV, SV, OCC, IPR, OPR}),
    "Skater":       frozenset({SV, DEF, IPR, OPR}),
    "Skater2":      frozenset({SV, OCC, DEF, IPR, OPR}),
    "Skating1":     frozenset({OCC, DEF, OPR, BC}),
    "Skating2":     frozenset({SV, OCC, DEF, OPR}),
    "Skiing":       frozenset({OCC, IPR, OPR}),
    "Soccer":       frozenset({IV, SV, OCC, MB, FM, OPR, BC}),
    "Subway":       frozenset({OCC, DEF, BC}),
    "Surfer":       frozenset({SV, OPR, BC}),
    "Suv":          frozenset({IV, OCC, FM, OV, BC}),
    "Sylvester":    frozenset({IV, IPR, OPR}),
    "Tiger1":       frozenset({OCC, MB, FM, IPR, OPR}),
    "Tiger2":       frozenset({IV, OCC, MB, FM, IPR, OPR}),
    "Toy":          frozenset({IV, SV, IPR, OPR}),
    "Trans":        frozenset({IV, SV, OCC, DEF}),
    "Trellis":      frozenset({IV, SV, IPR, OPR, BC}),
    "Twinnings":    frozenset({OCC, BC}),
    "Vase":         frozenset({SV, IPR, OPR}),
    "Walking":      frozenset({OCC, DEF, OPR}),
    "Walking2":     frozenset({SV, OCC, LR}),
    "Woman":        frozenset({IV, SV, OCC, DEF, OPR, BC}),
}

# OTB-50 sequence names (Wu et al., CVPR 2013)
_OTB50_NAMES: FrozenSet[str] = frozenset({
    "Basketball", "Biker",       "Bird1",       "BlurBody",    "BlurCar2",
    "BlurFace",   "BlurOwl",     "Bolt",        "Box",         "Boy",
    "Car1",       "Car4",        "CarDark",     "CarScale",    "ClifBar",
    "Couple",     "Crowds",      "David",       "Deer",        "Diving",
    "DragonBaby", "Dudek",       "Football",    "Freeman4",    "Girl",
    "Human3",     "Human4",      "Human6",      "Human9",      "Ironman",
    "Jump",       "Jumping",     "Liquor",      "Matrix",      "MotorRolling",
    "Panda",      "RedTeam",     "Shaking",     "Singer2",     "Skating1",
    "Skiing",     "Soccer",      "Surfer",      "Sylvester",   "Tiger1",
    "Tiger2",     "Trellis",     "Walking",     "Walking2",    "Woman",
})


class AttributedSequence(Sequence):
    """A tracking sequence annotated with per-sequence challenge attributes."""

    def __init__(
        self,
        name: str,
        frame_paths: list,
        ground_truth,
        attributes: FrozenSet[str] = frozenset(),
    ) -> None:
        super().__init__(name, frame_paths, ground_truth)
        self.attributes: FrozenSet[str] = frozenset(attributes)

    def has_attribute(self, attr: str) -> bool:
        """Return True if this sequence is annotated with *attr*."""
        return attr in self.attributes

    def __repr__(self) -> str:
        return (
            f"AttributedSequence(name={self.name!r}, frames={len(self)}, "
            f"attrs={sorted(self.attributes)})"
        )


class OTB100Dataset(OTBDataset):
    """OTB-100 dataset loader with canonical per-sequence attribute annotations.

    Each sequence is returned as an :class:`AttributedSequence` carrying the
    challenge attributes defined by Wu et al. (TPAMI 2015).  Unknown sequence
    names (not in the canonical 100) receive an empty attribute set.

    Args:
        root: Path to the OTB-100 dataset root directory.
    """

    def __getitem__(self, idx: int) -> AttributedSequence:
        seq = super().__getitem__(idx)
        return AttributedSequence(
            name=seq.name,
            frame_paths=seq._frame_paths,
            ground_truth=seq.ground_truth,
            attributes=_OTB100_ATTRIBUTES.get(seq.name, frozenset()),
        )

    def filter_by_attribute(self, attribute: str) -> List[AttributedSequence]:
        """Return all sequences that have *attribute* annotated.

        Args:
            attribute: One of the 11 OTB attribute constants
                       (``IV``, ``SV``, ``OCC``, etc.).

        Raises:
            ValueError: If *attribute* is not a recognised OTB attribute.
        """
        if attribute not in ALL_ATTRIBUTES:
            raise ValueError(
                f"Unknown attribute {attribute!r}. "
                f"Valid attributes: {sorted(ALL_ATTRIBUTES)}"
            )
        return [s for s in self if s.has_attribute(attribute)]

    def attribute_summary(self) -> Dict[str, int]:
        """Return per-attribute sequence counts for this dataset.

        Returns:
            Dict mapping each of the 11 attribute codes to the number of
            sequences in this dataset that carry it.
        """
        return {
            attr: len(self.filter_by_attribute(attr))
            for attr in sorted(ALL_ATTRIBUTES)
        }


class OTB50Dataset(OTB100Dataset):
    """OTB-50 dataset (the original 50 sequences from Wu et al., CVPR 2013).

    Subsets the OTB-100 root directory to only the canonical 50 sequences.
    All attribute functionality from :class:`OTB100Dataset` is inherited.

    Args:
        root: Path to the full OTB-100 dataset root directory
              (the 50 OTB-50 sequences must be present within it).
    """

    def _discover(self):
        return [
            (name, path)
            for name, path in super()._discover()
            if name in _OTB50_NAMES
        ]

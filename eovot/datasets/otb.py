"""Full-featured OTB-50 / OTB-100 dataset loader with challenge attribute annotations.

Extends the minimal :class:`~eovot.datasets.base.OTBDataset` with:

- **Split-aware loading**: ``"otb50"`` restricts to the 50 OTB-2013 sequences;
  ``"otb100"`` loads all 100 OTB-2015 sequences; ``"all"`` loads every
  valid sequence found on disk.
- **Official attribute annotations**: the 11 per-sequence challenge attributes
  from Wu et al. (TPAMI 2015) are stored as frozensets and usable for
  challenge-specific sub-evaluation.
- **Attribute-based filtering**: :meth:`OTB100Dataset.filter_by_attribute`
  returns a lightweight view restricted to sequences with the given attributes.
- **Convenience API**: :attr:`sequence_names`, :attr:`categories`,
  :meth:`attribute_coverage`, and :meth:`sequences_with_attribute`.

Dataset directory layout::

    <root>/
      Basketball/
        img/
          0001.jpg
          0002.jpg
          ...
        groundtruth_rect.txt   # x y w h per line (comma or space separated)
      Biker/
        ...

Challenge attributes (11 codes)::

    IV   Illumination Variation
    SV   Scale Variation
    OCC  Occlusion
    DEF  Deformation
    MB   Motion Blur
    FM   Fast Motion
    IPR  In-Plane Rotation
    OPR  Out-of-Plane Rotation
    OV   Out-of-View
    BC   Background Clutter
    LR   Low Resolution

References:
    Wu et al., "Object Tracking Benchmark." IEEE TPAMI 36(7):1320-1335, 2015.
    Wu et al., "Online Object Tracking: A Benchmark." CVPR 2013.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set

import numpy as np

from .base import BaseDataset, Sequence

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_GT_FILENAME = "groundtruth_rect.txt"
_IMG_DIR = "img"


# ---------------------------------------------------------------------------
# Attribute constants
# ---------------------------------------------------------------------------

#: Human-readable descriptions of the 11 OTB challenge attributes.
ATTRIBUTE_DESCRIPTIONS: Dict[str, str] = {
    "IV":  "Illumination Variation",
    "SV":  "Scale Variation",
    "OCC": "Occlusion",
    "DEF": "Deformation",
    "MB":  "Motion Blur",
    "FM":  "Fast Motion",
    "IPR": "In-Plane Rotation",
    "OPR": "Out-of-Plane Rotation",
    "OV":  "Out-of-View",
    "BC":  "Background Clutter",
    "LR":  "Low Resolution",
}

VALID_ATTRIBUTES: FrozenSet[str] = frozenset(ATTRIBUTE_DESCRIPTIONS)


# ---------------------------------------------------------------------------
# Canonical sequence lists
# ---------------------------------------------------------------------------

#: 50 sequences from OTB-2013 (Wu et al., CVPR 2013).
OTB50_SEQUENCES: FrozenSet[str] = frozenset([
    "Basketball", "Biker", "Bird1", "BlurBody", "BlurCar2", "BlurFace",
    "BlurOwl", "Bolt", "Box", "Car1", "Car4", "CarDark", "CarScale",
    "ClifBar", "Couple", "Crowds", "David", "Deer", "Diving", "DragonBaby",
    "Dudek", "Football", "Freeman4", "Girl", "Human3", "Human4-2", "Human6",
    "Human9", "Ironman", "Jump", "Jumping", "KiteSurf", "Lemming", "Man",
    "MotorRolling", "Panda", "RedTeam", "Shaking", "Singer2", "Skating1",
    "Skating2-1", "Skiing", "Soccer", "Surfer", "Sylvester", "Tiger2",
    "Trellis", "Walking", "Walking2", "Woman",
])

#: Additional sequences present in OTB-2015 (OTB-100) but NOT OTB-2013 (OTB-50).
_OTB100_EXTRA: FrozenSet[str] = frozenset([
    "Bird2", "BlurCar1", "BlurCar3", "BlurCar4", "Board", "Bolt2", "Boy",
    "Car2", "Car24", "Coke", "Coupon", "Crossing", "David2", "David3",
    "Dog", "Dog1", "FaceOcc1", "FaceOcc2", "Fish", "FleetFace", "Football1",
    "Freeman1", "Freeman3", "Girl2", "Gym", "Human1", "Human2", "Human4",
    "Human5", "Human7", "Human8", "Jogging-1", "Jogging-2", "Liquor",
    "Matrix", "MountainBike", "Polo", "Rubik", "Singer1", "Skater",
    "Skater2", "Skating2-2", "Subway", "Tiger1", "Toy", "Trans",
    "Twinnings", "Vase", "Wiper", "Yo-Yo",
])

#: 100 sequences from OTB-2015 (Wu et al., TPAMI 2015).
OTB100_SEQUENCES: FrozenSet[str] = OTB50_SEQUENCES | _OTB100_EXTRA


# ---------------------------------------------------------------------------
# Per-sequence attribute annotations (Wu et al., TPAMI 2015)
# ---------------------------------------------------------------------------

#: Official challenge attribute annotations.  Each key is a sequence directory
#: name; each value is a frozenset of attribute codes from
#: :data:`VALID_ATTRIBUTES`.  Sequences not present here return an empty set.
SEQUENCE_ATTRIBUTES: Dict[str, FrozenSet[str]] = {
    # -----------------------------------------------------------------------
    # OTB-50 sequences
    # -----------------------------------------------------------------------
    "Basketball":   frozenset(["OCC", "SV", "IPR", "OPR", "BC"]),
    "Biker":        frozenset(["SV", "OCC", "MB", "FM", "IPR", "OPR", "OV", "BC"]),
    "Bird1":        frozenset(["MB", "FM", "OV"]),
    "BlurBody":     frozenset(["DEF", "MB", "FM", "IPR", "OPR", "OV"]),
    "BlurCar2":     frozenset(["SV", "MB", "FM", "OV"]),
    "BlurFace":     frozenset(["MB", "FM", "IPR"]),
    "BlurOwl":      frozenset(["SV", "MB", "FM", "IPR", "OPR"]),
    "Bolt":         frozenset(["SV", "OCC", "FM", "IPR", "OPR"]),
    "Box":          frozenset(["IV", "SV", "OCC", "MB", "IPR", "OPR", "OV", "BC", "LR"]),
    "Car1":         frozenset(["SV", "MB", "FM", "BC", "LR"]),
    "Car4":         frozenset(["SV", "MB"]),
    "CarDark":      frozenset(["IV", "SV", "MB", "BC", "LR"]),
    "CarScale":     frozenset(["SV", "FM", "OV", "BC"]),
    "ClifBar":      frozenset(["SV", "MB", "FM", "IPR", "OPR", "OV", "BC"]),
    "Couple":       frozenset(["SV", "FM", "IPR", "OV", "BC"]),
    "Crowds":       frozenset(["SV", "OCC", "BC"]),
    "David":        frozenset(["IV", "SV", "OCC", "DEF", "MB", "IPR", "OPR"]),
    "Deer":         frozenset(["MB", "FM", "BC", "LR"]),
    "Diving":       frozenset(["SV", "DEF", "IPR", "OPR"]),
    "DragonBaby":   frozenset(["SV", "MB", "FM", "IPR", "OPR", "OV", "BC"]),
    "Dudek":        frozenset(["SV", "OCC", "DEF", "FM", "IPR", "OPR", "OV", "BC"]),
    "Football":     frozenset(["SV", "OCC", "MB", "IPR", "OPR", "BC"]),
    "Freeman4":     frozenset(["SV", "OCC", "IPR", "OPR"]),
    "Girl":         frozenset(["SV", "OCC", "IPR", "OPR"]),
    "Human3":       frozenset(["SV", "OCC", "DEF", "FM", "IPR", "OPR", "OV", "BC"]),
    "Human4-2":     frozenset(["SV", "DEF", "FM", "IPR", "OPR"]),
    "Human6":       frozenset(["SV", "OCC", "DEF", "FM", "OV", "BC"]),
    "Human9":       frozenset(["IV", "SV", "DEF", "MB", "FM", "OPR"]),
    "Ironman":      frozenset(["SV", "OCC", "MB", "IPR", "OPR", "OV", "BC"]),
    "Jump":         frozenset(["SV", "OCC", "DEF", "MB", "FM", "IPR", "OPR"]),
    "Jumping":      frozenset(["MB", "FM"]),
    "KiteSurf":     frozenset(["SV", "OCC", "IPR", "OPR"]),
    "Lemming":      frozenset(["SV", "OCC", "MB", "FM", "IPR", "OPR", "OV", "BC"]),
    "Man":          frozenset(["IV", "BC"]),
    "MotorRolling": frozenset(["IV", "SV", "MB", "FM", "BC"]),
    "Panda":        frozenset(["SV", "OCC", "DEF", "OV", "LR"]),
    "RedTeam":      frozenset(["SV", "OCC", "OV", "LR"]),
    "Shaking":      frozenset(["IV", "SV", "IPR", "OPR", "BC"]),
    "Singer2":      frozenset(["SV", "IPR", "OPR", "BC"]),
    "Skating1":     frozenset(["SV", "OCC", "DEF", "FM", "IPR", "OPR"]),
    "Skating2-1":   frozenset(["SV", "FM", "IPR", "OPR"]),
    "Skiing":       frozenset(["SV", "DEF", "IPR", "OPR"]),
    "Soccer":       frozenset(["IV", "SV", "OCC", "MB", "FM", "IPR", "OPR", "BC"]),
    "Surfer":       frozenset(["SV", "OCC", "IPR", "OPR", "LR"]),
    "Sylvester":    frozenset(["IV", "SV", "OCC", "IPR", "OPR", "BC"]),
    "Tiger2":       frozenset(["IV", "SV", "OCC", "DEF", "MB", "FM", "IPR", "OPR", "BC"]),
    "Trellis":      frozenset(["IV", "SV", "IPR", "OPR", "BC"]),
    "Walking":      frozenset(["SV", "OCC", "DEF", "FM", "OPR"]),
    "Walking2":     frozenset(["SV", "OCC", "LR"]),
    "Woman":        frozenset(["IV", "SV", "OCC", "DEF", "FM", "OPR"]),
    # -----------------------------------------------------------------------
    # OTB-100 additional sequences
    # -----------------------------------------------------------------------
    "Bird2":        frozenset(["MB", "FM", "IPR", "OPR", "BC"]),
    "BlurCar1":     frozenset(["MB", "FM"]),
    "BlurCar3":     frozenset(["MB", "FM"]),
    "BlurCar4":     frozenset(["MB", "FM", "OV"]),
    "Board":        frozenset(["IV", "SV", "MB", "FM", "OPR", "OV", "BC", "LR"]),
    "Bolt2":        frozenset(["DEF", "FM"]),
    "Boy":          frozenset(["SV", "MB", "FM", "IPR", "OPR"]),
    "Car2":         frozenset(["SV", "MB", "FM", "BC"]),
    "Car24":        frozenset(["IV", "SV", "MB", "BC", "LR"]),
    "Coke":         frozenset(["IV", "OCC", "IPR", "OPR", "BC"]),
    "Coupon":       frozenset(["OCC", "BC"]),
    "Crossing":     frozenset(["SV", "FM", "OV", "BC"]),
    "David2":       frozenset(["IPR", "OPR"]),
    "David3":       frozenset(["SV", "OCC", "FM", "BC"]),
    "Dog":          frozenset(["SV", "FM", "IPR", "OPR"]),
    "Dog1":         frozenset(["SV", "FM", "IPR", "OPR"]),
    "FaceOcc1":     frozenset(["OCC"]),
    "FaceOcc2":     frozenset(["SV", "OCC", "IPR", "OPR"]),
    "Fish":         frozenset(["SV", "IPR", "OPR"]),
    "FleetFace":    frozenset(["SV", "DEF", "MB", "FM", "IPR", "OPR"]),
    "Football1":    frozenset(["IPR", "OPR", "BC"]),
    "Freeman1":     frozenset(["SV", "OCC", "IPR", "OPR"]),
    "Freeman3":     frozenset(["SV", "IPR", "OPR"]),
    "Girl2":        frozenset(["SV", "OCC", "DEF", "FM", "IPR", "OPR"]),
    "Gym":          frozenset(["SV", "DEF", "IPR", "OPR"]),
    "Human1":       frozenset(["IV", "SV", "OCC", "DEF", "OV", "BC"]),
    "Human2":       frozenset(["IV", "SV", "DEF", "OPR"]),
    "Human4":       frozenset(["SV", "DEF", "FM", "IPR", "OPR"]),
    "Human5":       frozenset(["SV", "OCC", "DEF", "FM", "OPR", "BC"]),
    "Human7":       frozenset(["IV", "SV", "OCC", "DEF", "FM", "OPR", "BC"]),
    "Human8":       frozenset(["SV", "OCC", "DEF", "FM", "OPR"]),
    "Jogging-1":    frozenset(["OCC", "DEF", "OPR"]),
    "Jogging-2":    frozenset(["OCC", "DEF", "OPR"]),
    "Liquor":       frozenset(["IV", "SV", "OCC", "MB", "FM", "IPR", "OPR", "OV", "BC"]),
    "Matrix":       frozenset(["IV", "SV", "OCC", "MB", "FM", "IPR", "OPR", "BC"]),
    "MountainBike": frozenset(["SV", "OCC", "BC"]),
    "Polo":         frozenset(["SV", "OCC", "IPR", "OPR", "BC"]),
    "Rubik":        frozenset(["SV", "OCC", "IPR", "OPR", "BC"]),
    "Singer1":      frozenset(["IV", "SV", "IPR", "OPR", "BC"]),
    "Skater":       frozenset(["SV", "DEF", "FM", "IPR", "OPR"]),
    "Skater2":      frozenset(["SV", "DEF", "FM", "IPR", "OPR"]),
    "Skating2-2":   frozenset(["SV", "FM", "IPR", "OPR"]),
    "Subway":       frozenset(["OCC", "DEF", "BC"]),
    "Tiger1":       frozenset(["IV", "SV", "OCC", "DEF", "MB", "FM", "IPR", "OPR", "BC"]),
    "Toy":          frozenset(["SV", "IPR", "OPR", "BC"]),
    "Trans":        frozenset(["IV", "SV", "LR"]),
    "Twinnings":    frozenset(["SV", "IPR", "OPR"]),
    "Vase":         frozenset(["SV", "IPR", "OPR"]),
    "Wiper":        frozenset(["IV", "SV", "OCC", "DEF", "IPR"]),
    "Yo-Yo":        frozenset(["SV", "IPR", "OPR"]),
}


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------

class OTB100Dataset(BaseDataset):
    """OTB-50 / OTB-100 dataset loader with per-sequence attribute annotations.

    Supports split-aware loading and attribute-based sequence filtering for
    challenge-specific sub-evaluation (e.g., occlusion-only sequences).

    Args:
        root: Path to the OTB root directory.  Must exist.
        split: One of ``"otb50"``, ``"otb100"``, or ``"all"``.
            ``"otb50"`` — only the 50 OTB-2013 sequences.
            ``"otb100"`` — the full 100 OTB-2015 sequences.
            ``"all"`` — every sequence discovered on disk regardless of split.
            Default: ``"otb100"``.
        max_sequences: Optional cap applied after split filtering.
            Useful for quick smoke-tests.  Default: ``None``.
        name: Optional override for the :attr:`name` property.

    Raises:
        ValueError: If *split* is not a recognised value.
        FileNotFoundError: If *root* does not exist.

    Example::

        from eovot.datasets.otb import OTB100Dataset

        dataset = OTB100Dataset("/data/OTB100", split="otb100")
        print(len(dataset))           # ≤ 100

        # Evaluate only on occluded + fast-motion sequences
        hard = dataset.filter_by_attribute("OCC", "FM")
        print(len(hard))

        for seq in hard:
            print(seq.name, dataset.get_attributes(seq.name))
    """

    SPLITS = ("otb50", "otb100", "all")

    def __init__(
        self,
        root: str,
        split: str = "otb100",
        max_sequences: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        if split not in self.SPLITS:
            raise ValueError(
                f"split must be one of {self.SPLITS!r}, got {split!r}"
            )
        if not os.path.isdir(root):
            raise FileNotFoundError(f"OTB root directory not found: {root}")
        self.root = Path(root)
        self.split = split
        self.max_sequences = max_sequences
        self._name = name
        self._seq_dirs: List[Path] = self._discover()

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._seq_dirs)

    def __getitem__(self, idx: int) -> Sequence:
        if idx < 0 or idx >= len(self._seq_dirs):
            raise IndexError(
                f"Sequence index {idx} out of range [0, {len(self._seq_dirs)})"
            )
        return _load_sequence(self._seq_dirs[idx])

    def __repr__(self) -> str:
        return (
            f"OTB100Dataset(root={str(self.root)!r}, split={self.split!r}, "
            f"sequences={len(self)})"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable dataset name used in benchmark reports."""
        if self._name:
            return self._name
        if self.split == "otb50":
            return "OTB-50"
        if self.split == "otb100":
            return "OTB-100"
        return "OTB"

    @property
    def sequence_names(self) -> List[str]:
        """Sorted list of sequence directory names in this split."""
        return [d.name for d in self._seq_dirs]

    @property
    def categories(self) -> List[str]:
        """Sorted list of unique challenge attributes present in this split.

        Only sequences whose names appear in :data:`SEQUENCE_ATTRIBUTES` are
        considered.  Returns all 11 attributes when the full OTB-100 is loaded.
        """
        attrs: Set[str] = set()
        for seq_name in self.sequence_names:
            attrs |= set(SEQUENCE_ATTRIBUTES.get(seq_name, frozenset()))
        return sorted(attrs)

    # ------------------------------------------------------------------
    # Attribute API
    # ------------------------------------------------------------------

    def get_attributes(self, seq_name: str) -> FrozenSet[str]:
        """Return the official challenge attributes for a sequence by name.

        Args:
            seq_name: Sequence directory name (e.g. ``"Basketball"``).

        Returns:
            Frozenset of attribute codes, or an empty frozenset for unknown
            sequence names.
        """
        return SEQUENCE_ATTRIBUTES.get(seq_name, frozenset())

    def filter_by_attribute(self, *attrs: str) -> "OTB100Dataset":
        """Return a view containing only sequences that have ALL given attributes.

        Args:
            attrs: One or more attribute codes from :data:`VALID_ATTRIBUTES`
                (``"IV"``, ``"SV"``, ``"OCC"``, ``"DEF"``, ``"MB"``,
                ``"FM"``, ``"IPR"``, ``"OPR"``, ``"OV"``, ``"BC"``, ``"LR"``).

        Returns:
            A new :class:`OTB100Dataset` instance restricted to matching
            sequences.  Shares the parent's *root* and *split*; sets
            *max_sequences* to ``None``.

        Raises:
            ValueError: If any code in *attrs* is unrecognised.

        Example::

            occ_fm = dataset.filter_by_attribute("OCC", "FM")
            print(f"{len(occ_fm)} sequences with occlusion AND fast motion")
        """
        unknown = set(attrs) - VALID_ATTRIBUTES
        if unknown:
            raise ValueError(
                f"Unknown attribute(s) {sorted(unknown)!r}. "
                f"Valid: {sorted(VALID_ATTRIBUTES)}"
            )
        target = frozenset(attrs)
        filtered = [
            d for d in self._seq_dirs
            if target.issubset(SEQUENCE_ATTRIBUTES.get(d.name, frozenset()))
        ]
        view = object.__new__(OTB100Dataset)
        view.root = self.root
        view.split = self.split
        view.max_sequences = None
        view._name = f"{self.name}[{','.join(sorted(attrs))}]"
        view._seq_dirs = filtered
        return view

    def sequences_with_attribute(self, attr: str) -> List[str]:
        """Return a sorted list of sequence names that have *attr*.

        Args:
            attr: One attribute code from :data:`VALID_ATTRIBUTES`.

        Returns:
            Sorted list of matching sequence names (may be empty).

        Raises:
            ValueError: If *attr* is unrecognised.
        """
        if attr not in VALID_ATTRIBUTES:
            raise ValueError(
                f"Unknown attribute {attr!r}. Valid: {sorted(VALID_ATTRIBUTES)}"
            )
        return sorted(
            d.name for d in self._seq_dirs
            if attr in SEQUENCE_ATTRIBUTES.get(d.name, frozenset())
        )

    def attribute_coverage(self) -> Dict[str, int]:
        """Count the number of sequences in this split that carry each attribute.

        Returns:
            Dict mapping each attribute code to its sequence count, sorted
            alphabetically by code key.

        Example::

            cov = dataset.attribute_coverage()
            for attr, n in cov.items():
                print(f"{attr}: {n} sequences")
        """
        counts: Dict[str, int] = {a: 0 for a in sorted(VALID_ATTRIBUTES)}
        for seq_name in self.sequence_names:
            for a in SEQUENCE_ATTRIBUTES.get(seq_name, frozenset()):
                if a in counts:
                    counts[a] += 1
        return counts

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _discover(self) -> List[Path]:
        """Scan *root* for valid sequence directories and apply split filtering."""
        known: Optional[FrozenSet[str]] = None
        if self.split == "otb50":
            known = OTB50_SEQUENCES
        elif self.split == "otb100":
            known = OTB100_SEQUENCES

        seq_dirs: List[Path] = []
        for candidate in sorted(self.root.iterdir()):
            if not candidate.is_dir() or candidate.name.startswith("."):
                continue
            if not (candidate / _GT_FILENAME).is_file():
                continue
            if not (candidate / _IMG_DIR).is_dir():
                continue
            if known is not None and candidate.name not in known:
                continue
            seq_dirs.append(candidate)

        if not seq_dirs:
            raise ValueError(
                f"No valid sequences found in {self.root!r} for split={self.split!r}. "
                "Ensure the directory contains subdirectories with "
                "'groundtruth_rect.txt' and 'img/' inside."
            )

        if self.max_sequences is not None:
            seq_dirs = seq_dirs[: self.max_sequences]

        return seq_dirs


# ---------------------------------------------------------------------------
# Private loading helper
# ---------------------------------------------------------------------------

def _load_sequence(seq_dir: Path) -> Sequence:
    """Build a :class:`~eovot.datasets.base.Sequence` from *seq_dir*."""
    gt_path = seq_dir / _GT_FILENAME
    img_dir = seq_dir / _IMG_DIR

    try:
        gt = np.loadtxt(str(gt_path), delimiter=",")
    except ValueError:
        gt = np.loadtxt(str(gt_path))
    if gt.ndim == 1:
        gt = gt[np.newaxis, :]

    frame_paths = sorted(
        str(p) for p in img_dir.iterdir()
        if p.suffix.lower() in _IMG_EXTS
    )
    if not frame_paths:
        raise FileNotFoundError(f"No image frames found in {img_dir}")

    n = min(len(frame_paths), len(gt))
    return Sequence(
        name=seq_dir.name,
        frame_paths=frame_paths[:n],
        ground_truth=gt[:n].astype(np.float64),
    )

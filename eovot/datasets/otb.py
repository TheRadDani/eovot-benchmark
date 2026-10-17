"""OTB dataset loader for EOVOT.

OTB (Object Tracking Benchmark) is the foundational single-object tracking
benchmark, available in two variants:

- OTB-50: 50 representative sequences
- OTB-100: 100 sequences (superset of OTB-50)

Each sequence is annotated with up to 11 challenge attributes enabling
fine-grained analysis of tracker performance under specific visual conditions.

Challenge Attributes
--------------------
IV   – Illumination Variation
SV   – Scale Variation
OCC  – Occlusion
DEF  – Deformation
MB   – Motion Blur
FM   – Fast Motion
IPR  – In-Plane Rotation
OPR  – Out-of-Plane Rotation
OV   – Out-of-View
BC   – Background Clutter
LR   – Low Resolution

Dataset directory layout::

    OTB100/
    ├── Basketball/
    │   ├── img/
    │   │   ├── 0001.jpg
    │   │   └── ...
    │   └── groundtruth_rect.txt   # x,y,w,h per frame (comma or space delimited)
    ├── Biker/
    └── ...

References:
    Wu et al., "Object Tracking Benchmark." IEEE TPAMI 2015.
    Wu et al., "Online Object Tracking: A Benchmark." CVPR 2013.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from .base import BaseDataset, BBox, Sequence


# ---------------------------------------------------------------------------
# Attribute vocabulary
# ---------------------------------------------------------------------------

#: The 11 standard challenge attributes used in OTB-100.
OTB_ATTRIBUTES: FrozenSet[str] = frozenset([
    "IV",   # Illumination Variation
    "SV",   # Scale Variation
    "OCC",  # Occlusion
    "DEF",  # Deformation
    "MB",   # Motion Blur
    "FM",   # Fast Motion
    "IPR",  # In-Plane Rotation
    "OPR",  # Out-of-Plane Rotation
    "OV",   # Out-of-View
    "BC",   # Background Clutter
    "LR",   # Low Resolution
])

#: Human-readable names for each attribute code.
OTB_ATTRIBUTE_NAMES: Dict[str, str] = {
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

# ---------------------------------------------------------------------------
# Per-sequence attribute annotations (source: Wu et al. TPAMI 2015, Table 1)
# ---------------------------------------------------------------------------

#: Challenge attributes for every OTB-100 sequence.
OTB100_ATTRIBUTES: Dict[str, FrozenSet[str]] = {
    "Basketball":    frozenset(["IV", "OCC", "DEF", "BC"]),
    "Biker":         frozenset(["SV", "OCC", "MB", "FM", "OPR", "BC"]),
    "Bird1":         frozenset(["DEF", "FM", "OV"]),
    "Bird2":         frozenset(["SV", "OCC", "FM", "IPR", "OPR"]),
    "BlurBody":      frozenset(["SV", "DEF", "MB", "FM", "OPR"]),
    "BlurCar1":      frozenset(["SV", "MB", "FM"]),
    "BlurCar2":      frozenset(["SV", "MB", "FM"]),
    "BlurCar3":      frozenset(["MB", "FM"]),
    "BlurCar4":      frozenset(["MB", "FM"]),
    "BlurFace":      frozenset(["MB", "FM", "IPR", "OPR"]),
    "BlurOwl":       frozenset(["SV", "MB", "FM", "IPR", "OPR"]),
    "Board":         frozenset(["SV", "MB", "FM", "OPR", "OV", "BC", "LR"]),
    "Bolt":          frozenset(["OCC", "DEF", "FM", "IPR", "OPR"]),
    "Bolt2":         frozenset(["DEF", "BC"]),
    "Box":           frozenset(["IV", "SV", "OCC", "MB", "IPR", "OPR", "OV", "BC"]),
    "Boy":           frozenset(["SV", "MB", "FM", "IPR", "OPR"]),
    "Car1":          frozenset(["IV", "SV", "MB", "FM", "BC", "LR"]),
    "Car2":          frozenset(["IV", "SV", "FM", "BC"]),
    "Car24":         frozenset(["IV", "SV", "BC"]),
    "Car4":          frozenset(["IV", "SV"]),
    "CarDark":       frozenset(["IV", "BC"]),
    "CarScale":      frozenset(["SV", "OCC", "FM"]),
    "ClifBar":       frozenset(["SV", "OCC", "MB", "FM", "IPR", "OPR", "OV"]),
    "Coke":          frozenset(["IV", "OCC", "FM", "IPR", "OPR", "BC"]),
    "Couple":        frozenset(["SV", "DEF", "FM", "BC"]),
    "Crossing":      frozenset(["SV", "DEF", "FM", "BC"]),
    "Crowds":        frozenset(["DEF", "FM", "BC"]),
    "Dancer":        frozenset(["SV", "DEF", "IPR", "OPR"]),
    "Dancer2":       frozenset(["DEF", "IPR", "OPR"]),
    "David":         frozenset(["IV", "DEF", "MB", "IPR", "OPR"]),
    "David2":        frozenset(["IPR", "OPR"]),
    "David3":        frozenset(["OCC", "DEF", "FM", "IPR", "OPR", "BC"]),
    "Deer":          frozenset(["MB", "FM", "BC", "LR"]),
    "Diving":        frozenset(["SV", "DEF", "IPR", "OPR"]),
    "Dog":           frozenset(["SV", "DEF", "OPR"]),
    "Dog1":          frozenset(["SV", "FM", "IPR", "OPR"]),
    "Doll":          frozenset(["SV", "OCC", "IPR", "OPR"]),
    "DragonBaby":    frozenset(["SV", "OCC", "MB", "FM", "IPR", "OPR", "OV"]),
    "Dudek":         frozenset(["SV", "OCC", "DEF", "FM", "IPR", "OPR", "OV", "BC"]),
    "FaceOcc1":      frozenset(["OCC"]),
    "FaceOcc2":      frozenset(["IV", "OCC"]),
    "Fish":          frozenset(["IV", "SV", "DEF", "BC"]),
    "FleetFace":     frozenset(["SV", "DEF", "MB", "FM", "IPR", "OPR"]),
    "Football":      frozenset(["OCC", "IPR", "OPR", "BC"]),
    "Football1":     frozenset(["IV", "OCC", "IPR", "BC"]),
    "Freeman1":      frozenset(["SV", "OPR"]),
    "Freeman3":      frozenset(["SV", "OCC", "IPR", "OPR"]),
    "Freeman4":      frozenset(["SV", "OCC", "IPR", "OPR"]),
    "Girl":          frozenset(["SV", "OCC", "IPR", "OPR"]),
    "Girl2":         frozenset(["SV", "OCC", "DEF", "MB", "FM", "IPR"]),
    "Gym":           frozenset(["SV", "DEF", "IPR", "OPR"]),
    "Human2":        frozenset(["IV", "SV", "DEF", "OPR"]),
    "Human3":        frozenset(["SV", "OCC", "DEF", "FM", "IPR", "OPR", "LR"]),
    "Human4":        frozenset(["SV", "DEF", "OCC", "OPR"]),
    "Human5":        frozenset(["SV", "OCC", "DEF", "FM", "OPR"]),
    "Human6":        frozenset(["SV", "OCC", "DEF", "FM", "OPR"]),
    "Human7":        frozenset(["SV", "OCC", "DEF", "MB", "FM", "OPR"]),
    "Human8":        frozenset(["SV", "DEF", "OPR"]),
    "Human9":        frozenset(["IV", "SV", "DEF", "OPR"]),
    "Ironman":       frozenset(["IV", "SV", "OCC", "MB", "FM", "IPR", "OPR", "OV", "BC"]),
    "Jogging_1":     frozenset(["OCC", "DEF", "OPR"]),
    "Jogging_2":     frozenset(["OCC", "DEF", "OPR"]),
    "Jump":          frozenset(["SV", "DEF", "MB", "FM", "IPR", "OPR"]),
    "Jumping":       frozenset(["MB", "FM"]),
    "KiteSurf":      frozenset(["IV", "SV", "OCC", "IPR", "OPR"]),
    "Lemming":       frozenset(["IV", "SV", "OCC", "FM", "OV"]),
    "Liquor":        frozenset(["IV", "SV", "OCC", "FM", "IPR", "OPR", "OV", "BC"]),
    "Man":           frozenset(["IV", "DEF"]),
    "Matrix":        frozenset(["IV", "SV", "OCC", "MB", "FM", "IPR", "OPR", "OV", "BC"]),
    "Mhyang":        frozenset(["IV", "DEF", "BC"]),
    "MotorRolling":  frozenset(["IV", "SV", "MB", "FM", "IPR", "BC"]),
    "MountainBike":  frozenset(["IPR", "OPR", "BC"]),
    "Panda":         frozenset(["SV", "OCC", "DEF", "FM", "OPR", "OV", "LR"]),
    "RedTeam":       frozenset(["SV", "OCC", "OV", "LR"]),
    "Rubik":         frozenset(["SV", "IPR", "OPR"]),
    "Shaking":       frozenset(["IV", "SV", "IPR", "OPR", "BC"]),
    "Sim":           frozenset(["IV", "SV", "IPR", "OPR", "OV", "BC"]),
    "Singer1":       frozenset(["IV", "SV", "OCC", "IPR"]),
    "Singer2":       frozenset(["IV", "DEF", "IPR", "OPR"]),
    "Skater":        frozenset(["SV", "DEF", "IPR", "OPR"]),
    "Skater2":       frozenset(["SV", "DEF", "FM", "IPR", "OPR"]),
    "Skating1":      frozenset(["IV", "SV", "OCC", "DEF", "OPR"]),
    "Skating2_1":    frozenset(["SV", "OCC", "DEF", "FM", "IPR", "OPR"]),
    "Skating2_2":    frozenset(["SV", "OCC", "DEF", "FM", "IPR", "OPR"]),
    "Skiing":        frozenset(["IV", "DEF", "IPR", "OPR"]),
    "Soccer":        frozenset(["IV", "SV", "OCC", "MB", "FM", "IPR", "OPR", "BC"]),
    "Solid":         frozenset(["IV", "SV", "OV", "BC"]),
    "Spider":        frozenset(["SV", "DEF", "MB", "FM", "OV"]),
    "Sphere":        frozenset(["MB", "FM", "IPR"]),
    "Stone":         frozenset(["SV", "FM", "IPR", "BC", "LR"]),
    "Subway":        frozenset(["OCC", "DEF", "BC"]),
    "Surfer":        frozenset(["SV", "OCC", "IPR", "OPR", "OV", "BC", "LR"]),
    "Suv":           frozenset(["OCC", "IPR", "OV"]),
    "Sylvester":     frozenset(["IV", "IPR", "OPR"]),
    "Tiger1":        frozenset(["IV", "OCC", "DEF", "MB", "FM", "IPR", "OPR"]),
    "Tiger2":        frozenset(["IV", "SV", "OCC", "DEF", "MB", "FM", "IPR", "OPR"]),
    "Toy":           frozenset(["SV", "OCC", "MB", "FM", "IPR", "OPR"]),
    "Trans":         frozenset(["SV", "IPR", "OPR"]),
    "Trellis":       frozenset(["IV", "SV", "IPR", "OPR", "BC"]),
    "Truck":         frozenset(["IV", "SV", "BC"]),
    "Vase":          frozenset(["SV", "IPR", "OPR"]),
    "Walking":       frozenset(["SV", "OCC", "DEF"]),
    "Walking2":      frozenset(["SV", "OCC", "LR"]),
    "Woman":         frozenset(["IV", "SV", "OCC", "DEF", "FM", "OPR"]),
}

#: The 50 sequences that constitute the OTB-50 subset.
OTB50_SEQUENCES: FrozenSet[str] = frozenset([
    "Basketball", "Biker", "Bird1", "BlurBody", "BlurCar2", "BlurFace",
    "BlurOwl", "Board", "Bolt", "Box", "Car1", "Car4", "CarDark",
    "CarScale", "ClifBar", "Coke", "Couple", "Crossing", "David",
    "David2", "David3", "Deer", "Dog1", "Doll", "Dudek", "FaceOcc1",
    "FaceOcc2", "Fish", "FleetFace", "Football", "Football1",
    "Freeman1", "Freeman3", "Girl", "Human2", "Ironman", "Jogging_1",
    "Jogging_2", "Jump", "Jumping", "Lemming", "Liquor", "Matrix",
    "Mhyang", "MotorRolling", "MountainBike", "Shaking", "Singer1",
    "Skating1", "Skiing",
])

_IMG_EXTS: FrozenSet[str] = frozenset([".jpg", ".jpeg", ".png", ".bmp"])


class OTBDataset(BaseDataset):
    """Loader for OTB-50 and OTB-100 tracking benchmarks.

    Extends the basic OTB layout support with OTB-50 / OTB-100 variant
    selection, per-sequence challenge attribute annotations sourced from the
    original paper, and attribute-based filtering for per-challenge analysis.

    Args:
        root: Path to the OTB dataset root directory.  Must contain sequence
            subdirectories each with an ``img/`` folder and a
            ``groundtruth_rect.txt`` file.
        version: Dataset variant — ``"100"`` (default, all sequences) or
            ``"50"`` (50-sequence representative subset).
        attributes: Optional list of challenge attribute codes to filter on.
            Only sequences annotated with **all** listed attributes are kept.
            Valid codes: ``IV SV OCC DEF MB FM IPR OPR OV BC LR``.
            Pass ``None`` (default) to disable filtering.
        max_sequences: Optional cap on the number of sequences loaded.  Applied
            after version and attribute filtering.  Useful for quick smoke-tests.

    Example::

        # Full OTB-100
        dataset = OTBDataset("/data/OTB100")

        # OTB-50 subset only
        dataset = OTBDataset("/data/OTB100", version="50")

        # Only sequences with both Fast Motion and Occlusion
        dataset = OTBDataset("/data/OTB100", attributes=["FM", "OCC"])

        # Quick smoke-test: 5 sequences
        dataset = OTBDataset("/data/OTB100", max_sequences=5)
    """

    _GT_FILENAME = "groundtruth_rect.txt"
    _IMG_DIR = "img"

    def __init__(
        self,
        root: str,
        version: str = "100",
        attributes: Optional[List[str]] = None,
        max_sequences: Optional[int] = None,
    ) -> None:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"OTB dataset root not found: {root}")
        if version not in ("50", "100"):
            raise ValueError(f"version must be '50' or '100', got {version!r}")
        if attributes is not None:
            unknown = set(attributes) - OTB_ATTRIBUTES
            if unknown:
                raise ValueError(
                    f"Unknown attribute codes: {sorted(unknown)}. "
                    f"Valid codes: {sorted(OTB_ATTRIBUTES)}"
                )
        self.root = Path(root)
        self.version = version
        self.filter_attributes: Optional[FrozenSet[str]] = (
            frozenset(attributes) if attributes else None
        )
        self.max_sequences = max_sequences
        self._sequences: List[Tuple[str, Path]] = self._discover()

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> Sequence:
        if idx < 0 or idx >= len(self._sequences):
            raise IndexError(
                f"Sequence index {idx} out of range [0, {len(self._sequences)})"
            )
        name, seq_dir = self._sequences[idx]
        return self._load_sequence(name, seq_dir)

    # ------------------------------------------------------------------
    # Metadata API
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Dataset name for reports (e.g. ``"OTB-100"`` or ``"OTB-50[FM+OCC]"``)."""
        base = f"OTB-{self.version}"
        if self.filter_attributes:
            tag = "+".join(sorted(self.filter_attributes))
            return f"{base}[{tag}]"
        return base

    def sequence_names(self) -> List[str]:
        """Return an ordered list of sequence names for this dataset instance."""
        return [name for name, _ in self._sequences]

    def get_attributes(self, seq_name: str) -> FrozenSet[str]:
        """Return the challenge attributes annotated for *seq_name*.

        Returns an empty frozenset for sequences not in the OTB-100 annotation
        table (e.g. custom sequences added to the directory).

        Args:
            seq_name: Sequence folder name (e.g. ``"Basketball"``).
        """
        return OTB100_ATTRIBUTES.get(seq_name, frozenset())

    def attribute_map(self) -> Dict[str, FrozenSet[str]]:
        """Return a mapping of every loaded sequence name to its attributes.

        Useful for passing to :class:`~eovot.analysis.attribute_analyzer.AttributeAnalyzer`.
        """
        return {name: self.get_attributes(name) for name, _ in self._sequences}

    def sequences_by_attribute(self, attribute: str) -> List[str]:
        """Return all loaded sequence names that carry *attribute*.

        Args:
            attribute: One of the 11 OTB challenge attribute codes.

        Raises:
            ValueError: If *attribute* is not a recognised OTB code.
        """
        if attribute not in OTB_ATTRIBUTES:
            raise ValueError(
                f"Unknown attribute {attribute!r}. "
                f"Valid codes: {sorted(OTB_ATTRIBUTES)}"
            )
        return sorted(
            name
            for name, _ in self._sequences
            if attribute in OTB100_ATTRIBUTES.get(name, frozenset())
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _discover(self) -> List[Tuple[str, Path]]:
        """Collect valid (name, path) pairs applying all active filters."""
        entries: List[Tuple[str, Path]] = []
        for seq_dir in sorted(self.root.iterdir()):
            if not seq_dir.is_dir() or seq_dir.name.startswith("."):
                continue
            if not (seq_dir / self._GT_FILENAME).is_file():
                continue
            if not (seq_dir / self._IMG_DIR).is_dir():
                continue

            seq_name = seq_dir.name

            if self.version == "50" and seq_name not in OTB50_SEQUENCES:
                continue

            if self.filter_attributes:
                seq_attrs = OTB100_ATTRIBUTES.get(seq_name, frozenset())
                if not self.filter_attributes.issubset(seq_attrs):
                    continue

            entries.append((seq_name, seq_dir))

        if self.max_sequences is not None:
            entries = entries[: self.max_sequences]

        return entries

    def _load_sequence(self, name: str, seq_dir: Path) -> Sequence:
        gt = _load_groundtruth(seq_dir / self._GT_FILENAME)
        frame_paths = sorted(
            p for p in (seq_dir / self._IMG_DIR).iterdir()
            if p.suffix.lower() in _IMG_EXTS
        )
        if not frame_paths:
            raise FileNotFoundError(f"No image frames found in {seq_dir / self._IMG_DIR}")

        n = min(len(frame_paths), len(gt))
        return Sequence(
            name=name,
            frame_paths=[str(p) for p in frame_paths[:n]],
            ground_truth=np.array(gt[:n], dtype=np.float64),
        )


def _load_groundtruth(gt_path: Path) -> List[BBox]:
    """Parse a ``groundtruth_rect.txt`` into a list of ``(x, y, w, h)`` tuples.

    Handles comma-separated, space-separated, and tab-separated files.
    Malformed lines are skipped rather than raising, matching the tolerant
    style used by GOT-10k and LaSOT loaders.
    """
    boxes: List[BBox] = []
    with open(gt_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = [p for p in line.replace("\t", ",").replace(" ", ",").split(",") if p]
            if len(parts) < 4:
                continue
            try:
                x, y, w, h = (
                    float(parts[0]), float(parts[1]),
                    float(parts[2]), float(parts[3]),
                )
                boxes.append((x, y, w, h))
            except ValueError:
                continue
    return boxes

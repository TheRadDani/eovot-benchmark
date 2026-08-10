"""OTB-100 attribute-aware dataset loader for EOVOT.

Extends the basic :class:`~eovot.datasets.base.OTBDataset` with per-sequence
challenge attribute metadata drawn from the OTB benchmark paper:

    Wu, Y., Lim, J., & Yang, M.-H. (2015).
    Object tracking benchmark.
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    37(9), 1834–1848.

The eleven challenge attributes are:

    IV  — Illumination Variation
    SV  — Scale Variation
    OCC — Occlusion
    DEF — Deformation
    MB  — Motion Blur
    FM  — Fast Motion
    IPR — In-Plane Rotation
    OPR — Out-of-Plane Rotation
    OV  — Out-of-View
    BC  — Background Clutter
    LR  — Low Resolution

Usage::

    from eovot.datasets.otb import OTB100Dataset, ATTRIBUTE_DESCRIPTIONS

    dataset = OTB100Dataset("/data/OTB100")

    # Subset: sequences with fast motion
    fm_ds = dataset.filter_by_attribute("FM")
    print(f"{len(fm_ds)} fast-motion sequences")

    # Subset: both occlusion and scale variation
    hard_ds = dataset.filter_by_attribute("OCC", "SV")

    # Per-attribute sequence counts across the whole dataset
    dist = dataset.attribute_distribution()
    for attr, count in sorted(dist.items(), key=lambda kv: -kv[1]):
        print(f"  {attr} ({ATTRIBUTE_DESCRIPTIONS[attr]}): {count}")
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional

from .base import BaseDataset, OTBDataset, Sequence

# ---------------------------------------------------------------------------
# Public attribute constants
# ---------------------------------------------------------------------------

VALID_ATTRIBUTES: FrozenSet[str] = frozenset({
    "IV", "SV", "OCC", "DEF", "MB", "FM", "IPR", "OPR", "OV", "BC", "LR",
})

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

# ---------------------------------------------------------------------------
# Canonical OTB-100 attribute table
# Source: Wu et al. (TPAMI 2015) Table I and the official OTB toolkit.
# Keys are sequence directory names exactly as they appear on disk.
# ---------------------------------------------------------------------------
_ATTRIBUTES: Dict[str, FrozenSet[str]] = {
    "Basketball":   frozenset({"IV", "OCC", "DEF", "OPR", "BC"}),
    "Biker":        frozenset({"SV", "OCC", "MB", "FM", "IPR", "OPR", "OV", "LR"}),
    "Bird1":        frozenset({"DEF", "FM", "IPR", "OPR", "OV"}),
    "Bird2":        frozenset({"SV", "OCC", "FM", "IPR", "OPR"}),
    "BlurBody":     frozenset({"SV", "DEF", "MB", "FM", "IPR", "OPR"}),
    "BlurCar1":     frozenset({"SV", "MB", "FM"}),
    "BlurCar2":     frozenset({"SV", "MB", "FM"}),
    "BlurCar3":     frozenset({"SV", "MB", "FM"}),
    "BlurCar4":     frozenset({"SV", "MB", "FM"}),
    "BlurFace":     frozenset({"MB", "FM", "IPR", "OPR"}),
    "BlurOwl":      frozenset({"SV", "MB", "FM", "IPR", "OPR"}),
    "Board":        frozenset({"SV", "MB", "IPR", "OPR", "OV", "BC"}),
    "Boy":          frozenset({"SV", "MB", "FM", "IPR", "OPR"}),
    "Car1":         frozenset({"SV", "OCC", "MB", "FM", "LR"}),
    "Car2":         frozenset({"SV", "MB", "FM", "BC"}),
    "Car24":        frozenset({"SV", "MB", "FM", "BC"}),
    "Car4":         frozenset({"SV", "OCC", "MB", "FM"}),
    "CarDark":      frozenset({"IV", "SV", "OCC", "LR", "BC"}),
    "CarScale":     frozenset({"SV", "OCC", "FM", "IPR", "OPR"}),
    "ClifBar":      frozenset({"SV", "OCC", "MB", "FM", "IPR", "OPR", "BC"}),
    "Coke":         frozenset({"IV", "OCC", "FM", "IPR", "OPR", "BC"}),
    "Couple":       frozenset({"SV", "DEF", "FM", "BC"}),
    "Coupon":       frozenset({"OCC", "BC"}),
    "Crossing":     frozenset({"SV", "OCC", "DEF", "FM", "IPR"}),
    "Crowds":       frozenset({"DEF", "FM", "BC"}),
    "David":        frozenset({"IV", "OCC", "DEF", "MB", "IPR", "OPR", "LR"}),
    "David2":       frozenset({"IPR", "OPR"}),
    "David3":       frozenset({"OCC", "DEF", "IPR", "OPR", "BC"}),
    "Deer":         frozenset({"MB", "FM", "BC", "LR"}),
    "Diving":       frozenset({"SV", "DEF", "IPR"}),
    "Dog":          frozenset({"SV", "OCC", "DEF", "FM"}),
    "Dog1":         frozenset({"SV", "FM"}),
    "Doll":         frozenset({"SV", "OCC", "IPR", "OPR", "BC"}),
    "DragonBaby":   frozenset({"SV", "OCC", "MB", "FM", "IPR", "OPR"}),
    "Dudek":        frozenset({"SV", "OCC", "DEF", "FM", "IPR", "OPR", "BC"}),
    "FaceOcc1":     frozenset({"OCC"}),
    "FaceOcc2":     frozenset({"IV", "OCC", "DEF", "FM", "IPR", "OPR"}),
    "Fish":         frozenset({"IV", "SV", "OCC", "DEF", "BC"}),
    "FleetFace":    frozenset({"SV", "DEF", "MB", "FM", "IPR", "OPR"}),
    "Football":     frozenset({"IV", "OCC", "DEF", "FM", "IPR", "OPR", "BC"}),
    "Football1":    frozenset({"IV", "OCC", "IPR", "OPR", "BC"}),
    "Freeman1":     frozenset({"SV", "OCC", "IPR", "OPR"}),
    "Freeman3":     frozenset({"SV", "OCC", "DEF", "IPR", "OPR"}),
    "Freeman4":     frozenset({"SV", "OCC", "DEF", "IPR", "OPR"}),
    "Girl":         frozenset({"SV", "OCC", "IPR", "OPR"}),
    "Girl2":        frozenset({"SV", "OCC", "DEF", "MB", "FM", "IPR", "OPR"}),
    "Gym":          frozenset({"SV", "DEF", "IPR"}),
    "Hand":         frozenset({"SV", "DEF", "MB", "FM", "IPR", "OPR"}),
    "Ironman":      frozenset({"SV", "OCC", "MB", "FM", "IPR", "OPR", "BC"}),
    "Jogging1":     frozenset({"OCC", "DEF", "OPR", "BC"}),
    "Jogging2":     frozenset({"OCC", "DEF", "OPR", "BC"}),
    "Jump":         frozenset({"SV", "OCC", "DEF", "MB", "FM", "IPR", "OPR"}),
    "Jumping":      frozenset({"MB", "FM"}),
    "KiteSurf":     frozenset({"SV", "OCC", "DEF", "IPR", "OPR", "LR"}),
    "Lemming":      frozenset({"IV", "SV", "OCC", "DEF", "FM", "IPR", "OPR"}),
    "Liquor":       frozenset({"IV", "SV", "OCC", "FM", "BC"}),
    "Man":          frozenset({"SV", "FM", "LR", "BC"}),
    "Matrix":       frozenset({"IV", "OCC", "FM", "IPR", "OPR", "BC"}),
    "Mhyang":       frozenset({"IV", "DEF", "BC"}),
    "MotorRolling": frozenset({"IV", "SV", "MB", "FM", "IPR", "BC", "LR"}),
    "MountainBike": frozenset({"IPR", "OPR", "BC"}),
    "Panda":        frozenset({"SV", "OCC", "DEF", "FM", "OPR", "OV", "LR", "BC"}),
    "RedTeam":      frozenset({"SV", "OCC", "FM", "LR"}),
    "Rubik":        frozenset({"IV", "SV", "IPR", "OPR", "BC"}),
    "Shaking":      frozenset({"IV", "SV", "IPR", "OPR", "BC"}),
    "Singer1":      frozenset({"IV", "SV", "OCC", "BC"}),
    "Singer2":      frozenset({"IV", "DEF", "IPR", "OPR"}),
    "Skating1":     frozenset({"IV", "OCC", "DEF", "BC"}),
    "Skating2_1":   frozenset({"SV", "OCC", "DEF", "MB", "FM", "IPR", "OPR"}),
    "Skating2_2":   frozenset({"SV", "OCC", "DEF", "MB", "FM", "IPR", "OPR"}),
    "Skiing":       frozenset({"IV", "SV", "DEF", "MB", "FM", "IPR", "OPR"}),
    "Soccer":       frozenset({"IV", "SV", "OCC", "MB", "FM", "IPR", "OPR", "BC"}),
    "Subway":       frozenset({"OCC", "DEF", "BC"}),
    "Surfer":       frozenset({"SV", "FM", "IPR", "OPR", "BC"}),
    "Suv":          frozenset({"SV", "OCC", "FM", "BC"}),
    "Sylvester":    frozenset({"IV", "IPR", "OPR"}),
    "Tiger1":       frozenset({"IV", "OCC", "DEF", "MB", "FM", "IPR", "OPR", "BC"}),
    "Tiger2":       frozenset({"IV", "OCC", "DEF", "MB", "FM", "IPR", "OPR", "BC"}),
    "Toy":          frozenset({"SV", "IPR", "OPR", "BC"}),
    "Trans":        frozenset({"IV", "SV", "DEF", "BC"}),
    "Trellis":      frozenset({"IV", "SV", "IPR", "OPR", "BC"}),
    "Twinnings":    frozenset({"SV", "OCC", "IPR", "OPR"}),
    "Vase":         frozenset({"SV", "IPR", "OPR"}),
    "Walking":      frozenset({"IV", "OCC", "DEF", "OPR", "LR"}),
    "Walking2":     frozenset({"SV", "OCC", "LR"}),
    "Woman":        frozenset({"IV", "SV", "OCC", "DEF", "FM", "OPR", "BC"}),
}


# ---------------------------------------------------------------------------
# AttributedSequence
# ---------------------------------------------------------------------------

class AttributedSequence(Sequence):
    """A :class:`~eovot.datasets.base.Sequence` enriched with OTB challenge attributes.

    The ``attributes`` frozenset contains the OTB attribute codes that apply
    to this sequence (e.g. ``frozenset({"FM", "OCC", "SV"})``).

    Attributes are set at construction time and never mutated.

    Example::

        seq = dataset[0]
        if seq.has_attribute("FM"):
            print(f"{seq.name} contains fast motion")
    """

    def __init__(
        self,
        name: str,
        frame_paths: list,
        ground_truth,
        attributes: FrozenSet[str] = frozenset(),
    ) -> None:
        super().__init__(name=name, frame_paths=frame_paths, ground_truth=ground_truth)
        self.attributes: FrozenSet[str] = attributes

    def has_attribute(self, attr: str) -> bool:
        """Return ``True`` if this sequence carries challenge attribute *attr*.

        Args:
            attr: OTB attribute code, e.g. ``"FM"`` or ``"OCC"``.
        """
        return attr in self.attributes

    def __repr__(self) -> str:
        attrs = ", ".join(sorted(self.attributes)) if self.attributes else "none"
        return (
            f"AttributedSequence(name={self.name!r}, "
            f"frames={len(self)}, attrs=[{attrs}])"
        )


# ---------------------------------------------------------------------------
# OTB100Dataset
# ---------------------------------------------------------------------------

class OTB100Dataset(OTBDataset):
    """OTB-100 dataset loader with per-sequence challenge attribute support.

    Inherits all file-discovery and GT-loading logic from
    :class:`~eovot.datasets.base.OTBDataset` and adds the attribute
    metadata from Wu et al. (TPAMI 2015).

    Sequences not found in the built-in attribute table (e.g. custom
    sequences added alongside the standard 100) are loaded with an empty
    ``frozenset()`` and are included in ``filter_by_attribute`` only when
    no attributes are required (i.e. zero-attribute filter).

    Args:
        root:       Path to the OTB-100 root directory.
                    Expected layout::

                        <root>/<seq_name>/img/*.jpg
                        <root>/<seq_name>/groundtruth_rect.txt

        attributes: Optional mapping ``{seq_name: frozenset(codes)}`` that
                    overrides or extends the built-in table.  Useful for
                    correcting entries or adding custom sequence metadata.

    Example::

        ds = OTB100Dataset("/data/OTB100")

        # Only fast-motion sequences
        fm = ds.filter_by_attribute("FM")

        # Sequences with both occlusion and scale change
        hard = ds.filter_by_attribute("OCC", "SV")

        # Attribute counts across all discovered sequences
        print(ds.attribute_distribution())
    """

    def __init__(
        self,
        root: str,
        attributes: Optional[Dict[str, FrozenSet[str]]] = None,
    ) -> None:
        super().__init__(root)
        self._attr_table: Dict[str, FrozenSet[str]] = dict(_ATTRIBUTES)
        if attributes:
            self._attr_table.update(attributes)

    # ------------------------------------------------------------------
    # Overridden accessor
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> AttributedSequence:
        """Return sequence *idx* as an :class:`AttributedSequence`."""
        base: Sequence = super().__getitem__(idx)
        attrs = self._attr_table.get(base.name, frozenset())
        return AttributedSequence(
            name=base.name,
            frame_paths=base._frame_paths,
            ground_truth=base.ground_truth,
            attributes=attrs,
        )

    # ------------------------------------------------------------------
    # Attribute-based filtering
    # ------------------------------------------------------------------

    def filter_by_attribute(self, *attrs: str) -> "_OTBSubset":
        """Return a view containing only sequences that have ALL specified attributes.

        Args:
            *attrs: One or more OTB attribute codes (e.g. ``"FM"``, ``"OCC"``).

        Returns:
            :class:`_OTBSubset` — a lightweight index view of this dataset.

        Raises:
            ValueError: If any code is not a recognised OTB attribute.

        Example::

            # Fast-motion AND out-of-view sequences
            ds.filter_by_attribute("FM", "OV")
        """
        unknown = set(attrs) - VALID_ATTRIBUTES
        if unknown:
            raise ValueError(
                f"Unknown attribute code(s): {unknown}. "
                f"Valid codes: {sorted(VALID_ATTRIBUTES)}"
            )
        required = frozenset(attrs)
        indices = [
            i for i in range(len(self))
            if required <= self._attr_table.get(self._entries[i][0], frozenset())
        ]
        return _OTBSubset(self, indices)

    def filter_by_names(self, names: List[str]) -> "_OTBSubset":
        """Return a view restricted to the named sequences.

        Args:
            names: Sequence directory names (case-sensitive).

        Returns:
            :class:`_OTBSubset` preserving the original dataset ordering.
        """
        name_set = set(names)
        indices = [
            i for i in range(len(self))
            if self._entries[i][0] in name_set
        ]
        return _OTBSubset(self, indices)

    # ------------------------------------------------------------------
    # Attribute statistics
    # ------------------------------------------------------------------

    def attribute_distribution(self) -> Dict[str, int]:
        """Count how many *discovered* sequences carry each OTB attribute.

        Only sequences actually present on disk are counted; entries in
        ``_ATTRIBUTES`` that are not in ``self._entries`` are ignored.

        Returns:
            Dict mapping each of the 11 OTB attribute codes to the number
            of sequences in this dataset that carry it.

        Example::

            dist = dataset.attribute_distribution()
            # {"BC": 31, "OCC": 49, "SV": 64, ...}
        """
        counts: Dict[str, int] = {attr: 0 for attr in VALID_ATTRIBUTES}
        for seq_name, _ in self._entries:
            for attr in self._attr_table.get(seq_name, frozenset()):
                if attr in counts:
                    counts[attr] += 1
        return counts

    def sequences_with_attribute(self, attr: str) -> List[str]:
        """Return the names of all *discovered* sequences that carry *attr*.

        Args:
            attr: An OTB attribute code (e.g. ``"LR"``).

        Returns:
            Sorted list of sequence directory names.

        Raises:
            ValueError: If *attr* is not a recognised OTB attribute code.
        """
        if attr not in VALID_ATTRIBUTES:
            raise ValueError(
                f"Unknown attribute {attr!r}. "
                f"Valid codes: {sorted(VALID_ATTRIBUTES)}"
            )
        present = {name for name, _ in self._entries}
        return sorted(
            name
            for name, attrs in self._attr_table.items()
            if attr in attrs and name in present
        )

    def __repr__(self) -> str:
        return f"OTB100Dataset(root={self.root!r}, sequences={len(self)})"


# ---------------------------------------------------------------------------
# _OTBSubset
# ---------------------------------------------------------------------------

class _OTBSubset(BaseDataset):
    """Index-filtered view of an :class:`OTB100Dataset`.

    Returned by :meth:`OTB100Dataset.filter_by_attribute` and
    :meth:`OTB100Dataset.filter_by_names`; not constructed directly.

    All sequences in the subset are :class:`AttributedSequence` instances
    so attribute-aware code can work uniformly on both the full dataset
    and any filtered view.
    """

    def __init__(self, parent: OTB100Dataset, indices: List[int]) -> None:
        self._parent = parent
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> AttributedSequence:
        if idx < 0 or idx >= len(self._indices):
            raise IndexError(
                f"Subset index {idx} out of range [0, {len(self._indices)})"
            )
        return self._parent[self._indices[idx]]

    def attribute_distribution(self) -> Dict[str, int]:
        """Attribute distribution restricted to this subset.

        Useful for verifying that a filtered view actually contains the
        expected challenge mix.

        Returns:
            Dict mapping each OTB attribute code to its count within the subset.
        """
        counts: Dict[str, int] = {attr: 0 for attr in VALID_ATTRIBUTES}
        for i in range(len(self)):
            for attr in self[i].attributes:
                if attr in counts:
                    counts[attr] += 1
        return counts

    def __repr__(self) -> str:
        return f"_OTBSubset(size={len(self._indices)})"

"""UAV123 / UAV20L dataset loader for EOVOT.

UAV123 is an aerial object tracking benchmark comprising 123 sequences (UAV123
split) and a 20-sequence long subset (UAV20L) filmed from drone/UAV camera
platforms.  It is particularly relevant for edge deployment benchmarks where
inference must run on airborne, resource-constrained hardware.

Dataset directory layout expected by :class:`UAV123Dataset`::

    <root>/
    ├── data_seq/
    │   └── UAV123/
    │       ├── bike1/
    │       │   ├── img00001.jpg
    │       │   └── ...
    │       └── ...
    └── anno/
        ├── UAV123/
        │   ├── bike1.txt          # x,y,w,h  (comma-sep, one row per frame)
        │   └── ...
        └── UAV20L/
            ├── bike1.txt
            └── ...

Ground-truth format:
    Each annotation file contains one line per frame with four comma-separated
    values: ``x,y,w,h`` (top-left corner + width/height, 1-based integer pixel
    coordinates as published by the benchmark).

Reference:
    Mueller et al., "A Benchmark and Simulator for UAV Tracking."
    ECCV 2016. https://doi.org/10.1007/978-3-319-46448-0_27
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

from .base import BaseDataset, BBox, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attribute catalogue (curated from the UAV123 paper supplemental material)
# ---------------------------------------------------------------------------

#: Mapping from attribute name → set of UAV123 sequence names that carry it.
#: Used by :meth:`UAV123Dataset.filter_by_attribute` and
#: :attr:`UAV123Dataset.attributes`.
SEQUENCE_ATTRIBUTES: Dict[str, Set[str]] = {
    "aerial_view": {
        "bike1", "bike2", "bike3", "bird1",
        "building1", "building2", "building3",
        "car1", "car1_s", "car2", "car2_s", "car3", "car3_s",
        "car4", "car4_s", "car5", "car6_1", "car6_2", "car6_3", "car6_4", "car6_5",
        "car7", "car8_1", "car8_2", "car9", "car10", "car11", "car12", "car13",
        "car14", "car15", "car16_1", "car16_2", "car17", "car18",
        "group1_1", "group1_2", "group1_3", "group1_4",
        "group2_1", "group2_2", "group2_3",
        "group3_1", "group3_2", "group3_3", "group3_4",
        "group4_1", "group4_2", "group4_3",
        "person1", "person2_1", "person2_2", "person2_3",
        "person3", "person4_1", "person4_2", "person5_1", "person5_2",
        "person6", "person7_1", "person7_2", "person8_1", "person8_2",
        "person9", "person10", "person11", "person12_1", "person12_2",
        "person13", "person14_1", "person14_2", "person14_3",
        "person15", "person16", "person17", "person18",
        "person19_1", "person19_2", "person19_3", "person20",
        "truck1", "truck2", "truck3", "truck4_1", "truck4_2",
        "uav1_1", "uav1_2", "uav1_3",
        "wakeboard1", "wakeboard2", "wakeboard3", "wakeboard4", "wakeboard5",
        "wakeboard6", "wakeboard7", "wakeboard8", "wakeboard9", "wakeboard10",
    },
    "fast_motion": {
        "bike1", "bird1", "car1", "car2", "car3", "car7", "car9",
        "wakeboard1", "wakeboard2", "wakeboard3", "wakeboard4",
        "wakeboard5", "wakeboard6", "wakeboard7", "wakeboard8",
        "wakeboard9", "wakeboard10",
    },
    "full_occlusion": {
        "person1", "person2_1", "person2_2", "person2_3",
        "person4_1", "person4_2", "person5_1", "person5_2",
        "person7_1", "person7_2",
        "car6_1", "car6_2", "car6_3", "car6_4", "car6_5",
    },
    "partial_occlusion": {
        "bike2", "bike3", "building1", "building2", "building3",
        "car4", "car5", "car8_1", "car8_2", "car10", "car11",
        "group1_1", "group1_2", "group1_3", "group1_4",
        "group2_1", "group2_2", "group2_3",
        "person3", "person6", "person9", "person10", "person11",
        "person13", "person15", "person16", "person18",
    },
    "illumination_variation": {
        "bike1", "car1", "car5", "car9", "car14", "car15",
        "person1", "person3", "person15", "person16",
    },
    "camera_motion": {
        "bird1", "car1", "car2", "car3", "car7", "car9",
        "group1_1", "group1_2", "group1_3", "group1_4",
        "uav1_1", "uav1_2", "uav1_3",
        "wakeboard1", "wakeboard2", "wakeboard3", "wakeboard4",
    },
    "low_resolution": {
        "car1_s", "car2_s", "car3_s", "car4_s",
        "person2_1", "person2_2", "person2_3",
        "person4_1", "person4_2",
        "person5_1", "person5_2",
        "person7_1", "person7_2",
        "person8_1", "person8_2",
    },
    "out_of_view": {
        "car6_1", "car6_2", "car6_3", "car6_4", "car6_5",
        "car8_1", "car8_2",
        "group3_1", "group3_2", "group3_3", "group3_4",
        "person4_1", "person4_2",
        "person12_1", "person12_2",
        "person14_1", "person14_2", "person14_3",
        "person19_1", "person19_2", "person19_3",
    },
    "similar_objects": {
        "car11", "car12", "car13",
        "group1_1", "group1_2", "group1_3", "group1_4",
        "group2_1", "group2_2", "group2_3",
        "group3_1", "group3_2", "group3_3", "group3_4",
        "group4_1", "group4_2", "group4_3",
    },
    "viewpoint_change": {
        "bike1", "bike2", "bike3", "bird1",
        "car7", "car9", "car10", "car16_1", "car16_2",
        "uav1_1", "uav1_2", "uav1_3",
        "wakeboard5", "wakeboard6", "wakeboard7", "wakeboard8",
    },
    "aspect_ratio_change": {
        "bike1", "bird1",
        "car7", "car9", "car14", "car16_1", "car16_2",
        "group1_1", "group1_2", "group1_3", "group1_4",
        "person14_1", "person14_2", "person14_3",
    },
    "background_clutter": {
        "building1", "building2", "building3",
        "car11", "car12", "car13",
        "group3_1", "group3_2", "group3_3", "group3_4",
        "group4_1", "group4_2", "group4_3",
        "person17", "person18", "person19_1", "person19_2", "person19_3",
        "truck1", "truck2", "truck3", "truck4_1", "truck4_2",
    },
}

#: Sequences belonging to the UAV20L (long-sequence) subset.
_UAV20L_SEQUENCES: Set[str] = {
    "bike1", "bike3", "bird1",
    "car1", "car3", "car6_1", "car7", "car9",
    "group1_1", "group2_1", "group3_1",
    "person2_1", "person4_1", "person5_1", "person7_1",
    "person14_1", "person17", "person18",
    "uav1_1", "wakeboard1",
}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class UAV123Dataset(BaseDataset):
    """Dataset loader for UAV123 / UAV20L aerial object tracking benchmarks.

    Implements the :class:`~eovot.datasets.base.BaseDataset` interface and
    integrates seamlessly with :class:`~eovot.benchmark.engine.BenchmarkEngine`.

    Frames are loaded lazily (path references only) — no images are buffered in
    memory during dataset construction, keeping memory usage constant for both
    splits.

    Args:
        root: Path to the UAV123 dataset root (must contain ``anno/`` and
            ``data_seq/`` subdirectories).
        split: Dataset split: ``"UAV123"`` (full 123-sequence set, default)
            or ``"UAV20L"`` (20 long-sequence subset).
        max_sequences: Optional cap on the number of sequences returned.
            Useful for quick smoke-tests without downloading the full dataset.

    Raises:
        FileNotFoundError: If *root* or its required subdirectories are missing.
        ValueError: If *split* is not ``"UAV123"`` or ``"UAV20L"``.

    Example::

        dataset = UAV123Dataset("/data/UAV123")
        print(f"{len(dataset)} sequences in UAV123")
        seq = dataset[0]
        print(seq.name, len(seq), seq.init_bbox)

        # Filter to sequences with partial occlusion
        occ = dataset.filter_by_attribute("partial_occlusion")
        print(len(occ), "sequences with partial occlusion")

        # Inspect available attributes
        print(dataset.attributes)
    """

    SPLITS = ("UAV123", "UAV20L")

    def __init__(
        self,
        root: str,
        split: str = "UAV123",
        max_sequences: Optional[int] = None,
    ) -> None:
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS!r}, got {split!r}")

        self.root = Path(root)
        self.split = split
        self.max_sequences = max_sequences

        self._anno_dir = self.root / "anno" / split
        self._seq_dir = self.root / "data_seq" / "UAV123"

        if not self.root.is_dir():
            raise FileNotFoundError(f"UAV123 root directory not found: {root}")

        self._seq_names: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.list_sequences())

    def __getitem__(self, idx: int) -> Sequence:
        names = self.list_sequences()
        if not (0 <= idx < len(names)):
            raise IndexError(f"Sequence index {idx} out of range [0, {len(names)})")
        return self._load_sequence(names[idx])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"UAV123-{self.split}"

    @property
    def attributes(self) -> List[str]:
        """Sorted list of trackable attribute names available for filtering."""
        return sorted(SEQUENCE_ATTRIBUTES.keys())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_sequences(self) -> List[str]:
        """Return the ordered list of sequence names for this split.

        Discovers sequences by scanning the annotation directory (``anno/<split>/*.txt``).
        Results are cached after the first call.

        Returns:
            Sorted list of sequence name strings (without ``.txt`` extension).
        """
        if self._seq_names is not None:
            return self._seq_names

        if not self._anno_dir.is_dir():
            raise FileNotFoundError(
                f"Annotation directory not found: {self._anno_dir}\n"
                f"Expected layout: <root>/anno/{self.split}/<seq_name>.txt"
            )

        names = sorted(p.stem for p in self._anno_dir.glob("*.txt"))

        if self.split == "UAV20L":
            # Some UAV20L anno directories contain only long-subset files;
            # if the directory is otherwise unpopulated, fall back to filtering
            # from the known _UAV20L_SEQUENCES set.
            if not names:
                uav123_anno = self.root / "anno" / "UAV123"
                if uav123_anno.is_dir():
                    names = sorted(
                        p.stem for p in uav123_anno.glob("*.txt")
                        if p.stem in _UAV20L_SEQUENCES
                    )

        if self.max_sequences is not None:
            names = names[: self.max_sequences]

        self._seq_names = names
        return self._seq_names

    def filter_by_attribute(self, *attrs: str) -> "UAV123Dataset":
        """Return a new dataset view containing only sequences with all given attributes.

        Performs an *intersection* filter: a sequence is included only if it
        carries every attribute listed in *attrs*.  Attribute names must appear
        in :attr:`attributes`.

        Args:
            *attrs: One or more attribute names (e.g. ``"fast_motion"``,
                ``"partial_occlusion"``).

        Returns:
            A new :class:`UAV123Dataset` with a pre-filtered sequence list.

        Raises:
            ValueError: If *attrs* is empty or an unknown attribute name is given.

        Example::

            fast_occ = dataset.filter_by_attribute("fast_motion", "partial_occlusion")
        """
        if not attrs:
            raise ValueError("filter_by_attribute() requires at least one attribute name.")
        unknown = set(attrs) - set(SEQUENCE_ATTRIBUTES)
        if unknown:
            raise ValueError(
                f"Unknown attribute(s): {unknown!r}. "
                f"Available: {self.attributes}"
            )

        # Start from the sequences present in all requested attribute sets
        candidate_sets = [SEQUENCE_ATTRIBUTES[a] for a in attrs]
        keep = candidate_sets[0].intersection(*candidate_sets[1:])
        filtered = [s for s in self.list_sequences() if s in keep]

        view = UAV123Dataset.__new__(UAV123Dataset)
        view.root = self.root
        view.split = self.split
        view.max_sequences = None
        view._anno_dir = self._anno_dir
        view._seq_dir = self._seq_dir
        view._seq_names = filtered
        return view

    def sequence_attributes(self, seq_name: str) -> List[str]:
        """Return the list of attributes assigned to a single sequence.

        Args:
            seq_name: Sequence name (without extension), e.g. ``"bike1"``.

        Returns:
            Sorted list of attribute names that include *seq_name*.
        """
        return sorted(
            attr for attr, seqs in SEQUENCE_ATTRIBUTES.items() if seq_name in seqs
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_sequence(self, seq_name: str) -> Sequence:
        """Load a single sequence by name.

        Resolves the annotation file path (trying the configured split first,
        then ``UAV123`` as a fallback for UAV20L), reads ground-truth boxes,
        and discovers image paths under ``data_seq/UAV123/<seq_name>/``.

        Args:
            seq_name: Sequence folder / annotation stem name.

        Returns:
            :class:`~eovot.datasets.base.Sequence` with lazy frame paths.

        Raises:
            FileNotFoundError: If the annotation file or image directory is missing.
        """
        gt_path = self._anno_dir / f"{seq_name}.txt"
        if not gt_path.exists():
            # UAV20L annotations are sometimes stored only under UAV123/
            fallback = self.root / "anno" / "UAV123" / f"{seq_name}.txt"
            if fallback.exists():
                gt_path = fallback
            else:
                raise FileNotFoundError(
                    f"Annotation not found: {gt_path}\n"
                    f"Also tried: {fallback}"
                )

        gt = _load_groundtruth(gt_path)

        img_dir = self._seq_dir / seq_name
        if not img_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory not found: {img_dir}\n"
                f"Expected layout: <root>/data_seq/UAV123/{seq_name}/"
            )

        frame_paths = _discover_frames(img_dir)
        if not frame_paths:
            raise FileNotFoundError(f"No JPEG/PNG frames found in {img_dir}")

        n = min(len(frame_paths), len(gt))
        return Sequence(
            name=seq_name,
            frame_paths=frame_paths[:n],
            ground_truth=np.array(gt[:n], dtype=np.float64),
        )


# ---------------------------------------------------------------------------
# Standalone helpers (module-private)
# ---------------------------------------------------------------------------

def _load_groundtruth(gt_path: Path) -> List[BBox]:
    """Parse a UAV123 annotation file into a list of ``(x, y, w, h)`` tuples.

    Handles both comma-separated and whitespace-delimited formats and
    skips blank or incomplete lines.

    Args:
        gt_path: Path to the ``.txt`` annotation file.

    Returns:
        List of ``(x, y, w, h)`` tuples in pixel coordinates.
    """
    boxes: List[BBox] = []
    with open(gt_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # Normalise delimiters: tabs and spaces → commas
            parts = [p for p in line.replace("\t", ",").replace(" ", ",").split(",") if p]
            if len(parts) < 4:
                logger.debug("Skipping malformed annotation line: %r", line)
                continue
            try:
                x, y, w, h = (float(parts[i]) for i in range(4))
            except ValueError:
                logger.debug("Skipping non-numeric annotation line: %r", line)
                continue
            boxes.append((x, y, w, h))
    return boxes


def _discover_frames(img_dir: Path) -> List[str]:
    """Return sorted absolute frame paths from an image directory.

    Collects ``.jpg`` and ``.png`` files (case-insensitive) and sorts
    them lexicographically, which matches UAV123's ``img%05d.jpg`` naming.

    Args:
        img_dir: Directory containing image frames.

    Returns:
        Sorted list of absolute path strings.
    """
    paths: List[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(img_dir.glob(ext))
        paths.extend(img_dir.glob(ext.upper()))
    return [str(p) for p in sorted(set(paths))]

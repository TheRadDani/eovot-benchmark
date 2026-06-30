"""Unit tests for the centralized tracker registry.

These guard against the exact bug this module fixes: a tracker module
(e.g. MIL) existing but being unreachable from one of the config-driven
entry points because that entry point kept its own stale copy of the
tracker name -> class mapping.
"""

import pytest

from eovot.trackers.base import BaseTracker
from eovot.trackers.registry import (
    TRACKER_REGISTRY,
    available_trackers,
    build_tracker,
)


class TestAvailableTrackers:
    def test_includes_all_classical_trackers(self):
        names = available_trackers()
        for expected in ("MOSSE", "KCF", "CSRT", "MIL", "MedianFlow"):
            assert expected in names

    def test_sorted_and_deduplicated(self):
        names = available_trackers()
        assert names == sorted(set(names))


class TestBuildTracker:
    @pytest.mark.parametrize("name", ["MOSSE", "KCF", "CSRT", "MIL", "MedianFlow"])
    def test_builds_registered_classical_tracker(self, name):
        # CSRT/MedianFlow require an OpenCV build with the contrib/legacy
        # tracking module; skip gracefully on minimal opencv-python builds
        # that lack it (mirrors MedianFlowTracker's own ImportError).
        try:
            tracker = build_tracker(name)
        except ImportError as exc:
            pytest.skip(str(exc))
        assert isinstance(tracker, BaseTracker)
        assert tracker.name == name

    def test_forwards_constructor_params(self):
        tracker = build_tracker("MOSSE", learning_rate=0.2, sigma=3.0)
        assert tracker.name == "MOSSE"

    def test_unknown_tracker_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown tracker"):
            build_tracker("NotARealTracker")

    def test_registry_dict_matches_available_trackers(self):
        assert set(TRACKER_REGISTRY) == set(available_trackers())

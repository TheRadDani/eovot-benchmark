"""Central tracker registry for EOVOT.

Provides a single authoritative mapping from tracker name strings to
:class:`~eovot.trackers.base.BaseTracker` subclasses, used by:

* :class:`~eovot.experiment.runner.ExperimentRunner` (YAML-driven experiments)
* CLI scripts (``compare_trackers.py``, ``run_benchmark.py``)
* Programmatic multi-tracker sweeps

New trackers can be registered at runtime via :meth:`TrackerRegistry.register`,
making the system extensible without modifying core source files.

Example::

    from eovot.trackers.registry import TrackerRegistry

    # Instantiate a tracker by name
    tracker = TrackerRegistry.create("MOSSE")

    # Register a custom tracker
    from myproject.trackers import MyTracker
    TrackerRegistry.register("MyTracker", MyTracker)
    tracker = TrackerRegistry.create("MyTracker", learning_rate=0.05)

    # List all registered names
    print(TrackerRegistry.list_available())
    # ['CSRT', 'KCF', 'MedianFlow', 'MIL', 'MOSSE']
"""

from __future__ import annotations

from typing import Dict, List, Type

from .base import BaseTracker


class TrackerRegistry:
    """Central registry mapping tracker names to ``BaseTracker`` classes.

    All methods are class-level so no instance is needed — the registry
    is a module-level singleton backed by a class-level dict.

    Built-in trackers (MOSSE, KCF, CSRT, MedianFlow, MIL) are pre-registered
    at import time.  Deep-learning trackers that require optional dependencies
    (DaSiamRPN, NanoTrack) are registered lazily to avoid import errors on
    machines that lack those dependencies.
    """

    _registry: Dict[str, Type[BaseTracker]] = {}

    @classmethod
    def register(cls, name: str, tracker_cls: Type[BaseTracker]) -> None:
        """Register a tracker class under *name*.

        Args:
            name: Lookup key used in :meth:`create` and YAML configs.
            tracker_cls: A :class:`~eovot.trackers.base.BaseTracker` subclass.

        Raises:
            TypeError: If *tracker_cls* is not a subclass of ``BaseTracker``.
        """
        if not (isinstance(tracker_cls, type) and issubclass(tracker_cls, BaseTracker)):
            raise TypeError(
                f"tracker_cls must be a BaseTracker subclass, got {tracker_cls!r}"
            )
        cls._registry[name] = tracker_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseTracker:
        """Instantiate the tracker registered under *name*.

        Args:
            name: Tracker key (case-sensitive).
            **kwargs: Passed to the tracker constructor.

        Returns:
            A new :class:`~eovot.trackers.base.BaseTracker` instance.

        Raises:
            KeyError: If *name* is not registered.

        Example::

            mosse = TrackerRegistry.create("MOSSE")
            kcf   = TrackerRegistry.create("KCF", learning_rate=0.125)
        """
        if name not in cls._registry:
            available = cls.list_available()
            raise KeyError(
                f"Tracker '{name}' is not registered. "
                f"Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def list_available(cls) -> List[str]:
        """Return a sorted list of all registered tracker names."""
        return sorted(cls._registry)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Return ``True`` if *name* is registered."""
        return name in cls._registry

    @classmethod
    def get_class(cls, name: str) -> Type[BaseTracker]:
        """Return the tracker class for *name* without instantiating it.

        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in cls._registry:
            raise KeyError(
                f"Tracker '{name}' is not registered. "
                f"Available: {cls.list_available()}"
            )
        return cls._registry[name]


# ---------------------------------------------------------------------------
# Bootstrap: register built-in trackers at import time
# ---------------------------------------------------------------------------

def _bootstrap_registry() -> None:
    """Register all built-in trackers.  Called once at module import."""
    from .mosse import MOSSETracker
    from .kcf import KCFTracker
    from .csrt import CSRTTracker
    from .median_flow import MedianFlowTracker
    from .mil import MILTracker

    for name, cls in [
        ("MOSSE", MOSSETracker),
        ("KCF", KCFTracker),
        ("CSRT", CSRTTracker),
        ("MedianFlow", MedianFlowTracker),
        ("MIL", MILTracker),
    ]:
        TrackerRegistry.register(name, cls)


_bootstrap_registry()

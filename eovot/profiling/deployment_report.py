"""Deployment feasibility analysis across hardware profiles.

Provides two public helpers:

- :func:`evaluate_deployment` — check a **single tracker's** benchmark
  results against one or more hardware profiles and return a per-device
  feasibility DataFrame.

- :func:`compare_trackers_on_hardware` — compare **multiple trackers** across
  all hardware profiles and return a tracker × device deployment-score matrix.

Usage::

    from eovot.profiling.deployment_report import evaluate_deployment
    from eovot.profiling.hardware_profiles import get_profile

    report = evaluate_deployment(
        tracker_results={"fps": 22.5, "mean_latency_ms": 44.2, "peak_memory_mb": 185.0},
        profiles=[get_profile("raspberry_pi4"), get_profile("jetson_nano")],
    )
    print(report.to_string(index=False))
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .hardware_profiles import BUILTIN_PROFILES, HardwareProfile


def evaluate_deployment(
    tracker_results: Dict,
    profiles: Optional[List[HardwareProfile]] = None,
) -> pd.DataFrame:
    """Evaluate one tracker's benchmark results against hardware profiles.

    Args:
        tracker_results: Dict with at least the keys:
            - ``'fps'`` (float)
            - ``'mean_latency_ms'`` (float)
            - ``'peak_memory_mb'`` (float)
        profiles: Hardware profiles to evaluate against.  Defaults to all
            five built-in profiles.

    Returns:
        DataFrame with columns:
        ``['profile', 'suitable', 'deployment_score',
          'fps_ok', 'latency_ok', 'memory_ok',
          'target_fps', 'max_latency_ms', 'max_memory_mb']``
        sorted by ``deployment_score`` (descending).
    """
    if profiles is None:
        profiles = list(BUILTIN_PROFILES.values())

    fps = float(tracker_results.get("fps", 0.0))
    latency = float(tracker_results.get("mean_latency_ms", float("inf")))
    memory = float(tracker_results.get("peak_memory_mb", float("inf")))

    rows = []
    for profile in profiles:
        rows.append(
            {
                "profile": profile.name,
                "suitable": profile.is_tracker_suitable(fps, latency, memory),
                "deployment_score": profile.deployment_score(fps, latency, memory),
                "fps_ok": fps >= profile.target_fps,
                "latency_ok": latency <= profile.max_latency_ms,
                "memory_ok": memory <= profile.max_memory_mb,
                "target_fps": profile.target_fps,
                "max_latency_ms": profile.max_latency_ms,
                "max_memory_mb": profile.max_memory_mb,
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values("deployment_score", ascending=False).reset_index(drop=True)


def compare_trackers_on_hardware(
    benchmark_results: Dict[str, Dict],
    profiles: Optional[List[HardwareProfile]] = None,
) -> pd.DataFrame:
    """Compare multiple trackers via deployment scores across hardware profiles.

    Args:
        benchmark_results: Mapping ``{tracker_name: result_dict}`` where each
            result_dict has keys ``'fps'``, ``'mean_latency_ms'``, and
            ``'peak_memory_mb'``.
        profiles: Hardware profiles to evaluate against.  Defaults to all
            five built-in profiles.

    Returns:
        DataFrame indexed by tracker name, one column per hardware profile,
        values are deployment scores in [0, 1].  A cell value of 1.0 means the
        tracker fully satisfies all constraints for that device.
    """
    if profiles is None:
        profiles = list(BUILTIN_PROFILES.values())

    rows: Dict[str, Dict[str, float]] = {}
    for tracker_name, results in benchmark_results.items():
        fps = float(results.get("fps", 0.0))
        latency = float(results.get("mean_latency_ms", float("inf")))
        memory = float(results.get("peak_memory_mb", float("inf")))

        rows[tracker_name] = {
            profile.name: profile.deployment_score(fps, latency, memory)
            for profile in profiles
        }

    df = pd.DataFrame(rows).T
    df.index.name = "tracker"
    return df

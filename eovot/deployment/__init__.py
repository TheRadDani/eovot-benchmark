"""Deployment suitability scoring for edge-device tracker selection.

Combines accuracy, throughput, memory, and energy metrics from a
:class:`~eovot.benchmark.engine.BenchmarkResult` with device simulation
to rank trackers for deployment on a specific edge target.

Typical usage::

    from eovot.deployment import DeploymentConstraints, DeploymentSuitabilityScorer

    constraints = DeploymentConstraints(min_fps=15.0, target_fps=30.0)
    scorer = DeploymentSuitabilityScorer(device="rpi4", constraints=constraints)
    scores = scorer.rank([result_mosse, result_kcf, result_csrt])
    print(scorer.to_markdown_table(scores))
"""

from .scorer import DeploymentConstraints, DeploymentScore, DeploymentSuitabilityScorer

__all__ = [
    "DeploymentConstraints",
    "DeploymentScore",
    "DeploymentSuitabilityScorer",
]

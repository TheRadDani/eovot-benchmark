"""Tracker failure correlation analysis for EOVOT.

Quantifies how similarly two trackers fail across the same sequence set,
enabling:

- **Ensemble potential scoring** — uncorrelated trackers improve more when
  combined because their failure modes do not overlap.
- **Redundancy detection** — highly correlated trackers are interchangeable
  from an information-theoretic standpoint.
- **Tracker clustering** — group trackers by failure-mode similarity using
  single-linkage clustering on the Pearson-r matrix.

Pearson r of per-frame IoU vectors is the similarity measure:
r = 1 means both trackers fail the same frames; r = 0 means independent
failures; r = -1 means one succeeds exactly where the other fails (perfect
complementary pair — the theoretical ideal for an oracle ensemble).

Typical usage::

    from eovot.analysis.tracker_correlation import TrackerCorrelationAnalyzer

    analyzer = TrackerCorrelationAnalyzer()
    report = analyzer.analyze({"MOSSE": mosse_result, "KCF": kcf_result})
    print(report.to_markdown())
    t1, t2 = report.most_complementary_pair()
    print(f"Best ensemble: {t1} + {t2}  gain=+{report.ensemble_potential(t1, t2):.4f} IoU")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..benchmark.engine import BenchmarkResult


@dataclass
class CorrelationReport:
    """Result of a pairwise tracker IoU-correlation analysis.

    Attributes:
        tracker_names: Ordered list of tracker names matching matrix axes.
        correlation_matrix: ``(N, N)`` symmetric matrix of Pearson r values.
            Diagonal is always 1.0.
        ensemble_gain_matrix: ``(N, N)`` matrix of oracle-union IoU gains.
            ``ensemble_gain_matrix[i, j]`` is the expected mean-IoU improvement
            when an oracle selects the better prediction per frame between
            tracker i and tracker j.
        cluster_labels: Integer cluster ID per tracker.  Trackers in the same
            cluster have correlation >= the threshold used in :meth:`analyze`.
    """

    tracker_names: List[str]
    correlation_matrix: np.ndarray
    ensemble_gain_matrix: np.ndarray
    cluster_labels: List[int]

    def correlation(self, t1: str, t2: str) -> float:
        """Return Pearson r between two named trackers' concatenated IoU sequences."""
        i = self.tracker_names.index(t1)
        j = self.tracker_names.index(t2)
        return float(self.correlation_matrix[i, j])

    def ensemble_potential(self, t1: str, t2: str) -> float:
        """Estimated oracle-union mean-IoU gain when combining tracker t1 and t2."""
        i = self.tracker_names.index(t1)
        j = self.tracker_names.index(t2)
        return float(self.ensemble_gain_matrix[i, j])

    def most_complementary_pair(self) -> Tuple[str, str]:
        """Return the tracker pair with the lowest Pearson r (most complementary)."""
        n = len(self.tracker_names)
        best_i, best_j, best_r = 0, 1, float("inf")
        for i in range(n):
            for j in range(i + 1, n):
                r = float(self.correlation_matrix[i, j])
                if r < best_r:
                    best_r, best_i, best_j = r, i, j
        return self.tracker_names[best_i], self.tracker_names[best_j]

    def most_redundant_pair(self) -> Tuple[str, str]:
        """Return the tracker pair with the highest Pearson r (most redundant)."""
        n = len(self.tracker_names)
        best_i, best_j, best_r = 0, 1, float("-inf")
        for i in range(n):
            for j in range(i + 1, n):
                r = float(self.correlation_matrix[i, j])
                if r > best_r:
                    best_r, best_i, best_j = r, i, j
        return self.tracker_names[best_i], self.tracker_names[best_j]

    def to_markdown(self) -> str:
        """Format the correlation and ensemble-gain matrices as Markdown tables."""
        n = len(self.tracker_names)
        names = self.tracker_names
        lines: List[str] = []

        # ---- Correlation matrix ----
        lines.append("## Tracker Failure Correlation Matrix (Pearson r)\n")
        lines.append("| Tracker | " + " | ".join(names) + " |")
        lines.append("|" + "---------|" * (n + 1))
        for i, name in enumerate(names):
            row = " | ".join(f"{self.correlation_matrix[i, j]:.3f}" for j in range(n))
            lines.append(f"| {name} | {row} |")

        lines.append("")

        # ---- Ensemble-gain matrix ----
        lines.append("## Oracle-Union Ensemble IoU Gain Matrix\n")
        lines.append(
            "*Estimated mean-IoU gain when an oracle selects the "
            "better tracker per frame.*\n"
        )
        lines.append("| Tracker | " + " | ".join(names) + " |")
        lines.append("|" + "---------|" * (n + 1))
        for i, name in enumerate(names):
            row = " | ".join(
                f"{self.ensemble_gain_matrix[i, j]:+.4f}" for j in range(n)
            )
            lines.append(f"| {name} | {row} |")

        lines.append("")

        # ---- Summary ----
        if n >= 2:
            comp = self.most_complementary_pair()
            red = self.most_redundant_pair()
            gain = self.ensemble_potential(comp[0], comp[1])
            lines.append(
                f"**Most complementary pair**: `{comp[0]}` + `{comp[1]}`  "
                f"r={self.correlation(comp[0], comp[1]):.3f}  "
                f"oracle gain=**+{gain:.4f}** IoU"
            )
            lines.append(
                f"**Most redundant pair**: `{red[0]}` + `{red[1]}`  "
                f"r={self.correlation(red[0], red[1]):.3f}"
            )

        # ---- Cluster summary ----
        if n >= 3 and self.cluster_labels:
            lines.append("\n## Tracker Clusters (single-linkage on Pearson r)\n")
            clusters: Dict[int, List[str]] = {}
            for name, label in zip(names, self.cluster_labels):
                clusters.setdefault(label, []).append(name)
            for cid, members in sorted(clusters.items()):
                lines.append(f"- **Cluster {cid}**: {', '.join(members)}")

        return "\n".join(lines)


class TrackerCorrelationAnalyzer:
    """Compute pairwise tracker failure correlation from benchmark results.

    Args:
        cluster_threshold: Pearson r at or above which two trackers are placed
            in the same cluster.  Default: ``0.85``.
    """

    def __init__(self, cluster_threshold: float = 0.85) -> None:
        if not 0.0 <= cluster_threshold <= 1.0:
            raise ValueError(
                f"cluster_threshold must be in [0, 1], got {cluster_threshold}"
            )
        self.cluster_threshold = cluster_threshold

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _concat_ious(result: BenchmarkResult) -> np.ndarray:
        """Concatenate per-frame IoU arrays from all sequences in a result."""
        parts = [r.ious for r in result.sequence_results if len(r.ious) > 0]
        if not parts:
            return np.array([], dtype=np.float64)
        return np.concatenate(parts).astype(np.float64)

    @staticmethod
    def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
        """Pearson correlation coefficient, safe for constant arrays."""
        n = min(len(a), len(b))
        if n < 2:
            return 1.0
        a, b = a[:n], b[:n]
        std_a, std_b = float(a.std()), float(b.std())
        if std_a == 0.0 or std_b == 0.0:
            return 1.0 if std_a == std_b else 0.0
        return float(np.corrcoef(a, b)[0, 1])

    @staticmethod
    def _oracle_union_gain(a: np.ndarray, b: np.ndarray) -> float:
        """Expected IoU gain from oracle-selecting the better tracker per frame.

        ``gain = E[max(IoU_a, IoU_b)] - 0.5 * E[IoU_a + IoU_b]``

        A large positive value means the trackers are complementary: one
        recovers where the other fails.  A value near zero means they fail
        (and succeed) on the same frames.
        """
        n = min(len(a), len(b))
        if n == 0:
            return 0.0
        a, b = a[:n], b[:n]
        oracle = np.maximum(a, b)
        baseline = 0.5 * (a + b)
        return float((oracle - baseline).mean())

    def _cluster(self, corr: np.ndarray) -> List[int]:
        """Single-linkage clustering via union-find on Pearson r >= threshold."""
        n = len(corr)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            parent[find(x)] = find(y)

        for i in range(n):
            for j in range(i + 1, n):
                if corr[i, j] >= self.cluster_threshold:
                    union(i, j)

        root_map: Dict[int, int] = {}
        labels: List[int] = []
        for i in range(n):
            root = find(i)
            if root not in root_map:
                root_map[root] = len(root_map)
            labels.append(root_map[root])
        return labels

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        results: Dict[str, BenchmarkResult],
    ) -> CorrelationReport:
        """Compute pairwise Pearson correlation and oracle-union ensemble gains.

        All results must come from the **same dataset** evaluated in the same
        sequence order so that per-frame IoU arrays are aligned.

        Args:
            results: Mapping ``{tracker_name: BenchmarkResult}``.

        Returns:
            :class:`CorrelationReport` with the full N×N correlation matrix,
            N×N ensemble-gain matrix, and per-tracker cluster labels.

        Raises:
            ValueError: If ``results`` is empty or contains a single tracker.
        """
        if len(results) < 2:
            raise ValueError(
                f"analyze() requires at least 2 trackers, got {len(results)}."
            )

        names = list(results.keys())
        n = len(names)
        ious_map = {name: self._concat_ious(results[name]) for name in names}

        corr = np.eye(n, dtype=np.float64)
        gain = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                a = ious_map[names[i]]
                b = ious_map[names[j]]
                corr[i, j] = self._pearson_r(a, b)
                gain[i, j] = self._oracle_union_gain(a, b)

        labels = self._cluster(corr) if n >= 3 else list(range(n))

        return CorrelationReport(
            tracker_names=names,
            correlation_matrix=corr,
            ensemble_gain_matrix=gain,
            cluster_labels=labels,
        )

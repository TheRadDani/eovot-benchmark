"""Sequence-discriminativeness-weighted leaderboard for tracker comparison.

A standard benchmark leaderboard averages each tracker's per-sequence IoU
over all sequences with equal weight.  This is simple but can be dominated
by easy sequences where every tracker scores near 1.0 — those sequences
contribute little information about which tracker is better.

The :class:`SequenceWeightedLeaderboard` assigns each sequence a weight
based on how *discriminative* it is: sequences where trackers spread far
apart in performance (high inter-tracker IoU variance) receive more weight,
and sequences where everyone performs similarly (trivial wins or trivial
failures) receive less.  The result is a ranking that emphasises the
sequences that actually differentiate trackers.

Two weighting schemes are provided:

* **variance** — weight proportional to the cross-tracker variance of
  per-sequence mean IoU.  The intuition: if MOSSE gets 0.9 and KCF gets
  0.2 on a sequence, that sequence is highly discriminative.  If both
  get 0.85, it is not.

* **difficulty** — weight proportional to the cross-tracker *mean* IoU
  subtracted from 1.0 (i.e. 1 - mean_iou).  This up-weights sequences
  that are hard on average, following the VOT protocol's emphasis on
  challenging scenarios.  Degenerate all-easy or all-hard sequences get
  low weight: a sequence where every tracker fails (mean IoU ≈ 0) is
  less informative than one where results spread between 0.2 and 0.8.

Both schemes are applied to the same set of :class:`~eovot.benchmark.engine.BenchmarkResult`
objects without re-running any tracker.

Typical usage::

    from eovot.analysis.weighted_leaderboard import SequenceWeightedLeaderboard

    results = [mosse_result, kcf_result, csrt_result]
    lb = SequenceWeightedLeaderboard(results)

    print(lb.to_markdown_table())           # variance-weighted
    print(lb.to_markdown_table("difficulty"))  # difficulty-weighted
    print(lb.rank_delta_table())           # weighted vs unweighted rank shift

The leaderboard only requires that the BenchmarkResults all cover the same
set of sequence names; if they differ, only the intersection is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence as TypingSequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult

WeightingScheme = Literal["variance", "difficulty"]
_VALID_SCHEMES = ("variance", "difficulty")


@dataclass
class LeaderboardEntry:
    """A single tracker's entry in the weighted leaderboard.

    Attributes:
        tracker_name:         Name of the tracker.
        unweighted_mean_iou:  Simple arithmetic mean IoU over all shared sequences.
        weighted_mean_iou:    Sequence-weighted mean IoU.
        unweighted_rank:      Rank in the unweighted leaderboard (1 = best).
        weighted_rank:        Rank in the weighted leaderboard (1 = best).
        rank_delta:           ``unweighted_rank - weighted_rank``; positive means
                              the tracker improves in rank when discrimination
                              weighting is applied (it performs better on harder
                              or more discriminative sequences), negative means
                              it drops.
        num_sequences:        Number of shared sequences used for scoring.
    """

    tracker_name: str
    unweighted_mean_iou: float
    weighted_mean_iou: float
    unweighted_rank: int
    weighted_rank: int
    rank_delta: int
    num_sequences: int

    def __str__(self) -> str:
        delta_str = (
            f"+{self.rank_delta}" if self.rank_delta > 0
            else str(self.rank_delta)
        )
        return (
            f"LeaderboardEntry({self.tracker_name!r}  "
            f"rank_w={self.weighted_rank} ({delta_str})  "
            f"IoU_w={self.weighted_mean_iou:.4f}  "
            f"IoU_uw={self.unweighted_mean_iou:.4f})"
        )


@dataclass
class WeightedLeaderboardResult:
    """Full output of :class:`SequenceWeightedLeaderboard`.

    Attributes:
        scheme:           Weighting scheme used (``"variance"`` or
            ``"difficulty"``).
        entries:          Leaderboard rows, sorted by *weighted* rank ascending.
        sequence_weights: Dict mapping each shared sequence name to its
            normalised weight in ``[0, 1]`` (sum = 1.0).
        shared_sequences: Ordered list of sequence names used for scoring.
    """

    scheme: WeightingScheme
    entries: List[LeaderboardEntry] = field(default_factory=list)
    sequence_weights: Dict[str, float] = field(default_factory=dict)
    shared_sequences: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def to_markdown_table(self) -> str:
        """Render a Markdown comparison table.

        Returns:
            Multi-line Markdown table with both unweighted and weighted
            rankings side by side.
        """
        header = (
            "| W-Rank | Tracker | Weighted IoU | UW-Rank | Unweighted IoU | Rank Δ |"
        )
        sep = "|--------|---------|-------------|---------|----------------|--------|"
        rows = []
        for e in self.entries:
            delta_str = (
                f"+{e.rank_delta}" if e.rank_delta > 0
                else ("—" if e.rank_delta == 0 else str(e.rank_delta))
            )
            rows.append(
                f"| {e.weighted_rank} "
                f"| {e.tracker_name} "
                f"| {e.weighted_mean_iou:.4f} "
                f"| {e.unweighted_rank} "
                f"| {e.unweighted_mean_iou:.4f} "
                f"| {delta_str} |"
            )
        title = f"\n### Leaderboard (scheme: {self.scheme})\n\n"
        note = (
            f"\n_Sequences: {len(self.shared_sequences)} "
            f"| Weighting: {self.scheme} "
            f"| Rank Δ = unweighted − weighted (positive = improved with weighting)_"
        )
        return title + header + "\n" + sep + "\n" + "\n".join(rows) + note

    def __str__(self) -> str:
        lines = [f"WeightedLeaderboard(scheme={self.scheme!r}, {len(self.entries)} trackers)"]
        for e in self.entries:
            lines.append(f"  {e}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SequenceWeightedLeaderboard:
    """Build a discriminativeness-weighted leaderboard from BenchmarkResult objects.

    Each :class:`~eovot.benchmark.engine.BenchmarkResult` provides per-sequence
    IoU arrays.  The leaderboard:

    1. Finds the *intersection* of sequence names across all results.
    2. Computes a per-sequence weight under the chosen scheme.
    3. Computes each tracker's weighted-mean IoU as the dot product of
       its per-sequence mean IoUs with the normalised weights.
    4. Ranks trackers by weighted-mean IoU (descending) and by standard
       unweighted mean IoU.

    Args:
        results: Two or more :class:`~eovot.benchmark.engine.BenchmarkResult`
            objects.  Must contain at least 1 result.
        eps: Small constant added to weights before normalisation to avoid
            a zero-weight degenerate case where every sequence is equally
            trivial.  Default: ``1e-6``.

    Raises:
        ValueError: If ``results`` is empty, or if no sequences are shared
            across all results.

    Example::

        from eovot.analysis.weighted_leaderboard import SequenceWeightedLeaderboard

        lb = SequenceWeightedLeaderboard([mosse_result, kcf_result, csrt_result])
        result = lb.rank(scheme="variance")
        print(result.to_markdown_table())
        delta_md = lb.rank_delta_table()
    """

    def __init__(
        self,
        results: TypingSequence["BenchmarkResult"],
        eps: float = 1e-6,
    ) -> None:
        if len(results) == 0:
            raise ValueError("SequenceWeightedLeaderboard requires at least 1 BenchmarkResult.")
        self._results = list(results)
        self._eps = eps
        self._shared_names: List[str] = self._find_shared_sequences()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def shared_sequence_names(self) -> List[str]:
        """Sequence names present in every result, sorted alphabetically."""
        return list(self._shared_names)

    def rank(self, scheme: WeightingScheme = "variance") -> WeightedLeaderboardResult:
        """Compute a weighted leaderboard under *scheme*.

        Args:
            scheme: ``"variance"`` (discriminativeness-weighted) or
                ``"difficulty"`` (hardness-weighted).

        Returns:
            :class:`WeightedLeaderboardResult` with ranked entries and
            per-sequence weights.

        Raises:
            ValueError: If *scheme* is not a valid option.
        """
        if scheme not in _VALID_SCHEMES:
            raise ValueError(
                f"Unknown weighting scheme '{scheme}'. "
                f"Valid options: {_VALID_SCHEMES}"
            )

        iou_matrix = self._build_iou_matrix()
        weights = self._compute_weights(iou_matrix, scheme)
        norm_weights = weights / weights.sum()

        tracker_names = [r.tracker_name for r in self._results]
        unweighted_ious = iou_matrix.mean(axis=1)      # (n_trackers,)
        weighted_ious = iou_matrix @ norm_weights       # (n_trackers,)

        unweighted_ranks = _rank_descending(unweighted_ious)
        weighted_ranks = _rank_descending(weighted_ious)

        entries = []
        for i, name in enumerate(tracker_names):
            entries.append(LeaderboardEntry(
                tracker_name=name,
                unweighted_mean_iou=float(unweighted_ious[i]),
                weighted_mean_iou=float(weighted_ious[i]),
                unweighted_rank=int(unweighted_ranks[i]),
                weighted_rank=int(weighted_ranks[i]),
                rank_delta=int(unweighted_ranks[i]) - int(weighted_ranks[i]),
                num_sequences=len(self._shared_names),
            ))

        entries.sort(key=lambda e: e.weighted_rank)

        weight_dict = {
            name: float(norm_weights[j])
            for j, name in enumerate(self._shared_names)
        }

        return WeightedLeaderboardResult(
            scheme=scheme,
            entries=entries,
            sequence_weights=weight_dict,
            shared_sequences=list(self._shared_names),
        )

    def rank_delta_table(self) -> str:
        """Compare variance and difficulty schemes in a single Markdown table.

        Runs both schemes and shows the unweighted IoU, both weighted IoUs,
        and both rank deltas for every tracker — useful for understanding how
        much the choice of weighting scheme affects the final ranking.

        Returns:
            Multi-line Markdown table string.
        """
        var_result = self.rank("variance")
        diff_result = self.rank("difficulty")

        # Build lookup dicts keyed by tracker name
        var_map = {e.tracker_name: e for e in var_result.entries}
        diff_map = {e.tracker_name: e for e in diff_result.entries}
        tracker_names = sorted(var_map.keys())

        header = (
            "| Tracker | UW-IoU | UW-Rank | "
            "Var-IoU | Var-Rank | Var-Δ | "
            "Diff-IoU | Diff-Rank | Diff-Δ |"
        )
        sep = (
            "|---------|--------|---------|"
            "--------|----------|-------|"
            "----------|-----------|--------|"
        )
        rows = []
        for name in tracker_names:
            v = var_map[name]
            d = diff_map[name]

            def _delta_str(delta: int) -> str:
                return f"+{delta}" if delta > 0 else ("—" if delta == 0 else str(delta))

            rows.append(
                f"| {name} "
                f"| {v.unweighted_mean_iou:.4f} "
                f"| {v.unweighted_rank} "
                f"| {v.weighted_mean_iou:.4f} "
                f"| {v.weighted_rank} "
                f"| {_delta_str(v.rank_delta)} "
                f"| {d.weighted_mean_iou:.4f} "
                f"| {d.weighted_rank} "
                f"| {_delta_str(d.rank_delta)} |"
            )

        title = "\n### Weighting Scheme Comparison\n\n"
        note = (
            "\n_Var = variance-weighted, Diff = difficulty-weighted, "
            "Δ = unweighted rank − weighted rank_"
        )
        return title + header + "\n" + sep + "\n" + "\n".join(rows) + note

    def sequence_weight_table(self, scheme: WeightingScheme = "variance") -> str:
        """Render per-sequence weights as a Markdown table.

        Args:
            scheme: Weighting scheme to display.  Default: ``"variance"``.

        Returns:
            Markdown table with sequence names and their normalised weights,
            sorted by weight descending (most discriminative first).
        """
        result = self.rank(scheme)
        items = sorted(
            result.sequence_weights.items(), key=lambda kv: kv[1], reverse=True
        )
        header = "| Sequence | Weight | Rank |"
        sep = "|----------|--------|------|"
        rows = []
        for rank, (seq_name, w) in enumerate(items, start=1):
            rows.append(f"| {seq_name} | {w:.6f} | {rank} |")
        return (
            f"\n### Sequence Weights (scheme: {scheme})\n\n"
            + header + "\n" + sep + "\n" + "\n".join(rows)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_shared_sequences(self) -> List[str]:
        """Compute the sorted intersection of sequence names across all results."""
        sets = [
            {sr.sequence_name for sr in r.sequence_results}
            for r in self._results
        ]
        shared = sets[0]
        for s in sets[1:]:
            shared = shared & s
        if not shared:
            raise ValueError(
                "No sequences are shared across all BenchmarkResults. "
                "Ensure every result was computed on the same dataset."
            )
        return sorted(shared)

    def _build_iou_matrix(self) -> np.ndarray:
        """Build a (n_trackers, n_sequences) matrix of per-sequence mean IoU.

        Rows correspond to trackers (in the order of ``self._results``).
        Columns correspond to shared sequences (alphabetically sorted).

        Returns:
            Float64 array of shape ``(n_trackers, n_sequences)``.
        """
        n_trackers = len(self._results)
        n_seqs = len(self._shared_names)
        matrix = np.zeros((n_trackers, n_seqs), dtype=np.float64)

        for i, result in enumerate(self._results):
            seq_iou = {
                sr.sequence_name: sr.mean_iou
                for sr in result.sequence_results
            }
            for j, seq_name in enumerate(self._shared_names):
                matrix[i, j] = seq_iou.get(seq_name, 0.0)

        return matrix

    def _compute_weights(
        self, iou_matrix: np.ndarray, scheme: WeightingScheme
    ) -> np.ndarray:
        """Compute raw (unnormalised) per-sequence weights.

        Args:
            iou_matrix: Float array of shape ``(n_trackers, n_sequences)``.
            scheme:     ``"variance"`` or ``"difficulty"``.

        Returns:
            1-D float array of length ``n_sequences``.
        """
        if scheme == "variance":
            # Weight = cross-tracker variance of per-sequence mean IoU.
            # High variance → trackers disagree → discriminative.
            weights = iou_matrix.var(axis=0) + self._eps
        else:  # difficulty
            # Weight = (1 - mean_iou) * mean_iou
            # This is maximised at mean_iou=0.5 and zero at 0 or 1,
            # penalising both degenerate all-pass and all-fail sequences.
            col_means = iou_matrix.mean(axis=0)
            weights = col_means * (1.0 - col_means) + self._eps

        return weights


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------

def _rank_descending(scores: np.ndarray) -> np.ndarray:
    """Rank an array of scores descending (rank 1 = highest score).

    Ties are broken by the natural sort order (first occurrence = lower rank
    number).

    Args:
        scores: 1-D float array of length N.

    Returns:
        Integer rank array of length N.
    """
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks

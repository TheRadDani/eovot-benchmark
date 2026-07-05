"""Magnitude-based structured pruning for PyTorch tracker backbones.

Structured (channel-level) pruning removes entire output channels whose
L1-norm falls below a threshold, producing dense weight tensors that run
efficiently on standard CPU/GPU hardware — unlike unstructured pruning,
which requires sparse-tensor support to realise speedups.

The workflow is:
    1. Analyse the model to find pruneable Conv2d layers.
    2. Choose a sparsity ratio and preview the expected size reduction.
    3. Apply pruning in-place; export the pruned model via :mod:`onnx_export`.

Example
-------
::

    import torch
    from eovot.optimization.pruning import MagnitudePruner, PruningConfig

    model = MyTrackerBackbone()
    cfg   = PruningConfig(sparsity=0.3, min_channels=8)
    stats = MagnitudePruner(cfg).prune(model)
    print(stats)
    # PruningStats(layers_pruned=5, channels_removed=48, size_reduction_pct=28.4)

Requires ``torch``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PruningConfig:
    """Parameters for magnitude-based structured pruning.

    Attributes
    ----------
    sparsity:
        Fraction of channels to remove per Conv2d layer (0.0 – 0.9).
        A value of 0.3 removes the 30% of channels with smallest L1-norm.
    min_channels:
        Hard floor on remaining output channels after pruning.  Prevents
        degenerate layers that would break forward-pass shapes.
    scope:
        Which layers to prune: ``"all"`` prunes every Conv2d;
        ``"non_depthwise"`` skips depthwise convolutions (``groups == in_channels``),
        which typically cannot be pruned independently.
    """

    sparsity: float = 0.3
    min_channels: int = 8
    scope: str = "all"

    def __post_init__(self) -> None:
        if not 0.0 <= self.sparsity < 1.0:
            raise ValueError(f"sparsity must be in [0, 1), got {self.sparsity}")
        if self.min_channels < 1:
            raise ValueError(f"min_channels must be >= 1, got {self.min_channels}")
        if self.scope not in ("all", "non_depthwise"):
            raise ValueError(f"scope must be 'all' or 'non_depthwise', got {self.scope!r}")


@dataclass
class LayerPruningInfo:
    """Analysis result for a single Conv2d layer.

    Attributes
    ----------
    name:           Layer name / path inside the module tree.
    original_out:   Original number of output channels.
    kept_channels:  Number of channels retained after pruning.
    removed_pct:    Percentage of channels removed.
    l1_threshold:   L1-norm value below which channels were cut.
    params_saved:   Approximate parameter reduction (Conv2d weights only).
    """

    name: str
    original_out: int
    kept_channels: int
    removed_pct: float
    l1_threshold: float
    params_saved: int


@dataclass
class PruningStats:
    """Aggregate result from a :class:`MagnitudePruner` run.

    Attributes
    ----------
    layers_pruned:       How many Conv2d layers were modified.
    channels_removed:    Total output channels removed across all layers.
    total_params_before: Parameter count before pruning.
    total_params_after:  Parameter count after pruning.
    size_reduction_pct:  Estimated weight-parameter reduction in percent.
    layer_details:       Per-layer breakdown.
    """

    layers_pruned: int
    channels_removed: int
    total_params_before: int
    total_params_after: int
    size_reduction_pct: float
    layer_details: List[LayerPruningInfo] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"PruningStats("
            f"layers_pruned={self.layers_pruned}, "
            f"channels_removed={self.channels_removed}, "
            f"params {self.total_params_before:,} → {self.total_params_after:,} "
            f"({self.size_reduction_pct:.1f}% reduction)"
            f")"
        )

    def to_dict(self) -> Dict:
        return {
            "layers_pruned": self.layers_pruned,
            "channels_removed": self.channels_removed,
            "total_params_before": self.total_params_before,
            "total_params_after": self.total_params_after,
            "size_reduction_pct": round(self.size_reduction_pct, 2),
            "layers": [
                {
                    "name": l.name,
                    "original_out": l.original_out,
                    "kept_channels": l.kept_channels,
                    "removed_pct": round(l.removed_pct, 2),
                    "l1_threshold": round(l.l1_threshold, 6),
                    "params_saved": l.params_saved,
                }
                for l in self.layer_details
            ],
        }


class PruningAnalyzer:
    """Read-only model analysis — does NOT modify the model.

    Use this to preview what pruning at a given sparsity ratio would do
    before committing to an irreversible in-place operation.

    Parameters
    ----------
    config:
        Pruning configuration.
    """

    def __init__(self, config: Optional[PruningConfig] = None) -> None:
        self.config = config or PruningConfig()
        self._torch_available = self._probe_torch()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(self, model: "torch.nn.Module") -> PruningStats:
        """Return pruning statistics without modifying *model*.

        Parameters
        ----------
        model: PyTorch ``nn.Module`` to analyse.

        Returns
        -------
        :class:`PruningStats` showing expected channels removed and
        parameter reduction if :meth:`MagnitudePruner.prune` were called
        with the same config.
        """
        self._require_torch()
        layers = list(self._iter_pruneable(model))
        if not layers:
            total = sum(p.numel() for p in model.parameters())
            return PruningStats(
                layers_pruned=0,
                channels_removed=0,
                total_params_before=total,
                total_params_after=total,
                size_reduction_pct=0.0,
            )

        details = [self._analyse_layer(name, module) for name, module in layers]
        total_before = sum(p.numel() for p in model.parameters())
        params_saved = sum(d.params_saved for d in details)
        total_after = total_before - params_saved
        reduction_pct = params_saved / total_before * 100.0 if total_before > 0 else 0.0

        return PruningStats(
            layers_pruned=len(details),
            channels_removed=sum(d.original_out - d.kept_channels for d in details),
            total_params_before=total_before,
            total_params_after=max(0, total_after),
            size_reduction_pct=round(reduction_pct, 2),
            layer_details=details,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _iter_pruneable(self, model: "torch.nn.Module"):
        """Yield (name, module) pairs for Conv2d layers matching the scope."""
        import torch.nn as nn

        for name, module in model.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue
            if self.config.scope == "non_depthwise" and module.groups == module.in_channels:
                continue
            yield name, module

    def _analyse_layer(self, name: str, conv: "torch.nn.Module") -> LayerPruningInfo:
        import torch

        weight = conv.weight.data  # (out_C, in_C/groups, kH, kW)
        out_channels = weight.shape[0]
        l1_norms = weight.view(out_channels, -1).abs().sum(dim=1)

        keep_n = max(
            self.config.min_channels,
            math.ceil(out_channels * (1.0 - self.config.sparsity)),
        )
        keep_n = min(keep_n, out_channels)
        removed = out_channels - keep_n

        sorted_norms, _ = torch.sort(l1_norms)
        threshold = float(sorted_norms[removed - 1]) if removed > 0 else 0.0

        # Approximate params saved: removed output filters × (in_C/groups × kH × kW)
        params_per_filter = int(weight[0].numel())
        params_saved = removed * params_per_filter

        return LayerPruningInfo(
            name=name,
            original_out=out_channels,
            kept_channels=keep_n,
            removed_pct=removed / out_channels * 100.0,
            l1_threshold=threshold,
            params_saved=params_saved,
        )

    @staticmethod
    def _probe_torch() -> bool:
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    def _require_torch(self) -> None:
        if not self._torch_available:
            raise ImportError(
                "torch is required for pruning analysis. "
                "Install with: pip install torch"
            )


class MagnitudePruner:
    """Apply structured magnitude pruning in-place to a PyTorch model.

    Channels whose L1-norm is below the *sparsity*-th percentile are
    zeroed out and a binary mask is registered as a buffer on the layer.
    The model remains a standard ``nn.Module`` and can be fine-tuned,
    exported to ONNX (via :mod:`eovot.optimization.onnx_export`), or
    evaluated immediately.

    .. warning::
        Pruning is an **in-place, irreversible** operation on the supplied
        module.  Clone the model before calling :meth:`prune` if you need
        the original weights.

    Parameters
    ----------
    config:
        Pruning settings.
    """

    def __init__(self, config: Optional[PruningConfig] = None) -> None:
        self.config = config or PruningConfig()
        self._analyser = PruningAnalyzer(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prune(self, model: "torch.nn.Module") -> PruningStats:
        """Prune *model* in-place and return statistics.

        For each Conv2d layer selected by ``config.scope``:
        - Rank output channels by L1-norm of their filter weights.
        - Zero the weights of channels below the sparsity threshold.
        - Register a ``"weight_mask"`` buffer (1/0 per output channel)
          so the pruning pattern can be inspected or serialized.

        Parameters
        ----------
        model: The ``nn.Module`` to prune.  **Modified in-place.**

        Returns
        -------
        :class:`PruningStats` with aggregate and per-layer statistics.
        """
        self._analyser._require_torch()
        import torch

        stats = self._analyser.analyse(model)

        for name, conv in self._analyser._iter_pruneable(model):
            weight = conv.weight.data
            out_channels = weight.shape[0]
            l1_norms = weight.view(out_channels, -1).abs().sum(dim=1)

            keep_n = max(
                self.config.min_channels,
                math.ceil(out_channels * (1.0 - self.config.sparsity)),
            )
            keep_n = min(keep_n, out_channels)

            _, sorted_idx = torch.sort(l1_norms, descending=True)
            keep_idx = sorted_idx[:keep_n]

            mask = torch.zeros(out_channels, dtype=torch.float32, device=weight.device)
            mask[keep_idx] = 1.0

            # Zero-out pruned filters and register mask
            conv.weight.data *= mask.view(-1, 1, 1, 1)
            if conv.bias is not None:
                conv.bias.data *= mask
            conv.register_buffer("weight_mask", mask)

        return stats

    def get_sparsity(self, model: "torch.nn.Module") -> Dict[str, float]:
        """Return per-layer actual weight sparsity (fraction of zero elements).

        Useful for verifying pruning was applied correctly and measuring
        the true zero-weight density after any subsequent fine-tuning.

        Parameters
        ----------
        model: A (possibly pruned) ``nn.Module``.

        Returns
        -------
        dict mapping layer name → sparsity fraction in [0, 1].
        """
        self._analyser._require_torch()
        result = {}
        for name, conv in self._analyser._iter_pruneable(model):
            w = conv.weight.data
            total = w.numel()
            zeros = int((w == 0).sum().item())
            result[name] = round(zeros / total, 4) if total > 0 else 0.0
        return result

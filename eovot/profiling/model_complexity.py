"""Static model complexity profiler for edge-deployment readiness.

Complements EOVOT's runtime profiling (latency, memory, energy) with
*static* analysis of deep-learning tracker models:

- **Parameter count** — total and trainable parameters.
- **Estimated FLOPs** — floating-point operations per forward pass,
  estimated via lightweight forward hooks on ``nn.Conv2d``, ``nn.Linear``,
  ``nn.BatchNorm2d``, and ``nn.ConvTranspose2d`` layers.
- **Model file size** — on-disk footprint of a saved checkpoint or ONNX file.
- **Edge readiness tier** — categorical label (Ultralight / Light / Medium /
  Heavy) based on parameter count and FLOPs, calibrated against typical
  constraints of Raspberry Pi 4, Jetson Nano, and mobile phones.

These metrics are orthogonal to runtime measurements: a model can be
parameter-light but slow (due to memory bandwidth), or FLOP-heavy but fast
on hardware with dedicated accelerators.  Reporting both dimensions enables
the accuracy × efficiency trade-off analysis that is central to EOVOT.

FLOPs estimation
~~~~~~~~~~~~~~~~
FLOPs are estimated via PyTorch forward hooks that count multiply-add
operations (MACs × 2 = FLOPs) for the most common layer types:

* **Conv2d**: ``2 × Cout × Cin/groups × kH × kW × H_out × W_out``
* **ConvTranspose2d**: same formula adjusted for output spatial size.
* **Linear**: ``2 × in_features × out_features``
* **BatchNorm2d**: ``2 × num_features × H × W`` (scale + shift)

Other layer types (activations, pooling, layer-norm, attention) are not
counted.  The result is therefore a *lower-bound* estimate; actual FLOPs
on a full transformer tracker will be higher.

Edge readiness tiers (parameter-count based)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
=============  ================  ===========================
Tier           Parameters         Representative hardware
=============  ================  ===========================
Ultralight     < 500 K            MCUs, RP2040
Light          500 K – 5 M        Raspberry Pi Zero, phones
Medium         5 M – 25 M         Jetson Nano, RPi 4
Heavy          > 25 M             Desktop GPU / cloud
=============  ================  ===========================

Typical usage::

    import torch.nn as nn
    from eovot.profiling.model_complexity import ModelComplexityAnalyzer

    analyzer = ModelComplexityAnalyzer()

    # Profile a PyTorch model (e.g., a SiamRPN backbone)
    model = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU())
    result = analyzer.profile_module(model, input_shape=(1, 3, 127, 127))
    print(result)

    # Profile an ONNX / checkpoint file on disk
    result = analyzer.profile_file("checkpoints/siamrpn.pt")
    print(result)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# PyTorch is optional in EOVOT (commented out in requirements.txt).
# All public API must degrade gracefully when torch is unavailable.
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelComplexityResult:
    """Static complexity summary for one deep-learning tracker model.

    Attributes:
        model_name:        Human-readable identifier (class name or file stem).
        total_params:      Total parameter count (trainable + frozen).
        trainable_params:  Parameters that receive gradients during training.
        frozen_params:     Parameters with ``requires_grad=False``.
        estimated_mflops:  Estimated megaFLOPs per forward pass for the
                           given ``input_shape``.  ``None`` when profiling
                           from a file without a forward pass.
        model_size_mb:     On-disk checkpoint / ONNX file size in megabytes.
                           ``None`` when no file path was provided.
        input_shape:       ``(N, C, H, W)`` shape used for FLOPs estimation.
        edge_tier:         Categorical label: ``"Ultralight"``, ``"Light"``,
                           ``"Medium"``, or ``"Heavy"``.
        layer_flops:       Per-layer FLOPs breakdown ``{layer_name: flops}``.
    """

    model_name: str
    total_params: int
    trainable_params: int
    frozen_params: int
    estimated_mflops: Optional[float]
    model_size_mb: Optional[float]
    input_shape: Optional[Tuple[int, ...]]
    edge_tier: str
    layer_flops: Dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived convenience properties
    # ------------------------------------------------------------------

    @property
    def total_params_m(self) -> float:
        """Total parameter count in millions."""
        return self.total_params / 1e6

    @property
    def trainable_params_m(self) -> float:
        """Trainable parameter count in millions."""
        return self.trainable_params / 1e6

    def __str__(self) -> str:
        size_str = f"{self.model_size_mb:.2f} MB" if self.model_size_mb is not None else "—"
        flops_str = f"{self.estimated_mflops:.1f} MFLOPs" if self.estimated_mflops is not None else "—"
        return (
            f"ModelComplexityResult[{self.model_name}]  "
            f"params={self.total_params_m:.3f}M  "
            f"trainable={self.trainable_params_m:.3f}M  "
            f"FLOPs={flops_str}  "
            f"size={size_str}  "
            f"tier={self.edge_tier}"
        )

    def to_dict(self) -> dict:
        """Serialise to a flat dict suitable for JSON export or leaderboard rows."""
        return {
            "model_name": self.model_name,
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
            "frozen_params": self.frozen_params,
            "estimated_mflops": round(self.estimated_mflops, 3) if self.estimated_mflops is not None else None,
            "model_size_mb": round(self.model_size_mb, 3) if self.model_size_mb is not None else None,
            "input_shape": list(self.input_shape) if self.input_shape is not None else None,
            "edge_tier": self.edge_tier,
        }


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------

def _classify_edge_tier(total_params: int) -> str:
    """Classify a model into an edge readiness tier by parameter count.

    Thresholds are calibrated against publicly available model sizes for
    trackers deployed on Raspberry Pi (≤5 M params), Jetson Nano (≤25 M),
    and desktop GPU (>25 M).

    Args:
        total_params: Total parameter count (int).

    Returns:
        One of ``"Ultralight"``, ``"Light"``, ``"Medium"``, ``"Heavy"``.
    """
    if total_params < 500_000:
        return "Ultralight"
    if total_params < 5_000_000:
        return "Light"
    if total_params < 25_000_000:
        return "Medium"
    return "Heavy"


# ---------------------------------------------------------------------------
# FLOPs estimation hooks (PyTorch-only)
# ---------------------------------------------------------------------------

def _make_flop_counter() -> Tuple[Dict[str, int], List]:
    """Return a (flop_dict, hook_handles) pair for forward-hook FLOPs counting."""
    flop_dict: Dict[str, int] = {}
    handles: List = []
    return flop_dict, handles


def _conv2d_hook(flop_dict: Dict[str, int], name: str):
    def hook(module, input, output):
        if not _TORCH_AVAILABLE:
            return
        _, _, h_out, w_out = output.shape
        cin = module.in_channels
        cout = module.out_channels
        kh, kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        groups = module.groups
        # MACs = Cout × (Cin/groups) × kH × kW × H_out × W_out; FLOPs = 2 × MACs
        macs = cout * (cin // groups) * kh * kw * h_out * w_out
        flop_dict[name] = flop_dict.get(name, 0) + int(macs * 2)
    return hook


def _conv_transpose2d_hook(flop_dict: Dict[str, int], name: str):
    def hook(module, input, output):
        if not _TORCH_AVAILABLE:
            return
        _, _, h_out, w_out = output.shape
        cin = module.in_channels
        cout = module.out_channels
        kh, kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        groups = module.groups
        macs = cin * (cout // groups) * kh * kw * h_out * w_out
        flop_dict[name] = flop_dict.get(name, 0) + int(macs * 2)
    return hook


def _linear_hook(flop_dict: Dict[str, int], name: str):
    def hook(module, input, output):
        flop_dict[name] = flop_dict.get(name, 0) + int(module.in_features * module.out_features * 2)
    return hook


def _batchnorm2d_hook(flop_dict: Dict[str, int], name: str):
    def hook(module, input, output):
        if not _TORCH_AVAILABLE:
            return
        _, c, h, w = output.shape
        # scale + shift = 2 ops per element
        flop_dict[name] = flop_dict.get(name, 0) + int(c * h * w * 2)
    return hook


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class ModelComplexityAnalyzer:
    """Measure static complexity properties of PyTorch-based tracking models.

    Provides two primary entry points:

    * :meth:`profile_module` — profile a live ``nn.Module`` with a synthetic
      forward pass (requires PyTorch).
    * :meth:`profile_file` — load a checkpoint/ONNX from disk, count its
      parameters, and measure its file size (requires PyTorch for ``.pt``/
      ``.pth`` files; file-size-only works for any format).

    Both methods return a :class:`ModelComplexityResult` dataclass.

    Args:
        device: Device string for the synthetic forward pass (``"cpu"``
            or ``"cuda"``).  Default: ``"cpu"``.

    Example::

        import torch.nn as nn
        from eovot.profiling.model_complexity import ModelComplexityAnalyzer

        analyzer = ModelComplexityAnalyzer()
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 128 * 128, 10),
        )
        result = analyzer.profile_module(model, input_shape=(1, 3, 128, 128))
        print(result)
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile_module(
        self,
        model: "nn.Module",  # type: ignore[name-defined]
        input_shape: Tuple[int, ...],
        model_name: Optional[str] = None,
    ) -> ModelComplexityResult:
        """Profile a live PyTorch ``nn.Module``.

        Counts parameters, registers forward hooks on ``Conv2d``, ``Linear``,
        ``BatchNorm2d``, and ``ConvTranspose2d`` layers, runs one synthetic
        forward pass with random data, then removes the hooks.

        Args:
            model:       PyTorch model to profile.
            input_shape: Input tensor shape ``(N, C, H, W)`` (or any shape
                         the model accepts — it is passed directly to
                         ``torch.randn``).
            model_name:  Optional name to store in the result.  Defaults to
                         the model's class name.

        Returns:
            :class:`ModelComplexityResult` with all fields populated.

        Raises:
            ImportError: If PyTorch is not installed.
            RuntimeError: If the forward pass fails (e.g., wrong input shape).
        """
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for ModelComplexityAnalyzer.profile_module(). "
                "Install it with: pip install torch"
            )

        name = model_name or model.__class__.__name__
        total_p, trainable_p, frozen_p = self._count_params(model)

        flop_dict, handles = _make_flop_counter()
        self._register_hooks(model, flop_dict, handles)

        model.eval()
        with torch.no_grad():
            dummy = torch.randn(*input_shape, device=self.device)
            try:
                model.to(self.device)(dummy)
            finally:
                for h in handles:
                    h.remove()

        total_flops = sum(flop_dict.values())
        mflops = total_flops / 1e6

        return ModelComplexityResult(
            model_name=name,
            total_params=total_p,
            trainable_params=trainable_p,
            frozen_params=frozen_p,
            estimated_mflops=mflops,
            model_size_mb=None,
            input_shape=tuple(input_shape),
            edge_tier=_classify_edge_tier(total_p),
            layer_flops=dict(flop_dict),
        )

    def profile_file(
        self,
        path: str,
        input_shape: Optional[Tuple[int, ...]] = None,
        model_name: Optional[str] = None,
    ) -> ModelComplexityResult:
        """Profile a model checkpoint or ONNX file on disk.

        Measures the file size for any file format.  For ``.pt`` / ``.pth``
        files, also counts parameters from the saved state dict.  A forward
        pass (and thus FLOPs estimate) requires the caller to pass
        ``input_shape`` *and* have the model's class importable; without
        those, ``estimated_mflops`` is ``None``.

        Args:
            path:        Path to the checkpoint or ONNX file.
            input_shape: Optional input shape for FLOPs estimation.
                         Only used if the file contains a full ``nn.Module``
                         (not just a state dict).
            model_name:  Optional display name.  Defaults to the file stem.

        Returns:
            :class:`ModelComplexityResult`.  ``estimated_mflops`` is ``None``
            when no forward pass was performed.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")

        name = model_name or p.stem
        size_mb = p.stat().st_size / (1024 ** 2)

        total_p = trainable_p = frozen_p = 0
        estimated_mflops: Optional[float] = None

        if _TORCH_AVAILABLE and p.suffix.lower() in {".pt", ".pth"}:
            try:
                obj = torch.load(path, map_location="cpu")
                if isinstance(obj, nn.Module):
                    total_p, trainable_p, frozen_p = self._count_params(obj)
                    if input_shape is not None:
                        result = self.profile_module(obj, input_shape, model_name=name)
                        total_p = result.total_params
                        trainable_p = result.trainable_params
                        frozen_p = result.frozen_params
                        estimated_mflops = result.estimated_mflops
                elif isinstance(obj, dict):
                    # State dict — count parameters without a model instance.
                    for v in obj.values():
                        if hasattr(v, "numel"):
                            total_p += v.numel()
                    trainable_p = total_p
            except Exception:  # pragma: no cover — handles corrupt / incompatible files
                pass

        return ModelComplexityResult(
            model_name=name,
            total_params=total_p,
            trainable_params=trainable_p,
            frozen_params=frozen_p,
            estimated_mflops=estimated_mflops,
            model_size_mb=size_mb,
            input_shape=tuple(input_shape) if input_shape is not None else None,
            edge_tier=_classify_edge_tier(total_p),
        )

    def compare(
        self,
        results: List[ModelComplexityResult],
    ) -> str:
        """Format a comparison table of multiple model profiles as Markdown.

        Args:
            results: List of :class:`ModelComplexityResult` objects.

        Returns:
            Markdown table string suitable for README or PR comments.
        """
        header = "| Model | Params (M) | MFLOPs | Size (MB) | Tier |"
        sep    = "|-------|:----------:|:------:|:---------:|:----:|"
        rows = [header, sep]
        for r in results:
            params_str = f"{r.total_params_m:.3f}"
            flops_str  = f"{r.estimated_mflops:.1f}" if r.estimated_mflops is not None else "—"
            size_str   = f"{r.model_size_mb:.2f}" if r.model_size_mb is not None else "—"
            rows.append(
                f"| {r.model_name} | {params_str} | {flops_str} | {size_str} | {r.edge_tier} |"
            )
        return "\n".join(rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_params(model: "nn.Module") -> Tuple[int, int, int]:  # type: ignore[name-defined]
        """Return (total, trainable, frozen) parameter counts."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable, total - trainable

    @staticmethod
    def _register_hooks(
        model: "nn.Module",  # type: ignore[name-defined]
        flop_dict: Dict[str, int],
        handles: List,
    ) -> None:
        """Attach forward hooks to supported layer types."""
        for full_name, module in model.named_modules():
            name = full_name or module.__class__.__name__
            if isinstance(module, nn.Conv2d):
                handles.append(module.register_forward_hook(_conv2d_hook(flop_dict, name)))
            elif isinstance(module, nn.ConvTranspose2d):
                handles.append(module.register_forward_hook(_conv_transpose2d_hook(flop_dict, name)))
            elif isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(_linear_hook(flop_dict, name)))
            elif isinstance(module, nn.BatchNorm2d):
                handles.append(module.register_forward_hook(_batchnorm2d_hook(flop_dict, name)))

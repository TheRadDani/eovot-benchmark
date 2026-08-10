"""Computational complexity profiler for edge deployment analysis.

Provides FLOPs/MACs estimation for both classical and deep-learning trackers
to enable hardware-agnostic complexity comparison — a key requirement when
benchmarking across diverse edge devices where the same tracker runs at
different wall-clock speeds depending on the hardware.

For **classical trackers** (MOSSE, KCF, CSRT, CamShift, MedianFlow, MIL),
FLOPs are estimated analytically using closed-form formulas derived from each
algorithm's mathematical structure (FFT-based correlation filters, optical
flow pyramids, histogram back-projection, etc.).

For **PyTorch models**, lightweight forward hooks count MACs during a single
forward pass with a dummy input tensor.  Supported layer types: ``Conv2d``,
``Linear``, ``ConvTranspose2d``, ``BatchNorm2d``, ``MultiheadAttention``.
Unknown layers contribute 0 MACs — the total is a lower bound.

The profiler is intentionally decoupled from the benchmark engine so it can
be applied to any tracker without modifying existing evaluation code.

Typical usage::

    from eovot.profiling.complexity import ComplexityAnalyzer

    analyzer = ComplexityAnalyzer()

    # Classical tracker — analytical estimate
    profile = analyzer.estimate_classical(
        name="MOSSE",
        frame_h=240, frame_w=320,
        filter_size=64,
        algorithm="mosse",
    )
    print(profile)
    # ComplexityProfile(MOSSE [classical]  FLOPs=2.51M  MACs=1.26M  ...)

    # Leaderboard across trackers
    configs = [
        ("MOSSE",      "mosse",     240, 320, 64),
        ("KCF",        "kcf",       240, 320, 64),
        ("CSRT",       "csrt",      240, 320, 64),
        ("CamShift",   "camshift",  240, 320, 64),
        ("MedianFlow", "medianflow",240, 320, 64),
    ]
    profiles = analyzer.estimate_tracker_fleet(configs)
    print(ComplexityAnalyzer.to_markdown_table(profiles))

    # PyTorch model
    import torch, torch.nn as nn
    model = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU())
    dummy = torch.zeros(1, 3, 127, 127)
    profile = analyzer.profile_torch_model(model, dummy, name="SiamFC-backbone")
    print(profile)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ComplexityProfile:
    """Computational complexity summary for one tracker or model component.

    Attributes:
        name: Human-readable tracker/model identifier.
        flops: Total floating-point operations.  By convention,
            ``flops = 2 × macs`` (each multiply-accumulate counts as 2).
        macs: Multiply-accumulate operations (MACs / "multiply-adds").
        params: Number of trainable parameters.  Zero for classical trackers
            that have no learned weight tensors.
        memory_read_bytes: Estimated bytes read from memory during one forward
            pass — approximated as ``params × 4`` bytes for float32 models or
            ``template_pixels × 4 × 2`` for classical filters.
        is_classical: ``True`` when the estimate is analytical (classical
            tracker); ``False`` when measured via forward hooks (PyTorch).
    """

    name: str
    flops: int
    macs: int
    params: int
    memory_read_bytes: int
    is_classical: bool = True

    @property
    def gflops(self) -> float:
        """FLOPs expressed in giga-FLOPs (GFLOPs)."""
        return self.flops / 1e9

    @property
    def mflops(self) -> float:
        """FLOPs expressed in mega-FLOPs (MFLOPs)."""
        return self.flops / 1e6

    @property
    def param_millions(self) -> float:
        """Parameter count in millions (M)."""
        return self.params / 1e6

    def __str__(self) -> str:
        src = "classical" if self.is_classical else "measured"
        return (
            f"ComplexityProfile({self.name} [{src}]  "
            f"FLOPs={self.mflops:.2f}M  MACs={self.macs / 1e6:.2f}M  "
            f"params={self.param_millions:.3f}M  "
            f"mem_read={self.memory_read_bytes / 1024:.1f} KB)"
        )


class ComplexityAnalyzer:
    """Estimate or measure computational complexity of visual trackers.

    Two usage modes:

    * **Analytical** (:meth:`estimate_classical`): closed-form FLOPs formulas
      for correlation-filter and classical computer-vision trackers that operate
      on pixel data without a neural network.  Results scale correctly with
      filter size and frame resolution and agree with published algorithmic
      complexity analyses.

    * **Hook-based** (:meth:`profile_torch_model`): attaches temporary forward
      hooks to a PyTorch ``nn.Module`` to count MACs during a single forward
      pass.  The model is left unchanged after profiling.  Requires PyTorch;
      raises ``ImportError`` if PyTorch is not installed.
    """

    _KNOWN_ALGORITHMS = frozenset(
        {"mosse", "kcf", "csrt", "camshift", "medianflow", "mil"}
    )

    # ------------------------------------------------------------------
    # Classical tracker FLOPs estimation
    # ------------------------------------------------------------------

    def estimate_classical(
        self,
        name: str,
        frame_h: int,
        frame_w: int,
        filter_size: int = 64,
        algorithm: str = "mosse",
    ) -> ComplexityProfile:
        """Analytically estimate per-frame FLOPs for a classical tracker.

        FLOPs are computed for a *single tracker update* (one frame), which
        is the quantity that determines real-time feasibility on edge hardware.

        Algorithm-specific derivations
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Let **N = filter_size²** (template pixel count) and **F = frame_h × frame_w**.

        * **MOSSE** — minimum output sum of squared error filter:
          Dominant cost is 2-D FFT correlation.
          ``MACs = 2 × (5N log₂N) + 6N + 4N``
          (forward FFT + inverse FFT + complex element-wise multiply + update)

        * **KCF** — kernelised correlation filter:
          Adds Gaussian kernel map computation over multi-channel features
          (HOG-like linear spatial features) on top of MOSSE.
          ``MACs = 3 × (5N log₂N) + 12N`` (~2.5× MOSSE)

        * **CSRT** — discriminative correlation filter with channel and spatial
          reliability:  Multi-channel HOG features (~54 FLOPs/pixel) + spatial
          reliability mask (2N) + multi-channel FFT correlation (31 channels).
          ``MACs = 54N + 2N + 31 × 5N log₂N``

        * **CamShift** — histogram back-projection + mean shift:
          Full-frame back-projection (2F) + mean-shift search over the patch (10 × 2N).
          ``MACs = 2F + 20N``

        * **MedianFlow** — pyramidal Lucas-Kanade + forward-backward check:
          ~2100 FLOPs per tracked point, ~2× filter_size points tracked.
          ``MACs = filter_size × 2 × 1050``

        * **MIL** — multiple instance learning:
          Binary classifier (positive + negative bag) with ~128-dim features.
          ``MACs = 128 × filter_size × 4``  (scales with active region size)

        Args:
            name: Human-readable label for the profile.
            frame_h: Frame height in pixels (relevant for CamShift).
            frame_w: Frame width in pixels (relevant for CamShift).
            filter_size: Side length of the square correlation template in pixels
                (e.g. 64 for a 64×64 template, giving N = 4096).
            algorithm: One of ``"mosse"``, ``"kcf"``, ``"csrt"``,
                ``"camshift"``, ``"medianflow"``, ``"mil"``
                (case-insensitive).

        Returns:
            :class:`ComplexityProfile` with analytically derived values.

        Raises:
            ValueError: If *algorithm* is not in the known set.
        """
        algo = algorithm.lower()
        if algo not in self._KNOWN_ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                f"Supported: {sorted(self._KNOWN_ALGORITHMS)}"
            )

        N = filter_size * filter_size          # template pixel count
        F = frame_h * frame_w                  # full frame pixel count
        log2N = math.log2(max(N, 2))          # safe log₂ for small sizes

        if algo == "mosse":
            # 2 × FFT (forward + inverse) + complex multiply + model update
            fft_macs = int(5 * N * log2N)
            macs = 2 * fft_macs + 6 * N + 4 * N
            mem_bytes = N * 4 * 4  # filter num/denom (complex, float32)

        elif algo == "kcf":
            # HOG-like linear features + Gaussian kernel + 3 FFT passes
            fft_macs = int(5 * N * log2N)
            macs = 3 * fft_macs + 12 * N
            mem_bytes = N * 4 * 8  # template + kernel (complex, float32)

        elif algo == "csrt":
            # 9-bin HOG (54 FLOPs/px) + spatial reliability (2/px)
            # + 31-channel frequency-domain solve
            hog_macs = 54 * N
            reliability_macs = 2 * N
            fft_macs = int(31 * 5 * N * log2N)
            macs = hog_macs + reliability_macs + fft_macs
            mem_bytes = N * 4 * 31 * 2  # 31-ch complex filters

        elif algo == "camshift":
            # Full-frame histogram back-projection + mean-shift iterations
            backproj_macs = 2 * F
            meanshift_macs = 10 * 2 * N  # 10 iterations, 2 MACs/px
            macs = backproj_macs + meanshift_macs
            mem_bytes = 256 * 4  # 256-bin hue histogram

        elif algo == "medianflow":
            # Pyramidal LK (~1050 MACs/point) × tracked points, ×2 for FB check
            num_points = filter_size * 2   # active tracking points
            macs = num_points * 1050 * 2  # forward + backward pass
            mem_bytes = num_points * 8 * 4  # point coords (x,y) float32

        else:  # mil
            # SVM-like binary classifier with 128-dim features
            feat_dim = 128
            macs = feat_dim * filter_size * 4  # positive + negative bags × 2
            mem_bytes = feat_dim * filter_size * 4  # weight matrix

        flops = 2 * macs  # FLOPs = 2 × MACs (standard convention)
        return ComplexityProfile(
            name=name,
            flops=flops,
            macs=macs,
            params=0,
            memory_read_bytes=mem_bytes,
            is_classical=True,
        )

    # ------------------------------------------------------------------
    # PyTorch model complexity (hook-based)
    # ------------------------------------------------------------------

    def profile_torch_model(
        self,
        model: object,
        input_tensor: object,
        name: str = "model",
    ) -> ComplexityProfile:
        """Measure MACs and parameter count for a PyTorch model via forward hooks.

        Forward hooks are registered before inference and removed afterward;
        the model's state and training mode are fully restored.

        Supported layer types and their MACs formula:

        * ``Conv2d``: ``batch × out_ch × out_H × out_W × (in_ch/groups) × kH × kW``
        * ``ConvTranspose2d``: same formula (treats it as a transposed convolution)
        * ``Linear``: ``batch × out_features × in_features``
        * ``BatchNorm2d``: ``2 × numel(output)`` (normalise + affine)
        * ``MultiheadAttention``: Q/K/V projections + softmax approximation
          + output projection

        Unknown layers contribute 0 MACs — the total is a conservative
        lower bound on true computational cost.

        Args:
            model: PyTorch ``nn.Module`` to profile.
            input_tensor: Example input with the correct shape (typically
                ``(1, C, H, W)`` for image models).
            name: Human-readable label for the returned profile.

        Returns:
            :class:`ComplexityProfile` with ``is_classical=False`` and
            measured ``macs``, ``params``, and ``memory_read_bytes``.

        Raises:
            ImportError: If PyTorch (``torch``) is not installed.
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PyTorch is required for profile_torch_model(). "
                "Install it with: pip install torch"
            ) from exc

        macs_acc: List[int] = [0]
        hooks: list = []

        def _conv_hook(module: object, inp: tuple, out: object) -> None:
            m = module  # type: ignore[assignment]
            batch = out.shape[0]  # type: ignore[union-attr]
            out_h, out_w = out.shape[2], out.shape[3]  # type: ignore[union-attr]
            kh, kw = (
                m.kernel_size
                if isinstance(m.kernel_size, tuple)
                else (m.kernel_size, m.kernel_size)
            )  # type: ignore[union-attr]
            in_ch = m.in_channels // m.groups  # type: ignore[union-attr]
            macs_acc[0] += int(batch * m.out_channels * out_h * out_w * in_ch * kh * kw)  # type: ignore[union-attr]

        def _linear_hook(module: object, inp: tuple, out: object) -> None:
            m = module  # type: ignore[assignment]
            batch = out.numel() // out.shape[-1]  # type: ignore[union-attr]
            macs_acc[0] += int(batch * m.out_features * m.in_features)  # type: ignore[union-attr]

        def _bn_hook(module: object, inp: tuple, out: object) -> None:
            macs_acc[0] += int(2 * out.numel())  # type: ignore[union-attr]

        def _mha_hook(module: object, inp: tuple, out: object) -> None:
            m = module  # type: ignore[assignment]
            q = inp[0]
            seq_len, embed = q.shape[0], m.embed_dim  # type: ignore[union-attr]
            # Q/K/V projections + scaled dot-product + output projection
            macs_acc[0] += int(
                3 * seq_len * embed * embed   # Q, K, V projections
                + seq_len * seq_len * embed   # attention weights
                + seq_len * embed * embed     # output projection
            )

        for mod in model.modules():  # type: ignore[union-attr]
            if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
                hooks.append(mod.register_forward_hook(_conv_hook))
            elif isinstance(mod, nn.Linear):
                hooks.append(mod.register_forward_hook(_linear_hook))
            elif isinstance(mod, nn.BatchNorm2d):
                hooks.append(mod.register_forward_hook(_bn_hook))
            elif isinstance(mod, nn.MultiheadAttention):
                hooks.append(mod.register_forward_hook(_mha_hook))

        was_training = model.training  # type: ignore[union-attr]
        model.eval()  # type: ignore[union-attr]
        try:
            with torch.no_grad():
                model(input_tensor)  # type: ignore[operator]
        finally:
            for h in hooks:
                h.remove()
            if was_training:
                model.train()  # type: ignore[union-attr]

        total_macs = macs_acc[0]
        total_params = sum(p.numel() for p in model.parameters())  # type: ignore[union-attr]

        return ComplexityProfile(
            name=name,
            flops=2 * total_macs,
            macs=total_macs,
            params=total_params,
            memory_read_bytes=total_params * 4,  # float32
            is_classical=False,
        )

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def estimate_tracker_fleet(
        self,
        tracker_configs: List[Tuple[str, str, int, int, int]],
    ) -> List[ComplexityProfile]:
        """Estimate complexity for a list of classical trackers.

        Args:
            tracker_configs: Each entry is a tuple
                ``(name, algorithm, frame_h, frame_w, filter_size)``.

        Returns:
            Ordered list of :class:`ComplexityProfile` objects in the same
            order as *tracker_configs*.
        """
        return [
            self.estimate_classical(
                name=name,
                frame_h=fh,
                frame_w=fw,
                filter_size=fs,
                algorithm=algo,
            )
            for name, algo, fh, fw, fs in tracker_configs
        ]

    @staticmethod
    def to_markdown_table(profiles: List[ComplexityProfile]) -> str:
        """Format a list of complexity profiles as a Markdown comparison table.

        Columns: Tracker, Source, MFLOPs, MMACs, Params (M), Mem Read (KB).

        Args:
            profiles: List of :class:`ComplexityProfile` objects to tabulate.

        Returns:
            Multi-line Markdown string suitable for README files or paper
            appendices.
        """
        lines = [
            "| Tracker | Source | MFLOPs | MMACs | Params (M) | Mem Read (KB) |",
            "|---------|--------|-------:|------:|-----------:|--------------:|",
        ]
        for p in profiles:
            src = "classical" if p.is_classical else "measured"
            lines.append(
                f"| {p.name} | {src} "
                f"| {p.mflops:.3f} | {p.macs / 1e6:.3f} "
                f"| {p.param_millions:.4f} "
                f"| {p.memory_read_bytes / 1024:.2f} |"
            )
        return "\n".join(lines)

"""SiamFC (Siamese Fully Convolutional) tracker for EOVOT.

Implements the original SiamFC architecture from Bertinetto et al. (2016)
with a lightweight backbone designed for edge deployment.

Design choices
--------------
* **Pure PyTorch** — no external ONNX runtime required; runs on any device
  with a standard PyTorch install (CPU or CUDA).
* **Lightweight backbone** — 5 convolutional layers (~2.3 M parameters) vs
  the full AlexNet used in the original paper, making it viable on Jetson
  Nano and similar hardware.
* **Random initialisation by default** — the tracker can be instantiated
  and run immediately for latency/memory benchmarking without any weight
  file.  Provide ``model_path`` for accuracy evaluation.
* **Context-padded crops** — follows the original paper's convention of
  padding by half the mean side-length before extracting template/search
  crops, producing a square region that encodes spatial context.
* **Upsampled score map** — the 17×17 correlation score map is bicubically
  upsampled before peak localisation to achieve sub-pixel displacement
  estimates.

Architecture
------------
Template z:  (B, 3, 127, 127) → backbone → (B, 256, 6, 6)
Search   x:  (B, 3, 255, 255) → backbone → (B, 256, 22, 22)
Score map:   F.conv2d(x_feat, z_feat) → (B, 1, 17, 17)

Reference
---------
Bertinetto et al., "Fully-Convolutional Siamese Networks for Object
Tracking." ECCV Workshop on Visual Object Challenge, 2016.
arXiv:1606.09549.

Usage
-----
::

    # Latency / memory benchmark (no weights needed):
    tracker = SiamFCTracker()

    # Accuracy evaluation (requires SiamFC pretrained weights):
    tracker = SiamFCTracker(model_path="weights/siamfc_alexnet.pth")

    # Explicit device placement:
    tracker = SiamFCTracker(device="cuda:0")
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Backbone
# ──────────────────────────────────────────────────────────────────────────────


class _SiamFCBackbone(nn.Module):
    """Lightweight 5-layer AlexNet-inspired CNN backbone for SiamFC.

    Spatial resolution is reduced by a factor of ~8 through two max-pool
    layers (stride 2 each) and the conv1 stride of 2, matching the original
    SiamFC paper:

        Input 127×127  →  output  6×6×256  (template branch)
        Input 255×255  →  output 22×22×256  (search branch)

    The same weights are shared between the template and search branches at
    inference time (Siamese structure).

    Parameter count: ~2.3 M (FP32), ~0.6 MB when quantised to INT8.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, bias=False),
            nn.BatchNorm2d(256),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # noqa: F821
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.conv5(x)


# ──────────────────────────────────────────────────────────────────────────────
# Tracker
# ──────────────────────────────────────────────────────────────────────────────


class SiamFCTracker(BaseTracker):
    """SiamFC visual object tracker.

    On :meth:`initialize`, a context-padded exemplar crop is extracted,
    passed through the backbone, and stored as the template embedding.
    On each subsequent :meth:`update`, a larger search-region crop is
    extracted at the previous position, encoded by the same backbone, and
    cross-correlated with the template.  The peak of the resulting 17×17
    score map (upsampled for sub-pixel precision) gives the displacement
    estimate.

    Args:
        model_path: Path to a ``.pth`` checkpoint containing backbone
            weights.  Supported formats:

            * Raw ``state_dict`` (keys matching :class:`_SiamFCBackbone`).
            * ``{"backbone": state_dict}``
            * ``{"state_dict": state_dict}``

            If ``None`` (default), random weights are used — suitable for
            latency/memory benchmarking but **not** for accuracy evaluation.
        device: PyTorch device string (``"cpu"``, ``"cuda:0"``, …).
            Defaults to CUDA if available, otherwise CPU.
        template_size: Side length (px) of the template crop fed to the
            backbone.  Must match the training convention (default: 127).
        search_size: Side length (px) of the search crop fed to the
            backbone.  Must match the training convention (default: 255).
        response_upscale: Integer upsampling factor applied to the 17×17
            score map before peak localisation.  Larger values give finer
            sub-pixel displacement estimates at negligible extra cost.
            Default: 16.
        context_amount: Fraction of the mean target side-length added as
            context padding on each side of the target when computing crop
            dimensions.  Default: 0.5 (matches the original paper).

    Example::

        tracker = SiamFCTracker(device="cpu")
        tracker.initialize(frame, (x, y, w, h))
        for frame in sequence:
            bbox = tracker.update(frame)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        template_size: int = 127,
        search_size: int = 255,
        response_upscale: int = 16,
        context_amount: float = 0.5,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for SiamFCTracker. "
                "Install it with: pip install torch"
            )
        super().__init__(name="SiamFC")

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.template_size = template_size
        self.search_size = search_size
        self.response_upscale = response_upscale
        self.context_amount = context_amount

        self._backbone = _SiamFCBackbone().to(self.device).eval()
        if model_path is not None:
            self._load_weights(model_path)

        # State cleared on each initialize() call.
        self._z_feat: Optional["torch.Tensor"] = None  # noqa: F821
        self._pos: Optional[Tuple[float, float]] = None
        self._target_sz: Optional[Tuple[float, float]] = None
        self._z_crop_side: Optional[float] = None

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the tracker on the first frame.

        Extracts a context-padded template crop, encodes it through the
        backbone, and stores the resulting feature tensor.

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 numpy array.
            bbox: Initial bounding box ``(x, y, w, h)``.
        """
        x, y, w, h = (float(v) for v in bbox)
        cx, cy = x + w / 2.0, y + h / 2.0
        self._pos = (cx, cy)
        self._target_sz = (w, h)

        # Context-padded square crop side length for the template.
        context = self.context_amount * (w + h)
        self._z_crop_side = math.sqrt((w + context) * (h + context))

        patch = self._get_crop(frame, cx, cy, self._z_crop_side, self.template_size)
        with torch.no_grad():
            self._z_feat = self._encode(patch)

    def update(self, frame: np.ndarray) -> BBox:
        """Locate the target in a new frame via cross-correlation.

        The search region is centred at the previous estimated position and
        scaled to ``search_size / template_size`` times the template crop
        size, providing a proportionally larger context window for search.

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 numpy array.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._pos is None or self._z_feat is None:
            raise RuntimeError(
                "SiamFCTracker is not initialised. Call initialize() first."
            )

        cx, cy = self._pos
        tw, th = self._target_sz

        # Scale the search region proportionally to the template crop.
        scale = self.search_size / self.template_size
        search_crop_side = self._z_crop_side * scale  # type: ignore[operator]

        patch = self._get_crop(frame, cx, cy, search_crop_side, self.search_size)
        with torch.no_grad():
            x_feat = self._encode(patch)
            # Cross-correlation: search (1,C,22,22) ⋆ template (1,C,6,6) → (1,1,17,17)
            score = F.conv2d(x_feat, self._z_feat)

        score_np: np.ndarray = score.squeeze().cpu().numpy()  # (17, 17)
        h_s, w_s = score_np.shape

        # Bicubic upsampling for sub-pixel localisation.
        score_up = cv2.resize(
            score_np,
            (w_s * self.response_upscale, h_s * self.response_upscale),
            interpolation=cv2.INTER_CUBIC,
        )
        H_up, W_up = score_up.shape

        # Peak position relative to the score-map centre.
        peak_y, peak_x = np.unravel_index(np.argmax(score_up), score_up.shape)
        dy = peak_y - H_up // 2
        dx = peak_x - W_up // 2

        # Each upsampled cell = (search_crop_side / W_up) image pixels.
        new_cx = cx + dx * (search_crop_side / W_up)
        new_cy = cy + dy * (search_crop_side / H_up)
        self._pos = (new_cx, new_cy)

        return (new_cx - tw / 2.0, new_cy - th / 2.0, tw, th)

    def reset(self) -> None:
        """Clear internal state so the tracker can be re-initialised."""
        self._z_feat = None
        self._pos = None
        self._target_sz = None
        self._z_crop_side = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_crop(
        self,
        frame: np.ndarray,
        cx: float,
        cy: float,
        crop_side: float,
        out_size: int,
    ) -> np.ndarray:
        """Extract a square crop of side ``crop_side`` px, resize to ``out_size``.

        Out-of-bounds regions are filled with edge-replicated values
        (``cv2.BORDER_REPLICATE``) to avoid introducing a zero-valued
        artefact at image borders that would corrupt feature statistics.

        Args:
            frame: Source BGR image.
            cx, cy: Crop centre in image coordinates.
            crop_side: Desired crop side length in image pixels.
            out_size: Target output size (both dimensions).

        Returns:
            RGB float32 array of shape ``(out_size, out_size, 3)``,
            values in ``[0, 255]``.
        """
        fh, fw = frame.shape[:2]
        half = crop_side / 2.0

        x1 = int(round(cx - half))
        y1 = int(round(cy - half))
        x2 = int(round(cx + half))
        y2 = int(round(cy + half))

        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - fw)
        pad_b = max(0, y2 - fh)

        x1c = max(0, x1)
        y1c = max(0, y1)
        x2c = min(fw, x2)
        y2c = min(fh, y2)

        patch = frame[y1c:y2c, x1c:x2c]

        if pad_l or pad_t or pad_r or pad_b:
            patch = cv2.copyMakeBorder(
                patch, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE
            )

        if patch.size == 0:
            patch = np.zeros((out_size, out_size, 3), dtype=np.uint8)
        elif patch.shape[0] != out_size or patch.shape[1] != out_size:
            patch = cv2.resize(patch, (out_size, out_size))

        # BGR → RGB, float32 in [0, 255].
        return cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32)

    def _encode(self, patch: np.ndarray) -> "torch.Tensor":  # noqa: F821
        """Run the backbone on a cropped patch.

        Args:
            patch: ``(H, W, 3)`` float32 RGB array, values in ``[0, 255]``.

        Returns:
            Feature tensor of shape ``(1, 256, h, w)``.
        """
        t = (
            torch.from_numpy(patch / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )
        return self._backbone(t)

    def _load_weights(self, path: str) -> None:
        """Load backbone weights from a ``.pth`` checkpoint file.

        Supports three common checkpoint formats so the tracker can be
        used with weights saved by different training pipelines.

        Args:
            path: Path to the checkpoint file.
        """
        ckpt = torch.load(path, map_location=self.device)
        if isinstance(ckpt, dict) and "backbone" in ckpt:
            state = ckpt["backbone"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
        missing, unexpected = self._backbone.load_state_dict(state, strict=False)
        if missing:
            import warnings
            warnings.warn(
                f"SiamFCTracker: {len(missing)} backbone keys not found in "
                f"checkpoint: {missing[:5]}{'...' if len(missing) > 5 else ''}",
                UserWarning,
                stacklevel=2,
            )

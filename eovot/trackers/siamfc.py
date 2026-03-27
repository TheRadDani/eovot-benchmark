"""SiamFC fully-convolutional Siamese tracker for EOVOT.

Implements the tracker from:
    Bertinetto et al., "Fully-Convolutional Siamese Networks for Object
    Tracking", ECCVW 2016.  https://arxiv.org/abs/1606.09549

The tracker uses an AlexNet-style backbone with no padding convolutions to
extract deep feature maps, then localises the target via cross-correlation
between the template (first-frame crop) and each subsequent search-region
crop.

PyTorch is an **optional** dependency.  Attempting to instantiate
:class:`SiamFCTracker` without PyTorch installed raises a clear
``ImportError`` with installation instructions.  All other EOVOT modules
remain usable without it.

Typical usage::

    from eovot.trackers.siamfc import SiamFCTracker

    tracker = SiamFCTracker(device="cpu")
    # Optional: load pretrained backbone weights
    # tracker.load_weights("path/to/siamfc_backbone.pth")

    tracker.initialize(first_frame, init_bbox)
    for frame in subsequent_frames:
        bbox = tracker.update(frame)

Architecture notes:
- Backbone: 5-layer AlexNet without padding (conv only, no FC layers)
- Template features: (256, 6, 6) for z_size=127
- Search features: (256, 22, 22) for x_size=255
- Response map: (1, 17, 17) from cross-correlation
- Total feature stride: 8 (stride-2 conv × 2× max-pool)
- Scale estimation: 3-scale pyramid (default step=1.05)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox


def _require_torch() -> None:
    """Raise a helpful ImportError when PyTorch is not installed."""
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "SiamFCTracker requires PyTorch.  Install it with:\n"
            "  pip install torch torchvision\n"
            "or visit https://pytorch.org/get-started/locally/ for GPU builds."
        ) from exc


# ---------------------------------------------------------------------------
# Backbone network
# ---------------------------------------------------------------------------

def _build_backbone():
    """Return an AlexNet-style backbone nn.Module (requires torch)."""
    import torch.nn as nn

    class _AlexNetBackbone(nn.Module):
        """5-layer AlexNet backbone with no-padding convolutions.

        Spatial output sizes (no padding, no dilation):
          Input z=127 → (256, 6, 6)
          Input x=255 → (256, 22, 22)
        Total stride: 2 × 2 × 2 = 8
        """

        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                # --- Block 1 ---
                nn.Conv2d(3, 96, kernel_size=11, stride=2, bias=False),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                # --- Block 2 ---
                nn.Conv2d(96, 256, kernel_size=5, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                # --- Block 3 ---
                nn.Conv2d(256, 384, kernel_size=3, bias=False),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
                # --- Block 4 ---
                nn.Conv2d(384, 384, kernel_size=3, bias=False),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
                # --- Block 5 (no ReLU — following original SiamFC) ---
                nn.Conv2d(384, 256, kernel_size=3, bias=False),
                nn.BatchNorm2d(256),
            )

        def forward(self, x):
            return self.features(x)

    return _AlexNetBackbone()


# ---------------------------------------------------------------------------
# Cross-correlation
# ---------------------------------------------------------------------------

def _xcorr(search_feat, template_feat):
    """Compute cross-correlation response map via grouped convolution.

    Args:
        search_feat:   (1, C, H, W) search feature map.
        template_feat: (1, C, h, w) template feature map.

    Returns:
        (1, 1, H-h+1, W-w+1) response map.
    """
    import torch.nn.functional as F

    b, c, h, w = template_feat.shape
    # Treat each channel of the template as a conv filter, group by batch.
    response = F.conv2d(
        search_feat.view(1, b * c, *search_feat.shape[-2:]),
        template_feat.view(b * c, 1, h, w),
        groups=b * c,
    )
    # Average channel responses → (1, 1, H', W')
    return response.view(b, c, *response.shape[-2:]).mean(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------

class SiamFCTracker(BaseTracker):
    """SiamFC fully-convolutional Siamese tracker.

    Args:
        template_size: Size of the square template crop fed to the backbone.
            Default: 127 (standard SiamFC).
        search_size: Size of the square search-region crop.
            Default: 255 (standard SiamFC).
        context_amount: Fraction of ``(w + h)`` added as context margin when
            computing crop sizes.  Default: 0.5.
        scale_step: Multiplicative step between scale candidates.
            Three scales ``[1/step, 1, step]`` are evaluated each frame.
            Default: 1.05.
        scale_lr: Exponential moving-average weight for the scale update.
            Default: 0.59 (from original implementation).
        scale_penalty: Multiplicative penalty applied to off-scale responses
            to avoid erratic size changes.  Default: 0.9745.
        response_up: Bilinear upsampling factor applied to the response map
            before peak finding, enabling sub-pixel localisation.
            Default: 16.
        device: PyTorch device string, e.g. ``"cpu"`` or ``"cuda:0"``.
        weights_path: Optional path to a ``.pth`` checkpoint.  The file may
            contain the full ``state_dict`` or a nested dict with a
            ``"backbone"`` key.

    Example::

        tracker = SiamFCTracker(device="cpu")
        tracker.initialize(first_frame, (x, y, w, h))
        for frame in video_frames:
            bbox = tracker.update(frame)
    """

    _STRIDE = 8  # total backbone stride (2 × pool × pool)

    def __init__(
        self,
        template_size: int = 127,
        search_size: int = 255,
        context_amount: float = 0.5,
        scale_step: float = 1.05,
        scale_lr: float = 0.59,
        scale_penalty: float = 0.9745,
        response_up: int = 16,
        device: str = "cpu",
        weights_path: Optional[str] = None,
    ) -> None:
        super().__init__(name="SiamFC")
        _require_torch()
        import torch

        self.template_size = template_size
        self.search_size = search_size
        self.context_amount = context_amount
        self.scale_step = scale_step
        self.scale_lr = scale_lr
        self.scale_penalty = scale_penalty
        self.response_up = response_up
        self.device = torch.device(device)

        self._backbone = _build_backbone().to(self.device).eval()
        if weights_path is not None:
            self.load_weights(weights_path)

        # Tracker state — set on initialize()
        self._template_feat = None
        self._bbox: Optional[BBox] = None
        self._mean_color: Optional[np.ndarray] = None
        self._scales = np.array([
            scale_step ** (-1), 1.0, scale_step ** 1
        ])

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the tracker on the first frame.

        Args:
            frame: BGR uint8 image ``(H, W, 3)``.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        import torch

        self._bbox = bbox
        # Per-channel mean used to pad crops near image boundaries.
        self._mean_color = frame.mean(axis=(0, 1))

        z_crop = self._get_subwindow(frame, bbox, self.template_size)
        z_tensor = self._to_tensor(z_crop)
        with torch.no_grad():
            self._template_feat = self._backbone(z_tensor)

    def update(self, frame: np.ndarray) -> BBox:
        """Predict the target location in the current frame.

        Args:
            frame: BGR uint8 image ``(H, W, 3)``.

        Returns:
            Updated bounding box ``(x, y, w, h)``.
        """
        import torch
        import torch.nn.functional as F

        x, y, w, h = self._bbox

        # Evaluate response at three scale candidates.
        responses = []
        for scale in self._scales:
            scaled_bbox = (x, y, w * scale, h * scale)
            x_crop = self._get_subwindow(frame, scaled_bbox, self.search_size)
            x_tensor = self._to_tensor(x_crop)
            with torch.no_grad():
                x_feat = self._backbone(x_tensor)
                resp = _xcorr(x_feat, self._template_feat)
                if self.response_up > 1:
                    resp = F.interpolate(
                        resp, scale_factor=self.response_up,
                        mode="bicubic", align_corners=False,
                    )
            responses.append(resp.squeeze().cpu().numpy())

        # Apply scale penalty to non-identity scales.
        penalties = np.full(len(self._scales), self.scale_penalty)
        penalties[len(self._scales) // 2] = 1.0  # no penalty for scale == 1
        responses = [r * p for r, p in zip(responses, penalties)]

        # Pick the scale with the highest peak response.
        peaks = [r.max() for r in responses]
        best_scale_idx = int(np.argmax(peaks))
        best_scale = self._scales[best_scale_idx]
        resp_map = responses[best_scale_idx]

        # Find peak location.
        peak_idx = np.unravel_index(resp_map.argmax(), resp_map.shape)
        center = np.array(resp_map.shape) / 2.0
        # Displacement in response-map units (after upsampling).
        disp_resp = np.array(peak_idx, dtype=float) - center
        # Convert to search-image pixel displacement.
        disp_search = disp_resp * (self._STRIDE / self.response_up)
        # Scale to original-frame pixels.
        s_x = self._get_crop_size(w * best_scale, h * best_scale) * (
            self.search_size / self.template_size
        )
        disp_orig = disp_search * s_x / self.search_size

        # Update center position.
        cx = x + w / 2 + disp_orig[1]
        cy = y + h / 2 + disp_orig[0]

        # Smooth scale update via EMA.
        new_w = w * (1 - self.scale_lr + self.scale_lr * best_scale)
        new_h = h * (1 - self.scale_lr + self.scale_lr * best_scale)

        self._bbox = (cx - new_w / 2, cy - new_h / 2, new_w, new_h)
        return self._bbox

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def load_weights(self, path: str) -> None:
        """Load backbone weights from a PyTorch checkpoint.

        The checkpoint may be:
        - A bare ``state_dict`` for the backbone.
        - A nested dict with a ``"backbone"`` or ``"net"`` key.

        Args:
            path: Path to the ``.pth`` file.
        """
        import torch

        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict):
            for key in ("backbone", "net", "model"):
                if key in state:
                    state = state[key]
                    break
        self._backbone.load_state_dict(state, strict=False)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_crop_size(self, w: float, h: float) -> float:
        """Template crop size s_z = sqrt((w + context)(h + context))."""
        context = self.context_amount * (w + h)
        return math.sqrt((w + context) * (h + context))

    def _get_subwindow(
        self, frame: np.ndarray, bbox: BBox, out_size: int
    ) -> np.ndarray:
        """Crop a square sub-window centred on *bbox*, pad with mean colour.

        For the template we use the target's natural scale (s_z = template_size
        in the original frame).  For the search region we use s_x = s_z ×
        (search_size / template_size).

        Args:
            frame:    BGR uint8 image.
            bbox:     Bounding box ``(x, y, w, h)`` in the original frame.
            out_size: Output square size (either template_size or search_size).

        Returns:
            ``out_size × out_size`` BGR uint8 crop.
        """
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2

        # Compute crop region in the original frame.
        s_z = self._get_crop_size(w, h)
        scale = out_size / s_z
        half = s_z / 2

        H, W = frame.shape[:2]
        x0, y0 = cx - half, cy - half
        x1, y1 = cx + half, cy + half

        # Padding required when crop extends outside the frame.
        pad_l = max(0.0, -x0)
        pad_t = max(0.0, -y0)
        pad_r = max(0.0, x1 - W)
        pad_b = max(0.0, y1 - H)

        x0c = int(max(0, round(x0)))
        y0c = int(max(0, round(y0)))
        x1c = int(min(W, round(x1)))
        y1c = int(min(H, round(y1)))

        crop = frame[y0c:y1c, x0c:x1c].copy()

        if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
            mean = (
                self._mean_color.astype(int).tolist()
                if self._mean_color is not None
                else [128, 128, 128]
            )
            crop = cv2.copyMakeBorder(
                crop,
                int(round(pad_t)), int(round(pad_b)),
                int(round(pad_l)), int(round(pad_r)),
                cv2.BORDER_CONSTANT,
                value=mean,
            )

        if crop.size == 0:
            crop = np.full((out_size, out_size, 3), 128, dtype=np.uint8)
        else:
            crop = cv2.resize(crop, (out_size, out_size),
                              interpolation=cv2.INTER_LINEAR)
        return crop

    def _to_tensor(self, img: np.ndarray):
        """Convert a BGR uint8 crop to a normalised float32 tensor (1, 3, H, W).

        Uses ImageNet mean/std normalisation consistent with the standard
        pretrained backbone expectations.
        """
        import torch

        # BGR → RGB, (H, W, C) → (C, H, W), [0, 255] → [0, 1]
        rgb = img[:, :, ::-1].copy()
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).div_(255.0)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)
        return tensor.to(self.device).sub_(mean).div_(std).unsqueeze(0)

"""SiamFC-Lite — lightweight Siamese cross-correlation tracker.

Implements a simplified version of:

    Bertinetto et al., "Fully-Convolutional Siamese Networks for Object
    Tracking." ECCV 2016 Workshop. https://arxiv.org/abs/1606.09549

Architecture
------------
* **Backbone** — 3 convolutional blocks (Conv → BatchNorm → ReLU) followed by
  L2 normalisation.  Total ~30 k parameters; designed to fit within the
  memory envelope of Raspberry Pi 4 and Jetson Nano.

  ::

    Conv(3→32, 11×11, stride=2) → BN → ReLU
    Conv(32→64,  5×5, stride=2) → BN → ReLU
    Conv(64→64,  3×3, stride=2) → BN  ← no ReLU; L2-norm applied after

  Effective stride = 8, receptive field ≈ 47 px.

* **Template branch** — extracts ``z``-features from a context-padded crop
  centred on the initial target (127 × 127 pixels → 13 × 13 feature map).

* **Search branch** — extracts ``x``-features from a larger search region
  centred on the last known position (255 × 255 pixels → 29 × 29 feature map).

* **Response map** — depth-wise cross-correlation between ``z``- and
  ``x``-features produces a 17 × 17 score map.  The argmax, scaled back to
  image coordinates, gives the new target centre.

Usage (no pre-trained weights — full pipeline test)
----------------------------------------------------
::

    import numpy as np
    from eovot.trackers.siamfc_lite import SiamFCLiteTracker

    tracker = SiamFCLiteTracker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracker.initialize(frame, (200, 150, 80, 60))
    bbox = tracker.update(frame)   # exercises full forward-pass pipeline

Loading pre-trained weights
---------------------------
::

    tracker = SiamFCLiteTracker.from_checkpoint("checkpoints/siamfc_lite.pth")

The checkpoint must be a ``state_dict`` saved with::

    torch.save(tracker._backbone.state_dict(), "checkpoints/siamfc_lite.pth")

Note
----
Without pre-trained weights the response map is random noise and target
localisation is meaningless.  The implementation is intentionally provided
without weights to:

1. Keep the repository lightweight (no large binary assets).
2. Allow researchers to fine-tune on their own data (e.g. GOT-10k).
3. Let the benchmark engine measure inference latency / memory correctly
   regardless of tracking accuracy.

Requires
--------
    torch >= 1.13
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .base import BaseTracker, BBox

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


class _SiamFCBackbone(nn.Module):
    """Lightweight 3-block CNN backbone for SiamFC-Lite.

    Accepts ``(B, 3, H, W)`` RGB tensors normalised to ``[0, 1]`` and returns
    L2-normalised feature maps suitable for cross-correlation scoring.

    The architecture is deliberately minimal:

    * No pooling layers — downsampling is entirely via strided convolutions.
    * BatchNorm after every conv for stable training and deterministic eval.
    * L2 normalisation along the channel dimension to bound correlation scores.

    Args:
        n_channels: Number of output feature channels.  Changing this value
            will alter the backbone stride and receptive field — only the
            default of ``64`` is guaranteed to produce a 13 × 13 feature map
            from a 127 × 127 input.
    """

    def __init__(self, n_channels: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1 — large receptive field for coarse localisation
            nn.Conv2d(3, 32, kernel_size=11, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Block 2 — medium receptive field
            nn.Conv2d(32, n_channels, kernel_size=5, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
            # Block 3 — fine-grained feature discrimination
            nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(n_channels),
            # No ReLU before L2-norm — preserves sign for cross-correlation
        )
        self.n_channels = n_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalised feature maps.

        Args:
            x: Input tensor ``(B, 3, H, W)`` in ``[0, 1]``.

        Returns:
            Feature map ``(B, C, H', W')`` with unit L2 norm per channel.
        """
        feats = self.features(x)
        return F.normalize(feats, p=2, dim=1)


class SiamFCLiteTracker(BaseTracker):
    """Lightweight Siamese cross-correlation tracker (SiamFC-Lite).

    See module docstring for architecture details and literature reference.

    Args:
        template_size: Side length (px) of the template crop fed to the
            backbone.  Default: ``127`` (matches the original SiamFC paper).
        search_size: Side length (px) of the search region.  Must be larger
            than ``template_size``.  Default: ``255``.
        context_amount: Context padding fraction — how much background to
            include around the target when cropping.  Default: ``0.5``
            (matches the original SiamFC paper).
        stride: Total stride of the backbone network (8).  Used to convert
            response-map peak coordinates back to image coordinates.  This
            must match the actual backbone stride; do not change unless you
            also change the backbone architecture.
        device: PyTorch device string.  ``"cpu"`` by default; set to
            ``"cuda"`` for GPU inference if a compatible GPU is present.

    Raises:
        ImportError: If PyTorch (``torch``) is not installed.

    Example::

        tracker = SiamFCLiteTracker(device="cpu")
        tracker.initialize(first_frame, (x, y, w, h))
        for frame in subsequent_frames:
            bbox = tracker.update(frame)
    """

    def __init__(
        self,
        template_size: int = 127,
        search_size: int = 255,
        context_amount: float = 0.5,
        stride: int = 8,
        device: str = "cpu",
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError(
                "SiamFCLiteTracker requires PyTorch >= 1.13. "
                "Install it with: pip install torch"
            )
        super().__init__(name="SiamFC-Lite")
        self.template_size = template_size
        self.search_size = search_size
        self.context_amount = context_amount
        self.stride = stride
        self.device = torch.device(device)

        self._backbone = _SiamFCBackbone().to(self.device)
        self._backbone.eval()

        # Per-sequence state populated in initialize()
        self._template_feat: Optional[torch.Tensor] = None
        self._bbox: Optional[list] = None  # [cx, cy, w, h] in image coords

    # ------------------------------------------------------------------
    # Class-level factory
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, path: str, **kwargs) -> "SiamFCLiteTracker":
        """Load a pre-trained backbone from a state-dict checkpoint.

        Args:
            path: Path to a ``.pth`` file saved via::

                    torch.save(tracker._backbone.state_dict(), path)

            **kwargs: Forwarded to the :class:`SiamFCLiteTracker` constructor
                (e.g. ``device="cuda"``).

        Returns:
            Tracker instance with loaded backbone weights in ``eval`` mode.
        """
        tracker = cls(**kwargs)
        state = torch.load(path, map_location=tracker.device)
        tracker._backbone.load_state_dict(state)
        tracker._backbone.eval()
        return tracker

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the tracker on the first frame of a sequence.

        Crops a context-padded template patch centred on ``bbox``, extracts
        its features with the backbone, and caches them for subsequent
        cross-correlation calls.

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 array.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        cx, cy = x + w / 2.0, y + h / 2.0
        self._bbox = [cx, cy, w, h]

        z_crop = self._crop_patch(frame, cx, cy, w, h, self.template_size)
        with torch.no_grad():
            z_tensor = self._to_tensor(z_crop)
            self._template_feat = self._backbone(z_tensor)  # (1, C, hz, wz)

    def update(self, frame: np.ndarray) -> BBox:
        """Predict the target location in the next frame.

        Crops a search region around the last known position, runs both
        branches through the backbone, computes the cross-correlation response
        map, and converts the peak back to image coordinates.

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 array.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._template_feat is None or self._bbox is None:
            raise RuntimeError(
                "SiamFCLiteTracker is not initialised. Call initialize() first."
            )

        cx, cy, w, h = self._bbox

        x_crop = self._crop_patch(frame, cx, cy, w, h, self.search_size)
        with torch.no_grad():
            x_tensor = self._to_tensor(x_crop)
            x_feat = self._backbone(x_tensor)  # (1, C, hx, wx)
            response = self._cross_correlate(self._template_feat, x_feat)

        # Locate argmax on the score map
        resp_np = response.squeeze().cpu().numpy()  # (H_resp, W_resp)
        resp_h, resp_w = resp_np.shape
        peak_y, peak_x = np.unravel_index(resp_np.argmax(), resp_np.shape)

        # Displacement in search-crop pixel space (origin = centre of map)
        disp_x = (float(peak_x) - resp_w / 2.0) * self.stride
        disp_y = (float(peak_y) - resp_h / 2.0) * self.stride

        # Scale from search-crop space to image space
        scale = self._context_scale(w, h, self.search_size)
        new_cx = cx + disp_x * scale
        new_cy = cy + disp_y * scale

        self._bbox = [new_cx, new_cy, w, h]
        return (new_cx - w / 2.0, new_cy - h / 2.0, w, h)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _context_scale(self, w: float, h: float, crop_size: int) -> float:
        """Ratio of image pixels per search-crop pixel.

        Derived from the context-padded crop size used for the search region.
        """
        context = self.context_amount * (w + h)
        sz = float(np.sqrt((w + context) * (h + context)))
        return sz / crop_size

    def _crop_patch(
        self,
        frame: np.ndarray,
        cx: float,
        cy: float,
        w: float,
        h: float,
        out_size: int,
    ) -> np.ndarray:
        """Crop a context-padded square patch and resize to ``out_size × out_size``.

        Extends beyond frame edges using the mean pixel value as padding,
        matching the original SiamFC approach and avoiding edge artefacts.

        Args:
            frame:    BGR source image.
            cx, cy:  Target centre in image coordinates.
            w, h:    Target width and height.
            out_size: Output patch side length in pixels.

        Returns:
            RGB patch of shape ``(out_size, out_size, 3)`` as uint8.
        """
        context = self.context_amount * (w + h)
        half = float(np.sqrt((w + context) * (h + context))) / 2.0

        fh, fw = frame.shape[:2]
        x1 = int(round(cx - half))
        y1 = int(round(cy - half))
        x2 = int(round(cx + half))
        y2 = int(round(cy + half))

        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - fw)
        pad_b = max(0, y2 - fh)

        if pad_l or pad_t or pad_r or pad_b:
            mean_px = frame.mean(axis=(0, 1)).astype(np.uint8)
            padded = np.full(
                (y2 - y1, x2 - x1, frame.shape[2]), mean_px, dtype=np.uint8
            )
            src_x1, src_y1 = max(0, x1), max(0, y1)
            src_x2, src_y2 = min(fw, x2), min(fh, y2)
            dst_x1, dst_y1 = pad_l, pad_t
            dst_x2 = dst_x1 + (src_x2 - src_x1)
            dst_y2 = dst_y1 + (src_y2 - src_y1)
            padded[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]
            crop = padded
        else:
            crop = frame[y1:y2, x1:x2]

        # Convert BGR → RGB then resize
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (out_size, out_size), interpolation=cv2.INTER_LINEAR)

    def _to_tensor(self, patch: np.ndarray) -> torch.Tensor:
        """Convert a ``(H, W, 3)`` uint8 RGB patch to a ``(1, 3, H, W)`` float tensor."""
        t = torch.from_numpy(patch).permute(2, 0, 1).float().div(255.0)
        return t.unsqueeze(0).to(self.device)

    @staticmethod
    def _cross_correlate(
        z_feat: torch.Tensor,
        x_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Depth-wise cross-correlation: response = corr(z_feat, x_feat).

        Implements the SiamFC scoring layer as a grouped convolution so that
        each channel of ``z_feat`` acts as a separate 2-D filter applied to
        the corresponding channel of ``x_feat``.  Per-channel responses are
        summed to produce a single score map.

        Args:
            z_feat: Template features ``(1, C, hz, wz)``.
            x_feat: Search features  ``(1, C, hx, wx)`` where hx > hz.

        Returns:
            Score map ``(1, 1, hx-hz+1, wx-wz+1)``.
        """
        b, c, hz, wz = z_feat.shape
        kernel = z_feat.reshape(c, 1, hz, wz)           # (C, 1, hz, wz)
        score = F.conv2d(x_feat, kernel, groups=c)       # (1, C, H_resp, W_resp)
        return score.sum(dim=1, keepdim=True)             # (1, 1, H_resp, W_resp)

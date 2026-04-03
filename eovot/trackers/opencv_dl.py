"""OpenCV deep-learning tracker wrappers for EOVOT.

Provides ``BaseTracker``-compatible wrappers for the DL-based trackers
bundled with OpenCV:

* **DaSiamRPNTracker** — DaSiamRPN (Siamese RPN + distractor-aware training).
* **NanoTracker** — lightweight backbone+neckhead network, optimised for
  edge devices.

Both trackers require pre-trained ONNX model files that are **not**
bundled with OpenCV.  Download links are provided in each class docstring.

References
----------
Zhu, Z., Wang, Q., Li, B., Wu, W., Yan, J., & Hu, W. (2018).
Distractor-aware Siamese Networks for Visual Object Tracking.
ECCV 2018.

"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .base import BaseTracker, BBox


class DaSiamRPNTracker(BaseTracker):
    """DaSiamRPN tracker — Siamese RPN with distractor-aware training.

    Significantly more accurate than classical trackers (MOSSE, KCF, MIL)
    on long-term sequences and challenging scenarios (e.g. distractors,
    fast motion, large appearance change).

    Requires three ONNX model files (≈20 MB total).  Download from the
    OpenCV Zoo::

        # Model download (one-time)
        wget -P models/ \\
          https://storage.openvinotoolkit.org/repositories/open_model_zoo/\\
          public/dasiamrpn-vot/dasiamrpn_model.onnx \\
          https://storage.openvinotoolkit.org/repositories/open_model_zoo/\\
          public/dasiamrpn-vot/dasiamrpn_kernel_r1.onnx \\
          https://storage.openvinotoolkit.org/repositories/open_model_zoo/\\
          public/dasiamrpn-vot/dasiamrpn_kernel_cls1.onnx

    Args:
        model:       Path to ``dasiamrpn_model.onnx``.
        kernel_r1:   Path to ``dasiamrpn_kernel_r1.onnx``.
        kernel_cls1: Path to ``dasiamrpn_kernel_cls1.onnx``.
        name:        Human-readable identifier in benchmark reports.
                     Default: ``"DaSiamRPN"``.

    Raises:
        FileNotFoundError: If any model file does not exist.
        RuntimeError:      If OpenCV was built without DaSiamRPN support.

    Example::

        tracker = DaSiamRPNTracker(
            model="models/dasiamrpn_model.onnx",
            kernel_r1="models/dasiamrpn_kernel_r1.onnx",
            kernel_cls1="models/dasiamrpn_kernel_cls1.onnx",
        )
        tracker.initialize(first_frame, init_bbox)
        for frame in sequence:
            pred = tracker.update(frame)
    """

    def __init__(
        self,
        model: str,
        kernel_r1: str,
        kernel_cls1: str,
        name: str = "DaSiamRPN",
    ) -> None:
        super().__init__(name)
        for path, label in [(model, "model"), (kernel_r1, "kernel_r1"), (kernel_cls1, "kernel_cls1")]:
            if not Path(path).is_file():
                raise FileNotFoundError(
                    f"DaSiamRPN {label} file not found: {path}\n"
                    "Download from the OpenCV Zoo — see class docstring."
                )
        if not hasattr(cv2, "TrackerDaSiamRPN_create"):
            raise RuntimeError(
                "DaSiamRPN tracker is not available in this OpenCV build. "
                "Install opencv-python >= 4.5 or opencv-contrib-python."
            )
        params = cv2.TrackerDaSiamRPN_Params()
        params.model = model
        params.kernel_r1 = kernel_r1
        params.kernel_cls1 = kernel_cls1
        self._tracker: cv2.TrackerDaSiamRPN = cv2.TrackerDaSiamRPN_create(params)
        self._last_bbox: BBox = (0.0, 0.0, 1.0, 1.0)

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        x, y, w, h = (max(0, int(v)) for v in bbox)
        self._last_bbox = (float(x), float(y), float(w), float(h))
        self._tracker.init(frame, (x, y, w, h))

    def update(self, frame: np.ndarray) -> BBox:
        ok, bbox = self._tracker.update(frame)
        if ok:
            self._last_bbox = tuple(float(v) for v in bbox)  # type: ignore[assignment]
        return self._last_bbox


class NanoTracker(BaseTracker):
    """NanoTracker — ultra-lightweight deep tracker for edge deployment.

    NanoTracker uses a compact backbone + neck-head architecture optimised
    for low-latency inference.  It offers a middle ground between classical
    trackers (high FPS, moderate accuracy) and heavier DL trackers (lower
    FPS, high accuracy).

    Requires two ONNX model files (≈5 MB total)::

        # Model download (one-time)
        wget -P models/ \\
          https://github.com/HonglinChu/NanoTrack/raw/master/ncnn_models/\\
          nanotrack_backbone_sim.onnx \\
          https://github.com/HonglinChu/NanoTrack/raw/master/ncnn_models/\\
          nanotrack_head_sim.onnx

    Args:
        backbone:  Path to backbone ONNX model file.
        neckhead:  Path to neck+head ONNX model file.
        name:      Human-readable identifier in benchmark reports.
                   Default: ``"NanoTrack"``.

    Raises:
        FileNotFoundError: If any model file does not exist.
        RuntimeError:      If OpenCV was built without NanoTracker support.

    Example::

        tracker = NanoTracker(
            backbone="models/nanotrack_backbone_sim.onnx",
            neckhead="models/nanotrack_head_sim.onnx",
        )
        tracker.initialize(first_frame, init_bbox)
        for frame in sequence:
            pred = tracker.update(frame)
    """

    def __init__(
        self,
        backbone: str,
        neckhead: str,
        name: str = "NanoTrack",
    ) -> None:
        super().__init__(name)
        for path, label in [(backbone, "backbone"), (neckhead, "neckhead")]:
            if not Path(path).is_file():
                raise FileNotFoundError(
                    f"NanoTracker {label} file not found: {path}\n"
                    "Download from github.com/HonglinChu/NanoTrack — see class docstring."
                )
        if not hasattr(cv2, "TrackerNano_create"):
            raise RuntimeError(
                "NanoTracker is not available in this OpenCV build. "
                "Install opencv-python >= 4.6 or opencv-contrib-python."
            )
        params = cv2.TrackerNano_Params()
        params.backbone = backbone
        params.neckhead = neckhead
        self._tracker: cv2.TrackerNano = cv2.TrackerNano_create(params)
        self._last_bbox: BBox = (0.0, 0.0, 1.0, 1.0)

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        x, y, w, h = (max(0, int(v)) for v in bbox)
        self._last_bbox = (float(x), float(y), float(w), float(h))
        self._tracker.init(frame, (x, y, w, h))

    def update(self, frame: np.ndarray) -> BBox:
        ok, bbox = self._tracker.update(frame)
        if ok:
            self._last_bbox = tuple(float(v) for v in bbox)  # type: ignore[assignment]
        return self._last_bbox

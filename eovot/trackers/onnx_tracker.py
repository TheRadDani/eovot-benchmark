"""ONNX Runtime tracker wrapper for inference-optimized deep learning models.

Bridges any tracking model exported to ONNX format with the EOVOT
:class:`~eovot.trackers.base.BaseTracker` interface.  Because ONNX tracking
models differ widely in their I/O layout (SiamFC, SiamRPN, STARK, OSTrack
all have different input names and output shapes), this wrapper is intentionally
model-agnostic: the caller supplies lightweight callback functions that handle
preprocessing and postprocessing while this class owns the session lifecycle
and state threading.

Design
------
Two callbacks shape the tracker's behaviour:

``init_fn(session, frame, bbox) -> state``
    Called once on the first frame.  Should extract template features or
    encode whatever the model needs as persistent state.  Returns an arbitrary
    object (dict, namedtuple, …) stored internally as ``self._state``.

``update_fn(session, frame, state) -> (bbox, new_state)``
    Called on every subsequent frame.  Receives the current session and the
    state produced by the previous call.  Returns the predicted bounding box
    and the updated state.

The ONNX :class:`onnxruntime.InferenceSession` is created once at construction
time and shared across all frames (and sequences when the tracker is reused).
Calling :meth:`initialize` resets the per-sequence state.

Hardware acceleration
---------------------
Pass ``providers`` to select execution providers in priority order:

* ``["CPUExecutionProvider"]`` — default, universally available
* ``["CUDAExecutionProvider", "CPUExecutionProvider"]`` — GPU with CPU fallback
* ``["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]``

Example
-------
::

    import cv2
    import numpy as np
    from eovot.trackers.onnx_tracker import OnnxTracker

    def init_fn(session, frame, bbox):
        x, y, w, h = bbox
        patch = frame[int(y):int(y+h), int(x):int(x+w)]
        template = cv2.resize(patch, (127, 127)).astype(np.float32)[None] / 255.0
        return {"template": template, "last_bbox": bbox}

    def update_fn(session, frame, state):
        inputs = {
            "template": state["template"],
            "search":   preprocess_search(frame, state["last_bbox"]),
        }
        response, = session.run(None, inputs)
        new_bbox = decode_response(response, state["last_bbox"])
        return new_bbox, {**state, "last_bbox": new_bbox}

    tracker = OnnxTracker(
        model_path="siamfc_r22.onnx",
        init_fn=init_fn,
        update_fn=update_fn,
    )

References
----------
* ONNX Runtime: https://onnxruntime.ai
* SiamFC: Bertinetto et al., ECCVW 2016
* SiamRPN: Li et al., CVPR 2018
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from .base import BaseTracker, BBox

# Type aliases for callback signatures
InitFn = Callable[["ort.InferenceSession", np.ndarray, BBox], Any]  # noqa: F821
UpdateFn = Callable[["ort.InferenceSession", np.ndarray, Any], Tuple[BBox, Any]]  # noqa: F821


def _load_onnxruntime() -> Any:
    """Import onnxruntime with a clear error message if unavailable."""
    try:
        import onnxruntime as ort
        return ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for OnnxTracker but is not installed.\n"
            "Install it with one of:\n"
            "  pip install onnxruntime          # CPU-only\n"
            "  pip install onnxruntime-gpu      # CUDA GPU support\n"
        ) from exc


class OnnxTracker(BaseTracker):
    """Model-agnostic ONNX Runtime tracker wrapper.

    Loads any tracking model exported to ONNX format and wraps it in the
    :class:`~eovot.trackers.base.BaseTracker` interface.  State management
    (template features, search region context, etc.) is delegated to
    caller-supplied callback functions, keeping this class free of
    model-specific assumptions.

    Args:
        model_path: Path to the ``.onnx`` model file.
        init_fn:    Callback called at :meth:`initialize` time.
                    Signature: ``(session, frame, bbox) -> state``.
                    The returned ``state`` is stored and forwarded to
                    ``update_fn`` at each subsequent frame.
        update_fn:  Callback called at :meth:`update` time.
                    Signature: ``(session, frame, state) -> (bbox, new_state)``.
                    Must return the predicted bounding box and the updated state.
        name:       Human-readable tracker name used in reports.
                    Defaults to the model file's stem (e.g. ``"siamfc_r22"``
                    for ``"siamfc_r22.onnx"``).
        providers:  ONNX Runtime execution providers in priority order.
                    Defaults to ``["CPUExecutionProvider"]``.
                    Example for GPU: ``["CUDAExecutionProvider",
                    "CPUExecutionProvider"]``.

    Raises:
        ImportError: If ``onnxruntime`` is not installed (raised at
            construction time).
        FileNotFoundError: If ``model_path`` does not exist.

    Example::

        tracker = OnnxTracker(
            model_path="model.onnx",
            init_fn=my_init,
            update_fn=my_update,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        tracker.initialize(first_frame, (x, y, w, h))
        for frame in sequence[1:]:
            bbox = tracker.update(frame)
    """

    def __init__(
        self,
        model_path: str,
        init_fn: InitFn,
        update_fn: UpdateFn,
        name: Optional[str] = None,
        providers: Optional[List[str]] = None,
    ) -> None:
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        tracker_name = name if name is not None else model_path_obj.stem
        super().__init__(name=tracker_name)

        self._init_fn = init_fn
        self._update_fn = update_fn
        self._providers = providers or ["CPUExecutionProvider"]
        self._state: Any = None

        ort = _load_onnxruntime()
        self._session: Any = ort.InferenceSession(
            str(model_path_obj),
            providers=self._providers,
        )

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the tracker on the first frame of a new sequence.

        Calls ``init_fn(session, frame, bbox)`` and stores the returned state.
        Any state from a previous sequence is discarded, allowing the same
        tracker instance to be reused across multiple sequences.

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 numpy array.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        self._state = self._init_fn(self._session, frame, bbox)

    def update(self, frame: np.ndarray) -> BBox:
        """Predict the target location in the current frame.

        Calls ``update_fn(session, frame, state)`` with the session and the
        state stored by the previous :meth:`initialize` or :meth:`update` call.

        Args:
            frame: BGR image as a ``(H, W, 3)`` uint8 numpy array.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If :meth:`initialize` has not been called yet.
        """
        if self._state is None:
            raise RuntimeError(
                f"{self.__class__.__name__} is not initialised. "
                "Call initialize() before update()."
            )
        bbox, self._state = self._update_fn(self._session, frame, self._state)
        return bbox

    def reset(self) -> None:
        """Clear the per-sequence state.

        After calling this method the tracker can be re-initialised on a new
        sequence by calling :meth:`initialize` again.  The ONNX session
        remains loaded, so subsequent :meth:`initialize` calls do not reload
        the model from disk.
        """
        self._state = None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def input_names(self) -> List[str]:
        """Names of the ONNX model's input nodes."""
        return [inp.name for inp in self._session.get_inputs()]

    @property
    def output_names(self) -> List[str]:
        """Names of the ONNX model's output nodes."""
        return [out.name for out in self._session.get_outputs()]

    @property
    def active_providers(self) -> List[str]:
        """Execution providers actually used by the session (may differ from requested)."""
        return self._session.get_providers()

    def __repr__(self) -> str:
        return (
            f"OnnxTracker(name={self.name!r}, "
            f"providers={self.active_providers}, "
            f"inputs={self.input_names})"
        )

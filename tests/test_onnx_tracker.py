"""Unit tests for eovot.trackers.onnx_tracker.OnnxTracker.

All tests mock onnxruntime.InferenceSession so no real ONNX model or
GPU hardware is required.  The test suite verifies:

- Construction: session creation, name defaulting, FileNotFoundError guard
- BaseTracker protocol: initialize() / update() contract
- State threading: init_fn return value forwarded to update_fn
- Return value: update() returns the bbox from update_fn
- RuntimeError guard: update() before initialize()
- reset(): clears state so the tracker can be re-used
- Provider plumbing: providers kwarg forwarded to InferenceSession
- ImportError: clear message when onnxruntime is absent
- Introspection properties: input_names, output_names, active_providers
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_MODEL_PATH = "/tmp/fake_model.onnx"


def _make_fake_model(tmp_path: Path) -> Path:
    """Create a zero-byte file that satisfies OnnxTracker's existence check."""
    p = tmp_path / "model.onnx"
    p.write_bytes(b"")
    return p


def _simple_init_fn(session, frame: np.ndarray, bbox) -> Dict:
    """Minimal init callback: stores initial bbox as state."""
    return {"bbox": bbox, "call_count": 1}


def _simple_update_fn(session, frame: np.ndarray, state: Dict) -> Tuple[tuple, Dict]:
    """Minimal update callback: returns the stored bbox unchanged."""
    new_state = {**state, "call_count": state["call_count"] + 1}
    return state["bbox"], new_state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_ort():
    """Patch onnxruntime with a MagicMock for the full test."""
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [
        MagicMock(name="template"),
        MagicMock(name="search"),
    ]
    mock_session.get_outputs.return_value = [MagicMock(name="response")]
    mock_session.get_providers.return_value = ["CPUExecutionProvider"]

    mock_module = MagicMock()
    mock_module.InferenceSession.return_value = mock_session

    with patch.dict(sys.modules, {"onnxruntime": mock_module}):
        yield mock_module, mock_session


@pytest.fixture()
def tracker(tmp_path, mock_ort):
    """Build a ready-to-use OnnxTracker backed by the mock session."""
    from eovot.trackers.onnx_tracker import OnnxTracker
    model_path = _make_fake_model(tmp_path)
    return OnnxTracker(
        model_path=str(model_path),
        init_fn=_simple_init_fn,
        update_fn=_simple_update_fn,
    )


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_session_created_with_model_path(self, tmp_path, mock_ort):
        from eovot.trackers.onnx_tracker import OnnxTracker
        mock_module, mock_session = mock_ort
        model_path = _make_fake_model(tmp_path)

        OnnxTracker(str(model_path), _simple_init_fn, _simple_update_fn)

        mock_module.InferenceSession.assert_called_once()
        args, kwargs = mock_module.InferenceSession.call_args
        assert str(model_path) in args or str(model_path) == kwargs.get("path_or_bytes")

    def test_name_defaults_to_model_stem(self, tmp_path, mock_ort):
        from eovot.trackers.onnx_tracker import OnnxTracker
        model_path = _make_fake_model(tmp_path)

        tracker = OnnxTracker(str(model_path), _simple_init_fn, _simple_update_fn)

        assert tracker.name == "model"

    def test_explicit_name_overrides_stem(self, tmp_path, mock_ort):
        from eovot.trackers.onnx_tracker import OnnxTracker
        model_path = _make_fake_model(tmp_path)

        tracker = OnnxTracker(
            str(model_path), _simple_init_fn, _simple_update_fn, name="SiamFC"
        )

        assert tracker.name == "SiamFC"

    def test_file_not_found_raises(self, mock_ort):
        from eovot.trackers.onnx_tracker import OnnxTracker

        with pytest.raises(FileNotFoundError, match="ONNX model not found"):
            OnnxTracker(
                "/nonexistent/path/model.onnx",
                _simple_init_fn,
                _simple_update_fn,
            )

    def test_providers_forwarded_to_session(self, tmp_path, mock_ort):
        from eovot.trackers.onnx_tracker import OnnxTracker
        mock_module, _ = mock_ort
        model_path = _make_fake_model(tmp_path)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        OnnxTracker(
            str(model_path), _simple_init_fn, _simple_update_fn, providers=providers
        )

        _, kwargs = mock_module.InferenceSession.call_args
        assert kwargs.get("providers") == providers

    def test_default_provider_is_cpu(self, tmp_path, mock_ort):
        from eovot.trackers.onnx_tracker import OnnxTracker
        mock_module, _ = mock_ort
        model_path = _make_fake_model(tmp_path)

        OnnxTracker(str(model_path), _simple_init_fn, _simple_update_fn)

        _, kwargs = mock_module.InferenceSession.call_args
        assert "CPUExecutionProvider" in kwargs.get("providers", [])


# ---------------------------------------------------------------------------
# BaseTracker protocol tests
# ---------------------------------------------------------------------------

class TestTrackerProtocol:
    def test_initialize_calls_init_fn(self, tracker, mock_ort):
        _, mock_session = mock_ort
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        bbox = (10.0, 20.0, 50.0, 40.0)

        tracker.initialize(frame, bbox)

        # State should be populated after initialize
        assert tracker._state is not None

    def test_init_fn_receives_session_frame_bbox(self, tmp_path, mock_ort):
        from eovot.trackers.onnx_tracker import OnnxTracker
        _, mock_session = mock_ort
        model_path = _make_fake_model(tmp_path)

        received: list = []

        def recording_init(session, frame, bbox):
            received.append((session, frame, bbox))
            return {"bbox": bbox}

        tracker = OnnxTracker(str(model_path), recording_init, _simple_update_fn)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        bbox = (5.0, 10.0, 30.0, 20.0)

        tracker.initialize(frame, bbox)

        assert len(received) == 1
        sess, f, b = received[0]
        assert sess is mock_session
        np.testing.assert_array_equal(f, frame)
        assert b == bbox

    def test_update_returns_bbox_from_update_fn(self, tracker):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        init_bbox = (10.0, 20.0, 50.0, 40.0)
        tracker.initialize(frame, init_bbox)

        result = tracker.update(frame)

        assert result == init_bbox

    def test_update_fn_receives_session_frame_state(self, tmp_path, mock_ort):
        from eovot.trackers.onnx_tracker import OnnxTracker
        _, mock_session = mock_ort
        model_path = _make_fake_model(tmp_path)

        update_received: list = []

        def recording_update(session, frame, state):
            update_received.append((session, frame, state))
            return state["bbox"], state

        tracker = OnnxTracker(str(model_path), _simple_init_fn, recording_update)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        bbox = (1.0, 2.0, 10.0, 10.0)
        tracker.initialize(frame, bbox)
        tracker.update(frame)

        assert len(update_received) == 1
        sess, f, state = update_received[0]
        assert sess is mock_session
        np.testing.assert_array_equal(f, frame)
        assert state["bbox"] == bbox

    def test_state_is_threaded_between_updates(self, tmp_path, mock_ort):
        from eovot.trackers.onnx_tracker import OnnxTracker
        model_path = _make_fake_model(tmp_path)

        call_log: list = []

        def init_fn(session, frame, bbox):
            return {"step": 0, "bbox": bbox}

        def update_fn(session, frame, state):
            new_step = state["step"] + 1
            call_log.append(new_step)
            new_state = {"step": new_step, "bbox": state["bbox"]}
            return state["bbox"], new_state

        tracker = OnnxTracker(str(model_path), init_fn, update_fn)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        tracker.initialize(frame, (0, 0, 10, 10))

        for _ in range(3):
            tracker.update(frame)

        assert call_log == [1, 2, 3], "State counter not threaded correctly"


# ---------------------------------------------------------------------------
# Guard / error tests
# ---------------------------------------------------------------------------

class TestGuards:
    def test_update_before_initialize_raises(self, tracker):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        with pytest.raises(RuntimeError, match="not initialised"):
            tracker.update(frame)

    def test_reset_clears_state(self, tracker):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        tracker.initialize(frame, (0.0, 0.0, 10.0, 10.0))
        assert tracker._state is not None

        tracker.reset()

        assert tracker._state is None

    def test_update_after_reset_raises(self, tracker):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        tracker.initialize(frame, (0.0, 0.0, 10.0, 10.0))
        tracker.reset()

        with pytest.raises(RuntimeError, match="not initialised"):
            tracker.update(frame)

    def test_reinitialize_after_reset_works(self, tracker):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        tracker.initialize(frame, (0.0, 0.0, 10.0, 10.0))
        tracker.reset()
        tracker.initialize(frame, (5.0, 5.0, 20.0, 20.0))

        bbox = tracker.update(frame)
        assert bbox == (5.0, 5.0, 20.0, 20.0)

    def test_missing_onnxruntime_raises_import_error(self, tmp_path):
        model_path = _make_fake_model(tmp_path)

        # Simulate onnxruntime not being installed
        with patch.dict(sys.modules, {"onnxruntime": None}):
            # Re-import so the module-level guard triggers
            if "eovot.trackers.onnx_tracker" in sys.modules:
                del sys.modules["eovot.trackers.onnx_tracker"]
            from eovot.trackers.onnx_tracker import OnnxTracker

            with pytest.raises(ImportError, match="onnxruntime"):
                OnnxTracker(
                    str(model_path), _simple_init_fn, _simple_update_fn
                )


# ---------------------------------------------------------------------------
# Introspection tests
# ---------------------------------------------------------------------------

class TestIntrospection:
    def test_input_names_delegated_to_session(self, tracker, mock_ort):
        _, mock_session = mock_ort
        mock_session.get_inputs.return_value = [
            MagicMock(name="template"),
            MagicMock(name="search"),
        ]
        names = tracker.input_names
        assert "template" in names
        assert "search" in names

    def test_output_names_delegated_to_session(self, tracker, mock_ort):
        _, mock_session = mock_ort
        mock_session.get_outputs.return_value = [MagicMock(name="response")]
        names = tracker.output_names
        assert "response" in names

    def test_active_providers_delegated_to_session(self, tracker, mock_ort):
        _, mock_session = mock_ort
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        assert "CPUExecutionProvider" in tracker.active_providers

    def test_repr_contains_name_and_provider(self, tracker):
        r = repr(tracker)
        assert "OnnxTracker" in r
        assert tracker.name in r

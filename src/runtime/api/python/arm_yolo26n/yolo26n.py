"""Low-level tensor access to the native YOLO26n runtime."""

from __future__ import annotations

import ctypes
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np
from numpy.typing import NDArray

from ._model import _load_model
from ._runtime import _runtime_path
from .errors import Yolo26nError

__all__ = ["Yolo26nTensorSession"]

# The fixed topology is deliberately partitioned into nineteen XNNPACK
# subgraphs so the native schedule can run custom work between graph segments.
_EXPECTED_GRAPH_COUNT = 19
_NATIVE_STATUS_OK = 0
type _FloatArray = NDArray[np.float32]
type _Uint8Array = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class _PixelFormat:
    """Python view of one y26_native_pixel_format value."""

    native_value: int
    channels: int


# Native values must remain synchronized with y26_native_pixel_format in
# arm_yolo26n.h. Channel counts validate the interleaved source array.
_PIXEL_FORMATS = {
    "BGR": _PixelFormat(native_value=0, channels=3),
    "RGB": _PixelFormat(native_value=1, channels=3),
    "BGRX": _PixelFormat(native_value=2, channels=4),
}


def _elements(shape: tuple[int, ...]) -> int:
    result = 1
    for extent in shape:
        result *= extent
    return result


@dataclass(frozen=True, slots=True)
class _NativeModelContract:
    """Model facts reported by a newly created native handle."""

    graph_count: int
    resolution: int
    class_count: int
    input_bytes: int
    output_bytes: int
    calls: int


class _NativeLibrary:
    """Bind the YOLO26n C API and translate calls to Python values."""

    def __init__(self, path: Path) -> None:
        self._raw = ctypes.CDLL(str(path), mode=os.RTLD_NOW | os.RTLD_LOCAL)
        self._raw.y26_native_create.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self._raw.y26_native_create.restype = ctypes.c_void_p
        self._raw.y26_native_execute.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self._raw.y26_native_execute.restype = ctypes.c_int
        self._raw.y26_native_preprocess_image_no_resize.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self._raw.y26_native_preprocess_image_no_resize.restype = ctypes.c_int
        self._raw.y26_native_destroy.argtypes = [ctypes.c_void_p]
        self._raw.y26_native_destroy.restype = None
        self._raw.y26_native_calls.argtypes = [ctypes.c_void_p]
        self._raw.y26_native_calls.restype = ctypes.c_uint64
        self._raw.y26_native_graphs.argtypes = [ctypes.c_void_p]
        self._raw.y26_native_graphs.restype = ctypes.c_uint32
        for name in ("y26_native_resolution", "y26_native_class_count"):
            function = getattr(self._raw, name)
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_uint32
        for name in ("y26_native_input_bytes", "y26_native_output_bytes"):
            function = getattr(self._raw, name)
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_size_t

    def create(self, model: bytes) -> ctypes.c_void_p:
        model_pointer = ctypes.cast(ctypes.c_char_p(model), ctypes.c_void_p)
        return ctypes.c_void_p(self._raw.y26_native_create(model_pointer, len(model)))

    def model_contract(self, handle: ctypes.c_void_p) -> _NativeModelContract:
        return _NativeModelContract(
            graph_count=int(self._raw.y26_native_graphs(handle)),
            resolution=int(self._raw.y26_native_resolution(handle)),
            class_count=int(self._raw.y26_native_class_count(handle)),
            input_bytes=int(self._raw.y26_native_input_bytes(handle)),
            output_bytes=int(self._raw.y26_native_output_bytes(handle)),
            calls=int(self._raw.y26_native_calls(handle)),
        )

    def execute(
        self,
        handle: ctypes.c_void_p,
        model_input: _FloatArray,
        output: _FloatArray,
    ) -> int:
        return int(
            self._raw.y26_native_execute(
                handle,
                model_input.ctypes.data,
                model_input.nbytes,
                output.ctypes.data,
                output.nbytes,
            )
        )

    def preprocess_image_no_resize(
        self,
        frame: _Uint8Array,
        pixel_format: int,
        output: _FloatArray,
    ) -> int:
        """Invoke native preprocessing while preserving padded source rows."""
        return int(
            self._raw.y26_native_preprocess_image_no_resize(
                frame.ctypes.data,
                _image_span_bytes(frame),
                frame.shape[0],
                frame.shape[1],
                frame.strides[0],
                pixel_format,
                output.ctypes.data,
                output.nbytes,
            )
        )

    def destroy(self, handle: ctypes.c_void_p) -> None:
        self._raw.y26_native_destroy(handle)


def _image_span_bytes(frame: _Uint8Array) -> int:
    """Return bytes through the final pixel, including padding between rows."""
    final_row_offset = (frame.shape[0] - 1) * frame.strides[0]
    visible_row_bytes = frame.shape[1] * frame.shape[2]
    return int(final_row_offset + visible_row_bytes)


class Yolo26nTensorSession:
    """Execute raw tensors with the native runtime and converted model.

    This is the low-level Python interface. Use it when the caller already
    owns preprocessing and postprocessing and needs direct FP32 NCHW tensor
    execution. `Yolo26Detector` is the image-oriented API.

    A session owns one native model handle and its worker resources. Calls are
    deliberately non-reentrant; create independent sessions in separate
    processes when concurrent isolation is required. Prefer using the session
    as a context manager so the native resources are released deterministically.
    Release wheels provide the runtime library. The optional ``runtime``
    override is for source builds and runtime development.
    """

    def __init__(
        self,
        *,
        model: str | Path,
        runtime: str | Path | None = None,
    ) -> None:
        runtime_path = _runtime_path(runtime)
        model_data = _load_model(model)
        self._runtime_path = runtime_path
        self.input_shape = model_data.input_shape
        self.output_shape = model_data.output_shape
        self.class_names = model_data.class_names
        self._lifecycle_lock = threading.Lock()
        self._handle = ctypes.c_void_p()
        try:
            self._native = _NativeLibrary(runtime_path)
            self._handle = self._native.create(model_data.model_bytes)
            if not self._handle.value:
                raise Yolo26nError("YOLO26n runtime rejected its exported model")
            expected_input_bytes = 4 * _elements(self.input_shape)
            expected_output_bytes = 4 * _elements(self.output_shape)
            native_contract = self._native.model_contract(self._handle)
            if (
                native_contract.graph_count != _EXPECTED_GRAPH_COUNT
                or native_contract.resolution != self.input_shape[2]
                or native_contract.class_count != len(self.class_names)
                or native_contract.input_bytes != expected_input_bytes
                or native_contract.output_bytes != expected_output_bytes
            ):
                raise Yolo26nError("YOLO26n runtime and model contracts disagree")
            if native_contract.calls != 0:
                raise Yolo26nError("YOLO26n runtime call counter was not initialized")
        except BaseException:
            self.close()
            raise

    @property
    def closed(self) -> bool:
        return not bool(self._handle.value)

    @property
    def runtime_path(self) -> Path:
        """Return the runtime library selected for this session."""
        return self._runtime_path

    def execute(self, tensor: _FloatArray) -> _FloatArray:
        """Execute one FP32 NCHW tensor and return the raw YOLO26n prediction.

        The input must match `input_shape`. A non-contiguous array is copied
        into contiguous storage before entering native code. The returned FP32
        array always has `output_shape`.
        """
        if self.closed:
            raise Yolo26nError("YOLO26n runtime is closed")
        if not isinstance(tensor, np.ndarray):
            raise TypeError("input must be a numpy.ndarray")
        if tensor.dtype != np.float32:
            raise TypeError("input array must have dtype float32")
        if tuple(tensor.shape) != self.input_shape:
            raise ValueError(f"input shape {tuple(tensor.shape)} does not match {self.input_shape}")
        tensor = np.ascontiguousarray(tensor)
        if not self._lifecycle_lock.acquire(blocking=False):
            raise Yolo26nError("concurrent execution or close is not supported")
        try:
            if self.closed:
                raise Yolo26nError("YOLO26n runtime is closed")
            output = np.empty(self.output_shape, dtype=np.float32)
            status = self._native.execute(self._handle, tensor, output)
            if status != _NATIVE_STATUS_OK:
                raise Yolo26nError(f"YOLO26n runtime execution failed with status {status}")
            return output
        finally:
            self._lifecycle_lock.release()

    def _prepare_image_without_resize(
        self,
        frame: _Uint8Array,
        *,
        pixel_format: str,
        output: _FloatArray,
    ) -> None:
        """Fill a model-input tensor from an image that needs padding but no resize.

        The detector uses this package-internal fast path after determining
        that the source image's longest side already equals the model
        resolution. Native code fuses centered value-114 padding, BGR/BGRX to
        RGB conversion, HWC-to-NCHW conversion, and FP32 normalization. Callers
        of the public tensor API should prepare their own input instead.
        """
        if self.closed:
            raise Yolo26nError("YOLO26n runtime is closed")
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            raise TypeError("frame must be a uint8 numpy.ndarray")
        try:
            format_contract = _PIXEL_FORMATS[pixel_format]
        except KeyError as error:
            raise ValueError("pixel_format must be 'BGR', 'RGB', or 'BGRX'") from error
        if frame.ndim != 3 or frame.shape[2] != format_contract.channels:
            raise ValueError(
                f"{pixel_format} frame must have shape [height, width, {format_contract.channels}]"
            )
        if frame.strides[1:] != (format_contract.channels, 1) or frame.strides[0] <= 0:
            raise ValueError("frame must have interleaved, forward uint8 pixels")
        if (
            not isinstance(output, np.ndarray)
            or output.dtype != np.float32
            or tuple(output.shape) != self.input_shape
            or not output.flags.c_contiguous
        ):
            raise ValueError(
                f"output must be a contiguous FP32 array with shape {self.input_shape}"
            )
        if not self._lifecycle_lock.acquire(blocking=False):
            raise Yolo26nError("concurrent execution or close is not supported")
        try:
            if self.closed:
                raise Yolo26nError("YOLO26n runtime is closed")
            status = self._native.preprocess_image_no_resize(
                frame,
                format_contract.native_value,
                output,
            )
            if status != _NATIVE_STATUS_OK:
                raise Yolo26nError(f"YOLO26n runtime preprocessing failed with status {status}")
        finally:
            self._lifecycle_lock.release()

    def close(self) -> None:
        """Release the native model handle; repeated calls are safe."""
        lock = getattr(self, "_lifecycle_lock", None)
        if lock is None:
            return
        with lock:
            handle = getattr(self, "_handle", None)
            native = getattr(self, "_native", None)
            if handle is not None and handle.value and native is not None:
                native.destroy(handle)
                self._handle = ctypes.c_void_p()

    def __enter__(self) -> Self:
        if self.closed:
            raise Yolo26nError("YOLO26n runtime is closed")
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

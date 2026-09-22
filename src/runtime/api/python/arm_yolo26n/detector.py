"""High-level YOLO26n detection for uint8 NumPy image frames."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import TracebackType
from typing import Literal, Protocol, Self, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .errors import Yolo26nError
from .yolo26n import Yolo26nTensorSession

__all__ = [
    "Detection",
    "DetectionResult",
    "DetectorConfig",
    "DetectorTimings",
    "PixelFormat",
    "Yolo26Detector",
]

PixelFormat = Literal["BGR", "RGB", "BGRX"]
_BOX_COORDINATE_CHANNELS = 4
_LETTERBOX_VALUE = 114
_MAX_NMS_CANDIDATES = 30_000
_CLASS_BOX_OFFSET = 7680
type _FloatArray = NDArray[np.float32]
type _IndexArray = NDArray[np.int64]
type _Uint8Array = NDArray[np.uint8]


class _TensorSession(Protocol):
    input_shape: tuple[int, int, int, int]
    output_shape: tuple[int, int, int]
    class_names: tuple[str, ...]

    @property
    def closed(self) -> bool: ...

    def execute(self, tensor: _FloatArray) -> _FloatArray: ...

    def _prepare_image_without_resize(
        self,
        frame: _Uint8Array,
        *,
        pixel_format: str,
        output: _FloatArray,
    ) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    """Postprocessing settings for one detector instance."""

    confidence: float = 0.25
    iou: float = 0.45
    max_detections: int = 300
    multi_label: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not 0.0 <= self.iou <= 1.0:
            raise ValueError("iou must be between 0 and 1")
        if self.max_detections <= 0:
            raise ValueError("max_detections must be positive")


@dataclass(frozen=True, slots=True)
class Detection:
    """One detection in source-image pixel coordinates (x1, y1, x2, y2)."""

    box: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


@dataclass(frozen=True, slots=True)
class DetectorTimings:
    """Wall-clock time spent in each detector stage."""

    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Detections and timing information for one source frame."""

    detections: tuple[Detection, ...]
    source_shape: tuple[int, int]
    timings: DetectorTimings


class Yolo26Detector:
    """Convert image frames into labelled detections in source coordinates.

    The detector applies letterboxing, normalization, native inference,
    non-maximum suppression, and coordinate scaling. Each result also reports
    time spent preprocessing, executing the runtime, and postprocessing. One
    input tensor is reused between calls, so ``detect`` and ``close`` are
    deliberately non-reentrant; use separate processes for concurrent
    isolation. The runtime and model paths are keyword-only so callers cannot
    silently reverse two path-like arguments.
    """

    def __init__(
        self,
        *,
        runtime: str | Path,
        model: str | Path,
        config: DetectorConfig | None = None,
    ) -> None:
        self.config = config or DetectorConfig()
        self._session: _TensorSession = Yolo26nTensorSession(runtime=runtime, model=model)
        self._lifecycle_lock = threading.Lock()
        try:
            class_names = tuple(self._session.class_names)
            input_shape = tuple(self._session.input_shape)
            output_shape = tuple(self._session.output_shape)
            if (
                len(input_shape) != 4
                or input_shape[:2] != (1, 3)
                or input_shape[2] <= 0
                or input_shape[2] != input_shape[3]
            ):
                raise ValueError(
                    f"detector requires a square [1, 3, S, S] input, got {input_shape}"
                )
            if (
                len(output_shape) != 3
                or output_shape[0] != 1
                or output_shape[1] <= _BOX_COORDINATE_CHANNELS
                or output_shape[2] <= 0
            ):
                raise ValueError(
                    "detector requires output shape [1, 4 + classes, positions], "
                    f"got {output_shape}"
                )
            output_class_count = output_shape[1] - _BOX_COORDINATE_CHANNELS
            if len(class_names) != output_class_count or any(
                not isinstance(name, str) or not name for name in class_names
            ):
                raise ValueError(
                    "detector class names must be non-empty strings matching the "
                    f"{output_class_count} output classes; got {len(class_names)}"
                )
        except BaseException:
            self._session.close()
            raise
        self.class_names = class_names
        self.input_shape = input_shape
        self.output_shape = output_shape
        self._input_hw = (input_shape[2], input_shape[3])
        self._input_tensor = np.empty(input_shape, dtype=np.float32)

    @property
    def closed(self) -> bool:
        return self._session.closed

    def detect(
        self,
        frame: _Uint8Array,
        *,
        pixel_format: PixelFormat,
        on_frame_consumed: Callable[[], None] | None = None,
    ) -> DetectionResult:
        """Detect objects in one uint8 HWC BGR, RGB, or BGRX frame.

        Pixel-order conversion, letterboxing, normalization, native execution,
        non-maximum suppression, and box scaling are all included. Frames whose
        longest side already matches the square model input take the native
        no-resize path: centered value-114 padding, channel conversion,
        HWC-to-CHW conversion, and FP32 normalization are fused into the reused
        input tensor. Frames requiring interpolation use centered letterboxing.
        If provided, ``on_frame_consumed`` is called exactly once after the
        input tensor no longer refers to the source frame and before inference.
        """
        source_shape = self._validate_frame(frame, pixel_format)
        if not self._lifecycle_lock.acquire(blocking=False):
            raise Yolo26nError("concurrent detection or close is not supported")

        try:
            if self.closed:
                raise Yolo26nError("detector is closed")

            preprocess_started_at = perf_counter()
            self._prepare_input(frame, source_shape, pixel_format)
            preprocess_ms = (perf_counter() - preprocess_started_at) * 1000
            if on_frame_consumed is not None:
                on_frame_consumed()

            inference_started_at = perf_counter()
            raw = self._run_inference()
            inference_ms = (perf_counter() - inference_started_at) * 1000

            postprocess_started_at = perf_counter()
            detections = self._postprocess(raw, source_shape)
            postprocess_ms = (perf_counter() - postprocess_started_at) * 1000
            return DetectionResult(
                detections=detections,
                source_shape=source_shape,
                timings=DetectorTimings(
                    preprocess_ms=preprocess_ms,
                    inference_ms=inference_ms,
                    postprocess_ms=postprocess_ms,
                ),
            )
        finally:
            self._lifecycle_lock.release()

    @staticmethod
    def _validate_frame(
        frame: _Uint8Array,
        pixel_format: PixelFormat,
    ) -> tuple[int, int]:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy.ndarray")
        if frame.dtype != np.uint8:
            raise TypeError("frame must have dtype uint8")
        if pixel_format not in ("BGR", "RGB", "BGRX"):
            raise ValueError("pixel_format must be 'BGR', 'RGB', or 'BGRX'")
        expected_channels = 4 if pixel_format == "BGRX" else 3
        if frame.ndim != 3 or frame.shape[2] != expected_channels:
            raise ValueError(
                f"{pixel_format} frame must have shape [height, width, {expected_channels}]"
            )
        if frame.shape[0] == 0 or frame.shape[1] == 0:
            raise ValueError("frame dimensions must be positive")
        return (int(frame.shape[0]), int(frame.shape[1]))

    def _prepare_input(
        self,
        frame: _Uint8Array,
        source_shape: tuple[int, int],
        pixel_format: PixelFormat,
    ) -> None:
        if max(source_shape) == self._input_hw[0]:
            self._session._prepare_image_without_resize(
                frame,
                pixel_format=pixel_format,
                output=self._input_tensor,
            )
            return

        three_channel_frame = frame[..., :3] if pixel_format == "BGRX" else frame
        image = _letterbox(three_channel_frame, self._input_hw)
        image_rgb = image[..., ::-1] if pixel_format in ("BGR", "BGRX") else image
        np.copyto(
            self._input_tensor[0],
            image_rgb.transpose(2, 0, 1),
            casting="unsafe",
        )
        np.divide(self._input_tensor, np.float32(255.0), out=self._input_tensor)

    def _run_inference(self) -> _FloatArray:
        return self._session.execute(self._input_tensor)

    def _postprocess(
        self,
        raw: _FloatArray,
        source_shape: tuple[int, int],
    ) -> tuple[Detection, ...]:
        processed = _non_max_suppression(
            raw[0],
            confidence=self.config.confidence,
            iou=self.config.iou,
            max_detections=self.config.max_detections,
            multi_label=self.config.multi_label,
        )
        _scale_boxes(processed[:, :4], self._input_hw, source_shape)
        detections = []
        for row in processed:
            class_id = int(row[5])
            if not 0 <= class_id < len(self.class_names):
                raise Yolo26nError(f"postprocessing returned unknown class ID {class_id}")
            x1, y1, x2, y2 = (float(value) for value in row[:4])
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(
                    box=(x1, y1, x2, y2),
                    confidence=float(row[4]),
                    class_id=class_id,
                    class_name=self.class_names[class_id],
                )
            )
        return tuple(detections)

    def close(self) -> None:
        """Release the detector's tensor session; repeated calls are safe."""
        with self._lifecycle_lock:
            self._session.close()

    def __enter__(self) -> Self:
        if self.closed:
            raise Yolo26nError("detector is closed")
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()


def _letterbox(image: _Uint8Array, target_shape: tuple[int, int]) -> _Uint8Array:
    """Resize without distortion, then center the image on a square canvas."""
    source_height, source_width = image.shape[:2]
    target_height, target_width = target_shape
    scale = min(target_height / source_height, target_width / source_width)
    resized_width = round(source_width * scale)
    resized_height = round(source_height * scale)
    if (source_width, source_height) != (resized_width, resized_height):
        image = cast(
            _Uint8Array,
            cv2.resize(
                image,
                (resized_width, resized_height),
                interpolation=cv2.INTER_LINEAR,
            ),
        )

    horizontal_padding = (target_width - resized_width) / 2
    vertical_padding = (target_height - resized_height) / 2
    left = round(horizontal_padding - 0.1)
    right = round(horizontal_padding + 0.1)
    top = round(vertical_padding - 0.1)
    bottom = round(vertical_padding + 0.1)
    return cast(
        _Uint8Array,
        cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(_LETTERBOX_VALUE,) * 3,
        ),
    )


def _xywh_to_xyxy(boxes: _FloatArray) -> _FloatArray:
    converted = np.empty_like(boxes)
    half_size = boxes[:, 2:] / np.float32(2.0)
    converted[:, :2] = boxes[:, :2] - half_size
    converted[:, 2:] = boxes[:, :2] + half_size
    return converted


def _candidate_detections(
    prediction: _FloatArray,
    confidence: float,
    multi_label: bool,
) -> _FloatArray:
    class_scores = prediction[_BOX_COORDINATE_CHANNELS:].T
    candidate_positions = np.max(class_scores, axis=1) > confidence
    if not np.any(candidate_positions):
        return np.empty((0, 6), dtype=np.float32)

    boxes = _xywh_to_xyxy(prediction[:_BOX_COORDINATE_CHANNELS].T[candidate_positions])
    scores = class_scores[candidate_positions]
    if multi_label and scores.shape[1] > 1:
        box_indices, class_ids = np.nonzero(scores > confidence)
        return np.column_stack(
            (boxes[box_indices], scores[box_indices, class_ids], class_ids.astype(np.float32))
        ).astype(np.float32, copy=False)

    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]
    accepted = confidences > confidence
    return np.column_stack(
        (boxes[accepted], confidences[accepted], class_ids[accepted].astype(np.float32))
    ).astype(np.float32, copy=False)


def _nms_indices(
    boxes: _FloatArray,
    scores: _FloatArray,
    iou: float,
) -> _IndexArray:
    """Return descending-confidence indices after class-aware box suppression."""
    if not len(boxes):
        return np.empty(0, dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    # Reversing a stable ascending order makes exact score ties deterministic:
    # the later model candidate takes precedence.
    order = np.argsort(scores, kind="stable")[::-1]
    kept: list[int] = []
    while len(order):
        current = int(order[0])
        kept.append(current)
        if len(order) == 1:
            break
        remaining = order[1:]
        intersection_width = np.maximum(
            np.minimum(x2[current], x2[remaining]) - np.maximum(x1[current], x1[remaining]),
            np.float32(0.0),
        )
        intersection_height = np.maximum(
            np.minimum(y2[current], y2[remaining]) - np.maximum(y1[current], y1[remaining]),
            np.float32(0.0),
        )
        intersection = intersection_width * intersection_height
        overlap = intersection / (areas[current] + areas[remaining] - intersection)
        order = remaining[overlap <= iou]
    return np.asarray(kept, dtype=np.int64)


def _non_max_suppression(
    prediction: _FloatArray,
    *,
    confidence: float,
    iou: float,
    max_detections: int,
    multi_label: bool,
) -> _FloatArray:
    detections = _candidate_detections(prediction, confidence, multi_label)
    if not len(detections):
        return detections
    if len(detections) > _MAX_NMS_CANDIDATES:
        highest_confidence = np.argsort(detections[:, 4], kind="stable")[::-1][:_MAX_NMS_CANDIDATES]
        detections = detections[highest_confidence]

    # Offsetting coordinates by class prevents boxes from different classes
    # suppressing one another while retaining a single vectorized NMS pass.
    class_offsets = detections[:, 5:6] * np.float32(_CLASS_BOX_OFFSET)
    boxes = detections[:, :4] + class_offsets
    kept = _nms_indices(boxes, detections[:, 4], iou)[:max_detections]
    return detections[kept]


def _scale_boxes(
    boxes: _FloatArray,
    model_shape: tuple[int, int],
    source_shape: tuple[int, int],
) -> None:
    """Map letterboxed model coordinates back to source-image pixels."""
    if not len(boxes):
        return
    model_height, model_width = model_shape
    source_height, source_width = source_shape
    scale = min(model_height / source_height, model_width / source_width)
    horizontal_padding = round((model_width - round(source_width * scale)) / 2 - 0.1)
    vertical_padding = round((model_height - round(source_height * scale)) / 2 - 0.1)
    boxes[:, (0, 2)] -= horizontal_padding
    boxes[:, (1, 3)] -= vertical_padding
    boxes /= np.float32(scale)
    boxes[:, (0, 2)] = boxes[:, (0, 2)].clip(0, source_width)
    boxes[:, (1, 3)] = boxes[:, (1, 3)].clip(0, source_height)

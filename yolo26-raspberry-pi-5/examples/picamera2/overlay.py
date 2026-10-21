"""Rendering and rolling performance metrics for the Picamera2 live demo."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Protocol

import cv2
import numpy as np


class DetectionLike(Protocol):
    """The portion of ``Detection`` needed by the overlay renderer."""

    box: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


class RollingPerformance:
    """Rolling detection throughput and mean inference latency."""

    def __init__(self, window_seconds: float = 1.5) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.window_seconds = window_seconds
        self._samples: deque[tuple[float, float]] = deque()

    def add(self, completed_at: float, inference_ms: float) -> None:
        if self._samples and completed_at < self._samples[-1][0]:
            raise ValueError("completed_at must be monotonic")
        if inference_ms < 0:
            raise ValueError("inference_ms must be non-negative")
        self._samples.append((completed_at, inference_ms))
        cutoff = completed_at - self.window_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    @property
    def detection_fps(self) -> float | None:
        if len(self._samples) < 2:
            return None
        elapsed = self._samples[-1][0] - self._samples[0][0]
        return (len(self._samples) - 1) / elapsed if elapsed > 0 else None

    @property
    def inference_mean_ms(self) -> float | None:
        if not self._samples:
            return None
        return sum(sample[1] for sample in self._samples) / len(self._samples)


class LatestOverlayUploader:
    """Render and upload only the newest requested overlay off-thread."""

    def __init__(
        self,
        renderer: OverlayRenderer,
        width: int,
        height: int,
        upload: Callable[[np.ndarray], None],
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("overlay dimensions must be positive")
        self._renderer = renderer
        self._upload = upload
        self._buffer = np.empty((height, width, 4), dtype=np.uint8)
        self._requested = threading.Event()
        self._closed = threading.Event()
        self._metrics_lock = threading.Lock()
        self._upload_count = 0
        self._upload_seconds = 0.0
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="picamera2-overlay-upload",
            daemon=True,
        )
        self._thread.start()

    def request(self) -> None:
        """Request an upload of the newest renderer state without blocking."""
        self._raise_if_failed()
        if self._closed.is_set():
            raise RuntimeError("overlay uploader is closed")
        self._requested.set()

    def close(self) -> None:
        """Stop after the active upload, discarding any pending request."""
        self._closed.set()
        self._requested.set()
        self._thread.join()
        self._raise_if_failed()

    @property
    def upload_mean_ms(self) -> float:
        with self._metrics_lock:
            if not self._upload_count:
                return 0.0
            return self._upload_seconds * 1000 / self._upload_count

    @property
    def upload_count(self) -> int:
        with self._metrics_lock:
            return self._upload_count

    def _run(self) -> None:
        try:
            while True:
                self._requested.wait()
                self._requested.clear()
                if self._closed.is_set():
                    return
                self._renderer.draw_rgba(self._buffer)
                started_at = perf_counter()
                self._upload(self._buffer)
                elapsed = perf_counter() - started_at
                with self._metrics_lock:
                    self._upload_count += 1
                    self._upload_seconds += elapsed
        except BaseException as error:
            self._error = error
            self._closed.set()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("overlay upload worker failed") from self._error


# OpenCV writes these bytes directly into Picamera2's XRGB8888 buffer, whose
# NumPy representation is BGRX on this little-endian platform.
_COLORS_BGRX = (
    (255, 200, 70, 0),
    (90, 120, 255, 0),
    (145, 235, 105, 0),
    (255, 130, 220, 0),
    (75, 210, 255, 0),
    (225, 220, 90, 0),
    (190, 135, 255, 0),
    (80, 220, 165, 0),
)


class OverlayRenderer:
    """Draw the latest detections into a Picamera2 XRGB8888 preview buffer."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        show_confidence: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("overlay dimensions must be positive")
        self.width = width
        self.height = height
        self.show_confidence = show_confidence
        self._lock = threading.Lock()
        self._detections: tuple[DetectionLike, ...] = ()
        self._status = self._format_status("--", "--")
        self._render_count = 0
        self._render_seconds = 0.0

    def update(
        self,
        detections: Sequence[DetectionLike],
        performance: RollingPerformance,
    ) -> None:
        """Publish one immutable result for subsequent preview frames."""
        fps = performance.detection_fps
        inference_ms = performance.inference_mean_ms
        fps_text = "--" if fps is None else f"{fps:.1f}"
        inference_text = "--" if inference_ms is None else f"{inference_ms:.1f}"
        with self._lock:
            self._detections = tuple(detections)
            self._status = self._format_status(fps_text, inference_text)

    def _format_status(self, fps_text: str, inference_text: str) -> str:
        return f"Detect {fps_text} FPS  |  Inference {inference_text} ms"

    def draw(self, image: np.ndarray) -> None:
        """Draw the published result in place on one camera preview frame."""
        if image.dtype != np.uint8 or image.shape != (self.height, self.width, 4):
            raise ValueError(
                "preview image must be a uint8 XRGB8888 array with shape "
                f"[{self.height}, {self.width}, 4]"
            )
        started_at = perf_counter()
        with self._lock:
            detections = self._detections
            status = self._status

        for detection in detections:
            x1, y1, x2, y2 = detection.box
            left = min(max(int(round(x1)), 0), self.width - 1)
            top = min(max(int(round(y1)), 0), self.height - 1)
            right = min(max(int(round(x2)), 0), self.width - 1)
            bottom = min(max(int(round(y2)), 0), self.height - 1)
            if right <= left or bottom <= top:
                continue

            color = _COLORS_BGRX[detection.class_id % len(_COLORS_BGRX)]
            cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            label = detection.class_name
            if self.show_confidence:
                label += f" {detection.confidence:.2f}"
            self._draw_label(image, label, left, top, color)

        self._draw_status(image, status)
        self._record_render_time(perf_counter() - started_at)

    def snapshot(self) -> tuple[tuple[DetectionLike, ...], str]:
        """Return the latest immutable overlay state for another renderer."""
        with self._lock:
            return self._detections, self._status

    def draw_rgba(self, image: np.ndarray) -> None:
        """Draw a native-resolution transparent preview overlay in place."""
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 4:
            raise ValueError("overlay image must be a uint8 RGBA array")
        started_at = perf_counter()
        detections, status = self.snapshot()
        image.fill(0)
        height, width = image.shape[:2]
        scale = min(width / self.width, height / self.height)
        x_offset = (width - self.width * scale) / 2
        y_offset = (height - self.height * scale) / 2

        for detection in detections:
            self._draw_rgba_detection(image, detection, scale, x_offset, y_offset)
        self._draw_rgba_status(image, status, scale, y_offset)
        self._record_render_time(perf_counter() - started_at)

    def _draw_rgba_detection(
        self,
        image: np.ndarray,
        detection: DetectionLike,
        scale: float,
        x_offset: float,
        y_offset: float,
    ) -> None:
        x1, y1, x2, y2 = detection.box
        left = round(x_offset + min(max(x1, 0), self.width - 1) * scale)
        top = round(y_offset + min(max(y1, 0), self.height - 1) * scale)
        right = round(x_offset + min(max(x2, 0), self.width - 1) * scale)
        bottom = round(y_offset + min(max(y2, 0), self.height - 1) * scale)
        if right <= left or bottom <= top:
            return
        blue, green, red, _ = _COLORS_BGRX[detection.class_id % len(_COLORS_BGRX)]
        color = (red, green, blue, 255)
        cv2.rectangle(image, (left, top), (right, bottom), color, max(2, round(2 * scale)))
        label = detection.class_name
        if self.show_confidence:
            label += f" {detection.confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45 * scale
        thickness = max(1, round(scale))
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        image_height, image_width = image.shape[:2]
        box_width = min(text_width + round(8 * scale), image_width)
        box_height = text_height + baseline + round(6 * scale)
        label_left = min(left, image_width - box_width)
        label_top = max(round(y_offset), top - box_height)
        label_bottom = min(label_top + box_height, image_height - 1)
        cv2.rectangle(
            image,
            (label_left, label_top),
            (label_left + box_width, label_bottom),
            (0, 0, 0, 220),
            cv2.FILLED,
        )
        cv2.putText(
            image,
            label,
            (label_left + round(4 * scale), label_bottom - baseline - round(3 * scale)),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _draw_rgba_status(
        self,
        image: np.ndarray,
        status: str,
        scale: float,
        y_offset: float,
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.435 * scale
        thickness = max(1, round(scale))
        (text_width, text_height), baseline = cv2.getTextSize(status, font, font_scale, thickness)
        image_height, image_width = image.shape[:2]
        panel_width = min(text_width + round(15 * scale), image_width)
        panel_height = text_height + baseline + round(14 * scale)
        panel_left = max(0, round((image_width - panel_width) / 2))
        panel_right = min(panel_left + panel_width, image_width - 1)
        panel_bottom = min(round(y_offset + self.height * scale - 6 * scale), image_height - 1)
        panel_top = max(round(y_offset), panel_bottom - panel_height)
        cv2.rectangle(
            image,
            (panel_left, panel_top),
            (panel_right, panel_bottom),
            (0, 0, 0, 220),
            cv2.FILLED,
        )
        cv2.putText(
            image,
            status,
            (panel_left + round(5 * scale), panel_bottom - baseline - round(5 * scale)),
            font,
            font_scale,
            (255, 255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    def _record_render_time(self, elapsed_seconds: float) -> None:
        with self._lock:
            self._render_count += 1
            self._render_seconds += elapsed_seconds

    @property
    def render_mean_ms(self) -> float:
        with self._lock:
            if not self._render_count:
                return 0.0
            return self._render_seconds * 1000 / self._render_count

    @staticmethod
    def _draw_label(
        image: np.ndarray,
        label: str,
        left: int,
        top: int,
        color: tuple[int, int, int, int],
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        image_height, image_width = image.shape[:2]
        box_width = min(text_width + 8, image_width)
        box_height = text_height + baseline + 6
        label_left = min(left, image_width - box_width)
        label_top = max(0, top - box_height)
        label_bottom = min(label_top + box_height, image_height)
        cv2.rectangle(
            image,
            (label_left, label_top),
            (label_left + box_width, label_bottom),
            (0, 0, 0, 0),
            cv2.FILLED,
        )
        cv2.putText(
            image,
            label,
            (label_left + 4, label_bottom - baseline - 3),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_status(image: np.ndarray, status: str) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.435
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(status, font, font_scale, thickness)
        image_height, image_width = image.shape[:2]
        panel_width = min(text_width + 15, image_width)
        panel_height = text_height + baseline + 14
        panel_left = max(0, (image_width - panel_width) // 2)
        panel_right = min(panel_left + panel_width, image_width - 1)
        panel_bottom = image_height - 6
        panel_top = max(0, panel_bottom - panel_height)
        cv2.rectangle(
            image,
            (panel_left, panel_top),
            (panel_right, panel_bottom),
            (0, 0, 0, 0),
            cv2.FILLED,
        )
        cv2.putText(
            image,
            status,
            (panel_left + 5, panel_bottom - baseline - 5),
            font,
            font_scale,
            (255, 255, 255, 0),
            thickness,
            cv2.LINE_AA,
        )

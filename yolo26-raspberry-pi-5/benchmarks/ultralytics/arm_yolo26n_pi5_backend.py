"""Ultralytics backend for the Arm-optimized YOLO26n Raspberry Pi 5 runtime."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from arm_yolo26n import Yolo26nTensorSession
from ultralytics.nn.backends.base import BaseBackend
from ultralytics.utils import LOGGER, YAML

_YOLO26N_SUFFIX = "arm_yolo26n_pi5_model"


class ArmYolo26nPi5Backend(BaseBackend):
    """Run a converted YOLO26n model through the native tensor session."""

    def load_model(self, weight: str | Path) -> None:
        model_directory = Path(weight).expanduser().resolve()
        if not model_directory.is_dir():
            raise ValueError(f"Arm YOLO26n Pi 5 model must be a directory: {model_directory}")
        if not model_directory.name.endswith(_YOLO26N_SUFFIX):
            raise ValueError(
                f"Arm YOLO26n Pi 5 model directory must end with {_YOLO26N_SUFFIX}: "
                f"{model_directory}"
            )
        self._load_yolo26n(model_directory)

    @staticmethod
    def _descriptor(model_directory: Path, filename: str) -> tuple[Path, Mapping[str, object]]:
        descriptor_path = model_directory / filename
        if not descriptor_path.is_file():
            raise FileNotFoundError(f"missing model descriptor: {descriptor_path}")
        descriptor = YAML.load(descriptor_path)
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"model descriptor must be a mapping: {descriptor_path}")
        return descriptor_path, descriptor

    def _load_yolo26n(self, model_directory: Path) -> None:
        descriptor_path, descriptor = self._descriptor(model_directory, "arm-yolo26n-pi5.yaml")
        if descriptor.get("backend") != "arm_yolo26n_pi5" or set(descriptor) != {
            "backend",
            "model",
            "runtime",
        }:
            raise ValueError(f"invalid YOLO26n runtime descriptor: {descriptor_path}")
        model_path = _descriptor_path(model_directory, descriptor.get("model"), "model")
        runtime_path = _descriptor_path(model_directory, descriptor.get("runtime"), "runtime")
        # The native runtime already occupies all four Pi 5 cores. Keep
        # Ultralytics' surrounding Torch work from creating a competing
        # four-thread intra-op pool between inference calls.
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        LOGGER.info(f"Loading {model_directory} with ArmYolo26nPi5Backend...")
        self.model = Yolo26nTensorSession(runtime=runtime_path, model=model_path)
        height, width = self.model.input_shape[2:]
        self.apply_metadata(
            {
                "task": "detect",
                "stride": 32,
                "batch": 1,
                "imgsz": [height, width],
                "names": dict(enumerate(self.model.class_names)),
            }
        )

    def forward(self, im: torch.Tensor) -> list[torch.Tensor]:
        if im.device.type != "cpu":
            raise ValueError("ArmYolo26nPi5Backend requires a CPU input tensor")
        if im.dtype is not torch.float32:
            raise TypeError("ArmYolo26nPi5Backend requires an FP32 input tensor")
        model_input = im.detach().numpy()
        prediction = self.model.execute(model_input)
        return [torch.from_numpy(prediction)]

    def close(self) -> None:
        if self.model is not None:
            self.model.close()


def _descriptor_path(directory: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"YOLO26n descriptor {name} must be a non-empty path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (directory / path).resolve()

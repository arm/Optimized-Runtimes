"""Typed contracts between YOLO26n checkpoint intake and model conversion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .converter_io import JsonObject

# The native schedule accepts these square inputs and derives all intermediate
# tensor extents from them. Values outside this closed set are rejected by both
# the exporter and runtime before graph construction.
SUPPORTED_RESOLUTIONS = (320, 480, 640, 800, 960, 1280)

# YOLO26n emits one detection branch at each of these fixed reductions. The
# branch count and output-position calculation are derived from this tuple.
DETECTION_STRIDES = (8, 16, 32)
DETECTION_BRANCH_COUNT = len(DETECTION_STRIDES)

# Each detection position contains four box-coordinate channels followed by one
# confidence channel for each class.
BOX_COORDINATE_CHANNELS = 4
_CLASSIFICATION_CHANNEL_FLOOR = 64

# These are the 102 convolution outputs consumed by the fixed runtime graphs.
# Every convolution contributes a weight and a bias after BatchNorm folding,
# except that two weights are split at the input boundaries listed below. The
# tuple is lexicographically ordered to match validated checkpoint paths.
RUNTIME_CONVOLUTION_PATHS = (
    "model.0.conv",
    "model.1.conv",
    "model.10.cv1.conv",
    "model.10.cv2.conv",
    "model.10.m.0.attn.pe.conv",
    "model.10.m.0.attn.proj.conv",
    "model.10.m.0.attn.qkv.conv",
    "model.10.m.0.ffn.0.conv",
    "model.10.m.0.ffn.1.conv",
    "model.13.cv1.conv",
    "model.13.cv2.conv",
    "model.13.m.0.cv1.conv",
    "model.13.m.0.cv2.conv",
    "model.13.m.0.cv3.conv",
    "model.13.m.0.m.0.cv1.conv",
    "model.13.m.0.m.0.cv2.conv",
    "model.13.m.0.m.1.cv1.conv",
    "model.13.m.0.m.1.cv2.conv",
    "model.16.cv1.conv",
    "model.16.cv2.conv",
    "model.16.m.0.cv1.conv",
    "model.16.m.0.cv2.conv",
    "model.16.m.0.cv3.conv",
    "model.16.m.0.m.0.cv1.conv",
    "model.16.m.0.m.0.cv2.conv",
    "model.16.m.0.m.1.cv1.conv",
    "model.16.m.0.m.1.cv2.conv",
    "model.17.conv",
    "model.19.cv1.conv",
    "model.19.cv2.conv",
    "model.19.m.0.cv1.conv",
    "model.19.m.0.cv2.conv",
    "model.19.m.0.cv3.conv",
    "model.19.m.0.m.0.cv1.conv",
    "model.19.m.0.m.0.cv2.conv",
    "model.19.m.0.m.1.cv1.conv",
    "model.19.m.0.m.1.cv2.conv",
    "model.2.cv1.conv",
    "model.2.cv2.conv",
    "model.2.m.0.cv1.conv",
    "model.2.m.0.cv2.conv",
    "model.20.conv",
    "model.22.cv1.conv",
    "model.22.cv2.conv",
    "model.22.m.0.0.cv1.conv",
    "model.22.m.0.0.cv2.conv",
    "model.22.m.0.1.attn.pe.conv",
    "model.22.m.0.1.attn.proj.conv",
    "model.22.m.0.1.attn.qkv.conv",
    "model.22.m.0.1.ffn.0.conv",
    "model.22.m.0.1.ffn.1.conv",
    "model.23.cv2.0.0.conv",
    "model.23.cv2.0.1.conv",
    "model.23.cv2.0.2",
    "model.23.cv2.1.0.conv",
    "model.23.cv2.1.1.conv",
    "model.23.cv2.1.2",
    "model.23.cv2.2.0.conv",
    "model.23.cv2.2.1.conv",
    "model.23.cv2.2.2",
    "model.23.cv3.0.0.0.conv",
    "model.23.cv3.0.0.1.conv",
    "model.23.cv3.0.1.0.conv",
    "model.23.cv3.0.1.1.conv",
    "model.23.cv3.0.2",
    "model.23.cv3.1.0.0.conv",
    "model.23.cv3.1.0.1.conv",
    "model.23.cv3.1.1.0.conv",
    "model.23.cv3.1.1.1.conv",
    "model.23.cv3.1.2",
    "model.23.cv3.2.0.0.conv",
    "model.23.cv3.2.0.1.conv",
    "model.23.cv3.2.1.0.conv",
    "model.23.cv3.2.1.1.conv",
    "model.23.cv3.2.2",
    "model.3.conv",
    "model.4.cv1.conv",
    "model.4.cv2.conv",
    "model.4.m.0.cv1.conv",
    "model.4.m.0.cv2.conv",
    "model.5.conv",
    "model.6.cv1.conv",
    "model.6.cv2.conv",
    "model.6.m.0.cv1.conv",
    "model.6.m.0.cv2.conv",
    "model.6.m.0.cv3.conv",
    "model.6.m.0.m.0.cv1.conv",
    "model.6.m.0.m.0.cv2.conv",
    "model.6.m.0.m.1.cv1.conv",
    "model.6.m.0.m.1.cv2.conv",
    "model.7.conv",
    "model.8.cv1.conv",
    "model.8.cv2.conv",
    "model.8.m.0.cv1.conv",
    "model.8.m.0.cv2.conv",
    "model.8.m.0.cv3.conv",
    "model.8.m.0.m.0.cv1.conv",
    "model.8.m.0.m.0.cv2.conv",
    "model.8.m.0.m.1.cv1.conv",
    "model.8.m.0.m.1.cv2.conv",
    "model.9.cv1.conv",
    "model.9.cv2.conv",
)

SPLIT_PROJECTION_CHANNEL_RANGES = (
    ("model.13.cv1.conv.weight", 0, 256),
    ("model.13.cv1.conv.weight", 256, 384),
    ("model.16.cv1.conv.weight", 0, 128),
    ("model.16.cv1.conv.weight", 128, 256),
)
SPLIT_PROJECTION_ZERO_BIAS_CHANNELS = (128, 64)


def _runtime_weight_names() -> tuple[str, ...]:
    """Derive every ordered runtime slot name from the fixed model topology."""
    split_sources = {source for source, _begin, _end in SPLIT_PROJECTION_CHANNEL_RANGES}
    names = [f"{path}.bias" for path in RUNTIME_CONVOLUTION_PATHS]
    names.extend(
        f"{path}.weight"
        for path in RUNTIME_CONVOLUTION_PATHS
        if f"{path}.weight" not in split_sources
    )
    names.extend(
        f"{source}[input:{begin}:{end}]" for source, begin, end in SPLIT_PROJECTION_CHANNEL_RANGES
    )
    names.extend(
        f"projection.zero_bias.{channels}" for channels in SPLIT_PROJECTION_ZERO_BIAS_CHANNELS
    )
    return tuple(sorted(names))


RUNTIME_WEIGHT_NAMES = _runtime_weight_names()
# Stable slot numbers are assigned by semantic name. The fixed topology above
# therefore determines the slot count and order used by both exporter and runtime.
RUNTIME_WEIGHT_SLOT_COUNT = len(RUNTIME_WEIGHT_NAMES)

_RUNTIME_TARGET = "linux-aarch64-raspberry-pi-5"
_RUNTIME_THREAD_COUNT = 4


def classification_width_for_classes(class_count: int) -> int:
    """Return the classification-branch width implemented by YOLO26n."""
    if type(class_count) is not int or class_count <= 0:
        raise ValueError(f"unsupported YOLO26n class count: {class_count}")
    return max(_CLASSIFICATION_CHANNEL_FLOOR, class_count)


def detection_output_channels(class_count: int) -> int:
    """Return four box coordinates plus one confidence channel per class."""
    if type(class_count) is not int or class_count <= 0:
        raise ValueError(f"unsupported YOLO26n class count: {class_count}")
    return BOX_COORDINATE_CHANNELS + class_count


class ConvolutionActivation(Enum):
    """Activation fused into or scheduled immediately after a convolution."""

    IDENTITY = "identity"
    SILU = "silu"


@dataclass(frozen=True)
class LearnedTensor:
    """One named checkpoint tensor copied out of Torch-owned storage."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    data: bytes

    def __post_init__(self) -> None:
        element_bytes = {"float16": 2, "int64": 8}.get(self.dtype)
        elements = 1
        for extent in self.shape:
            if type(extent) is not int or extent <= 0:
                raise ValueError("invalid learned tensor shape")
            elements *= extent
        if not self.name or element_bytes is None or len(self.data) != elements * element_bytes:
            raise ValueError("invalid learned tensor storage")


@dataclass(frozen=True)
class ConvolutionSpec:
    """Convolution geometry needed to validate and lay out its learned weights."""

    path: str
    input_channels: int
    output_channels: int
    kernel: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]
    groups: int
    activation: ConvolutionActivation

    def __post_init__(self) -> None:
        if (
            not self.path
            or min(self.input_channels, self.output_channels, self.groups) <= 0
            or self.input_channels % self.groups
            or self.output_channels % self.groups
            or any(value <= 0 for value in self.kernel + self.stride + self.dilation)
            or any(value < 0 for value in self.padding)
        ):
            raise ValueError("invalid validated convolution")


@dataclass(frozen=True)
class BatchNormFusion:
    """A BatchNorm wrapper whose statistics are folded into its convolution."""

    convolution_path: str
    epsilon: float

    def __post_init__(self) -> None:
        if not self.convolution_path or not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("invalid BatchNorm fusion contract")


@dataclass(frozen=True)
class AttentionSpec:
    """Attention dimensions reproduced by one of the fixed runtime graphs."""

    path: str
    heads: int
    head_channels: int
    key_channels: int
    query_scale: float

    def __post_init__(self) -> None:
        if (
            not self.path
            or min(self.heads, self.head_channels, self.key_channels) <= 0
            or self.head_channels % 2
            or self.key_channels != self.head_channels // 2
            or self.query_scale != self.key_channels**-0.5
        ):
            raise ValueError("invalid validated attention")


@dataclass(frozen=True)
class ValidatedYolo26n:
    """Checkpoint data copied into the exact facts needed by later stages."""

    learned_tensors: tuple[LearnedTensor, ...]
    convolutions: tuple[ConvolutionSpec, ...]
    batch_norm_fusions: tuple[BatchNormFusion, ...]
    attention_blocks: tuple[AttentionSpec, ...]
    class_names: tuple[str, ...]
    classification_width: int
    detection_strides: tuple[int, int, int]

    def __post_init__(self) -> None:
        if not self.class_names or any(not name for name in self.class_names):
            raise ValueError("validated YOLO26n class names must be non-empty")
        if self.classification_width != classification_width_for_classes(self.class_count):
            raise ValueError("validated YOLO26n classification width changed")
        if self.detection_strides != DETECTION_STRIDES:
            raise ValueError("validated YOLO26n detection strides changed")
        inventories = (
            [row.name for row in self.learned_tensors],
            [row.path for row in self.convolutions],
            [row.convolution_path for row in self.batch_norm_fusions],
            [row.path for row in self.attention_blocks],
        )
        if any(len(rows) != len(set(rows)) for rows in inventories):
            raise ValueError("validated YOLO26n inventory contains duplicates")

    @property
    def class_count(self) -> int:
        return len(self.class_names)

    def convolution(self, path: str) -> ConvolutionSpec:
        matches = [row for row in self.convolutions if row.path == path]
        if len(matches) != 1:
            raise ValueError(f"unknown validated convolution: {path}")
        return matches[0]

    def attention(self, path: str) -> AttentionSpec:
        matches = [row for row in self.attention_blocks if row.path == path]
        if len(matches) != 1:
            raise ValueError(f"unknown validated attention: {path}")
        return matches[0]


@dataclass(frozen=True)
class ModelResolution:
    """One validated square input size and its derived detector dimensions."""

    resolution: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.resolution, int)
            or isinstance(self.resolution, bool)
            or self.resolution not in SUPPORTED_RESOLUTIONS
        ):
            raise ValueError(f"unsupported YOLO26n resolution: {self.resolution}")

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        return (1, 3, self.resolution, self.resolution)

    @property
    def detection_positions(self) -> int:
        extents = (self.resolution // stride for stride in DETECTION_STRIDES)
        return sum(extent * extent for extent in extents)

    def output_shape(self, class_count: int) -> tuple[int, int, int]:
        """Return the runtime output shape for a model's class inventory."""
        return (1, detection_output_channels(class_count), self.detection_positions)

    def target_manifest(self) -> JsonObject:
        """Describe the fixed runtime target paired with this model artifact."""
        return {
            "version": 1,
            "target": _RUNTIME_TARGET,
            "input_shape": list(self.input_shape),
            "input_dtype": "fp32",
            "output_dtype": "fp32",
            "thread_count": _RUNTIME_THREAD_COUNT,
        }

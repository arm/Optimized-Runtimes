"""Semantic runtime weight identities shared by export and graph maintenance."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class WeightLayout(Enum):
    """Static-data ordering expected by the receiving XNNPACK operator."""

    IDENTITY = "identity"
    CONVOLUTION = "oihw-to-ohwi"
    DEPTHWISE_CONVOLUTION = "depthwise-oihw-to-mhwc"


def _valid_shape(shape: tuple[int, ...]) -> bool:
    return bool(shape) and all(
        isinstance(extent, int) and not isinstance(extent, bool) and extent > 0 for extent in shape
    )


@dataclass(frozen=True)
class CheckpointTensor:
    """A complete checkpoint tensor, optionally reordered for XNNPACK."""

    source: str
    source_shape: tuple[int, ...]
    layout: WeightLayout

    def __post_init__(self) -> None:
        if not self.source or not _valid_shape(self.source_shape):
            raise ValueError("invalid checkpoint tensor derivation")


@dataclass(frozen=True)
class SplitProjectionWeight:
    """Channels extracted when the runtime evaluates one projection in two parts."""

    source: str
    input_channel_slice: tuple[int, int]

    def __post_init__(self) -> None:
        begin, end = self.input_channel_slice
        if (
            not self.source
            or type(begin) is not int
            or type(end) is not int
            or not 0 <= begin < end
        ):
            raise ValueError("invalid split-projection weight derivation")


@dataclass(frozen=True)
class ZeroBias:
    """A synthesized zero bias required by one split projection."""

    channels: int

    def __post_init__(self) -> None:
        if type(self.channels) is not int or self.channels <= 0:
            raise ValueError("invalid zero-bias derivation")


type WeightDerivation = CheckpointTensor | SplitProjectionWeight | ZeroBias


def weight_derivation_name(derivation: WeightDerivation) -> str:
    """Name a runtime weight by model meaning, never learned content."""
    if isinstance(derivation, CheckpointTensor):
        return derivation.source
    if isinstance(derivation, SplitProjectionWeight):
        begin, end = derivation.input_channel_slice
        return f"{derivation.source}[input:{begin}:{end}]"
    return f"projection.zero_bias.{derivation.channels}"


@dataclass(frozen=True)
class WeightBinding:
    """The stable semantic identity and byte extent of one runtime weight."""

    name: str
    size: int
    derivation: WeightDerivation

    def __post_init__(self) -> None:
        if (
            self.name != weight_derivation_name(self.derivation)
            or not isinstance(self.size, int)
            or isinstance(self.size, bool)
            or self.size <= 0
        ):
            raise ValueError("invalid runtime weight binding")


@dataclass(frozen=True)
class LoweredWeight:
    """One named, layout-converted byte payload ready for a runtime slot."""

    binding: WeightBinding
    payload: bytes

    def __post_init__(self) -> None:
        if self.binding.size != len(self.payload):
            raise ValueError("lowered weight size disagrees with payload")

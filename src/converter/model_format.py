"""Write the fixed-topology YOLO26n model container consumed by the runtime."""

from __future__ import annotations

import struct
from dataclasses import dataclass

from .model_contract import (
    BOX_COORDINATE_CHANNELS,
    RUNTIME_WEIGHT_NAMES,
    RUNTIME_WEIGHT_SLOT_COUNT,
    SUPPORTED_RESOLUTIONS,
)
from .weight_contract import LoweredWeight

# This module owns the writer's wire-format constants; model_contract.h mirrors
# them in the C++ reader. Native validation exercises the complete writer/reader
# boundary for every supported resolution.
#
# The magic and version let the runtime reject unrelated or incompatible model
# files before interpreting offsets. Both are checked by ModelStore::Load.
_MAGIC = b"Y26MOD01"
_VERSION = 1

# XNNPACK uses 64-byte-aligned static buffers. The header occupies one aligned
# block. Directory position identifies the semantic slot, so each entry needs
# only the payload offset and byte count.
_ALIGNMENT = 64
_HEADER_BYTES = _ALIGNMENT
_ENTRY_BYTES = 16
_HEADER = struct.Struct("<8sIIIIQ")
_ENTRY = struct.Struct("<QQ")
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1
_FP32_BYTES = 4
_CLASSIFICATION_CHANNEL_FLOOR = 64


@dataclass(frozen=True)
class RuntimeModelHeader:
    """Validated header fields from the runtime's serialized ``model.bin``."""

    resolution: int
    class_count: int
    slot_count: int
    payload_offset: int


@dataclass(frozen=True)
class _ModelWeight:
    slot: int
    name: str
    payload: bytes

    def __post_init__(self) -> None:
        if (
            type(self.slot) is not int
            or self.slot < 0
            or not self.name
            or not isinstance(self.payload, bytes)
            or not self.payload
        ):
            raise ValueError("invalid model weight")


def _build_model_weight_table(
    derived_weights: tuple[LoweredWeight, ...],
) -> tuple[_ModelWeight, ...]:
    """Build the runtime's stable table from its closed weight inventory."""
    named = []
    for weight in derived_weights:
        named.append((weight.binding.name, weight.payload))
    named.sort(key=lambda row: row[0])
    names = [name for name, _ in named]
    if len(names) != len(set(names)):
        raise ValueError("semantic model weight slots are not unique")
    if tuple(names) != RUNTIME_WEIGHT_NAMES:
        missing = sorted(set(RUNTIME_WEIGHT_NAMES) - set(names))
        unexpected = sorted(set(names) - set(RUNTIME_WEIGHT_NAMES))
        raise ValueError(
            "lowered weights do not match the fixed runtime inventory: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    weights = tuple(_ModelWeight(slot, name, payload) for slot, (name, payload) in enumerate(named))
    return weights


def _round_up_to_alignment(value: int) -> int:
    """Round a byte offset up to the next 64-byte model-file boundary."""
    return ((value + _ALIGNMENT - 1) // _ALIGNMENT) * _ALIGNMENT


def _class_dependent_weight_bytes_fit(class_count: int) -> bool:
    """Return whether the target runtime can size its largest learned tensor."""
    width = max(_CLASSIFICATION_CHANNEL_FLOOR, class_count)
    return (
        0 < class_count <= _UINT32_MAX - BOX_COORDINATE_CHANNELS
        and class_count <= _UINT64_MAX // _FP32_BYTES // width
    )


def read_runtime_model_header(payload: bytes) -> RuntimeModelHeader:
    """Read and validate the fixed header of an exported runtime ``model.bin``."""
    if len(payload) < _HEADER_BYTES:
        raise ValueError("model file is smaller than its required 64-byte header")
    (
        magic,
        version,
        resolution,
        class_count,
        slot_count,
        payload_offset,
    ) = _HEADER.unpack_from(payload)
    maximum_directory_entries = (len(payload) - _HEADER_BYTES) // _ENTRY_BYTES
    expected_payload_offset = _round_up_to_alignment(_HEADER_BYTES + _ENTRY_BYTES * slot_count)
    if (
        magic != _MAGIC
        or version != _VERSION
        or resolution not in SUPPORTED_RESOLUTIONS
        or not _class_dependent_weight_bytes_fit(class_count)
        or slot_count == 0
        or slot_count != RUNTIME_WEIGHT_SLOT_COUNT
        or slot_count > maximum_directory_entries
        or payload_offset != expected_payload_offset
        or payload[_HEADER.size : _HEADER_BYTES] != bytes(_HEADER_BYTES - _HEADER.size)
    ):
        raise ValueError("model header does not match the fixed runtime format")
    return RuntimeModelHeader(
        resolution=resolution,
        class_count=class_count,
        slot_count=slot_count,
        payload_offset=payload_offset,
    )


def _pack_model_weights(
    weights: tuple[_ModelWeight, ...],
    resolution: int,
    class_count: int,
) -> bytes:
    """Pack validated slot records into the binary model container."""
    if len(weights) != RUNTIME_WEIGHT_SLOT_COUNT or [weight.slot for weight in weights] != list(
        range(RUNTIME_WEIGHT_SLOT_COUNT)
    ):
        raise ValueError("model weight slots must be complete and contiguous")
    if len({weight.name for weight in weights}) != len(weights):
        raise ValueError("model weight names must be unique")
    if (
        resolution not in SUPPORTED_RESOLUTIONS
        or type(class_count) is not int
        or not _class_dependent_weight_bytes_fit(class_count)
    ):
        raise ValueError("invalid model dimensions")

    directory_offset = _HEADER_BYTES
    payload_offset = _round_up_to_alignment(directory_offset + _ENTRY_BYTES * len(weights))
    packed = bytearray(payload_offset)
    for weight in weights:
        while len(packed) % _ALIGNMENT:
            packed.append(0)
        offset = len(packed)
        packed.extend(weight.payload)
        _ENTRY.pack_into(
            packed,
            directory_offset + weight.slot * _ENTRY_BYTES,
            offset,
            len(weight.payload),
        )

    _HEADER.pack_into(
        packed,
        0,
        _MAGIC,
        _VERSION,
        resolution,
        class_count,
        len(weights),
        payload_offset,
    )
    return bytes(packed)


def pack_model(
    lowered_weights: tuple[LoweredWeight, ...],
    resolution: int,
    class_count: int,
) -> bytes:
    """Validate stable weight identities and serialize the runtime model file."""
    return _pack_model_weights(
        _build_model_weight_table(lowered_weights),
        resolution,
        class_count,
    )

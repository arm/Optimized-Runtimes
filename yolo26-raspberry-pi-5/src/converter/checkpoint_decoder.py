"""Decode a Torch ZIP/pickle checkpoint without importing its model code."""

from __future__ import annotations

import collections
import io
import math
import pickle
import pickletools
import struct
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Never

from .converter_io import RejectedInput

# Torch's ZIP format can use several pickle protocols for the same object
# graph. Accept their passive container and memoization opcodes; find_class
# below independently restricts every callable referenced by the stream.
_ALLOWED_OPCODES = {
    "ADDITEMS",
    "APPEND",
    "APPENDS",
    "BINBYTES",
    "BINBYTES8",
    "BINFLOAT",
    "BINGET",
    "BININT",
    "BININT1",
    "BININT2",
    "BINPERSID",
    "BINPUT",
    "BINUNICODE",
    "BUILD",
    "EMPTY_DICT",
    "EMPTY_LIST",
    "EMPTY_SET",
    "EMPTY_TUPLE",
    "FRAME",
    "GLOBAL",
    "LONG_BINGET",
    "LONG_BINPUT",
    "LONG1",
    "LONG4",
    "MARK",
    "MEMOIZE",
    "NEWFALSE",
    "NEWOBJ",
    "NEWOBJ_EX",
    "NEWTRUE",
    "NONE",
    "PROTO",
    "REDUCE",
    "SETITEM",
    "SETITEMS",
    "FROZENSET",
    "SHORT_BINBYTES",
    "SHORT_BINUNICODE",
    "STACK_GLOBAL",
    "STOP",
    "TUPLE",
    "TUPLE1",
    "TUPLE2",
    "TUPLE3",
}
# Format code and byte width for the only tensor dtypes consumed by validation
# and weight lowering. Unsupported storage types cannot become DecodedTensor.
_STORAGE_FORMATS = {
    "torch.HalfStorage": ("float16", 2, "e"),
    "torch.FloatStorage": ("float32", 4, "f"),
    "torch.LongStorage": ("int64", 8, "q"),
}

# Torch stores tensor bytes under this directory. Other archive metadata is
# deliberately ignored: it does not affect the tensors exported to model.bin.
_STORAGE_DIRECTORY = "data"

type _EmptyHooks = collections.OrderedDict[Never, Never]


class DecodedModule:
    """Attribute container for an allowlisted model class, with no executable code."""

    _global = ""

    def __new__(cls, *_args: object, **_kwargs: object) -> DecodedModule:
        """Accept constructor data from pickle without retaining or executing it."""
        return super().__new__(cls)

    def __getattr__(self, name: str) -> object:
        raise AttributeError(name)


@dataclass
class _Storage:
    """One copied Torch storage payload and the dtype needed to decode it."""

    kind: str
    elements: int
    raw: bytes


@dataclass
class DecodedTensor:
    """A validated strided tensor view into copied checkpoint storage."""

    storage: _Storage
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]

    @property
    def dtype(self) -> str:
        return _STORAGE_FORMATS[self.storage.kind][0]

    @property
    def elements(self) -> int:
        return math.prod(self.shape) if self.shape else 1

    def contiguous_bytes(self) -> bytes:
        """Materialize this tensor view after validating every storage index.

        Torch tensors may reference a strided view into a shared storage file.
        The exporter needs packed tensor bytes, so this method reconstructs the
        view and rejects malformed metadata before reading outside that file.
        """
        _, width, _ = _STORAGE_FORMATS[self.storage.kind]
        if len(self.storage.raw) != self.storage.elements * width:
            raise RejectedInput("storage payload does not match its declared extent")
        if any(not isinstance(v, int) or v < 0 for v in self.shape + self.stride):
            raise RejectedInput("negative or non-integer tensor metadata")
        if self.elements == 0:
            if not 0 <= self.storage_offset <= self.storage.elements:
                raise RejectedInput("tensor view exceeds its declared storage")
            return b""
        indices = [self.storage_offset]
        for extent, step in zip(self.shape, self.stride, strict=True):
            indices = [base + index * step for base in indices for index in range(extent)]
        if not self.shape:
            indices = [self.storage_offset]
        if len(indices) != self.elements or not indices or max(indices) >= self.storage.elements:
            raise RejectedInput("tensor view exceeds its declared storage")
        return b"".join(self.storage.raw[index * width : (index + 1) * width] for index in indices)

    def values(self) -> list[float | int]:
        _, width, code = _STORAGE_FORMATS[self.storage.kind]
        raw = self.contiguous_bytes()
        return [row[0] for row in struct.iter_unpack("<" + code, raw)]


def _validate_tensor_extent(value: tuple[int, ...], label: str) -> None:
    if not isinstance(value, tuple) or any(type(item) is not int or item < 0 for item in value):
        raise RejectedInput(
            f"tensor {label} must be a tuple of non-negative integers; received {value!r}"
        )


class _CheckpointUnpickler(pickle.Unpickler):
    """Decode only data required to inspect a YOLO26n checkpoint.

    Model classes are represented by empty local objects. No Torch or
    Ultralytics class is imported or constructed while reading the checkpoint.
    """

    def __init__(self, stream: io.BytesIO, archive: zipfile.ZipFile, root: str) -> None:
        super().__init__(stream)
        self.archive = archive
        self.root = root
        self.classes: dict[str, type[DecodedModule]] = {}
        self.storages: dict[str, _Storage] = {}

    def _inert_class(self, name: str) -> type[DecodedModule]:
        if name not in self.classes:
            self.classes[name] = type(name.replace(".", "_"), (DecodedModule,), {"_global": name})
        return self.classes[name]

    def find_class(self, module: str, name: str) -> object:
        """Replace every pickle global with a local passive equivalent."""
        global_name = module + "." + name
        if global_name in {"__builtin__.set", "builtins.set"}:
            return set
        if global_name == "collections.OrderedDict":
            return collections.OrderedDict
        if global_name == "torch._utils._rebuild_tensor_v2":
            return self._rebuild_tensor
        if global_name == "torch._utils._rebuild_parameter":
            return self._rebuild_parameter
        return self._inert_class(global_name)

    def persistent_load(self, value: object) -> _Storage:
        """Load one validated Torch storage record from the owning ZIP archive."""
        if not isinstance(value, tuple) or len(value) != 5 or value[0] != "storage":
            raise RejectedInput(
                f"Torch persistent ID must be a five-item storage record; received {value!r}"
            )
        _, storage_class, key, location, elements = value
        kind = getattr(storage_class, "_global", None)
        if (
            kind not in _STORAGE_FORMATS
            or not isinstance(key, str)
            or not key
            or PurePosixPath(key).name != key
            or "\\" in key
        ):
            raise RejectedInput(
                "storage record requires a supported Torch dtype and a simple key; "
                f"received dtype={kind!r}, key={key!r}"
            )
        if not isinstance(location, str) or type(elements) is not int or elements < 0:
            raise RejectedInput(
                "storage record requires a named source location and a non-negative extent; "
                f"location={location!r}, elements={elements!r}"
            )
        existing = self.storages.get(key)
        if existing is not None:
            if existing.kind != kind or existing.elements != elements:
                raise RejectedInput(f"conflicting metadata for shared storage: {key}")
            return existing
        member = f"{self.root}/{_STORAGE_DIRECTORY}/{key}"
        matches = [info for info in self.archive.infolist() if info.filename == member]
        if len(matches) != 1:
            raise RejectedInput(f"storage {key} must have one ZIP member; found {len(matches)}")
        info = matches[0]
        expected = elements * _STORAGE_FORMATS[kind][1]
        if info.file_size != expected:
            raise RejectedInput(
                f"storage extent mismatch for {key}: {info.file_size} != {expected}"
            )
        storage = _Storage(kind, elements, self.archive.read(info))
        self.storages[key] = storage
        return storage

    @staticmethod
    def _rebuild_tensor(
        storage: _Storage,
        offset: int,
        shape: tuple[int, ...],
        stride: tuple[int, ...],
        requires_grad: bool,
        hooks: _EmptyHooks,
    ) -> DecodedTensor:
        """Reconstruct a tensor from the metadata required by Torch's pickle protocol."""
        if not isinstance(storage, _Storage):
            raise RejectedInput(
                f"tensor storage must be a decoded Torch storage; received {type(storage).__name__}"
            )
        if type(offset) is not int or offset < 0:
            raise RejectedInput(
                f"tensor storage offset must be a non-negative integer; received {offset!r}"
            )
        _validate_tensor_extent(shape, "shape")
        _validate_tensor_extent(stride, "stride")
        if len(shape) != len(stride):
            raise RejectedInput(
                "tensor shape and stride must have equal rank; "
                f"received {len(shape)} and {len(stride)}"
            )
        if type(requires_grad) is not bool:
            raise RejectedInput(
                f"tensor requires_grad must be a boolean; received {requires_grad!r}"
            )
        if not isinstance(hooks, collections.OrderedDict) or hooks:
            raise RejectedInput("tensor hooks must be an empty OrderedDict")
        tensor = DecodedTensor(storage, offset, shape, stride)
        tensor.contiguous_bytes()
        return tensor

    @staticmethod
    def _rebuild_parameter(
        tensor: DecodedTensor,
        requires_grad: bool,
        hooks: _EmptyHooks,
    ) -> DecodedTensor:
        if not isinstance(tensor, DecodedTensor):
            raise RejectedInput(
                f"parameter rebuild requires a decoded tensor; received {type(tensor).__name__}"
            )
        if type(requires_grad) is not bool:
            raise RejectedInput(
                f"parameter requires_grad must be a boolean; received {requires_grad!r}"
            )
        if not isinstance(hooks, collections.OrderedDict) or hooks:
            raise RejectedInput("parameter hooks must be an empty OrderedDict")
        return tensor


def _validate_pickle(data: bytes) -> None:
    """Preflight the pickle program before allowing the inert decoder to run.

    Pickle normally permits arbitrary constructor calls. The supported
    checkpoint needs only passive object, container, and memoization opcodes;
    every referenced global is independently replaced by find_class above.
    """
    try:
        operations = list(pickletools.genops(data))
    except Exception as exc:
        raise RejectedInput(f"invalid pickle stream: {exc}") from exc
    for operation, _argument, _ in operations:
        if operation.name not in _ALLOWED_OPCODES:
            raise RejectedInput(f"unsupported pickle opcode: {operation.name}")


def _checkpoint_root(archive: zipfile.ZipFile) -> str:
    """Find an unambiguous Torch payload while ignoring unrelated members."""
    payloads = []
    for info in archive.infolist():
        parts = PurePosixPath(info.filename).parts
        if (
            len(parts) == 2
            and parts[1] == "data.pkl"
            and parts[0] not in {"", ".", "..", "/"}
            and info.filename == f"{parts[0]}/data.pkl"
        ):
            payloads.append(parts[0])
    if len(payloads) != 1:
        raise RejectedInput(
            "checkpoint ZIP must contain one unambiguous Torch data.pkl payload; "
            f"found {len(payloads)}"
        )
    return payloads[0]


def _pickle_payload(archive: zipfile.ZipFile, root: str) -> bytes:
    """Read metadata needed by this decoder and ignore unrelated Torch records."""
    required = {}
    for name in ("byteorder", "data.pkl"):
        member = f"{root}/{name}"
        matches = [info for info in archive.infolist() if info.filename == member]
        if len(matches) != 1:
            raise RejectedInput(f"checkpoint requires one {name} member; found {len(matches)}")
        required[name] = archive.read(matches[0])
    byteorder = required["byteorder"]
    if byteorder.strip() != b"little":
        raise RejectedInput(
            "checkpoint tensor storage must be little-endian for this exporter; "
            f"received {byteorder!r}"
        )
    return required["data.pkl"]


def _decode_pickle(data: bytes, archive: zipfile.ZipFile, root: str) -> object:
    _validate_pickle(data)
    unpickler = _CheckpointUnpickler(io.BytesIO(data), archive, root)
    try:
        return unpickler.load()
    except RejectedInput:
        raise
    except Exception as exc:
        raise RejectedInput(f"inert pickle decode failed: {exc}") from exc


def decode_checkpoint_archive(snapshot: bytes) -> object:
    """Decode the inert Torch object graph and copy all referenced tensor bytes."""
    try:
        with zipfile.ZipFile(io.BytesIO(snapshot)) as archive:
            root = _checkpoint_root(archive)
            return _decode_pickle(_pickle_payload(archive, root), archive, root)
    except (OSError, zipfile.BadZipFile) as exc:
        raise RejectedInput(f"not a supported ZIP: {exc}") from exc

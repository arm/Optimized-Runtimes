#!/usr/bin/env python3
"""Fuse and layout-convert validated YOLO26n learned tensors."""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import struct
from dataclasses import dataclass
from typing import cast

from .converter_io import RejectedInput
from .model_contract import (
    BOX_COORDINATE_CHANNELS,
    DETECTION_BRANCH_COUNT,
    SPLIT_PROJECTION_CHANNEL_RANGES,
    SPLIT_PROJECTION_ZERO_BIAS_CHANNELS,
    BatchNormFusion,
    ConvolutionSpec,
    LearnedTensor,
    ValidatedYolo26n,
)
from .weight_contract import (
    CheckpointTensor,
    LoweredWeight,
    SplitProjectionWeight,
    WeightBinding,
    WeightDerivation,
    WeightLayout,
    ZeroBias,
    weight_derivation_name,
)

# The fixed runtime evaluates these 1x1 projections before their two inputs are
# concatenated. Splitting the original weights at the input-channel boundary
# makes the two operations mathematically equivalent to the checkpoint
# convolution. These ranges are the channel widths of the paired inputs.
_SPLIT_PROJECTION_WEIGHTS = tuple(
    SplitProjectionWeight(source, (begin, end))
    for source, begin, end in SPLIT_PROJECTION_CHANNEL_RANGES
)

# XNNPACK's two independent projections each need a bias. Their source
# convolutions have no learned bias, so the exporter supplies exact zeros at
# the corresponding output-channel widths.
_PROJECTION_ZERO_BIASES = tuple(
    ZeroBias(channels) for channels in SPLIT_PROJECTION_ZERO_BIAS_CHANNELS
)
_ONE_TO_ONE_HEAD = ".one2one_"

# BatchNorm fusion rounds every primitive operation to binary32. Python's
# math.sqrt accepts and returns binary64, whereas C `sqrtf` rounds its result
# directly to binary32. Calling the platform C library gives square root the
# same binary32 boundary as the other fusion operations.
# `find_library` resolves the platform SONAME instead of assuming libm.so.6.
_libm_name = ctypes.util.find_library("m")
if _libm_name is None:
    raise RuntimeError("the exporter requires a C math library with sqrtf")
_libm = ctypes.CDLL(_libm_name)
_libm.sqrtf.argtypes = [ctypes.c_float]
_libm.sqrtf.restype = ctypes.c_float


@dataclass(frozen=True)
class _EffectiveTensor:
    shape: tuple[int, ...]
    values: tuple[float, ...]


def _float32(value: float) -> float:
    """Round one Python binary64 value to the exporter's binary32 contract."""
    return cast(float, struct.unpack("<f", struct.pack("<f", value))[0])


def _float32_add(a: float, b: float) -> float:
    return _float32(float(a) + float(b))


def _float32_subtract(a: float, b: float) -> float:
    return _float32(float(a) - float(b))


def _float32_multiply(a: float, b: float) -> float:
    return _float32(float(a) * float(b))


def _float32_divide(a: float, b: float) -> float:
    return _float32(float(a) / float(b))


def _float32_sqrt(a: float) -> float:
    """Evaluate C sqrtf rather than Python's binary64 math.sqrt."""
    return float(_libm.sqrtf(ctypes.c_float(a)))


def _quantize_to_float16(value: float) -> float:
    """Apply the FP16 quantization used by the original fused-weight path."""
    return cast(float, struct.unpack("<e", struct.pack("<e", _float32(value)))[0])


def _tensor_float32_values(
    tensor: LearnedTensor, expected_shape: tuple[int, ...], label: str
) -> list[float]:
    """Decode one FP16 checkpoint tensor and reject the wrong role or shape."""
    if tensor.dtype != "float16" or tensor.shape != expected_shape:
        raise RejectedInput(f"invalid tensor dtype/shape: {label}")
    values = [_float32(value[0]) for value in struct.iter_unpack("<e", tensor.data)]
    if not all(math.isfinite(value) for value in values):
        raise RejectedInput(f"non-finite tensor: {label}")
    return values


def _convolution_weight_shape(spec: ConvolutionSpec) -> tuple[int, int, int, int]:
    return (spec.output_channels, spec.input_channels // spec.groups, *spec.kernel)


def _convolution_bias(
    state: dict[str, LearnedTensor],
    bias_name: str,
    output_channels: int,
) -> tuple[list[float], set[str]]:
    """Read an optional learned bias, using zeros for bias-free convolutions."""
    tensor = state.get(bias_name)
    if tensor is None:
        return [0.0] * output_channels, set()
    return _tensor_float32_values(tensor, (output_channels,), bias_name), {bias_name}


def _batch_norm_values(
    state: dict[str, LearnedTensor],
    wrapper_path: str,
    output_channels: int,
) -> tuple[list[float], list[float], list[float], list[float], set[str]]:
    """Read the four vectors needed for inference-time BatchNorm folding.

    `num_batches_tracked` affects training only. It is still required and
    accounted for so an unexpected learned-state inventory cannot be silently
    discarded during export.
    """
    names = {
        name: wrapper_path + ".bn." + name
        for name in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")
    }
    if any(name not in state for name in names.values()):
        raise RejectedInput(f"missing BatchNorm state: {wrapper_path}")
    shape = (output_channels,)
    gamma = _tensor_float32_values(state[names["weight"]], shape, names["weight"])
    beta = _tensor_float32_values(state[names["bias"]], shape, names["bias"])
    mean = _tensor_float32_values(state[names["running_mean"]], shape, names["running_mean"])
    variance = _tensor_float32_values(state[names["running_var"]], shape, names["running_var"])
    tracked = state[names["num_batches_tracked"]]
    if tracked.dtype != "int64" or tracked.shape != ():
        raise RejectedInput(f"invalid BatchNorm state: {wrapper_path}")
    return gamma, beta, mean, variance, set(names.values())


def _fuse_batch_norm(
    fusion: BatchNormFusion,
    model: ValidatedYolo26n,
    state: dict[str, LearnedTensor],
) -> tuple[str, _EffectiveTensor, str, _EffectiveTensor, set[str]]:
    """Fold one convolution and BatchNorm using the specified FP32 order.

    For each output channel, scale = gamma / sqrt(variance + epsilon), then
    fused_weight = weight * scale and fused_bias = bias * scale + beta -
    gamma * mean / sqrt(variance + epsilon). Each primitive operation is
    rounded to binary32 before its result feeds the next operation; changing
    this order changes model.bin and can change inference results.
    """
    path = fusion.convolution_path
    spec = model.convolution(path)
    if spec.groups not in {1, spec.input_channels} or (
        spec.groups != 1 and spec.output_channels != spec.input_channels
    ):
        raise RejectedInput(f"unsupported convolution grouping: {path}")
    weight_name = path + ".weight"
    bias_name = path + ".bias"
    weight_shape = _convolution_weight_shape(spec)
    weight_tensor = state.get(weight_name)
    if weight_tensor is None:
        raise RejectedInput(f"missing convolution weight: {weight_name}")
    convolution_weight = _tensor_float32_values(weight_tensor, weight_shape, weight_name)
    bias, consumed = _convolution_bias(state, bias_name, spec.output_channels)
    gamma, beta, mean, variance, batch_norm_names = _batch_norm_values(
        state, path.removesuffix(".conv"), spec.output_channels
    )
    scales: list[float] = []
    fused_bias: list[float] = []
    for index in range(spec.output_channels):
        root = _float32_sqrt(_float32_add(variance[index], _float32(fusion.epsilon)))
        if not math.isfinite(root) or root <= 0.0:
            raise RejectedInput(f"invalid BatchNorm statistics: {path.removesuffix('.conv')}")
        scale = _float32_divide(gamma[index], root)
        batch_norm_bias = _float32_subtract(
            beta[index],
            _float32_divide(_float32_multiply(gamma[index], mean[index]), root),
        )
        scales.append(scale)
        fused_bias.append(_float32_add(_float32_multiply(scale, bias[index]), batch_norm_bias))
    per_channel = math.prod(weight_shape[1:])
    fused_weight = [
        _float32_multiply(value, scales[index // per_channel])
        for index, value in enumerate(convolution_weight)
    ]
    consumed.update(batch_norm_names)
    consumed.add(weight_name)
    return (
        weight_name,
        _EffectiveTensor(weight_shape, tuple(fused_weight)),
        bias_name,
        _EffectiveTensor((spec.output_channels,), tuple(fused_bias)),
        consumed,
    )


def _read_direct_convolution_parameter(
    name: str,
    tensor: LearnedTensor,
    model: ValidatedYolo26n,
) -> _EffectiveTensor:
    """Validate and copy a convolution parameter that needs no BatchNorm fold."""
    if tensor.dtype != "float16" or not (name.endswith(".weight") or name.endswith(".bias")):
        raise RejectedInput(f"unsupported direct learned tensor: {name}")
    owner_name, role = name.rsplit(".", 1)
    try:
        owner = model.convolution(owner_name)
    except ValueError:
        raise RejectedInput(f"unsupported direct tensor owner: {name}") from None
    expected = _convolution_weight_shape(owner) if role == "weight" else (owner.output_channels,)
    return _EffectiveTensor(expected, tuple(_tensor_float32_values(tensor, expected, name)))


def _validate_detection_head(output: dict[str, _EffectiveTensor], model: ValidatedYolo26n) -> None:
    """Confirm class-derived output convolutions exist at all three scales."""
    for branch in range(DETECTION_BRANCH_COUNT):
        for prefix, expected_out in (
            (f"model.23.cv2.{branch}.2", BOX_COORDINATE_CHANNELS),
            (f"model.23.cv3.{branch}.2", model.class_count),
        ):
            if model.convolution(prefix).output_channels != expected_out:
                raise RejectedInput("class-derived detection-head schema mismatch")
            if any(prefix + "." + role not in output for role in ("weight", "bias")):
                raise RejectedInput("class-derived detection-head tensor missing")


def _one_to_one_head_state_names(model: ValidatedYolo26n) -> set[str]:
    """List learned state owned by the intentionally unused head branch."""
    convolution_state = {
        f"{spec.path}.{role}"
        for spec in model.convolutions
        if _ONE_TO_ONE_HEAD in spec.path
        for role in ("weight", "bias")
    }
    batch_norm_state = {
        f"{fusion.convolution_path.removesuffix('.conv')}.bn.{role}"
        for fusion in model.batch_norm_fusions
        if _ONE_TO_ONE_HEAD in fusion.convolution_path
        for role in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")
    }
    return convolution_state | batch_norm_state


def _fold_batch_norm(model: ValidatedYolo26n) -> dict[str, _EffectiveTensor]:
    """Create the exact convolution tensors consumed by the fixed runtime.

    The runtime implements the one-to-many detection path. Ultralytics also
    stores a training/inference one-to-one head in the same checkpoint; those
    tensors do not feed this runtime's output and are intentionally ignored.
    Every other learned tensor must be consumed by a folded BatchNorm group or
    by one unfused convolution.
    """
    state = {row.name: row for row in model.learned_tensors}
    output: dict[str, _EffectiveTensor] = {}
    consumed: set[str] = set()
    ignored_one_to_one_state = _one_to_one_head_state_names(model)
    for fusion in model.batch_norm_fusions:
        if _ONE_TO_ONE_HEAD in fusion.convolution_path:
            continue
        weight_name, weight, bias_name, bias, fusion_consumed = _fuse_batch_norm(
            fusion, model, state
        )
        output[weight_name] = weight
        output[bias_name] = bias
        consumed.update(fusion_consumed)

    for name, tensor in state.items():
        if name in consumed:
            continue
        if _ONE_TO_ONE_HEAD in name:
            if name not in ignored_one_to_one_state:
                raise RejectedInput(f"unexpected one-to-one head tensor: {name}")
            continue
        output[name] = _read_direct_convolution_parameter(name, tensor, model)

    _validate_detection_head(output, model)
    return output


def _xnnpack_weight_bytes(tensor: _EffectiveTensor, layout: WeightLayout) -> bytes:
    """Quantize and arrange one tensor for XNNPACK's static-weight ABI.

    Checkpoints use OIHW convolution order. XNNPACK consumes regular kernels as
    OHWI and depthwise kernels as multiplier-height-width-channel. Biases remain
    in their original one-dimensional order. Each value is first quantized to
    FP16, which is the runtime weight-precision contract, then stored as FP32
    because these XNNPACK graph values use FP32 static data.
    """
    shape = tensor.shape
    values = [_quantize_to_float16(value) for value in tensor.values]
    if layout is WeightLayout.IDENTITY:
        ordered = values
    elif layout is WeightLayout.CONVOLUTION:
        o, i, h, w = shape
        ordered = [
            values[((oo * i + ii) * h + hh) * w + ww]
            for oo in range(o)
            for hh in range(h)
            for ww in range(w)
            for ii in range(i)
        ]
    elif layout is WeightLayout.DEPTHWISE_CONVOLUTION:
        o, multiplier, h, w = shape
        ordered = [
            values[((oo * multiplier + mm) * h + hh) * w + ww]
            for mm in range(multiplier)
            for hh in range(h)
            for ww in range(w)
            for oo in range(o)
        ]
    else:
        raise RejectedInput(f"unsupported XNNPACK weight layout: {layout!r}")
    return b"".join(struct.pack("<f", value) for value in ordered)


def _lower_weight(payload: bytes, derivation: WeightDerivation) -> LoweredWeight:
    """Attach one converted payload to its stable semantic identity."""
    name = weight_derivation_name(derivation)
    return LoweredWeight(WeightBinding(name, len(payload), derivation), payload)


def _layout_for_convolution_parameter(
    name: str,
    tensor: _EffectiveTensor,
    model: ValidatedYolo26n,
) -> WeightLayout:
    """Select the XNNPACK storage order for one effective model parameter."""
    if name.endswith(".weight") and len(tensor.shape) == 4:
        owner = model.convolution(name.removesuffix(".weight"))
        if owner.groups == 1:
            return WeightLayout.CONVOLUTION
        if owner.groups == owner.input_channels == owner.output_channels:
            return WeightLayout.DEPTHWISE_CONVOLUTION
        raise RejectedInput(f"unsupported exported grouping: {name}")
    if name.endswith(".bias") and len(tensor.shape) == 1:
        return WeightLayout.IDENTITY
    raise RejectedInput(f"unsupported effective tensor role: {name}")


def _lower_checkpoint_weights(
    effective: dict[str, _EffectiveTensor],
    model: ValidatedYolo26n,
) -> tuple[LoweredWeight, ...]:
    """Lower every effective parameter not consumed by split projection."""
    split_sources = {row.source for row in _SPLIT_PROJECTION_WEIGHTS}
    weights: list[LoweredWeight] = []
    for name, tensor in sorted(effective.items()):
        if name in split_sources:
            continue
        layout = _layout_for_convolution_parameter(name, tensor, model)
        weights.append(
            _lower_weight(
                _xnnpack_weight_bytes(tensor, layout),
                CheckpointTensor(source=name, source_shape=tensor.shape, layout=layout),
            )
        )
    return tuple(weights)


def _extract_split_projection_weight(
    effective: dict[str, _EffectiveTensor], derivation: SplitProjectionWeight
) -> _EffectiveTensor:
    """Extract one input-channel range from a checkpoint's 1x1 projection."""
    source = derivation.source
    begin, end = derivation.input_channel_slice
    sliced_source = effective.get(source)
    if (
        sliced_source is None
        or len(sliced_source.shape) != 4
        or sliced_source.shape[2:] != (1, 1)
        or not 0 <= begin < end <= sliced_source.shape[1]
    ):
        raise RejectedInput(f"projection slice bounds changed: {source}")
    output_channels, input_channels, _, _ = sliced_source.shape
    values = tuple(
        sliced_source.values[output * input_channels + input_channel]
        for output in range(output_channels)
        for input_channel in range(begin, end)
    )
    return _EffectiveTensor((output_channels, end - begin, 1, 1), values)


def _lower_split_projection_weights(
    effective: dict[str, _EffectiveTensor],
) -> tuple[LoweredWeight, ...]:
    """Lower the four weight ranges consumed by split runtime projections."""
    return tuple(
        _lower_weight(
            _xnnpack_weight_bytes(
                _extract_split_projection_weight(effective, derivation),
                WeightLayout.CONVOLUTION,
            ),
            derivation,
        )
        for derivation in _SPLIT_PROJECTION_WEIGHTS
    )


def _lower_projection_zero_biases() -> tuple[LoweredWeight, ...]:
    """Create the zero biases required by the two bias-free projections."""
    return tuple(
        _lower_weight(bytes(derivation.channels * 4), derivation)
        for derivation in _PROJECTION_ZERO_BIASES
    )


def lower_yolo26n_weights(model: ValidatedYolo26n) -> tuple[LoweredWeight, ...]:
    """Fold and serialize the complete weight inventory for the fixed runtime."""
    effective = _fold_batch_norm(model)
    weights = (
        *_lower_checkpoint_weights(effective, model),
        *_lower_split_projection_weights(effective),
        *_lower_projection_zero_biases(),
    )
    names = [weight.binding.name for weight in weights]
    if len(names) != len(set(names)):
        raise RejectedInput("lowering produced duplicate semantic model weights")
    return weights

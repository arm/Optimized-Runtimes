"""Validate the decoded inert object graph as the supported YOLO26n model."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .checkpoint_decoder import DecodedModule, DecodedTensor
from .converter_io import RejectedInput
from .model_contract import (
    DETECTION_BRANCH_COUNT,
    DETECTION_STRIDES,
    RUNTIME_CONVOLUTION_PATHS,
    AttentionSpec,
    BatchNormFusion,
    ConvolutionActivation,
    ConvolutionSpec,
    LearnedTensor,
    ValidatedYolo26n,
    classification_width_for_classes,
    detection_output_channels,
)

_CLASSIFICATION_HIDDEN_CONVOLUTION = "model.23.cv3.{branch}.0.1.conv"
_SPPF_MAX_POOL_PATH = "model.9.m"
_NEAREST_UPSAMPLE_PATHS = ("model.11", "model.14")
_CHANNEL_CONCATENATION_PATHS = ("model.12", "model.15", "model.18", "model.21")
_ATTENTION_PATHS = ("model.10.m.0.attn", "model.22.m.0.1.attn")
_ATTENTION_RESIDUAL_PATHS = ("model.10.m.0", "model.22.m.0.1")
# These are precisely the Bottleneck instances whose shortcut output is added
# by fixed_graph.cpp. Other Bottleneck instances in this model concatenate or
# forward their convolution result without that addition.
_RESIDUAL_BOTTLENECK_PATHS = (
    "model.2.m.0",
    "model.4.m.0",
    "model.6.m.0.m.0",
    "model.6.m.0.m.1",
    "model.8.m.0.m.0",
    "model.8.m.0.m.1",
    "model.13.m.0.m.0",
    "model.13.m.0.m.1",
    "model.16.m.0.m.0",
    "model.16.m.0.m.1",
    "model.19.m.0.m.0",
    "model.19.m.0.m.1",
    "model.22.m.0.0",
)


@dataclass(frozen=True)
class _ValidatedTopology:
    convolutions: tuple[ConvolutionSpec, ...]
    batch_norm_fusions: tuple[BatchNormFusion, ...]
    attention_blocks: tuple[AttentionSpec, ...]


# Each row is (module index, input source, operator kind). This is the top-level
# routing implemented by fixed_graph.cpp; nested convolution geometry and the
# remaining parameter-free operations are validated separately below.
_MODULE_GRAPH = (
    (0, -1, "ultralytics.nn.modules.conv.Conv"),
    (1, -1, "ultralytics.nn.modules.conv.Conv"),
    (2, -1, "ultralytics.nn.modules.block.C3k2"),
    (3, -1, "ultralytics.nn.modules.conv.Conv"),
    (4, -1, "ultralytics.nn.modules.block.C3k2"),
    (5, -1, "ultralytics.nn.modules.conv.Conv"),
    (6, -1, "ultralytics.nn.modules.block.C3k2"),
    (7, -1, "ultralytics.nn.modules.conv.Conv"),
    (8, -1, "ultralytics.nn.modules.block.C3k2"),
    (9, -1, "ultralytics.nn.modules.block.SPPF"),
    (10, -1, "ultralytics.nn.modules.block.C2PSA"),
    (11, -1, "torch.nn.modules.upsampling.Upsample"),
    (12, [-1, 6], "ultralytics.nn.modules.conv.Concat"),
    (13, -1, "ultralytics.nn.modules.block.C3k2"),
    (14, -1, "torch.nn.modules.upsampling.Upsample"),
    (15, [-1, 4], "ultralytics.nn.modules.conv.Concat"),
    (16, -1, "ultralytics.nn.modules.block.C3k2"),
    (17, -1, "ultralytics.nn.modules.conv.Conv"),
    (18, [-1, 13], "ultralytics.nn.modules.conv.Concat"),
    (19, -1, "ultralytics.nn.modules.block.C3k2"),
    (20, -1, "ultralytics.nn.modules.conv.Conv"),
    (21, [-1, 10], "ultralytics.nn.modules.conv.Concat"),
    (22, -1, "ultralytics.nn.modules.block.C3k2"),
    (23, [16, 19, 22], "ultralytics.nn.modules.head.Detect"),
)


def _module_children(module: DecodedModule) -> dict[str, DecodedModule | None]:
    """Read Torch's child-module mapping as decoded inert module values."""
    children = getattr(module, "_modules", None)
    if not isinstance(children, dict):
        raise RejectedInput(
            "module _modules must be a mapping in the supported YOLO26n checkpoint; "
            f"received {type(children).__name__}"
        )
    if any(
        not isinstance(name, str)
        or not name
        or "." in name
        or (child is not None and not isinstance(child, DecodedModule))
        for name, child in children.items()
    ):
        raise RejectedInput(
            "module children must use non-empty, dot-free names and decoded module values"
        )
    return children


def _positive_int_attribute(module: DecodedModule, name: str, label: str) -> int:
    value = getattr(module, name, None)
    if type(value) is not int or value <= 0:
        raise RejectedInput(f"{label} must be a positive integer; received {value!r}")
    return value


def _module_state(module: DecodedModule, prefix: str) -> list[tuple[str, DecodedTensor]]:
    """Collect persistent parameters and buffers owned directly by one module."""
    parameters = getattr(module, "_parameters", None)
    buffers = getattr(module, "_buffers", None)
    ignored = getattr(module, "_non_persistent_buffers_set", None)
    if (
        not isinstance(parameters, dict)
        or not isinstance(buffers, dict)
        or not isinstance(ignored, set)
    ):
        raise RejectedInput("unsupported module state containers")
    rows: list[tuple[str, DecodedTensor]] = []
    for label, entries in (("parameter", parameters), ("buffer", buffers)):
        for name, tensor in entries.items():
            if not isinstance(name, str) or not name or "." in name:
                raise RejectedInput(f"invalid {label} name: {name!r}")
            if tensor is None or (label == "buffer" and name in ignored):
                continue
            if not isinstance(tensor, DecodedTensor):
                raise RejectedInput(f"non-tensor {label}")
            rows.append((prefix + name, tensor))
    return rows


def _state_dict(model: DecodedModule) -> list[tuple[str, DecodedTensor]]:
    """Flatten the model's persistent learned state under stable module paths."""
    rows: list[tuple[str, DecodedTensor]] = []
    ancestors: set[int] = set()

    def visit(module: DecodedModule, prefix: str) -> None:
        identity = id(module)
        if identity in ancestors:
            raise RejectedInput(f"module graph contains a cycle at {prefix or '<root>'}")
        ancestors.add(identity)
        rows.extend(_module_state(module, prefix))
        for name, child in _module_children(module).items():
            if child is not None:
                visit(child, prefix + name + ".")
        ancestors.remove(identity)

    visit(model, "")
    if len(rows) != len({name for name, _ in rows}):
        raise RejectedInput("duplicate learned-state name")
    return rows


def _all_modules(model: DecodedModule) -> dict[str, DecodedModule]:
    """Index the decoded module tree by the paths used by the fixed graph."""
    result = {"": model}
    ancestors: set[int] = set()

    def visit(module: DecodedModule, prefix: str) -> None:
        identity = id(module)
        if identity in ancestors:
            raise RejectedInput(f"module graph contains a cycle at {prefix or '<root>'}")
        ancestors.add(identity)
        for name, nested in _module_children(module).items():
            if nested is None:
                continue
            path = name if not prefix else f"{prefix}.{name}"
            result[path] = nested
            visit(nested, path)
        ancestors.remove(identity)

    visit(model, "")
    return result


def _required_module(
    modules: dict[str, DecodedModule], path: str, expected_kind: str
) -> DecodedModule:
    module = modules.get(path)
    observed_kind = getattr(type(module), "_global", None) if module is not None else None
    if module is None or observed_kind != expected_kind:
        raise RejectedInput(f"module {path} must be {expected_kind}; received {observed_kind!r}")
    return module


def _read_convolution_pair(value: object, label: str, *, allow_zero: bool) -> tuple[int, int]:
    """Narrow one decoded convolution attribute to a pair of integers."""
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(type(item) is not int or item < (0 if allow_zero else 1) for item in value)
    ):
        minimum = "non-negative" if allow_zero else "positive"
        raise RejectedInput(
            f"convolution {label} must contain two {minimum} integers; received {value!r}"
        )
    return value


def _read_convolution(
    path: str,
    convolution: DecodedModule,
    activation: ConvolutionActivation,
) -> ConvolutionSpec:
    """Validate one decoded Conv2d and copy the geometry the runtime uses."""
    if getattr(type(convolution), "_global", None) != "torch.nn.modules.conv.Conv2d":
        raise RejectedInput(f"invalid convolution module: {path}")
    input_channels = _positive_int_attribute(convolution, "in_channels", f"input channels: {path}")
    output_channels = _positive_int_attribute(
        convolution, "out_channels", f"output channels: {path}"
    )
    groups = _positive_int_attribute(convolution, "groups", f"convolution groups: {path}")
    if input_channels % groups or output_channels % groups:
        raise RejectedInput(
            f"convolution {path} groups={groups} must divide input={input_channels} "
            f"and output={output_channels} channels"
        )
    metadata = (
        getattr(convolution, "transposed", None),
        getattr(convolution, "output_padding", None),
        getattr(convolution, "padding_mode", None),
    )
    if metadata != (False, (0, 0), "zeros"):
        raise RejectedInput(
            f"convolution {path} must be non-transposed with zero output padding and "
            f"zero padding mode; received {metadata!r}"
        )
    return ConvolutionSpec(
        path=path,
        input_channels=input_channels,
        output_channels=output_channels,
        kernel=_read_convolution_pair(convolution.kernel_size, "kernel", allow_zero=False),
        stride=_read_convolution_pair(convolution.stride, "stride", allow_zero=False),
        padding=_read_convolution_pair(convolution.padding, "padding", allow_zero=True),
        dilation=_read_convolution_pair(convolution.dilation, "dilation", allow_zero=False),
        groups=groups,
        activation=activation,
    )


def _read_convolution_block(
    path: str, wrapper: DecodedModule
) -> tuple[ConvolutionSpec, BatchNormFusion]:
    """Read a YOLO Conv block as its convolution and required BatchNorm fold."""
    nested = _module_children(wrapper)
    convolution = nested.get("conv")
    batch_norm = nested.get("bn")
    activation = nested.get("act")
    if not all(isinstance(value, DecodedModule) for value in (convolution, batch_norm, activation)):
        raise RejectedInput(f"incomplete convolution wrapper: {path}")
    assert isinstance(convolution, DecodedModule)
    assert isinstance(batch_norm, DecodedModule)
    assert isinstance(activation, DecodedModule)
    activation_kind = getattr(type(activation), "_global", None)
    if activation_kind == "torch.nn.modules.activation.SiLU":
        if getattr(activation, "inplace", None) is not True:
            raise RejectedInput(f"unsupported SiLU metadata: {path}")
        semantic_activation = ConvolutionActivation.SILU
    elif activation_kind == "torch.nn.modules.linear.Identity":
        semantic_activation = ConvolutionActivation.IDENTITY
    else:
        raise RejectedInput(f"unsupported convolution activation: {path}")
    convolution_path = f"{path}.conv"
    spec = _read_convolution(convolution_path, convolution, semantic_activation)
    if getattr(type(batch_norm), "_global", None) != "torch.nn.modules.batchnorm.BatchNorm2d":
        raise RejectedInput(f"unsupported BatchNorm module: {path}")
    epsilon = getattr(batch_norm, "eps", None)
    if not isinstance(epsilon, float) or not math.isfinite(epsilon) or epsilon <= 0.0:
        raise RejectedInput(f"invalid BatchNorm epsilon: {path}")
    return spec, BatchNormFusion(convolution_path, epsilon)


def _convolution_inventory(
    modules: dict[str, DecodedModule],
) -> tuple[dict[str, ConvolutionSpec], list[BatchNormFusion]]:
    """Extract every convolution and each wrapper BatchNorm that must be fused."""
    convolutions: dict[str, ConvolutionSpec] = {}
    fusions: list[BatchNormFusion] = []
    wrappers = {"ultralytics.nn.modules.conv.Conv", "ultralytics.nn.modules.conv.DWConv"}
    for path, module in modules.items():
        if getattr(type(module), "_global", None) in wrappers:
            spec, fusion = _read_convolution_block(path, module)
            convolutions[spec.path] = spec
            fusions.append(fusion)
    for path, module in modules.items():
        if (
            getattr(type(module), "_global", None) == "torch.nn.modules.conv.Conv2d"
            and path not in convolutions
        ):
            convolutions[path] = _read_convolution(path, module, ConvolutionActivation.IDENTITY)
    return convolutions, fusions


def _read_attention_block(
    path: str,
    module: DecodedModule,
    convolutions: dict[str, ConvolutionSpec],
) -> AttentionSpec:
    """Validate attention dimensions and the QKV convolution they require."""
    heads = _positive_int_attribute(module, "num_heads", f"attention heads: {path}")
    head_channels = _positive_int_attribute(module, "head_dim", f"attention head channels: {path}")
    key_channels = _positive_int_attribute(module, "key_dim", f"attention key channels: {path}")
    scale = getattr(module, "scale", None)
    if not isinstance(scale, float) or not math.isfinite(scale):
        raise RejectedInput(f"invalid attention metadata: {path}")
    spec = AttentionSpec(path, heads, head_channels, key_channels, scale)
    qkv_path = f"{path}.qkv.conv"
    qkv = convolutions.get(qkv_path)
    if qkv is None:
        raise RejectedInput(f"attention block {path} is missing QKV convolution {qkv_path}")
    observed_qkv = (
        qkv.input_channels,
        qkv.output_channels,
        qkv.kernel,
        qkv.stride,
        qkv.padding,
        qkv.dilation,
        qkv.groups,
        qkv.activation,
    )
    expected_qkv = (
        heads * head_channels,
        heads * (head_channels + 2 * key_channels),
        (1, 1),
        (1, 1),
        (0, 0),
        (1, 1),
        1,
        ConvolutionActivation.IDENTITY,
    )
    if observed_qkv != expected_qkv:
        raise RejectedInput(
            f"attention block {path} QKV convolution must be {expected_qkv!r}; "
            f"received {observed_qkv!r}"
        )
    return spec


def _validate_sppf_pool(modules: dict[str, DecodedModule]) -> None:
    """Validate the size-preserving pool repeated by the runtime's SPPF graph."""
    pool = _required_module(modules, _SPPF_MAX_POOL_PATH, "torch.nn.modules.pooling.MaxPool2d")
    # SPPF applies the same 5x5, stride-one pool three times. Padding by two
    # preserves spatial dimensions so the original and three pooled tensors can
    # be concatenated by the fixed graph.
    expected_pool_attributes = {
        "kernel_size": 5,
        "stride": 1,
        "padding": 2,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
    }
    for attribute, expected_value in expected_pool_attributes.items():
        observed = getattr(pool, attribute, None)
        if observed != expected_value:
            raise RejectedInput(
                f"SPPF max-pool {_SPPF_MAX_POOL_PATH} {attribute} must be "
                f"{expected_value!r}; received {observed!r}"
            )


def _validate_upsampling(modules: dict[str, DecodedModule]) -> None:
    """Validate the two nearest-neighbor expansions in the detector neck."""
    for path in _NEAREST_UPSAMPLE_PATHS:
        upsample = _required_module(modules, path, "torch.nn.modules.upsampling.Upsample")
        observed = (
            getattr(upsample, "size", None),
            getattr(upsample, "scale_factor", None),
            getattr(upsample, "mode", None),
            getattr(upsample, "align_corners", None),
            getattr(upsample, "recompute_scale_factor", None),
        )
        expected_upsample = (None, 2.0, "nearest", None, None)
        if observed != expected_upsample:
            raise RejectedInput(
                f"upsample {path} must use 2x nearest-neighbor scaling; received {observed!r}"
            )


def _validate_concatenation(modules: dict[str, DecodedModule]) -> None:
    """Validate that every neck merge joins its inputs along channels."""
    for path in _CHANNEL_CONCATENATION_PATHS:
        concatenate = _required_module(modules, path, "ultralytics.nn.modules.conv.Concat")
        if getattr(concatenate, "d", None) != 1:
            raise RejectedInput(
                f"concatenation {path} must join the channel axis; "
                f"received axis={getattr(concatenate, 'd', None)!r}"
            )


def _validate_residual_connections(modules: dict[str, DecodedModule]) -> None:
    """Validate every shortcut addition reproduced by the fixed graphs."""
    for path in _ATTENTION_RESIDUAL_PATHS:
        block = _required_module(modules, path, "ultralytics.nn.modules.block.PSABlock")
        if getattr(block, "add", None) is not True:
            raise RejectedInput(f"attention block {path} must retain its residual connection")
    for path in _RESIDUAL_BOTTLENECK_PATHS:
        block = _required_module(modules, path, "ultralytics.nn.modules.block.Bottleneck")
        if getattr(block, "add", None) is not True:
            raise RejectedInput(f"bottleneck {path} must retain its residual connection")


def _validate_fixed_blocks(modules: dict[str, DecodedModule]) -> None:
    """Validate parameter-free model behavior reproduced by the runtime."""
    _validate_sppf_pool(modules)
    _validate_upsampling(modules)
    _validate_concatenation(modules)
    _validate_residual_connections(modules)


def _validated_topology(model: DecodedModule) -> _ValidatedTopology:
    """Extract every operator attribute needed by fixed graph construction."""
    modules = _all_modules(model)
    convolutions, fusions = _convolution_inventory(modules)
    runtime_convolutions = tuple(sorted(path for path in convolutions if ".one2one_" not in path))
    if runtime_convolutions != RUNTIME_CONVOLUTION_PATHS:
        missing = sorted(set(RUNTIME_CONVOLUTION_PATHS) - set(runtime_convolutions))
        unexpected = sorted(set(runtime_convolutions) - set(RUNTIME_CONVOLUTION_PATHS))
        raise RejectedInput(
            "checkpoint convolutions do not match the fixed runtime: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    attention = tuple(
        _read_attention_block(
            path,
            _required_module(modules, path, "ultralytics.nn.modules.block.Attention"),
            convolutions,
        )
        for path in _ATTENTION_PATHS
    )
    _validate_fixed_blocks(modules)
    return _ValidatedTopology(
        convolutions=tuple(convolutions[path] for path in sorted(convolutions)),
        batch_norm_fusions=tuple(sorted(fusions, key=lambda row: row.convolution_path)),
        attention_blocks=attention,
    )


def _validate_model_kind(model: DecodedModule) -> None:
    """Require the Ultralytics model type whose fields are read below."""
    model_kind = getattr(type(model), "_global", None)
    if model_kind != "ultralytics.nn.tasks.DetectionModel":
        raise RejectedInput(
            f"checkpoint model must be ultralytics.nn.tasks.DetectionModel; received {model_kind!r}"
        )


def _class_names(model: DecodedModule) -> tuple[str, ...]:
    """Read the class labels that determine the runtime's output width."""
    names = getattr(model, "names", None)
    class_count = len(names) if isinstance(names, dict) else 0
    if (
        not isinstance(names, dict)
        or class_count < 1
        or list(names) != list(range(class_count))
        or not all(isinstance(value, str) and value for value in names.values())
    ):
        raise RejectedInput(
            "class names must map contiguous indices starting at zero to non-empty strings; "
            f"received {names!r}"
        )
    return tuple(names[index] for index in range(class_count))


def _validate_module_graph(model: DecodedModule, class_count: int) -> None:
    """Match the routing and operator kinds implemented by the fixed runtime."""
    sequence = _module_children(model).get("model")
    sequence_children = _module_children(sequence) if sequence is not None else {}
    expected_names = tuple(str(index) for index in range(len(_MODULE_GRAPH)))
    if tuple(sequence_children) != expected_names or any(
        module is None for module in sequence_children.values()
    ):
        raise RejectedInput(
            "top-level model sequence must contain modules named 0 through 23 in order; "
            f"received {tuple(sequence_children)!r}"
        )
    modules = [module for module in sequence_children.values() if module is not None]
    graph = tuple(
        (
            getattr(row, "i", None),
            getattr(row, "f", None),
            getattr(type(row), "_global", None),
        )
        for row in modules
    )
    if graph != _MODULE_GRAPH:
        mismatch = next(
            (
                index
                for index, (expected, observed) in enumerate(
                    zip(_MODULE_GRAPH, graph, strict=False)
                )
                if expected != observed
            ),
            min(len(_MODULE_GRAPH), len(graph)),
        )
        expected = _MODULE_GRAPH[mismatch] if mismatch < len(_MODULE_GRAPH) else None
        observed = graph[mismatch] if mismatch < len(graph) else None
        raise RejectedInput(
            "module topology does not match YOLO26n: "
            f"module {mismatch} expected {expected!r}, received {observed!r}; "
            f"expected {len(_MODULE_GRAPH)} modules, received {len(graph)}"
        )
    head = modules[-1]
    expected_head = (class_count, 1, detection_output_channels(class_count))
    observed_head = (
        getattr(head, "nc", None),
        getattr(head, "reg_max", None),
        getattr(head, "no", None),
    )
    if observed_head != expected_head:
        raise RejectedInput(
            "detection head must contain (classes, reg_max, outputs) "
            f"{expected_head!r}; received {observed_head!r}"
        )
    stride = getattr(model, "stride", None)
    if (
        not isinstance(stride, DecodedTensor)
        or stride.dtype != "float32"
        or stride.values() != list(DETECTION_STRIDES)
    ):
        observed_stride = stride.values() if isinstance(stride, DecodedTensor) else stride
        raise RejectedInput(
            f"detection strides must be {DETECTION_STRIDES!r} as float32; "
            f"received {observed_stride!r}"
        )


def _classification_width(topology: _ValidatedTopology, class_count: int) -> int:
    """Validate the hidden width shared by the three classification branches."""
    width = classification_width_for_classes(class_count)
    convolutions = {row.path: row for row in topology.convolutions}
    paths = tuple(
        _CLASSIFICATION_HIDDEN_CONVOLUTION.format(branch=branch)
        for branch in range(DETECTION_BRANCH_COUNT)
    )
    observed = tuple(
        convolutions[path].output_channels if path in convolutions else None for path in paths
    )
    if observed != (width,) * DETECTION_BRANCH_COUNT:
        raise RejectedInput(
            "YOLO26n classification branches have incompatible hidden widths: "
            f"expected {width} channels in each branch, observed {observed}"
        )
    return width


def _learned_tensors(model: DecodedModule) -> tuple[LearnedTensor, ...]:
    """Copy finite tensor data out of the decoded Torch object graph."""
    learned = _state_dict(model)
    if not learned:
        raise RejectedInput("YOLO26n checkpoint does not contain learned tensors")
    for name, tensor in sorted(learned):
        if tensor.dtype not in {"float16", "int64"}:
            raise RejectedInput(
                f"learned tensor {name} must use float16 or int64; received {tensor.dtype}"
            )
        if tensor.dtype == "float16" and not all(
            math.isfinite(float(value)) for value in tensor.values()
        ):
            raise RejectedInput(f"non-finite learned tensor: {name}")
    return tuple(
        LearnedTensor(name, tensor.shape, tensor.dtype, tensor.contiguous_bytes())
        for name, tensor in learned
    )


def validate_yolo26n_model(model: DecodedModule) -> ValidatedYolo26n:
    """Copy a decoded YOLO26n model into the complete fixed-runtime contract."""
    _validate_model_kind(model)
    class_names = _class_names(model)
    class_count = len(class_names)
    _validate_module_graph(model, class_count)
    topology = _validated_topology(model)
    width = _classification_width(topology, class_count)
    return ValidatedYolo26n(
        learned_tensors=_learned_tensors(model),
        convolutions=topology.convolutions,
        batch_norm_fusions=topology.batch_norm_fusions,
        attention_blocks=topology.attention_blocks,
        class_names=class_names,
        classification_width=width,
        detection_strides=DETECTION_STRIDES,
    )

# YOLO26n converter and runtime

This is the purpose-built native YOLO26n path for Raspberry Pi 5:

```text
supported YOLO26n checkpoint + supported resolution
    -> model.bin + manifest.json

fixed YOLO26n runtime source + pinned dependencies
    -> one libarm-yolo26n.so.0

libarm-yolo26n.so.0 + converted model directory
    -> one runtime session
```

The exporter uses Python 3.13's standard library and never invokes a native
toolchain. Each exported model has one static input resolution. Supported
resolutions are 320, 480, 640, 800, 960, and 1280. Export and validate a
checkpoint by declaring the resolution explicitly:

```sh
make model RESOLUTION=640 CHECKPOINT=/absolute/yolo26n.pt
make model-check RESOLUTION=640 CHECKPOINT=/absolute/yolo26n.pt
```

The checkpoint-validation target performs one complete in-memory export for
the requested resolution. `make test-converter CHECKPOINT=...` separately runs
the checkpoint-backed converter behavior and rejection tests.

The runtime is built independently from all checkpoints and resolutions. CMake
fetches its pinned dependencies and builds it natively on Linux AArch64:

```sh
make runtime
```

The same installed native library can execute any supported converted model. The
native CTest suite and Raspberry Pi public-interface tests own runtime behavior
and rejection coverage. `make ultralytics-benchmark` records accuracy and
performance together with the Raspberry Pi host context for that run.

The fixed topology and schedule are ordinary C++ runtime source. Checkpoint
conversion, native build, installation, and inference have no code-generation
step, and learned values are never compiled into the runtime.

## Converter pipeline

The converter is the `converter` package. Its production export
path is deliberately linear:

1. `checkpoint_decoder` reads the supported Torch archive without importing
   Torch or Ultralytics.
2. `model_validation` parses the instantiated operators, routing, tensor
   inventory, and class-head geometry that the fixed runtime actually uses.
   Unrelated training and YAML metadata do not affect acceptance.
3. `weight_lowering` folds BatchNorm, converts convolution layouts, creates the
   two split projections and their zero biases, and returns named weight slots.
4. `model_format` writes those slots plus resolution and class metadata to
   `model.bin`; `export_yolo26n` writes the model and manifest atomically.

The converter stops at `model.bin`. The runtime directly owns the one supported
YOLO26n program, including its nineteen XNNPACK partitions, execution phases,
external bindings, and memory plan.

## Weight conversion contract

Learned FP16 checkpoint values are decoded into explicit float32 values.
BatchNorm folding evaluates each add, subtract, multiply, divide, and `sqrtf`
at float32 precision in the same order as the implementation. Before storage,
every effective value is rounded through FP16 and then written as FP32 for
XNNPACK. This order is part of the reproducible artifact contract.

The runtime computes the one-to-many detection path, so the checkpoint's
separate one-to-one head tensors are deliberately excluded. Two 1x1
projections are divided at their concatenated-input channel boundaries for the
runtime's split-projection execution; each half receives an explicit zero bias
because the source convolution has no learned bias.

Regular convolution weights change from OIHW to OHWI. Depthwise convolution
weights change from OIHW to multiplier-height-width-channel order. Biases keep
their one-dimensional order. Unsupported dtypes, shapes, groups, non-finite
values, or incomplete BatchNorm state are rejected before an artifact is
written.

# YOLO26n model converter

The converter turns a compatible YOLO26n PyTorch checkpoint into the
`model.bin` and `manifest.json` files consumed by the native runtime. It uses
the Python 3.13 standard library and does not import Torch or Ultralytics.

Export and validate a checkpoint from the repository root:

```sh
make model RESOLUTION=640 CHECKPOINT=/absolute/yolo26n.pt
make model-check RESOLUTION=640 CHECKPOINT=/absolute/yolo26n.pt
```

Supported resolutions are 320, 480, 640, 800, 960, and 1280. Each converted
model has one static input resolution. The same native runtime library can
execute models at any supported resolution.

## Conversion pipeline

1. `checkpoint_decoder` reads the supported Torch archive without importing
   executable model code.
2. `model_validation` validates the model type, operator topology, routing,
   tensor inventory, class labels, and detection-head geometry required by the
   fixed runtime.
3. `weight_lowering` folds BatchNorm, converts convolution layouts, and
   creates the split projections used by the runtime.
4. `model_format` serializes the named weight slots, resolution, and class
   metadata to `model.bin`.
5. `export_yolo26n` publishes `model.bin` and `manifest.json` together.

The converter stops at `model.bin`. The runtime independently owns the fixed
YOLO26n program, including its nineteen XNNPACK partitions, execution schedule,
external bindings, and memory plan.

## Weight conversion contract

Learned FP16 checkpoint values are decoded into explicit float32 values.
BatchNorm folding evaluates each operation at float32 precision. Before
storage, every effective value is rounded through FP16 and then written as FP32
for XNNPACK. This order is part of the reproducible artifact contract.

The runtime computes the one-to-many detection path, so the checkpoint's
separate one-to-one head tensors are excluded. Two 1x1 projections are divided
at their concatenated-input channel boundaries for split-projection execution;
each half receives an explicit zero bias because the source convolution has no
learned bias.

Regular convolution weights change from OIHW to OHWI. Depthwise convolution
weights change from OIHW to multiplier-height-width-channel order. Biases retain
their one-dimensional order. Unsupported dtypes, shapes, groups, non-finite
values, or incomplete BatchNorm state are rejected before an artifact is
written.

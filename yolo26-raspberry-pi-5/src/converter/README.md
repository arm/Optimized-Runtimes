# YOLO26n model converter

The converter turns a compatible YOLO26n PyTorch checkpoint into the
`model.bin` and `manifest.json` files consumed by the native runtime. It uses
the Python 3.13 standard library and does not import Torch or Ultralytics.

Export and validate a checkpoint from the `yolo26-raspberry-pi-5/` runtime
directory:

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

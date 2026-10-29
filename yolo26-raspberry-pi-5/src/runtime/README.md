# Native runtime

This runtime owns the fixed YOLO26n topology and execution schedule. It builds
the model's nineteen XNNPACK graphs when a runtime instance is created.

## File layout

- `arm_yolo26n.h` defines the public C API.
- `arm_yolo26n.cpp` owns the lifetime of a loaded model, its graphs, worker
  threads, and scratch memory.
- `model_contract.h` defines supported resolutions, tensor alignment, and
  output dimensions.
- `model_store.cpp` validates `model.bin` and resolves its weight slots.
- `fixed_graph.cpp` and the `fixed_graph_*.inc` files define the backbone,
  neck, detection, and early-projection graphs.
- `fixed_schedule.cpp` executes the graphs and performs data movement between
  them.
- `fixed_memory_plan.cpp` assigns intermediate tensors to the shared
  activation arena.
- `preprocess.cpp` pads and converts packed RGB, BGR, or BGRX images into the
  planar floating-point input expected by the runtime. It does not resize.
- `projection_optimization.cpp` avoids two intermediate tensors and
  transposes by writing two early 1x1 projections directly into the layout
  consumed by the next graph. It is the only runtime source permitted to use
  private XNNPACK interfaces.
- `patches/` contains the project-owned XNNPACK and pthreadpool changes.
- `tests/` contains the public C API behavior test.
- `CMakeLists.txt` owns native dependency acquisition and patching,
  compilation, testing, ABI verification, installation, and packaging.

## Building

From the `yolo26-raspberry-pi-5/` runtime directory, run:

```sh
make runtime
```

The top-level Make target configures CMake, builds and tests the native runtime,
and installs it under `dist/runtime/`. CMake fetches the pinned native
dependencies and includes their license notices from `licenses/` in the
installed runtime.

## Runtime details

### Fixed runtime topology

YOLO26n builds progressively smaller feature maps in its backbone, combines
them from coarse to fine in its neck, then predicts boxes and classes at three
scales in its detection head. This runtime uses conventional pyramid names for
those feature levels:

| Level | Spatial size for resolution `R` | Role |
| --- | --- | --- |
| P2 | `R / 4` | Early backbone features used by the split-projection optimization. |
| P3 | `R / 8` | Finest detection scale; preserves the most spatial detail. |
| P4 | `R / 16` | Middle backbone, neck, and detection scale. |
| P5 | `R / 32` | Coarsest backbone and detection scale; contains the attention blocks. |

The topology is divided into nineteen XNNPACK subgraphs. These partitions are
not independently configurable models; together they form one fixed YOLO26n
program. Boundaries occur where the runtime performs explicit data movement
for attention reshaping, upsampling, output conversion, or split-projection
optimization.

### Memory planning

The standard 640-pixel, 80-class model uses a compact placement benchmarked on
Raspberry Pi 5. Other supported resolutions or class counts use an aligned
sequential placement calculated from their tensor sizes, with one reuse where
`P3Fusion` takes over the no-longer-needed `BackboneP3` storage. This uses
more memory but avoids applying size-specific offsets to incompatible tensor
shapes.

### Runtime lifecycle and buffers

`y26_native_create` copies and validates caller-provided `model.bin` bytes,
derives tensor sizes, allocates scratch memory, creates a four-worker thread
pool, and builds the nineteen graphs. The caller can release its byte buffer
after creation returns. This work happens once per runtime instance; the native
runtime does not access the filesystem.

The public input and output buffers contain FP32 values; most intermediate
tensors use FP16. Callers query exact byte counts from the runtime before
calling `y26_native_execute`.

`y26_native_execute` returns raw predictions, not final detections. Its output
has shape `[1, 4 + classes, positions]`; the first four channels contain
`xywh` boxes and the remaining channels contain per-class scores. Native API
callers must apply confidence filtering, non-maximum suppression, and any
required coordinate scaling. The high-level Python `Yolo26Detector` performs
these steps and returns final detections; the low-level Python
`Yolo26nTensorSession` exposes the same raw output as the native API.

Concurrent inference requires one instance per caller. A caller must not destroy
an instance while a call is using it. `y26_native_destroy` releases the
complete instance.

`y26_native_preprocess_image_no_resize` accepts images whose longer side
already matches the model resolution. It normalizes pixels, separates color
channels, and pads the shorter side, but deliberately does not resize.

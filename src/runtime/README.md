# Native runtime

This runtime owns the fixed YOLO26n topology and execution schedule. It builds
the model's nineteen XNNPACK graphs when a runtime instance is created.

## File Layout

- `arm_yolo26n.h` defines the public C API; `arm_yolo26n.cpp` owns the
  lifetime of a loaded model, its graphs, worker threads, and scratch memory.
- `model_contract.h` defines the resolutions, tensor alignment, and output
  dimensions the runtime supports. `model_store.cpp` validates `model.bin` and
  resolves its weight slots.
- `fixed_graph.cpp` provides the graph-building vocabulary. The
  `fixed_graph_*.inc` files use it to define the backbone, neck, detection, and
  early-projection graphs.
- `fixed_schedule.cpp` runs the graphs in model order and performs the data
  movement between them. `fixed_memory_plan.cpp` assigns the intermediate
  tensors to the shared activation arena.
- `preprocess.cpp` pads and converts packed RGB, BGR, or BGRX images into the
  planar floating-point input expected by the runtime. It does not resize.
- `projection_optimization.cpp` avoids two intermediate tensors and transposes
  by writing each half of two early 1x1 projections directly into the layout
  consumed by the next graph. It is the only runtime source allowed to use
  private XNNPACK interfaces.
- `patches/` contains the project-owned XNNPACK and pthreadpool changes, with
  their responsibilities documented in `patches/README.md`.
- `tests/` contains the public C API behavior test. CTest skips its model-backed
  execution during an ordinary build; the end-to-end workflow supplies a
  converted model and runs it against the same binary.
- `CMakeLists.txt` owns dependency acquisition and patching, native compilation,
  testing, ABI verification, installation, and packaging. The script in
  `cmake/` verifies the built library.

## Building

Build the checkpoint-independent runtime once on Raspberry Pi 5:

```sh
cmake -S . -B /absolute/build -DCMAKE_BUILD_TYPE=Release
cmake --build /absolute/build --target arm_yolo26n --parallel
ctest --test-dir /absolute/build --output-on-failure
```

The standard install and CPack targets publish separate runtime and development
components. Converted models remain independent and never invoke CMake:

```sh
cmake --install /absolute/build --prefix /absolute/runtime-install
cpack --config /absolute/build/CPackConfig.cmake -C Release
```

Accuracy and performance are measured through the Ultralytics integration:

```sh
make ultralytics-benchmark \
  RUNTIME=/absolute/runtime-install/lib/libarm-yolo26n.so.0 \
  MODEL=/absolute/converted/model \
  BENCHMARK_OUTPUT=/absolute/current-benchmark
```

Runtime correctness belongs to the native CTest suite and the Raspberry Pi
tests of the public Python API. The benchmark reports its normal Ultralytics
results while the surrounding recorder writes before-and-after host context.

## Dependencies

`CMakeLists.txt` owns the pinned dependency sources, revisions, and archive
hashes. The corresponding license notices under `licenses/` are included in
the runtime package.
The developer-only performance evidence workflow lives under
[`benchmarks/reporting/`](../../benchmarks/reporting/).

## Runtime Details

### Fixed runtime topology

YOLO26n builds progressively smaller feature maps in its backbone, combines
them from coarse to fine in its neck, then predicts boxes and classes at three
scales in its detection head. This runtime uses the conventional pyramid names
for those feature levels:

| Level | Spatial size for resolution `R` | Role |
| --- | --- | --- |
| P2 | `R / 4` | Early backbone features used by the split-projection optimization. |
| P3 | `R / 8` | Finest detection scale; preserves the most spatial detail. |
| P4 | `R / 16` | Middle backbone, neck, and detection scale. |
| P5 | `R / 32` | Coarsest backbone and detection scale; contains the attention blocks. |

The topology is divided into nineteen XNNPACK subgraphs. Those partitions are
not independently configurable models and together are one fixed YOLO26n
program. Boundaries exist where the runtime must perform explicit data motion,
attention reshaping, upsampling, output conversion, or split-projection
optimization between XNNPACK invocations.

### Memory planning

The schedule owns activation-memory placement. The standard 640-pixel,
80-class model uses a compact placement benchmarked on Raspberry Pi 5. Other
supported resolutions or class counts use an aligned sequential placement
calculated from their tensor sizes, with one reuse where `P3Fusion` takes over
the no-longer-needed `BackboneP3` storage. This uses more memory but avoids
applying the size-specific 640-pixel offsets to incompatible tensor shapes.

Every placement is checked during initialization. Two tensors may share arena
storage only when their required lifetimes do not overlap.

### Runtime lifecycle and buffers

`y26_native_create` copies and validates caller-provided `model.bin` bytes,
derives all tensor sizes, allocates scratch memory, creates a four-worker
thread pool, and builds the nineteen graphs. The caller can release its byte
buffer after creation returns. This work happens once for each runtime
instance; the native runtime does not access the filesystem.

The public input and output buffers contain FP32 values; most intermediate
tensors use FP16. Callers query the exact byte counts from the runtime before
calling `y26_native_execute`. An instance owns mutable graph and scratch state,
so it cannot execute concurrent calls; independent concurrent inference
requires one instance per caller. The caller must not destroy an instance
while a call is using it. `y26_native_destroy` releases the complete instance.

`y26_native_preprocess_image_no_resize` is provided for images whose longer
side already matches the model resolution. It normalizes the pixels, separates
their color channels, and pads the shorter side, but deliberately does not own
image resizing.

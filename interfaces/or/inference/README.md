<!-- SPDX-License-Identifier: Apache-2.0 -->
# Standalone OR inference and timing

Small Linux C++17 client for OR ABI 10. No OPK, image library, Python runtime,
GTest or model-specific code is needed to build or run it. It loads a trusted
shared library, loads a model package from memory, binds reusable host IO, runs
synchronous inference and saves raw outputs. The same executable handles one
input for latency measurements or a list of inputs for accuracy evaluation.

## Build

```sh
make -C interfaces/or/inference
make -C interfaces/or/inference test
```

Requirements: Linux, a C++17 compiler, Make and libdl. Tests additionally require
Python 3 and a native C++ compiler. Make builds `test_runtime.cpp` with the
existing upstream mock into `test_runtime.so`; Python only runs the client and
checks results. The tests do not require GTest. Override `CXX`, `CPPFLAGS`, `CXXFLAGS`, `LDFLAGS` or
`OR_INCLUDE` as needed. For example:

```sh
make -C interfaces/or/inference clean
make -C interfaces/or/inference CXX=aarch64-linux-gnu-g++
```

Run cross-built binaries on the target. Emulation checks functionality only.
The runtime library, its dependencies and the model package must match the target
and each other; this tool does not download or build them.

## Inputs and inference

Prepare a text file containing one raw tensor file path per line. Relative paths
are resolved beside the text file; absolute paths work too. Use LF line endings,
no quoting and no comments; empty lines are ignored. Files contain native-endian
dense Float32 tensor bytes, with **no header**. Each file must exactly match the
model's resolved input shape and layout. A `.npy` header is not accepted.

The caller owns image decoding, resizing, letterboxing, normalization and tensor
layout. Do those consistently for reference and candidate. For example, after
preprocessing in Python:

```python
import numpy as np

tensor = np.load('prepared.npy', allow_pickle=False)
np.ascontiguousarray(tensor, dtype=np.float32).tofile('prepared.bin')
```

Given `inputs.txt` containing `prepared.bin`, run from the repository root:

```sh
interfaces/or/inference/or-run ./libmy-runtime.so model.bin inputs.txt results
```

The output directory must not exist. This defaults to zero warmups and one
inference per input. The model and session are loaded once and reused while each
input's results are saved before the next inference. Inputs must be independent:
stateful models are explicitly rejected.

For each nonempty input line, outputs are numbered from zero:

- `0.json`: input path, per-call timing samples, median/mean and output metadata.
- `0-0.bin`, `0-1.bin`, ...: raw output tensor views indexed by model output index.
- `1.json`, `1-0.bin`, ...: the next input's results.
- `run.json`: written only after the entire input list succeeds.

Treat results as complete only if the process exits zero **and** `run.json`
exists. A failed run may leave diagnostic/partial files; rerun into a fresh
directory rather than overwriting them. Errors go to stderr and return nonzero.

Output metadata contains tensor `index`, numeric OR `dtype`, actual `shape`,
`byte_strides`, `byte_length` and raw filename. Strides are bytes; use them when
constructing a tensor view. The file starts at the reported output view's byte
offset and may include capacity/padding beyond the shape's elements. Actual shape
and strides determine valid elements. Scalar outputs have empty shape/strides.
Data-dependent output shapes use the runtime's bounded allocation and are written
with the actual post-inference shape. No output is truncated or silently padded to
a guessed detection count.

The dtype values are defined by [or_runtime.h](../or_runtime.h): Float32=1,
Float16=2, BFloat16=3, Int8=4, UInt8=5, Int16=6, Int32=7, Int64=8, Bool8=9.
Quantization parameters, class names and decoding semantics remain in the model
contract; the runner does not interpret them. Convert raw tensors to detections
and COCO predictions in a separate evaluation script.

## Latency

Supply warmup and measurement counts after the output directory:

```sh
interfaces/or/inference/or-run ./libmy-runtime.so model.bin inputs.txt timing 20 200
```

Use a one-line input list for a conventional warmed latency trial. For a multi-line
list, the warmup/measurement loop runs separately for each input and saves the
**last invocation's** outputs for that input. Use defaults for accuracy inference.

The monotonic timer covers `infer`, checking terminal execution success,
`get_outputs`, and releasing the completion. Model loading, allocation, binding,
input file reads/copies, sorting/statistics and output serialization are excluded.
Each sample reuses already-bound buffers; compare an ONNX baseline with equivalent
IO reuse/materialization and also report end-to-end processing separately. Samples
are milliseconds, not throughput or a target-performance claim.

OR's core ABI has no thread-count option. Configure the concrete runtime through
its documented package/environment/build settings, record its actual worker count,
and match reference settings. This runner does not invent a generic `--threads`.
Use an outer `timeout` to bound benchmarks; synchronous inference cannot be
cancelled through this tool. Profile with the target's `perf` separately from
clean latency trials.

## Supported subset

- Exact ABI negotiation and synchronous completion/result checking.
- One static dense unquantized Float32 input, any number of output tensors.
- Runtime-owned, host-accessible, direct POINTER allocations for every IO tensor;
  input/output options are selected independently, preserving accessibility bits.
- Borrowed runtime outputs, including bounded data-dependent output shapes.
- Proper completion/session/buffer/model/context cleanup before library unloading.

Unsupported models, memory profiles and execution modes fail explicitly. There is
no async fallback, host import, borrowed input binding, device mapping, shape
override, preprocessing, NMS, detection scorer or special YOLO dispatch. Cleanup
failure terminates without unloading code/storage still owned by the runtime.
The [OR contract](../runtime-contract.md) defines the lifecycle and memory rules.

# Repository layout

The repository is organized by ownership rather than by the order in which
commands happen to run.

```text
src/
  converter/                Checkpoint validation and model conversion
  runtime/                  CMake-owned native runtime
    api/python/arm_yolo26n/  Public application-facing Python API
benchmarks/
  ultralytics/              Optional Ultralytics adapter and benchmark
  reporting/                Performance evidence recorded around benchmarks
examples/
  video/                    Prerecorded-video example
  picamera2/                Camera examples
tests/
  unit/                     Portable API and command behavior
  converter/                Checkpoint-backed conversion behavior
  end_to_end/               Native runtime and example behavior
docs/                       Design, maintenance, and review documentation
```

`src/converter` turns a supported checkpoint into the model format consumed by
the runtime. It is repository tooling, not part of the application-facing API.

`src/runtime` owns native model loading and execution. Its Python API lives
with the runtime implementation and is imported by applications as
`arm_yolo26n`.

`benchmarks/ultralytics` owns everything needed only to expose the runtime to
Ultralytics: the backend, upstream patch, isolated environment, temporary
descriptor, and benchmark command. `benchmarks/reporting` records the
surrounding performance evidence. Torch and Ultralytics remain confined to the
benchmark environment.

The top-level example workflow obtains its declared inputs before invoking the
commands under `examples`. Example preparation is not part of native
compilation, Python API installation, or internal validation.

The top-level Makefile remains the discoverable command index. CMake continues
to own native dependency acquisition, compilation, tests, and packaging.

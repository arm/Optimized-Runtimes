<!--
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Host-memory API conformance

These GTest tests exercise the public [runtime ABI](../loom_runtime.h) without
application runtime classes, models or plugins. The default provider is an in-process
mock network that doubles four Float32 values. Passing the mock demonstrates
that the suite works; it does not certify another runtime.

## Build and run independently

Requirements: CMake 3.24 or newer and C/C++ compilers supporting C11 and C++17.
CMake reuses an existing `GTest::gtest_main` target or an installed GTest package.
If neither is available, it downloads the pinned GTest source. For an offline
build, supply a populated source directory with
`-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/path/to/googletest`.

Run from the repository root:

```sh
cmake -S interfaces/loom -B interfaces/loom/build
cmake --build interfaces/loom/build --parallel
ctest --test-dir interfaces/loom/build --output-on-failure
```

The standalone build runs declaration checks, the normal mock suite and
asymmetric-memory baselines, plus expected-failure checks for wrong output,
a clobbered bootstrap table and input/output memory coupling. CTest reports
success for a negative check only when the defective implementation is rejected.

## Test your implementation

Implement `lrt_conformance::make_provider()` from [provider.h](provider.h) in a
replacement source file. Keep runtime-specific setup out of the test file.
Use the shared helper from your runtime's CMake build:

```cmake
include(CTest)
include("${RUNTIMES_ROOT}/interfaces/loom/conformance/Conformance.cmake")
loom_add_conformance(my_conformance "${CMAKE_CURRENT_SOURCE_DIR}/my_provider.cpp" my_runtime)
add_test(NAME my_conformance COMMAND my_conformance)
set_tests_properties(my_conformance PROPERTIES TIMEOUT 60)
```

`RUNTIMES_ROOT` is the path to this repository, and `my_runtime` is your runtime
library target. Add further adapter sources with `target_sources`, includes with
`target_include_directories`, and dependencies with `target_link_libraries`.
The helper creates the executable; the parent registers its fixture-specific
CTest commands and timeouts. YOLO26n uses this same helper for shared and static providers.

The provider may call statically linked runtime code or load a shared library.
For a dynamic loader, keep the library loaded until process exit: failed
conformance must never unload code still used by a defective runtime.

The provider supplies:

- `get_api` and its expected contract version (the exported `lrt_runtime_version`
  for a shared plugin, or the named getter's contract version for static linking).
- Self-contained serialized model bytes. The suite overwrites/frees a temporary
  source copy after loading to check the source-borrow lifetime.
- An optional existing path for a PATH source of the same model. Empty paths
  explicitly skip the path test; memory loading remains mandatory for this profile.
- Complete input byte payloads and an output oracle that checks bytes using the
  runtime's model semantics and numerical tolerances. The test itself assumes no
  network or dtype. Padding/input initialization belongs in these fixture bytes.
- `input_option_indices`/`output_option_indices`: one known-compatible complete
  host POINTER configuration set from this model's `get_io_requirements` options.
  Different tensors may use different methods/configurations. The baseline must
  run; it is never silently skipped because all host paths are absent.
- Optional `arm_pending`, `finish_pending`, `fail_next_execution` controls. Omit
  unsupported controls; the corresponding deterministic timing/failure tests skip.
  These are fixture controls, not new runtime ABI operations.

The model fixture must have nonempty static tensors, non-data-dependent outputs,
at least one input/output, and deterministic output for the supplied input across
repeated execution. Stateful models are suitable only when the supplied fixture
has that behavior. The mock is stateless. Async-only runtimes are supported;
normal invocation waits have a finite two-second deadline.

## Coverage and exclusions

The suite checks exact-version bootstrap, all core entries, model/session/source
lifetime, host configurations, allocation/import/borrow methods, shapes/strides,
capacity/alignment, rejected binding atomicity, imported subrange offsets,
complete-set validation, repeated outputs, output metadata, stale completions,
parent/buffer busy lifetimes, context mismatch, completion failures and reset.

Per-tensor advertised methods need not form a compatible complete combination.
Optional homogeneous allocation/import/borrow tests skip an incompatible full
combination. They also skip when a method is not advertised for every fixture
tensor. Skips are printed by GTest and must be reviewed alongside pass counts.
The provider-selected baseline remains mandatory.

The mock's async gate is a deterministic simulation: the first poll remains
pending and a subsequent nonzero wait completes work. It does not test a real
worker thread or concurrent memory visibility. Failure injection reports one
accepted execution failure, followed by reset and recovery. `LRT_MOCK_FAULT`
selects intentionally broken mock behavior only for negative runner checks;
generic conformance tests never inspect that variable.

This first profile excludes Vulkan, DMA-BUF, all device-memory configurations,
dynamic shapes/data-dependent outputs, exhaustive concurrency/resource stress and allocator/device
failure coverage. Passing means the exercised host profile passed, not
that every optional ABI feature conforms.

When a destructive call unexpectedly succeeds, or work cannot quiesce before
cleanup, the runner exits immediately with failure rather than calling a stale
handle or freeing in-flight host storage. Use a process timeout around external
runtimes; no test can recover safely from a runtime that blocks forever inside
an ABI call. The standalone CTest checks use a 60-second process timeout, or 10 seconds
for expected-failure checks.

## Independent input and output memory

Each provider selects its baseline options independently for every input and
output. Matching option indices across tensors do not identify matching storage
or transfer modes. Homogeneous method tests are optional coverage, not a
requirement that both directions advertise the same methods.

The mock also runs baselines with external input/runtime output and the reverse,
including STAGED external storage on only one side. DIRECT external profiles
advertise two external choices on one side and one runtime choice on the other.
Each case checks repeated outputs and independent sessions. A negative test
intentionally requires matching input/output ownership and transfer, and must
fail conformance. These profiles verify selection semantics without claiming
DMA-BUF or Vulkan execution support.

## Consumer integration regressions

The standalone checks configure an external parent project that includes the
CMake helper from sibling directories, without shared directory-scope path
variables. Additional mock profiles verify that single-session providers pass
while the parallel-session test skips when its capability is absent. The
parallel test still requires two independently executable sessions when that
capability is advertised.

STAGED options can require different capacity and alignment from DIRECT options.
The suite allocates or imports replacement storage for the selected option and
refills fixture inputs before rebinding. Regression checks use larger and more
aligned staged storage with both borrowed and imported host bindings. Repeated
parallel submissions exercise the mock's atomically consumed one-shot pending
and failure controls.

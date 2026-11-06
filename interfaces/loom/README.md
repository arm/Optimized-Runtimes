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

# Runtime consumer interface

C ABI, contract version **10**, for applications and inference runtimes. A native
runtime may implement it directly or through an adapter. Preprocessing and output
interpretation belong to the application unless included in the model package.

| File | Purpose |
| --- | --- |
| [loom_runtime.h](loom_runtime.h) | Authoritative core types, lifecycle, memory binding and execution rules |
| [loom_runtime_vulkan.h](loom_runtime_vulkan.h) | Optional typed Vulkan buffer/image profile and handoff rules |
| [Contract](runtime-contract.md) | Responsibilities, lifecycle and execution semantics |
| [Memory guide](runtime-memory-use-cases.md) | Binding choices, output reuse and synchronization |
| [Declaration checks](tests/) | Standalone C11/C++11 compatibility and include-order checks |
| [Host conformance suite](conformance/README.md) | Replaceable GTest runtime provider and host-memory contract tests |

## Consumer lifecycle

1. Load the trusted runtime library and resolve `lrt_get_api`. Request exactly
   `LRT_RUNTIME_VERSION`; reject failure before reading the function table.
   Verify `api.get_runtime_version()` matches. Keep the library loaded until
   every runtime object is released. The runtime also exports the immutable
   `lrt_runtime_version` value for optional pre-bootstrap inspection.
2. Create a context, query capabilities, load a path or self-contained memory
   package, and describe its tensors. Package format is runtime-specific.
3. Create a session. Pass NULL options for static input shapes; resolve dynamic
   inputs through indexed shape overrides. Query resolved IO requirements.
4. Select an advertised memory configuration for every input and output.
   Borrow host storage, import external resources, or allocate runtime storage.
   Bind every tensor, then call `validate_bindings`.
5. Fill inputs and establish external readiness. Call `infer` for blocking
   execution or, if supported, `infer_async` followed by `wait`. Check execution
   success before consuming outputs. Release each terminal completion.
6. Refill and run repeatedly while bindings remain valid. Rebinding requires
   validation again. Do not overwrite outputs still used downstream.
7. Once idle, release completions, destroy sessions, release owned buffer handles,
   unload models and destroy contexts. Then unload the runtime library.

The runtime returns `LRT_UNSUPPORTED` for unavailable operations. Core function
pointers remain nonnull. Rejected submission accepts no work; accepted execution
reports failure through its completion. Timeout does not cancel work.

## Memory and output rules

Choose storage independently per tensor. `DIRECT` prohibits boundary staging;
`STAGED` explicitly requests an advertised staging path. Neither permits silent
fallback. Shape, dtype, strides, capacity, alignment and resource compatibility
must match that tensor's own requirements. Input and output option lists, ownership,
accessibility, resource references and transfer modes may differ, including between
tensors on the same side. There is no requirement for equal tensor counts or
matching option indices across inputs and outputs. Validate the complete selected
set; independent selection does not imply every combination is supported. IO
aliasing is unsupported.

Host pointers and native owners/mappings remain valid while bound/imported.
Imports do not retain external pool slots. Completion establishes production,
not downstream release. Output metadata is borrowed until the next accepted
invocation, binding change or session destruction. Retained output leases are
not supported. See the memory guide for pool-slot and native-resource reuse.

DMA-BUF and Vulkan are optional profiles, not zero-copy guarantees. The Vulkan
companion requires compatible device/queue ownership and explicit synchronization;
including the header does not enable those capabilities in a backend.

## Runtime capabilities

The YOLO26n runtime implements static Float32 tensors, synchronous execution
and runtime-owned direct host allocations. Other implementations may advertise
different per-tensor memory options or optional capabilities. Query each runtime;
interface declarations alone do not establish backend or hardware support.

## Host conformance tests

The [conformance suite](conformance/README.md) checks runtime behavior through
the public API, using a mock host runtime by default. Implementors replace its
provider adapter with their API and known model fixtures. The suite needs only
the interface headers, a C++ compiler and GTest.

See the suite README for standalone builds, provider requirements and coverage
limits. Passing the mock tests checks the harness, not a concrete runtime.

## Declaration checks

Run from this directory; no runtime library is linked:

```sh
cc -std=c11 -Wall -Wextra -Werror -pedantic -I . tests/header_contract.c -o /tmp/lrt-header-c
/tmp/lrt-header-c
c++ -x c++ -std=c++11 -Wall -Wextra -Werror -pedantic -I . tests/header_contract.c -o /tmp/lrt-header-cpp
/tmp/lrt-header-cpp
```

With Vulkan-Headers installed, add its include path as needed. Run the following
both with and without `-DLRT_TEST_CORE_FIRST` to cover both include orders:

```sh
cc -std=c11 -Wall -Wextra -Werror -pedantic -I . tests/vulkan_header_contract.c -o /tmp/lrt-vulkan-c
/tmp/lrt-vulkan-c
c++ -x c++ -std=c++11 -Wall -Wextra -Werror -pedantic -I . tests/vulkan_header_contract.c -o /tmp/lrt-vulkan-cpp
/tmp/lrt-vulkan-cpp
```

Before relying on a runtime/profile, validate known outputs, repeated execution,
binding replacement, invalid bounds, execution failures and owner lifetime.
Native-memory paths also need target testing of import, synchronization and reuse.

## Repository ownership and validation

This directory is the canonical shared contract in Optimized Runtimes. Native
builds reference it directly; installed development packages include its headers.
Source builds require the repository layout, including this directory.

Run portable declarations and mock conformance with:

```sh
cmake -S interfaces/loom -B interfaces/loom/build
cmake --build interfaces/loom/build --parallel
ctest --test-dir interfaces/loom/build --output-on-failure
```

Vulkan declaration tests are added when Vulkan-Headers are installed; ARM CI
installs them explicitly. Mock conformance tests the contract harness. Runtime
qualification requires each real provider and valid model fixtures.

YOLO26n accepts its existing `model.bin` through PATH or MEMORY; no manifest is
needed at this native boundary. It implements synchronous execution with
runtime-owned direct host allocations.
Optional asynchronous, borrowed-host, DMA-BUF and Vulkan paths are unsupported.
Load each library locally and resolve bootstrap symbols through its own handle.

## Static and shared linking

`LRT_EXPORT` specifies symbol visibility/import/export attributes. It does not
select a backend or enable plugin entry points. The ABI types and v10 table
layout are identical for both linking forms.

Each runtime provides a unique getter in its own public header:

| Runtime | Header | Getter | Source CMake target |
| --- | --- | --- | --- |
| YOLO26n | `y26_lrt.h` | `y26_lrt_get_api` | `arm_yolo26n_static` |

Static consumers include those headers and request one independent function
table per backend. Link the source CMake targets to retain their transitive
XNNPACK/pthreadpool dependencies. These are build-tree targets; this change
does not introduce a self-contained installed static SDK or merge native
dependencies into a single distributable archive.

```c
#include "y26_lrt.h"

int initialize_runtime(lrt_api *yolo) {
    return y26_lrt_get_api(LRT_RUNTIME_VERSION, yolo) == LRT_OK;
}
```

The static archives omit `lrt_get_api` and `lrt_runtime_version`. Their selected
version is checked by each named getter and its returned `get_runtime_version`
entry. Keep every handle paired with the table that created it; handles are
opaque and cannot be exchanged between implementations.

Shared libraries export both their unique getter and the existing generic
`lrt_get_api`/`lrt_runtime_version` plugin bootstrap. Dynamic loaders continue to
resolve the generic names on each library handle. Callers directly linking
multiple shared libraries use the unique getters. The generic bootstrap is a
separate source compiled only into shared targets, preserving existing plugin
consumers without generic symbols in static archives.

Backend handle implementations and helpers have translation-unit-local linkage;
only their opaque C handles cross the ABI boundary. A static consumer checks the
named getter and context lifecycle, and verifies that neither generic bootstrap
symbol is present:

```sh
cmake -S interfaces/loom/tests/static-link -B interfaces/loom/build/static -DCMAKE_BUILD_TYPE=Release
cmake --build interfaces/loom/build/static --target loom_static_link --parallel 4
ctest --test-dir interfaces/loom/build/static --output-on-failure
```

Run that build on supported AArch64 Linux (or configure a suitable cross
toolchain; execute the resulting tests only on a compatible runner).

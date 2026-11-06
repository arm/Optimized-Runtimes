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

# Runtime consumer contract

Contract version 10 defines an application/runtime boundary. A native runtime can
implement it directly or through an adapter. Backend capabilities determine the
supported execution and memory paths.

The declarations and operation-specific rules in [loom_runtime.h](loom_runtime.h)
are authoritative. The optional [Vulkan companion](loom_runtime_vulkan.h) defines
its native-resource profile. [README.md](README.md) is the usage entry point.

## Responsibilities

The runtime owns model execution, internal partitioning, scratch, weight packing,
placement and internal synchronization. The application owns pipeline scheduling,
frame identity, preprocessing and interpretation of output tensors, unless the
model package explicitly includes those operations. A camera allocation is not
necessarily a compatible model input.

This is a local library interface. Pointers, numeric file descriptors and opaque
handles are meaningful only in their process/platform context. Service transport,
remote resource transfer and device placement controls are outside this contract.

## Objects and lifetime

| Object | Owns or retains | Release condition |
| --- | --- | --- |
| Context | Runtime execution environment | All children released |
| Model | Loaded package, immutable metadata, potentially shared weights | All sessions destroyed |
| Session | Resolved shapes, bindings, scratch and mutable network state | Idle, all completions released |
| Buffer | External import reference or runtime allocation | Unbound, idle and unmapped |
| Completion | One accepted invocation's result | Terminal, no concurrent users |

One invocation may be outstanding per session. Additional sessions and parallel
execution are backend capabilities. Callers serialize mutation of shared objects;
completion waits/queries may run concurrently, but release must not race them.
Destruction never waits implicitly: `LRT_BUSY` preserves the object.

## Model and session preparation

`load_model` accepts a local path or self-contained serialized bytes. Source data
is borrowed only until the call returns. Runtime/package formats are backend
specific; this interface does not prescribe ONNX, PTE or VGF.

`create_session(model, NULL, ...)` uses intrinsic static input shapes. Dynamic
inputs need indexed concrete shape overrides within model constraints. Session
shapes are immutable; a different shape needs another session. Compilation is
permitted only by the selected policy. Preparation leaves the declared initial
network state intact and does not access application IO.

Data-dependent outputs need a capacity bound at session creation. Their binding
uses the requirement's `{0, NULL}` shape marker and NULL strides, with enough
storage for the bound. `get_outputs` supplies the actual concrete shape and
strides within that storage. The marker is distinct from a scalar tensor.
Inference never truncates output, silently recompiles or reallocates IO.

## Memory negotiation

`get_io_requirements` returns shape, required strides, capacity, alignment and
ranked memory options for each tensor. Each option is uniquely selected by
`(owner, accessibility, reference, transfer)`. Ranking is advisory. The caller
chooses an exact advertised configuration independently for every input/output.
Input and output counts and memory-option lists need not match. Ownership,
accessibility, resource reference, transfer mode, layout, capacity and alignment
are described for each tensor; matching option indices on two tensors have no
shared meaning. A runtime may support a memory category only for inputs, only
for outputs, or for a subset of tensors. There is no input-to-output memory
mapping implied by the API. Complete-set validation still checks whether the
selected combination can execute.

- `DIRECT` prohibits staging at the application/runtime IO boundary. It does
  not prohibit internal network transfers, packing or scratch.
- `STAGED` explicitly selects the advertised staging path. Required output
  transfers finish before successful completion.

Bind all inputs and outputs, then call `validate_bindings`. It checks the full
combination and finishes binding-dependent setup without executing inference or
accessing IO. Every successful bind/unbind invalidates validation and old output
views. A failed bind preserves the previous association and validation state.
There is no automatic substitution, allocation or fallback. IO aliasing is
unsupported, including overlaps between host and native-resource bindings.

Use [the memory guide](runtime-memory-use-cases.md) to choose between borrowed
host pointers, reusable imports and runtime allocations. Imports retain resource
references, not external producer/pool leases. Owners keep allocations and any
required mappings alive while bound and prevent reuse until all consumers finish.

## Execution and results

Both `infer` and `infer_async` require a validated binding set and return a
completion for accepted work. `infer` blocks until terminal completion;
`infer_async` returns after enqueue. At least one mode must be supported.
Unsupported modes return `LRT_UNSUPPORTED` without accepting work or switching modes.

A submission error means no work was accepted and the output completion is
untouched. After acceptance, execution success/failure is reported through the
completion. `wait` covers CPU work, device work, output transfers and promised
visibility. Timeout does not cancel work or permit early reuse. Terminal status,
including failure, means the runtime no longer accesses invocation IO.

`get_outputs` accepts only the latest successful invocation of that session.
Its metadata is borrowed until another accepted invocation, binding change or
session destruction. Keeping a completion does not pin output storage against
reuse. Externally owned output storage remains owned by the caller.

Failed accepted execution invalidates network state until reset or session
recreation. Failed outputs must not be consumed. Stateful models document their
initial/reset state; stateless reset is a no-op. No automatic retry is permitted.

## Extensions and compatibility

The exact global version covers the core and extension layouts. Shared plugins
export `lrt_runtime_version` and `lrt_get_api`; static consumers use a
runtime-specific getter as described in the [linking guide](README.md#static-and-shared-linking).
Each getter rejects a mismatched version before writing the caller's table. Verify the table's version query
before using other entries. All core entries are nonnull; unsupported operations
return `LRT_UNSUPPORTED`. Optional extension entries follow their capability bits.

DMA-BUF import does not imply CPU mapping, implicit synchronization or direct
execution. The Linux extension can supply producer sync-file fences, whole-
invocation fence export and bracketed CPU access. The Vulkan profile requires
the same logical device/queue, completed external work at handoff and explicit
runtime memory dependencies. Its bounded image representation does not cover
camera planes/modifiers or foreign-device handles.

Deferred: output leases (H2), cancellation, execution chaining, arbitrary image
preprocessing, foreign-device Vulkan sharing and native semaphore handoff.
Optional capabilities must be advertised by the concrete runtime before use.

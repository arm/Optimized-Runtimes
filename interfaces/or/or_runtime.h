/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * @file or_runtime.h
 * @brief Inference runtime C ABI, contract version 10.
 *
 * @details
 * A runtime may implement this consumer contract directly or through an adapter.
 * Preprocessing and output interpretation belong
 * to the application unless the model package explicitly includes them.
 * See README.md for usage and runtime-memory-use-cases.md for memory paths.
 *
 * ABI rules:
 * - C11 / C++11 compatible, using the native platform calling ABI and pointer
 *   width. This is not a wire format or a cross-architecture binary format.
 * - No C++ objects, exceptions, allocator ownership or packing pragmas cross
 *   the boundary. Implementations catch their own exceptions.
 * - One exact global version covers all records, signatures and extension
 *   tables. Layout or behavioral contract changes require a version increment.
 *   There is no per-record size/version or partial-table negotiation.
 * - Callers zero-initialize input records. Implementations validate counts,
 *   bounds and arithmetic overflow. Nonzero counts require nonnull arrays.
 *   Rank zero is a scalar, except the documented data-dependent output marker.
 * - Unless stated otherwise, output parameters are untouched on failure.
 *   Input descriptors are borrowed for the call; returned metadata is borrowed
 *   for its documented lifetime and must never be freed by the caller.
 *
 * @section or_category_paths Binding and lifetime
 *
 * create_session resolves shapes and creates independent mutable state.
 * get_io_requirements describes each tensor's layout and ranked memory options.
 * The caller selects and binds every input/output, then validates the full set.
 * bind_host_tensor borrows CPU storage; import_buffer retains an external
 * resource reference; allocate_buffer creates runtime storage. Imports and
 * bindings do not acquire an external pool-slot lease.
 *
 * infer returns a terminal completion; infer_async may return a pending one.
 * Completion covers all runtime IO access, not downstream consumer release.
 * Timeout never cancels work. Keep storage and mappings valid while bound,
 * and prevent external producers/consumers from racing runtime access.
 * Output handoff is borrowed runtime storage (H1) or external storage (H3).
 * Retained output leases (H2) are reserved and must not be advertised.
 *
 * Optional DMA-BUF and Vulkan profiles require explicit compatibility and
 * synchronization. A camera frame still needs the model's tensor format;
 * a handle or DIRECT transfer mode is not an end-to-end zero-copy guarantee.
 */
#ifndef OR_RUNTIME_H
#define OR_RUNTIME_H

#include <stdint.h>

#if defined(_WIN32)
#define OR_CALL __cdecl
#else
#define OR_CALL
#endif

/* Override when building a shared library, e.g. __declspec(dllexport). */
#ifndef OR_EXPORT
#define OR_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Consumer compile-time expectation; distinct from implementation release ID. */
#define OR_RUNTIME_VERSION UINT32_C(10)

/* Defined by the runtime library, immutable for its entire loaded lifetime.
 * Must equal both OR_RUNTIME_VERSION and api.get_runtime_version().
 * This and or_get_api are the stable bootstrap exports. */
OR_EXPORT extern const uint32_t or_runtime_version;
#define OR_WAIT_FOREVER UINT64_MAX
#define OR_DIM_DYNAMIC INT64_C(-1)

typedef struct or_context_impl *or_context;
typedef struct or_model_impl *or_model;
typedef struct or_session_impl *or_session;
typedef struct or_buffer_impl *or_buffer;
typedef struct or_completion_impl *or_completion;

/* Optional typed payloads; definitions and Vulkan includes stay in the companion. */
typedef struct or_vulkan_requirements or_vulkan_requirements;
typedef struct or_vulkan_resource or_vulkan_resource;
typedef struct or_vulkan_allocation_desc or_vulkan_allocation_desc;

/* Fixed-width tags deliberately avoid compiler-dependent C enum storage. */
typedef int32_t or_status;
enum {
    OR_OK = 0,
    OR_INVALID_ARGUMENT = 1,
    OR_UNSUPPORTED = 2,
    OR_INCOMPATIBLE = 3,
    OR_OUT_OF_MEMORY = 4,
    OR_BUSY = 5,
    OR_TIMEOUT = 6,
    OR_EXECUTION_FAILED = 7,
    OR_STATE_INVALID = 8,
    OR_BUFFER_TOO_SMALL = 9,
    OR_IO_ERROR = 10,
    OR_DEVICE_LOST = 11
};

typedef uint32_t or_dtype;
enum {
    OR_FLOAT32 = 1,
    OR_FLOAT16 = 2,
    OR_BFLOAT16 = 3,
    OR_INT8 = 4,
    OR_UINT8 = 5,
    OR_INT16 = 6,
    OR_INT32 = 7,
    OR_INT64 = 8,
    OR_BOOL8 = 9
};

typedef uint32_t or_io;
enum { OR_INPUT = 1, OR_OUTPUT = 2 };
typedef uint32_t or_access;
enum { OR_READ = 1, OR_WRITE = 2, OR_READ_WRITE = 3 };
/** @brief Who manages the backing allocation, relative to this ABI boundary. */
typedef uint32_t or_memory_owner;
enum { OR_OWNER_RUNTIME = 1, OR_OWNER_EXTERNAL = 2 };

/** @brief Minimum accessibility guaranteed by a selected configuration.
 *
 * @details
 * Bitmask: HOST | RUNTIME_DEVICE expresses access to the same allocation.
 * These are access guarantees, not simultaneous access or physical placement.
 * RUNTIME_DEVICE means at least one compatible runtime execution resource,
 * not all resources inside a heterogeneous runtime. OTHER_DEVICE requires a
 * defined sharing profile identifying that external access; not any device.
 * Zero is invalid. Additional access may exist; absent bits do not prohibit it.
 */
typedef uint32_t or_memory_accessibility;
enum {
    OR_ACCESSIBLE_HOST = 1,
    OR_ACCESSIBLE_RUNTIME_DEVICE = 2,
    OR_ACCESSIBLE_OTHER_DEVICE = 4
};

/** @brief Typed reference used to name storage, not an extra location field.
 *
 * @details
 * POINTER and DMA_BUF have core payloads. VULKAN_BUFFER and VULKAN_IMAGE use
 * typed pointers defined in or_runtime_vulkan.h. An unsupported profile must
 * not be advertised. No generic DEVICE address is accepted.
 */
typedef uint32_t or_memory_reference;
enum {
    OR_REFERENCE_POINTER = 1,
    OR_REFERENCE_DMA_BUF = 2,
    OR_REFERENCE_VULKAN_BUFFER = 3,
    OR_REFERENCE_VULKAN_IMAGE = 4
};

/** @brief Exact transfer behavior chosen by the caller, not a fallback policy.
 *
 * @details
 * DIRECT: no IO staging copy at the application/runtime boundary. Internal
 * network transfers, packing, scratch and partitioning remain runtime-owned.
 * STAGED: execution uses IO staging between the supplied allocation and internal
 * tensor storage. Output copies finish before successful wait returns.
 * Zero/unknown values are invalid; neither mode permits automatic substitution.
 */
typedef uint32_t or_memory_transfer;
enum { OR_TRANSFER_DIRECT = 1, OR_TRANSFER_STAGED = 2 };

/** @brief Classification and exact selection key; contains no opaque option ID.
 *
 * @details
 * Equality compares these four fields, never padding bytes. Accessibility is
 * a minimum guarantee of the selected path, not a caller's assertion. Binding
 * checks the concrete allocation. Owner does not transfer on binding/import.
 */
typedef struct or_memory_configuration {
    or_memory_owner owner;
    or_memory_accessibility accessibility;
    or_memory_reference reference;
    or_memory_transfer transfer;
} or_memory_configuration;

typedef uint32_t or_output_handoff;
enum {
    OR_HANDOFF_NONE = 0,     /* Input: no output handoff. */
    OR_HANDOFF_BORROWED = 1, /* H1, runtime-owned output. */
    OR_HANDOFF_RETAINED = 2, /* H2 reserved; MUST NOT be advertised. */
    OR_HANDOFF_EXTERNAL = 3  /* H3, externally managed output storage. */
};

typedef uint32_t or_compile_policy;
enum { OR_ALLOW_COMPILE = 0, OR_PRECOMPILED_ONLY = 1 };

/** @brief How a POINTER option can be supplied. At least one flag is required.
 *
 * @details
 * Runtime-owned POINTER options use allocate_buffer plus bind_tensor. External
 * POINTER options may permit borrowed bind_host_tensor, reusable imports, or
 * both. These flags do not change resource ownership or copy policy.
 */
typedef uint32_t or_pointer_binding_flags;
enum {
    OR_POINTER_BORROWED_BIND = 1,
    OR_POINTER_REUSABLE_IMPORT = 2,
    OR_POINTER_RUNTIME_ALLOCATION = 4
};

/** @brief One ranked configuration and the requirements needed to select it.
 *
 * @details
 * All counts/arrays and nested metadata are immutable until session destruction.
 * Each option specifies one exact transfer mode in configuration. Neither rank
 * nor DIRECT is a measured end-to-end performance claim. The caller selects the
 * complete configuration; bind accepts that exact choice or returns an error.
 * Runtime never substitutes another configuration or transfer mode. DIRECT and
 * STAGED options with otherwise equal fields are distinct selectable options.
 *
 * If two alternatives with the same configuration cannot share these
 * constraints, advertise a conservative common subset, not duplicate keys.
 * This limits expressiveness intentionally; distinct profiles may be needed
 * later. Capacity/alignment are additional to the containing tensor requirement
 * (satisfy both). Every option uses that tensor's single required stride layout.
 * Dtype/layout come from model metadata; no implicit conversion is allowed.
 *
 * POINTER uses pointer_binding_flags, other references set that field to zero.
 * DMA_BUF covers tensor byte ranges, not arbitrary video/image plane layouts.
 * HOST accessibility for DMA_BUF requires the CPU_MAP extension capability.
 *
 * vulkan is required for Vulkan references and NULL otherwise. It identifies
 * the device/queue and concrete native constraints, with companion definitions.
 * Native options require OR_EXTENSION_VULKAN support. Runtime-owned POINTER
 * options require HOST access; runtime-owned Vulkan options may be device-only.
 * Runtime-owned DMA-BUF export still needs a separately defined profile.
 * OTHER_DEVICE accessibility cannot be advertised without a defined profile
 * describing which external resource/device can access the allocation.
 *
 * Advertised pointer methods must agree with context capabilities: BORROWED_BIND needs HOST_BIND,
 * REUSABLE_IMPORT needs HOST_IMPORT, and RUNTIME_ALLOCATION requires that the
 * runtime can allocate the requested configuration. No capability bit alone
 * substitutes for an advertised per-tensor option.
 *
 * Handoff is NONE for inputs, BORROWED for runtime outputs, EXTERNAL for external
 * outputs. H2 is not implemented. Ranking does not depend on current bindings;
 * options are returned in tensor requirements for resolved session shapes and
 * the actual host environment, rather than generic backend capabilities.
 */
typedef struct or_memory_option {
    or_memory_configuration configuration;
    or_output_handoff output_handoff;
    uint64_t minimum_byte_capacity;
    uint64_t byte_alignment; /* Nonzero power of two. */
    or_pointer_binding_flags pointer_binding_flags;
    const or_vulkan_requirements *vulkan;
} or_memory_option;

/* Context capabilities describe available facilities, not compatibility of
 * every model/layout/allocation. Session creation/import/bind still validates. */
typedef uint64_t or_capability_flags;
#define OR_CAP_HOST_IMPORT (UINT64_C(1) << 0)
#define OR_CAP_DMA_BUF_IMPORT (UINT64_C(1) << 1)
#define OR_CAP_ASYNC_INFERENCE (UINT64_C(1) << 2)
#define OR_CAP_PARALLEL_SESSIONS (UINT64_C(1) << 3)
#define OR_CAP_DYNAMIC_SHAPES (UINT64_C(1) << 4)
/* Direct pointer binding is independent of reusable HOST_IMPORT support. */
#define OR_CAP_HOST_BIND (UINT64_C(1) << 5)
#define OR_CAP_SYNC_INFERENCE (UINT64_C(1) << 6)
/* At least one inference mode is implemented. Both core entries are nonnull;
 * an unimplemented mode returns UNSUPPORTED without accepting work. There is
 * no caller-visible fallback from one execution mode to the other. */

typedef struct or_capabilities {
    or_capability_flags flags;
    uint32_t max_rank;
    /* Borrowed UTF-8 strings, valid until context destruction. */
    const char *runtime_id;
    const char *runtime_version; /* Implementation release; not ABI compatibility. */
} or_capabilities;

/** @brief Source of a model/package, independent of inference tensor memory. */
typedef uint32_t or_model_source_kind;
enum { OR_MODEL_SOURCE_PATH = 1, OR_MODEL_SOURCE_MEMORY = 2 };

/** @brief Load from a local path or a contiguous, host-readable byte range.
 *
 * @details
 * PATH is a nonempty NUL-terminated UTF-8 local package/file path, not an RPC
 * endpoint. MEMORY supplies nonnull data and a nonzero byte_length representable
 * in the runtime's host address space. Bytes are the backend's serialized model
 * or self-contained package format, not an executable pointer or tensor input.
 * The tag selects the only active union member; unknown tags are invalid.
 *
 * Both the descriptor and its selected storage are borrowed only for load_model.
 * Caller keeps memory readable and unchanged until the call returns. The runtime
 * owns any retained bytes/state on success (copying or decoding as needed), and
 * never frees caller memory. The caller may free/unmap the source on return,
 * including on failure. No deferred access to the original range is permitted.
 * This deliberately omits a borrowed-until-unload mode and lifetime callbacks.
 *
 * MEMORY must be self-contained: there is no implicit base path, external-file
 * resolver or automatic filesystem fallback. Unsupported package formats or
 * packages requiring external resources fail explicitly. Implementations must
 * support MEMORY for their loadable self-contained serialized format; consumers
 * need not materialize a temporary file. Format/security validation applies to
 * both sources; no generic ONNX/PTE/VGF format is implied by this ABI.
 */
typedef struct or_model_source {
    or_model_source_kind kind;
    union {
        const char *path;
        struct {
            const void *data;
            uint64_t byte_length;
        } memory;
    } source;
} or_model_source;

typedef struct or_quantization {
    uint32_t parameter_count; /* 0 = none, 1 = per-tensor, otherwise per-axis. */
    int32_t axis;             /* -1 for none/per-tensor. */
    const double *scales;
    const int64_t *zero_points;
} or_quantization;

/* Inputs/outputs are network tensors, not structured perception results.
 * Optional interpretation metadata belongs in the model/package descriptor.
 * The application owns output decoding and structured results. If decoding/NMS is fused
 * into the network, the descriptor must document the resulting tensor format.
 */
typedef struct or_tensor_info {
    const char *name; /* Stable name; indices follow description order. */
    or_dtype dtype;
    uint32_t rank;
    const int64_t *dimensions;          /* OR_DIM_DYNAMIC where runtime-selected. */
    const uint64_t *minimum_dimensions; /* rank entries, including fixed dims. */
    const uint64_t *maximum_dimensions; /* 0 = no declared upper bound. */
    const char *layout;                 /* Versioned identifier, e.g. "tensor/nchw/v1". */
    or_quantization quantization;
} or_tensor_info;

typedef struct or_model_info {
    const char *model_id;
    uint32_t input_count;
    uint32_t output_count;
    /* Tensor descriptors are borrowed for the model lifetime. */
    const or_tensor_info *const *inputs;
    const or_tensor_info *const *outputs;
    uint32_t stateful; /* 0 or 1; state is session-local. */
    const char *state_schema;
    const char *initial_and_reset_state_json;
} or_model_info;

typedef struct or_shape {
    uint32_t rank;
    const uint64_t *dimensions;
} or_shape;

/** @brief Resolve one dynamic input using its model-description index.
 *
 * @details
 * Supplies the entire concrete shape for that input; fixed dimensions must
 * match the model and dynamic ones must satisfy its constraints. Batch size
 * is adjustable only if that dimension is dynamic. This cannot redefine the
 * network's input count or dtype/layout. Repeating a static shape is allowed
 * but unnecessary; changing a static dimension is rejected.
 */
typedef struct or_input_shape_override {
    uint32_t input_index;
    or_shape shape;
} or_input_shape_override;

/** @brief Optional session settings; not a second model input description.
 *
 * @details
 * Zero initialization means ALLOW_COMPILE and no overrides. Each input_index
 * may occur at most once. Unmentioned inputs use their intrinsic concrete model
 * shapes. An unresolved dynamic input requires an override; no batch/dimension
 * value is guessed from minimum/maximum bounds. Counts describe only this
 * sparse override list, never the number of network inputs.
 * All descriptors/dimensions are borrowed for create_session only; the runtime
 * copies resolved shapes. There is no externally visible warmup operation.
 */
typedef struct or_session_options {
    or_compile_policy compile_policy;
    uint32_t shape_override_count;
    const or_input_shape_override *shape_overrides;
} or_session_options;

/** @brief Resolved tensor requirements, including ranked memory choices.
 *
 * @details
 * Returned by get_io_requirements, with all nested metadata immutable and
 * borrowed until session destruction. memory_options is best-to-worst runtime
 * preference, unique by all four configuration fields; the client may prefer another
 * option for its pipeline. Zero options means this tensor cannot be bound.
 * Binding checks allocation compatibility; validate_bindings checks the complete
 * set. Per-tensor support does not guarantee every combination works.
 * Input and output option lists are independent; neither corresponding tensor
 * indices nor option indices imply a pairing. Counts, ownership, accessibility,
 * reference types, transfer modes and constraints may differ for every tensor.
 * A category advertised for an input need not be available for any output, and
 * vice versa. The ABI imposes no matching-configuration rule across directions.
 *
 * All memory options share this tensor's required strides. A supplied binding
 * stride array must match; NULL selects these strides, not another layout.
 * Different physical layouts are not negotiated for a single tensor.
 * For a data-dependent output, shape is {0, NULL} and byte_strides is NULL.
 * Bind that same marker with NULL strides and capacity satisfying both the
 * tensor and selected option. This marker is not a scalar output shape.
 * get_outputs returns the actual concrete shape and byte
 * strides within the bound capacity; no truncation or IO reallocation occurs.
 * A backend unable to establish a capacity bound rejects session creation.
 * All inputs and outputs start unbound. There is no implicit output option or
 * allocation. For runtime-owned storage, caller selects a configuration, calls
 * allocate_buffer and binds its handle. NULL binding only unbinds the tensor.
 */
typedef struct or_tensor_requirement {
    or_shape shape;         /* Concrete; or rank=0 and data_dependent=1. */
    uint32_t data_dependent; /* Output shape is known only after execution. */
    uint64_t minimum_byte_capacity;
    uint64_t byte_alignment;
    /* Required strides for concrete shape (rank entries); NULL for rank zero. */
    const uint64_t *byte_strides;
    uint32_t memory_option_count;
    const or_memory_option *memory_options;
} or_tensor_requirement;

typedef struct or_io_requirements {
    uint32_t input_count;
    uint32_t output_count;
    const or_tensor_requirement *const *inputs;
    const or_tensor_requirement *const *outputs;
} or_io_requirements;

/** @brief Description of externally owned storage to import as a reusable view.
 *
 * @details
 * Import creates a runtime resource reference, not ownership of the producer's
 * pool slot. The caller prevents external reuse while execution/consumers need
 * the contents. Host storage remains alive while imported; DMA-BUF import
 * duplicates the FD but does not acquire a producer-buffer lease.
 */
typedef struct or_memory_desc {
    or_memory_reference reference; /* Tagged payload; ownership is External. */
    or_access access;              /* Access by inference, not CPU mapping rights. */
    uint64_t allocation_byte_size;
    uint64_t allocation_offset;
    uint64_t view_byte_length;
    union {
        void *host_base;    /* Allocation byte zero; borrowed, never freed. */
        int32_t dma_buf_fd; /* Duplicated with close-on-exec on import success. */
        const or_vulkan_resource *vulkan;
    } handle;
    /* Vulkan: typed payload contains bounds; the three generic range fields
     * above must be zero. Payload graph is copied on success; native objects
     * remain externally owned and alive until the import is released. */
} or_memory_desc;

/** @brief Request reusable runtime-managed storage in a supported profile.
 *
 * @details
 * Configuration requests minimum accessibility and eligibility for the selected
 * transfer mode, not physical placement/export. Binding validates tensor-specific
 * compatibility. Transfer is not a permanent property of the backing allocation.
 * Unsupported allocation profiles return OR_UNSUPPORTED; DMA-BUF import
 * support alone does not imply DMA-BUF allocation/export support.
 */
typedef struct or_allocation_desc {
    or_memory_configuration configuration; /* owner must be RUNTIME. */
    or_access access;
    uint64_t byte_length;
    uint64_t byte_alignment;
    /* Required for Vulkan references; NULL otherwise. Borrowed for call;
     * runtime copies any native metadata it retains. */
    const or_vulkan_allocation_desc *vulkan;
} or_allocation_desc;

/** @brief Actual owner/reference and guaranteed accessibility of a buffer view.
 *
 * @details
 * Import may initially establish only limited accessibility; tensor binding
 * validates any additional access promised by its selected configuration.
 * host_data alone does not establish coherence or consumer readiness.
 */
typedef struct or_buffer_info {
    /* Storage facts; transfer mode belongs to a tensor binding, not an allocation. */
    or_memory_owner owner;
    or_memory_accessibility accessibility;
    or_memory_reference reference;
    or_access access;
    uint64_t view_byte_length;
    /* Nonnull only for POINTER with HOST access. Borrowed for buffer lifetime. An idle
     * buffer still bound to a session may be filled/read by the caller. */
    void *host_data;
    /* Nonnull only for Vulkan references. Borrowed metadata for buffer lifetime;
     * native handles do not transfer destruction ownership. */
    const or_vulkan_resource *vulkan;
} or_buffer_info;

typedef struct or_tensor_binding {
    or_buffer buffer;
    uint64_t byte_offset; /* Relative to the imported view, not allocation. */
    uint64_t byte_length;
    or_shape shape;
    const uint64_t *byte_strides;           /* Must match requirements; NULL uses them. */
    or_memory_configuration configuration; /* Exact advertised configuration. */
} or_tensor_binding;

/** @brief Borrowed host tensor, independent of or_buffer.
 *
 * @details
 * data points at the first
 * tensor byte; byte_length is accessible capacity from that address, with no
 * additional allocation offset. Dtype/layout come from model metadata.
 * Inference reads inputs without modifying them; outputs must be writable.
 * Shape/stride descriptors are copied at binding; storage remains caller-owned.
 */
typedef struct or_host_tensor_binding {
    void *data;
    uint64_t byte_length;
    or_shape shape;
    const uint64_t *byte_strides;           /* Must match requirements; NULL uses them. */
    or_memory_configuration configuration; /* EXTERNAL / includes HOST / POINTER. */
} or_host_tensor_binding;

typedef struct or_completion_info {
    uint32_t terminal; /* 0 = pending, 1 = finished including failure. */
    or_status result; /* Meaningful only when terminal=1. */
} or_completion_info;

/** @brief One result view; metadata lifetime is separate from external storage.
 *
 * @details
 * @see or_output_handoff
 */
typedef struct or_output {
    uint32_t tensor_index;
    or_tensor_binding tensor; /* Actual shape, range, strides and buffer. */
    /* Direct host output: tensor.buffer=NULL, byte_offset=0, and this points to
     * caller storage. tensor still describes actual shape/strides and capacity.
     * Otherwise NULL; access tensor.buffer normally. Reporting a direct host
     * output does not create an import handle. */
    void *host_data;
} or_output;

typedef struct or_outputs {
    uint32_t count;
    const or_output *const *items;
} or_outputs;

/* Optional Linux DMA-BUF synchronization / CPU access extension. Kernel and
 * driver support must be validated; importing a DMA-BUF does not imply
 * implicit synchronization, CPU mapping or direct binding support. */
#define OR_EXTENSION_LINUX_DMA_BUF "or.runtime.linux-dma-buf/1"
typedef uint64_t or_dma_buf_capability_flags;
/* SYNC_FILE covers asynchronous producer-fence submission and whole-invocation
 * fence export; advertise only with ASYNC_INFERENCE and both entries present. */
#define OR_DMA_BUF_CAP_SYNC_FILE (UINT64_C(1) << 0)
#define OR_DMA_BUF_CAP_CPU_MAP (UINT64_C(1) << 1)

typedef struct or_cpu_mapping_impl *or_cpu_mapping;
typedef struct or_cpu_mapping_info {
    void *data; /* First byte of imported view, not allocation. */
    uint64_t byte_length;
} or_cpu_mapping_info;

typedef struct or_linux_dma_buf_api {
    or_dma_buf_capability_flags capabilities;
    /**
     * @brief Enqueue inference with external producer sync-file dependencies.
     *
     * @details
     * Honors producer fences before touching IO without waiting for fences/inference on this
     * calling thread. Only external fence FDs are accepted; runtime-completion chaining is
     * unsupported. Duplicate FDs on acceptance only; originals stay caller-owned. Zero
     * count/NULL supplies no fences.
     *
     * @pre ASYNC_INFERENCE and SYNC_FILE are advertised. The session is idle, state/bindings
     * validated, FD array valid and owners/mappings protected through completion. Native
     * profiles must permit this fence combination.
     *
     * @post Acceptance returns a pending or terminal completion, invalidates previous outputs
     * and retains duplicate FDs for the invocation. Use wait/get_completion_info,
     * get_outputs on success, then release_completion. Rejection retains no FDs/work
     * and leaves out_completion untouched.
     *
     * @return OR_OK: work accepted, not necessarily complete/successful. OR_UNSUPPORTED:
     * mode/fence/resource combination unavailable. OR_BUSY: session/resource in use.
     * OR_STATE_INVALID: state/bindings not ready. OR_INVALID_ARGUMENT: invalid
     * fence arguments. FD/allocation failures return OR_IO_ERROR or
     * OR_OUT_OF_MEMORY without acceptance.
     */
    or_status(OR_CALL *infer_async_with_fences)(or_session session,
                                                  uint32_t fence_count,
                                                  const int32_t *acquire_sync_file_fds,
                                                  or_completion *out_completion);
    /**
     * @brief Export whole-invocation ordering as a caller-owned sync-file FD.
     *
     * @details
     * Signalling represents ordering, not execution success; also inspect the completion
     * result. The fence covers CPU work, device work and output transfers. Export is
     * unsupported if sync_file cannot represent all that ordering.
     *
     * @pre A live completion, writable FD output and advertised SYNC_FILE support; no
     * concurrent completion release.
     *
     * @post On success caller owns a new close-on-exec FD and must close it. It may be
     * supplied to a compatible downstream consumer; still inspect execution success and
     * prevent premature storage reuse. Failure leaves the FD output untouched.
     *
     * @return OR_OK: fence FD exported. OR_UNSUPPORTED: ordering cannot be represented.
     * OR_INVALID_ARGUMENT: invalid arguments. OR_IO_ERROR or OR_OUT_OF_MEMORY:
     * fence export failed.
     */
    or_status(OR_CALL *export_completion_fence)(or_completion completion,
                                                  int32_t *out_sync_file_fd);
    /**
     * @brief Begin bracketed CPU access to an imported DMA-BUF view.
     *
     * @details
     * This performs the profile CPU mapping/cache-access step, not an execution fence.
     * Mapping may fail for a particular resource. The returned pointer starts at imported-
     * view byte zero.
     *
     * @pre CPU_MAP is advertised. The buffer is idle; external producers/consumers have
     * completed. Requested access is valid and no conflicting mapping/access exists.
     *
     * @post On success mapping/info are returned and buffer binding, inference and release
     * are blocked until end_cpu_access. Access only within the returned range and
     * requested mode. Failure creates no mapping and leaves outputs untouched.
     *
     * @return OR_OK: mapping/access begun. OR_UNSUPPORTED: resource mapping unavailable.
     * OR_BUSY: conflicting use/mapping. OR_INVALID_ARGUMENT: invalid
     * arguments/access. OR_IO_ERROR or OR_OUT_OF_MEMORY: mapping/cache-access setup
     * failed.
     */
    or_status(OR_CALL *begin_cpu_access)(or_buffer buffer,
                                           or_access access,
                                           or_cpu_mapping *out_mapping,
                                           or_cpu_mapping_info *out_info);
    /**
     * @brief End a DMA-BUF CPU access interval and release its mapping.
     *
     * @details
     * Completes required cache-access bracketing. This does not establish downstream
     * execution completion or acquire/release an external pool-slot lease.
     *
     * @pre A live mapping created by this table; all CPU accesses through it have finished
     * and no concurrent users remain.
     *
     * @post On success mapping and pointer are invalid; the buffer may be bound/submitted or
     * released under normal lifetime rules. On failure output arguments/state follow
     * the core failure rules; do not assume access has ended.
     *
     * @return OR_OK: access ended and mapping released. OR_INVALID_ARGUMENT: invalid
     * mapping argument. OR_IO_ERROR: cache-access/unmap operation failed.
     */
    or_status(OR_CALL *end_cpu_access)(or_cpu_mapping mapping);
} or_linux_dma_buf_api;

typedef struct or_api {
    /**
     * @brief Query the global ABI contract version.
     *
     * @details
     * This context-free query returns the immutable version also exported as
     * or_runtime_version. It is distinct from the implementation release string.
     *
     * @pre Bootstrap must have succeeded for the requested version; the runtime library
     * remains loaded.
     *
     * @post No objects or state change. Verify equality with OR_RUNTIME_VERSION before
     * calling other table entries.
     *
     * @return The runtime contract version. A mismatch means the table must not be used.
     */
    uint32_t(OR_CALL *get_runtime_version)(void);

    /**
     * @brief Create a runtime-owned execution environment.
     *
     * @details
     * The environment may span CPUs and accelerators. Runtime/package requirements determine
     * placement. Hardware availability is checked here, at model loading or at session
     * creation as requirements become known.
     *
     * @pre Bootstrap/version verification succeeded; out_context points to writable handle
     * storage.
     *
     * @post On success the context owns subsequent models, sessions, buffers and completions
     * and must outlive them. Call get_capabilities, load_model or buffer operations
     * next. Failure leaves out_context untouched.
     *
     * @return OR_OK: context created. OR_INVALID_ARGUMENT: invalid output argument.
     * OR_UNSUPPORTED or OR_INCOMPATIBLE: required execution environment
     * unavailable. OR_OUT_OF_MEMORY: context allocation failed.
     */
    or_status(OR_CALL *create_context)(or_context *out_context);
    /**
     * @brief Query context facilities and implementation identity.
     *
     * @details
     * Capabilities describe facilities, not compatibility of every tensor, package or
     * allocation. At least one inference mode is advertised; per-tensor requirements and
     * binding validation remain authoritative.
     *
     * @pre A live context and writable output record; serialize mutation of that context.
     *
     * @post No runtime state changes. Returned strings are borrowed until context
     * destruction. Load a model or inspect supported facilities next.
     *
     * @return OR_OK: capabilities returned. OR_INVALID_ARGUMENT: invalid arguments.
     */
    or_status(OR_CALL *get_capabilities)(or_context, or_capabilities *out);
    /**
     * @brief Load and validate a path-backed or memory-backed model package.
     *
     * @details
     * Packaging and hardware compatibility are backend-owned. Source descriptors and bytes
     * are borrowed only during this call. The runtime copies or decodes retained data; memory
     * packages are self-contained and have no implicit filesystem fallback. See
     * or_model_source.
     *
     * @pre A live context, a valid source tag with its required path or byte range, and
     * writable out_model. Source memory remains readable and unchanged until return.
     *
     * @post On success the model belongs to the context; describe_model or create_session may
     * follow. On either success or failure source storage may be released. Failure
     * retains no source references and leaves out_model untouched.
     *
     * @return OR_OK: model loaded. OR_INVALID_ARGUMENT: invalid source tag, required
     * pointer or byte length. OR_UNSUPPORTED: format/resource mechanism unavailable.
     * OR_INCOMPATIBLE: model/hardware incompatible. OR_IO_ERROR: package cannot be
     * read. OR_OUT_OF_MEMORY: loading allocation failed.
     */
    or_status(OR_CALL *load_model)(or_context, const or_model_source *, or_model *out_model);
    /**
     * @brief Query immutable model and tensor metadata.
     *
     * @details
     * Tensor order defines binding indices. Shapes may contain dynamic dimensions; resolve
     * those through session options rather than guessing from bounds.
     *
     * @pre A live loaded model and writable output record.
     *
     * @post No state changes. Returned metadata is borrowed until unload_model. Use it to
     * create a session and configure application preprocessing/postprocessing.
     *
     * @return OR_OK: description returned. OR_INVALID_ARGUMENT: invalid arguments.
     */
    or_status(OR_CALL *describe_model)(or_model, or_model_info *out);

    /**
     * @brief Create independent execution state and resolve input shapes.
     *
     * @details
     * NULL options select static model shapes. Dynamic inputs require indexed overrides;
     * fixed dimensions cannot change. Models may share weights/compiled state, but sessions
     * have independent bindings, scratch and network state. Shapes are immutable for the
     * session. Compilation occurs only when permitted. Warmup uses private storage and
     * preserves initial state; no application IO is accessed. Final binding-dependent setup
     * occurs in bind/validate operations. Data-dependent outputs require a capacity bound. No
     * inference-time compilation or IO reallocation is permitted.
     *
     * @pre A live model and writable out_session. Options and shape arrays remain valid for
     * this call. Overrides are unique, in range and satisfy model constraints;
     * PRECOMPILED_ONLY forbids required compilation.
     *
     * @post On success the session is idle, initialized and unbound. Query
     * get_io_requirements, then select/bind storage. Another shape requires another
     * session. Failure leaves out_session untouched.
     *
     * @return OR_OK: session created. OR_INVALID_ARGUMENT: invalid/duplicate index, rank or
     * shape constraint, or unresolved input shape. OR_UNSUPPORTED: valid
     * shape/policy, additional session or bounded output cannot be supported.
     * OR_OUT_OF_MEMORY: preparation allocation failed. OR_INCOMPATIBLE: required
     * execution resources incompatible.
     */
    or_status(OR_CALL *create_session)(or_model,
                                         const or_session_options *options,
                                         or_session *out_session);
    /**
     * @brief Query resolved tensor requirements and ranked memory options.
     *
     * @details
     * Each input/output has its own shape, capacity, alignment, strides and options. Ranking
     * is advisory; matching reference types or option row numbers do not establish
     * compatibility of the complete binding set. See or_category_paths.
     *
     * @pre A live session and writable output record.
     *
     * @post No state changes. Returned nested metadata is immutable and borrowed until
     * session destruction. Select exact configurations and allocate/import/bind
     * resources next.
     *
     * @return OR_OK: requirements returned. OR_INVALID_ARGUMENT: invalid arguments.
     */
    or_status(OR_CALL *get_io_requirements)(or_session, or_io_requirements *out);
    /**
     * @brief Create a reusable reference to externally owned storage.
     *
     * @details
     * Import once per stable allocation, then bind compatible views or pool slots. Use
     * bind_host_tensor instead for changing ordinary CPU pointers. Import does not acquire an
     * external pool-slot lease or establish tensor compatibility. Host storage is borrowed; a
     * DMA-BUF FD is duplicated with close-on-exec on success. Vulkan metadata is copied,
     * while native objects stay caller-owned.
     *
     * @pre A live context, valid or_memory_desc and writable out_buffer. External owners
     * keep storage/native objects alive until the import is released and prevent
     * conflicting reuse.
     *
     * @post On success the buffer belongs to this context; query get_buffer_info or
     * bind_tensor next. Binding to another context requires a supported new import.
     * Failure consumes no caller handle and leaves out_buffer untouched.
     *
     * @return OR_OK: import retained. OR_INVALID_ARGUMENT: malformed descriptor/range.
     * OR_UNSUPPORTED: import mechanism unavailable. OR_INCOMPATIBLE: resource
     * cannot be imported into this environment. OR_OUT_OF_MEMORY: import allocation
     * failed. OR_IO_ERROR: external handle operation failed.
     */
    or_status(OR_CALL *import_buffer)(or_context,
                                        const or_memory_desc *,
                                        or_buffer *out_buffer);
    /**
     * @brief Allocate runtime-owned storage for a selected memory configuration.
     *
     * @details
     * The request describes minimum accessibility and eligibility for a transfer mode, not
     * physical placement or export. Binding still validates tensor-specific compatibility.
     * DMA-BUF import support does not imply allocation/export support. Native allocations
     * require the companion descriptor.
     *
     * @pre A live context, valid runtime-owned allocation configuration with required
     * size/alignment and writable out_buffer.
     *
     * @post On success the caller owns an or_buffer handle, while the runtime manages its
     * allocation. Query get_buffer_info, then bind_tensor. Release the handle after
     * unbinding. Failure leaves out_buffer untouched.
     *
     * @return OR_OK: allocation created. OR_INVALID_ARGUMENT: malformed
     * configuration/size/alignment. OR_UNSUPPORTED: allocation profile unavailable.
     * OR_INCOMPATIBLE: requested resource constraints incompatible.
     * OR_OUT_OF_MEMORY: allocation failed.
     */
    or_status(OR_CALL *allocate_buffer)(or_context,
                                          const or_allocation_desc *,
                                          or_buffer *out_buffer);
    /**
     * @brief Query buffer storage properties and borrowed access views.
     *
     * @details
     * Transfer mode belongs to a tensor binding, not the allocation. host_data is exposed
     * only for host-accessible POINTER storage; Vulkan references return native metadata
     * instead. A pointer alone does not establish readiness or coherence.
     *
     * @pre A live buffer and writable output record.
     *
     * @post No state changes. Metadata/native views are borrowed until release_buffer; use
     * them only under the applicable synchronization rules. Fill idle host storage or
     * bind the buffer next.
     *
     * @return OR_OK: properties returned. OR_INVALID_ARGUMENT: invalid arguments.
     */
    or_status(OR_CALL *get_buffer_info)(or_buffer, or_buffer_info *out);

    /**
     * @brief Bind or unbind one input/output using a buffer view.
     *
     * @details
     * All four configuration fields must match an advertised option. Validate shape, strides,
     * bounds, alignment, access and non-overlap; IO aliasing is unsupported. POINTER methods
     * must be permitted by pointer_binding_flags. DIRECT excludes boundary staging, not
     * internal transfers/packing. NULL binding unbinds without choosing or allocating a
     * replacement. Descriptors are copied and the buffer is retained while bound. Binding
     * never reads inputs or writes outputs; staging occurs during execution.
     *
     * @pre The session is idle. For nonnull binding, buffer/session contexts match and
     * storage satisfies the tensor and selected option. No conflicting mapping or
     * external access is active.
     *
     * @post On success this association replaces either previous binding path, invalidating
     * prior result views and complete-set validation, even for identical arguments.
     * Call validate_bindings once all IO is bound. Failure preserves the old binding
     * and validation state.
     *
     * @return OR_OK: binding replaced or removed. OR_BUSY: session/resource is in use.
     * OR_INVALID_ARGUMENT: invalid tensor index, descriptor, shape, strides or
     * alignment. OR_BUFFER_TOO_SMALL: insufficient view capacity. OR_UNSUPPORTED:
     * unadvertised configuration/method. OR_INCOMPATIBLE:
     * context/resource cannot meet the selection. Access/aliasing failures may
     * instead report OR_INVALID_ARGUMENT. Setup failures
     * return their relevant status.
     */
    or_status(OR_CALL *bind_tensor)(or_session,
                                      or_io,
                                      uint32_t index,
                                      const or_tensor_binding *binding);

    /**
     * @brief Borrow CPU tensor storage without creating an import handle.
     *
     * @details
     * Select an advertised EXTERNAL / includes HOST / POINTER configuration permitting
     * BORROWED_BIND, with exact transfer mode. Shape, strides, capacity, alignment, non-
     * overlap and failure-atomic replacement follow bind_tensor. Registration/setup may have
     * cost. NULL unbinds even without HOST_BIND support and never allocates replacement
     * output. A fixed pointer can be bound/validated once, then refilled between completed
     * invocations; a changing pointer requires rebinding and validation.
     *
     * @pre The session is idle. Nonnull binding requires HOST_BIND. Keep
     * owner/storage/mapping alive while bound and exclude conflicting
     * producers/consumers. Required strides and configuration match advertised
     * requirements.
     *
     * @post On success previous association is replaced; old outputs and binding-set
     * validation become invalid. Validate before inference. After terminal execution
     * and unbinding, caller may unmap/release storage. No release_buffer call is
     * needed. Failure preserves the previous association and validation.
     *
     * @return OR_OK: pointer bound or tensor unbound. OR_BUSY: session/resource is in use.
     * OR_INVALID_ARGUMENT: invalid index, pointer, shape, strides or alignment.
     * OR_BUFFER_TOO_SMALL: insufficient capacity. OR_UNSUPPORTED: HOST_BIND or
     * selected method/configuration unavailable. OR_INVALID_ARGUMENT or
     * OR_INCOMPATIBLE: resource/access/non-overlap requirements cannot be met.
     * Setup failures return their relevant status.
     */
    or_status(OR_CALL *bind_host_tensor)(or_session,
                                           or_io,
                                           uint32_t index,
                                           const or_host_tensor_binding *binding);

    /**
     * @brief Validate and finish setting up the complete IO binding set.
     *
     * @details
     * Checks every tensor is bound, configuration combinations, aliasing and shared resource
     * limits. May reserve explicitly selected staging and prepare backend binding snapshots,
     * but never compiles, executes or accesses IO. This is not an atomic batch bind/rollback.
     * Revalidating an unchanged valid set is a no-op; refilling existing input storage does
     * not invalidate it.
     *
     * ```mermaid
     * stateDiagram-v2
     *     [*] --> Unvalidated
     *     Unvalidated --> Ready: validate_bindings succeeds
     *     Ready --> Unvalidated: bind or unbind succeeds
     *     Ready --> Executing: inference accepted
     *     Executing --> Ready: terminal success
     *     Executing --> InvalidState: terminal failure
     *     InvalidState --> Ready: reset succeeds, bindings still validated
     *     InvalidState --> Unvalidated: reset succeeds, bindings changed
     * ```
     *
     * @pre The session is idle; selected buffers and owners remain valid.
     *
     * @post On success exact selections are preserved and inference may follow without more
     * binding setup. On failure associations remain, but the set is invalid and no
     * inference can be accepted; change bindings and retry. Every successful
     * bind/unbind invalidates validation.
     *
     * @return OR_OK: set validated. OR_BUSY: session/resource in use. OR_STATE_INVALID:
     * missing binding. OR_INCOMPATIBLE: configurations/aliasing/resources
     * incompatible. OR_OUT_OF_MEMORY or another setup error: preparation failed.
     */
    or_status(OR_CALL *validate_bindings)(or_session);

    /**
     * @brief Execute synchronously and return a terminal completion.
     *
     * @details
     * Blocks until the runtime no longer accesses invocation IO. Accepted work returns OR_OK
     * even when execution fails: the terminal completion carries that result. Successful
     * execution establishes output visibility without a separate wait. Core submission
     * accepts no runtime-completion dependencies; producer fences belong to the optional
     * asynchronous extension. Neither completion nor fence owns external pool slots.
     *
     * @pre SYNC_INFERENCE is supported. Session is idle, state is valid and all IO has passed
     * validation since the last binding change. External producers and prior consumers
     * have completed. Keep IO storage/mappings alive and unchanged during execution.
     *
     * @post On acceptance previous output views become invalid; out_completion is terminal.
     * Inspect get_completion_info or call get_outputs, then release_completion. Failed
     * execution invalidates state/outputs until reset/recreation. Rejection accepts no
     * work, retains no invocation resources and leaves out_completion untouched;
     * unsupported mode leaves bindings, outputs and state unchanged.
     *
     * @return OR_OK: work accepted and completed; inspect completion/result for execution
     * success. OR_UNSUPPORTED: synchronous mode unavailable. OR_BUSY:
     * session/resources already in use. OR_STATE_INVALID: state invalid or binding
     * set unvalidated/incomplete. Invalid arguments or acceptance/setup failures
     * return their relevant status with no accepted work.
     */
    or_status(OR_CALL *infer)(or_session, or_completion *out_completion);

    /**
     * @brief Enqueue inference without waiting for execution completion.
     *
     * @details
     * Uses the same acceptance and lifetime rules as infer. No blocking synchronous fallback
     * or wait for inference/producer fences runs on the calling thread. Validation/enqueue
     * overhead remains; no real-time latency bound is promised. A fast execution may already
     * be terminal at return.
     *
     * @pre ASYNC_INFERENCE is supported. Session is idle with valid state and a complete
     * validated binding set. External producers/prior consumers are ready.
     * Owners/mappings remain valid and IO is protected until terminal completion.
     *
     * @post On acceptance out_completion may be pending and previous outputs are invalid. Use
     * wait or get_completion_info, then get_outputs on success and release_completion
     * when terminal. Rejection accepts no work and leaves out_completion untouched.
     * Accepted failure requires state reset/recreation.
     *
     * @return OR_OK: work accepted, not necessarily finished or successful. OR_UNSUPPORTED:
     * asynchronous mode unavailable. OR_BUSY: session/resources already in use.
     * OR_STATE_INVALID: state/bindings not ready. Invalid arguments or
     * acceptance/setup failures return their relevant status without accepting work.
     */
    or_status(OR_CALL *infer_async)(or_session, or_completion *out_completion);
    /**
     * @brief Wait for invocation completion and promised output visibility.
     *
     * @details
     * Zero timeout polls; OR_WAIT_FOREVER has no deadline. Timeout does not cancel or
     * release work. Completion covers CPU/device work and required output transfers, not
     * downstream consumer release or DMA-BUF CPU cache-access bracketing. Terminal status,
     * including failure/device loss, guarantees no further runtime access to invocation IO;
     * until safe quiescence it remains pending.
     *
     * @pre A live unreleased completion; no concurrent release. Storage/mappings stay valid
     * while work is pending.
     *
     * @post On terminal success outputs may be obtained/consumed under their memory profile.
     * On terminal failure discard outputs and reset/recreate state. Release terminal
     * completion. After timeout keep resources alive and wait/query again.
     *
     * @return OR_OK: successful terminal execution and promised visibility. OR_TIMEOUT:
     * still pending at deadline. OR_INVALID_ARGUMENT: invalid arguments. A terminal
     * execution status, such as OR_EXECUTION_FAILED, OR_DEVICE_LOST or
     * OR_BUFFER_TOO_SMALL, reports failed execution with no remaining IO access.
     */
    or_status(OR_CALL *wait)(or_completion, uint64_t timeout_ns);
    /**
     * @brief Query whether an invocation has finished and its result.
     *
     * @details
     * Polls without waiting for inference. out.result is meaningful only when out.terminal is
     * one. A successful query is not itself successful execution; inspect both fields.
     *
     * @pre A live completion and writable output record; release does not race this query.
     *
     * @post No object ownership changes. Pending work must retain its resources; terminal
     * work may be handled as after wait, using the applicable memory visibility/access
     * rules, then released.
     *
     * @return OR_OK: completion information returned, including when execution failed or
     * remains pending. OR_INVALID_ARGUMENT: invalid arguments.
     */
    or_status(OR_CALL *get_completion_info)(or_completion, or_completion_info *out);
    /**
     * @brief Borrow actual output metadata from the latest successful invocation.
     *
     * @details
     * Returns actual shape/range/strides without acquiring an output lease. Direct host
     * outputs use host_data rather than a buffer handle. Do not release borrowed output
     * buffer handles. External allocations keep caller ownership; no output truncation is
     * allowed.
     *
     * @pre A live session and its latest successful terminal completion, without a subsequent
     * accepted invocation or binding change. Apply the required native-memory
     * access/synchronization before reading storage.
     *
     * @post On success metadata and runtime storage are borrowed until the next accepted
     * invocation, binding change or session destruction. Consume/copy results before
     * reuse; retain external output slots until downstream completion. Release the
     * terminal completion when no longer queried.
     *
     * @return OR_OK: result views returned. OR_BUSY: invocation pending.
     * OR_INVALID_ARGUMENT: wrong/stale completion or invalid arguments. The
     * completion execution error is returned if that invocation failed.
     */
    or_status(OR_CALL *get_outputs)(or_session, or_completion, or_outputs *out);
    /**
     * @brief Restore the session network state after completion or failure.
     *
     * @details
     * Restores the declared initial/reset state. Stateless models accept reset as a no-op.
     * Failed accepted work requires reset or session recreation; no silent inference retry
     * occurs.
     *
     * @pre The session is idle; no external access conflicts with reset.
     *
     * @post On success network state is initialized and a validated binding set may be used
     * for another invocation. Reset does not replace or revalidate bindings. On failure
     * state must not be assumed usable; recreate the session or handle the reported
     * error.
     *
     * @return OR_OK: initial state restored, or stateless no-op. OR_BUSY: session
     * executing. OR_INVALID_ARGUMENT: invalid arguments. Backend reset failure
     * returns its relevant execution/device status.
     */
    or_status(OR_CALL *reset_state)(or_session);

    /**
     * @brief Release a terminal invocation handle.
     *
     * @details
     * Destruction never waits or cancels pending work. Retaining a completion does not lease
     * output storage against reuse.
     *
     * @pre A live terminal completion with no concurrent waits/queries/other users.
     *
     * @post On success the handle is invalid; it cannot be used for get_outputs or further
     * queries. Destroy an otherwise idle session or submit its next invocation as
     * appropriate. On OR_BUSY the handle remains valid.
     *
     * @return OR_OK: handle released. OR_BUSY: completion pending. OR_INVALID_ARGUMENT:
     * invalid arguments.
     */
    or_status(OR_CALL *release_completion)(or_completion);
    /**
     * @brief Release an owned runtime allocation or external import handle.
     *
     * @details
     * Releases runtime storage or the retained import reference/duplicated FD. It never frees
     * caller-owned imported storage or external native objects. Do not release borrowed
     * handles returned by get_outputs.
     *
     * @pre A live owned buffer handle, unbound, not in flight and without active mappings. No
     * concurrent users.
     *
     * @post On success the handle is invalid; external owners may release their resources
     * once all other imports/consumers finish. On OR_BUSY the object and ownership
     * remain unchanged.
     *
     * @return OR_OK: handle released. OR_BUSY: buffer bound, executing or mapped.
     * OR_INVALID_ARGUMENT: invalid arguments.
     */
    or_status(OR_CALL *release_buffer)(or_buffer);
    /**
     * @brief Destroy idle session state and remove its bindings.
     *
     * @details
     * Destruction never waits implicitly. Runtime scratch and session-owned result metadata
     * are released; caller-owned output storage retains its normal ownership.
     *
     * @pre A live idle session with every completion released and no concurrent users.
     *
     * @post On success session/result views are invalid and bindings are removed. Release
     * owned buffer handles and unload_model when no other sessions remain. On OR_BUSY
     * the session remains valid.
     *
     * @return OR_OK: session destroyed. OR_BUSY: session executing or completions
     * unreleased. OR_INVALID_ARGUMENT: invalid arguments.
     */
    or_status(OR_CALL *destroy_session)(or_session);
    /**
     * @brief Release a loaded model and its immutable metadata.
     *
     * @details
     * Shared immutable weights/compiled state may be reclaimed according to backend
     * ownership. No outstanding session may refer to the model.
     *
     * @pre A live model with all sessions destroyed and no concurrent users.
     *
     * @post On success model and description pointers are invalid. Load another model or
     * destroy the context once its other children are released. On OR_BUSY the model
     * remains valid.
     *
     * @return OR_OK: model unloaded. OR_BUSY: model still has sessions.
     * OR_INVALID_ARGUMENT: invalid arguments.
     */
    or_status(OR_CALL *unload_model)(or_model);
    /**
     * @brief Release an execution environment after its children.
     *
     * @details
     * The context owns its execution resources. No implicit waits or recursive child
     * destruction occur.
     *
     * @pre A live context with models, sessions, buffers, mappings and completions released;
     * no concurrent users.
     *
     * @post On success context-owned pointers and extension tables are invalid. The runtime
     * library may be unloaded after every context/object is gone. On OR_BUSY the
     * context remains valid.
     *
     * @return OR_OK: context destroyed. OR_BUSY: context still has children.
     * OR_INVALID_ARGUMENT: invalid arguments.
     */
    or_status(OR_CALL *destroy_context)(or_context);

    /**
     * @brief Query an optional capability table by extension identifier.
     *
     * @details
     * The exact global version covers the returned table layout. Pass a local const void *
     * variable by address, then cast the result to the known table type. Check its capability
     * bits before optional functions; unsupported entries are NULL.
     *
     * @pre A live context, supported identifier format and writable out_table. Core version
     * verification succeeded.
     *
     * @post On success table storage is borrowed until context destruction. Use only
     * advertised operations; no object ownership changes. Failure leaves out_table
     * untouched.
     *
     * @return OR_OK: extension table returned. OR_UNSUPPORTED: unknown/unavailable
     * extension. OR_INVALID_ARGUMENT: invalid arguments.
     */
    or_status(OR_CALL *get_extension)(or_context,
                                        const char *extension_id,
                                        const void **out_table);
    /**
     * @brief Borrow diagnostic text for the last failed call on this thread.
     *
     * @details
     * Includes bootstrap failures. This is thread-local diagnostic text; status codes, not
     * message text, drive program logic.
     *
     * @pre The runtime library is loaded. Copy the text before another API call on the same
     * thread if it must be retained.
     *
     * @post No execution/object state changes. The borrowed text expires at the next API call
     * on this thread.
     *
     * @return A borrowed diagnostic string describing the last failed call on this thread;
     * not an execution-status code.
     */
    const char *(OR_CALL *last_error_message)(void);
} or_api;

/**
 * @brief Obtain the complete function table for an exact contract version.
 *
 * @details
 * The stable bootstrap checks requested_version against or_runtime_version before
 * interpreting/writing the table. Success populates every core entry; unsupported operations
 * still have nonnull entries. The global version covers all records and extensions, not
 * library release or platform compatibility. Both sides need matching architecture, calling
 * convention and layout. Serialize mutation of shared context/model/session/buffer objects.
 * Concurrent completion waits/queries are permitted, but release/destruction must not race
 * users or descendants. Different sessions may execute independently when supported.
 *
 * @pre Runtime library is loaded, out_api is writable and requested_version equals the
 * consumer OR_RUNTIME_VERSION. No context is needed.
 *
 * @post On success verify api.get_runtime_version() matches before other entries, then
 * create_context. Library remains loaded until every runtime object is released.
 * Failure leaves out_api untouched.
 *
 * @return OR_OK: complete matching table returned. OR_INCOMPATIBLE: requested global
 * version differs. OR_INVALID_ARGUMENT: invalid out_api argument.
 */
OR_EXPORT or_status OR_CALL or_get_api(uint32_t requested_version, or_api *out_api);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OR_RUNTIME_H */

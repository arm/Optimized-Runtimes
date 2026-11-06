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
 * @file loom_runtime_vulkan.h
 * @brief Optional native Vulkan resource profile for loom_runtime.h.
 *
 * @details
 * Resource declarations require support advertised by the concrete runtime.
 * Include this header only in Vulkan-aware code. The core header includes no
 * Vulkan headers and refers to these records through typed pointers only.
 * Every layout below is covered by LRT_RUNTIME_VERSION, not a separate ABI.
 *
 * @section lrt_vk_usage Discovery, import, allocation and native views
 *
 * 1. Query core get_io_requirements. A Vulkan memory option has a nonnull
 *    lrt_memory_option::vulkan identifying its device/queue and requirements.
 *    Copy the caller-selected configuration exactly; ranking is advisory.
 * 2. For external storage, fill lrt_vulkan_resource and pass its address in
 *    lrt_memory_desc::handle.vulkan, with a matching Vulkan reference tag.
 *    Set allocation_byte_size/allocation_offset/view_byte_length to zero: the
 *    typed resource provides native bounds. Import borrows native objects;
 *    it copies descriptors, not VkBuffer/VkImage/VkDeviceMemory ownership.
 * 3. For runtime storage, pass lrt_vulkan_allocation_desc through the core
 *    allocation descriptor. Runtime creates the resource and backing memory.
 *    Both input and output storage are explicitly selected and bound.
 * 4. get_buffer_info returns a borrowed lrt_vulkan_resource pointer for either
 *    source. The application can use its native handle under the handoff rules
 *    below. This is a native view, not external-memory/DMA-BUF export.
 * 5. Use ordinary bind_tensor and validate_bindings, then infer or infer_async/wait and
 *    get_outputs. Both modes return completions for the common result interface.
 *    Runtime owns descriptors, internal transfers and inference execution.
 *
 * Input/output selection is independent. An input may use a Vulkan image while
 * an output uses a host pointer, a Vulkan buffer, or another advertised path.
 * Native-resource options and their constraints may differ for every tensor;
 * there is no implied pairing of input and output resource representations.
 * Validate the complete selected binding set. DIRECT excludes boundary staging;
 * STAGED explicitly selects it. This profile does not add automatic fallback,
 * H2 leases or preprocessing. The caller must not destroy borrowed runtime
 * objects; release the original owned lrt_buffer handle through release_buffer.
 * Imported objects and backing memory must outlive the import. In particular,
 * importing a GStreamer Vulkan handle does not retain its GstBuffer/pool slot.
 *
 * @section lrt_vk_handoff Initial device and queue handoff contract
 *
 * This profile uses the exact same logical device and queue for external
 * use and runtime access. Matching physical GPUs alone is insufficient. It
 * neither wraps an arbitrary application device nor imports foreign-device
 * handles; use a separately supported external-memory route for that case.
 * A context can expose different Vulkan device associations per option, but
 * each imported/bound resource must match the option actually selected.
 *
 * The application may submit to the exposed queue only while the entire LRT
 * context is idle: no pending completions and no concurrent API calls, including
 * allocation/setup calls. The runtime must not use that queue in background
 * work outside API calls or outstanding completions. The application completes
 * its own queue work before calling the runtime. This supplies external queue
 * host synchronization without exposing a queue scheduler or lock protocol.
 *
 * Resources use EXCLUSIVE sharing and belong to queue_family_index at every
 * handoff. Other queue families, concurrent-sharing resources, external queue
 * ownership transfers and native semaphore dependencies are outside this initial
 * profile. The application may not access imported/allocated resources through
 * another queue while this profile is in use. These restrictions are explicit;
 * supporting a VkBuffer tag alone is not permission to skip synchronization.
 *
 * Both core inference modes require completed external producer/prior-consumer work;
 * wait for an application Vulkan fence/queue before handing resources back.
 * Successful core wait covers the full invocation, including CPU work and any
 * staged output copies. Runtime records incoming device memory dependencies
 * for previous writes on the shared queue, outgoing dependencies for subsequent
 * reads/writes on that queue, and any internal barriers/layout transitions.
 * HOST-accessible buffers also receive the device-to-host visibility dependency
 * required before CPU reads; begin_cpu_access handles cache invalidation.
 * Thus a host wait is not being used as a substitute for device visibility.
 * Nonempty external sync-file lists with Vulkan bindings are UNSUPPORTED in
 * this profile; asynchronous Vulkan semaphore handoffs need a future extension.
 *
 * Images are in GENERAL layout outside runtime use. Runtime restores GENERAL
 * before completion and returns newly allocated images already transitioned
 * to GENERAL (contents unspecified). Caller restores GENERAL after application
 * use. No layout-changing or ownership-changing commands may overlap runtime
 * access. Completion with execution failure/device loss gives no valid output;
 * failed resources must not be reused until the backend establishes/recreates
 * safe state. A completed runtime output is still not a released consumer slot.
 *
 * @section lrt_vk_images Bounded image representation
 *
 * Images are non-sparse, unprotected, 2D, single-sample, single-mip, single-layer
 * color images created with flags=0 and EXCLUSIVE sharing. Only the full color
 * subresource is imported/bound; no planes, modifiers, mip chains or crops.
 * Format is an uncompressed numeric format matching tensor dtype exactly,
 * without normalization, sRGB conversion or preprocessing. Supported formats
 * are advertised by the backend, not implied by this header.
 *
 * The logical tensor is [height,width] for a scalar texel, or
 * [height,width,components] for one to four scalar components per texel in
 * format channel order. Each texel component is one tensor element. Byte strides
 * in core requirements describe this logical dense tensor, not an optimal
 * image's physical row pitch. An optimal image has no ordinary tensor host view.
 * Models needing another mapping must not advertise this image profile.
 * This restriction keeps the core's one required logical layout authoritative.
 *
 * Core image binding requires byte_offset=0 and byte_length equal to the full
 * logical tensor byte capacity; its shape/strides match core requirements.
 * Allocation descriptor byte_length likewise denotes logical capacity, not the
 * driver's physical image allocation size. Runtime queries actual Vulkan memory
 * requirements and allocates/binds compatible memory. HOST accessibility is not
 * advertised for images in this profile. Buffer CPU mapping is optional below.
 *
 * @section lrt_vk_sources Primary Vulkan references
 * - [Resource creation](https://docs.vulkan.org/spec/latest/chapters/resources.html)
 * - [Buffer
 * ranges](https://docs.vulkan.org/refpages/latest/refpages/source/VkDescriptorBufferInfo.html)
 * - [Device dependencies](https://docs.vulkan.org/guide/latest/synchronization_examples.html)
 * - [Image
 * transitions](https://docs.vulkan.org/refpages/latest/refpages/source/VkImageMemoryBarrier2.html)
 */
#ifndef LOOM_RUNTIME_VULKAN_H
#define LOOM_RUNTIME_VULKAN_H

#include "loom_runtime.h"
#include <vulkan/vulkan_core.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LRT_EXTENSION_VULKAN "loom.runtime.vulkan/1"

/** @brief Borrowed runtime Vulkan association, immutable for context lifetime.
 *
 * @details
 * Caller must use the same Vulkan implementation/loader and may not destroy
 * these objects. queue belongs to device and queue_family_index. Native API
 * calls/queue submission obey lrt_vk_handoff, including calls from GStreamer.
 */
typedef struct lrt_vulkan_context {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    uint32_t queue_family_index;
} lrt_vulkan_context;

/** @brief Required properties for a buffer option, not an allocation policy. */
typedef struct lrt_vulkan_buffer_requirements {
    VkBufferUsageFlags usage; /* Superset required; e.g. STORAGE_BUFFER/TRANSFER. */
} lrt_vulkan_buffer_requirements;

/** @brief Full 2D image description; image usage may include additional bits.
 *
 * @details
 * Format/extent/tiling must match the advertised option exactly. All other
 * creation parameters and tensor mapping are fixed by lrt_vk_images.
 */
typedef struct lrt_vulkan_image_spec {
    VkFormat format;
    VkExtent2D extent;
    VkImageTiling tiling;
    VkImageUsageFlags usage;
} lrt_vulkan_image_spec;

/** @brief Typed requirements attached to one advertised core memory option.
 *
 * @details
 * The containing configuration.reference selects the union member. POINTER
 * and DMA_BUF options never use this record. Records/context are borrowed for
 * the core's documented metadata lifetimes. Import/allocation must match this
 * context. Imported usage and memory flags must include the required flags.
 */
struct lrt_vulkan_requirements {
    const lrt_vulkan_context *context;
    VkMemoryPropertyFlags memory_properties; /* Required actual memory-type flags. */
    union {
        lrt_vulkan_buffer_requirements buffer;
        lrt_vulkan_image_spec image;
    } resource;
};

/** @brief Backing allocation identity/range, needed to validate aliasing/mapping.
 *
 * @details
 * Caller supplies truthful binding metadata: Vulkan cannot generally retrieve
 * all original creation/binding properties from opaque handles. Sparse,
 * disjoint, protected and dedicated-external-memory special cases are outside
 * this profile. memory_type_index names the actual allocation memory type.
 * binding_offset is relative to VkDeviceMemory, never to VkBuffer. Allocation
 * size must cover the native memory requirements from that offset, and offset
 * must satisfy Vulkan's binding alignment. Imported memory remains caller-owned.
 */
typedef struct lrt_vulkan_backing_memory {
    VkDeviceMemory memory;
    VkDeviceSize allocation_size;
    VkDeviceSize binding_offset;
    uint32_t memory_type_index;
} lrt_vulkan_backing_memory;

typedef struct lrt_vulkan_buffer_resource {
    VkBuffer buffer;
    VkDeviceSize byte_size; /* Actual VkBuffer creation size. */
    VkBufferUsageFlags usage;
    VkDeviceSize view_offset; /* Relative to VkBuffer start, not its memory. */
    VkDeviceSize view_size;   /* Nonzero, bounded by byte_size-view_offset. */
} lrt_vulkan_buffer_resource;

typedef struct lrt_vulkan_image_resource {
    VkImage image;
    lrt_vulkan_image_spec spec;
} lrt_vulkan_image_resource;

/** @brief External import descriptor or borrowed native view of an lrt_buffer.
 *
 * @details
 * The outer core reference selects buffer/image. Every handle is nonnull and
 * belongs to context->device. Resource is already backed by memory. Native
 * descriptors/backing metadata are copied on import; no Vulkan object is copied
 * or refcounted. Application retains originals until release_buffer succeeds.
 * Returned descriptors are immutable buffer-lifetime metadata; native handles
 * may only be used under the ownership and handoff rules above.
 *
 * For buffers, core tensor offsets are relative to view_offset. Physical memory
 * offsets are binding_offset+view_offset+tensor offset, applied exactly once.
 * Views/aliases of overlapping backing ranges cannot be bound as disjoint IO.
 * For images, the view is the full subresource with GENERAL outside runtime use.
 * Runtime may create its own image views/samplers and destroys only those it owns.
 */
struct lrt_vulkan_resource {
    const lrt_vulkan_context *context;
    lrt_vulkan_backing_memory backing;
    union {
        lrt_vulkan_buffer_resource buffer;
        lrt_vulkan_image_resource image;
    } resource;
};

/** @brief Native details of a core runtime allocation request.
 *
 * @details
 * context matches the selected option. Memory flags include all option flags;
 * HOST accessibility also requires HOST_VISIBLE and CPU_MAP support. Runtime
 * chooses an actual compatible memory type satisfying these minimum flags.
 * Vulkan's memory requirements/alignment remain authoritative independently
 * of the core tensor byte_alignment. Runtime returns exact actual metadata
 * through get_buffer_info; no ownership of native handles transfers to caller.
 * BUFFER uses core byte_length as resource/view size and usage as a superset
 * of option requirements. IMAGE matches spec and logical capacity exactly.
 */
struct lrt_vulkan_allocation_desc {
    const lrt_vulkan_context *context;
    VkMemoryPropertyFlags memory_properties;
    union {
        VkBufferUsageFlags buffer_usage;
        lrt_vulkan_image_spec image;
    } resource;
};

typedef uint64_t lrt_vulkan_capability_flags;
#define LRT_VULKAN_CAP_BUFFER_CPU_MAP (UINT64_C(1) << 0)

/** @brief Optional profile table returned by core get_extension.
 *
 * @details
 * Presence permits native options; absence prohibits advertising them. This
 * table does not replace core import/allocate/bind/query/inference functions.
 * CPU-map entries are both nonnull only when BUFFER_CPU_MAP is advertised.
 * Images cannot advertise HOST; buffers may do so only with compatible memory
 * and this capability. Runtime derives actual flags from memory_type_index.
 */
typedef struct lrt_vulkan_api {
    lrt_vulkan_capability_flags capabilities;
    /**
     * @brief Map an idle Vulkan buffer with required cache maintenance.
     *
     * @details
     * READ invalidates non-coherent memory as needed; end_cpu_access flushes WRITE access.
     * Runtime handles non-coherent atom alignment and returns view byte zero. Mapping is not
     * a production fence. Caller must not directly map/unmap/free runtime memory through
     * Vulkan.
     *
     * @pre BUFFER_CPU_MAP is advertised. Producers/consumers have completed; no other mapping
     * of the backing allocation or concurrent external access to overlapping non-
     * coherent atoms exists. Resource supports the requested access.
     *
     * @post On success mapping/info are returned. Binding, inference and release involving
     * this buffer are blocked until end_cpu_access. Failure creates no mapping and
     * leaves outputs untouched.
     *
     * @return LRT_OK: mapped for requested access. LRT_UNSUPPORTED: resource cannot be
     * mapped. LRT_BUSY: conflicting mapping/use. LRT_INVALID_ARGUMENT: invalid
     * access/arguments. LRT_OUT_OF_MEMORY, LRT_DEVICE_LOST or other mapping failure
     * reports the backend error.
     */
    lrt_status(LRT_CALL *begin_cpu_access)(lrt_buffer buffer,
                                           lrt_access access,
                                           lrt_cpu_mapping *out_mapping,
                                           lrt_cpu_mapping_info *out_info);
    /**
     * @brief Flush required Vulkan host writes and release a CPU mapping.
     *
     * @details
     * WRITE access flushes non-coherent ranges as required. Runtime manages mapping lifetime;
     * application must not call Vulkan unmap/free on runtime storage.
     *
     * @pre A live mapping returned by this table; all CPU accesses have ended and no
     * concurrent mapping users remain.
     *
     * @post On success mapping/pointer are invalid. The buffer may be rebound, submitted or
     * released after establishing appropriate consumer dependencies. Failure does not
     * permit assuming successful handoff.
     *
     * @return LRT_OK: cache maintenance and unmapping completed. LRT_INVALID_ARGUMENT:
     * invalid mapping argument. LRT_DEVICE_LOST or another backend mapping/cache
     * error reports failure.
     */
    lrt_status(LRT_CALL *end_cpu_access)(lrt_cpu_mapping mapping);
} lrt_vulkan_api;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LOOM_RUNTIME_VULKAN_H */

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
/* Declaration checks against real Vulkan-Headers; no driver or runtime linked. */
#ifdef LRT_TEST_CORE_FIRST
#include "loom_runtime.h"
#endif
#include "loom_runtime_vulkan.h"
#include <stddef.h>

#ifdef __cplusplus
#define ABI_ASSERT static_assert
#define ABI_ZERO                                                                                   \
    {                                                                                              \
    }
#else
#define ABI_ASSERT _Static_assert
#define ABI_ZERO {0}
#endif

ABI_ASSERT(LRT_RUNTIME_VERSION == 10, "one global contract covers both headers");

int main(void) {
    lrt_vulkan_context context = ABI_ZERO;
    lrt_vulkan_requirements requirements = ABI_ZERO;
    lrt_vulkan_resource resource = ABI_ZERO;
    lrt_vulkan_allocation_desc allocation = ABI_ZERO;
    lrt_memory_option option = ABI_ZERO;
    lrt_memory_desc imported = ABI_ZERO;
    lrt_allocation_desc request = ABI_ZERO;
    lrt_buffer_info info = ABI_ZERO;
    lrt_vulkan_api api = ABI_ZERO;
    lrt_status(LRT_CALL * map)(lrt_buffer, lrt_access, lrt_cpu_mapping *, lrt_cpu_mapping_info *) =
        api.begin_cpu_access;
    lrt_status(LRT_CALL * unmap)(lrt_cpu_mapping) = api.end_cpu_access;

    requirements.context = &context;
    requirements.resource.buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    resource.context = &context;
    resource.resource.buffer.byte_size = 4096;
    resource.resource.buffer.view_offset = 512;
    resource.resource.buffer.view_size = 1024;
    option.configuration.owner = LRT_OWNER_EXTERNAL;
    option.configuration.reference = LRT_REFERENCE_VULKAN_BUFFER;
    option.configuration.accessibility = LRT_ACCESSIBLE_RUNTIME_DEVICE;
    option.configuration.transfer = LRT_TRANSFER_DIRECT;
    option.vulkan = &requirements;
    imported.reference = option.configuration.reference;
    imported.handle.vulkan = &resource;
    request.configuration = option.configuration;
    request.configuration.owner = LRT_OWNER_RUNTIME;
    allocation.context = &context;
    allocation.resource.buffer_usage = requirements.resource.buffer.usage;
    request.vulkan = &allocation;
    info.vulkan = &resource;
    if (option.vulkan->context != imported.handle.vulkan->context ||
        request.vulkan->context != info.vulkan->context)
        return 1;

    option.configuration.reference = LRT_REFERENCE_VULKAN_IMAGE;
    imported.reference = option.configuration.reference;
    request.configuration.reference = option.configuration.reference;
    requirements.resource.image.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    requirements.resource.image.extent.width = 64;
    requirements.resource.image.extent.height = 32;
    requirements.resource.image.tiling = VK_IMAGE_TILING_OPTIMAL;
    requirements.resource.image.usage = VK_IMAGE_USAGE_STORAGE_BIT;
    allocation.resource.image = requirements.resource.image;
    resource.resource.image.spec = requirements.resource.image;
    return map != NULL || unmap != NULL || resource.resource.image.spec.extent.width != 64;
}

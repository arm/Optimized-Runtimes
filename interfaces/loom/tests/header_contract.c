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
/* Compile and run as C11 and C++11: declaration/layout checks, not inference. */
#include "loom_runtime.h"
#include <stddef.h>

#ifdef VK_VERSION_1_0
#error "The core ABI must not include Vulkan headers"
#endif

#ifdef __cplusplus
#define ABI_ASSERT static_assert
#define ABI_ZERO                                                                                   \
    {                                                                                              \
    }
#else
#define ABI_ASSERT _Static_assert
#define ABI_ZERO {0}
#endif

ABI_ASSERT(LRT_RUNTIME_VERSION == 10, "separate inference modes change the ABI");
ABI_ASSERT((LRT_CAP_SYNC_INFERENCE & LRT_CAP_ASYNC_INFERENCE) == 0,
           "execution modes are independent capabilities");
ABI_ASSERT(sizeof(lrt_memory_configuration) == 4 * sizeof(uint32_t),
           "configuration has four fixed-width fields");
ABI_ASSERT(offsetof(lrt_api, get_runtime_version) == 0, "version query is first");

int main(void) {
    lrt_api api = ABI_ZERO;
    lrt_memory_option option = ABI_ZERO;
    lrt_tensor_requirement requirement = ABI_ZERO;
    lrt_host_tensor_binding binding = ABI_ZERO;
    lrt_allocation_desc allocation = ABI_ZERO;
    lrt_buffer_info info = ABI_ZERO;
    lrt_linux_dma_buf_api dma_buf_api = ABI_ZERO;
    const unsigned char model_bytes[] = {1, 2, 3};
    lrt_model_source memory_model = ABI_ZERO;
    lrt_model_source path_model = ABI_ZERO;
    lrt_session_options session_options = ABI_ZERO;
    const uint64_t dimensions[] = {4, 128};
    const lrt_input_shape_override overrides[] = {{0, {2, dimensions}}};
    const uint64_t strides[] = {128 * sizeof(float), sizeof(float)};
    lrt_status(LRT_CALL * query)(lrt_session, lrt_io_requirements *) = api.get_io_requirements;
    lrt_status(LRT_CALL * validate)(lrt_session) = api.validate_bindings;
    lrt_status(LRT_CALL * load)(lrt_context, const lrt_model_source *, lrt_model *) =
        api.load_model;
    lrt_status(LRT_CALL * create_session)(lrt_model, const lrt_session_options *, lrt_session *) =
        api.create_session;
    lrt_status(LRT_CALL * bind)(lrt_session, lrt_io, uint32_t, const lrt_tensor_binding *) =
        api.bind_tensor;
    lrt_status(LRT_CALL * bind_host)(
        lrt_session, lrt_io, uint32_t, const lrt_host_tensor_binding *) = api.bind_host_tensor;
    lrt_status(LRT_CALL * infer)(lrt_session, lrt_completion *) = api.infer;
    lrt_status(LRT_CALL * infer_async)(lrt_session, lrt_completion *) = api.infer_async;
    lrt_status(LRT_CALL * infer_fenced)(lrt_session, uint32_t, const int32_t *, lrt_completion *) =
        dma_buf_api.infer_async_with_fences;

    session_options.shape_override_count = 1;
    session_options.shape_overrides = overrides;

    memory_model.kind = LRT_MODEL_SOURCE_MEMORY;
    memory_model.source.memory.data = model_bytes;
    memory_model.source.memory.byte_length = sizeof(model_bytes);
    path_model.kind = LRT_MODEL_SOURCE_PATH;
    path_model.source.path = "model.bin";

    option.configuration.owner = LRT_OWNER_EXTERNAL;
    option.configuration.accessibility = LRT_ACCESSIBLE_HOST;
    option.configuration.reference = LRT_REFERENCE_POINTER;
    option.configuration.transfer = LRT_TRANSFER_DIRECT;
    option.output_handoff = LRT_HANDOFF_NONE;
    option.pointer_binding_flags = LRT_POINTER_BORROWED_BIND;
    requirement.shape = overrides[0].shape;
    requirement.byte_strides = strides;
    requirement.memory_option_count = 1;
    requirement.memory_options = &option;
    binding.configuration = requirement.memory_options[0].configuration;
    binding.byte_strides = requirement.byte_strides;
    allocation.configuration = binding.configuration;
    allocation.configuration.owner = LRT_OWNER_RUNTIME;
    allocation.configuration.accessibility = LRT_ACCESSIBLE_HOST | LRT_ACCESSIBLE_RUNTIME_DEVICE;
    info.owner = allocation.configuration.owner;
    info.accessibility = allocation.configuration.accessibility;
    info.reference = allocation.configuration.reference;
    return query != NULL || validate != NULL || load != NULL || create_session != NULL ||
           bind != NULL || bind_host != NULL || infer != NULL || infer_async != NULL ||
           infer_fenced != NULL || requirement.memory_option_count != 1 ||
           binding.byte_strides != requirement.byte_strides ||
           session_options.compile_policy != LRT_ALLOW_COMPILE ||
           session_options.shape_overrides[0].shape.dimensions[0] != 4 ||
           memory_model.source.memory.data != model_bytes ||
           memory_model.source.memory.byte_length != sizeof(model_bytes) ||
           path_model.source.path == NULL || info.owner != LRT_OWNER_RUNTIME ||
           binding.configuration.transfer != LRT_TRANSFER_DIRECT ||
           binding.configuration.owner != LRT_OWNER_EXTERNAL;
}

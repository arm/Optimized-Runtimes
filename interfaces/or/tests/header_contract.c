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
#include "or_runtime.h"
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

ABI_ASSERT(OR_RUNTIME_VERSION == 10, "separate inference modes change the ABI");
ABI_ASSERT((OR_CAP_SYNC_INFERENCE & OR_CAP_ASYNC_INFERENCE) == 0,
           "execution modes are independent capabilities");
ABI_ASSERT(sizeof(or_memory_configuration) == 4 * sizeof(uint32_t),
           "configuration has four fixed-width fields");
ABI_ASSERT(offsetof(or_api, get_runtime_version) == 0, "version query is first");

int main(void) {
    or_api api = ABI_ZERO;
    or_memory_option option = ABI_ZERO;
    or_tensor_requirement requirement = ABI_ZERO;
    or_host_tensor_binding binding = ABI_ZERO;
    or_allocation_desc allocation = ABI_ZERO;
    or_buffer_info info = ABI_ZERO;
    or_linux_dma_buf_api dma_buf_api = ABI_ZERO;
    const unsigned char model_bytes[] = {1, 2, 3};
    or_model_source memory_model = ABI_ZERO;
    or_model_source path_model = ABI_ZERO;
    or_session_options session_options = ABI_ZERO;
    const uint64_t dimensions[] = {4, 128};
    const or_input_shape_override overrides[] = {{0, {2, dimensions}}};
    const uint64_t strides[] = {128 * sizeof(float), sizeof(float)};
    or_status(OR_CALL * query)(or_session, or_io_requirements *) = api.get_io_requirements;
    or_status(OR_CALL * validate)(or_session) = api.validate_bindings;
    or_status(OR_CALL * load)(or_context, const or_model_source *, or_model *) =
        api.load_model;
    or_status(OR_CALL * create_session)(or_model, const or_session_options *, or_session *) =
        api.create_session;
    or_status(OR_CALL * bind)(or_session, or_io, uint32_t, const or_tensor_binding *) =
        api.bind_tensor;
    or_status(OR_CALL * bind_host)(
        or_session, or_io, uint32_t, const or_host_tensor_binding *) = api.bind_host_tensor;
    or_status(OR_CALL * infer)(or_session, or_completion *) = api.infer;
    or_status(OR_CALL * infer_async)(or_session, or_completion *) = api.infer_async;
    or_status(OR_CALL * infer_fenced)(or_session, uint32_t, const int32_t *, or_completion *) =
        dma_buf_api.infer_async_with_fences;

    session_options.shape_override_count = 1;
    session_options.shape_overrides = overrides;

    memory_model.kind = OR_MODEL_SOURCE_MEMORY;
    memory_model.source.memory.data = model_bytes;
    memory_model.source.memory.byte_length = sizeof(model_bytes);
    path_model.kind = OR_MODEL_SOURCE_PATH;
    path_model.source.path = "model.bin";

    option.configuration.owner = OR_OWNER_EXTERNAL;
    option.configuration.accessibility = OR_ACCESSIBLE_HOST;
    option.configuration.reference = OR_REFERENCE_POINTER;
    option.configuration.transfer = OR_TRANSFER_DIRECT;
    option.output_handoff = OR_HANDOFF_NONE;
    option.pointer_binding_flags = OR_POINTER_BORROWED_BIND;
    requirement.shape = overrides[0].shape;
    requirement.byte_strides = strides;
    requirement.memory_option_count = 1;
    requirement.memory_options = &option;
    binding.configuration = requirement.memory_options[0].configuration;
    binding.byte_strides = requirement.byte_strides;
    allocation.configuration = binding.configuration;
    allocation.configuration.owner = OR_OWNER_RUNTIME;
    allocation.configuration.accessibility = OR_ACCESSIBLE_HOST | OR_ACCESSIBLE_RUNTIME_DEVICE;
    info.owner = allocation.configuration.owner;
    info.accessibility = allocation.configuration.accessibility;
    info.reference = allocation.configuration.reference;
    return query != NULL || validate != NULL || load != NULL || create_session != NULL ||
           bind != NULL || bind_host != NULL || infer != NULL || infer_async != NULL ||
           infer_fenced != NULL || requirement.memory_option_count != 1 ||
           binding.byte_strides != requirement.byte_strides ||
           session_options.compile_policy != OR_ALLOW_COMPILE ||
           session_options.shape_overrides[0].shape.dimensions[0] != 4 ||
           memory_model.source.memory.data != model_bytes ||
           memory_model.source.memory.byte_length != sizeof(model_bytes) ||
           path_model.source.path == NULL || info.owner != OR_OWNER_RUNTIME ||
           binding.configuration.transfer != OR_TRANSFER_DIRECT ||
           binding.configuration.owner != OR_OWNER_EXTERNAL;
}

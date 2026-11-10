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
#include "provider.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <vector>

struct or_context_impl {
    unsigned models = 0, buffers = 0;
};
struct or_model_impl {
    or_context context;
    unsigned sessions = 0;
};
struct or_buffer_impl {
    or_context context;
    void *data;
    uint64_t bytes;
    or_access access;
    bool owned;
    unsigned bindings = 0;
};
struct Binding {
    or_buffer buffer = nullptr;
    void *data = nullptr;
    uint64_t bytes = 0;
    or_memory_configuration config{};
};
struct or_session_impl {
    or_model model;
    Binding bindings[2];
    bool validated = false, invalid_state = false;
    unsigned completions = 0;
    uint64_t generation = 0;
    or_completion pending = nullptr;
    or_memory_option options[2][3]{};
    or_tensor_requirement requirements[2]{};
    const or_tensor_requirement *input_requirements[1], *output_requirements[1];
    or_output output{};
    const or_output *output_ptr[1];
};
struct or_completion_impl {
    or_session session;
    uint64_t generation;
    bool terminal;
    or_status result;
};

namespace {
thread_local std::string error;
std::atomic<bool> hold_next{false}, fail_next{false};
bool single_session_profile() {
    const char *profile = std::getenv("OR_MOCK_SESSION_PROFILE");
    return profile && std::strcmp(profile, "single") == 0;
}
bool import_only_profile() {
    const char *profile = std::getenv("OR_MOCK_HOST_METHOD");
    return profile && std::strcmp(profile, "import") == 0;
}
constexpr uint64_t count = 4, bytes = count * sizeof(float), alignment = 16;
const uint64_t dims[] = {count}, strides[] = {sizeof(float)};
const int64_t model_dims[] = {count};
const or_tensor_info tensor = {
    "values", OR_FLOAT32, 1, model_dims, dims, dims, "tensor/dense/v1", {0, -1, nullptr, nullptr}};
const or_tensor_info *tensors[] = {&tensor};
const uint8_t package[] = {'L', 'R', 'T', 'C', 1};

or_status status(or_status code) {
    if (code != OR_OK)
        error = "mock status " + std::to_string(code);
    return code;
}
bool same(const or_memory_configuration &a, const or_memory_configuration &b) {
    return a.owner == b.owner && a.accessibility == b.accessibility && a.reference == b.reference &&
           a.transfer == b.transfer;
}
bool power_two(uint64_t n) {
    return n && !(n & (n - 1));
}
void complete(or_completion c) {
    if (c->terminal)
        return;
    auto s = c->session;
    if (c->result == OR_OK) {
        float staged_input[4], staged_output[4];
        const uint8_t *input = static_cast<const uint8_t *>(s->bindings[0].data);
        uint8_t *output = static_cast<uint8_t *>(s->bindings[1].data);
        if (s->bindings[0].config.transfer == OR_TRANSFER_STAGED) {
            std::memcpy(staged_input, input, bytes);
            input = reinterpret_cast<const uint8_t *>(staged_input);
        }
        auto destination = output;
        if (s->bindings[1].config.transfer == OR_TRANSFER_STAGED)
            destination = reinterpret_cast<uint8_t *>(staged_output);
        const char *fault = std::getenv("OR_MOCK_FAULT");
        for (unsigned i = 0; i < 4; ++i) {
            float value;
            std::memcpy(&value, input + i * sizeof(float), sizeof(value));
            value *= 2;
            if (i == 0 && fault && std::string(fault) == "wrong-output")
                value += 1;
            std::memcpy(destination + i * sizeof(float), &value, sizeof(value));
        }
        if (destination != output)
            std::memcpy(output, destination, bytes);
    } else
        s->invalid_state = true;
    c->terminal = true;
    s->pending = nullptr;
}
uint32_t OR_CALL version() {
    return OR_RUNTIME_VERSION;
}
or_status OR_CALL create_context(or_context *out) {
    if (!out)
        return status(OR_INVALID_ARGUMENT);
    auto c = new (std::nothrow) or_context_impl{};
    if (!c)
        return status(OR_OUT_OF_MEMORY);
    *out = c;
    return OR_OK;
}
or_status OR_CALL capabilities(or_context c, or_capabilities *out) {
    if (!c || !out)
        return status(OR_INVALID_ARGUMENT);
    *out = {OR_CAP_HOST_BIND | OR_CAP_HOST_IMPORT | OR_CAP_SYNC_INFERENCE |
                OR_CAP_ASYNC_INFERENCE | OR_CAP_PARALLEL_SESSIONS,
            1,
            "host-conformance-mock",
            "1"};
    if (single_session_profile())
        out->flags &= ~OR_CAP_PARALLEL_SESSIONS;
    if (import_only_profile())
        out->flags &= ~OR_CAP_HOST_BIND;
    return OR_OK;
}
or_status OR_CALL load(or_context c, const or_model_source *source, or_model *out) {
    if (!c || !source || !out)
        return status(OR_INVALID_ARGUMENT);
    if (source->kind != OR_MODEL_SOURCE_MEMORY)
        return status(source->kind == OR_MODEL_SOURCE_PATH ? OR_UNSUPPORTED
                                                            : OR_INVALID_ARGUMENT);
    if (!source->source.memory.data || !source->source.memory.byte_length)
        return status(OR_INVALID_ARGUMENT);
    if (source->source.memory.byte_length != sizeof(package) ||
        std::memcmp(source->source.memory.data, package, sizeof(package)))
        return status(OR_INCOMPATIBLE);
    auto m = new (std::nothrow) or_model_impl{c};
    if (!m)
        return status(OR_OUT_OF_MEMORY);
    ++c->models;
    *out = m;
    return OR_OK;
}
or_status OR_CALL describe(or_model m, or_model_info *out) {
    if (!m || !out)
        return status(OR_INVALID_ARGUMENT);
    *out = {"double-four", 1, 1, tensors, tensors, 0, nullptr, nullptr};
    return OR_OK;
}
or_status OR_CALL create_session(or_model m,
                                   const or_session_options *options,
                                   or_session *out) {
    if (!m || !out)
        return status(OR_INVALID_ARGUMENT);
    if (options) {
        if (options->compile_policy != OR_ALLOW_COMPILE &&
            options->compile_policy != OR_PRECOMPILED_ONLY)
            return status(OR_INVALID_ARGUMENT);
        if (options->shape_override_count > 1 ||
            (options->shape_override_count && !options->shape_overrides))
            return status(OR_INVALID_ARGUMENT);
        if (options->shape_override_count) {
            auto &o = options->shape_overrides[0];
            if (o.input_index || o.shape.rank != 1 || !o.shape.dimensions ||
                o.shape.dimensions[0] != count)
                return status(OR_INVALID_ARGUMENT);
        }
    }
    if (single_session_profile() && m->sessions)
        return status(OR_UNSUPPORTED);
    auto s = new (std::nothrow) or_session_impl{};
    if (!s)
        return status(OR_OUT_OF_MEMORY);
    s->model = m;
    for (unsigned io = 0; io < 2; ++io) {
        for (unsigned method = 0; method < 3; ++method) {
            auto &o = s->options[io][method];
            o.configuration = {method ? OR_OWNER_EXTERNAL : OR_OWNER_RUNTIME,
                               OR_ACCESSIBLE_HOST,
                               OR_REFERENCE_POINTER,
                               OR_TRANSFER_DIRECT};
            o.output_handoff =
                io ? (method ? OR_HANDOFF_EXTERNAL : OR_HANDOFF_BORROWED) : OR_HANDOFF_NONE;
            o.minimum_byte_capacity = bytes;
            o.byte_alignment = alignment;
            o.pointer_binding_flags = method
                                          ? OR_POINTER_BORROWED_BIND | OR_POINTER_REUSABLE_IMPORT
                                          : OR_POINTER_RUNTIME_ALLOCATION;
        }
        // External DIRECT and STAGED choices both support import and borrowed binding.
        s->options[io][2].configuration.transfer = OR_TRANSFER_STAGED;
        if (import_only_profile()) {
            s->options[io][1].pointer_binding_flags = OR_POINTER_REUSABLE_IMPORT;
            s->options[io][2].pointer_binding_flags = OR_POINTER_REUSABLE_IMPORT;
        }
        if (std::getenv("OR_MOCK_LARGER_STAGED")) {
            s->options[io][2].minimum_byte_capacity = bytes * 2;
            s->options[io][2].byte_alignment = alignment * 4;
        }
        s->requirements[io] = {{1, dims}, 0, bytes, alignment, strides, 3, s->options[io]};
    }
    // Deliberately asymmetric option sets exercise independent tensor selection.
    // These are test profiles, not capabilities required of concrete runtimes.
    const char *profile = std::getenv("OR_MOCK_MEMORY_PROFILE");
    if (profile) {
        const bool external_input = std::strcmp(profile, "external-input") == 0;
        const bool external_output = std::strcmp(profile, "external-output") == 0;
        const bool staged_input = std::strcmp(profile, "staged-input") == 0;
        const bool staged_output = std::strcmp(profile, "staged-output") == 0;
        if (external_input || external_output || staged_input || staged_output) {
            const unsigned external_side = external_input || staged_input ? 0 : 1;
            const unsigned runtime_side = 1 - external_side;
            const bool staged_only = staged_input || staged_output;
            s->requirements[external_side].memory_options =
                &s->options[external_side][staged_only ? 2 : 1];
            s->requirements[external_side].memory_option_count = staged_only ? 1 : 2;
            s->requirements[runtime_side].memory_option_count = 1;
        }
    }
    s->input_requirements[0] = &s->requirements[0];
    s->output_requirements[0] = &s->requirements[1];
    s->output_ptr[0] = &s->output;
    ++m->sessions;
    *out = s;
    return OR_OK;
}
or_status OR_CALL requirements(or_session s, or_io_requirements *out) {
    if (!s || !out)
        return status(OR_INVALID_ARGUMENT);
    *out = {1, 1, s->input_requirements, s->output_requirements};
    return OR_OK;
}
or_status OR_CALL import_buffer(or_context c, const or_memory_desc *d, or_buffer *out) {
    if (!c || !d || !out)
        return status(OR_INVALID_ARGUMENT);
    if (d->reference != OR_REFERENCE_POINTER)
        return status(OR_UNSUPPORTED);
    if (!d->handle.host_base || !d->view_byte_length || d->access < 1 || d->access > 3 ||
        d->allocation_offset > d->allocation_byte_size ||
        d->view_byte_length > d->allocation_byte_size - d->allocation_offset)
        return status(OR_INVALID_ARGUMENT);
    auto b = new (std::nothrow)
        or_buffer_impl{c,
                        static_cast<uint8_t *>(d->handle.host_base) + d->allocation_offset,
                        d->view_byte_length,
                        d->access,
                        false};
    if (!b)
        return status(OR_OUT_OF_MEMORY);
    ++c->buffers;
    *out = b;
    return OR_OK;
}
or_status OR_CALL allocate(or_context c, const or_allocation_desc *d, or_buffer *out) {
    if (!c || !d || !out || !d->byte_length || !power_two(d->byte_alignment) || d->access < 1 ||
        d->access > 3)
        return status(OR_INVALID_ARGUMENT);
    const or_memory_configuration supported = {
        OR_OWNER_RUNTIME, OR_ACCESSIBLE_HOST, OR_REFERENCE_POINTER, OR_TRANSFER_DIRECT};
    if (!same(d->configuration, supported))
        return status(OR_UNSUPPORTED);
    if (d->byte_length > std::numeric_limits<size_t>::max())
        return status(OR_OUT_OF_MEMORY);
    void *data = nullptr;
    if (posix_memalign(
            &data, std::max<uint64_t>(sizeof(void *), d->byte_alignment), d->byte_length))
        return status(OR_OUT_OF_MEMORY);
    auto b = new (std::nothrow) or_buffer_impl{c, data, d->byte_length, d->access, true};
    if (!b) {
        std::free(data);
        return status(OR_OUT_OF_MEMORY);
    }
    ++c->buffers;
    *out = b;
    return OR_OK;
}
or_status OR_CALL buffer_info(or_buffer b, or_buffer_info *out) {
    if (!b || !out)
        return status(OR_INVALID_ARGUMENT);
    *out = {b->owned ? OR_OWNER_RUNTIME : OR_OWNER_EXTERNAL,
            OR_ACCESSIBLE_HOST,
            OR_REFERENCE_POINTER,
            b->access,
            b->bytes,
            b->data,
            nullptr};
    return OR_OK;
}
or_status bind(or_session s,
                or_io io,
                uint32_t index,
                const or_memory_configuration *config,
                void *data,
                uint64_t capacity,
                or_shape shape,
                const uint64_t *stride,
                or_buffer buffer) {
    if (!s || index || (io != OR_INPUT && io != OR_OUTPUT))
        return status(OR_INVALID_ARGUMENT);
    if (s->pending)
        return status(OR_BUSY);
    auto slot = io == OR_INPUT ? 0 : 1;
    if (data) {
        const or_memory_option *advertised = nullptr;
        const auto &requirement = s->requirements[slot];
        for (uint32_t i = 0; i < requirement.memory_option_count; ++i)
            if (same(*config, requirement.memory_options[i].configuration))
                advertised = &requirement.memory_options[i];
        if (!advertised)
            return status(OR_UNSUPPORTED);
        if (config->owner != (buffer && buffer->owned ? OR_OWNER_RUNTIME : OR_OWNER_EXTERNAL))
            return status(OR_INCOMPATIBLE);
        if (shape.rank != 1 || !shape.dimensions || shape.dimensions[0] != count ||
            (stride && stride[0] != sizeof(float)) ||
            reinterpret_cast<uintptr_t>(data) % std::max(alignment, advertised->byte_alignment))
            return status(OR_INVALID_ARGUMENT);
        if (capacity < std::max(bytes, advertised->minimum_byte_capacity))
            return status(OR_BUFFER_TOO_SMALL);
        auto &other = s->bindings[1 - slot];
        if (other.data) {
            const char *profile = std::getenv("OR_MOCK_MEMORY_PROFILE");
            // These profiles share one external or staged resource between input and output.
            if (profile &&
                ((std::strcmp(profile, "single-external") == 0 &&
                  config->owner == OR_OWNER_EXTERNAL &&
                  other.config.owner == OR_OWNER_EXTERNAL) ||
                 (std::strcmp(profile, "single-staged") == 0 &&
                  config->transfer == OR_TRANSFER_STAGED &&
                  other.config.transfer == OR_TRANSFER_STAGED)))
                return status(OR_INCOMPATIBLE);
            auto a = reinterpret_cast<uintptr_t>(data), b = reinterpret_cast<uintptr_t>(other.data);
            if (a < b + bytes && b < a + bytes)
                return status(OR_INCOMPATIBLE);
        }
    }
    if (s->bindings[slot].buffer)
        --s->bindings[slot].buffer->bindings;
    s->bindings[slot] = {buffer, data, capacity, config ? *config : or_memory_configuration{}};
    if (buffer)
        ++buffer->bindings;
    s->validated = false;
    ++s->generation;
    return OR_OK;
}
or_status OR_CALL bind_tensor(or_session s,
                                or_io io,
                                uint32_t index,
                                const or_tensor_binding *b) {
    if (!b)
        return bind(s, io, index, nullptr, nullptr, 0, {}, nullptr, nullptr);
    if (!s || !b->buffer)
        return status(OR_INVALID_ARGUMENT);
    if (b->buffer->context != s->model->context)
        return status(OR_INCOMPATIBLE);
    if (b->byte_offset > b->buffer->bytes || b->byte_length > b->buffer->bytes - b->byte_offset)
        return status(OR_BUFFER_TOO_SMALL);
    if (!(b->buffer->access & (io == OR_INPUT ? OR_READ : OR_WRITE)))
        return status(OR_INCOMPATIBLE);
    return bind(s,
                io,
                index,
                &b->configuration,
                static_cast<uint8_t *>(b->buffer->data) + b->byte_offset,
                b->byte_length,
                b->shape,
                b->byte_strides,
                b->buffer);
}
or_status OR_CALL bind_host(or_session s,
                              or_io io,
                              uint32_t index,
                              const or_host_tensor_binding *b) {
    if (!b)
        return bind(s, io, index, nullptr, nullptr, 0, {}, nullptr, nullptr);
    if (!b->data)
        return status(OR_INVALID_ARGUMENT);
    return bind(s,
                io,
                index,
                &b->configuration,
                b->data,
                b->byte_length,
                b->shape,
                b->byte_strides,
                nullptr);
}
or_status OR_CALL validate(or_session s) {
    if (!s)
        return status(OR_INVALID_ARGUMENT);
    if (s->pending)
        return status(OR_BUSY);
    if (!s->bindings[0].data || !s->bindings[1].data)
        return status(OR_STATE_INVALID);
    const char *fault = std::getenv("OR_MOCK_FAULT");
    if (fault && std::strcmp(fault, "paired-memory") == 0 &&
        (s->bindings[0].config.owner != s->bindings[1].config.owner ||
         s->bindings[0].config.transfer != s->bindings[1].config.transfer))
        return status(OR_INCOMPATIBLE);
    s->validated = true;
    return OR_OK;
}
or_status submit(or_session s, or_completion *out, bool async) {
    if (!s || !out)
        return status(OR_INVALID_ARGUMENT);
    if (s->pending)
        return status(OR_BUSY);
    if (!s->validated || s->invalid_state)
        return status(OR_STATE_INVALID);
    auto c = new (std::nothrow)
        or_completion_impl{s, s->generation + 1, false, OR_OK};
    if (!c)
        return status(OR_OUT_OF_MEMORY);
    if (fail_next.exchange(false, std::memory_order_relaxed))
        c->result = OR_EXECUTION_FAILED;
    const bool hold = hold_next.exchange(false, std::memory_order_relaxed);
    ++s->generation;
    ++s->completions;
    s->pending = c;
    *out = c;
    if (!async || !hold)
        complete(c);
    return OR_OK;
}
or_status OR_CALL infer(or_session s, or_completion *out) {
    return submit(s, out, false);
}
or_status OR_CALL infer_async(or_session s, or_completion *out) {
    return submit(s, out, true);
}
or_status OR_CALL wait(or_completion c, uint64_t timeout) {
    if (!c)
        return status(OR_INVALID_ARGUMENT);
    if (!c->terminal && timeout != 0)
        complete(c);
    return c->terminal ? status(c->result) : status(OR_TIMEOUT);
}
or_status OR_CALL completion_info(or_completion c, or_completion_info *out) {
    if (!c || !out)
        return status(OR_INVALID_ARGUMENT);
    *out = {c->terminal ? 1u : 0u, c->result};
    return OR_OK;
}
or_status OR_CALL outputs(or_session s, or_completion c, or_outputs *out) {
    if (!s || !c || !out || c->session != s || c->generation != s->generation)
        return status(OR_INVALID_ARGUMENT);
    if (!c->terminal)
        return status(OR_BUSY);
    if (c->result != OR_OK)
        return status(c->result);
    auto &b = s->bindings[1];
    s->output = {
        0, {b.buffer, 0, b.bytes, {1, dims}, strides, b.config}, b.buffer ? nullptr : b.data};
    *out = {1, s->output_ptr};
    return OR_OK;
}
or_status OR_CALL reset(or_session s) {
    if (!s)
        return status(OR_INVALID_ARGUMENT);
    if (s->pending)
        return status(OR_BUSY);
    s->invalid_state = false;
    return OR_OK;
}
or_status OR_CALL release_completion(or_completion c) {
    if (!c)
        return status(OR_INVALID_ARGUMENT);
    if (!c->terminal)
        return status(OR_BUSY);
    --c->session->completions;
    delete c;
    return OR_OK;
}
or_status OR_CALL release_buffer(or_buffer b) {
    if (!b)
        return status(OR_INVALID_ARGUMENT);
    if (b->bindings)
        return status(OR_BUSY);
    --b->context->buffers;
    if (b->owned)
        std::free(b->data);
    delete b;
    return OR_OK;
}
or_status OR_CALL destroy_session(or_session s) {
    if (!s)
        return status(OR_INVALID_ARGUMENT);
    if (s->pending || s->completions)
        return status(OR_BUSY);
    for (auto &b : s->bindings)
        if (b.buffer)
            --b.buffer->bindings;
    --s->model->sessions;
    delete s;
    return OR_OK;
}
or_status OR_CALL unload(or_model m) {
    if (!m)
        return status(OR_INVALID_ARGUMENT);
    if (m->sessions)
        return status(OR_BUSY);
    --m->context->models;
    delete m;
    return OR_OK;
}
or_status OR_CALL destroy_context(or_context c) {
    if (!c)
        return status(OR_INVALID_ARGUMENT);
    if (c->models || c->buffers)
        return status(OR_BUSY);
    delete c;
    return OR_OK;
}
or_status OR_CALL extension(or_context c, const char *name, const void **out) {
    return status(c && name && out ? OR_UNSUPPORTED : OR_INVALID_ARGUMENT);
}
const char *OR_CALL last_error() {
    return error.c_str();
}
or_status OR_CALL get_api(uint32_t requested, or_api *out) {
    if (requested != OR_RUNTIME_VERSION) {
        if (const char *fault = std::getenv("OR_MOCK_FAULT");
            fault && std::string(fault) == "clobber-bootstrap" && out)
            std::memset(out, 0, sizeof(*out));
        return status(OR_INCOMPATIBLE);
    }
    if (!out)
        return status(OR_INVALID_ARGUMENT);
    *out = {version,         create_context,  capabilities, load,
            describe,        create_session,  requirements, import_buffer,
            allocate,        buffer_info,     bind_tensor,  bind_host,
            validate,        infer,           infer_async,  wait,
            completion_info, outputs,         reset,        release_completion,
            release_buffer,  destroy_session, unload,       destroy_context,
            extension,       last_error};
    return OR_OK;
}
} // namespace

namespace or_conformance {
Provider make_provider() {
    hold_next = false;
    fail_next = false;
    Provider p{};
    p.get_api = get_api;
    p.exported_version = OR_RUNTIME_VERSION;
    p.model_bytes.assign(std::begin(package), std::end(package));
    p.input_option_indices = {0};
    p.output_option_indices = {0};
    const float input[] = {1, -2, 3, 4};
    p.inputs.emplace_back(reinterpret_cast<const uint8_t *>(input),
                          reinterpret_cast<const uint8_t *>(input) + sizeof(input));
    p.verify_output = [](uint32_t index, const void *data, size_t length) {
        const float expected[] = {2, -4, 6, 8};
        return index == 0 && length >= sizeof(expected) &&
               !std::memcmp(data, expected, sizeof(expected));
    };
    p.arm_pending = [] { hold_next = true; };
    p.finish_pending = [] {}; // wait(nonzero) releases the deterministic mock gate.
    p.fail_next_execution = [] { fail_next = true; };
    return p;
}
} // namespace or_conformance

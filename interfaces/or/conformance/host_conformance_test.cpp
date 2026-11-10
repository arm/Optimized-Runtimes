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
#include <future>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace or_conformance {
namespace {
constexpr uint64_t wait_ns = UINT64_C(2000000000);
[[noreturn]] void unsafe_runtime(const char *message) {
    std::fprintf(stderr, "Conformance stopped to avoid unsafe cleanup: %s\n", message);
    std::fflush(stderr);
    std::_Exit(2);
}
void cleanup_ok(or_status result) {
    if (result != OR_OK)
        unsafe_runtime("runtime refused safe teardown");
}
void expect_busy(or_status result) {
    EXPECT_EQ(result, OR_BUSY);
    if (result == OR_OK)
        unsafe_runtime("destructive call unexpectedly succeeded");
}
enum class Method { Any, Runtime, Import, Borrow };

class HostConformance : public testing::Test {
  protected:
    Provider provider;
    or_api api{};
    or_context context = nullptr;
    or_model model = nullptr;
    or_session session = nullptr;
    or_capabilities caps{};
    or_model_info metadata{};
    or_io_requirements requirements{};
    std::vector<or_session> extra_sessions;
    std::vector<or_buffer> buffers;
    std::vector<or_completion> completions;
    std::vector<std::vector<uint8_t>> storage;
    struct Tensor {
        or_io io;
        uint32_t index;
        const or_tensor_requirement *requirement;
        const or_memory_option *option;
        or_buffer buffer;
        void *data;
        uint64_t capacity;
    };
    std::vector<Tensor> bound;
    bool prepared = false;

    void SetUp() override {
        provider = make_provider();
        ASSERT_NE(provider.get_api, nullptr);
        ASSERT_EQ(provider.exported_version, OR_RUNTIME_VERSION);
        ASSERT_EQ(provider.get_api(OR_RUNTIME_VERSION, &api), OR_OK);
#define REQUIRE_ENTRY(name) ASSERT_NE(api.name, nullptr) << #name
        REQUIRE_ENTRY(get_runtime_version);
        REQUIRE_ENTRY(create_context);
        REQUIRE_ENTRY(get_capabilities);
        REQUIRE_ENTRY(load_model);
        REQUIRE_ENTRY(describe_model);
        REQUIRE_ENTRY(create_session);
        REQUIRE_ENTRY(get_io_requirements);
        REQUIRE_ENTRY(import_buffer);
        REQUIRE_ENTRY(allocate_buffer);
        REQUIRE_ENTRY(get_buffer_info);
        REQUIRE_ENTRY(bind_tensor);
        REQUIRE_ENTRY(bind_host_tensor);
        REQUIRE_ENTRY(validate_bindings);
        REQUIRE_ENTRY(infer);
        REQUIRE_ENTRY(infer_async);
        REQUIRE_ENTRY(wait);
        REQUIRE_ENTRY(get_completion_info);
        REQUIRE_ENTRY(get_outputs);
        REQUIRE_ENTRY(reset_state);
        REQUIRE_ENTRY(release_completion);
        REQUIRE_ENTRY(release_buffer);
        REQUIRE_ENTRY(destroy_session);
        REQUIRE_ENTRY(unload_model);
        REQUIRE_ENTRY(destroy_context);
        REQUIRE_ENTRY(get_extension);
        REQUIRE_ENTRY(last_error_message);
#undef REQUIRE_ENTRY
        ASSERT_EQ(api.get_runtime_version(), OR_RUNTIME_VERSION);
        ASSERT_EQ(api.create_context(&context), OR_OK);
        ASSERT_EQ(api.get_capabilities(context, &caps), OR_OK);
        ASSERT_TRUE(caps.flags & (OR_CAP_SYNC_INFERENCE | OR_CAP_ASYNC_INFERENCE));
        ASSERT_FALSE(provider.model_bytes.empty());
        ASSERT_TRUE(provider.verify_output);
        auto source = memory_source(provider.model_bytes);
        ASSERT_EQ(api.load_model(context, &source, &model), OR_OK);
        ASSERT_EQ(api.describe_model(model, &metadata), OR_OK);
        ASSERT_GT(metadata.input_count, 0u);
        ASSERT_GT(metadata.output_count, 0u);
        ASSERT_NE(metadata.inputs, nullptr);
        ASSERT_NE(metadata.outputs, nullptr);
        ASSERT_EQ(provider.inputs.size(), metadata.input_count);
        ASSERT_EQ(provider.input_option_indices.size(), metadata.input_count);
        ASSERT_EQ(provider.output_option_indices.size(), metadata.output_count);
        ASSERT_EQ(api.create_session(model, nullptr, &session), OR_OK);
        ASSERT_EQ(api.get_io_requirements(session, &requirements), OR_OK);
        ASSERT_EQ(requirements.input_count, metadata.input_count);
        ASSERT_EQ(requirements.output_count, metadata.output_count);
        ASSERT_NE(requirements.inputs, nullptr);
        ASSERT_NE(requirements.outputs, nullptr);
        for (or_io io : {OR_INPUT, OR_OUTPUT}) {
            auto count = io == OR_INPUT ? requirements.input_count : requirements.output_count;
            auto tensors = io == OR_INPUT ? requirements.inputs : requirements.outputs;
            for (uint32_t i = 0; i < count; ++i) {
                ASSERT_NE(tensors[i], nullptr);
                if (tensors[i]->shape.rank) {
                    ASSERT_NE(tensors[i]->shape.dimensions, nullptr);
                    ASSERT_NE(tensors[i]->byte_strides, nullptr);
                }
                ASSERT_FALSE(tensors[i]->data_dependent)
                    << "Fixture profile requires static tensors";
                ASSERT_GT(tensors[i]->minimum_byte_capacity, 0u);
                ASSERT_NE(select(*tensors[i], Method::Any), nullptr)
                    << "Fixture must provide a usable host POINTER option for every tensor";
            }
        }
    }
    void TearDown() override {
        if (!api.release_completion)
            return;
        for (auto c : completions) {
            if (provider.finish_pending)
                provider.finish_pending();
            or_completion_info info{};
            if (api.get_completion_info(c, &info) != OR_OK)
                unsafe_runtime("completion query failed during cleanup");
            if (!info.terminal)
                api.wait(c, wait_ns);
            if (api.get_completion_info(c, &info) != OR_OK || !info.terminal) {
                unsafe_runtime("pending work did not quiesce before cleanup deadline");
            }
            cleanup_ok(api.release_completion(c));
        }
        for (auto extra : extra_sessions)
            cleanup_ok(api.destroy_session(extra));
        if (session) {
            cleanup_ok(api.destroy_session(session));
        }
        for (auto b : buffers)
            cleanup_ok(api.release_buffer(b));
        if (model) {
            cleanup_ok(api.unload_model(model));
        }
        if (context) {
            cleanup_ok(api.destroy_context(context));
        }
    }
    static or_model_source memory_source(const std::vector<uint8_t> &bytes) {
        or_model_source source{};
        source.kind = OR_MODEL_SOURCE_MEMORY;
        source.source.memory = {bytes.data(), bytes.size()};
        return source;
    }
    const or_memory_option *select(const or_tensor_requirement &r, Method method) const {
        if (!r.memory_options)
            return nullptr;
        for (uint32_t i = 0; i < r.memory_option_count; ++i) {
            const auto &o = r.memory_options[i];
            const auto &c = o.configuration;
            if (c.reference != OR_REFERENCE_POINTER || !(c.accessibility & OR_ACCESSIBLE_HOST))
                continue;
            if ((method == Method::Any || method == Method::Runtime) &&
                c.owner == OR_OWNER_RUNTIME &&
                (o.pointer_binding_flags & OR_POINTER_RUNTIME_ALLOCATION))
                return &o;
            if ((method == Method::Any || method == Method::Import) &&
                c.owner == OR_OWNER_EXTERNAL && (caps.flags & OR_CAP_HOST_IMPORT) &&
                (o.pointer_binding_flags & OR_POINTER_REUSABLE_IMPORT))
                return &o;
            if ((method == Method::Any || method == Method::Borrow) &&
                c.owner == OR_OWNER_EXTERNAL && (caps.flags & OR_CAP_HOST_BIND) &&
                (o.pointer_binding_flags & OR_POINTER_BORROWED_BIND))
                return &o;
        }
        return nullptr;
    }
    bool supports(Method method) const {
        for (or_io io : {OR_INPUT, OR_OUTPUT}) {
            auto count = io == OR_INPUT ? requirements.input_count : requirements.output_count;
            auto tensors = io == OR_INPUT ? requirements.inputs : requirements.outputs;
            for (uint32_t i = 0; i < count; ++i)
                if (!select(*tensors[i], method))
                    return false;
        }
        return true;
    }
    void *host_storage(uint64_t capacity, uint64_t align) {
        EXPECT_GT(align, 0u);
        EXPECT_EQ(align & (align - 1), 0u);
        if (!align || align > UINT64_C(1048576) || capacity > UINT64_C(268435456))
            return nullptr;
        storage.emplace_back(static_cast<size_t>(capacity + align), 0);
        auto p = reinterpret_cast<uintptr_t>(storage.back().data());
        return reinterpret_cast<void *>((p + align - 1) & ~(align - 1));
    }
    or_tensor_binding binding(const Tensor &t) const {
        return {t.buffer,
                0,
                t.capacity,
                t.requirement->shape,
                t.requirement->byte_strides,
                t.option->configuration};
    }
    or_host_tensor_binding host_binding(const Tensor &t) const {
        return {t.data,
                t.capacity,
                t.requirement->shape,
                t.requirement->byte_strides,
                t.option->configuration};
    }
    or_status rebind(const Tensor &t) {
        if (t.buffer) {
            auto b = binding(t);
            return api.bind_tensor(session, t.io, t.index, &b);
        }
        auto b = host_binding(t);
        return api.bind_host_tensor(session, t.io, t.index, &b);
    }
    void allocate_storage(Tensor &t, Method method) {
        const auto *r = t.requirement;
        const auto *o = t.option;
        const auto io = t.io;
        const auto i = t.index;
        t.capacity = std::max(r->minimum_byte_capacity, o->minimum_byte_capacity);
        t.buffer = nullptr;
        t.data = nullptr;
        auto align = std::max(r->byte_alignment, o->byte_alignment);
        ASSERT_GT(align, 0u);
        ASSERT_EQ(align & (align - 1), 0u);
        if (o->configuration.owner == OR_OWNER_RUNTIME) {
            or_allocation_desc d{};
            d.configuration = o->configuration;
            d.access = io == OR_INPUT ? OR_READ : OR_WRITE;
            d.byte_length = t.capacity;
            d.byte_alignment = align;
            ASSERT_EQ(api.allocate_buffer(context, &d, &t.buffer), OR_OK);
            buffers.push_back(t.buffer);
            or_buffer_info info{};
            ASSERT_EQ(api.get_buffer_info(t.buffer, &info), OR_OK);
            EXPECT_EQ(info.owner, OR_OWNER_RUNTIME);
            EXPECT_EQ(info.reference, OR_REFERENCE_POINTER);
            EXPECT_TRUE(info.accessibility & OR_ACCESSIBLE_HOST);
            ASSERT_GE(info.view_byte_length, t.capacity);
            ASSERT_NE(info.host_data, nullptr);
            t.data = info.host_data;
        } else {
            t.data = host_storage(t.capacity, align);
            ASSERT_NE(t.data, nullptr);
            bool import = method == Method::Import ||
                          (method == Method::Any &&
                           (!(o->pointer_binding_flags & OR_POINTER_BORROWED_BIND) ||
                            !(caps.flags & OR_CAP_HOST_BIND)));
            if (import) {
                or_memory_desc d{};
                d.reference = OR_REFERENCE_POINTER;
                d.access = io == OR_INPUT ? OR_READ : OR_WRITE;
                d.allocation_byte_size = t.capacity;
                d.view_byte_length = t.capacity;
                d.handle.host_base = t.data;
                ASSERT_EQ(api.import_buffer(context, &d, &t.buffer), OR_OK);
                buffers.push_back(t.buffer);
            }
        }
        ASSERT_EQ(reinterpret_cast<uintptr_t>(t.data) % align, 0u);
        if (io == OR_INPUT) {
            ASSERT_LE(provider.inputs[i].size(), t.capacity);
            std::memcpy(t.data, provider.inputs[i].data(), provider.inputs[i].size());
        }
    }
    void prepare(Method method = Method::Any) {
        ASSERT_TRUE(bound.empty());
        for (or_io io : {OR_INPUT, OR_OUTPUT}) {
            auto count = io == OR_INPUT ? requirements.input_count : requirements.output_count;
            auto tensors = io == OR_INPUT ? requirements.inputs : requirements.outputs;
            for (uint32_t i = 0; i < count; ++i) {
                auto r = tensors[i];
                auto o = select(*r, method);
                ASSERT_NE(o, nullptr);
                if (method == Method::Any) {
                    auto selected = io == OR_INPUT ? provider.input_option_indices[i]
                                                    : provider.output_option_indices[i];
                    ASSERT_LT(selected, r->memory_option_count);
                    o = &r->memory_options[selected];
                    ASSERT_EQ(o->configuration.reference, OR_REFERENCE_POINTER);
                    ASSERT_TRUE(o->configuration.accessibility & OR_ACCESSIBLE_HOST);
                }
                Tensor t{io,
                         i,
                         r,
                         o,
                         nullptr,
                         nullptr,
                         0};
                ASSERT_NO_FATAL_FAILURE(allocate_storage(t, method));
                auto result = rebind(t);
                if (method != Method::Any && result == OR_INCOMPATIBLE) {
                    prepared = false;
                    return;
                }
                ASSERT_EQ(result, OR_OK);
                bound.push_back(t);
            }
        }
        auto result = api.validate_bindings(session);
        if (method != Method::Any && result == OR_INCOMPATIBLE) {
            prepared = false;
            return;
        }
        ASSERT_EQ(result, OR_OK);
        prepared = true;
    }
    or_completion execute(bool asynchronous = false) {
        or_completion c = nullptr;
        auto status = ((caps.flags & OR_CAP_SYNC_INFERENCE) && !asynchronous)
                          ? api.infer(session, &c)
                          : api.infer_async(session, &c);
        EXPECT_EQ(status, OR_OK);
        if (status != OR_OK)
            return nullptr;
        EXPECT_NE(c, nullptr);
        if (!c)
            return nullptr;
        completions.push_back(c);
        if ((caps.flags & OR_CAP_SYNC_INFERENCE) && !asynchronous) {
            or_completion_info info{};
            EXPECT_EQ(api.get_completion_info(c, &info), OR_OK);
            EXPECT_EQ(info.terminal, 1u) << "infer must return a terminal completion";
            or_outputs outputs{};
            EXPECT_EQ(api.get_outputs(session, c, &outputs), OR_OK)
                << "Synchronous results must be available without wait";
        }
        EXPECT_EQ(api.wait(c, wait_ns), OR_OK);
        return c;
    }
    void verify(or_completion c) {
        ASSERT_NE(c, nullptr);
        or_outputs out{};
        ASSERT_EQ(api.get_outputs(session, c, &out), OR_OK);
        ASSERT_EQ(out.count, metadata.output_count);
        ASSERT_NE(out.items, nullptr);
        std::vector<bool> seen(out.count, false);
        for (uint32_t i = 0; i < out.count; ++i) {
            ASSERT_NE(out.items[i], nullptr);
            auto &o = *out.items[i];
            ASSERT_LT(o.tensor_index, out.count);
            ASSERT_FALSE(seen[o.tensor_index]);
            seen[o.tensor_index] = true;
            const auto &selected = bound[requirements.input_count + o.tensor_index];
            ASSERT_GT(o.tensor.byte_length, 0u);
            ASSERT_LE(o.tensor.byte_length, selected.capacity);
            if (selected.buffer) {
                ASSERT_NE(o.tensor.buffer, nullptr);
                ASSERT_EQ(o.host_data, nullptr);
            } else {
                ASSERT_EQ(o.tensor.buffer, nullptr);
                ASSERT_EQ(o.tensor.byte_offset, 0u);
            }
            void *data = o.host_data;
            if (o.tensor.buffer) {
                or_buffer_info info{};
                ASSERT_EQ(api.get_buffer_info(o.tensor.buffer, &info), OR_OK);
                ASSERT_NE(info.host_data, nullptr);
                ASSERT_LE(o.tensor.byte_offset, info.view_byte_length);
                ASSERT_LE(o.tensor.byte_length, info.view_byte_length - o.tensor.byte_offset);
                data = static_cast<uint8_t *>(info.host_data) + o.tensor.byte_offset;
            }
            ASSERT_NE(data, nullptr);
            ASSERT_EQ(data, selected.data);
            auto required = requirements.outputs[o.tensor_index];
            ASSERT_EQ(o.tensor.shape.rank, required->shape.rank);
            ASSERT_EQ(o.tensor.configuration.owner,
                      bound[requirements.input_count + o.tensor_index].option->configuration.owner);
            ASSERT_EQ(o.tensor.configuration.reference, OR_REFERENCE_POINTER);
            ASSERT_EQ(o.tensor.configuration.accessibility,
                      bound[requirements.input_count + o.tensor_index]
                          .option->configuration.accessibility);
            ASSERT_EQ(
                o.tensor.configuration.transfer,
                bound[requirements.input_count + o.tensor_index].option->configuration.transfer);
            if (o.tensor.shape.rank) {
                ASSERT_NE(o.tensor.shape.dimensions, nullptr);
                ASSERT_NE(o.tensor.byte_strides, nullptr);
                for (uint32_t dim = 0; dim < o.tensor.shape.rank; ++dim) {
                    EXPECT_EQ(o.tensor.shape.dimensions[dim], required->shape.dimensions[dim]);
                    EXPECT_EQ(o.tensor.byte_strides[dim], required->byte_strides[dim]);
                }
            }
            ASSERT_TRUE(provider.verify_output(o.tensor_index, data, o.tensor.byte_length))
                << "Output " << o.tensor_index;
        }
    }
    void expect_rejected_binding(Tensor t, or_tensor_binding b, or_host_tensor_binding h) {
        auto status = t.buffer ? api.bind_tensor(session, t.io, t.index, &b)
                               : api.bind_host_tensor(session, t.io, t.index, &h);
        ASSERT_NE(status, OR_OK);
        ASSERT_NO_FATAL_FAILURE(verify(execute()));
    }
};

/** @brief Reject mismatched ABI versions without altering the caller table.
 * @details Both older and newer versions are rejected before table interpretation.
 * @pre A provider entry point and a byte-pattern-filled table are available.
 * @post OR_INCOMPATIBLE is returned and every table byte remains unchanged.
 */
TEST_F(HostConformance, BootstrapVersionMismatchPreservesTable) {
    for (auto v : {OR_RUNTIME_VERSION - 1, OR_RUNTIME_VERSION + 1}) {
        or_api table;
        std::memset(&table, 0xa5, sizeof(table));
        std::array<unsigned char, sizeof(table)> original{};
        std::memcpy(original.data(), &table, sizeof(table));
        EXPECT_EQ(provider.get_api(v, &table), OR_INCOMPATIBLE);
        EXPECT_EQ(std::memcmp(&table, original.data(), sizeof(table)), 0);
    }
}

/** @brief Require a complete core table and a usable host fixture.
 * @details Setup verifies every core entry, model counts and per-tensor host options.
 * @pre The replaceable provider supplies static deterministic model fixtures.
 * @post Runtime identity, supported execution mode and host requirements are valid.
 */
TEST_F(HostConformance, CoreTableAndHostProfile) {
    EXPECT_NE(caps.runtime_id, nullptr);
    EXPECT_NE(caps.runtime_version, nullptr);
    EXPECT_GT(caps.max_rank, 0u);
    for (uint32_t i = 0; i < metadata.input_count; ++i) {
        ASSERT_NE(metadata.inputs[i], nullptr);
        EXPECT_NE(metadata.inputs[i]->name, nullptr);
        EXPECT_NE(metadata.inputs[i]->layout, nullptr);
    }
}

/** @brief Preserve output handles on malformed model-source rejection.
 * @details Invalid source tags and empty memory sources cannot create a model.
 * @pre A live context and a sentinel output handle are provided.
 * @post Each call returns INVALID_ARGUMENT and leaves the sentinel untouched.
 */
TEST_F(HostConformance, InvalidSourcePreservesOutput) {
    or_model_source source{};
    or_model sentinel = model;
    EXPECT_EQ(api.load_model(context, &source, &sentinel), OR_INVALID_ARGUMENT);
    EXPECT_EQ(sentinel, model);
    source.kind = OR_MODEL_SOURCE_MEMORY;
    EXPECT_EQ(api.load_model(context, &source, &sentinel), OR_INVALID_ARGUMENT);
    EXPECT_EQ(sentinel, model);
}

/** @brief Stop borrowing serialized model bytes when loading returns.
 * @details A temporary package copy is overwritten and freed before execution.
 * @pre The original fixture model is unloaded after its session is destroyed.
 * @post Metadata and known output remain valid using runtime-retained model state.
 */
TEST_F(HostConformance, ModelSourceIsNotBorrowedAfterLoad) {
    ASSERT_EQ(api.destroy_session(session), OR_OK);
    session = nullptr;
    ASSERT_EQ(api.unload_model(model), OR_OK);
    model = nullptr;
    {
        auto temporary = provider.model_bytes;
        auto source = memory_source(temporary);
        ASSERT_EQ(api.load_model(context, &source, &model), OR_OK);
        std::fill(temporary.begin(), temporary.end(), 0xa5);
    }
    ASSERT_EQ(api.describe_model(model, &metadata), OR_OK);
    ASSERT_EQ(api.create_session(model, nullptr, &session), OR_OK);
    ASSERT_EQ(api.get_io_requirements(session, &requirements), OR_OK);
    ASSERT_NO_FATAL_FAILURE(prepare());
    ASSERT_NO_FATAL_FAILURE(verify(execute()));
}

/** @brief Load a provider-supplied path-backed model when available.
 * @details PATH uses the same known IO fixture as the MEMORY package.
 * @pre The provider supplies an existing path; otherwise the test explicitly skips.
 * @post The path model can be described and released without affecting the original.
 */
TEST_F(HostConformance, PathModelSource) {
    if (provider.model_path.empty())
        GTEST_SKIP() << "Provider has no PATH fixture";
    or_model_source source{};
    source.kind = OR_MODEL_SOURCE_PATH;
    source.source.path = provider.model_path.c_str();
    or_model loaded = nullptr;
    ASSERT_EQ(api.load_model(context, &source, &loaded), OR_OK);
    or_model_info info{};
    EXPECT_EQ(api.describe_model(loaded, &info), OR_OK);
    EXPECT_EQ(info.input_count, metadata.input_count);
    EXPECT_EQ(api.unload_model(loaded), OR_OK);
}

/** @brief Reject invalid session overrides without replacing the output handle.
 * @details An out-of-range input index is invalid even on static models.
 * @pre A loaded model and deliberately invalid indexed override are available.
 * @post INVALID_ARGUMENT is returned and the existing session handle is preserved.
 */
TEST_F(HostConformance, InvalidShapeOverridePreservesOutput) {
    uint64_t dimension = 1;
    or_input_shape_override override{metadata.input_count, {1, &dimension}};
    or_session_options options{OR_ALLOW_COMPILE, 1, &override};
    auto sentinel = session;
    EXPECT_EQ(api.create_session(model, &options, &sentinel), OR_INVALID_ARGUMENT);
    EXPECT_EQ(sentinel, session);
}

/** @brief Require explicit binding and complete-set validation before inference.
 * @details Fresh sessions cannot submit work with missing IO.
 * @pre An unbound session and an output sentinel are provided.
 * @post STATE_INVALID leaves the completion output unchanged and accepts no work.
 */
TEST_F(HostConformance, UnboundSessionRejectsInference) {
    EXPECT_EQ(api.validate_bindings(session), OR_STATE_INVALID);
    or_completion c = nullptr;
    auto status = (caps.flags & OR_CAP_SYNC_INFERENCE) ? api.infer(session, &c)
                                                        : api.infer_async(session, &c);
    EXPECT_EQ(status, OR_STATE_INVALID);
    EXPECT_EQ(c, nullptr);
}

/** @brief Run the known fixture repeatedly through advertised host configurations.
 * @details Each invocation reuses validated storage, and output metadata is checked.
 * @pre Every tensor has a provider-compatible advertised host memory option.
 * @post Both invocations succeed with oracle-approved outputs and unchanged inputs.
 */
TEST_F(HostConformance, RepeatedHostExecution) {
    ASSERT_NO_FATAL_FAILURE(prepare());
    for (unsigned n = 0; n < 2; ++n) {
        ASSERT_NO_FATAL_FAILURE(verify(execute()));
        for (auto &t : bound) {
            if (t.io == OR_INPUT) {
                EXPECT_EQ(std::memcmp(t.data,
                                      provider.inputs[t.index].data(),
                                      provider.inputs[t.index].size()),
                          0);
            }
        }
    }
}

/** @brief Exercise asynchronous execution whenever its capability is advertised.
 * @details This uses bounded completion waits without requiring mock timing controls.
 * @pre ASYNC_INFERENCE is supported and the provider baseline host set is validated.
 * @post Accepted asynchronous work completes successfully and outputs pass the oracle.
 */
TEST_F(HostConformance, AdvertisedAsynchronousExecution) {
    if (!(caps.flags & OR_CAP_ASYNC_INFERENCE))
        GTEST_SKIP() << "ASYNC_INFERENCE not advertised";
    ASSERT_NO_FATAL_FAILURE(prepare());
    ASSERT_NO_FATAL_FAILURE(verify(execute(true)));
}

class HostBindingMethods : public HostConformance, public testing::WithParamInterface<Method> {};
/** @brief Exercise runtime allocations, external imports and direct borrowed pointers.
 * @details The same public API tests run independently for each host storage method.
 * @pre All tensors advertise that method; unsupported optional methods explicitly skip.
 * @post Bound storage produces known outputs; imports/borrowed outputs keep caller ownership.
 */
TEST_P(HostBindingMethods, KnownOutputsForSelectedMethod) {
    if (!supports(GetParam()))
        GTEST_SKIP() << "Memory method not advertised for every fixture tensor";
    ASSERT_NO_FATAL_FAILURE(prepare(GetParam()));
    if (!prepared)
        GTEST_SKIP() << "Advertised per-tensor methods do not form a compatible complete set";
    ASSERT_NO_FATAL_FAILURE(verify(execute()));
}
INSTANTIATE_TEST_SUITE_P(HostMemory,
                         HostBindingMethods,
                         testing::Values(Method::Runtime, Method::Import, Method::Borrow));

/** @brief Reject unsupported transfer configurations without silently staging.
 * @details An unknown transfer tag cannot replace an existing valid association.
 * @pre A complete validated host binding set exists.
 * @post Binding rejects the selection; original validation and known execution remain valid.
 */
TEST_F(HostConformance, RejectedConfigurationPreservesBinding) {
    ASSERT_NO_FATAL_FAILURE(prepare());
    auto t = bound.front();
    auto b = binding(t);
    auto h = host_binding(t);
    b.configuration.transfer = h.configuration.transfer = UINT32_MAX;
    ASSERT_NO_FATAL_FAILURE(expect_rejected_binding(t, b, h));
}

/** @brief Reject overlapping input and output storage without replacing bindings.
 * @details Separate tensor shapes do not permit aliasing the same host allocation.
 * @pre A compatible borrowed-host set is validated and shared storage can hold either tensor.
 * @post The overlapping output bind fails; the prior output binding and known results remain valid.
 */
TEST_F(HostConformance, OverlappingHostBindingsAreRejected) {
    if (!supports(Method::Borrow))
        GTEST_SKIP() << "Borrowed host pointers unavailable";
    ASSERT_NO_FATAL_FAILURE(prepare(Method::Borrow));
    if (!prepared)
        GTEST_SKIP() << "Borrowed-pointer combination incompatible";
    auto &input = bound.front();
    const auto &output = bound[requirements.input_count];
    const auto capacity = std::max(input.capacity, output.capacity);
    const auto alignment =
        std::max(std::max(input.requirement->byte_alignment, input.option->byte_alignment),
                 std::max(output.requirement->byte_alignment, output.option->byte_alignment));
    auto *shared = host_storage(capacity, alignment);
    ASSERT_NE(shared, nullptr);
    std::memcpy(shared, provider.inputs.front().data(), provider.inputs.front().size());
    input.data = shared;
    input.capacity = capacity;
    ASSERT_EQ(rebind(input), OR_OK);
    ASSERT_EQ(api.validate_bindings(session), OR_OK);
    auto overlapping = host_binding(output);
    overlapping.data = shared;
    overlapping.byte_length = capacity;
    ASSERT_NE(api.bind_host_tensor(session, output.io, output.index, &overlapping), OR_OK);
    ASSERT_NO_FATAL_FAILURE(verify(execute()));
}

/** @brief Reject insufficient tensor capacity without replacing valid storage.
 * @details The attempted view is smaller than its required tensor capacity.
 * @pre A complete validated host binding set exists.
 * @post Rejection preserves the previous binding and successful known-output execution.
 */
TEST_F(HostConformance, RejectedCapacityPreservesBinding) {
    ASSERT_NO_FATAL_FAILURE(prepare());
    auto t = bound.front();
    auto b = binding(t);
    auto h = host_binding(t);
    b.byte_length = h.byte_length = t.requirement->minimum_byte_capacity - 1;
    ASSERT_NO_FATAL_FAILURE(expect_rejected_binding(t, b, h));
}

/** @brief Reject incompatible tensor shapes without corrupting prior bindings.
 * @details A changed dimension does not match the session's resolved shape.
 * @pre A non-scalar static host tensor and validated binding set exist.
 * @post The shape change is rejected and the original output oracle still passes.
 */
TEST_F(HostConformance, RejectedShapePreservesBinding) {
    ASSERT_NO_FATAL_FAILURE(prepare());
    auto t = bound.front();
    if (!t.requirement->shape.rank)
        GTEST_SKIP() << "Fixture has scalar first input";
    auto dims = std::vector<uint64_t>(t.requirement->shape.dimensions,
                                      t.requirement->shape.dimensions + t.requirement->shape.rank);
    dims[0] ^= 1;
    auto b = binding(t);
    auto h = host_binding(t);
    b.shape.dimensions = h.shape.dimensions = dims.data();
    ASSERT_NO_FATAL_FAILURE(expect_rejected_binding(t, b, h));
}

/** @brief Reject incompatible strides without losing prior validation.
 * @details One stride differs from the requirement rather than requesting conversion.
 * @pre A non-scalar host tensor and complete validated set exist.
 * @post Rejection leaves the old binding executable with known outputs.
 */
TEST_F(HostConformance, RejectedStridesPreserveBinding) {
    ASSERT_NO_FATAL_FAILURE(prepare());
    auto t = bound.front();
    if (!t.requirement->shape.rank)
        GTEST_SKIP() << "Fixture has scalar first input";
    ASSERT_NE(t.requirement->byte_strides, nullptr);
    auto strides = std::vector<uint64_t>(t.requirement->byte_strides,
                                         t.requirement->byte_strides + t.requirement->shape.rank);
    strides[0] ^= 1;
    auto b = binding(t);
    auto h = host_binding(t);
    b.byte_strides = h.byte_strides = strides.data();
    ASSERT_NO_FATAL_FAILURE(expect_rejected_binding(t, b, h));
}

/** @brief Reject misaligned borrowed pointers while preserving the previous binding.
 * @details A one-byte-shifted host pointer violates an advertised alignment greater than one.
 * @pre A compatible borrowed-pointer set with a nontrivial input alignment exists.
 * @post Binding fails without replacing storage or invalidating the previous valid set.
 */
TEST_F(HostConformance, MisalignedPointerPreservesBinding) {
    if (!supports(Method::Borrow))
        GTEST_SKIP() << "Borrowed pointers unavailable";
    ASSERT_NO_FATAL_FAILURE(prepare(Method::Borrow));
    if (!prepared)
        GTEST_SKIP() << "Borrowed-pointer combination incompatible";
    auto t = bound.front();
    if (std::max(t.requirement->byte_alignment, t.option->byte_alignment) <= 1)
        GTEST_SKIP() << "Fixture input permits byte alignment";
    auto bad = host_binding(t);
    bad.data = static_cast<uint8_t *>(bad.data) + 1;
    EXPECT_NE(api.bind_host_tensor(session, t.io, t.index, &bad), OR_OK);
    ASSERT_NO_FATAL_FAILURE(verify(execute()));
}

/** @brief Apply imported-view and tensor offsets exactly once.
 * @details Import an interior host view, then bind an interior tensor subrange in that view.
 * @pre HOST_IMPORT and a compatible all-import set are advertised for the fixture.
 * @post Buffer info identifies view byte zero and execution reads the final tensor bytes.
 */
TEST_F(HostConformance, ImportedSubrangeOffsets) {
    if (!supports(Method::Import))
        GTEST_SKIP() << "Reusable host imports unavailable";
    ASSERT_NO_FATAL_FAILURE(prepare(Method::Import));
    if (!prepared)
        GTEST_SKIP() << "Import combination incompatible";
    auto t = bound.front();
    auto align = std::max(t.requirement->byte_alignment, t.option->byte_alignment);
    auto base = host_storage(t.capacity + 2 * align, align);
    ASSERT_NE(base, nullptr);
    auto tensor_data = static_cast<uint8_t *>(base) + 2 * align;
    std::memcpy(tensor_data, provider.inputs[t.index].data(), provider.inputs[t.index].size());
    or_memory_desc imported{};
    imported.reference = OR_REFERENCE_POINTER;
    imported.access = OR_READ;
    imported.allocation_byte_size = t.capacity + 2 * align;
    imported.allocation_offset = align;
    imported.view_byte_length = t.capacity + align;
    imported.handle.host_base = base;
    or_buffer buffer = nullptr;
    ASSERT_EQ(api.import_buffer(context, &imported, &buffer), OR_OK);
    buffers.push_back(buffer);
    or_buffer_info info{};
    ASSERT_EQ(api.get_buffer_info(buffer, &info), OR_OK);
    EXPECT_EQ(info.host_data, static_cast<uint8_t *>(base) + align);
    EXPECT_EQ(info.view_byte_length, t.capacity + align);
    auto b = binding(t);
    b.buffer = buffer;
    b.byte_offset = align;
    ASSERT_EQ(api.bind_tensor(session, t.io, t.index, &b), OR_OK);
    ASSERT_EQ(api.validate_bindings(session), OR_OK);
    ASSERT_NO_FATAL_FAILURE(verify(execute()));
}

/** @brief Execute an explicitly advertised external STAGED host configuration.
 * @details The caller chooses STAGED without silently replacing another transfer mode.
 * @pre A compatible borrowed or imported host set has a STAGED option for one tensor.
 * @post The staged set validates and produces known outputs with the selected result configuration.
 */
TEST_F(HostConformance, AdvertisedStagedHostTransfer) {
    auto method = supports(Method::Borrow) ? Method::Borrow : Method::Import;
    if (!supports(method))
        GTEST_SKIP() << "No compatible external host method advertised";
    ASSERT_NO_FATAL_FAILURE(prepare(method));
    if (!prepared)
        GTEST_SKIP() << "External host combination incompatible";
    bool changed = false;
    for (auto &t : bound) {
        for (uint32_t i = 0; i < t.requirement->memory_option_count; ++i) {
            auto &option = t.requirement->memory_options[i];
            const auto &a = option.configuration;
            const auto &b = t.option->configuration;
            auto flag = t.buffer ? OR_POINTER_REUSABLE_IMPORT : OR_POINTER_BORROWED_BIND;
            if (a.owner == b.owner && a.accessibility == b.accessibility &&
                a.reference == b.reference && a.transfer == OR_TRANSFER_STAGED &&
                (option.pointer_binding_flags & flag)) {
                auto replacement = t;
                replacement.option = &option;
                ASSERT_NO_FATAL_FAILURE(allocate_storage(replacement, method));
                auto result = rebind(replacement);
                if (result == OR_INCOMPATIBLE)
                    GTEST_SKIP() << "Staged options do not form a compatible set";
                ASSERT_EQ(result, OR_OK);
                t = replacement;
                changed = true;
                break;
            }
        }
    }
    if (!changed)
        GTEST_SKIP() << "No STAGED option for selected external host storage";
    auto result = api.validate_bindings(session);
    if (result == OR_INCOMPATIBLE)
        GTEST_SKIP() << "Staged options do not form a compatible set";
    ASSERT_EQ(result, OR_OK);
    ASSERT_NO_FATAL_FAILURE(verify(execute()));
}

/** @brief Successful rebinding invalidates validation and prior result views.
 * @details Repeating the same binding arguments still counts as a binding change.
 * @pre A successful invocation and its unreleased terminal completion exist.
 * @post Old completion is stale; inference rejects until validate_bindings succeeds.
 */
TEST_F(HostConformance, RebindInvalidatesValidationAndOutputs) {
    ASSERT_NO_FATAL_FAILURE(prepare());
    auto old = execute();
    ASSERT_NE(old, nullptr);
    ASSERT_EQ(rebind(bound.front()), OR_OK);
    or_outputs out{};
    EXPECT_EQ(api.get_outputs(session, old, &out), OR_INVALID_ARGUMENT);
    or_completion rejected = nullptr;
    auto status = (caps.flags & OR_CAP_SYNC_INFERENCE) ? api.infer(session, &rejected)
                                                        : api.infer_async(session, &rejected);
    EXPECT_EQ(status, OR_STATE_INVALID);
    EXPECT_EQ(rejected, nullptr);
    ASSERT_EQ(api.validate_bindings(session), OR_OK);
    ASSERT_NO_FATAL_FAILURE(verify(execute()));
}

/** @brief Unbinding removes an association without allocating substitute storage.
 * @details A null binding must also invalidate complete-set validation.
 * @pre A complete validated host set exists.
 * @post Validation fails for missing IO until the original tensor is rebound.
 */
TEST_F(HostConformance, NullBindingOnlyUnbinds) {
    ASSERT_NO_FATAL_FAILURE(prepare());
    auto &t = bound.front();
    EXPECT_EQ(api.bind_tensor(session, t.io, t.index, nullptr), OR_OK);
    EXPECT_EQ(api.validate_bindings(session), OR_STATE_INVALID);
    ASSERT_EQ(rebind(t), OR_OK);
    ASSERT_EQ(api.validate_bindings(session), OR_OK);
    ASSERT_NO_FATAL_FAILURE(verify(execute()));
}

/** @brief A later accepted invocation makes previous completions stale for outputs.
 * @details Completion handles remain queryable independently of output-view lifetime.
 * @pre One successful invocation completion is retained while another runs.
 * @post The older get_outputs call rejects staleness; the newer outputs pass the oracle.
 */
TEST_F(HostConformance, NewInvocationInvalidatesOldOutputs) {
    ASSERT_NO_FATAL_FAILURE(prepare());
    auto old = execute();
    auto latest = execute();
    ASSERT_NE(old, nullptr);
    or_outputs out{};
    EXPECT_EQ(api.get_outputs(session, old, &out), OR_INVALID_ARGUMENT);
    or_completion_info info{};
    EXPECT_EQ(api.get_completion_info(old, &info), OR_OK);
    EXPECT_EQ(info.terminal, 1u);
    ASSERT_NO_FATAL_FAILURE(verify(latest));
}

/** @brief Terminal completions prevent session destruction until released.
 * @details Destruction reports busy without implicitly releasing child objects.
 * @pre A session has a successful but unreleased completion.
 * @post BUSY preserves the session, completion queries and output access.
 */
TEST_F(HostConformance, UnreleasedCompletionKeepsSessionAlive) {
    ASSERT_NO_FATAL_FAILURE(prepare());
    auto c = execute();
    ASSERT_NE(c, nullptr);
    expect_busy(api.destroy_session(session));
    ASSERT_NO_FATAL_FAILURE(verify(c));
}

/** @brief Bound resource handles cannot be released prematurely.
 * @details Buffer release is distinct from dropping an external owner or completion.
 * @pre A host resource method is available and its buffers are bound.
 * @post BUSY preserves every resource and known-output execution still succeeds.
 */
TEST_F(HostConformance, BoundBuffersRemainBusy) {
    if (!supports(Method::Runtime) && !supports(Method::Import))
        GTEST_SKIP() << "Only borrowed pointers advertised";
    ASSERT_NO_FATAL_FAILURE(prepare(supports(Method::Runtime) ? Method::Runtime : Method::Import));
    if (!prepared)
        GTEST_SKIP() << "No compatible complete resource-method set";
    for (auto b : buffers) {
        expect_busy(api.release_buffer(b));
    }
    ASSERT_NO_FATAL_FAILURE(verify(execute()));
}

/** @brief Parent objects stay alive while they have children.
 * @details Context/model destruction cannot recursively discard owned descendants.
 * @pre A context contains a loaded model with a live session.
 * @post Both destruction attempts report BUSY and metadata remains queryable.
 */
TEST_F(HostConformance, ParentDestructionIsBusy) {
    expect_busy(api.unload_model(model));
    expect_busy(api.destroy_context(context));
    or_model_info info{};
    EXPECT_EQ(api.describe_model(model, &info), OR_OK);
}

/** @brief Imported ranges reject overflow without changing the output handle.
 * @details Subtraction-safe range checks reject offsets/lengths outside an allocation.
 * @pre HOST_IMPORT is advertised and a valid aligned host block exists.
 * @post Invalid descriptors fail; caller storage and output sentinel remain owned unchanged.
 */
TEST_F(HostConformance, ImportRangeOverflowPreservesOutput) {
    if (!(caps.flags & OR_CAP_HOST_IMPORT))
        GTEST_SKIP() << "HOST_IMPORT not advertised";
    auto data = host_storage(64, 64);
    ASSERT_NE(data, nullptr);
    or_memory_desc d{};
    d.reference = OR_REFERENCE_POINTER;
    d.access = OR_READ;
    d.handle.host_base = data;
    d.allocation_byte_size = 64;
    d.allocation_offset = 63;
    d.view_byte_length = UINT64_MAX;
    or_buffer out = nullptr;
    EXPECT_NE(api.import_buffer(context, &d, &out), OR_OK);
    EXPECT_EQ(out, nullptr);
}

/** @brief A foreign-context buffer cannot bind directly to a session.
 * @details An independently allocated host resource still belongs to its creating context.
 * @pre A runtime-owned host option and second context are available.
 * @post INCOMPATIBLE leaves the original binding executable and both contexts clean up.
 */
TEST_F(HostConformance, ContextMismatchRejectsBinding) {
    if (!supports(Method::Runtime))
        GTEST_SKIP() << "Runtime allocations not advertised for this fixture";
    ASSERT_NO_FATAL_FAILURE(prepare(Method::Runtime));
    if (!prepared)
        GTEST_SKIP() << "No compatible runtime-allocation set";
    auto t = bound.front();
    or_context other = nullptr;
    ASSERT_EQ(api.create_context(&other), OR_OK);
    or_allocation_desc d{};
    d.configuration = t.option->configuration;
    d.access = OR_READ;
    d.byte_length = t.capacity;
    d.byte_alignment = std::max(t.requirement->byte_alignment, t.option->byte_alignment);
    or_buffer buffer = nullptr;
    auto allocated = api.allocate_buffer(other, &d, &buffer);
    if (allocated == OR_OK) {
        auto b = binding(t);
        b.buffer = buffer;
        EXPECT_EQ(api.bind_tensor(session, t.io, t.index, &b), OR_INCOMPATIBLE);
        EXPECT_EQ(api.release_buffer(buffer), OR_OK);
    } else
        ADD_FAILURE() << "Advertised allocation unavailable in second context: " << allocated;
    EXPECT_EQ(api.destroy_context(other), OR_OK);
    ASSERT_NO_FATAL_FAILURE(verify(execute()));
}

/** @brief Unsupported inference modes reject without accepting work.
 * @details Core entries remain nonnull even when their capability is absent.
 * @pre A validated host session exists; at least one execution mode is supported.
 * @post Each unavailable mode returns UNSUPPORTED with an unchanged completion output.
 */
TEST_F(HostConformance, UnsupportedExecutionModeDoesNotFallback) {
    ASSERT_NO_FATAL_FAILURE(prepare());
    if ((caps.flags & (OR_CAP_SYNC_INFERENCE | OR_CAP_ASYNC_INFERENCE)) ==
        (OR_CAP_SYNC_INFERENCE | OR_CAP_ASYNC_INFERENCE))
        GTEST_SKIP() << "Both modes advertised";
    or_completion out = nullptr;
    if (!(caps.flags & OR_CAP_SYNC_INFERENCE)) {
        EXPECT_EQ(api.infer(session, &out), OR_UNSUPPORTED);
    }
    if (!(caps.flags & OR_CAP_ASYNC_INFERENCE)) {
        EXPECT_EQ(api.infer_async(session, &out), OR_UNSUPPORTED);
    }
    EXPECT_EQ(out, nullptr);
    ASSERT_NO_FATAL_FAILURE(verify(execute()));
}

/** @brief A polling timeout does not cancel work or permit premature destruction.
 * @details Provider controls hold an async invocation pending deterministically.
 * @pre ASYNC_INFERENCE and explicit pending/release controls are supplied.
 * @post Poll returns TIMEOUT, mutations/destruction report BUSY, then execution completes.
 */
TEST_F(HostConformance, PendingCompletionRetainsResources) {
    if (!(caps.flags & OR_CAP_ASYNC_INFERENCE) || !provider.arm_pending ||
        !provider.finish_pending)
        GTEST_SKIP() << "No deterministic pending-execution fixture controls";
    ASSERT_NO_FATAL_FAILURE(prepare());
    provider.arm_pending();
    or_completion c = nullptr;
    ASSERT_EQ(api.infer_async(session, &c), OR_OK);
    ASSERT_NE(c, nullptr);
    completions.push_back(c);
    or_completion_info info{};
    ASSERT_EQ(api.get_completion_info(c, &info), OR_OK);
    EXPECT_EQ(info.terminal, 0u);
    EXPECT_EQ(api.wait(c, 0), OR_TIMEOUT);
    expect_busy(api.release_completion(c));
    expect_busy(api.destroy_session(session));
    EXPECT_EQ(api.reset_state(session), OR_BUSY);
    EXPECT_EQ(rebind(bound.front()), OR_BUSY);
    or_outputs out{};
    EXPECT_EQ(api.get_outputs(session, c, &out), OR_BUSY);
    provider.finish_pending();
    EXPECT_EQ(api.wait(c, wait_ns), OR_OK);
    ASSERT_NO_FATAL_FAILURE(verify(c));
}

/** @brief Accepted execution failure is separate from submission success.
 * @details The provider induces one execution error, after which network state needs reset.
 * @pre A validated host session and deterministic failure control are available.
 * @post Submission succeeds, completion/output report failure, reset permits a known-good run.
 */
TEST_F(HostConformance, ExecutionFailureRequiresReset) {
    if (!provider.fail_next_execution)
        GTEST_SKIP() << "No deterministic failure fixture control";
    ASSERT_NO_FATAL_FAILURE(prepare());
    provider.fail_next_execution();
    or_completion c = nullptr;
    auto accepted = (caps.flags & OR_CAP_SYNC_INFERENCE) ? api.infer(session, &c)
                                                          : api.infer_async(session, &c);
    ASSERT_EQ(accepted, OR_OK);
    ASSERT_NE(c, nullptr);
    completions.push_back(c);
    auto result = api.wait(c, wait_ns);
    EXPECT_NE(result, OR_OK);
    ASSERT_NE(result, OR_TIMEOUT);
    or_completion_info info{};
    ASSERT_EQ(api.get_completion_info(c, &info), OR_OK);
    EXPECT_EQ(info.terminal, 1u);
    EXPECT_EQ(info.result, result);
    or_outputs out{};
    EXPECT_EQ(api.get_outputs(session, c, &out), result);
    or_completion rejected = nullptr;
    auto rejection = (caps.flags & OR_CAP_SYNC_INFERENCE) ? api.infer(session, &rejected)
                                                           : api.infer_async(session, &rejected);
    EXPECT_EQ(rejection, OR_STATE_INVALID);
    EXPECT_EQ(rejected, nullptr);
    ASSERT_EQ(api.reset_state(session), OR_OK);
    ASSERT_NO_FATAL_FAILURE(verify(execute()));
}

/** @brief Unknown extension lookup preserves the caller's output pointer.
 * @details Host conformance does not require any native-memory extension.
 * @pre A context and deliberately unrecognized extension identifier are available.
 * @post UNSUPPORTED leaves the output untouched and provides a diagnostic string.
 */
TEST_F(HostConformance, UnknownExtensionPreservesOutput) {
    const void *out = &metadata;
    auto original = out;
    EXPECT_EQ(api.get_extension(context, "conformance.invalid.extension/999", &out),
              OR_UNSUPPORTED);
    EXPECT_EQ(out, original);
    EXPECT_NE(api.last_error_message(), nullptr);
}

TEST_F(HostConformance, IndependentSessionsExecuteConcurrently) {
    if (!(caps.flags & OR_CAP_SYNC_INFERENCE))
        GTEST_SKIP() << "Requires synchronous execution";
    if (!(caps.flags & OR_CAP_PARALLEL_SESSIONS))
        GTEST_SKIP() << "Parallel sessions not advertised";
    ASSERT_NO_FATAL_FAILURE(prepare());
    auto first = session;
    struct RestoreSession {
        or_session& slot;
        or_session original;
        ~RestoreSession() { slot = original; }
    } restore{session, first};
    auto first_bound = bound;
    auto first_requirements = requirements;
    or_session second = nullptr;
    ASSERT_EQ(api.create_session(model, nullptr, &second), OR_OK);
    extra_sessions.push_back(second);
    // Keep the primary session for fixture cleanup even when an assertion fails.
    session = second;
    ASSERT_EQ(api.get_io_requirements(second, &requirements), OR_OK);
    bound.clear();
    ASSERT_NO_FATAL_FAILURE(prepare());
    auto second_bound = bound;
    session = first;
    auto run_first = std::async(std::launch::async, [&]() {
        or_completion c = nullptr;
        auto result = api.infer(first, &c);
        return std::make_pair(result, c);
    });
    or_completion second_completion = nullptr;
    auto second_result = api.infer(second, &second_completion);
    auto first_result = run_first.get();
    if (first_result.second) completions.push_back(first_result.second);
    if (second_completion) completions.push_back(second_completion);
    ASSERT_EQ(first_result.first, OR_OK);
    ASSERT_EQ(second_result, OR_OK);
    bound = first_bound;
    requirements = first_requirements;
    ASSERT_NO_FATAL_FAILURE(verify(first_result.second));
    session = second;
    bound = second_bound;
    ASSERT_EQ(api.get_io_requirements(second, &requirements), OR_OK);
    ASSERT_NO_FATAL_FAILURE(verify(second_completion));
    session = first;
}

} // namespace
} // namespace or_conformance

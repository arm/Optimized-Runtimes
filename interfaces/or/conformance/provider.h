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
#pragma once

#include "../or_runtime.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace or_conformance {

/** @brief Runtime entry point and model-specific test data.
 * @details Replace make_provider(), not the conformance tests, to exercise another
 * runtime. Model fixtures have static, nonempty, dense tensors and at least one
 * input and output. The oracle compares actual output bytes without assuming a
 * network, dtype or numerical tolerance. Fixture inputs must produce the same
 * outputs on repeated execution; reset is used between independent checks.
 * Retain any dynamically loaded library for this record's lifetime.
 */
struct Provider {
    or_status(OR_CALL *get_api)(uint32_t, or_api *);
    uint32_t exported_version;
    std::vector<uint8_t> model_bytes;
    std::string model_path; // Optional PATH fixture of the same model.
    std::vector<std::vector<uint8_t>> inputs;
    // A known-compatible complete host set, indexed in get_io_requirements options.
    std::vector<uint32_t> input_option_indices;
    std::vector<uint32_t> output_option_indices;
    std::function<bool(uint32_t, const void *, size_t)> verify_output;
    // Optional deterministic controls; never part of the public runtime ABI.
    std::function<void()> arm_pending;
    std::function<void()> finish_pending;
    std::function<void()> fail_next_execution;
};

Provider make_provider();

} // namespace or_conformance

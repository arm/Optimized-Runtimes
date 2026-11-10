// SPDX-License-Identifier: Apache-2.0
#include <cstdlib>

#include "conformance/provider.h"
static or_api original;
static or_shape output_shape;
static const uint64_t* output_strides;

static or_status requirements(or_session session, or_io_requirements* out) {
  const auto result = original.get_io_requirements(session, out);
  if (result != OR_OK) {
    return result;
  }
  static or_tensor_requirement dynamic;
  static const or_tensor_requirement* outputs[] = {&dynamic};
  dynamic = *out->outputs[0];
  output_shape = dynamic.shape;
  output_strides = dynamic.byte_strides;
  dynamic.data_dependent = 1;
  dynamic.shape = {0, nullptr};
  dynamic.byte_strides = nullptr;
  out->outputs = outputs;
  return OR_OK;
}

static or_status bind(or_session session, or_io direction, uint32_t index,
                      const or_tensor_binding* binding) {
  auto concrete = *binding;
  if (direction == OR_OUTPUT) {
    if (binding->shape.rank || binding->shape.dimensions || binding->byte_strides) {
      return OR_INVALID_ARGUMENT;
    }
    concrete.shape = output_shape;
    concrete.byte_strides = output_strides;
  }
  return original.bind_tensor(session, direction, index, &concrete);
}

extern "C" or_status or_get_api(uint32_t version, or_api* out) {
  auto provider = or_conformance::make_provider();
  if (std::getenv("TEST_EXECUTION_FAILURE")) {
    provider.fail_next_execution();
  }
  const auto result = provider.get_api(version, out);
  if (result == OR_OK && std::getenv("TEST_DYNAMIC_OUTPUT")) {
    original = *out;
    out->get_io_requirements = requirements;
    out->bind_tensor = bind;
  }
  return result;
}

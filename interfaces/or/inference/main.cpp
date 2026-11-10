// SPDX-License-Identifier: Apache-2.0
#include <dlfcn.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "or_runtime.h"

namespace fs = std::filesystem;

namespace {

std::string quoted(const std::string& value) {
  std::ostringstream output;
  output << '"';

  for (const unsigned char character : value) {
    if (character == '"' || character == '\\') {
      output << '\\' << character;
    } else if (character < 0x20) {
      output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(character);
    } else {
      output << character;
    }
  }

  return output.str() + '"';
}

std::ofstream outputFile(const fs::path& path) {
  std::ofstream output;
  output.exceptions(std::ios::failbit | std::ios::badbit);
  output.open(path, std::ios::binary);
  return output;
}

uint32_t number(const char* const text) {
  uint32_t result = 0;
  const char* const end = text + std::strlen(text);
  const auto parsed = std::from_chars(text, end, result);

  if (parsed.ec != std::errc{} || parsed.ptr != end) {
    throw std::runtime_error("Expected a nonnegative integer");
  }

  return result;
}

struct Buffer {
  or_buffer handle = nullptr;
  or_buffer_info info{};
};

class Runtime {
 public:
  Runtime() = default;
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  ~Runtime() {
    // Do not unload code or backing storage if the runtime still owns live work.
    if (!cleanup()) {
      std::fputs("OR cleanup failed; terminating without unloading live runtime objects\n", stderr);
      std::_Exit(2);
    }
  }

  void setup(const fs::path& libraryPath, const fs::path& modelPath) {
    library = dlopen(fs::absolute(libraryPath).c_str(), RTLD_NOW | RTLD_LOCAL);

    if (!library) {
      throw std::runtime_error(dlerror());
    }

    using Getter = or_status(OR_CALL*)(uint32_t, or_api*);
    const auto getter = reinterpret_cast<Getter>(dlsym(library, "or_get_api"));

    if (!getter) {
      throw std::runtime_error("Library does not export or_get_api");
    }

    check(getter(OR_RUNTIME_VERSION, &api));

    if (!api.get_runtime_version || api.get_runtime_version() != OR_RUNTIME_VERSION ||
        !api.create_context || !api.get_capabilities || !api.load_model || !api.describe_model ||
        !api.create_session || !api.get_io_requirements || !api.allocate_buffer ||
        !api.get_buffer_info || !api.bind_tensor || !api.validate_bindings || !api.infer ||
        !api.get_completion_info || !api.get_outputs || !api.release_completion ||
        !api.destroy_session || !api.release_buffer || !api.unload_model || !api.destroy_context) {
      throw std::runtime_error("Incomplete or incompatible OR function table");
    }

    check(api.create_context(&context));
    or_capabilities capabilities{};
    check(api.get_capabilities(context, &capabilities));

    if (!(capabilities.flags & OR_CAP_SYNC_INFERENCE)) {
      throw std::runtime_error("This runner requires synchronous inference");
    }

    loadModel(modelPath);
    check(api.describe_model(model, &description));

    if (description.stateful || description.input_count != 1 || !description.inputs ||
        !description.inputs[0] || description.inputs[0]->dtype != OR_FLOAT32 ||
        description.inputs[0]->quantization.parameter_count || !description.output_count ||
        !description.outputs) {
      throw std::runtime_error("Expected stateless model with one unquantized Float32 input");
    }

    check(api.create_session(model, nullptr, &session));
    or_io_requirements requirements{};
    check(api.get_io_requirements(session, &requirements));

    if (requirements.input_count != 1 || requirements.output_count != description.output_count ||
        !requirements.inputs || !requirements.inputs[0] || !requirements.outputs) {
      throw std::runtime_error("Invalid IO requirements");
    }

    inputBytes = denseInputBytes(*requirements.inputs[0]);
    buffers.reserve(1 + requirements.output_count);
    allocate(*requirements.inputs[0], OR_INPUT, 0);

    for (uint32_t index = 0; index < requirements.output_count; ++index) {
      if (!requirements.outputs[index] || !description.outputs[index]) {
        throw std::runtime_error("Missing output requirement");
      }

      allocate(*requirements.outputs[index], OR_OUTPUT, index);
    }

    check(api.validate_bindings(session));
  }

  void readInput(const fs::path& path) const {
    if (fs::file_size(path) != inputBytes) {
      throw std::runtime_error("Input byte size does not match resolved dense Float32 shape");
    }

    std::ifstream input;
    input.exceptions(std::ios::failbit | std::ios::badbit);
    input.open(path, std::ios::binary);
    input.read(static_cast<char*>(buffers.front().info.host_data), inputBytes);
  }

  void infer() {
    check(api.infer(session, &completion));
    or_completion_info information{};
    check(api.get_completion_info(completion, &information));

    if (!information.terminal) {
      throw std::runtime_error("Synchronous runtime returned a pending completion");
    }

    check(information.result);
    check(api.get_outputs(session, completion, &outputs));
    check(api.release_completion(completion));
    completion = nullptr;
  }

  void writeOutputs(std::ostream& metadata, const fs::path& directory,
                    const std::string& prefix) const {
    if (outputs.count != description.output_count || !outputs.items) {
      throw std::runtime_error("Unexpected output count");
    }

    metadata << "[";
    std::vector<bool> seen(outputs.count, false);

    for (uint32_t index = 0; index < outputs.count; ++index) {
      const or_output* const output = outputs.items[index];

      if (!output || output->tensor_index >= outputs.count || seen[output->tensor_index]) {
        throw std::runtime_error("Invalid or duplicate output index");
      }

      const uint32_t tensorIndex = output->tensor_index;
      seen[tensorIndex] = true;
      const auto& tensor = output->tensor;
      const auto& buffer = buffers.at(tensorIndex + 1);

      if (tensor.buffer != buffer.handle || tensor.byte_offset > buffer.info.view_byte_length ||
          tensor.byte_length > buffer.info.view_byte_length - tensor.byte_offset ||
          (tensor.shape.rank && (!tensor.shape.dimensions || !tensor.byte_strides))) {
        throw std::runtime_error("Invalid output view");
      }

      const std::string filename = prefix + "-" + std::to_string(tensorIndex) + ".bin";
      auto binary = outputFile(directory / filename);
      binary.write(static_cast<const char*>(buffer.info.host_data) + tensor.byte_offset,
                   tensor.byte_length);
      binary.close();
      metadata << (index ? "," : "") << "{\"file\":" << quoted(filename)
               << ",\"index\":" << tensorIndex
               << ",\"dtype\":" << description.outputs[tensorIndex]->dtype
               << ",\"byte_length\":" << tensor.byte_length << ",\"shape\":[";

      for (uint32_t axis = 0; axis < tensor.shape.rank; ++axis) {
        metadata << (axis ? "," : "") << tensor.shape.dimensions[axis];
      }

      metadata << "],\"byte_strides\":[";

      for (uint32_t axis = 0; axis < tensor.shape.rank; ++axis) {
        metadata << (axis ? "," : "") << tensor.byte_strides[axis];
      }

      metadata << "]}";
    }

    metadata << "]";
  }

 private:
  void check(const or_status status) const {
    if (status != OR_OK) {
      const char* const detail = api.last_error_message ? api.last_error_message() : nullptr;
      throw std::runtime_error("OR status " + std::to_string(status) + ": " +
                               (detail ? detail : "no diagnostic"));
    }
  }

  void loadModel(const fs::path& path) {
    const auto size = fs::file_size(path);

    if (!size || size > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
      throw std::runtime_error("Invalid model file size");
    }

    std::vector<char> bytes(size);
    std::ifstream input;
    input.exceptions(std::ios::failbit | std::ios::badbit);
    input.open(path, std::ios::binary);
    input.read(bytes.data(), bytes.size());
    or_model_source source{};
    source.kind = OR_MODEL_SOURCE_MEMORY;
    source.source.memory = {bytes.data(), bytes.size()};
    check(api.load_model(context, &source, &model));
  }

  static uint64_t denseInputBytes(const or_tensor_requirement& requirement) {
    if (requirement.data_dependent || !requirement.shape.rank || !requirement.shape.dimensions ||
        !requirement.byte_strides) {
      throw std::runtime_error("Expected static dense input");
    }

    uint64_t bytes = sizeof(float);

    for (uint32_t axis = requirement.shape.rank; axis-- > 0;) {
      const uint64_t dimension = requirement.shape.dimensions[axis];

      if (!dimension || requirement.byte_strides[axis] != bytes ||
          dimension > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()) / bytes) {
        throw std::runtime_error("Unsupported input shape or strides");
      }

      bytes *= dimension;
    }

    if (bytes > requirement.minimum_byte_capacity) {
      throw std::runtime_error("Input requirement capacity is too small");
    }

    return bytes;
  }

  void allocate(const or_tensor_requirement& requirement, const or_io direction,
                const uint32_t index) {
    if (requirement.memory_option_count && !requirement.memory_options) {
      throw std::runtime_error("Missing memory options");
    }

    for (uint32_t optionIndex = 0; optionIndex < requirement.memory_option_count; ++optionIndex) {
      const auto& option = requirement.memory_options[optionIndex];
      const auto& configuration = option.configuration;

      if (configuration.owner != OR_OWNER_RUNTIME ||
          !(configuration.accessibility & OR_ACCESSIBLE_HOST) ||
          configuration.reference != OR_REFERENCE_POINTER ||
          configuration.transfer != OR_TRANSFER_DIRECT ||
          !(option.pointer_binding_flags & OR_POINTER_RUNTIME_ALLOCATION) ||
          (direction == OR_OUTPUT && option.output_handoff != OR_HANDOFF_BORROWED)) {
        continue;
      }

      const uint64_t capacity =
          std::max(requirement.minimum_byte_capacity, option.minimum_byte_capacity);

      if (!capacity ||
          capacity > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("Unsupported buffer capacity");
      }

      const or_allocation_desc allocation{
          configuration, direction == OR_INPUT ? OR_READ : OR_WRITE, capacity,
          std::max(requirement.byte_alignment, option.byte_alignment), nullptr};
      buffers.emplace_back();
      auto& buffer = buffers.back();
      check(api.allocate_buffer(context, &allocation, &buffer.handle));
      check(api.get_buffer_info(buffer.handle, &buffer.info));

      if (!buffer.info.host_data || !(buffer.info.accessibility & OR_ACCESSIBLE_HOST) ||
          buffer.info.view_byte_length < capacity) {
        throw std::runtime_error("Runtime did not return requested host storage");
      }

      std::memset(buffer.info.host_data, 0, capacity);
      const or_tensor_binding binding{
          buffer.handle, 0, capacity, requirement.shape, requirement.byte_strides, configuration};
      check(api.bind_tensor(session, direction, index, &binding));
      return;
    }

    throw std::runtime_error("No runtime-owned direct host-pointer allocation option");
  }

  bool cleanup() noexcept {
    if (completion && api.release_completion(completion) != OR_OK) {
      return false;
    }

    if (session && api.destroy_session(session) != OR_OK) {
      return false;
    }

    for (const auto& buffer : buffers) {
      if (buffer.handle && api.release_buffer(buffer.handle) != OR_OK) {
        return false;
      }
    }

    if (model && api.unload_model(model) != OR_OK) {
      return false;
    }

    if (context && api.destroy_context(context) != OR_OK) {
      return false;
    }

    return !library || dlclose(library) == 0;
  }

  void* library = nullptr;
  or_api api{};
  or_context context = nullptr;
  or_model model = nullptr;
  or_session session = nullptr;
  or_completion completion = nullptr;
  or_model_info description{};
  or_outputs outputs{};
  std::vector<Buffer> buffers;
  uint64_t inputBytes = 0;
};

void run(const int argc, char** const argv) {
  if (argc != 5 && argc != 7) {
    throw std::runtime_error(
        "Usage: or-run LIBRARY MODEL INPUTS.txt NEW_OUTPUT_DIR [WARMUP ITERATIONS]");
  }

  const uint32_t warmup = argc == 7 ? number(argv[5]) : 0;
  const uint32_t iterations = argc == 7 ? number(argv[6]) : 1;

  if (!iterations) {
    throw std::runtime_error("ITERATIONS must be positive");
  }

  const fs::path list = fs::absolute(argv[3]);
  std::ifstream inputs(list);

  if (!inputs) {
    throw std::runtime_error("Cannot open input list");
  }

  const fs::path destination = argv[4];

  if (!fs::create_directory(destination)) {
    throw std::runtime_error("Output directory already exists");
  }

  Runtime runtime;
  runtime.setup(argv[1], argv[2]);
  std::string line;
  size_t inputIndex = 0;

  while (std::getline(inputs, line)) {
    if (line.empty()) {
      continue;
    }

    const fs::path path = list.parent_path() / line;
    runtime.readInput(path);

    for (uint32_t iteration = 0; iteration < warmup; ++iteration) {
      runtime.infer();
    }

    std::vector<double> samples(iterations);

    for (auto& sample : samples) {
      const auto start = std::chrono::steady_clock::now();
      runtime.infer();
      sample = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                   .count();
    }

    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const double median = (sorted[(sorted.size() - 1) / 2] + sorted[sorted.size() / 2]) / 2;
    const std::string prefix = std::to_string(inputIndex++);
    auto metadata = outputFile(destination / (prefix + ".json"));
    metadata << std::setprecision(std::numeric_limits<double>::max_digits10)
             << "{\"input\":" << quoted(path.string()) << ",\"p50_ms\":" << median
             << ",\"mean_ms\":" << mean << ",\"samples_ms\":[";

    for (size_t index = 0; index < samples.size(); ++index) {
      metadata << (index ? "," : "") << samples[index];
    }

    metadata << "],\"outputs\":";
    runtime.writeOutputs(metadata, destination, prefix);
    metadata << "}\n";
    metadata.close();
  }

  if (inputs.bad() || !inputIndex) {
    throw std::runtime_error("Empty or unreadable input list");
  }

  auto summary = outputFile(destination / "run.json");
  summary << "{\"abi\":" << OR_RUNTIME_VERSION << ",\"inputs\":" << inputIndex
          << ",\"warmup\":" << warmup << ",\"iterations\":" << iterations
          << ",\"boundary\":\"infer + terminal-status check + get_outputs + release_completion; "
             "reusable IO\"}\n";
  summary.close();
}

}  // namespace

int main(const int argc, char** const argv) {
  try {
    run(argc, argv);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

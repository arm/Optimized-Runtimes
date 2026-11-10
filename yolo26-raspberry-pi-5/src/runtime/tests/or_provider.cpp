// The unchanged legacy API supplies a same-library adapter oracle, not model-quality evidence.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <vector>

#include "arm_yolo26n.h"
#include "provider.h"
#include "y26_or.h"
namespace or_conformance {
Provider make_provider() {
  const char* path = std::getenv("Y26_OR_MODEL");
  if (!path || !*path) {
    std::fputs("Y26_OR_MODEL absent; provide model.bin for real conformance\n", stderr);
    std::exit(77);
  }
  Provider p;
#ifdef OR_STATIC_PROVIDER
  p.get_api = y26_or_get_api;
  p.exported_version = OR_RUNTIME_VERSION;
#else
  p.get_api = or_get_api;
  p.exported_version = or_runtime_version;
#endif
  p.model_path = path;
  std::ifstream stream(path, std::ios::binary);
  p.model_bytes = std::vector<uint8_t>(std::istreambuf_iterator<char>(stream), {});
  std::unique_ptr<y26_native_runtime, decltype(&y26_native_destroy)> native(
      y26_native_create(p.model_bytes.data(), p.model_bytes.size()), y26_native_destroy);
  if (!native) throw std::runtime_error("Invalid YOLO conformance fixture");
  std::vector<float> input(y26_native_input_bytes(native.get()) / sizeof(float));
  for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i % 257) / 256.0f;
  auto expected =
      std::make_shared<std::vector<float>>(y26_native_output_bytes(native.get()) / sizeof(float));
  if (y26_native_execute(native.get(), input.data(), input.size() * sizeof(float), expected->data(),
                         expected->size() * sizeof(float)) != Y26_NATIVE_OK)
    throw std::runtime_error("Legacy YOLO oracle failed");
  p.inputs.emplace_back(input.size() * sizeof(float));
  std::memcpy(p.inputs[0].data(), input.data(), p.inputs[0].size());
  p.input_option_indices = {0};
  p.output_option_indices = {0};
  p.verify_output = [expected](uint32_t index, const void* data, size_t size) {
    return index == 0 && data && size == expected->size() * sizeof(float) &&
           std::memcmp(data, expected->data(), size) == 0;
  };
  return p;
}
}  // namespace or_conformance

#ifndef YOLO26N_MODEL_CONTRACT_H
#define YOLO26N_MODEL_CONTRACT_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace yolo26 {

// Fixed topology, public tensor, and execution constraints mirrored by
// converter.model_contract.
inline constexpr std::array<uint32_t, 6> kSupportedResolutions{320, 480, 640, 800, 960, 1280};
inline constexpr std::array<uint32_t, 3> kDetectionStrides{8, 16, 32};
inline constexpr uint32_t kBoxCoordinateChannels = 4;
inline constexpr uint32_t kClassificationChannelFloor = 64;
inline constexpr size_t kThreadCount = 4;
inline constexpr size_t kTensorAlignment = 64;

// The fixed topology references slots 0 through 207. The exporter writes those
// slots in the same order as its semantic weight-path inventory.
inline constexpr size_t kRuntimeWeightSlotCount = 208;

// model.bin wire-format constants mirrored by
// converter.model_format. The sole target is little-endian.
inline constexpr std::array<uint8_t, 8> kModelMagic{'Y', '2', '6', 'M', 'O', 'D', '0', '1'};
inline constexpr uint32_t kModelFormatVersion = 1;
inline constexpr uint32_t kModelAlignment = kTensorAlignment;
inline constexpr uint32_t kModelHeaderBytes = kModelAlignment;
inline constexpr uint32_t kModelEntryBytes = 16;
inline constexpr uint64_t kModelDirectoryOffset = kModelHeaderBytes;

constexpr bool IsSupportedResolution(uint32_t resolution) {
  for (uint32_t supported : kSupportedResolutions) {
    if (resolution == supported) return true;
  }
  return false;
}

constexpr size_t DetectionPositions(uint32_t resolution) {
  size_t positions = 0;
  for (uint32_t stride : kDetectionStrides) {
    const size_t side = resolution / stride;
    positions += side * side;
  }
  return positions;
}

constexpr bool DetectionOutputChannelsFit(uint32_t class_count) {
  return class_count <= UINT32_MAX - kBoxCoordinateChannels;
}

constexpr uint32_t ClassificationWidth(uint32_t class_count) {
  return class_count < kClassificationChannelFloor ? kClassificationChannelFloor : class_count;
}

// The largest class-dependent learned tensor has shape
// [class_count, 1, 1, classification_width] in FP32. This is a format
// constraint rather than a resource policy: a class count is representable by
// model.bin only when the runtime can calculate that tensor's exact byte size.
constexpr bool ClassDependentWeightBytesFit(uint32_t class_count) {
  if (!DetectionOutputChannelsFit(class_count) || class_count == 0) return false;
  const size_t width = ClassificationWidth(class_count);
  return size_t{class_count} <= std::numeric_limits<size_t>::max() / sizeof(float) / width;
}

}  // namespace yolo26

#endif

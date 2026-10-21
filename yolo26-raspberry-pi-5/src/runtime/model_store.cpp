#include "model_store.h"

#include <cstring>

namespace {

template <typename T>
bool ReadScalar(const std::vector<uint8_t>& bytes, size_t offset, T* value) {
  if (offset > bytes.size() || sizeof(T) > bytes.size() - offset) return false;
  std::memcpy(value, bytes.data() + offset, sizeof(T));
  return true;
}

bool BytesAreZero(const std::vector<uint8_t>& bytes, size_t offset, size_t size) {
  if (offset > bytes.size() || size > bytes.size() - offset) return false;
  for (size_t index = offset; index < offset + size; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

uint64_t Align(uint64_t value) {
  const uint64_t alignment = yolo26::kModelAlignment;
  return ((value + alignment - 1) / alignment) * alignment;
}

struct ModelHeader {
  uint32_t version;
  uint32_t resolution;
  uint32_t class_count;
  uint32_t slot_count;
  uint64_t payload_offset;
};

bool ReadHeader(const std::vector<uint8_t>& bytes, ModelHeader* header) {
  // model.bin is little-endian, as is the sole Linux AArch64 target. Scalar
  // reads use memcpy to avoid alignment and aliasing assumptions.
  return bytes.size() >= yolo26::kModelHeaderBytes &&
         std::memcmp(bytes.data(), yolo26::kModelMagic.data(), yolo26::kModelMagic.size()) == 0 &&
         ReadScalar(bytes, 8, &header->version) && ReadScalar(bytes, 12, &header->resolution) &&
         ReadScalar(bytes, 16, &header->class_count) &&
         ReadScalar(bytes, 20, &header->slot_count) &&
         ReadScalar(bytes, 24, &header->payload_offset) &&
         BytesAreZero(bytes, 32, yolo26::kModelHeaderBytes - 32);
}

bool HeaderMatchesRuntime(const std::vector<uint8_t>& bytes, const ModelHeader& header) {
  const uint64_t expected_payload_offset =
      Align(yolo26::kModelDirectoryOffset +
            uint64_t{yolo26::kModelEntryBytes} * yolo26::kRuntimeWeightSlotCount);
  return header.version == yolo26::kModelFormatVersion &&
         header.slot_count == yolo26::kRuntimeWeightSlotCount &&
         header.payload_offset == expected_payload_offset &&
         header.payload_offset <= bytes.size() &&
         yolo26::IsSupportedResolution(header.resolution) &&
         yolo26::ClassDependentWeightBytesFit(header.class_count);
}

}  // namespace

void ModelStore::Reset() {
  bytes_.clear();
  slots_ = {};
  resolution_ = 0;
  class_count_ = 0;
}

bool ModelStore::Load(const void* data, size_t size) {
  // Replace the complete store rather than retaining any state from a previous
  // model. On success, bytes_ owns the file data and slots_ indexes each weight
  // payload inside it. Every failure below resets both so callers cannot use
  // stale metadata or pointers.
  Reset();
  if (!data || size < yolo26::kModelHeaderBytes) return false;

  // Keep one stable copy so Slot() can return pointers into the model without
  // retaining the caller's buffer or copying each weight separately.
  const auto* first = static_cast<const uint8_t*>(data);
  bytes_.assign(first, first + size);

  // Reject files whose header does not match the one model format and set of
  // resolutions supported by this runtime.
  ModelHeader header{};
  if (!ReadHeader(bytes_, &header) || !HeaderMatchesRuntime(bytes_, header)) {
    Reset();
    return false;
  }
  resolution_ = header.resolution;
  class_count_ = header.class_count;

  // The directory contains one offset and byte count for every weight expected
  // by the fixed graph. Entries must be nonempty, aligned, ordered without
  // overlap, and wholly contained in the file before their locations are kept.
  uint64_t previous_end = header.payload_offset;
  for (uint32_t index = 0; index < yolo26::kRuntimeWeightSlotCount; ++index) {
    const size_t entry = yolo26::kModelDirectoryOffset + size_t{index} * yolo26::kModelEntryBytes;
    uint64_t offset = 0;
    uint64_t size = 0;
    if (!ReadScalar(bytes_, entry, &offset) || !ReadScalar(bytes_, entry + 8, &size) || size == 0 ||
        offset % yolo26::kModelAlignment != 0 || offset < previous_end || offset > bytes_.size() ||
        size > bytes_.size() - offset) {
      Reset();
      return false;
    }
    slots_[index] = {offset, size};
    previous_end = offset + size;
  }
  // The final payload must end with the file; otherwise unaccounted trailing
  // data would sit outside the validated slot directory.
  if (previous_end != bytes_.size()) {
    Reset();
    return false;
  }
  return true;
}

const void* ModelStore::Slot(uint32_t slot, size_t expected_size) const {
  if (slot >= slots_.size() || slots_[slot].size != expected_size) return nullptr;
  return bytes_.data() + slots_[slot].offset;
}

#ifndef YOLO26N_MODEL_STORE_H
#define YOLO26N_MODEL_STORE_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "model_contract.h"

// Owns one validated model.bin image. Slot pointers refer into bytes_ and stay
// valid until the store is loaded again or destroyed. A failed Load resets the
// store; callers must not use metadata or slots from an earlier model.
class ModelStore {
 public:
  // Copies the caller's bytes, then accepts only the fixed format, supported
  // dimensions, complete ordered slot directory, aligned non-overlapping
  // payloads, and no trailing bytes.
  bool Load(const void* data, size_t size);

  // Returns a stable payload pointer only when both slot identity and expected
  // byte extent match the fixed graph's request.
  const void* Slot(uint32_t slot, size_t expected_size) const;

  uint32_t resolution() const { return resolution_; }
  uint32_t class_count() const { return class_count_; }
  uint32_t classification_width() const { return yolo26::ClassificationWidth(class_count_); }

 private:
  struct SlotRecord {
    uint64_t offset = 0;
    uint64_t size = 0;
  };

  void Reset();

  std::vector<uint8_t> bytes_;
  std::array<SlotRecord, yolo26::kRuntimeWeightSlotCount> slots_{};
  uint32_t resolution_ = 0;
  uint32_t class_count_ = 0;
};

#endif

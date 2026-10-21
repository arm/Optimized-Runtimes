#ifndef YOLO26N_BUFFER_VALIDATION_H
#define YOLO26N_BUFFER_VALIDATION_H

#include <cstddef>
#include <cstdint>

namespace yolo26 {

// Returns true when the two memory ranges contain at least one common byte.
// Preprocessing and inference use this check to reject input and output buffers
// that overlap: writing output could otherwise destroy input bytes that have
// not been read yet. Empty ranges never overlap, and buffers that meet exactly
// at one range's end are adjacent rather than overlapping.
inline bool BuffersOverlap(const void* first, size_t first_bytes, const void* second,
                           size_t second_bytes) {
  if (first_bytes == 0 || second_bytes == 0) return false;
  const auto first_address = reinterpret_cast<uintptr_t>(first);
  const auto second_address = reinterpret_cast<uintptr_t>(second);
  // Compare the distance between start addresses. Calculating start + size
  // instead could wrap around the largest address representable by uintptr_t.
  if (first_address <= second_address) return second_address - first_address < first_bytes;
  return first_address - second_address < second_bytes;
}

}  // namespace yolo26

#endif

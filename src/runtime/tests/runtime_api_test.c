#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arm_yolo26n.h"

enum {
  // Little-endian model.bin header and first-directory-entry fields.
  kResolutionOffset = 12,
  kClassCountOffset = 16,
  kSlotCountOffset = 20,
  kFirstSlotSizeOffset = 72,
};

static const uint32_t kUnrepresentableClassCount = UINT32_C(2147483648);

/*
 * The Python API validates tensor sizes and allocates disjoint input and output
 * buffers before calling the native runtime. Its end-to-end tests therefore
 * cannot reach the C API's own rejection paths. This test calls that public C
 * boundary directly and verifies invalid buffers are rejected without starting
 * an inference or changing the observable execution count.
 */

static void fail(const char* message) {
  fprintf(stderr, "%s\n", message);
  exit(EXIT_FAILURE);
}

static unsigned char* read_model(const char* path, size_t* size) {
  FILE* file = fopen(path, "rb");
  if (file == NULL || fseek(file, 0, SEEK_END) != 0) fail("could not open test model");
  const long end = ftell(file);
  if (end <= 0 || fseek(file, 0, SEEK_SET) != 0) fail("could not size test model");

  *size = (size_t)end;
  unsigned char* data = malloc(*size);
  if (data == NULL || fread(data, 1, *size, file) != *size) fail("could not read test model");
  if (fclose(file) != 0) fail("could not close test model");
  return data;
}

static void write_u32(unsigned char* data, size_t offset, uint32_t value) {
  memcpy(data + offset, &value, sizeof(value));
}

static void require_model_rejected(unsigned char* data, size_t size, const char* message) {
  if (y26_native_create(data, size) != NULL) fail(message);
}

int main(void) {
  const char* model_path = getenv("YOLO26N_RUNTIME_API_TEST_MODEL");
  if (model_path == NULL || model_path[0] == '\0') {
    printf("YOLO26N_RUNTIME_API_TEST_MODEL is not set\n");
    return 77;
  }

  // Setup: create the runtime from caller-owned bytes through the public C API.
  size_t model_size = 0;
  unsigned char* model_data = read_model(model_path, &model_size);
  if (y26_native_create(NULL, model_size) != NULL || y26_native_create(model_data, 0) != NULL) {
    fail("invalid model span was not rejected");
  }

  // These fields control graph shapes and weight lookup. Each unsupported
  // value must fail at the public creation boundary, before inference is
  // possible. Restore the valid bytes after every independent case.
  unsigned char original[4];
  memcpy(original, model_data + kResolutionOffset, sizeof(original));
  write_u32(model_data, kResolutionOffset, 321);
  require_model_rejected(model_data, model_size, "unsupported resolution was not rejected");
  memcpy(model_data + kResolutionOffset, original, sizeof(original));

  memcpy(original, model_data + kClassCountOffset, sizeof(original));
  write_u32(model_data, kClassCountOffset, 0);
  require_model_rejected(model_data, model_size, "zero class count was not rejected");
  write_u32(model_data, kClassCountOffset, kUnrepresentableClassCount);
  require_model_rejected(model_data, model_size,
                         "unrepresentable model dimensions were not rejected");
  memcpy(model_data + kClassCountOffset, original, sizeof(original));

  memcpy(original, model_data + kSlotCountOffset, sizeof(original));
  uint32_t slot_count = 0;
  memcpy(&slot_count, original, sizeof(slot_count));
  if (slot_count <= 1) fail("test model has an unexpectedly small slot directory");
  write_u32(model_data, kSlotCountOffset, slot_count - 1);
  require_model_rejected(model_data, model_size, "incomplete weight directory was not rejected");
  memcpy(model_data + kSlotCountOffset, original, sizeof(original));

  // Changing a slot extent without changing its payload range leaves a
  // structurally readable file, but it no longer matches the fixed graph.
  unsigned char original_size[8];
  memcpy(original_size, model_data + kFirstSlotSizeOffset, sizeof(original_size));
  uint64_t first_slot_size = 0;
  memcpy(&first_slot_size, original_size, sizeof(first_slot_size));
  if (first_slot_size <= 1) fail("test model has an unexpectedly small first weight");
  --first_slot_size;
  memcpy(model_data + kFirstSlotSizeOffset, &first_slot_size, sizeof(first_slot_size));
  require_model_rejected(model_data, model_size, "incorrect weight extent was not rejected");
  memcpy(model_data + kFirstSlotSizeOffset, original_size, sizeof(original_size));

  y26_native_runtime* runtime = y26_native_create(model_data, model_size);
  if (runtime == NULL) fail("could not create runtime from model");
  free(model_data);

  const size_t required_input_bytes = y26_native_input_bytes(runtime);
  const size_t required_output_bytes = y26_native_output_bytes(runtime);
  const size_t shared_bytes =
      required_input_bytes > required_output_bytes ? required_input_bytes : required_output_bytes;
  float* input = malloc(required_input_bytes);
  float* output = malloc(required_output_bytes);
  float* shared = malloc(shared_bytes);
  if (input == NULL || output == NULL || shared == NULL) fail("could not allocate test buffers");

  // Exercise the two invalid-buffer cases without performing inference.
  const y26_native_status wrong_size = y26_native_execute(
      runtime, input, required_input_bytes - sizeof(float), output, required_output_bytes);
  const y26_native_status overlap =
      y26_native_execute(runtime, shared, required_input_bytes, shared, required_output_bytes);

  // Verify both calls were rejected before changing observable runtime state.
  if (wrong_size != Y26_NATIVE_INVALID_ARGUMENT) fail("incorrect input size was not rejected");
  if (overlap != Y26_NATIVE_INVALID_ARGUMENT) fail("overlapping buffers were not rejected");
  if (y26_native_calls(runtime) != 0) fail("rejected calls changed the execution count");

  free(shared);
  free(output);
  free(input);
  y26_native_destroy(runtime);
  return EXIT_SUCCESS;
}

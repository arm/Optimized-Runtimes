#ifndef ARM_YOLO26N_H
#define ARM_YOLO26N_H

#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__)
#define Y26_NATIVE_API __attribute__((visibility("default")))
#else
#define Y26_NATIVE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct y26_native_runtime y26_native_runtime;
typedef int32_t y26_native_status;

enum {
  Y26_NATIVE_OK = 0,
  Y26_NATIVE_INVALID_ARGUMENT = 1,
  Y26_NATIVE_EXECUTION_FAILED = 2,
  Y26_NATIVE_NONFINITE_OUTPUT = 3,
};

typedef int32_t y26_native_pixel_format;

enum {
  Y26_NATIVE_BGR = 0,
  Y26_NATIVE_RGB = 1,
  Y26_NATIVE_BGRX = 2,
};

/*
 * Copy a supported model.bin byte span and construct its fixed XNNPACK graphs.
 * The caller may release or modify model_data after this function returns.
 * Returns null if the bytes do not satisfy the supported model contract. The
 * returned instance is owned by the caller and must be released with
 * y26_native_destroy.
 */
Y26_NATIVE_API y26_native_runtime* y26_native_create(const void* model_data, size_t model_size);

/*
 * Convert an RGB, BGR, or BGRX image to the runtime's planar float input.
 * The exact output byte count selects one supported square model resolution;
 * the image must have one side equal to it and neither side may exceed it. The
 * function adds centered 114-valued padding but never resizes. image_bytes
 * must cover the last addressed pixel, including row stride. output must
 * provide exactly 3 * resolution * resolution floats. Input and output
 * buffers must not overlap.
 */
Y26_NATIVE_API y26_native_status y26_native_preprocess_image_no_resize(
    const uint8_t* image, size_t image_bytes, size_t height, size_t width, size_t row_stride_bytes,
    y26_native_pixel_format pixel_format, float* output, size_t output_bytes);

/*
 * Execute one inference from planar NCHW float input to [1, 4 + classes,
 * positions] float output. Query the exact buffer sizes from the runtime.
 * A runtime owns its model weights and scratch storage and is not reentrant;
 * calls using one instance must not overlap. The caller owns both buffers.
 */
Y26_NATIVE_API y26_native_status y26_native_execute(y26_native_runtime* runtime, const float* input,
                                                    size_t input_bytes, float* output,
                                                    size_t output_bytes);

/* The caller must not destroy a runtime while any call is using it. */
Y26_NATIVE_API void y26_native_destroy(y26_native_runtime* runtime);

/* Runtime metadata queries return zero when passed a null runtime. */
Y26_NATIVE_API uint64_t y26_native_calls(const y26_native_runtime* runtime);
Y26_NATIVE_API uint32_t y26_native_graphs(const y26_native_runtime* runtime);
Y26_NATIVE_API uint32_t y26_native_resolution(const y26_native_runtime* runtime);
Y26_NATIVE_API uint32_t y26_native_class_count(const y26_native_runtime* runtime);
Y26_NATIVE_API size_t y26_native_input_bytes(const y26_native_runtime* runtime);
Y26_NATIVE_API size_t y26_native_output_bytes(const y26_native_runtime* runtime);
Y26_NATIVE_API size_t y26_native_arena_bytes(const y26_native_runtime* runtime);

#ifdef __cplusplus
}
#endif

#endif

#include <arm_neon.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "arm_yolo26n.h"
#include "buffer_validation.h"
#include "model_contract.h"

namespace {

// Preprocessing preserves image pixels and pads the shorter dimension with the
// same value used by YOLO letterboxing. Source bytes are normalized from
// [0, 255] to [0, 1] while packed RGB/BGR channels are separated into the
// planar red, green, and blue tensors consumed by the runtime.
constexpr uint8_t kLetterboxValue = 114;
constexpr float kByteScale = 1.0F / 255.0F;

// Scalar pixels and padding use a lookup table so every byte value receives
// the same floating-point conversion without repeating division in the loops.
const std::array<float, 256>& NormalizedValues() {
  static const std::array<float, 256> values = []() {
    std::array<float, 256> result{};
    for (size_t index = 0; index < result.size(); ++index) {
      result[index] = static_cast<float>(index) / 255.0F;
    }
    return result;
  }();
  return values;
}

// Normalize eight values at once for the vectorized body of each image row.
void StoreNormalized8(uint8x8_t bytes, float* output) {
  const uint16x8_t values = vmovl_u8(bytes);
  const float32x4_t scale = vdupq_n_f32(kByteScale);
  vst1q_f32(output, vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(values))), scale));
  vst1q_f32(output + 4, vmulq_f32(vcvtq_f32_u32(vmovl_high_u16(values)), scale));
}

}  // namespace

extern "C" y26_native_status y26_native_preprocess_image_no_resize(
    const uint8_t* image, size_t image_bytes, size_t height, size_t width, size_t row_stride_bytes,
    y26_native_pixel_format pixel_format, float* output, size_t output_bytes) {
  // Reject missing buffers, empty images, and output sizes that cannot contain
  // three equal float planes.
  if (!image || !output || !height || !width || output_bytes % (3 * sizeof(float))) {
    return Y26_NATIVE_INVALID_ARGUMENT;
  }

  // Derive the square model resolution from the caller's output allocation.
  // This no-resize entry point accepts only supported resolutions and images
  // whose longer side already equals that resolution.
  const size_t input_plane = output_bytes / (3 * sizeof(float));
  const size_t input_size = static_cast<size_t>(std::sqrt(static_cast<double>(input_plane)));
  if (input_size * input_size != input_plane || input_size > std::numeric_limits<uint32_t>::max() ||
      !yolo26::IsSupportedResolution(static_cast<uint32_t>(input_size)) || height > input_size ||
      width > input_size || std::max(height, width) != input_size ||
      pixel_format < Y26_NATIVE_BGR || pixel_format > Y26_NATIVE_BGRX ||
      output_bytes != 3 * input_plane * sizeof(float)) {
    return Y26_NATIVE_INVALID_ARGUMENT;
  }

  // Validate the packed source layout before calculating its final byte. A row
  // may include trailing stride padding, but it must contain every image pixel.
  const size_t pixel_stride = pixel_format == Y26_NATIVE_BGRX ? 4 : 3;
  if (width > std::numeric_limits<size_t>::max() / pixel_stride ||
      row_stride_bytes < width * pixel_stride) {
    return Y26_NATIVE_INVALID_ARGUMENT;
  }
  const size_t final_row_bytes = width * pixel_stride;
  if ((height - 1) > (std::numeric_limits<size_t>::max() - final_row_bytes) / row_stride_bytes) {
    return Y26_NATIVE_INVALID_ARGUMENT;
  }
  const size_t required_image_bytes = (height - 1) * row_stride_bytes + final_row_bytes;
  if (image_bytes < required_image_bytes ||
      yolo26::BuffersOverlap(image, required_image_bytes, output, output_bytes)) {
    return Y26_NATIVE_INVALID_ARGUMENT;
  }

  const auto& normalized = NormalizedValues();
  // Centered padding implements the no-resize half of letterboxing. The ABI
  // requires one image side to match the model, so only one axis is padded.
  const size_t top = (input_size - height) / 2;
  const size_t left = (input_size - width) / 2;
  float* red_plane = output;
  float* green_plane = output + input_plane;
  float* blue_plane = output + 2 * input_plane;
  for (float* plane : {red_plane, green_plane, blue_plane}) {
    std::fill(plane, plane + top * input_size, normalized[kLetterboxValue]);
    std::fill(plane + (top + height) * input_size, plane + input_plane,
              normalized[kLetterboxValue]);
    if (left) {
      for (size_t y = top; y < top + height; ++y) {
        float* row = plane + y * input_size;
        std::fill(row, row + left, normalized[kLetterboxValue]);
        std::fill(row + left + width, row + input_size, normalized[kLetterboxValue]);
      }
    }
  }

  const size_t red_index = pixel_format == Y26_NATIVE_RGB ? 0 : 2;
  const size_t blue_index = 2 - red_index;
  // Separate packed source channels into planar output. Complete groups of
  // eight pixels use NEON; the scalar loop handles the remaining row tail.
  for (size_t y = 0; y < height; ++y) {
    const uint8_t* source = image + y * row_stride_bytes;
    const size_t offset = (top + y) * input_size + left;
    float* red = red_plane + offset;
    float* green = green_plane + offset;
    float* blue = blue_plane + offset;
    size_t x = 0;
    for (; x + 8 <= width; x += 8) {
      if (pixel_format == Y26_NATIVE_BGRX) {
        const uint8x8x4_t pixels = vld4_u8(source + 4 * x);
        StoreNormalized8(pixels.val[2], red + x);
        StoreNormalized8(pixels.val[1], green + x);
        StoreNormalized8(pixels.val[0], blue + x);
      } else {
        const uint8x8x3_t pixels = vld3_u8(source + 3 * x);
        StoreNormalized8(pixels.val[red_index], red + x);
        StoreNormalized8(pixels.val[1], green + x);
        StoreNormalized8(pixels.val[blue_index], blue + x);
      }
    }
    for (; x < width; ++x) {
      const uint8_t* pixel = source + pixel_stride * x;
      if (pixel_format == Y26_NATIVE_RGB) {
        red[x] = normalized[pixel[0]];
        green[x] = normalized[pixel[1]];
        blue[x] = normalized[pixel[2]];
      } else {
        blue[x] = normalized[pixel[0]];
        green[x] = normalized[pixel[1]];
        red[x] = normalized[pixel[2]];
      }
    }
  }
  return Y26_NATIVE_OK;
}

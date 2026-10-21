#include "fixed_schedule.h"

#include <arm_neon.h>

#include <cmath>
#include <cstring>
#include <initializer_list>

#include "fixed_memory_plan.h"
#include "model_contract.h"
#include "projection_optimization.h"

namespace {

// ---- Tensor sizing and constant construction --------------------------------

enum class TensorShapeKind : uint8_t {
  kInvalid,
  kScalar,
  kFeatureMap,
  kAttentionChannels,
  kAttentionMatrix,
  kAnchors,
  kDetectionChannels,
  kDetections,
};

struct TensorShape {
  TensorShapeKind kind;
  uint16_t channels;
  uint8_t stride;
};

constexpr TensorShape Scalar() { return {TensorShapeKind::kScalar, 1, 0}; }

constexpr TensorShape FeatureMap(uint16_t channels, uint8_t stride) {
  return {TensorShapeKind::kFeatureMap, channels, stride};
}

constexpr TensorShape AttentionChannels(uint16_t channels) {
  return {TensorShapeKind::kAttentionChannels, channels, 32};
}

constexpr TensorShape AttentionMatrix() { return {TensorShapeKind::kAttentionMatrix, 0, 32}; }

constexpr TensorShape DetectionChannels(uint16_t channels) {
  return {TensorShapeKind::kDetectionChannels, channels, 0};
}

TensorShape ShapeOf(TensorId tensor) {
  switch (tensor) {
    case TensorId::kBackboneScale:
    case TensorId::kDetectionScale:
    case TensorId::kCenterDivisor:
      return Scalar();
    case TensorId::kInputHalf:
      return FeatureMap(3, 1);
    case TensorId::kBackboneP3:
      return FeatureMap(128, 8);
    case TensorId::kP4Entry:
    case TensorId::kBackboneP4:
    case TensorId::kP4Fusion:
    case TensorId::kNeckP4:
    case TensorId::kDetectionP4Projection:
    case TensorId::kDetectionP4:
    case TensorId::kP5UpsampledProjection:
      return FeatureMap(128, 16);
    case TensorId::kP4EntrySkip:
    case TensorId::kP4EntryMain:
    case TensorId::kP4FusionSkip:
    case TensorId::kP4FusionMain:
    case TensorId::kP4Projection:
    case TensorId::kDetectionP4Skip:
    case TensorId::kDetectionP4Main:
      return FeatureMap(64, 16);
    case TensorId::kP5Entry:
    case TensorId::kBackboneAttentionEntry:
    case TensorId::kBackboneAttentionOutput:
    case TensorId::kDetectionP5Projection:
    case TensorId::kDetectionP5:
      return FeatureMap(256, 32);
    case TensorId::kP5EntrySkip:
    case TensorId::kP5EntryMain:
    case TensorId::kBackboneAttentionBypass:
    case TensorId::kBackboneAttentionResidual:
    case TensorId::kP5Projection:
    case TensorId::kDetectionP5Skip:
    case TensorId::kDetectionP5Main:
    case TensorId::kDetectionP5Residual:
      return FeatureMap(128, 32);
    case TensorId::kP3Fusion:
    case TensorId::kDetectionP3:
    case TensorId::kP4UpsampledProjection:
      return FeatureMap(64, 8);
    case TensorId::kP3FusionSkip:
    case TensorId::kP3FusionMain:
      return FeatureMap(32, 8);
    case TensorId::kBackboneQkv:
    case TensorId::kDetectionP5Qkv:
      return AttentionChannels(128);
    case TensorId::kBackboneQuery:
    case TensorId::kBackboneKey:
    case TensorId::kBackboneScaledQuery:
    case TensorId::kBackboneKeyCopy:
    case TensorId::kBackboneQueryCopy:
    case TensorId::kDetectionP5Query:
    case TensorId::kDetectionP5Key:
    case TensorId::kDetectionP5ScaledQuery:
    case TensorId::kDetectionP5KeyCopy:
    case TensorId::kDetectionP5QueryCopy:
      return AttentionChannels(32);
    case TensorId::kBackboneValue:
    case TensorId::kBackbonePositionalInput:
    case TensorId::kDetectionP5Value:
    case TensorId::kDetectionP5ValueCopy:
      return AttentionChannels(64);
    case TensorId::kBackboneAttentionMatrix:
    case TensorId::kBackboneAttentionMatrixCopy:
    case TensorId::kDetectionP5AttentionMatrix:
    case TensorId::kDetectionP5AttentionMatrixCopy:
      return AttentionMatrix();
    case TensorId::kAnchors:
      return {TensorShapeKind::kAnchors, 2, 0};
    case TensorId::kBoxRegression:
      return DetectionChannels(4);
    case TensorId::kBoxLower:
    case TensorId::kBoxUpper:
      return DetectionChannels(2);
    case TensorId::kDetectionsHalf:
      return {TensorShapeKind::kDetections, 0, 0};
    default:
      return {TensorShapeKind::kInvalid, 0, 0};
  }
}

size_t TensorBytes(TensorShape shape, uint32_t resolution, uint32_t class_count) {
  const size_t positions = yolo26::DetectionPositions(resolution);
  switch (shape.kind) {
    case TensorShapeKind::kInvalid:
      return 0;
    case TensorShapeKind::kScalar:
      return 2 * shape.channels;
    case TensorShapeKind::kFeatureMap: {
      const size_t side = resolution / shape.stride;
      return 2 * shape.channels * side * side;
    }
    case TensorShapeKind::kAttentionChannels: {
      const size_t side = resolution / shape.stride;
      return 4 * shape.channels * side * side;
    }
    case TensorShapeKind::kAttentionMatrix: {
      const size_t side = resolution / shape.stride;
      const size_t flattened = side * side;
      return 4 * flattened * flattened;
    }
    case TensorShapeKind::kAnchors:
    case TensorShapeKind::kDetectionChannels:
      return 2 * shape.channels * positions;
    case TensorShapeKind::kDetections:
      return 2 * (class_count + yolo26::kBoxCoordinateChannels) * positions;
  }
  return 0;
}

uint16_t HalfBits(float value) {
  const __fp16 half = static_cast<__fp16>(value);
  uint16_t bits;
  std::memcpy(&bits, &half, sizeof(bits));
  return bits;
}

void* ResolveTensor(const ScheduleState& state, TensorId tensor, uint8_t* arena, const float* input,
                    float* output) {
  switch (tensor) {
    case TensorId::kAnchorConstant:
      return const_cast<uint16_t*>(state.anchors.data());
    case TensorId::kStrideConstant:
      return const_cast<uint16_t*>(state.strides.data());
    case TensorId::kBackboneScaleConstant:
    case TensorId::kDetectionScaleConstant:
      return const_cast<float*>(&state.attention_scale);
    case TensorId::kCenterDivisorConstant:
      return const_cast<int64_t*>(&state.center_divisor);
    case TensorId::kInput:
      return const_cast<float*>(input);
    case TensorId::kOutput:
      return output;
    default:
      return arena + state.offsets[TensorIndex(tensor)];
  }
}

void StoreFloatAsHalf(void* source, void* output) {
  float value;
  std::memcpy(&value, source, sizeof(value));
  const uint16_t half = HalfBits(value);
  std::memcpy(output, &half, sizeof(half));
}

void StoreInt64AsHalf(void* source, void* output) {
  int64_t value;
  std::memcpy(&value, source, sizeof(value));
  const uint16_t half = HalfBits(static_cast<float>(value));
  std::memcpy(output, &half, sizeof(half));
}

void ConvertInputNchwFloatToNhwcHalf(const ScheduleState& state, const float* source,
                                     uint16_t* output) {
  const size_t plane = size_t{state.resolution} * state.resolution;
  for (size_t pixel = 0; pixel < plane; ++pixel) {
    for (size_t channel = 0; channel < 3; ++channel) {
      output[pixel * 3 + channel] = HalfBits(source[channel * plane + pixel]);
    }
  }
}

struct NearestWork {
  const uint16_t* source;
  uint16_t* output;
  size_t width;
  size_t channels;
};

// The neck uses exact nearest-neighbor 2x upsampling. Keeping it here avoids
// constructing an XNNPACK partition for a fixed, contiguous four-way copy.
void NearestRows(void* opaque, size_t start, size_t count) {
  auto* work = static_cast<NearestWork*>(opaque);
  const size_t output_width = work->width * 2;
  for (size_t y = start; y < start + count; ++y) {
    for (size_t x = 0; x < work->width; ++x) {
      const auto* source = work->source + (y * work->width + x) * work->channels;
      auto* top = work->output + ((2 * y) * output_width + 2 * x) * work->channels;
      auto* bottom = top + output_width * work->channels;
      for (size_t channel = 0; channel < work->channels; channel += 8) {
        const uint16x8_t value = vld1q_u16(source + channel);
        vst1q_u16(top + channel, value);
        vst1q_u16(top + work->channels + channel, value);
        vst1q_u16(bottom + channel, value);
        vst1q_u16(bottom + work->channels + channel, value);
      }
    }
  }
}

void Nearest2x(const uint16_t* source, uint16_t* output, size_t side, size_t channels,
               pthreadpool_t threadpool) {
  NearestWork work{source, output, side, channels};
  pthreadpool_parallelize_1d_tile_1d(threadpool, NearestRows, &work, side, 1, 0);
}

void ConvertDetectionsHalfToFloat(const ScheduleState& state, const uint16_t* input,
                                  float* output) {
  size_t index = 0;
  for (; index + 8 <= state.output_elements; index += 8) {
    const float16x8_t half = vreinterpretq_f16_u16(vld1q_u16(input + index));
    vst1q_f32(output + index, vcvt_f32_f16(vget_low_f16(half)));
    vst1q_f32(output + index + 4, vcvt_f32_f16(vget_high_f16(half)));
  }
  for (; index < state.output_elements; ++index) {
    __fp16 half;
    std::memcpy(&half, input + index, sizeof(half));
    output[index] = static_cast<float>(half);
  }
}

struct ExternalBinding {
  ExternalValueId external_id;
  TensorId tensor;
};

// ---- Partition binding and layout bridges -----------------------------------

class ScheduleExecution {
 public:
  ScheduleExecution(const ScheduleState& state,
                    const std::array<xnn_runtime_t, kGraphCount>& runtimes,
                    pthreadpool_t threadpool, uint8_t* arena, uint16_t* projection_scratch,
                    const float* input, float* output)
      : state_(state),
        runtimes_(runtimes),
        threadpool_(threadpool),
        arena_(arena),
        projection_scratch_(projection_scratch),
        input_(input),
        output_(output) {}

  // Keeping these shared operations out of line prevents GCC from duplicating
  // them throughout the fixed phase program. Their current code layout affects
  // measured Pi latency, so changes require new performance evidence.
  [[gnu::noinline]] void Copy(TensorId source, TensorId destination) const {
    std::memcpy(Tensor(destination), Tensor(source), state_.sizes[TensorIndex(destination)]);
  }

  void ConvertFloatToHalf(TensorId source, TensorId destination) const {
    StoreFloatAsHalf(Tensor(source), Tensor(destination));
  }

  void ConvertInt64ToHalf(TensorId source, TensorId destination) const {
    StoreInt64AsHalf(Tensor(source), Tensor(destination));
  }

  void ConvertInputLayout() const {
    ConvertInputNchwFloatToNhwcHalf(state_, input_,
                                    static_cast<uint16_t*>(Tensor(TensorId::kInputHalf)));
  }

  void ConvertOutputType() const {
    ConvertDetectionsHalfToFloat(
        state_, static_cast<const uint16_t*>(Tensor(TensorId::kDetectionsHalf)), output_);
  }

  bool RunStemProjection() const {
    const size_t p2_pixels = size_t{state_.resolution / 4} * (state_.resolution / 4);
    const size_t p3_pixels = size_t{state_.resolution / 8} * (state_.resolution / 8);
    const ProjectionGraphs graphs{
        runtimes_[GraphIndex(GraphId::kStemToP2)],
        runtimes_[GraphIndex(GraphId::kP2ToP3)],
        runtimes_[GraphIndex(GraphId::kP3ToP4)],
    };
    const ProjectionDimensions dimensions{p2_pixels * 16, p3_pixels * 32, p2_pixels, p3_pixels};
    const ProjectionBuffers buffers{
        projection_scratch_,
        static_cast<uint16_t*>(Tensor(TensorId::kInputHalf)),
        static_cast<uint16_t*>(Tensor(TensorId::kBackboneP3)),
        static_cast<uint16_t*>(Tensor(TensorId::kP4Entry)),
    };
    return RunProjectionOptimization(graphs, threadpool_, dimensions, buffers);
  }

  bool SplitContiguous(TensorId source, std::initializer_list<TensorId> destinations) const {
    return Split(source, destinations, 1);
  }

  bool SplitAttentionHeads(TensorId source, std::initializer_list<TensorId> destinations) const {
    // Q, K, and V are interleaved by head. The outer dimension keeps the two
    // heads separate while Split copies each head's contiguous channel ranges.
    return Split(source, destinations, 2);
  }

  bool SplitFeatureChannels(TensorId source, size_t source_channels,
                            std::initializer_list<TensorId> destinations) const {
    const size_t source_bytes = state_.sizes[TensorIndex(source)];
    const size_t bytes_per_position = source_channels * sizeof(uint16_t);
    if (source_bytes % bytes_per_position != 0) return false;
    return Split(source, destinations, source_bytes / bytes_per_position);
  }

  [[gnu::noinline]] bool RunPartition(
      GraphId graph, std::initializer_list<ExternalBinding> requested_bindings) const {
    std::array<xnn_external_value, 10> bindings{};
    if (requested_bindings.size() > bindings.size()) return false;
    size_t index = 0;
    for (ExternalBinding binding : requested_bindings) {
      bindings[index++] = {binding.external_id.value, Tensor(binding.tensor)};
    }
    xnn_runtime_t runtime = runtimes_[GraphIndex(graph)];
    return runtime && xnn_reshape_runtime(runtime) == xnn_status_success &&
           xnn_setup_runtime_v2(runtime, requested_bindings.size(), bindings.data()) ==
               xnn_status_success &&
           xnn_invoke_runtime(runtime) == xnn_status_success;
  }

  bool Upsample2x(TensorId source, TensorId destination, size_t channels) const {
    if (channels == 0) return false;
    const size_t source_bytes = state_.sizes[TensorIndex(source)];
    const size_t bytes_per_pixel = channels * sizeof(uint16_t);
    if (source_bytes % bytes_per_pixel != 0) return false;
    const size_t pixels = source_bytes / bytes_per_pixel;
    const size_t side = static_cast<size_t>(std::sqrt(static_cast<double>(pixels)));
    if (side * side != pixels) return false;
    Nearest2x(static_cast<const uint16_t*>(Tensor(source)),
              static_cast<uint16_t*>(Tensor(destination)), side, channels, threadpool_);
    return true;
  }

 private:
  void* Tensor(TensorId tensor) const {
    return ResolveTensor(state_, tensor, arena_, input_, output_);
  }

  [[gnu::noinline]] bool Split(TensorId source, std::initializer_list<TensorId> destinations,
                               size_t outer) const {
    if (outer == 0) return false;
    const size_t source_bytes = state_.sizes[TensorIndex(source)];
    if (source_bytes % outer != 0) return false;
    const size_t source_stride = source_bytes / outer;
    size_t source_offset = 0;
    for (TensorId destination : destinations) {
      const size_t destination_bytes = state_.sizes[TensorIndex(destination)];
      if (destination_bytes % outer != 0) return false;
      const size_t copy_bytes = destination_bytes / outer;
      auto* output = static_cast<uint8_t*>(Tensor(destination));
      auto* input = static_cast<const uint8_t*>(Tensor(source));
      for (size_t outer_index = 0; outer_index < outer; ++outer_index) {
        std::memcpy(output + outer_index * copy_bytes,
                    input + outer_index * source_stride + source_offset, copy_bytes);
      }
      source_offset += copy_bytes;
    }
    return source_offset == source_stride;
  }

  const ScheduleState& state_;
  const std::array<xnn_runtime_t, kGraphCount>& runtimes_;
  pthreadpool_t threadpool_;
  uint8_t* arena_;
  uint16_t* projection_scratch_;
  const float* input_;
  float* output_;
};

// ---- Fixed YOLO26n execution phases -----------------------------------------

bool RunInitialization(ScheduleExecution& execution) {
  // XNNPACK consumes these values as FP16 tensor inputs even though the
  // natural initialization arithmetic is FP32/integer.
  execution.Copy(TensorId::kAnchorConstant, TensorId::kAnchors);
  execution.ConvertFloatToHalf(TensorId::kBackboneScaleConstant, TensorId::kBackboneScale);
  execution.ConvertFloatToHalf(TensorId::kDetectionScaleConstant, TensorId::kDetectionScale);
  execution.ConvertInt64ToHalf(TensorId::kCenterDivisorConstant, TensorId::kCenterDivisor);
  // The public ABI is planar NCHW FP32; convolution partitions use NHWC FP16.
  execution.ConvertInputLayout();
  return execution.RunStemProjection();
}

bool RunBackbone(ScheduleExecution& execution) {
  // Move through the remaining backbone from P4 into P5 attention. Each
  // SplitContiguous call copies adjacent skip and main channel ranges into the
  // separate buffers consumed by the next XNNPACK graph. The attention entry
  // is likewise split into a preserved bypass and a residual branch; that
  // residual is projected into query, key, and value tensors, and the query is
  // scaled before matrix multiplication. Stop immediately if any copy or graph
  // invocation fails.
  if (!execution.SplitContiguous(TensorId::kP4Entry,
                                 {TensorId::kP4EntrySkip, TensorId::kP4EntryMain}) ||
      !execution.RunPartition(GraphId::kBackboneP4ToP5,
                              {{BackboneP4ToP5Bindings::kMainInput, TensorId::kP4EntryMain},
                               {BackboneP4ToP5Bindings::kSkipInput, TensorId::kP4EntrySkip},
                               {BackboneP4ToP5Bindings::kP4Output, TensorId::kBackboneP4},
                               {BackboneP4ToP5Bindings::kP5EntryOutput, TensorId::kP5Entry}}) ||
      !execution.SplitContiguous(TensorId::kP5Entry,
                                 {TensorId::kP5EntrySkip, TensorId::kP5EntryMain}) ||
      !execution.RunPartition(GraphId::kBackboneP5ToAttention,
                              {{BackboneP5ToAttentionBindings::kMainInput, TensorId::kP5EntryMain},
                               {BackboneP5ToAttentionBindings::kSkipInput, TensorId::kP5EntrySkip},
                               {BackboneP5ToAttentionBindings::kAttentionEntryOutput,
                                TensorId::kBackboneAttentionEntry}}) ||
      !execution.SplitContiguous(
          TensorId::kBackboneAttentionEntry,
          {TensorId::kBackboneAttentionBypass, TensorId::kBackboneAttentionResidual}) ||
      !execution.RunPartition(
          GraphId::kBackboneAttentionQkv,
          {{BackboneAttentionQkvBindings::kResidualInput, TensorId::kBackboneAttentionResidual},
           {BackboneAttentionQkvBindings::kQkvOutput, TensorId::kBackboneQkv}}) ||
      !execution.SplitAttentionHeads(
          TensorId::kBackboneQkv,
          {TensorId::kBackboneQuery, TensorId::kBackboneKey, TensorId::kBackboneValue}) ||
      !execution.RunPartition(
          GraphId::kBackboneAttentionScale,
          {{BackboneAttentionScaleBindings::kQueryInput, TensorId::kBackboneQuery},
           {BackboneAttentionScaleBindings::kScaleInput, TensorId::kBackboneScale},
           {BackboneAttentionScaleBindings::kScaledQueryOutput, TensorId::kBackboneScaledQuery}})) {
    return false;
  }

  // These copies give the matrix and positional branches independent external
  // inputs. Their distinct lifetimes also let the tuned 640-pixel arena layout
  // reuse the original buffers after each consumer finishes.
  execution.Copy(TensorId::kBackboneKey, TensorId::kBackboneKeyCopy);
  execution.Copy(TensorId::kBackboneValue, TensorId::kBackbonePositionalInput);
  execution.Copy(TensorId::kBackboneScaledQuery, TensorId::kBackboneQueryCopy);
  if (!execution.RunPartition(
          GraphId::kBackboneAttentionMatrix,
          {{BackboneAttentionMatrixBindings::kKeyInput, TensorId::kBackboneKeyCopy},
           {BackboneAttentionMatrixBindings::kQueryInput, TensorId::kBackboneQueryCopy},
           {BackboneAttentionMatrixBindings::kMatrixOutput, TensorId::kBackboneAttentionMatrix}})) {
    return false;
  }
  execution.Copy(TensorId::kBackboneAttentionMatrix, TensorId::kBackboneAttentionMatrixCopy);
  return execution.RunPartition(
      GraphId::kBackboneAttentionOutput,
      {{BackboneAttentionOutputBindings::kValueInput, TensorId::kBackboneValue},
       {BackboneAttentionOutputBindings::kPositionalInput, TensorId::kBackbonePositionalInput},
       {BackboneAttentionOutputBindings::kMatrixInput, TensorId::kBackboneAttentionMatrixCopy},
       {BackboneAttentionOutputBindings::kResidualInput, TensorId::kBackboneAttentionResidual},
       {BackboneAttentionOutputBindings::kBypassInput, TensorId::kBackboneAttentionBypass},
       {BackboneAttentionOutputBindings::kBackboneOutput, TensorId::kBackboneAttentionOutput},
       {BackboneAttentionOutputBindings::kP5ProjectionOutput, TensorId::kP5Projection}});
}

bool RunNeck(ScheduleExecution& execution) {
  if (!execution.Upsample2x(TensorId::kP5Projection, TensorId::kP5UpsampledProjection, 128) ||
      !execution.RunPartition(
          GraphId::kNeckP4Fusion,
          {{NeckP4FusionBindings::kUpsampledP5Input, TensorId::kP5UpsampledProjection},
           {NeckP4FusionBindings::kBackboneP4Input, TensorId::kBackboneP4},
           {NeckP4FusionBindings::kFusionOutput, TensorId::kP4Fusion}}) ||
      !execution.SplitContiguous(TensorId::kP4Fusion,
                                 {TensorId::kP4FusionSkip, TensorId::kP4FusionMain}) ||
      !execution.RunPartition(
          GraphId::kNeckP4Block,
          {{NeckP4BlockBindings::kMainInput, TensorId::kP4FusionMain},
           {NeckP4BlockBindings::kSkipInput, TensorId::kP4FusionSkip},
           {NeckP4BlockBindings::kNeckP4Output, TensorId::kNeckP4},
           {NeckP4BlockBindings::kP4ProjectionOutput, TensorId::kP4Projection}})) {
    return false;
  }
  return execution.Upsample2x(TensorId::kP4Projection, TensorId::kP4UpsampledProjection, 64) &&
         execution.RunPartition(
             GraphId::kNeckP3Fusion,
             {{NeckP3FusionBindings::kUpsampledP4Input, TensorId::kP4UpsampledProjection},
              {NeckP3FusionBindings::kFusionOutput, TensorId::kP3Fusion},
              {NeckP3FusionBindings::kBackboneP3Input, TensorId::kBackboneP3}}) &&
         execution.SplitFeatureChannels(TensorId::kP3Fusion, 64,
                                        {TensorId::kP3FusionSkip, TensorId::kP3FusionMain});
}

bool RunDetectionPyramid(ScheduleExecution& execution) {
  return execution.RunPartition(
             GraphId::kDetectionP3Path,
             {{DetectionP3PathBindings::kMainInput, TensorId::kP3FusionMain},
              {DetectionP3PathBindings::kSkipInput, TensorId::kP3FusionSkip},
              {DetectionP3PathBindings::kNeckP4Input, TensorId::kNeckP4},
              {DetectionP3PathBindings::kDetectionP3Output, TensorId::kDetectionP3},
              {DetectionP3PathBindings::kP4ProjectionOutput, TensorId::kDetectionP4Projection}}) &&
         execution.SplitContiguous(TensorId::kDetectionP4Projection,
                                   {TensorId::kDetectionP4Skip, TensorId::kDetectionP4Main}) &&
         execution.RunPartition(
             GraphId::kDetectionP4Path,
             {{DetectionP4PathBindings::kMainInput, TensorId::kDetectionP4Main},
              {DetectionP4PathBindings::kSkipInput, TensorId::kDetectionP4Skip},
              {DetectionP4PathBindings::kBackboneP5Input, TensorId::kBackboneAttentionOutput},
              {DetectionP4PathBindings::kDetectionP4Output, TensorId::kDetectionP4},
              {DetectionP4PathBindings::kP5ProjectionOutput, TensorId::kDetectionP5Projection}});
}

bool RunDetectionHead(ScheduleExecution& execution) {
  if (!execution.SplitFeatureChannels(TensorId::kDetectionP5Projection, 256,
                                      {TensorId::kDetectionP5Skip, TensorId::kDetectionP5Main}) ||
      !execution.RunPartition(
          GraphId::kDetectionP5Qkv,
          {{DetectionP5QkvBindings::kMainInput, TensorId::kDetectionP5Main},
           {DetectionP5QkvBindings::kResidualOutput, TensorId::kDetectionP5Residual},
           {DetectionP5QkvBindings::kQkvOutput, TensorId::kDetectionP5Qkv}}) ||
      !execution.SplitAttentionHeads(
          TensorId::kDetectionP5Qkv,
          {TensorId::kDetectionP5Query, TensorId::kDetectionP5Key, TensorId::kDetectionP5Value}) ||
      !execution.RunPartition(
          GraphId::kDetectionP5AttentionScale,
          {{DetectionP5AttentionScaleBindings::kQueryInput, TensorId::kDetectionP5Query},
           {DetectionP5AttentionScaleBindings::kScaleInput, TensorId::kDetectionScale},
           {DetectionP5AttentionScaleBindings::kScaledQueryOutput,
            TensorId::kDetectionP5ScaledQuery}})) {
    return false;
  }

  // As in the backbone attention block, separate external buffers preserve
  // each branch until its final consumer and make the tuned 640-pixel layout
  // explicit rather than relying on an in-place XNNPACK operation.
  execution.Copy(TensorId::kDetectionP5Key, TensorId::kDetectionP5KeyCopy);
  execution.Copy(TensorId::kDetectionP5Value, TensorId::kDetectionP5ValueCopy);
  execution.Copy(TensorId::kDetectionP5ScaledQuery, TensorId::kDetectionP5QueryCopy);
  if (!execution.RunPartition(
          GraphId::kDetectionP5AttentionMatrix,
          {{DetectionP5AttentionMatrixBindings::kKeyInput, TensorId::kDetectionP5KeyCopy},
           {DetectionP5AttentionMatrixBindings::kQueryInput, TensorId::kDetectionP5QueryCopy},
           {DetectionP5AttentionMatrixBindings::kMatrixOutput,
            TensorId::kDetectionP5AttentionMatrix}})) {
    return false;
  }
  execution.Copy(TensorId::kDetectionP5AttentionMatrix, TensorId::kDetectionP5AttentionMatrixCopy);
  if (!execution.RunPartition(
          GraphId::kDetectionP5AndBoxes,
          {{DetectionP5AndBoxesBindings::kP3Input, TensorId::kDetectionP3},
           {DetectionP5AndBoxesBindings::kP4Input, TensorId::kDetectionP4},
           {DetectionP5AndBoxesBindings::kValueInput, TensorId::kDetectionP5Value},
           {DetectionP5AndBoxesBindings::kValueCopyInput, TensorId::kDetectionP5ValueCopy},
           {DetectionP5AndBoxesBindings::kMatrixInput, TensorId::kDetectionP5AttentionMatrixCopy},
           {DetectionP5AndBoxesBindings::kResidualInput, TensorId::kDetectionP5Residual},
           {DetectionP5AndBoxesBindings::kSkipInput, TensorId::kDetectionP5Skip},
           {DetectionP5AndBoxesBindings::kMainInput, TensorId::kDetectionP5Main},
           {DetectionP5AndBoxesBindings::kP5Output, TensorId::kDetectionP5},
           {DetectionP5AndBoxesBindings::kBoxRegressionOutput, TensorId::kBoxRegression}}) ||
      !execution.SplitContiguous(TensorId::kBoxRegression,
                                 {TensorId::kBoxLower, TensorId::kBoxUpper})) {
    return false;
  }
  if (!execution.RunPartition(
          GraphId::kClassificationAndDecode,
          {{ClassificationAndDecodeBindings::kP3Input, TensorId::kDetectionP3},
           {ClassificationAndDecodeBindings::kP4Input, TensorId::kDetectionP4},
           {ClassificationAndDecodeBindings::kP5Input, TensorId::kDetectionP5},
           {ClassificationAndDecodeBindings::kAnchorsInput, TensorId::kAnchors},
           {ClassificationAndDecodeBindings::kBoxLowerInput, TensorId::kBoxLower},
           {ClassificationAndDecodeBindings::kBoxUpperInput, TensorId::kBoxUpper},
           {ClassificationAndDecodeBindings::kCenterDivisorInput, TensorId::kCenterDivisor},
           {ClassificationAndDecodeBindings::kStridesInput, TensorId::kStrideConstant},
           {ClassificationAndDecodeBindings::kDetectionsOutput, TensorId::kDetectionsHalf}})) {
    return false;
  }
  // The fixed head emits contiguous FP16 [box coordinates, class scores]; the
  // public ABI exposes the same logical layout in FP32.
  execution.ConvertOutputType();
  return true;
}

}  // namespace

// ---- Initialization and public schedule entry points ------------------------

bool InitializeSchedule(uint32_t resolution, uint32_t class_count, ScheduleState* state) {
  if (!state || !yolo26::IsSupportedResolution(resolution) ||
      !yolo26::ClassDependentWeightBytesFit(class_count)) {
    return false;
  }
  *state = {};
  state->resolution = resolution;
  state->class_count = class_count;
  state->attention_scale = 1.0F / std::sqrt(32.0F);
  const size_t positions = yolo26::DetectionPositions(resolution);
  state->anchors.reserve(2 * positions);
  state->strides.reserve(positions);
  for (uint32_t stride : yolo26::kDetectionStrides) {
    const uint32_t side = resolution / stride;
    for (uint32_t row = 0; row < side; ++row) {
      for (uint32_t column = 0; column < side; ++column) {
        state->anchors.push_back(HalfBits(static_cast<float>(column) + 0.5F));
      }
    }
  }
  for (uint32_t stride : yolo26::kDetectionStrides) {
    const uint32_t side = resolution / stride;
    for (uint32_t row = 0; row < side; ++row) {
      for (uint32_t column = 0; column < side; ++column) {
        state->anchors.push_back(HalfBits(static_cast<float>(row) + 0.5F));
        state->strides.push_back(HalfBits(static_cast<float>(stride)));
      }
    }
  }
  if (state->anchors.size() != 2 * positions || state->strides.size() != positions) {
    return false;
  }

  state->sizes[TensorIndex(TensorId::kAnchorConstant)] = state->anchors.size() * 2;
  state->sizes[TensorIndex(TensorId::kStrideConstant)] = state->strides.size() * 2;
  state->sizes[TensorIndex(TensorId::kBackboneScaleConstant)] = sizeof(float);
  state->sizes[TensorIndex(TensorId::kDetectionScaleConstant)] = sizeof(float);
  state->sizes[TensorIndex(TensorId::kCenterDivisorConstant)] = sizeof(int64_t);
  state->input_bytes = 3 * size_t{resolution} * resolution * sizeof(float);
  state->output_elements = (class_count + yolo26::kBoxCoordinateChannels) * positions;
  state->output_bytes = state->output_elements * sizeof(float);
  state->sizes[TensorIndex(TensorId::kInput)] = state->input_bytes;
  state->sizes[TensorIndex(TensorId::kOutput)] = state->output_bytes;
  for (size_t index = TensorIndex(TensorId::kAnchors); index < TensorIndex(TensorId::kOutput);
       ++index) {
    const auto tensor = static_cast<TensorId>(index);
    const size_t bytes = TensorBytes(ShapeOf(tensor), resolution, class_count);
    if (bytes == 0) return false;
    state->sizes[index] = bytes;
  }

  return yolo26::runtime_internal::PlanScheduleMemory(state);
}

bool ConfigureFixedProjection(const std::array<xnn_runtime_t, kGraphCount>& runtimes) {
  return ConfigureProjectionOptimization({
      runtimes[GraphIndex(GraphId::kStemToP2)],
      runtimes[GraphIndex(GraphId::kP2ToP3)],
      runtimes[GraphIndex(GraphId::kP3ToP4)],
  });
}

bool RunFixedSchedule(const ScheduleState& state,
                      const std::array<xnn_runtime_t, kGraphCount>& runtimes,
                      pthreadpool_t threadpool, uint8_t* arena, uint16_t* project,
                      const float* input, float* output) {
  ScheduleExecution execution(state, runtimes, threadpool, arena, project, input, output);
  return RunInitialization(execution) && RunBackbone(execution) && RunNeck(execution) &&
         RunDetectionPyramid(execution) && RunDetectionHead(execution);
}

bool OutputIsFinite(const ScheduleState& state, const float* output) {
  if (!output) return false;
  for (size_t index = 0; index < state.output_elements; ++index) {
    if (!std::isfinite(output[index])) return false;
  }
  return true;
}

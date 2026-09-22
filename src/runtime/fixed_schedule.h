#ifndef YOLO26N_FIXED_SCHEDULE_H
#define YOLO26N_FIXED_SCHEDULE_H

#include <pthreadpool.h>
#include <xnnpack.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "fixed_graph.h"

enum class TensorId : uint8_t {
  // State-owned source constants. These are never placed in the arena.
  kAnchorConstant,
  kStrideConstant,
  kBackboneScaleConstant,
  kDetectionScaleConstant,
  kCenterDivisorConstant,

  // Caller-owned FP32 input.
  kInput,

  // Arena-resident initialization results and input layout bridge.
  kAnchors,
  kBackboneScale,
  kDetectionScale,
  kCenterDivisor,
  kInputHalf,

  // Backbone feature pyramid and P5 attention.
  kBackboneP3,
  kP4Entry,
  kP4EntrySkip,
  kP4EntryMain,
  kBackboneP4,
  kP5Entry,
  kP5EntrySkip,
  kP5EntryMain,
  kBackboneAttentionEntry,
  kBackboneAttentionBypass,
  kBackboneAttentionResidual,
  kBackboneQkv,
  kBackboneQuery,
  kBackboneKey,
  kBackboneValue,
  kBackboneScaledQuery,
  kBackboneKeyCopy,
  kBackbonePositionalInput,
  kBackboneQueryCopy,
  kBackboneAttentionMatrix,
  kBackboneAttentionMatrixCopy,
  kBackboneAttentionOutput,

  // Top-down neck from P5 to P3.
  kP5Projection,
  kP5UpsampledProjection,
  kP4Fusion,
  kP4FusionSkip,
  kP4FusionMain,
  kNeckP4,
  kP4Projection,
  kP4UpsampledProjection,
  kP3Fusion,
  kP3FusionSkip,
  kP3FusionMain,

  // Bottom-up detection pyramid and P5 attention.
  kDetectionP3,
  kDetectionP4Projection,
  kDetectionP4Skip,
  kDetectionP4Main,
  kDetectionP4,
  kDetectionP5Projection,
  kDetectionP5Skip,
  kDetectionP5Main,
  kDetectionP5Residual,
  kDetectionP5Qkv,
  kDetectionP5Query,
  kDetectionP5Key,
  kDetectionP5Value,
  kDetectionP5ScaledQuery,
  kDetectionP5KeyCopy,
  kDetectionP5ValueCopy,
  kDetectionP5QueryCopy,
  kDetectionP5AttentionMatrix,
  kDetectionP5AttentionMatrixCopy,
  kDetectionP5,
  kBoxRegression,
  kBoxLower,
  kBoxUpper,
  kDetectionsHalf,

  // Caller-owned FP32 output.
  kOutput,
  kCount,
};

constexpr size_t TensorIndex(TensorId tensor) { return static_cast<size_t>(tensor); }

constexpr size_t kTensorCount = TensorIndex(TensorId::kCount);

struct ScheduleState {
  // Model dimensions that determine every dynamic tensor extent.
  uint32_t resolution = 0;
  uint32_t class_count = 0;
  // Complete tensor inventory. External and state-owned entries have sizes but
  // no meaningful arena offset; arena-resident entries have both.
  std::array<size_t, kTensorCount> offsets{};
  std::array<size_t, kTensorCount> sizes{};
  // Precomputed model constants retained for the lifetime of the runtime.
  std::vector<uint16_t> anchors;
  std::vector<uint16_t> strides;
  float attention_scale = 0.0F;
  int64_t center_divisor = 2;
  // Allocation and public-buffer requirements derived during initialization.
  size_t arena_bytes = 0;
  size_t project_bytes = 0;
  size_t input_bytes = 0;
  size_t output_bytes = 0;
  size_t output_elements = 0;
};

// Validates the supported model dimensions and completely derives constants,
// tensor sizes, memory placement, and public buffer requirements. On failure,
// state must not be used for execution.
bool InitializeSchedule(uint32_t resolution, uint32_t class_count, ScheduleState* state);

// Extracts the three packed projection operators after all fixed graphs have
// been built. The graph runtimes and their workspace must remain alive.
bool ConfigureFixedProjection(const std::array<xnn_runtime_t, kGraphCount>& runtimes);

// Executes one inference using an initialized state and configured graph set.
// arena and project provide at least the corresponding byte counts in state;
// input and output provide input_bytes and output_bytes. The runtime and these
// scratch buffers are mutable during invocation, so calls are not reentrant.
bool RunFixedSchedule(const ScheduleState& state,
                      const std::array<xnn_runtime_t, kGraphCount>& runtimes,
                      pthreadpool_t threadpool, uint8_t* arena, uint16_t* project,
                      const float* input, float* output);

// Checks the complete caller-visible FP32 output after a successful run.
bool OutputIsFinite(const ScheduleState& state, const float* output);

#endif

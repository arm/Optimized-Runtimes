#ifndef YOLO26N_FIXED_GRAPH_H
#define YOLO26N_FIXED_GRAPH_H

#include <xnnpack.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "model_store.h"

// Identifies the nineteen XNNPACK partitions that together implement the fixed
// YOLO26n topology. Declaration order is also the index into GraphContext's
// runtime array; it is a storage/construction order, not execution order.
enum class GraphId : uint8_t {
  kBackboneP4ToP5,
  kBackboneP5ToAttention,
  kBackboneAttentionQkv,
  kBackboneAttentionScale,
  kBackboneAttentionMatrix,
  kBackboneAttentionOutput,
  kNeckP4Fusion,
  kNeckP4Block,
  kNeckP3Fusion,
  kDetectionP3Path,
  kDetectionP4Path,
  kDetectionP5Qkv,
  kDetectionP5AttentionScale,
  kDetectionP5AttentionMatrix,
  kDetectionP5AndBoxes,
  kClassificationAndDecode,
  kStemToP2,
  kP2ToP3,
  kP3ToP4,
  kCount,
};

constexpr size_t GraphIndex(GraphId graph) { return static_cast<size_t>(graph); }

constexpr size_t kGraphCount = GraphIndex(GraphId::kCount);

struct ExternalValueId {
  uint32_t value;
};

// Named external-value contracts shared by graph construction and execution.
// Each kCount is passed to XNNPACK when reserving that partition's IDs.
struct BackboneP4ToP5Bindings {
  static constexpr ExternalValueId kMainInput{0}, kSkipInput{1}, kP4Output{2}, kP5EntryOutput{3};
  static constexpr uint32_t kCount = 4;
};
struct BackboneP5ToAttentionBindings {
  static constexpr ExternalValueId kMainInput{0}, kSkipInput{1}, kAttentionEntryOutput{2};
  static constexpr uint32_t kCount = 3;
};
struct BackboneAttentionQkvBindings {
  static constexpr ExternalValueId kResidualInput{0}, kQkvOutput{1};
  static constexpr uint32_t kCount = 2;
};
struct BackboneAttentionScaleBindings {
  static constexpr ExternalValueId kQueryInput{0}, kScaleInput{1}, kScaledQueryOutput{2};
  static constexpr uint32_t kCount = 3;
};
struct BackboneAttentionMatrixBindings {
  static constexpr ExternalValueId kKeyInput{0}, kQueryInput{1}, kMatrixOutput{2};
  static constexpr uint32_t kCount = 3;
};
struct BackboneAttentionOutputBindings {
  static constexpr ExternalValueId kPositionalInput{0}, kValueInput{1}, kMatrixInput{2},
      kResidualInput{3}, kBypassInput{4}, kBackboneOutput{5}, kP5ProjectionOutput{6};
  static constexpr uint32_t kCount = 7;
};
struct NeckP4FusionBindings {
  static constexpr ExternalValueId kUpsampledP5Input{0}, kBackboneP4Input{1}, kFusionOutput{2};
  static constexpr uint32_t kCount = 3;
};
struct NeckP4BlockBindings {
  static constexpr ExternalValueId kMainInput{0}, kSkipInput{1}, kNeckP4Output{2},
      kP4ProjectionOutput{3};
  static constexpr uint32_t kCount = 4;
};
struct NeckP3FusionBindings {
  static constexpr ExternalValueId kUpsampledP4Input{0}, kBackboneP3Input{1}, kFusionOutput{2};
  static constexpr uint32_t kCount = 3;
};
struct DetectionP3PathBindings {
  static constexpr ExternalValueId kMainInput{0}, kSkipInput{1}, kNeckP4Input{2},
      kDetectionP3Output{3}, kP4ProjectionOutput{4};
  static constexpr uint32_t kCount = 5;
};
struct DetectionP4PathBindings {
  static constexpr ExternalValueId kMainInput{0}, kSkipInput{1}, kBackboneP5Input{2},
      kDetectionP4Output{3}, kP5ProjectionOutput{4};
  static constexpr uint32_t kCount = 5;
};
struct DetectionP5QkvBindings {
  static constexpr ExternalValueId kMainInput{0}, kResidualOutput{1}, kQkvOutput{2};
  static constexpr uint32_t kCount = 3;
};
struct DetectionP5AttentionScaleBindings {
  static constexpr ExternalValueId kQueryInput{0}, kScaleInput{1}, kScaledQueryOutput{2};
  static constexpr uint32_t kCount = 3;
};
struct DetectionP5AttentionMatrixBindings {
  static constexpr ExternalValueId kKeyInput{0}, kQueryInput{1}, kMatrixOutput{2};
  static constexpr uint32_t kCount = 3;
};
struct DetectionP5AndBoxesBindings {
  static constexpr ExternalValueId kP3Input{0}, kP4Input{1}, kValueInput{2}, kValueCopyInput{3},
      kMatrixInput{4}, kResidualInput{5}, kSkipInput{6}, kMainInput{7}, kP5Output{8},
      kBoxRegressionOutput{9};
  static constexpr uint32_t kCount = 10;
};
struct ClassificationAndDecodeBindings {
  static constexpr ExternalValueId kP3Input{0}, kP4Input{1}, kP5Input{2}, kAnchorsInput{3},
      kBoxLowerInput{4}, kBoxUpperInput{5}, kCenterDivisorInput{6}, kStridesInput{7},
      kDetectionsOutput{8};
  static constexpr uint32_t kCount = 9;
};
struct StemToP2Bindings {
  static constexpr ExternalValueId kInput{0}, kP2ProjectionOutput{1};
  static constexpr uint32_t kCount = 2;
};
struct P2ToP3Bindings {
  static constexpr ExternalValueId kProjectionRightInput{0}, kProjectionLeftInput{1}, kP3Output{2};
  static constexpr uint32_t kCount = 3;
};
struct P3ToP4Bindings {
  static constexpr ExternalValueId kProjectionRightInput{0}, kProjectionLeftInput{1},
      kP3SkipInput{2}, kP4EntryOutput{3};
  static constexpr uint32_t kCount = 4;
};

// Resources shared while the fixed partitions are constructed. The caller
// owns every pointer and keeps the model, workspace, and threadpool alive for
// the lifetime of the created runtimes. BuildFixedGraphs fills each runtime
// array entry corresponding to GraphId.
struct GraphContext {
  const ModelStore* model;
  xnn_workspace_t workspace;
  pthreadpool_t threadpool;
  std::array<xnn_runtime_t, kGraphCount>* runtimes;
};

// Builds all fixed partitions. Returns false on an unsupported model shape,
// missing or incorrectly sized weight slot, or any XNNPACK construction error.
// Successfully created entries remain owned by context.runtimes and are
// released by the caller.
bool BuildFixedGraphs(const GraphContext& context);

#endif

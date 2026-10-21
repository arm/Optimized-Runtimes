#include "fixed_memory_plan.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>

#include "model_contract.h"

namespace yolo26::runtime_internal {
namespace {

constexpr size_t kTuned640ArenaBytes = 7'406'512;

// Boundaries between fixed-schedule operations. A tensor may reuse the storage
// of an input whose final consumer produces that tensor: the old value is dead
// at the boundary where the new value becomes ready. This compact lifetime
// model validates memory; it is not a second executable schedule.
enum class ScheduleBoundary : uint8_t {
  kConstantsReady,
  kInputConverted,
  kStemProjectionComplete,
  kP4EntrySplit,
  kBackboneP4ToP5Complete,
  kP5EntrySplit,
  kBackboneAttentionEntryReady,
  kBackboneAttentionEntrySplit,
  kBackboneQkvReady,
  kBackboneQkvSplit,
  kBackboneQueryScaled,
  kBackboneAttentionInputsCopied,
  kBackboneAttentionMatrixReady,
  kBackboneAttentionMatrixCopied,
  kBackboneComplete,
  kP5ProjectionUpsampled,
  kP4FusionReady,
  kP4FusionSplit,
  kNeckP4Ready,
  kP4ProjectionUpsampled,
  kP3FusionReady,
  kP3FusionSplit,
  kDetectionP3Ready,
  kDetectionP4ProjectionSplit,
  kDetectionP4Ready,
  kDetectionP5ProjectionSplit,
  kDetectionP5QkvReady,
  kDetectionP5QkvSplit,
  kDetectionP5QueryScaled,
  kDetectionP5AttentionInputsCopied,
  kDetectionP5AttentionMatrixReady,
  kDetectionP5AttentionMatrixCopied,
  kDetectionP5AndBoxesReady,
  kBoxRegressionSplit,
  kDetectionsReady,
  kOutputConverted,
};

struct Lifetime {
  ScheduleBoundary ready;
  ScheduleBoundary released;
};

struct PlannedTensor {
  TensorId id;
  size_t tuned_640_offset;
  Lifetime lifetime;
};

using B = ScheduleBoundary;

// These offsets form the compact activation-memory layout benchmarked for the
// standard 640-pixel, 80-class model on Raspberry Pi 5. They cannot be reused
// at other resolutions or class counts because those settings change tensor
// sizes. For those other shapes, PlanScheduleMemory places tensors sequentially
// with the required alignment; only P3Fusion reuses BackboneP3 storage after
// that input's final use. Overlaps in this compact layout are safe only where
// the recorded lifetimes do not intersect. Changing offsets, lifetimes, or
// declaration order requires the public runtime tests and new Pi performance
// evidence.
constexpr std::array kArenaTensors{
    PlannedTensor{TensorId::kAnchors, 7'372'800, {B::kConstantsReady, B::kDetectionsReady}},
    PlannedTensor{
        TensorId::kBackboneScale, 7'406'432, {B::kConstantsReady, B::kBackboneQueryScaled}},
    PlannedTensor{
        TensorId::kDetectionScale, 7'406'416, {B::kConstantsReady, B::kDetectionP5QueryScaled}},
    PlannedTensor{TensorId::kCenterDivisor, 7'406'400, {B::kConstantsReady, B::kDetectionsReady}},
    PlannedTensor{
        TensorId::kInputHalf, 4'915'200, {B::kInputConverted, B::kStemProjectionComplete}},
    PlannedTensor{
        TensorId::kBackboneP3, 4'915'200, {B::kStemProjectionComplete, B::kP3FusionReady}},
    PlannedTensor{TensorId::kP4Entry, 0, {B::kStemProjectionComplete, B::kP4EntrySplit}},
    PlannedTensor{
        TensorId::kP4EntrySkip, 2'099'200, {B::kP4EntrySplit, B::kBackboneP4ToP5Complete}},
    PlannedTensor{
        TensorId::kP4EntryMain, 1'894'400, {B::kP4EntrySplit, B::kBackboneP4ToP5Complete}},
    PlannedTensor{
        TensorId::kBackboneP4, 1'280'000, {B::kBackboneP4ToP5Complete, B::kP4FusionReady}},
    PlannedTensor{TensorId::kP5Entry, 1'689'600, {B::kBackboneP4ToP5Complete, B::kP5EntrySplit}},
    PlannedTensor{
        TensorId::kP5EntrySkip, 1'996'800, {B::kP5EntrySplit, B::kBackboneAttentionEntryReady}},
    PlannedTensor{
        TensorId::kP5EntryMain, 1'894'400, {B::kP5EntrySplit, B::kBackboneAttentionEntryReady}},
    PlannedTensor{TensorId::kBackboneAttentionEntry,
                  1'689'600,
                  {B::kBackboneAttentionEntryReady, B::kBackboneAttentionEntrySplit}},
    PlannedTensor{TensorId::kBackboneAttentionBypass,
                  4'608'000,
                  {B::kBackboneAttentionEntrySplit, B::kBackboneComplete}},
    PlannedTensor{TensorId::kBackboneAttentionResidual,
                  4'505'600,
                  {B::kBackboneAttentionEntrySplit, B::kBackboneComplete}},
    PlannedTensor{TensorId::kBackboneQkv, 1'689'600, {B::kBackboneQkvReady, B::kBackboneQkvSplit}},
    PlannedTensor{
        TensorId::kBackboneQuery, 6'553'600, {B::kBackboneQkvSplit, B::kBackboneQueryScaled}},
    PlannedTensor{TensorId::kBackboneKey,
                  4'864'000,
                  {B::kBackboneQkvSplit, B::kBackboneAttentionInputsCopied}},
    PlannedTensor{
        TensorId::kBackboneValue, 4'403'200, {B::kBackboneQkvSplit, B::kBackboneComplete}},
    PlannedTensor{TensorId::kBackboneScaledQuery,
                  4'812'800,
                  {B::kBackboneQueryScaled, B::kBackboneAttentionInputsCopied}},
    PlannedTensor{TensorId::kBackboneKeyCopy,
                  4'761'600,
                  {B::kBackboneAttentionInputsCopied, B::kBackboneAttentionMatrixReady}},
    PlannedTensor{TensorId::kBackbonePositionalInput,
                  4'300'800,
                  {B::kBackboneAttentionInputsCopied, B::kBackboneComplete}},
    PlannedTensor{TensorId::kBackboneQueryCopy,
                  4'710'400,
                  {B::kBackboneAttentionInputsCopied, B::kBackboneAttentionMatrixReady}},
    PlannedTensor{TensorId::kBackboneAttentionMatrix,
                  640'000,
                  {B::kBackboneAttentionMatrixReady, B::kBackboneAttentionMatrixCopied}},
    PlannedTensor{TensorId::kBackboneAttentionMatrixCopy,
                  0,
                  {B::kBackboneAttentionMatrixCopied, B::kBackboneComplete}},
    PlannedTensor{TensorId::kBackboneAttentionOutput,
                  4'096'000,
                  {B::kBackboneComplete, B::kDetectionP4Ready}},
    PlannedTensor{
        TensorId::kP5Projection, 640'000, {B::kBackboneComplete, B::kP5ProjectionUpsampled}},
    PlannedTensor{
        TensorId::kP5UpsampledProjection, 0, {B::kP5ProjectionUpsampled, B::kP4FusionReady}},
    PlannedTensor{TensorId::kP4Fusion, 819'200, {B::kP4FusionReady, B::kP4FusionSplit}},
    PlannedTensor{TensorId::kP4FusionSkip, 3'891'200, {B::kP4FusionSplit, B::kNeckP4Ready}},
    PlannedTensor{TensorId::kP4FusionMain, 3'686'400, {B::kP4FusionSplit, B::kNeckP4Ready}},
    PlannedTensor{TensorId::kNeckP4, 3'276'800, {B::kNeckP4Ready, B::kDetectionP3Ready}},
    PlannedTensor{TensorId::kP4Projection, 0, {B::kNeckP4Ready, B::kP4ProjectionUpsampled}},
    PlannedTensor{
        TensorId::kP4UpsampledProjection, 204'800, {B::kP4ProjectionUpsampled, B::kP3FusionReady}},
    PlannedTensor{TensorId::kP3Fusion, 4'915'200, {B::kP3FusionReady, B::kP3FusionSplit}},
    PlannedTensor{TensorId::kP3FusionSkip, 2'867'200, {B::kP3FusionSplit, B::kDetectionP3Ready}},
    PlannedTensor{TensorId::kP3FusionMain, 2'457'600, {B::kP3FusionSplit, B::kDetectionP3Ready}},
    PlannedTensor{TensorId::kDetectionP3, 0, {B::kDetectionP3Ready, B::kDetectionsReady}},
    PlannedTensor{TensorId::kDetectionP4Projection,
                  4'915'200,
                  {B::kDetectionP3Ready, B::kDetectionP4ProjectionSplit}},
    PlannedTensor{TensorId::kDetectionP4Skip,
                  2'073'600,
                  {B::kDetectionP4ProjectionSplit, B::kDetectionP4Ready}},
    PlannedTensor{TensorId::kDetectionP4Main,
                  1'868'800,
                  {B::kDetectionP4ProjectionSplit, B::kDetectionP4Ready}},
    PlannedTensor{TensorId::kDetectionP4, 1'459'200, {B::kDetectionP4Ready, B::kDetectionsReady}},
    PlannedTensor{TensorId::kDetectionP5Projection,
                  4'915'200,
                  {B::kDetectionP4Ready, B::kDetectionP5ProjectionSplit}},
    PlannedTensor{TensorId::kDetectionP5Skip,
                  2'483'200,
                  {B::kDetectionP5ProjectionSplit, B::kDetectionP5AndBoxesReady}},
    PlannedTensor{TensorId::kDetectionP5Main,
                  2'380'800,
                  {B::kDetectionP5ProjectionSplit, B::kDetectionP5AndBoxesReady}},
    PlannedTensor{TensorId::kDetectionP5Residual,
                  2'278'400,
                  {B::kDetectionP5QkvReady, B::kDetectionP5AndBoxesReady}},
    PlannedTensor{
        TensorId::kDetectionP5Qkv, 4'915'200, {B::kDetectionP5QkvReady, B::kDetectionP5QkvSplit}},
    PlannedTensor{TensorId::kDetectionP5Query,
                  2'739'200,
                  {B::kDetectionP5QkvSplit, B::kDetectionP5QueryScaled}},
    PlannedTensor{TensorId::kDetectionP5Key,
                  2'688'000,
                  {B::kDetectionP5QkvSplit, B::kDetectionP5AttentionInputsCopied}},
    PlannedTensor{TensorId::kDetectionP5Value,
                  2'176'000,
                  {B::kDetectionP5QkvSplit, B::kDetectionP5AndBoxesReady}},
    PlannedTensor{TensorId::kDetectionP5ScaledQuery,
                  2'636'800,
                  {B::kDetectionP5QueryScaled, B::kDetectionP5AttentionInputsCopied}},
    PlannedTensor{TensorId::kDetectionP5KeyCopy,
                  2'585'600,
                  {B::kDetectionP5AttentionInputsCopied, B::kDetectionP5AttentionMatrixReady}},
    PlannedTensor{TensorId::kDetectionP5ValueCopy,
                  2'073'600,
                  {B::kDetectionP5AttentionInputsCopied, B::kDetectionP5AndBoxesReady}},
    PlannedTensor{TensorId::kDetectionP5QueryCopy,
                  4'915'200,
                  {B::kDetectionP5AttentionInputsCopied, B::kDetectionP5AttentionMatrixReady}},
    PlannedTensor{TensorId::kDetectionP5AttentionMatrix,
                  819'200,
                  {B::kDetectionP5AttentionMatrixReady, B::kDetectionP5AttentionMatrixCopied}},
    PlannedTensor{TensorId::kDetectionP5AttentionMatrixCopy,
                  4'915'200,
                  {B::kDetectionP5AttentionMatrixCopied, B::kDetectionP5AndBoxesReady}},
    PlannedTensor{
        TensorId::kDetectionP5, 1'868'800, {B::kDetectionP5AndBoxesReady, B::kDetectionsReady}},
    PlannedTensor{TensorId::kBoxRegression,
                  2'585'600,
                  {B::kDetectionP5AndBoxesReady, B::kBoxRegressionSplit}},
    PlannedTensor{TensorId::kBoxLower, 2'686'400, {B::kBoxRegressionSplit, B::kDetectionsReady}},
    PlannedTensor{TensorId::kBoxUpper, 2'652'800, {B::kBoxRegressionSplit, B::kDetectionsReady}},
    PlannedTensor{TensorId::kDetectionsHalf, 4'915'200, {B::kDetectionsReady, B::kOutputConverted}},
};

constexpr bool ArenaInventoryIsContiguous() {
  for (size_t index = 0; index < kArenaTensors.size(); ++index) {
    if (TensorIndex(kArenaTensors[index].id) != TensorIndex(TensorId::kAnchors) + index) {
      return false;
    }
  }
  return true;
}

static_assert(kArenaTensors.size() ==
              TensorIndex(TensorId::kOutput) - TensorIndex(TensorId::kAnchors));
static_assert(ArenaInventoryIsContiguous());

size_t Align(size_t value) {
  const size_t alignment = yolo26::kTensorAlignment;
  return ((value + alignment - 1) / alignment) * alignment;
}

bool ByteRangesOverlap(size_t first_offset, size_t first_size, size_t second_offset,
                       size_t second_size) {
  return first_offset < second_offset + second_size && second_offset < first_offset + first_size;
}

bool LifetimesOverlap(Lifetime first, Lifetime second) {
  return first.ready < second.released && second.ready < first.released;
}

bool PlacementIsValid(const ScheduleState& state) {
  for (size_t first = 0; first < kArenaTensors.size(); ++first) {
    const size_t first_index = TensorIndex(kArenaTensors[first].id);
    const size_t first_offset = state.offsets[first_index];
    const size_t first_size = state.sizes[first_index];
    if (first_size == 0 || first_offset > state.arena_bytes ||
        first_size > state.arena_bytes - first_offset) {
      return false;
    }
    for (size_t second = first + 1; second < kArenaTensors.size(); ++second) {
      const size_t second_index = TensorIndex(kArenaTensors[second].id);
      if (LifetimesOverlap(kArenaTensors[first].lifetime, kArenaTensors[second].lifetime) &&
          ByteRangesOverlap(first_offset, first_size, state.offsets[second_index],
                            state.sizes[second_index])) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

bool PlanScheduleMemory(ScheduleState* state) {
  if (!state) return false;
  // Preserve the compact, benchmarked layout for the standard model. Other
  // supported shapes receive a conservative sequential layout calculated from
  // their tensor sizes below.
  const bool use_tuned_640_offsets = state->resolution == 640 && state->class_count == 80;
  size_t cursor = 0;
  size_t high_water = 0;
  for (const PlannedTensor& tensor : kArenaTensors) {
    const size_t index = TensorIndex(tensor.id);
    const size_t bytes = state->sizes[index];
    if (bytes == 0 || bytes > std::numeric_limits<size_t>::max() - yolo26::kTensorAlignment) {
      return false;
    }
    if (use_tuned_640_offsets) {
      state->offsets[index] = tensor.tuned_640_offset;
    } else if (tensor.id == TensorId::kP3Fusion) {
      // The fusion graph consumes BackboneP3 before publishing P3Fusion, so the
      // output takes over that input's arena range at the same boundary.
      state->offsets[index] = state->offsets[TensorIndex(TensorId::kBackboneP3)];
    } else {
      state->offsets[index] = Align(cursor);
      if (state->offsets[index] >
          std::numeric_limits<size_t>::max() - bytes - yolo26::kTensorAlignment) {
        return false;
      }
      cursor = state->offsets[index] + bytes + yolo26::kTensorAlignment;
    }
    if (state->offsets[index] > std::numeric_limits<size_t>::max() - bytes) return false;
    high_water = std::max(high_water, state->offsets[index] + bytes);
  }

  if (use_tuned_640_offsets) {
    state->arena_bytes = kTuned640ArenaBytes;
  } else {
    if (high_water > std::numeric_limits<size_t>::max() - yolo26::kTensorAlignment) return false;
    state->arena_bytes = Align(high_water + yolo26::kTensorAlignment);
  }
  const size_t first_pixels = size_t{state->resolution / 4} * (state->resolution / 4);
  state->project_bytes = 4 * first_pixels * 16;
  return ValidateScheduleMemory(*state);
}

bool ValidateScheduleMemory(const ScheduleState& state) { return PlacementIsValid(state); }

}  // namespace yolo26::runtime_internal

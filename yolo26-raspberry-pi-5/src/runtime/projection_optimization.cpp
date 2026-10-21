// Private XNNPACK adapter for the YOLO26n split-projection execution.

#include "projection_optimization.h"

#include <array>
#include <iterator>

#include "fixed_graph.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/subgraph.h"

namespace {

// XNNPACK stores the tensors and operations of each compiled graph in arrays.
// These constants identify the array entries created by the named declarations
// in fixed_graph_projection.inc. The adapter stops each graph at its 1x1
// projection, runs that projection one output-channel half at a time, and then
// resumes the next graph using those halves as external inputs. A changed graph
// layout is rejected instead of reading the wrong private XNNPACK entry.
constexpr uint32_t kModel2Cv1Output = 17;
constexpr size_t kModel2Cv1Operation = 2;

constexpr std::array<uint32_t, 2> kModel2ProjectionInputs{3, 4};
constexpr uint32_t kModel4Cv1Output = 31;
constexpr size_t kModel2BlockFirstOperation = 2;
constexpr size_t kModel4Cv1Operation = 11;

constexpr std::array<uint32_t, 2> kModel4ProjectionInputs{4, 5};
constexpr size_t kModel4BlockFirstOperation = 2;
constexpr size_t kModel4BlockOperationEnd = 13;

constexpr size_t kStemProjectionChannels = 16;
constexpr size_t kP3ProjectionChannels = 32;
constexpr size_t kProjectionTilePixels = 6;

bool RunOperators(xnn_runtime_t runtime, size_t first, size_t last) {
  if (!runtime || first >= last || last > runtime->num_ops) return false;
  for (size_t index = first; index < last; ++index) {
    bool ran_operator = false;
    for (size_t object = 0; object < XNN_MAX_OPERATOR_OBJECTS; ++object) {
      auto operation = runtime->opdata[index].operator_objects[object];
      if (operation) {
        if (xnn_run_operator(operation, runtime->threadpool) != xnn_status_success) {
          return false;
        }
        ran_operator = true;
      }
    }
    if (!ran_operator) return false;
  }
  return true;
}

bool RunDualProjection(xnn_runtime_t runtime, pthreadpool_t threadpool, size_t operation_index,
                       size_t pixels, size_t channels, uint16_t* left_output,
                       uint16_t* right_output) {
  // operation_index comes from the fixed graph layout above. There is no safe
  // private operator to inspect if the runtime or that array entry is missing.
  if (!runtime || operation_index >= runtime->num_ops) return false;
  auto operation = runtime->opdata[operation_index].operator_objects[0];
  // This function invokes XNNPACK's packed GEMM task directly. Refuse to do so
  // unless XNNPACK created the expected ready FP16 fully-connected operator and
  // populated every private callback and context used below.
  if (!operation || operation->type != xnn_operator_type_fully_connected_nc_f16 ||
      operation->state != xnn_run_state_ready || !operation->dynamic_context.gemm ||
      !operation->compute || !operation->compute[0].task_2d_tile_2d_dynamic_with_id) {
    return false;
  }

  const gemm_context& stock = operation->dynamic_context.gemm->gemm;
  gemm_context left = stock;
  gemm_context right = stock;
  left.c = left_output;
  left.cm_stride = channels * 2;
  left.nc = channels;
  // The fixed fully-connected operator has 2 * channels outputs. XNNPACK packs
  // them consecutively, one w_stride block per output channel; advancing by
  // channels selects the second projection without repacking learned weights.
  right.packed_w = reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(stock.packed_w) +
                                                 channels * stock.w_stride);
  right.c = right_output;
  right.cm_stride = channels * 2;
  right.nc = channels;

  auto task = operation->compute[0].task_2d_tile_2d_dynamic_with_id;
  constexpr uint32_t flags = PTHREADPOOL_FLAG_DISABLE_DENORMALS;
  pthreadpool_parallelize_2d_tile_2d_dynamic_with_uarch(threadpool, task, &left, 0,
                                                        XNN_MAX_UARCH_TYPES - 1, channels, pixels,
                                                        channels, kProjectionTilePixels, flags);
  pthreadpool_parallelize_2d_tile_2d_dynamic_with_uarch(threadpool, task, &right, 0,
                                                        XNN_MAX_UARCH_TYPES - 1, channels, pixels,
                                                        channels, kProjectionTilePixels, flags);
  return true;
}

bool MakeExternalInput(xnn_runtime_t runtime, uint32_t value_id) {
  if (!runtime || value_id >= runtime->num_values) return false;
  runtime->values[value_id].allocation_type = xnn_allocation_type_external;
  runtime->values[value_id].flags |= XNN_VALUE_FLAG_EXTERNAL_INPUT;
  return true;
}

bool MakeExternalOutput(xnn_runtime_t runtime, uint32_t value_id) {
  if (!runtime || value_id >= runtime->num_values) return false;
  runtime->values[value_id].allocation_type = xnn_allocation_type_external;
  runtime->values[value_id].flags |= XNN_VALUE_FLAG_EXTERNAL_OUTPUT;
  return true;
}

}  // namespace

bool ConfigureProjectionOptimization(const ProjectionGraphs& graphs) {
  if (!graphs.stem_to_p2 || !graphs.p2_to_p3 || !graphs.p3_to_p4) return false;
  for (uint32_t value_id : kModel2ProjectionInputs) {
    if (!MakeExternalInput(graphs.p2_to_p3, value_id)) return false;
  }
  for (uint32_t value_id : kModel4ProjectionInputs) {
    if (!MakeExternalInput(graphs.p3_to_p4, value_id)) return false;
  }
  return MakeExternalOutput(graphs.p2_to_p3, kModel4Cv1Output) &&
         MakeExternalOutput(graphs.stem_to_p2, kModel2Cv1Output);
}

bool RunProjectionOptimization(const ProjectionGraphs& graphs, pthreadpool_t threadpool,
                               const ProjectionDimensions& dimensions,
                               const ProjectionBuffers& buffers) {
  if (!graphs.stem_to_p2 || !graphs.p2_to_p3 || !graphs.p3_to_p4 || !threadpool ||
      !buffers.scratch || !buffers.input || !buffers.backbone_p3 || !buffers.p4_entry) {
    return false;
  }
  if (xnn_reshape_runtime(graphs.stem_to_p2) != xnn_status_success ||
      xnn_reshape_runtime(graphs.p2_to_p3) != xnn_status_success ||
      xnn_reshape_runtime(graphs.p3_to_p4) != xnn_status_success) {
    return false;
  }

  // P2 is dead before P3 projection begins, so both projections reuse the same
  // scratch base with different half extents.
  uint16_t* p2_left = buffers.scratch;
  uint16_t* p2_right = buffers.scratch + dimensions.p2_half_elements;
  uint16_t* p3_left = buffers.scratch;
  uint16_t* p3_right = buffers.scratch + dimensions.p3_half_elements;
  const xnn_external_value stem_bindings[] = {
      {StemToP2Bindings::kInput.value, buffers.input},
      {StemToP2Bindings::kP2ProjectionOutput.value, p2_left},
      {kModel2Cv1Output, p2_left},
  };
  const xnn_external_value p3_bindings[] = {
      {P2ToP3Bindings::kProjectionRightInput.value, p2_right},
      {P2ToP3Bindings::kProjectionLeftInput.value, p2_left},
      {kModel2ProjectionInputs[0], p2_right},
      {kModel2ProjectionInputs[1], p2_left},
      {P2ToP3Bindings::kP3Output.value, buffers.backbone_p3},
      {kModel4Cv1Output, p3_left},
  };
  const xnn_external_value p4_bindings[] = {
      {P3ToP4Bindings::kProjectionRightInput.value, p3_right},
      {P3ToP4Bindings::kProjectionLeftInput.value, p3_left},
      {kModel4ProjectionInputs[0], p3_right},
      {kModel4ProjectionInputs[1], p3_left},
      {P3ToP4Bindings::kP3SkipInput.value, buffers.backbone_p3},
      {P3ToP4Bindings::kP4EntryOutput.value, buffers.p4_entry},
  };
  if (xnn_setup_runtime_v2(graphs.stem_to_p2, std::size(stem_bindings), stem_bindings) !=
          xnn_status_success ||
      xnn_setup_runtime_v2(graphs.p2_to_p3, std::size(p3_bindings), p3_bindings) !=
          xnn_status_success ||
      xnn_setup_runtime_v2(graphs.p3_to_p4, std::size(p4_bindings), p4_bindings) !=
          xnn_status_success) {
    return false;
  }
  return RunOperators(graphs.stem_to_p2, 0, kModel2Cv1Operation) &&
         RunDualProjection(graphs.stem_to_p2, threadpool, kModel2Cv1Operation, dimensions.p2_pixels,
                           kStemProjectionChannels, p2_left, p2_right) &&
         RunOperators(graphs.p2_to_p3, kModel2BlockFirstOperation, kModel4Cv1Operation) &&
         RunDualProjection(graphs.p2_to_p3, threadpool, kModel4Cv1Operation, dimensions.p3_pixels,
                           kP3ProjectionChannels, p3_left, p3_right) &&
         RunOperators(graphs.p3_to_p4, kModel4BlockFirstOperation, kModel4BlockOperationEnd);
}

#include "fixed_graph.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <limits>

#include "model_contract.h"

namespace {

// The YOLO26n topology is fixed, but model resolution and class count still
// determine several tensor dimensions. This is the complete shape vocabulary
// needed by that topology; it is deliberately not a general graph IR.
enum class DimensionKind : uint8_t {
  kConstant,
  kResolutionDivided,
  kResolutionDividedSquared,
  kClassCount,
  kDetectionChannels,
  kClassificationWidth,
  kDetectionPositions,
};

struct Dimension {
  DimensionKind kind;
  uint32_t argument = 0;
};

constexpr Dimension Constant(uint32_t value) { return {DimensionKind::kConstant, value}; }
constexpr Dimension ResolutionDivided(uint32_t divisor) {
  return {DimensionKind::kResolutionDivided, divisor};
}
constexpr Dimension ResolutionDividedSquared(uint32_t divisor) {
  return {DimensionKind::kResolutionDividedSquared, divisor};
}
constexpr Dimension ClassCount() { return {DimensionKind::kClassCount}; }
constexpr Dimension DetectionChannels() { return {DimensionKind::kDetectionChannels}; }
constexpr Dimension ClassificationWidth() { return {DimensionKind::kClassificationWidth}; }
constexpr Dimension DetectionPositions() { return {DimensionKind::kDetectionPositions}; }

enum class ValueType : uint8_t { kFp16, kFp32 };

struct Padding {
  Dimension top, right, bottom, left;
};
struct Kernel {
  Dimension height, width;
};
struct Stride {
  Dimension height, width;
};
struct Dilation {
  Dimension height, width;
};
struct Groups {
  Dimension count, input_channels, output_channels;
};

bool Failed(const char* operation, GraphId graph, size_t item) {
  std::fprintf(stderr, "native graph construction failed: %s graph=%zu item=%zu\n", operation,
               GraphIndex(graph), item);
  return false;
}

struct GraphValue {
  uint32_t id = XNN_INVALID_VALUE_ID;
};

// Each Build* function below uses this helper to construct one of the model's
// nineteen XNNPACK graphs. Begin creates an empty graph, the Define* methods add
// its tensors, methods such as Convolution add operations, and Finish compiles
// the completed graph into the runtime stored in GraphContext.
//
// XNNPACK assigns every tensor a numeric ID. External inputs and outputs use
// the IDs reserved when the graph is created, while XNNPACK chooses IDs for
// intermediate tensors. GraphValue stores the ID returned by XNNPACK so later
// operations can refer to the correct tensor without relying on declaration
// position. Declaration order remains explicit because it affects XNNPACK's
// workspace planning and runtime performance.
class FixedGraphBuilder {
 public:
  FixedGraphBuilder(const GraphContext& context, GraphId id, uint32_t external_count)
      : context_(context), id_(id), external_count_(external_count) {}
  FixedGraphBuilder(const FixedGraphBuilder&) = delete;
  FixedGraphBuilder& operator=(const FixedGraphBuilder&) = delete;
  ~FixedGraphBuilder() {
    if (subgraph_) xnn_delete_subgraph(subgraph_);
  }

  bool Begin() {
    return xnn_create_subgraph(external_count_, 0, &subgraph_) == xnn_status_success ||
           Failed("xnn_create_subgraph", id_, 0);
  }
  bool DefinitionsValid() const { return definitions_valid_; }
  GraphValue DefineInput(ValueType type, std::initializer_list<Dimension> shape,
                         ExternalValueId id) {
    return DefineValue(type, shape, nullptr, id.value, XNN_VALUE_FLAG_EXTERNAL_INPUT);
  }
  GraphValue DefineOutput(ValueType type, std::initializer_list<Dimension> shape,
                          ExternalValueId id) {
    return DefineValue(type, shape, nullptr, id.value, XNN_VALUE_FLAG_EXTERNAL_OUTPUT);
  }
  GraphValue DefineIntermediate(ValueType type, std::initializer_list<Dimension> shape) {
    return DefineValue(type, shape, nullptr, XNN_INVALID_VALUE_ID, 0);
  }
  template <typename Slot>
  GraphValue DefineWeight(ValueType type, std::initializer_list<Dimension> shape, Slot slot) {
    // Exported learned tensors are FP32. Requiring the exact resolved byte size
    // prevents a valid slot from being attached to the wrong fixed-graph value.
    std::array<size_t, 4> resolved{};
    size_t elements = 0;
    if (!ResolveShape(shape, &resolved, &elements) || type != ValueType::kFp32) {
      definitions_valid_ = false;
      return {};
    }
    if (elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
      definitions_valid_ = false;
      return {};
    }
    const void* data = context_.model->Slot(static_cast<uint32_t>(slot), elements * sizeof(float));
    if (!data) {
      definitions_valid_ = false;
      Failed("model weight slot", id_, value_count_);
      return {};
    }
    return DefineResolvedValue(type, shape.size(), resolved, data, XNN_INVALID_VALUE_ID, 0);
  }

  bool Convolution(GraphValue input, GraphValue filter, GraphValue bias, GraphValue output,
                   Padding padding, Kernel kernel, Stride stride, Dilation dilation, Groups groups,
                   uint32_t flags) {
    return Operation(
        "xnn_define_convolution_2d",
        xnn_define_convolution_2d(
            subgraph_, Evaluate32(padding.top), Evaluate32(padding.right),
            Evaluate32(padding.bottom), Evaluate32(padding.left), Evaluate32(kernel.height),
            Evaluate32(kernel.width), Evaluate32(stride.height), Evaluate32(stride.width),
            Evaluate32(dilation.height), Evaluate32(dilation.width), Evaluate32(groups.count),
            Evaluate32(groups.input_channels), Evaluate32(groups.output_channels), -INFINITY,
            INFINITY, Value(input), Value(filter), Value(bias), Value(output), flags));
  }
  bool DepthwiseConvolution(GraphValue input, GraphValue filter, GraphValue bias, GraphValue output,
                            Padding padding, Kernel kernel, Stride stride, Dilation dilation,
                            Groups groups, uint32_t flags) {
    const uint32_t input_channels = Evaluate32(groups.input_channels);
    const uint32_t output_channels = Evaluate32(groups.output_channels);
    if (input_channels == 0 || output_channels % input_channels != 0) return false;
    // XNNPACK accepts a depth multiplier here, whereas Groups records the
    // total input and output channel counts used by the model definition.
    return Operation(
        "xnn_define_depthwise_convolution_2d",
        xnn_define_depthwise_convolution_2d(
            subgraph_, Evaluate32(padding.top), Evaluate32(padding.right),
            Evaluate32(padding.bottom), Evaluate32(padding.left), Evaluate32(kernel.height),
            Evaluate32(kernel.width), Evaluate32(stride.height), Evaluate32(stride.width),
            Evaluate32(dilation.height), Evaluate32(dilation.width),
            output_channels / input_channels, Evaluate32(groups.count), -INFINITY, INFINITY,
            Value(input), Value(filter), Value(bias), Value(output), flags));
  }
  bool Silu(GraphValue input, GraphValue output, uint32_t flags) {
    return Unary("xnn_define_silu", xnn_unary_silu, input, output, flags);
  }
  bool Sigmoid(GraphValue input, GraphValue output, uint32_t flags) {
    return Unary("xnn_define_sigmoid", xnn_unary_sigmoid, input, output, flags);
  }
  bool Add(GraphValue left, GraphValue right, GraphValue output, uint32_t flags) {
    return Binary("xnn_define_add", xnn_binary_add, left, right, output, flags);
  }
  bool Subtract(GraphValue left, GraphValue right, GraphValue output, uint32_t flags) {
    return Binary("xnn_define_subtract", xnn_binary_subtract, left, right, output, flags);
  }
  bool Multiply(GraphValue left, GraphValue right, GraphValue output, uint32_t flags) {
    return Binary("xnn_define_multiply", xnn_binary_multiply, left, right, output, flags);
  }
  bool Divide(GraphValue left, GraphValue right, GraphValue output, uint32_t flags) {
    return Binary("xnn_define_divide", xnn_binary_divide, left, right, output, flags);
  }
  bool Concatenate(std::initializer_list<GraphValue> inputs, GraphValue output, Dimension axis,
                   uint32_t flags) {
    std::array<uint32_t, 5> input_ids{};
    if (inputs.size() < 2 || inputs.size() > input_ids.size()) return false;
    size_t index = 0;
    for (GraphValue input : inputs) input_ids[index++] = Value(input);
    return Operation("xnn_define_concatenate",
                     xnn_define_concatenate(subgraph_, static_cast<int32_t>(Evaluate32(axis)),
                                            inputs.size(), input_ids.data(), Value(output), flags));
  }
  bool Transpose(GraphValue input, GraphValue output, std::initializer_list<Dimension> permutation,
                 uint32_t flags) {
    return StaticShapeOperation("xnn_define_static_transpose", true, input, output, permutation,
                                flags);
  }
  bool Reshape(GraphValue input, GraphValue output, std::initializer_list<Dimension> shape,
               uint32_t flags) {
    return StaticShapeOperation("xnn_define_static_reshape", false, input, output, shape, flags);
  }
  bool MaxPool(GraphValue input, GraphValue output, Padding padding, Kernel kernel, Stride stride,
               Dilation dilation, uint32_t flags) {
    return Operation(
        "xnn_define_max_pooling_2d",
        xnn_define_max_pooling_2d(subgraph_, Evaluate32(padding.top), Evaluate32(padding.right),
                                  Evaluate32(padding.bottom), Evaluate32(padding.left),
                                  Evaluate32(kernel.height), Evaluate32(kernel.width),
                                  Evaluate32(stride.height), Evaluate32(stride.width),
                                  Evaluate32(dilation.height), Evaluate32(dilation.width),
                                  -INFINITY, INFINITY, Value(input), Value(output), flags));
  }
  bool BatchMatrixMultiply(GraphValue left, GraphValue right, GraphValue output, uint32_t flags) {
    return Operation("xnn_define_batch_matrix_multiply",
                     xnn_define_batch_matrix_multiply(subgraph_, Value(left), Value(right),
                                                      Value(output), flags));
  }
  bool Softmax(GraphValue input, GraphValue output, uint32_t flags) {
    return Operation("xnn_define_softmax",
                     xnn_define_softmax(subgraph_, Value(input), Value(output), flags));
  }

  bool Finish() {
    // Runtime creation materializes the operators that retain the shared
    // workspace and threadpool. The construction-only subgraph can then go.
    if (!definitions_valid_) return false;
    const xnn_status status =
        xnn_create_runtime_v4(subgraph_, nullptr, context_.workspace, context_.threadpool, 0,
                              &(*context_.runtimes)[GraphIndex(id_)]);
    if (status != xnn_status_success) return Failed("xnn_create_runtime_v4", id_, operation_);
    xnn_delete_subgraph(subgraph_);
    subgraph_ = nullptr;
    return true;
  }

 private:
  size_t Evaluate(Dimension dimension) const {
    switch (dimension.kind) {
      case DimensionKind::kConstant:
        return dimension.argument;
      case DimensionKind::kResolutionDivided:
        return context_.model->resolution() / dimension.argument;
      case DimensionKind::kResolutionDividedSquared: {
        const size_t side = context_.model->resolution() / dimension.argument;
        return side * side;
      }
      case DimensionKind::kClassCount:
        return context_.model->class_count();
      case DimensionKind::kDetectionChannels:
        return context_.model->class_count() + yolo26::kBoxCoordinateChannels;
      case DimensionKind::kClassificationWidth:
        return context_.model->classification_width();
      case DimensionKind::kDetectionPositions:
        return yolo26::DetectionPositions(context_.model->resolution());
    }
    return 0;
  }
  uint32_t Evaluate32(Dimension dimension) const {
    return static_cast<uint32_t>(Evaluate(dimension));
  }
  bool ResolveShape(std::initializer_list<Dimension> dimensions, std::array<size_t, 4>* resolved,
                    size_t* elements) const {
    if (dimensions.size() == 0 || dimensions.size() > resolved->size()) return false;
    *elements = 1;
    size_t axis = 0;
    for (Dimension dimension : dimensions) {
      const size_t extent = Evaluate(dimension);
      if (extent == 0 || *elements > std::numeric_limits<size_t>::max() / extent) return false;
      (*resolved)[axis++] = extent;
      *elements *= extent;
    }
    return true;
  }
  GraphValue DefineValue(ValueType type, std::initializer_list<Dimension> dimensions,
                         const void* data, uint32_t external_id, uint32_t flags) {
    std::array<size_t, 4> resolved{};
    size_t elements = 0;
    if (!ResolveShape(dimensions, &resolved, &elements)) {
      definitions_valid_ = false;
      return {};
    }
    return DefineResolvedValue(type, dimensions.size(), resolved, data, external_id, flags);
  }
  GraphValue DefineResolvedValue(ValueType type, size_t rank,
                                 const std::array<size_t, 4>& dimensions, const void* data,
                                 uint32_t external_id, uint32_t flags) {
    uint32_t value = XNN_INVALID_VALUE_ID;
    const xnn_datatype datatype = type == ValueType::kFp16 ? xnn_datatype_fp16 : xnn_datatype_fp32;
    if (xnn_define_tensor_value(subgraph_, datatype, rank, dimensions.data(), data, external_id,
                                flags, &value) != xnn_status_success) {
      definitions_valid_ = false;
      Failed("xnn_define_tensor_value", id_, value_count_);
      return {};
    }
    ++value_count_;
    return {value};
  }
  uint32_t Value(GraphValue value) const { return value.id; }
  bool Operation(const char* name, xnn_status status) {
    const size_t operation = operation_++;
    return status == xnn_status_success || Failed(name, id_, operation);
  }
  bool Unary(const char* name, xnn_unary_operator operation, GraphValue input, GraphValue output,
             uint32_t flags) {
    return Operation(
        name, xnn_define_unary(subgraph_, operation, nullptr, Value(input), Value(output), flags));
  }
  bool Binary(const char* name, xnn_binary_operator operation, GraphValue left, GraphValue right,
              GraphValue output, uint32_t flags) {
    const xnn_binary_params parameters{-INFINITY, INFINITY};
    return Operation(name, xnn_define_binary(subgraph_, operation, &parameters, Value(left),
                                             Value(right), Value(output), flags));
  }
  bool StaticShapeOperation(const char* name, bool transpose, GraphValue input, GraphValue output,
                            std::initializer_list<Dimension> dimensions, uint32_t flags) {
    std::array<size_t, 4> resolved{};
    if (dimensions.size() == 0 || dimensions.size() > resolved.size()) return false;
    size_t index = 0;
    for (Dimension dimension : dimensions) resolved[index++] = Evaluate(dimension);
    const xnn_status status =
        transpose ? xnn_define_static_transpose(subgraph_, dimensions.size(), resolved.data(),
                                                Value(input), Value(output), flags)
                  : xnn_define_static_reshape(subgraph_, dimensions.size(), resolved.data(),
                                              Value(input), Value(output), flags);
    return Operation(name, status);
  }

  const GraphContext& context_;
  GraphId id_;
  uint32_t external_count_;
  xnn_subgraph_t subgraph_ = nullptr;
  size_t value_count_ = 0;
  size_t operation_ = 0;
  bool definitions_valid_ = true;
};

// These are intentionally private textual fragments: they use the builder and
// shape vocabulary in this anonymous namespace, and no translation unit may
// compile or include them independently. Keeping one fragment per model region
// makes the fixed topology navigable without creating another graph format or
// generated-source path. Each Build* function declares one of the 19 runtime
// partitions. Value declarations retain the complete inventory benchmarked on
// Raspberry Pi 5—including entries not consumed by a node—because declaration
// order affects XNNPACK workspace planning and steady-state latency.
// clang-format off
#include "fixed_weight_slots.inc"
#include "fixed_graph_backbone.inc"
#include "fixed_graph_neck.inc"
#include "fixed_graph_detection.inc"
#include "fixed_graph_projection.inc"
// clang-format on

bool BuildAllFixedGraphs(const GraphContext& context) {
  // Keep construction explicit and in GraphId order. Runtime execution follows
  // the separate fixed schedule, which begins with the projection partitions.
  return BuildBackboneP4ToP5(context) && BuildBackboneP5ToAttention(context) &&
         BuildBackboneAttentionQkv(context) && BuildBackboneAttentionScale(context) &&
         BuildBackboneAttentionMatrix(context) && BuildBackboneAttentionOutput(context) &&
         BuildNeckP4Fusion(context) && BuildNeckP4Block(context) && BuildNeckP3Fusion(context) &&
         BuildDetectionP3Path(context) && BuildDetectionP4Path(context) &&
         BuildDetectionP5Qkv(context) && BuildDetectionP5AttentionScale(context) &&
         BuildDetectionP5AttentionMatrix(context) && BuildDetectionP5AndBoxes(context) &&
         BuildClassificationAndDecode(context) && BuildStemToP2(context) && BuildP2ToP3(context) &&
         BuildP3ToP4(context);
}

}  // namespace

bool BuildFixedGraphs(const GraphContext& context) { return BuildAllFixedGraphs(context); }

// Early backbone partitions used by the split-projection optimization. Each
// partition stops at an output-channel split that projection_optimization.cpp
// executes directly through XNNPACK's packed GEMM operator. Reserved values
// preserve XNNPACK's performance-sensitive declaration inventory.

// Normalizes the input layout, runs models 0 and 1, and prepares the two
// model.2 input projections at P2.
bool BuildStemToP2(const GraphContext& context) {
  FixedGraphBuilder graph(context, GraphId::kStemToP2, StemToP2Bindings::kCount);
  if (!graph.Begin()) return false;
  [[maybe_unused]] const GraphValue reserved_value_0 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(3), ResolutionDivided(1), ResolutionDivided(1)});
  const GraphValue input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), ResolutionDivided(1), ResolutionDivided(1), Constant(3)},
      StemToP2Bindings::kInput);
  const GraphValue model0_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(16), Constant(3), Constant(3), Constant(3)},
                         WeightSlot::kModel0ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_3 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(2), ResolutionDivided(2), Constant(16)});
  const GraphValue model0_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(16)}, WeightSlot::kModel0ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_5 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(2), ResolutionDivided(2), Constant(16)});
  const GraphValue model0_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(2), ResolutionDivided(2), Constant(16)});
  const GraphValue model1_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(32), Constant(3), Constant(3), Constant(16)},
                         WeightSlot::kModel1ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_8 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(4), ResolutionDivided(4), Constant(32)});
  const GraphValue model1_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(32)}, WeightSlot::kModel1ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_10 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(4), ResolutionDivided(4), Constant(32)});
  const GraphValue model1_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(4), ResolutionDivided(4), Constant(32)});
  const GraphValue model2_cv1_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(32), Constant(1), Constant(1), Constant(32)},
                         WeightSlot::kModel2Cv1ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_13 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(4), ResolutionDivided(4), Constant(32)});
  const GraphValue model2_cv1_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(32)}, WeightSlot::kModel2Cv1ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_15 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(4), ResolutionDivided(4), Constant(32)});
  const GraphValue model2_cv1_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(4), ResolutionDivided(4), Constant(32)});
  const GraphValue p2_projection_output = graph.DefineOutput(
      ValueType::kFp16, {Constant(1), Constant(32), ResolutionDivided(4), ResolutionDivided(4)},
      StemToP2Bindings::kP2ProjectionOutput);
  if (!graph.DefinitionsValid()) return false;
  if (!graph.Convolution(input, model0_conv_weight, model0_conv_bias, model0_conv_output,
                         Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
                         Kernel{Constant(3), Constant(3)}, Stride{Constant(2), Constant(2)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), Constant(3), Constant(16)},
                         XNN_FLAG_YOLO26_IGEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Convolution(
          model0_conv_output, model1_conv_weight, model1_conv_bias, model1_conv_output,
          Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
          Kernel{Constant(3), Constant(3)}, Stride{Constant(2), Constant(2)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(16), Constant(32)},
          XNN_FLAG_YOLO26_IGEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Convolution(
          model1_conv_output, model2_cv1_conv_weight, model2_cv1_conv_bias, model2_cv1_conv_output,
          Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(32), Constant(32)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Transpose(model2_cv1_conv_output, p2_projection_output,
                       {Constant(0), Constant(3), Constant(1), Constant(2)}, 0))
    return false;
  return graph.Finish();
}

// Completes model.2, downsamples through model.3, and prepares the two model.4
// input projections at P3.
bool BuildP2ToP3(const GraphContext& context) {
  FixedGraphBuilder graph(context, GraphId::kP2ToP3, P2ToP3Bindings::kCount);
  if (!graph.Begin()) return false;
  const GraphValue projection_right_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), Constant(16), ResolutionDivided(4), ResolutionDivided(4)},
      P2ToP3Bindings::kProjectionRightInput);
  const GraphValue transpose_1_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(4), ResolutionDivided(4), Constant(16)});
  const GraphValue projection_left_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), Constant(16), ResolutionDivided(4), ResolutionDivided(4)},
      P2ToP3Bindings::kProjectionLeftInput);
  const GraphValue transpose_2_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(4), ResolutionDivided(4), Constant(16)});
  const GraphValue model2_m0_cv1_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(8), Constant(3), Constant(3), Constant(16)},
                         WeightSlot::kModel2M0Cv1ConvWeight);
  const GraphValue model2_m0_cv1_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(4), ResolutionDivided(4), Constant(8)});
  const GraphValue model2_m0_cv1_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(8)}, WeightSlot::kModel2M0Cv1ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_7 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(4), ResolutionDivided(4), Constant(8)});
  const GraphValue model2_m0_cv1_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(4), ResolutionDivided(4), Constant(8)});
  const GraphValue model2_m0_cv2_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(16), Constant(3), Constant(3), Constant(8)},
                         WeightSlot::kModel2M0Cv2ConvWeight);
  const GraphValue model2_m0_cv2_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(4), ResolutionDivided(4), Constant(16)});
  const GraphValue model2_m0_cv2_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(16)}, WeightSlot::kModel2M0Cv2ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_12 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(4), ResolutionDivided(4), Constant(16)});
  const GraphValue model2_m0_cv2_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(4), ResolutionDivided(4), Constant(16)});
  const GraphValue residual_sum_1 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(4), ResolutionDivided(4), Constant(16)});
  const GraphValue concatenation_1 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(4), ResolutionDivided(4), Constant(48)});
  const GraphValue model2_cv2_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(64), Constant(1), Constant(1), Constant(48)},
                         WeightSlot::kModel2Cv2ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_17 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(4), ResolutionDivided(4), Constant(64)});
  const GraphValue model2_cv2_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(64)}, WeightSlot::kModel2Cv2ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_19 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(4), ResolutionDivided(4), Constant(64)});
  const GraphValue model2_cv2_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(4), ResolutionDivided(4), Constant(64)});
  const GraphValue model3_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(64), Constant(3), Constant(3), Constant(64)},
                         WeightSlot::kModel3ConvWeight);
  const GraphValue model3_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(64)});
  const GraphValue model3_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(64)}, WeightSlot::kModel3ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_24 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(64)});
  const GraphValue model3_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(64)});
  const GraphValue model4_cv1_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(64), Constant(1), Constant(1), Constant(64)},
                         WeightSlot::kModel4Cv1ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_27 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(64)});
  const GraphValue model4_cv1_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(64)}, WeightSlot::kModel4Cv1ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_29 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(64)});
  const GraphValue model4_cv1_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(64)});
  const GraphValue p3_output = graph.DefineOutput(
      ValueType::kFp16, {Constant(1), Constant(64), ResolutionDivided(8), ResolutionDivided(8)},
      P2ToP3Bindings::kP3Output);
  if (!graph.DefinitionsValid()) return false;
  if (!graph.Transpose(projection_right_input, transpose_1_output,
                       {Constant(0), Constant(2), Constant(3), Constant(1)}, 0))
    return false;
  if (!graph.Transpose(projection_left_input, transpose_2_output,
                       {Constant(0), Constant(2), Constant(3), Constant(1)}, 0))
    return false;
  if (!graph.Convolution(
          transpose_1_output, model2_m0_cv1_conv_weight, model2_m0_cv1_conv_bias,
          model2_m0_cv1_conv_output, Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
          Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(16), Constant(8)}, 0))
    return false;
  if (!graph.Silu(model2_m0_cv1_conv_output, model2_m0_cv1_conv_activation, 0)) return false;
  if (!graph.Convolution(
          model2_m0_cv1_conv_activation, model2_m0_cv2_conv_weight, model2_m0_cv2_conv_bias,
          model2_m0_cv2_conv_output, Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
          Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(8), Constant(16)}, 0))
    return false;
  if (!graph.Silu(model2_m0_cv2_conv_output, model2_m0_cv2_conv_activation, 0)) return false;
  if (!graph.Add(transpose_1_output, model2_m0_cv2_conv_activation, residual_sum_1, 0))
    return false;
  if (!graph.Concatenate({transpose_2_output, transpose_1_output, residual_sum_1}, concatenation_1,
                         Constant(3), XNN_FLAG_YOLO26_FUSED_C2F_CONCATENATE))
    return false;
  if (!graph.Convolution(
          concatenation_1, model2_cv2_conv_weight, model2_cv2_conv_bias, model2_cv2_conv_output,
          Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(48), Constant(64)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Convolution(
          model2_cv2_conv_output, model3_conv_weight, model3_conv_bias, model3_conv_output,
          Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
          Kernel{Constant(3), Constant(3)}, Stride{Constant(2), Constant(2)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(64), Constant(64)}, 0))
    return false;
  if (!graph.Silu(model3_conv_output, model3_conv_activation, 0)) return false;
  if (!graph.Convolution(
          model3_conv_activation, model4_cv1_conv_weight, model4_cv1_conv_bias,
          model4_cv1_conv_output, Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(64), Constant(64)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Transpose(model4_cv1_conv_output, p3_output,
                       {Constant(0), Constant(3), Constant(1), Constant(2)}, 0))
    return false;
  return graph.Finish();
}

// Completes model.4, downsamples through model.5, and prepares the two model.6
// input projections at P4.
bool BuildP3ToP4(const GraphContext& context) {
  FixedGraphBuilder graph(context, GraphId::kP3ToP4, P3ToP4Bindings::kCount);
  if (!graph.Begin()) return false;
  const GraphValue projection_right_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), Constant(32), ResolutionDivided(8), ResolutionDivided(8)},
      P3ToP4Bindings::kProjectionRightInput);
  const GraphValue transpose_1_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(32)});
  const GraphValue projection_left_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), Constant(32), ResolutionDivided(8), ResolutionDivided(8)},
      P3ToP4Bindings::kProjectionLeftInput);
  const GraphValue transpose_2_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(32)});
  const GraphValue model4_m0_cv1_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(16), Constant(3), Constant(3), Constant(32)},
                         WeightSlot::kModel4M0Cv1ConvWeight);
  const GraphValue model4_m0_cv1_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(16)});
  const GraphValue model4_m0_cv1_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(16)}, WeightSlot::kModel4M0Cv1ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_7 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(16)});
  const GraphValue model4_m0_cv1_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(16)});
  const GraphValue model4_m0_cv2_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(32), Constant(3), Constant(3), Constant(16)},
                         WeightSlot::kModel4M0Cv2ConvWeight);
  const GraphValue model4_m0_cv2_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(32)});
  const GraphValue model4_m0_cv2_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(32)}, WeightSlot::kModel4M0Cv2ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_12 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(32)});
  const GraphValue model4_m0_cv2_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(32)});
  const GraphValue residual_sum_1 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(32)});
  const GraphValue concatenation_1 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(96)});
  const GraphValue model4_cv2_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(128), Constant(1), Constant(1), Constant(96)},
                         WeightSlot::kModel4Cv2ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_17 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(128)});
  const GraphValue model4_cv2_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(128)}, WeightSlot::kModel4Cv2ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_19 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(128)});
  const GraphValue p3_skip_input = graph.DefineOutput(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(128)},
      P3ToP4Bindings::kP3SkipInput);
  [[maybe_unused]] const GraphValue reserved_value_21 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(128), ResolutionDivided(8), ResolutionDivided(8)});
  const GraphValue model5_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(128), Constant(3), Constant(3), Constant(128)},
                         WeightSlot::kModel5ConvWeight);
  const GraphValue model5_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)});
  const GraphValue model5_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(128)}, WeightSlot::kModel5ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_25 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)});
  const GraphValue model5_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)});
  const GraphValue model6_cv1_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(128), Constant(1), Constant(1), Constant(128)},
                         WeightSlot::kModel6Cv1ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_28 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)});
  const GraphValue model6_cv1_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(128)}, WeightSlot::kModel6Cv1ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_30 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)});
  const GraphValue model6_cv1_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)});
  const GraphValue p4_entry_output = graph.DefineOutput(
      ValueType::kFp16, {Constant(1), Constant(128), ResolutionDivided(16), ResolutionDivided(16)},
      P3ToP4Bindings::kP4EntryOutput);
  if (!graph.DefinitionsValid()) return false;
  if (!graph.Transpose(projection_right_input, transpose_1_output,
                       {Constant(0), Constant(2), Constant(3), Constant(1)}, 0))
    return false;
  if (!graph.Transpose(projection_left_input, transpose_2_output,
                       {Constant(0), Constant(2), Constant(3), Constant(1)}, 0))
    return false;
  if (!graph.Convolution(
          transpose_1_output, model4_m0_cv1_conv_weight, model4_m0_cv1_conv_bias,
          model4_m0_cv1_conv_output, Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
          Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(32), Constant(16)}, 0))
    return false;
  if (!graph.Silu(model4_m0_cv1_conv_output, model4_m0_cv1_conv_activation, 0)) return false;
  if (!graph.Convolution(
          model4_m0_cv1_conv_activation, model4_m0_cv2_conv_weight, model4_m0_cv2_conv_bias,
          model4_m0_cv2_conv_output, Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
          Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(16), Constant(32)}, 0))
    return false;
  if (!graph.Silu(model4_m0_cv2_conv_output, model4_m0_cv2_conv_activation, 0)) return false;
  if (!graph.Add(transpose_1_output, model4_m0_cv2_conv_activation, residual_sum_1, 0))
    return false;
  if (!graph.Concatenate({transpose_2_output, transpose_1_output, residual_sum_1}, concatenation_1,
                         Constant(3), 0))
    return false;
  if (!graph.Convolution(concatenation_1, model4_cv2_conv_weight, model4_cv2_conv_bias,
                         p3_skip_input, Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
                         Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), Constant(96), Constant(128)},
                         XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Convolution(p3_skip_input, model5_conv_weight, model5_conv_bias, model5_conv_output,
                         Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
                         Kernel{Constant(3), Constant(3)}, Stride{Constant(2), Constant(2)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), Constant(128), Constant(128)}, 0))
    return false;
  if (!graph.Silu(model5_conv_output, model5_conv_activation, 0)) return false;
  if (!graph.Convolution(
          model5_conv_activation, model6_cv1_conv_weight, model6_cv1_conv_bias,
          model6_cv1_conv_output, Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(128), Constant(128)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Transpose(model6_cv1_conv_output, p4_entry_output,
                       {Constant(0), Constant(3), Constant(1), Constant(2)}, 0))
    return false;
  return graph.Finish();
}

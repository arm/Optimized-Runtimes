// Fixed backbone partitions from the P4 feature level through the P5 attention
// block. Reserved values reproduce XNNPACK's performance-sensitive declaration
// inventory but intentionally do not participate in data flow.

// Completes the model.6 P4 block, downsamples through model.7, and begins the
// model.8 P5 block. The schedule supplies the two halves of the P4 entry.
bool BuildBackboneP4ToP5(const GraphContext& context) {
  FixedGraphBuilder graph(context, GraphId::kBackboneP4ToP5, BackboneP4ToP5Bindings::kCount);
  if (!graph.Begin()) return false;
  const GraphValue main_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), Constant(64), ResolutionDivided(16), ResolutionDivided(16)},
      BackboneP4ToP5Bindings::kMainInput);
  const GraphValue transpose_1_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(64)});
  const GraphValue skip_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), Constant(64), ResolutionDivided(16), ResolutionDivided(16)},
      BackboneP4ToP5Bindings::kSkipInput);
  const GraphValue transpose_2_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(64)});
  const GraphValue model6_m0_cv1_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(32), Constant(1), Constant(1), Constant(64)},
                         WeightSlot::kModel6M0Cv1ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_5 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model6_m0_cv1_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(32)}, WeightSlot::kModel6M0Cv1ConvBias);
  const GraphValue model6_m0_cv2_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(32), Constant(1), Constant(1), Constant(64)},
                         WeightSlot::kModel6M0Cv2ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_8 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model6_m0_cv2_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(32)}, WeightSlot::kModel6M0Cv2ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_10 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  [[maybe_unused]] const GraphValue reserved_value_11 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model6_m0_cv1_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model6_m0_cv2_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model6_m0_m0_cv1_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(32), Constant(3), Constant(3), Constant(32)},
                         WeightSlot::kModel6M0M0Cv1ConvWeight);
  const GraphValue model6_m0_m0_cv1_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model6_m0_m0_cv1_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(32)}, WeightSlot::kModel6M0M0Cv1ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_17 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model6_m0_m0_cv1_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model6_m0_m0_cv2_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(32), Constant(3), Constant(3), Constant(32)},
                         WeightSlot::kModel6M0M0Cv2ConvWeight);
  const GraphValue model6_m0_m0_cv2_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model6_m0_m0_cv2_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(32)}, WeightSlot::kModel6M0M0Cv2ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_22 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model6_m0_m0_cv2_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue residual_sum_1 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model6_m0_m1_cv1_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(32), Constant(3), Constant(3), Constant(32)},
                         WeightSlot::kModel6M0M1Cv1ConvWeight);
  const GraphValue model6_m0_m1_cv1_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model6_m0_m1_cv1_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(32)}, WeightSlot::kModel6M0M1Cv1ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_28 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model6_m0_m1_cv1_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model6_m0_m1_cv2_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(32), Constant(3), Constant(3), Constant(32)},
                         WeightSlot::kModel6M0M1Cv2ConvWeight);
  const GraphValue model6_m0_m1_cv2_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model6_m0_m1_cv2_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(32)}, WeightSlot::kModel6M0M1Cv2ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_33 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model6_m0_m1_cv2_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue residual_sum_2 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue concatenation_1 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(64)});
  const GraphValue model6_m0_cv3_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(64), Constant(1), Constant(1), Constant(64)},
                         WeightSlot::kModel6M0Cv3ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_38 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(64)});
  const GraphValue model6_m0_cv3_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(64)}, WeightSlot::kModel6M0Cv3ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_40 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(64)});
  const GraphValue model6_m0_cv3_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(64)});
  const GraphValue concatenation_2 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(192)});
  const GraphValue model6_cv2_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(128), Constant(1), Constant(1), Constant(192)},
                         WeightSlot::kModel6Cv2ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_44 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)});
  const GraphValue model6_cv2_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(128)}, WeightSlot::kModel6Cv2ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_46 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)});
  const GraphValue model6_cv2_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)});
  const GraphValue p4_output = graph.DefineOutput(
      ValueType::kFp16, {Constant(1), Constant(128), ResolutionDivided(16), ResolutionDivided(16)},
      BackboneP4ToP5Bindings::kP4Output);
  const GraphValue model7_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(256), Constant(3), Constant(3), Constant(128)},
                         WeightSlot::kModel7ConvWeight);
  const GraphValue model7_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue model7_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(256)}, WeightSlot::kModel7ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_52 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue model7_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue model8_cv1_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(256), Constant(1), Constant(1), Constant(256)},
                         WeightSlot::kModel8Cv1ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_55 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue model8_cv1_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(256)}, WeightSlot::kModel8Cv1ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_57 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue model8_cv1_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue p5_entry_output = graph.DefineOutput(
      ValueType::kFp16, {Constant(1), Constant(256), ResolutionDivided(32), ResolutionDivided(32)},
      BackboneP4ToP5Bindings::kP5EntryOutput);
  if (!graph.DefinitionsValid()) return false;
  if (!graph.Transpose(main_input, transpose_1_output,
                       {Constant(0), Constant(2), Constant(3), Constant(1)}, 0))
    return false;
  if (!graph.Transpose(skip_input, transpose_2_output,
                       {Constant(0), Constant(2), Constant(3), Constant(1)}, 0))
    return false;
  if (!graph.Convolution(
          transpose_1_output, model6_m0_cv1_conv_weight, model6_m0_cv1_conv_bias,
          model6_m0_cv1_conv_output, Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(64), Constant(32)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Convolution(
          transpose_1_output, model6_m0_cv2_conv_weight, model6_m0_cv2_conv_bias,
          model6_m0_cv2_conv_output, Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(64), Constant(32)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Convolution(
          model6_m0_cv1_conv_output, model6_m0_m0_cv1_conv_weight, model6_m0_m0_cv1_conv_bias,
          model6_m0_m0_cv1_conv_output, Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
          Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(32), Constant(32)}, 0))
    return false;
  if (!graph.Silu(model6_m0_m0_cv1_conv_output, model6_m0_m0_cv1_conv_activation, 0)) return false;
  if (!graph.Convolution(model6_m0_m0_cv1_conv_activation, model6_m0_m0_cv2_conv_weight,
                         model6_m0_m0_cv2_conv_bias, model6_m0_m0_cv2_conv_output,
                         Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
                         Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), Constant(32), Constant(32)}, 0))
    return false;
  if (!graph.Silu(model6_m0_m0_cv2_conv_output, model6_m0_m0_cv2_conv_activation, 0)) return false;
  if (!graph.Add(model6_m0_cv1_conv_output, model6_m0_m0_cv2_conv_activation, residual_sum_1, 0))
    return false;
  if (!graph.Convolution(
          residual_sum_1, model6_m0_m1_cv1_conv_weight, model6_m0_m1_cv1_conv_bias,
          model6_m0_m1_cv1_conv_output, Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
          Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(32), Constant(32)}, 0))
    return false;
  if (!graph.Silu(model6_m0_m1_cv1_conv_output, model6_m0_m1_cv1_conv_activation, 0)) return false;
  if (!graph.Convolution(model6_m0_m1_cv1_conv_activation, model6_m0_m1_cv2_conv_weight,
                         model6_m0_m1_cv2_conv_bias, model6_m0_m1_cv2_conv_output,
                         Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
                         Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), Constant(32), Constant(32)}, 0))
    return false;
  if (!graph.Silu(model6_m0_m1_cv2_conv_output, model6_m0_m1_cv2_conv_activation, 0)) return false;
  if (!graph.Add(residual_sum_1, model6_m0_m1_cv2_conv_activation, residual_sum_2, 0)) return false;
  if (!graph.Concatenate({residual_sum_2, model6_m0_cv2_conv_output}, concatenation_1, Constant(3),
                         0))
    return false;
  if (!graph.Convolution(
          concatenation_1, model6_m0_cv3_conv_weight, model6_m0_cv3_conv_bias,
          model6_m0_cv3_conv_output, Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(64), Constant(64)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Concatenate({transpose_2_output, transpose_1_output, model6_m0_cv3_conv_output},
                         concatenation_2, Constant(3), 0))
    return false;
  if (!graph.Convolution(
          concatenation_2, model6_cv2_conv_weight, model6_cv2_conv_bias, model6_cv2_conv_output,
          Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(192), Constant(128)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Transpose(model6_cv2_conv_output, p4_output,
                       {Constant(0), Constant(3), Constant(1), Constant(2)}, 0))
    return false;
  if (!graph.Convolution(
          model6_cv2_conv_output, model7_conv_weight, model7_conv_bias, model7_conv_output,
          Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
          Kernel{Constant(3), Constant(3)}, Stride{Constant(2), Constant(2)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(128), Constant(256)}, 0))
    return false;
  if (!graph.Silu(model7_conv_output, model7_conv_activation, 0)) return false;
  if (!graph.Convolution(
          model7_conv_activation, model8_cv1_conv_weight, model8_cv1_conv_bias,
          model8_cv1_conv_output, Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(256), Constant(256)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Transpose(model8_cv1_conv_output, p5_entry_output,
                       {Constant(0), Constant(3), Constant(1), Constant(2)}, 0))
    return false;
  return graph.Finish();
}

// Completes model.8 and model.10.cv1, producing the bypass and residual inputs
// consumed by the P5 attention partitions.
bool BuildBackboneP5ToAttention(const GraphContext& context) {
  FixedGraphBuilder graph(context, GraphId::kBackboneP5ToAttention,
                          BackboneP5ToAttentionBindings::kCount);
  if (!graph.Begin()) return false;
  const GraphValue main_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), Constant(128), ResolutionDivided(32), ResolutionDivided(32)},
      BackboneP5ToAttentionBindings::kMainInput);
  const GraphValue transpose_1_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue skip_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), Constant(128), ResolutionDivided(32), ResolutionDivided(32)},
      BackboneP5ToAttentionBindings::kSkipInput);
  const GraphValue transpose_2_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue model8_m0_cv1_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(64), Constant(1), Constant(1), Constant(128)},
                         WeightSlot::kModel8M0Cv1ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_5 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(64)});
  const GraphValue model8_m0_cv1_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(64)}, WeightSlot::kModel8M0Cv1ConvBias);
  const GraphValue model8_m0_cv2_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(64), Constant(1), Constant(1), Constant(128)},
                         WeightSlot::kModel8M0Cv2ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_8 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(64)});
  const GraphValue model8_m0_cv2_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(64)}, WeightSlot::kModel8M0Cv2ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_10 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(64)});
  [[maybe_unused]] const GraphValue reserved_value_11 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(64)});
  const GraphValue model8_m0_cv1_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(64)});
  const GraphValue model8_m0_cv2_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(64)});
  const GraphValue model8_m0_m0_cv1_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(64), Constant(3), Constant(3), Constant(64)},
                         WeightSlot::kModel8M0M0Cv1ConvWeight);
  const GraphValue model8_m0_m0_cv1_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(64)});
  const GraphValue model8_m0_m0_cv1_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(64)}, WeightSlot::kModel8M0M0Cv1ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_17 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(64)});
  const GraphValue model8_m0_m0_cv1_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(64)});
  const GraphValue model8_m0_m0_cv2_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(64), Constant(3), Constant(3), Constant(64)},
                         WeightSlot::kModel8M0M0Cv2ConvWeight);
  const GraphValue model8_m0_m0_cv2_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(64)});
  const GraphValue model8_m0_m0_cv2_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(64)}, WeightSlot::kModel8M0M0Cv2ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_22 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(64)});
  const GraphValue model8_m0_m0_cv2_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(64)});
  const GraphValue residual_sum_1 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(64)});
  const GraphValue model8_m0_m1_cv1_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(64), Constant(3), Constant(3), Constant(64)},
                         WeightSlot::kModel8M0M1Cv1ConvWeight);
  const GraphValue model8_m0_m1_cv1_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(64)});
  const GraphValue model8_m0_m1_cv1_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(64)}, WeightSlot::kModel8M0M1Cv1ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_28 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(64)});
  const GraphValue model8_m0_m1_cv1_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(64)});
  const GraphValue model8_m0_m1_cv2_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(64), Constant(3), Constant(3), Constant(64)},
                         WeightSlot::kModel8M0M1Cv2ConvWeight);
  const GraphValue model8_m0_m1_cv2_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(64)});
  const GraphValue model8_m0_m1_cv2_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(64)}, WeightSlot::kModel8M0M1Cv2ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_33 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(64)});
  const GraphValue model8_m0_m1_cv2_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(64)});
  const GraphValue residual_sum_2 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(64)});
  const GraphValue concatenation_1 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue model8_m0_cv3_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(128), Constant(1), Constant(1), Constant(128)},
                         WeightSlot::kModel8M0Cv3ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_38 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue model8_m0_cv3_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(128)}, WeightSlot::kModel8M0Cv3ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_40 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue model8_m0_cv3_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue concatenation_2 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(384)});
  const GraphValue model8_cv2_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(256), Constant(1), Constant(1), Constant(384)},
                         WeightSlot::kModel8Cv2ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_44 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue model8_cv2_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(256)}, WeightSlot::kModel8Cv2ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_46 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue model8_cv2_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue model9_cv1_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(128), Constant(1), Constant(1), Constant(256)},
                         WeightSlot::kModel9Cv1ConvWeight);
  const GraphValue model9_cv1_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue model9_cv1_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(128)}, WeightSlot::kModel9Cv1ConvBias);
  const GraphValue max_pool_1_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue max_pool_2_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue max_pool_3_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue concatenation_3 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(512)});
  const GraphValue model9_cv2_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(256), Constant(1), Constant(1), Constant(512)},
                         WeightSlot::kModel9Cv2ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_56 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue model9_cv2_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(256)}, WeightSlot::kModel9Cv2ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_58 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue model9_cv2_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue residual_sum_3 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue model10_cv1_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(256), Constant(1), Constant(1), Constant(256)},
                         WeightSlot::kModel10Cv1ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_62 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue model10_cv1_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(256)}, WeightSlot::kModel10Cv1ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_64 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue model10_cv1_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue attention_entry_output = graph.DefineOutput(
      ValueType::kFp16, {Constant(1), Constant(256), ResolutionDivided(32), ResolutionDivided(32)},
      BackboneP5ToAttentionBindings::kAttentionEntryOutput);
  if (!graph.DefinitionsValid()) return false;
  if (!graph.Transpose(main_input, transpose_1_output,
                       {Constant(0), Constant(2), Constant(3), Constant(1)}, 0))
    return false;
  if (!graph.Transpose(skip_input, transpose_2_output,
                       {Constant(0), Constant(2), Constant(3), Constant(1)}, 0))
    return false;
  if (!graph.Convolution(
          transpose_1_output, model8_m0_cv1_conv_weight, model8_m0_cv1_conv_bias,
          model8_m0_cv1_conv_output, Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(128), Constant(64)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Convolution(
          transpose_1_output, model8_m0_cv2_conv_weight, model8_m0_cv2_conv_bias,
          model8_m0_cv2_conv_output, Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(128), Constant(64)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Convolution(
          model8_m0_cv1_conv_output, model8_m0_m0_cv1_conv_weight, model8_m0_m0_cv1_conv_bias,
          model8_m0_m0_cv1_conv_output, Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
          Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(64), Constant(64)}, 0))
    return false;
  if (!graph.Silu(model8_m0_m0_cv1_conv_output, model8_m0_m0_cv1_conv_activation, 0)) return false;
  if (!graph.Convolution(model8_m0_m0_cv1_conv_activation, model8_m0_m0_cv2_conv_weight,
                         model8_m0_m0_cv2_conv_bias, model8_m0_m0_cv2_conv_output,
                         Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
                         Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), Constant(64), Constant(64)}, 0))
    return false;
  if (!graph.Silu(model8_m0_m0_cv2_conv_output, model8_m0_m0_cv2_conv_activation, 0)) return false;
  if (!graph.Add(model8_m0_cv1_conv_output, model8_m0_m0_cv2_conv_activation, residual_sum_1, 0))
    return false;
  if (!graph.Convolution(
          residual_sum_1, model8_m0_m1_cv1_conv_weight, model8_m0_m1_cv1_conv_bias,
          model8_m0_m1_cv1_conv_output, Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
          Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(64), Constant(64)}, 0))
    return false;
  if (!graph.Silu(model8_m0_m1_cv1_conv_output, model8_m0_m1_cv1_conv_activation, 0)) return false;
  if (!graph.Convolution(model8_m0_m1_cv1_conv_activation, model8_m0_m1_cv2_conv_weight,
                         model8_m0_m1_cv2_conv_bias, model8_m0_m1_cv2_conv_output,
                         Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
                         Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), Constant(64), Constant(64)}, 0))
    return false;
  if (!graph.Silu(model8_m0_m1_cv2_conv_output, model8_m0_m1_cv2_conv_activation, 0)) return false;
  if (!graph.Add(residual_sum_1, model8_m0_m1_cv2_conv_activation, residual_sum_2, 0)) return false;
  if (!graph.Concatenate({residual_sum_2, model8_m0_cv2_conv_output}, concatenation_1, Constant(3),
                         0))
    return false;
  if (!graph.Convolution(
          concatenation_1, model8_m0_cv3_conv_weight, model8_m0_cv3_conv_bias,
          model8_m0_cv3_conv_output, Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(128), Constant(128)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Concatenate({transpose_2_output, transpose_1_output, model8_m0_cv3_conv_output},
                         concatenation_2, Constant(3), 0))
    return false;
  if (!graph.Convolution(
          concatenation_2, model8_cv2_conv_weight, model8_cv2_conv_bias, model8_cv2_conv_output,
          Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(384), Constant(256)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Convolution(
          model8_cv2_conv_output, model9_cv1_conv_weight, model9_cv1_conv_bias,
          model9_cv1_conv_output, Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(256), Constant(128)}, 0))
    return false;
  if (!graph.MaxPool(model9_cv1_conv_output, max_pool_1_output,
                     Padding{Constant(2), Constant(2), Constant(2), Constant(2)},
                     Kernel{Constant(5), Constant(5)}, Stride{Constant(1), Constant(1)},
                     Dilation{Constant(1), Constant(1)}, 64U))
    return false;
  if (!graph.MaxPool(max_pool_1_output, max_pool_2_output,
                     Padding{Constant(2), Constant(2), Constant(2), Constant(2)},
                     Kernel{Constant(5), Constant(5)}, Stride{Constant(1), Constant(1)},
                     Dilation{Constant(1), Constant(1)}, 64U))
    return false;
  if (!graph.MaxPool(max_pool_2_output, max_pool_3_output,
                     Padding{Constant(2), Constant(2), Constant(2), Constant(2)},
                     Kernel{Constant(5), Constant(5)}, Stride{Constant(1), Constant(1)},
                     Dilation{Constant(1), Constant(1)}, 64U))
    return false;
  if (!graph.Concatenate(
          {model9_cv1_conv_output, max_pool_1_output, max_pool_2_output, max_pool_3_output},
          concatenation_3, Constant(3), 0))
    return false;
  if (!graph.Convolution(
          concatenation_3, model9_cv2_conv_weight, model9_cv2_conv_bias, model9_cv2_conv_output,
          Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(512), Constant(256)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Add(model9_cv2_conv_output, model8_cv2_conv_output, residual_sum_3, 0)) return false;
  if (!graph.Convolution(
          residual_sum_3, model10_cv1_conv_weight, model10_cv1_conv_bias, model10_cv1_conv_output,
          Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(256), Constant(256)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Transpose(model10_cv1_conv_output, attention_entry_output,
                       {Constant(0), Constant(3), Constant(1), Constant(2)}, 0))
    return false;
  return graph.Finish();
}

// Projects the P5 residual branch into two attention heads containing query,
// key, and value channels; the schedule performs the channel split.
bool BuildBackboneAttentionQkv(const GraphContext& context) {
  FixedGraphBuilder graph(context, GraphId::kBackboneAttentionQkv,
                          BackboneAttentionQkvBindings::kCount);
  if (!graph.Begin()) return false;
  const GraphValue residual_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), Constant(128), ResolutionDivided(32), ResolutionDivided(32)},
      BackboneAttentionQkvBindings::kResidualInput);
  const GraphValue transpose_1_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue model10_m0_attn_qkv_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(256), Constant(1), Constant(1), Constant(128)},
                         WeightSlot::kModel10M0AttnQkvConvWeight);
  const GraphValue model10_m0_attn_qkv_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue model10_m0_attn_qkv_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(256)}, WeightSlot::kModel10M0AttnQkvConvBias);
  const GraphValue transpose_2_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(256), ResolutionDivided(32), ResolutionDivided(32)});
  const GraphValue qkv_output = graph.DefineOutput(
      ValueType::kFp16, {Constant(1), Constant(2), Constant(128), ResolutionDividedSquared(32)},
      BackboneAttentionQkvBindings::kQkvOutput);
  if (!graph.DefinitionsValid()) return false;
  if (!graph.Transpose(residual_input, transpose_1_output,
                       {Constant(0), Constant(2), Constant(3), Constant(1)}, 0))
    return false;
  if (!graph.Convolution(transpose_1_output, model10_m0_attn_qkv_conv_weight,
                         model10_m0_attn_qkv_conv_bias, model10_m0_attn_qkv_conv_output,
                         Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
                         Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), Constant(128), Constant(256)}, 0))
    return false;
  if (!graph.Transpose(model10_m0_attn_qkv_conv_output, transpose_2_output,
                       {Constant(0), Constant(3), Constant(1), Constant(2)}, 0))
    return false;
  if (!graph.Reshape(transpose_2_output, qkv_output,
                     {Constant(1), Constant(2), Constant(128), ResolutionDividedSquared(32)}, 0))
    return false;
  return graph.Finish();
}

// Scales each query by 1 / sqrt(head width) and transposes it for QK matrix
// multiplication.
bool BuildBackboneAttentionScale(const GraphContext& context) {
  FixedGraphBuilder graph(context, GraphId::kBackboneAttentionScale,
                          BackboneAttentionScaleBindings::kCount);
  if (!graph.Begin()) return false;
  const GraphValue query_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), Constant(2), Constant(32), ResolutionDividedSquared(32)},
      BackboneAttentionScaleBindings::kQueryInput);
  const GraphValue scale_input = graph.DefineInput(ValueType::kFp16, {Constant(1)},
                                                   BackboneAttentionScaleBindings::kScaleInput);
  const GraphValue product_1 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(2), Constant(32), ResolutionDividedSquared(32)});
  const GraphValue scaled_query_output = graph.DefineOutput(
      ValueType::kFp16, {Constant(1), Constant(2), ResolutionDividedSquared(32), Constant(32)},
      BackboneAttentionScaleBindings::kScaledQueryOutput);
  if (!graph.DefinitionsValid()) return false;
  if (!graph.Multiply(query_input, scale_input, product_1, 0)) return false;
  if (!graph.Transpose(product_1, scaled_query_output,
                       {Constant(0), Constant(1), Constant(3), Constant(2)}, 0))
    return false;
  return graph.Finish();
}

// Computes and normalizes the two P5 attention matrices. Copies at the
// partition boundary are owned by the schedule.
bool BuildBackboneAttentionMatrix(const GraphContext& context) {
  FixedGraphBuilder graph(context, GraphId::kBackboneAttentionMatrix,
                          BackboneAttentionMatrixBindings::kCount);
  if (!graph.Begin()) return false;
  const GraphValue reshape_1_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(2), Constant(32), ResolutionDividedSquared(32)});
  const GraphValue key_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), Constant(2), Constant(32), ResolutionDividedSquared(32)},
      BackboneAttentionMatrixBindings::kKeyInput);
  const GraphValue reshape_2_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(2), ResolutionDividedSquared(32), Constant(32)});
  const GraphValue query_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), Constant(2), ResolutionDividedSquared(32), Constant(32)},
      BackboneAttentionMatrixBindings::kQueryInput);
  const GraphValue matrix_product_1 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(2), ResolutionDividedSquared(32), ResolutionDividedSquared(32)});
  const GraphValue reshape_3_output = graph.DefineIntermediate(
      ValueType::kFp16,
      {Constant(1), Constant(2), ResolutionDividedSquared(32), ResolutionDividedSquared(32)});
  const GraphValue softmax_1_output = graph.DefineIntermediate(
      ValueType::kFp16,
      {Constant(1), Constant(2), ResolutionDividedSquared(32), ResolutionDividedSquared(32)});
  const GraphValue matrix_output = graph.DefineOutput(
      ValueType::kFp16,
      {Constant(1), Constant(2), ResolutionDividedSquared(32), ResolutionDividedSquared(32)},
      BackboneAttentionMatrixBindings::kMatrixOutput);
  if (!graph.DefinitionsValid()) return false;
  if (!graph.Reshape(key_input, reshape_1_output,
                     {Constant(2), Constant(32), ResolutionDividedSquared(32)}, 0))
    return false;
  if (!graph.Reshape(query_input, reshape_2_output,
                     {Constant(2), ResolutionDividedSquared(32), Constant(32)}, 0))
    return false;
  if (!graph.BatchMatrixMultiply(reshape_2_output, reshape_1_output, matrix_product_1, 0))
    return false;
  if (!graph.Reshape(
          matrix_product_1, reshape_3_output,
          {Constant(1), Constant(2), ResolutionDividedSquared(32), ResolutionDividedSquared(32)},
          0))
    return false;
  if (!graph.Softmax(reshape_3_output, softmax_1_output, 0)) return false;
  if (!graph.Transpose(softmax_1_output, matrix_output,
                       {Constant(0), Constant(1), Constant(3), Constant(2)}, 0))
    return false;
  return graph.Finish();
}

// Applies attention to the values, adds positional encoding and residual/FFN
// paths, rejoins the model.10 bypass, and emits both P5 and its neck projection.
bool BuildBackboneAttentionOutput(const GraphContext& context) {
  FixedGraphBuilder graph(context, GraphId::kBackboneAttentionOutput,
                          BackboneAttentionOutputBindings::kCount);
  if (!graph.Begin()) return false;
  const GraphValue residual_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), Constant(128), ResolutionDivided(32), ResolutionDivided(32)},
      BackboneAttentionOutputBindings::kResidualInput);
  const GraphValue transpose_1_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue bypass_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), Constant(128), ResolutionDivided(32), ResolutionDivided(32)},
      BackboneAttentionOutputBindings::kBypassInput);
  const GraphValue transpose_2_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue reshape_1_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(2), Constant(64), ResolutionDividedSquared(32)});
  const GraphValue value_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), Constant(2), Constant(64), ResolutionDividedSquared(32)},
      BackboneAttentionOutputBindings::kValueInput);
  const GraphValue reshape_2_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(128), ResolutionDivided(32), ResolutionDivided(32)});
  const GraphValue positional_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), Constant(2), Constant(64), ResolutionDividedSquared(32)},
      BackboneAttentionOutputBindings::kPositionalInput);
  const GraphValue transpose_3_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue model10_m0_attn_pe_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(1), Constant(3), Constant(3), Constant(128)},
                         WeightSlot::kModel10M0AttnPeConvWeight);
  const GraphValue model10_m0_attn_pe_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue model10_m0_attn_pe_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(128)}, WeightSlot::kModel10M0AttnPeConvBias);
  const GraphValue reshape_3_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(2), ResolutionDividedSquared(32), ResolutionDividedSquared(32)});
  const GraphValue matrix_input = graph.DefineInput(
      ValueType::kFp16,
      {Constant(1), Constant(2), ResolutionDividedSquared(32), ResolutionDividedSquared(32)},
      BackboneAttentionOutputBindings::kMatrixInput);
  const GraphValue matrix_product_1 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(2), Constant(64), ResolutionDividedSquared(32)});
  const GraphValue reshape_4_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(2), Constant(64), ResolutionDividedSquared(32)});
  const GraphValue reshape_5_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(128), ResolutionDivided(32), ResolutionDivided(32)});
  const GraphValue transpose_4_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue residual_sum_1 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue model10_m0_attn_proj_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(128), Constant(1), Constant(1), Constant(128)},
                         WeightSlot::kModel10M0AttnProjConvWeight);
  const GraphValue model10_m0_attn_proj_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue model10_m0_attn_proj_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(128)}, WeightSlot::kModel10M0AttnProjConvBias);
  const GraphValue residual_sum_2 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue model10_m0_ffn0_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(256), Constant(1), Constant(1), Constant(128)},
                         WeightSlot::kModel10M0Ffn0ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_24 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue model10_m0_ffn0_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(256)}, WeightSlot::kModel10M0Ffn0ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_26 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue model10_m0_ffn0_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue model10_m0_ffn1_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(128), Constant(1), Constant(1), Constant(256)},
                         WeightSlot::kModel10M0Ffn1ConvWeight);
  const GraphValue model10_m0_ffn1_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue model10_m0_ffn1_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(128)}, WeightSlot::kModel10M0Ffn1ConvBias);
  const GraphValue residual_sum_3 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue concatenation_1 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue model10_cv2_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(256), Constant(1), Constant(1), Constant(256)},
                         WeightSlot::kModel10Cv2ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_34 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue model10_cv2_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(256)}, WeightSlot::kModel10Cv2ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_36 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue model10_cv2_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue backbone_output = graph.DefineOutput(
      ValueType::kFp16, {Constant(1), Constant(256), ResolutionDivided(32), ResolutionDivided(32)},
      BackboneAttentionOutputBindings::kBackboneOutput);
  const GraphValue model13_cv1_conv_weight_input0256 =
      graph.DefineWeight(ValueType::kFp32, {Constant(128), Constant(1), Constant(1), Constant(256)},
                         WeightSlot::kModel13Cv1ConvWeightInput0256);
  const GraphValue projection_zero_bias128 =
      graph.DefineWeight(ValueType::kFp32, {Constant(128)}, WeightSlot::kProjectionZeroBias128);
  const GraphValue p5_projection_output = graph.DefineOutput(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)},
      BackboneAttentionOutputBindings::kP5ProjectionOutput);
  [[maybe_unused]] const GraphValue reserved_value_42 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(128), ResolutionDivided(32), ResolutionDivided(32)});
  if (!graph.DefinitionsValid()) return false;
  if (!graph.Transpose(residual_input, transpose_1_output,
                       {Constant(0), Constant(2), Constant(3), Constant(1)}, 0))
    return false;
  if (!graph.Transpose(bypass_input, transpose_2_output,
                       {Constant(0), Constant(2), Constant(3), Constant(1)}, 0))
    return false;
  if (!graph.Reshape(value_input, reshape_1_output,
                     {Constant(2), Constant(64), ResolutionDividedSquared(32)}, 0))
    return false;
  if (!graph.Reshape(positional_input, reshape_2_output,
                     {Constant(1), Constant(128), ResolutionDivided(32), ResolutionDivided(32)}, 0))
    return false;
  if (!graph.Transpose(reshape_2_output, transpose_3_output,
                       {Constant(0), Constant(2), Constant(3), Constant(1)}, 0))
    return false;
  if (!graph.DepthwiseConvolution(
          transpose_3_output, model10_m0_attn_pe_conv_weight, model10_m0_attn_pe_conv_bias,
          model10_m0_attn_pe_conv_output,
          Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
          Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(128), Constant(1), Constant(1)}, 0))
    return false;
  if (!graph.Reshape(matrix_input, reshape_3_output,
                     {Constant(2), ResolutionDividedSquared(32), ResolutionDividedSquared(32)}, 0))
    return false;
  if (!graph.BatchMatrixMultiply(reshape_1_output, reshape_3_output, matrix_product_1, 0))
    return false;
  if (!graph.Reshape(matrix_product_1, reshape_4_output,
                     {Constant(1), Constant(2), Constant(64), ResolutionDividedSquared(32)}, 0))
    return false;
  if (!graph.Reshape(reshape_4_output, reshape_5_output,
                     {Constant(1), Constant(128), ResolutionDivided(32), ResolutionDivided(32)}, 0))
    return false;
  if (!graph.Transpose(reshape_5_output, transpose_4_output,
                       {Constant(0), Constant(2), Constant(3), Constant(1)}, 0))
    return false;
  if (!graph.Add(transpose_4_output, model10_m0_attn_pe_conv_output, residual_sum_1, 0))
    return false;
  if (!graph.Convolution(residual_sum_1, model10_m0_attn_proj_conv_weight,
                         model10_m0_attn_proj_conv_bias, model10_m0_attn_proj_conv_output,
                         Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
                         Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), Constant(128), Constant(128)}, 0))
    return false;
  if (!graph.Add(transpose_1_output, model10_m0_attn_proj_conv_output, residual_sum_2, 0))
    return false;
  if (!graph.Convolution(
          residual_sum_2, model10_m0_ffn0_conv_weight, model10_m0_ffn0_conv_bias,
          model10_m0_ffn0_conv_output, Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(128), Constant(256)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Convolution(
          model10_m0_ffn0_conv_output, model10_m0_ffn1_conv_weight, model10_m0_ffn1_conv_bias,
          model10_m0_ffn1_conv_output, Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(256), Constant(128)}, 0))
    return false;
  if (!graph.Add(residual_sum_2, model10_m0_ffn1_conv_output, residual_sum_3, 0)) return false;
  if (!graph.Concatenate({transpose_2_output, residual_sum_3}, concatenation_1, Constant(3), 0))
    return false;
  if (!graph.Convolution(
          concatenation_1, model10_cv2_conv_weight, model10_cv2_conv_bias, model10_cv2_conv_output,
          Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(256), Constant(256)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Transpose(model10_cv2_conv_output, backbone_output,
                       {Constant(0), Constant(3), Constant(1), Constant(2)}, 0))
    return false;
  if (!graph.Convolution(
          model10_cv2_conv_output, model13_cv1_conv_weight_input0256, projection_zero_bias128,
          p5_projection_output, Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(256), Constant(128)}, 0))
    return false;
  return graph.Finish();
}

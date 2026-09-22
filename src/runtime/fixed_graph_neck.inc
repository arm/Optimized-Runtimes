// Fixed top-down neck partitions. Reserved values reproduce XNNPACK's
// performance-sensitive declaration inventory but are not part of data flow.

// Projects the backbone P4 feature and adds the upsampled P5 projection to form
// the model.13 fusion input.
bool BuildNeckP4Fusion(const GraphContext& context) {
  FixedGraphBuilder graph(context, GraphId::kNeckP4Fusion, NeckP4FusionBindings::kCount);
  if (!graph.Begin()) return false;
  const GraphValue upsampled_p5_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)},
      NeckP4FusionBindings::kUpsampledP5Input);
  const GraphValue backbone_p4_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), Constant(128), ResolutionDivided(16), ResolutionDivided(16)},
      NeckP4FusionBindings::kBackboneP4Input);
  [[maybe_unused]] const GraphValue reserved_value_2 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)});
  const GraphValue transpose_1_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)});
  const GraphValue model13_cv1_conv_weight_input256384 =
      graph.DefineWeight(ValueType::kFp32, {Constant(128), Constant(1), Constant(1), Constant(128)},
                         WeightSlot::kModel13Cv1ConvWeightInput256384);
  [[maybe_unused]] const GraphValue reserved_value_5 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)});
  const GraphValue model13_cv1_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(128)}, WeightSlot::kModel13Cv1ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_7 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)});
  const GraphValue model13_cv1_conv_weight_input256384_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)});
  const GraphValue fusion_output = graph.DefineOutput(
      ValueType::kFp16, {Constant(1), Constant(128), ResolutionDivided(16), ResolutionDivided(16)},
      NeckP4FusionBindings::kFusionOutput);
  const GraphValue residual_sum_1 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)});
  const GraphValue residual_sum_1_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)});
  if (!graph.DefinitionsValid()) return false;
  if (!graph.Transpose(backbone_p4_input, transpose_1_output,
                       {Constant(0), Constant(2), Constant(3), Constant(1)}, 0))
    return false;
  if (!graph.Convolution(transpose_1_output, model13_cv1_conv_weight_input256384,
                         model13_cv1_conv_bias, model13_cv1_conv_weight_input256384_output,
                         Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
                         Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), Constant(128), Constant(128)}, 0))
    return false;
  if (!graph.Add(upsampled_p5_input, model13_cv1_conv_weight_input256384_output, residual_sum_1, 0))
    return false;
  if (!graph.Silu(residual_sum_1, residual_sum_1_activation, 0)) return false;
  if (!graph.Transpose(residual_sum_1_activation, fusion_output,
                       {Constant(0), Constant(3), Constant(1), Constant(2)}, 0))
    return false;
  return graph.Finish();
}

// Runs the model.13 C2f block and emits both the fused P4 feature and the
// narrower projection that the schedule upsamples toward P3.
bool BuildNeckP4Block(const GraphContext& context) {
  FixedGraphBuilder graph(context, GraphId::kNeckP4Block, NeckP4BlockBindings::kCount);
  if (!graph.Begin()) return false;
  const GraphValue main_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), Constant(64), ResolutionDivided(16), ResolutionDivided(16)},
      NeckP4BlockBindings::kMainInput);
  const GraphValue transpose_1_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(64)});
  const GraphValue skip_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), Constant(64), ResolutionDivided(16), ResolutionDivided(16)},
      NeckP4BlockBindings::kSkipInput);
  const GraphValue transpose_2_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(64)});
  const GraphValue model13_m0_cv1_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(32), Constant(1), Constant(1), Constant(64)},
                         WeightSlot::kModel13M0Cv1ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_5 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model13_m0_cv1_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(32)}, WeightSlot::kModel13M0Cv1ConvBias);
  const GraphValue model13_m0_cv2_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(32), Constant(1), Constant(1), Constant(64)},
                         WeightSlot::kModel13M0Cv2ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_8 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model13_m0_cv2_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(32)}, WeightSlot::kModel13M0Cv2ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_10 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  [[maybe_unused]] const GraphValue reserved_value_11 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model13_m0_cv1_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model13_m0_cv2_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model13_m0_m0_cv1_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(32), Constant(3), Constant(3), Constant(32)},
                         WeightSlot::kModel13M0M0Cv1ConvWeight);
  const GraphValue model13_m0_m0_cv1_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model13_m0_m0_cv1_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(32)}, WeightSlot::kModel13M0M0Cv1ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_17 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model13_m0_m0_cv1_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model13_m0_m0_cv2_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(32), Constant(3), Constant(3), Constant(32)},
                         WeightSlot::kModel13M0M0Cv2ConvWeight);
  const GraphValue model13_m0_m0_cv2_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model13_m0_m0_cv2_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(32)}, WeightSlot::kModel13M0M0Cv2ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_22 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model13_m0_m0_cv2_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue residual_sum_1 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model13_m0_m1_cv1_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(32), Constant(3), Constant(3), Constant(32)},
                         WeightSlot::kModel13M0M1Cv1ConvWeight);
  const GraphValue model13_m0_m1_cv1_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model13_m0_m1_cv1_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(32)}, WeightSlot::kModel13M0M1Cv1ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_28 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model13_m0_m1_cv1_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model13_m0_m1_cv2_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(32), Constant(3), Constant(3), Constant(32)},
                         WeightSlot::kModel13M0M1Cv2ConvWeight);
  const GraphValue model13_m0_m1_cv2_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model13_m0_m1_cv2_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(32)}, WeightSlot::kModel13M0M1Cv2ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_33 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model13_m0_m1_cv2_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue residual_sum_2 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue concatenation_1 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(64)});
  const GraphValue model13_m0_cv3_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(64), Constant(1), Constant(1), Constant(64)},
                         WeightSlot::kModel13M0Cv3ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_38 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(64)});
  const GraphValue model13_m0_cv3_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(64)}, WeightSlot::kModel13M0Cv3ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_40 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(64)});
  const GraphValue model13_m0_cv3_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(64)});
  const GraphValue concatenation_2 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(192)});
  const GraphValue model13_cv2_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(128), Constant(1), Constant(1), Constant(192)},
                         WeightSlot::kModel13Cv2ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_44 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)});
  const GraphValue model13_cv2_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(128)}, WeightSlot::kModel13Cv2ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_46 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)});
  const GraphValue model13_cv2_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)});
  const GraphValue neck_p4_output = graph.DefineOutput(
      ValueType::kFp16, {Constant(1), Constant(128), ResolutionDivided(16), ResolutionDivided(16)},
      NeckP4BlockBindings::kNeckP4Output);
  const GraphValue model16_cv1_conv_weight_input0128 =
      graph.DefineWeight(ValueType::kFp32, {Constant(64), Constant(1), Constant(1), Constant(128)},
                         WeightSlot::kModel16Cv1ConvWeightInput0128);
  const GraphValue projection_zero_bias64 =
      graph.DefineWeight(ValueType::kFp32, {Constant(64)}, WeightSlot::kProjectionZeroBias64);
  const GraphValue p4_projection_output = graph.DefineOutput(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(64)},
      NeckP4BlockBindings::kP4ProjectionOutput);
  [[maybe_unused]] const GraphValue reserved_value_52 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(64), ResolutionDivided(16), ResolutionDivided(16)});
  if (!graph.DefinitionsValid()) return false;
  if (!graph.Transpose(main_input, transpose_1_output,
                       {Constant(0), Constant(2), Constant(3), Constant(1)}, 0))
    return false;
  if (!graph.Transpose(skip_input, transpose_2_output,
                       {Constant(0), Constant(2), Constant(3), Constant(1)}, 0))
    return false;
  if (!graph.Convolution(
          transpose_1_output, model13_m0_cv1_conv_weight, model13_m0_cv1_conv_bias,
          model13_m0_cv1_conv_output, Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(64), Constant(32)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Convolution(
          transpose_1_output, model13_m0_cv2_conv_weight, model13_m0_cv2_conv_bias,
          model13_m0_cv2_conv_output, Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(64), Constant(32)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Convolution(model13_m0_cv1_conv_output, model13_m0_m0_cv1_conv_weight,
                         model13_m0_m0_cv1_conv_bias, model13_m0_m0_cv1_conv_output,
                         Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
                         Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), Constant(32), Constant(32)}, 0))
    return false;
  if (!graph.Silu(model13_m0_m0_cv1_conv_output, model13_m0_m0_cv1_conv_activation, 0))
    return false;
  if (!graph.Convolution(model13_m0_m0_cv1_conv_activation, model13_m0_m0_cv2_conv_weight,
                         model13_m0_m0_cv2_conv_bias, model13_m0_m0_cv2_conv_output,
                         Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
                         Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), Constant(32), Constant(32)}, 0))
    return false;
  if (!graph.Silu(model13_m0_m0_cv2_conv_output, model13_m0_m0_cv2_conv_activation, 0))
    return false;
  if (!graph.Add(model13_m0_cv1_conv_output, model13_m0_m0_cv2_conv_activation, residual_sum_1, 0))
    return false;
  if (!graph.Convolution(residual_sum_1, model13_m0_m1_cv1_conv_weight, model13_m0_m1_cv1_conv_bias,
                         model13_m0_m1_cv1_conv_output,
                         Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
                         Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), Constant(32), Constant(32)}, 0))
    return false;
  if (!graph.Silu(model13_m0_m1_cv1_conv_output, model13_m0_m1_cv1_conv_activation, 0))
    return false;
  if (!graph.Convolution(model13_m0_m1_cv1_conv_activation, model13_m0_m1_cv2_conv_weight,
                         model13_m0_m1_cv2_conv_bias, model13_m0_m1_cv2_conv_output,
                         Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
                         Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), Constant(32), Constant(32)}, 0))
    return false;
  if (!graph.Silu(model13_m0_m1_cv2_conv_output, model13_m0_m1_cv2_conv_activation, 0))
    return false;
  if (!graph.Add(residual_sum_1, model13_m0_m1_cv2_conv_activation, residual_sum_2, 0))
    return false;
  if (!graph.Concatenate({residual_sum_2, model13_m0_cv2_conv_output}, concatenation_1, Constant(3),
                         0))
    return false;
  if (!graph.Convolution(
          concatenation_1, model13_m0_cv3_conv_weight, model13_m0_cv3_conv_bias,
          model13_m0_cv3_conv_output, Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(64), Constant(64)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Concatenate({transpose_2_output, transpose_1_output, model13_m0_cv3_conv_output},
                         concatenation_2, Constant(3), 0))
    return false;
  if (!graph.Convolution(
          concatenation_2, model13_cv2_conv_weight, model13_cv2_conv_bias, model13_cv2_conv_output,
          Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(192), Constant(128)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Transpose(model13_cv2_conv_output, neck_p4_output,
                       {Constant(0), Constant(3), Constant(1), Constant(2)}, 0))
    return false;
  if (!graph.Convolution(
          model13_cv2_conv_output, model16_cv1_conv_weight_input0128, projection_zero_bias64,
          p4_projection_output, Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(128), Constant(64)}, 0))
    return false;
  return graph.Finish();
}

// Projects the backbone P3 feature and adds the upsampled P4 projection. The
// schedule splits the result into the two model.16 branches.
bool BuildNeckP3Fusion(const GraphContext& context) {
  FixedGraphBuilder graph(context, GraphId::kNeckP3Fusion, NeckP3FusionBindings::kCount);
  if (!graph.Begin()) return false;
  [[maybe_unused]] const GraphValue reserved_value_0 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(128), ResolutionDivided(8), ResolutionDivided(8)});
  [[maybe_unused]] const GraphValue reserved_value_1 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(128), ResolutionDivided(8), ResolutionDivided(8)});
  [[maybe_unused]] const GraphValue reserved_value_2 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(256), ResolutionDivided(8), ResolutionDivided(8)});
  const GraphValue upsampled_p4_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(64)},
      NeckP3FusionBindings::kUpsampledP4Input);
  const GraphValue model16_cv1_conv_weight_input128256 =
      graph.DefineWeight(ValueType::kFp32, {Constant(64), Constant(1), Constant(1), Constant(128)},
                         WeightSlot::kModel16Cv1ConvWeightInput128256);
  [[maybe_unused]] const GraphValue reserved_value_5 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(64)});
  const GraphValue model16_cv1_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(64)}, WeightSlot::kModel16Cv1ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_7 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(64)});
  const GraphValue model16_cv1_conv_weight_input128256_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(64)});
  [[maybe_unused]] const GraphValue reserved_value_9 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(64), ResolutionDivided(8), ResolutionDivided(8)});
  [[maybe_unused]] const GraphValue reserved_value_10 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(64)});
  const GraphValue fusion_output = graph.DefineInput(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(128)},
      NeckP3FusionBindings::kFusionOutput);
  [[maybe_unused]] const GraphValue reserved_value_12 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(128)});
  const GraphValue residual_sum_1 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(64)});
  const GraphValue backbone_p3_input = graph.DefineOutput(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(64)},
      NeckP3FusionBindings::kBackboneP3Input);
  if (!graph.DefinitionsValid()) return false;
  if (!graph.Convolution(fusion_output, model16_cv1_conv_weight_input128256, model16_cv1_conv_bias,
                         model16_cv1_conv_weight_input128256_output,
                         Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
                         Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), Constant(128), Constant(64)}, 0))
    return false;
  if (!graph.Add(upsampled_p4_input, model16_cv1_conv_weight_input128256_output, residual_sum_1, 0))
    return false;
  if (!graph.Silu(residual_sum_1, backbone_p3_input, 0)) return false;
  return graph.Finish();
}

// Fixed bottom-up detection-neck and head partitions. P3, P4, and P5 are the
// stride-8, stride-16, and stride-32 detection scales. Reserved values preserve
// XNNPACK's performance-sensitive declaration inventory without entering data
// flow.

// Completes the model.16 P3 block, emits the finest detection feature, and
// downsamples toward the P4 detection path.
bool BuildDetectionP3Path(const GraphContext& context) {
  FixedGraphBuilder graph(context, GraphId::kDetectionP3Path, DetectionP3PathBindings::kCount);
  if (!graph.Begin()) return false;
  const GraphValue main_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(32)},
      DetectionP3PathBindings::kMainInput);
  [[maybe_unused]] const GraphValue reserved_value_1 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(32)});
  const GraphValue skip_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(32)},
      DetectionP3PathBindings::kSkipInput);
  [[maybe_unused]] const GraphValue reserved_value_3 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(32)});
  const GraphValue neck_p4_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), Constant(128), ResolutionDivided(16), ResolutionDivided(16)},
      DetectionP3PathBindings::kNeckP4Input);
  const GraphValue transpose_1_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)});
  const GraphValue model16_m0_cv1_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(16), Constant(1), Constant(1), Constant(32)},
                         WeightSlot::kModel16M0Cv1ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_7 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(16)});
  const GraphValue model16_m0_cv1_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(16)}, WeightSlot::kModel16M0Cv1ConvBias);
  const GraphValue model16_m0_cv2_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(16), Constant(1), Constant(1), Constant(32)},
                         WeightSlot::kModel16M0Cv2ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_10 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(16)});
  const GraphValue model16_m0_cv2_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(16)}, WeightSlot::kModel16M0Cv2ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_12 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(16)});
  [[maybe_unused]] const GraphValue reserved_value_13 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(16)});
  const GraphValue model16_m0_cv1_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(16)});
  const GraphValue model16_m0_cv2_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(16)});
  const GraphValue model16_m0_m0_cv1_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(16), Constant(3), Constant(3), Constant(16)},
                         WeightSlot::kModel16M0M0Cv1ConvWeight);
  const GraphValue model16_m0_m0_cv1_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(16)});
  const GraphValue model16_m0_m0_cv1_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(16)}, WeightSlot::kModel16M0M0Cv1ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_19 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(16)});
  const GraphValue model16_m0_m0_cv1_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(16)});
  const GraphValue model16_m0_m0_cv2_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(16), Constant(3), Constant(3), Constant(16)},
                         WeightSlot::kModel16M0M0Cv2ConvWeight);
  const GraphValue model16_m0_m0_cv2_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(16)});
  const GraphValue model16_m0_m0_cv2_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(16)}, WeightSlot::kModel16M0M0Cv2ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_24 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(16)});
  const GraphValue model16_m0_m0_cv2_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(16)});
  const GraphValue residual_sum_1 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(16)});
  const GraphValue model16_m0_m1_cv1_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(16), Constant(3), Constant(3), Constant(16)},
                         WeightSlot::kModel16M0M1Cv1ConvWeight);
  const GraphValue model16_m0_m1_cv1_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(16)});
  const GraphValue model16_m0_m1_cv1_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(16)}, WeightSlot::kModel16M0M1Cv1ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_30 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(16)});
  const GraphValue model16_m0_m1_cv1_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(16)});
  const GraphValue model16_m0_m1_cv2_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(16), Constant(3), Constant(3), Constant(16)},
                         WeightSlot::kModel16M0M1Cv2ConvWeight);
  const GraphValue model16_m0_m1_cv2_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(16)});
  const GraphValue model16_m0_m1_cv2_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(16)}, WeightSlot::kModel16M0M1Cv2ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_35 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(16)});
  const GraphValue model16_m0_m1_cv2_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(16)});
  const GraphValue residual_sum_2 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(16)});
  const GraphValue concatenation_1 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(32)});
  const GraphValue model16_m0_cv3_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(32), Constant(1), Constant(1), Constant(32)},
                         WeightSlot::kModel16M0Cv3ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_40 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(32)});
  const GraphValue model16_m0_cv3_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(32)}, WeightSlot::kModel16M0Cv3ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_42 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(32)});
  const GraphValue model16_m0_cv3_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(32)});
  const GraphValue concatenation_2 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(96)});
  const GraphValue model16_cv2_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(64), Constant(1), Constant(1), Constant(96)},
                         WeightSlot::kModel16Cv2ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_46 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(64)});
  const GraphValue model16_cv2_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(64)}, WeightSlot::kModel16Cv2ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_48 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(64)});
  const GraphValue detection_p3_output = graph.DefineOutput(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(64)},
      DetectionP3PathBindings::kDetectionP3Output);
  [[maybe_unused]] const GraphValue reserved_value_50 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(64), ResolutionDivided(8), ResolutionDivided(8)});
  const GraphValue model17_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(64), Constant(3), Constant(3), Constant(64)},
                         WeightSlot::kModel17ConvWeight);
  const GraphValue model17_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(64)});
  const GraphValue model17_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(64)}, WeightSlot::kModel17ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_54 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(64)});
  const GraphValue model17_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(64)});
  const GraphValue concatenation_3 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(192)});
  const GraphValue model19_cv1_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(128), Constant(1), Constant(1), Constant(192)},
                         WeightSlot::kModel19Cv1ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_58 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)});
  const GraphValue model19_cv1_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(128)}, WeightSlot::kModel19Cv1ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_60 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)});
  const GraphValue model19_cv1_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)});
  const GraphValue p4_projection_output = graph.DefineOutput(
      ValueType::kFp16, {Constant(1), Constant(128), ResolutionDivided(16), ResolutionDivided(16)},
      DetectionP3PathBindings::kP4ProjectionOutput);
  if (!graph.DefinitionsValid()) return false;
  if (!graph.Transpose(neck_p4_input, transpose_1_output,
                       {Constant(0), Constant(2), Constant(3), Constant(1)}, 0))
    return false;
  if (!graph.Convolution(
          main_input, model16_m0_cv1_conv_weight, model16_m0_cv1_conv_bias,
          model16_m0_cv1_conv_output, Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(32), Constant(16)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Convolution(
          main_input, model16_m0_cv2_conv_weight, model16_m0_cv2_conv_bias,
          model16_m0_cv2_conv_output, Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(32), Constant(16)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Convolution(model16_m0_cv1_conv_output, model16_m0_m0_cv1_conv_weight,
                         model16_m0_m0_cv1_conv_bias, model16_m0_m0_cv1_conv_output,
                         Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
                         Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), Constant(16), Constant(16)}, 0))
    return false;
  if (!graph.Silu(model16_m0_m0_cv1_conv_output, model16_m0_m0_cv1_conv_activation, 0))
    return false;
  if (!graph.Convolution(model16_m0_m0_cv1_conv_activation, model16_m0_m0_cv2_conv_weight,
                         model16_m0_m0_cv2_conv_bias, model16_m0_m0_cv2_conv_output,
                         Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
                         Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), Constant(16), Constant(16)}, 0))
    return false;
  if (!graph.Silu(model16_m0_m0_cv2_conv_output, model16_m0_m0_cv2_conv_activation, 0))
    return false;
  if (!graph.Add(model16_m0_cv1_conv_output, model16_m0_m0_cv2_conv_activation, residual_sum_1, 0))
    return false;
  if (!graph.Convolution(residual_sum_1, model16_m0_m1_cv1_conv_weight, model16_m0_m1_cv1_conv_bias,
                         model16_m0_m1_cv1_conv_output,
                         Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
                         Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), Constant(16), Constant(16)}, 0))
    return false;
  if (!graph.Silu(model16_m0_m1_cv1_conv_output, model16_m0_m1_cv1_conv_activation, 0))
    return false;
  if (!graph.Convolution(model16_m0_m1_cv1_conv_activation, model16_m0_m1_cv2_conv_weight,
                         model16_m0_m1_cv2_conv_bias, model16_m0_m1_cv2_conv_output,
                         Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
                         Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), Constant(16), Constant(16)}, 0))
    return false;
  if (!graph.Silu(model16_m0_m1_cv2_conv_output, model16_m0_m1_cv2_conv_activation, 0))
    return false;
  if (!graph.Add(residual_sum_1, model16_m0_m1_cv2_conv_activation, residual_sum_2, 0))
    return false;
  if (!graph.Concatenate({residual_sum_2, model16_m0_cv2_conv_output}, concatenation_1, Constant(3),
                         0))
    return false;
  if (!graph.Convolution(
          concatenation_1, model16_m0_cv3_conv_weight, model16_m0_cv3_conv_bias,
          model16_m0_cv3_conv_output, Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(32), Constant(32)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Concatenate({skip_input, main_input, model16_m0_cv3_conv_output}, concatenation_2,
                         Constant(3), 0))
    return false;
  if (!graph.Convolution(
          concatenation_2, model16_cv2_conv_weight, model16_cv2_conv_bias, detection_p3_output,
          Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(96), Constant(64)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Convolution(
          detection_p3_output, model17_conv_weight, model17_conv_bias, model17_conv_output,
          Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
          Kernel{Constant(3), Constant(3)}, Stride{Constant(2), Constant(2)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(64), Constant(64)}, 0))
    return false;
  if (!graph.Silu(model17_conv_output, model17_conv_activation, 0)) return false;
  if (!graph.Concatenate({model17_conv_activation, transpose_1_output}, concatenation_3,
                         Constant(3), 0))
    return false;
  if (!graph.Convolution(
          concatenation_3, model19_cv1_conv_weight, model19_cv1_conv_bias, model19_cv1_conv_output,
          Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(192), Constant(128)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Transpose(model19_cv1_conv_output, p4_projection_output,
                       {Constant(0), Constant(3), Constant(1), Constant(2)}, 0))
    return false;
  return graph.Finish();
}

// Completes the model.19 P4 block, emits the middle detection feature, and
// downsamples toward the P5 attention head.
bool BuildDetectionP4Path(const GraphContext& context) {
  FixedGraphBuilder graph(context, GraphId::kDetectionP4Path, DetectionP4PathBindings::kCount);
  if (!graph.Begin()) return false;
  const GraphValue main_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), Constant(64), ResolutionDivided(16), ResolutionDivided(16)},
      DetectionP4PathBindings::kMainInput);
  const GraphValue transpose_1_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(64)});
  const GraphValue skip_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), Constant(64), ResolutionDivided(16), ResolutionDivided(16)},
      DetectionP4PathBindings::kSkipInput);
  const GraphValue transpose_2_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(64)});
  const GraphValue backbone_p5_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), Constant(256), ResolutionDivided(32), ResolutionDivided(32)},
      DetectionP4PathBindings::kBackboneP5Input);
  const GraphValue transpose_3_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue model19_m0_cv1_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(32), Constant(1), Constant(1), Constant(64)},
                         WeightSlot::kModel19M0Cv1ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_7 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model19_m0_cv1_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(32)}, WeightSlot::kModel19M0Cv1ConvBias);
  const GraphValue model19_m0_cv2_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(32), Constant(1), Constant(1), Constant(64)},
                         WeightSlot::kModel19M0Cv2ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_10 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model19_m0_cv2_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(32)}, WeightSlot::kModel19M0Cv2ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_12 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  [[maybe_unused]] const GraphValue reserved_value_13 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model19_m0_cv1_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model19_m0_cv2_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model19_m0_m0_cv1_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(32), Constant(3), Constant(3), Constant(32)},
                         WeightSlot::kModel19M0M0Cv1ConvWeight);
  const GraphValue model19_m0_m0_cv1_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model19_m0_m0_cv1_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(32)}, WeightSlot::kModel19M0M0Cv1ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_19 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model19_m0_m0_cv1_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model19_m0_m0_cv2_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(32), Constant(3), Constant(3), Constant(32)},
                         WeightSlot::kModel19M0M0Cv2ConvWeight);
  const GraphValue model19_m0_m0_cv2_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model19_m0_m0_cv2_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(32)}, WeightSlot::kModel19M0M0Cv2ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_24 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model19_m0_m0_cv2_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue residual_sum_1 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model19_m0_m1_cv1_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(32), Constant(3), Constant(3), Constant(32)},
                         WeightSlot::kModel19M0M1Cv1ConvWeight);
  const GraphValue model19_m0_m1_cv1_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model19_m0_m1_cv1_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(32)}, WeightSlot::kModel19M0M1Cv1ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_30 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model19_m0_m1_cv1_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model19_m0_m1_cv2_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(32), Constant(3), Constant(3), Constant(32)},
                         WeightSlot::kModel19M0M1Cv2ConvWeight);
  const GraphValue model19_m0_m1_cv2_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model19_m0_m1_cv2_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(32)}, WeightSlot::kModel19M0M1Cv2ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_35 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue model19_m0_m1_cv2_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue residual_sum_2 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(32)});
  const GraphValue concatenation_1 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(64)});
  const GraphValue model19_m0_cv3_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(64), Constant(1), Constant(1), Constant(64)},
                         WeightSlot::kModel19M0Cv3ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_40 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(64)});
  const GraphValue model19_m0_cv3_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(64)}, WeightSlot::kModel19M0Cv3ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_42 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(64)});
  const GraphValue model19_m0_cv3_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(64)});
  const GraphValue concatenation_2 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(192)});
  const GraphValue model19_cv2_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(128), Constant(1), Constant(1), Constant(192)},
                         WeightSlot::kModel19Cv2ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_46 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)});
  const GraphValue model19_cv2_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(128)}, WeightSlot::kModel19Cv2ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_48 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)});
  const GraphValue detection_p4_output = graph.DefineOutput(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)},
      DetectionP4PathBindings::kDetectionP4Output);
  [[maybe_unused]] const GraphValue reserved_value_50 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(128), ResolutionDivided(16), ResolutionDivided(16)});
  const GraphValue model20_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(128), Constant(3), Constant(3), Constant(128)},
                         WeightSlot::kModel20ConvWeight);
  const GraphValue model20_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue model20_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(128)}, WeightSlot::kModel20ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_54 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue model20_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue concatenation_3 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(384)});
  const GraphValue model22_cv1_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(256), Constant(1), Constant(1), Constant(384)},
                         WeightSlot::kModel22Cv1ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_58 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue model22_cv1_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(256)}, WeightSlot::kModel22Cv1ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_60 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue p5_projection_output = graph.DefineOutput(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)},
      DetectionP4PathBindings::kP5ProjectionOutput);
  [[maybe_unused]] const GraphValue reserved_value_62 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(256), ResolutionDivided(32), ResolutionDivided(32)});
  if (!graph.DefinitionsValid()) return false;
  if (!graph.Transpose(main_input, transpose_1_output,
                       {Constant(0), Constant(2), Constant(3), Constant(1)}, 0))
    return false;
  if (!graph.Transpose(skip_input, transpose_2_output,
                       {Constant(0), Constant(2), Constant(3), Constant(1)}, 0))
    return false;
  if (!graph.Transpose(backbone_p5_input, transpose_3_output,
                       {Constant(0), Constant(2), Constant(3), Constant(1)}, 0))
    return false;
  if (!graph.Convolution(
          transpose_1_output, model19_m0_cv1_conv_weight, model19_m0_cv1_conv_bias,
          model19_m0_cv1_conv_output, Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(64), Constant(32)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Convolution(
          transpose_1_output, model19_m0_cv2_conv_weight, model19_m0_cv2_conv_bias,
          model19_m0_cv2_conv_output, Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(64), Constant(32)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Convolution(model19_m0_cv1_conv_output, model19_m0_m0_cv1_conv_weight,
                         model19_m0_m0_cv1_conv_bias, model19_m0_m0_cv1_conv_output,
                         Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
                         Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), Constant(32), Constant(32)}, 0))
    return false;
  if (!graph.Silu(model19_m0_m0_cv1_conv_output, model19_m0_m0_cv1_conv_activation, 0))
    return false;
  if (!graph.Convolution(model19_m0_m0_cv1_conv_activation, model19_m0_m0_cv2_conv_weight,
                         model19_m0_m0_cv2_conv_bias, model19_m0_m0_cv2_conv_output,
                         Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
                         Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), Constant(32), Constant(32)}, 0))
    return false;
  if (!graph.Silu(model19_m0_m0_cv2_conv_output, model19_m0_m0_cv2_conv_activation, 0))
    return false;
  if (!graph.Add(model19_m0_cv1_conv_output, model19_m0_m0_cv2_conv_activation, residual_sum_1, 0))
    return false;
  if (!graph.Convolution(residual_sum_1, model19_m0_m1_cv1_conv_weight, model19_m0_m1_cv1_conv_bias,
                         model19_m0_m1_cv1_conv_output,
                         Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
                         Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), Constant(32), Constant(32)}, 0))
    return false;
  if (!graph.Silu(model19_m0_m1_cv1_conv_output, model19_m0_m1_cv1_conv_activation, 0))
    return false;
  if (!graph.Convolution(model19_m0_m1_cv1_conv_activation, model19_m0_m1_cv2_conv_weight,
                         model19_m0_m1_cv2_conv_bias, model19_m0_m1_cv2_conv_output,
                         Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
                         Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), Constant(32), Constant(32)}, 0))
    return false;
  if (!graph.Silu(model19_m0_m1_cv2_conv_output, model19_m0_m1_cv2_conv_activation, 0))
    return false;
  if (!graph.Add(residual_sum_1, model19_m0_m1_cv2_conv_activation, residual_sum_2, 0))
    return false;
  if (!graph.Concatenate({residual_sum_2, model19_m0_cv2_conv_output}, concatenation_1, Constant(3),
                         0))
    return false;
  if (!graph.Convolution(
          concatenation_1, model19_m0_cv3_conv_weight, model19_m0_cv3_conv_bias,
          model19_m0_cv3_conv_output, Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(64), Constant(64)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Concatenate({transpose_2_output, transpose_1_output, model19_m0_cv3_conv_output},
                         concatenation_2, Constant(3), 0))
    return false;
  if (!graph.Convolution(
          concatenation_2, model19_cv2_conv_weight, model19_cv2_conv_bias, detection_p4_output,
          Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(192), Constant(128)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Convolution(
          detection_p4_output, model20_conv_weight, model20_conv_bias, model20_conv_output,
          Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
          Kernel{Constant(3), Constant(3)}, Stride{Constant(2), Constant(2)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(128), Constant(128)}, 0))
    return false;
  if (!graph.Silu(model20_conv_output, model20_conv_activation, 0)) return false;
  if (!graph.Concatenate({model20_conv_activation, transpose_3_output}, concatenation_3,
                         Constant(3), 0))
    return false;
  if (!graph.Convolution(
          concatenation_3, model22_cv1_conv_weight, model22_cv1_conv_bias, p5_projection_output,
          Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(384), Constant(256)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  return graph.Finish();
}

// Projects the model.22 residual branch into two heads of query, key, and value
// channels; the schedule performs the channel split.
bool BuildDetectionP5Qkv(const GraphContext& context) {
  FixedGraphBuilder graph(context, GraphId::kDetectionP5Qkv, DetectionP5QkvBindings::kCount);
  if (!graph.Begin()) return false;
  [[maybe_unused]] const GraphValue reserved_value_0 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(128), ResolutionDivided(32), ResolutionDivided(32)});
  const GraphValue main_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)},
      DetectionP5QkvBindings::kMainInput);
  const GraphValue model22_m00_cv1_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(64), Constant(3), Constant(3), Constant(128)},
                         WeightSlot::kModel22M00Cv1ConvWeight);
  const GraphValue model22_m00_cv1_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(64)});
  const GraphValue model22_m00_cv1_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(64)}, WeightSlot::kModel22M00Cv1ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_5 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(64)});
  const GraphValue model22_m00_cv1_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(64)});
  const GraphValue model22_m00_cv2_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(128), Constant(3), Constant(3), Constant(64)},
                         WeightSlot::kModel22M00Cv2ConvWeight);
  const GraphValue model22_m00_cv2_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue model22_m00_cv2_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(128)}, WeightSlot::kModel22M00Cv2ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_10 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue model22_m00_cv2_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue residual_output = graph.DefineOutput(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)},
      DetectionP5QkvBindings::kResidualOutput);
  [[maybe_unused]] const GraphValue reserved_value_13 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(128), ResolutionDivided(32), ResolutionDivided(32)});
  const GraphValue model22_m01_attn_qkv_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(256), Constant(1), Constant(1), Constant(128)},
                         WeightSlot::kModel22M01AttnQkvConvWeight);
  const GraphValue model22_m01_attn_qkv_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue model22_m01_attn_qkv_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(256)}, WeightSlot::kModel22M01AttnQkvConvBias);
  const GraphValue transpose_1_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(256), ResolutionDivided(32), ResolutionDivided(32)});
  const GraphValue qkv_output = graph.DefineOutput(
      ValueType::kFp16, {Constant(1), Constant(2), Constant(128), ResolutionDividedSquared(32)},
      DetectionP5QkvBindings::kQkvOutput);
  if (!graph.DefinitionsValid()) return false;
  if (!graph.Convolution(
          main_input, model22_m00_cv1_conv_weight, model22_m00_cv1_conv_bias,
          model22_m00_cv1_conv_output, Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
          Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(128), Constant(64)}, 0))
    return false;
  if (!graph.Silu(model22_m00_cv1_conv_output, model22_m00_cv1_conv_activation, 0)) return false;
  if (!graph.Convolution(
          model22_m00_cv1_conv_activation, model22_m00_cv2_conv_weight, model22_m00_cv2_conv_bias,
          model22_m00_cv2_conv_output, Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
          Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(64), Constant(128)}, 0))
    return false;
  if (!graph.Silu(model22_m00_cv2_conv_output, model22_m00_cv2_conv_activation, 0)) return false;
  if (!graph.Add(main_input, model22_m00_cv2_conv_activation, residual_output, 0)) return false;
  if (!graph.Convolution(residual_output, model22_m01_attn_qkv_conv_weight,
                         model22_m01_attn_qkv_conv_bias, model22_m01_attn_qkv_conv_output,
                         Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
                         Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), Constant(128), Constant(256)}, 0))
    return false;
  if (!graph.Transpose(model22_m01_attn_qkv_conv_output, transpose_1_output,
                       {Constant(0), Constant(3), Constant(1), Constant(2)}, 0))
    return false;
  if (!graph.Reshape(transpose_1_output, qkv_output,
                     {Constant(1), Constant(2), Constant(128), ResolutionDividedSquared(32)}, 0))
    return false;
  return graph.Finish();
}

// Scales and transposes the detection-head queries for QK multiplication.
bool BuildDetectionP5AttentionScale(const GraphContext& context) {
  FixedGraphBuilder graph(context, GraphId::kDetectionP5AttentionScale,
                          DetectionP5AttentionScaleBindings::kCount);
  if (!graph.Begin()) return false;
  const GraphValue query_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), Constant(2), Constant(32), ResolutionDividedSquared(32)},
      DetectionP5AttentionScaleBindings::kQueryInput);
  const GraphValue scale_input = graph.DefineInput(ValueType::kFp16, {Constant(1)},
                                                   DetectionP5AttentionScaleBindings::kScaleInput);
  const GraphValue product_1 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(2), Constant(32), ResolutionDividedSquared(32)});
  const GraphValue scaled_query_output = graph.DefineOutput(
      ValueType::kFp16, {Constant(1), Constant(2), ResolutionDividedSquared(32), Constant(32)},
      DetectionP5AttentionScaleBindings::kScaledQueryOutput);
  if (!graph.DefinitionsValid()) return false;
  if (!graph.Multiply(query_input, scale_input, product_1, 0)) return false;
  if (!graph.Transpose(product_1, scaled_query_output,
                       {Constant(0), Constant(1), Constant(3), Constant(2)}, 0))
    return false;
  return graph.Finish();
}

// Computes and normalizes the detection-head attention matrices.
bool BuildDetectionP5AttentionMatrix(const GraphContext& context) {
  FixedGraphBuilder graph(context, GraphId::kDetectionP5AttentionMatrix,
                          DetectionP5AttentionMatrixBindings::kCount);
  if (!graph.Begin()) return false;
  const GraphValue reshape_1_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(2), Constant(32), ResolutionDividedSquared(32)});
  const GraphValue key_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), Constant(2), Constant(32), ResolutionDividedSquared(32)},
      DetectionP5AttentionMatrixBindings::kKeyInput);
  const GraphValue reshape_2_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(2), ResolutionDividedSquared(32), Constant(32)});
  const GraphValue query_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), Constant(2), ResolutionDividedSquared(32), Constant(32)},
      DetectionP5AttentionMatrixBindings::kQueryInput);
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
      DetectionP5AttentionMatrixBindings::kMatrixOutput);
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

// Applies attention, completes model.22, and produces box-distribution outputs
// at P3, P4, and P5 before concatenating and decoding them to four coordinates.
bool BuildDetectionP5AndBoxes(const GraphContext& context) {
  FixedGraphBuilder graph(context, GraphId::kDetectionP5AndBoxes,
                          DetectionP5AndBoxesBindings::kCount);
  if (!graph.Begin()) return false;
  [[maybe_unused]] const GraphValue reserved_value_0 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(64), ResolutionDivided(8), ResolutionDivided(8)});
  const GraphValue p3_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(64)},
      DetectionP5AndBoxesBindings::kP3Input);
  [[maybe_unused]] const GraphValue reserved_value_2 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(128), ResolutionDivided(16), ResolutionDivided(16)});
  const GraphValue p4_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)},
      DetectionP5AndBoxesBindings::kP4Input);
  [[maybe_unused]] const GraphValue reserved_value_4 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(128), ResolutionDivided(32), ResolutionDivided(32)});
  const GraphValue residual_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)},
      DetectionP5AndBoxesBindings::kResidualInput);
  [[maybe_unused]] const GraphValue reserved_value_6 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(128), ResolutionDivided(32), ResolutionDivided(32)});
  const GraphValue skip_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)},
      DetectionP5AndBoxesBindings::kSkipInput);
  [[maybe_unused]] const GraphValue reserved_value_8 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(128), ResolutionDivided(32), ResolutionDivided(32)});
  const GraphValue main_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)},
      DetectionP5AndBoxesBindings::kMainInput);
  const GraphValue model23_cv200_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(16), Constant(3), Constant(3), Constant(64)},
                         WeightSlot::kModel23Cv200ConvWeight);
  const GraphValue model23_cv200_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(16)});
  const GraphValue model23_cv200_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(16)}, WeightSlot::kModel23Cv200ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_13 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(16)});
  const GraphValue model23_cv200_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(16)});
  const GraphValue model23_cv201_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(16), Constant(3), Constant(3), Constant(16)},
                         WeightSlot::kModel23Cv201ConvWeight);
  const GraphValue model23_cv201_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(16)});
  const GraphValue model23_cv201_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(16)}, WeightSlot::kModel23Cv201ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_18 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(16)});
  const GraphValue model23_cv201_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(16)});
  const GraphValue model23_cv202_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(4), Constant(1), Constant(1), Constant(16)},
                         WeightSlot::kModel23Cv202Weight);
  const GraphValue model23_cv202_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(4)});
  const GraphValue model23_cv202_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(4)}, WeightSlot::kModel23Cv202Bias);
  const GraphValue transpose_1_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(4), ResolutionDivided(8), ResolutionDivided(8)});
  const GraphValue reshape_1_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(4), ResolutionDividedSquared(8)});
  const GraphValue model23_cv210_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(16), Constant(3), Constant(3), Constant(128)},
                         WeightSlot::kModel23Cv210ConvWeight);
  const GraphValue model23_cv210_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(16)});
  const GraphValue model23_cv210_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(16)}, WeightSlot::kModel23Cv210ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_28 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(16)});
  const GraphValue model23_cv210_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(16)});
  const GraphValue model23_cv211_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(16), Constant(3), Constant(3), Constant(16)},
                         WeightSlot::kModel23Cv211ConvWeight);
  const GraphValue model23_cv211_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(16)});
  const GraphValue model23_cv211_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(16)}, WeightSlot::kModel23Cv211ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_33 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(16)});
  const GraphValue model23_cv211_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(16)});
  const GraphValue model23_cv212_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(4), Constant(1), Constant(1), Constant(16)},
                         WeightSlot::kModel23Cv212Weight);
  const GraphValue model23_cv212_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(4)});
  const GraphValue model23_cv212_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(4)}, WeightSlot::kModel23Cv212Bias);
  const GraphValue transpose_2_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(4), ResolutionDivided(16), ResolutionDivided(16)});
  const GraphValue reshape_2_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(4), ResolutionDividedSquared(16)});
  const GraphValue reshape_3_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(2), Constant(64), ResolutionDividedSquared(32)});
  const GraphValue value_copy_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), Constant(2), Constant(64), ResolutionDividedSquared(32)},
      DetectionP5AndBoxesBindings::kValueCopyInput);
  const GraphValue reshape_4_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(128), ResolutionDivided(32), ResolutionDivided(32)});
  const GraphValue value_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), Constant(2), Constant(64), ResolutionDividedSquared(32)},
      DetectionP5AndBoxesBindings::kValueInput);
  const GraphValue transpose_3_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue model22_m01_attn_pe_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(1), Constant(3), Constant(3), Constant(128)},
                         WeightSlot::kModel22M01AttnPeConvWeight);
  const GraphValue model22_m01_attn_pe_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue model22_m01_attn_pe_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(128)}, WeightSlot::kModel22M01AttnPeConvBias);
  const GraphValue reshape_5_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(2), ResolutionDividedSquared(32), ResolutionDividedSquared(32)});
  const GraphValue matrix_input = graph.DefineInput(
      ValueType::kFp16,
      {Constant(1), Constant(2), ResolutionDividedSquared(32), ResolutionDividedSquared(32)},
      DetectionP5AndBoxesBindings::kMatrixInput);
  const GraphValue matrix_product_1 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(2), Constant(64), ResolutionDividedSquared(32)});
  const GraphValue reshape_6_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(2), Constant(64), ResolutionDividedSquared(32)});
  const GraphValue reshape_7_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(128), ResolutionDivided(32), ResolutionDivided(32)});
  const GraphValue transpose_4_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue residual_sum_1 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue model22_m01_attn_proj_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(128), Constant(1), Constant(1), Constant(128)},
                         WeightSlot::kModel22M01AttnProjConvWeight);
  const GraphValue model22_m01_attn_proj_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue model22_m01_attn_proj_conv_bias = graph.DefineWeight(
      ValueType::kFp32, {Constant(128)}, WeightSlot::kModel22M01AttnProjConvBias);
  const GraphValue residual_sum_2 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue model22_m01_ffn0_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(256), Constant(1), Constant(1), Constant(128)},
                         WeightSlot::kModel22M01Ffn0ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_60 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue model22_m01_ffn0_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(256)}, WeightSlot::kModel22M01Ffn0ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_62 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue model22_m01_ffn0_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue model22_m01_ffn1_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(128), Constant(1), Constant(1), Constant(256)},
                         WeightSlot::kModel22M01Ffn1ConvWeight);
  const GraphValue model22_m01_ffn1_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue model22_m01_ffn1_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(128)}, WeightSlot::kModel22M01Ffn1ConvBias);
  const GraphValue residual_sum_3 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(128)});
  const GraphValue concatenation_1 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(384)});
  const GraphValue model22_cv2_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(256), Constant(1), Constant(1), Constant(384)},
                         WeightSlot::kModel22Cv2ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_70 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue model22_cv2_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(256)}, WeightSlot::kModel22Cv2ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_72 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue p5_output = graph.DefineOutput(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)},
      DetectionP5AndBoxesBindings::kP5Output);
  [[maybe_unused]] const GraphValue reserved_value_74 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(256), ResolutionDivided(32), ResolutionDivided(32)});
  const GraphValue model23_cv220_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(16), Constant(3), Constant(3), Constant(256)},
                         WeightSlot::kModel23Cv220ConvWeight);
  const GraphValue model23_cv220_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(16)});
  const GraphValue model23_cv220_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(16)}, WeightSlot::kModel23Cv220ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_78 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(16)});
  const GraphValue model23_cv220_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(16)});
  const GraphValue model23_cv221_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(16), Constant(3), Constant(3), Constant(16)},
                         WeightSlot::kModel23Cv221ConvWeight);
  const GraphValue model23_cv221_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(16)});
  const GraphValue model23_cv221_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(16)}, WeightSlot::kModel23Cv221ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_83 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(16)});
  const GraphValue model23_cv221_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(16)});
  const GraphValue model23_cv222_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(4), Constant(1), Constant(1), Constant(16)},
                         WeightSlot::kModel23Cv222Weight);
  const GraphValue model23_cv222_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(4)});
  const GraphValue model23_cv222_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(4)}, WeightSlot::kModel23Cv222Bias);
  const GraphValue transpose_5_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(4), ResolutionDivided(32), ResolutionDivided(32)});
  const GraphValue reshape_8_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(4), ResolutionDividedSquared(32)});
  const GraphValue box_regression_output =
      graph.DefineOutput(ValueType::kFp16, {Constant(1), Constant(4), DetectionPositions()},
                         DetectionP5AndBoxesBindings::kBoxRegressionOutput);
  if (!graph.DefinitionsValid()) return false;
  if (!graph.Convolution(
          p3_input, model23_cv200_conv_weight, model23_cv200_conv_bias, model23_cv200_conv_output,
          Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
          Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(64), Constant(16)}, 0))
    return false;
  if (!graph.Silu(model23_cv200_conv_output, model23_cv200_conv_activation, 0)) return false;
  if (!graph.Convolution(
          model23_cv200_conv_activation, model23_cv201_conv_weight, model23_cv201_conv_bias,
          model23_cv201_conv_output, Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
          Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(16), Constant(16)}, 0))
    return false;
  if (!graph.Silu(model23_cv201_conv_output, model23_cv201_conv_activation, 0)) return false;
  if (!graph.Convolution(
          model23_cv201_conv_activation, model23_cv202_weight, model23_cv202_bias,
          model23_cv202_output, Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(16), Constant(4)}, 0))
    return false;
  if (!graph.Transpose(model23_cv202_output, transpose_1_output,
                       {Constant(0), Constant(3), Constant(1), Constant(2)}, 0))
    return false;
  if (!graph.Reshape(transpose_1_output, reshape_1_output, {Constant(1), Constant(4), Constant(0)},
                     0))
    return false;
  if (!graph.Convolution(
          p4_input, model23_cv210_conv_weight, model23_cv210_conv_bias, model23_cv210_conv_output,
          Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
          Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(128), Constant(16)}, 0))
    return false;
  if (!graph.Silu(model23_cv210_conv_output, model23_cv210_conv_activation, 0)) return false;
  if (!graph.Convolution(
          model23_cv210_conv_activation, model23_cv211_conv_weight, model23_cv211_conv_bias,
          model23_cv211_conv_output, Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
          Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(16), Constant(16)}, 0))
    return false;
  if (!graph.Silu(model23_cv211_conv_output, model23_cv211_conv_activation, 0)) return false;
  if (!graph.Convolution(
          model23_cv211_conv_activation, model23_cv212_weight, model23_cv212_bias,
          model23_cv212_output, Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(16), Constant(4)}, 0))
    return false;
  if (!graph.Transpose(model23_cv212_output, transpose_2_output,
                       {Constant(0), Constant(3), Constant(1), Constant(2)}, 0))
    return false;
  if (!graph.Reshape(transpose_2_output, reshape_2_output, {Constant(1), Constant(4), Constant(0)},
                     0))
    return false;
  if (!graph.Reshape(value_copy_input, reshape_3_output,
                     {Constant(2), Constant(64), ResolutionDividedSquared(32)}, 0))
    return false;
  if (!graph.Reshape(value_input, reshape_4_output,
                     {Constant(1), Constant(128), ResolutionDivided(32), ResolutionDivided(32)}, 0))
    return false;
  if (!graph.Transpose(reshape_4_output, transpose_3_output,
                       {Constant(0), Constant(2), Constant(3), Constant(1)}, 0))
    return false;
  if (!graph.DepthwiseConvolution(
          transpose_3_output, model22_m01_attn_pe_conv_weight, model22_m01_attn_pe_conv_bias,
          model22_m01_attn_pe_conv_output,
          Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
          Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(128), Constant(1), Constant(1)}, 0))
    return false;
  if (!graph.Reshape(matrix_input, reshape_5_output,
                     {Constant(2), ResolutionDividedSquared(32), ResolutionDividedSquared(32)}, 0))
    return false;
  if (!graph.BatchMatrixMultiply(reshape_3_output, reshape_5_output, matrix_product_1, 0))
    return false;
  if (!graph.Reshape(matrix_product_1, reshape_6_output,
                     {Constant(1), Constant(2), Constant(64), ResolutionDividedSquared(32)}, 0))
    return false;
  if (!graph.Reshape(reshape_6_output, reshape_7_output,
                     {Constant(1), Constant(128), ResolutionDivided(32), ResolutionDivided(32)}, 0))
    return false;
  if (!graph.Transpose(reshape_7_output, transpose_4_output,
                       {Constant(0), Constant(2), Constant(3), Constant(1)}, 0))
    return false;
  if (!graph.Add(transpose_4_output, model22_m01_attn_pe_conv_output, residual_sum_1, 0))
    return false;
  if (!graph.Convolution(residual_sum_1, model22_m01_attn_proj_conv_weight,
                         model22_m01_attn_proj_conv_bias, model22_m01_attn_proj_conv_output,
                         Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
                         Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), Constant(128), Constant(128)}, 0))
    return false;
  if (!graph.Add(residual_input, model22_m01_attn_proj_conv_output, residual_sum_2, 0))
    return false;
  if (!graph.Convolution(
          residual_sum_2, model22_m01_ffn0_conv_weight, model22_m01_ffn0_conv_bias,
          model22_m01_ffn0_conv_output, Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(128), Constant(256)},
          XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Convolution(
          model22_m01_ffn0_conv_output, model22_m01_ffn1_conv_weight, model22_m01_ffn1_conv_bias,
          model22_m01_ffn1_conv_output, Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(256), Constant(128)}, 0))
    return false;
  if (!graph.Add(residual_sum_2, model22_m01_ffn1_conv_output, residual_sum_3, 0)) return false;
  if (!graph.Concatenate({skip_input, main_input, residual_sum_3}, concatenation_1, Constant(3), 0))
    return false;
  if (!graph.Convolution(concatenation_1, model22_cv2_conv_weight, model22_cv2_conv_bias, p5_output,
                         Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
                         Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), Constant(384), Constant(256)},
                         XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Convolution(
          p5_output, model23_cv220_conv_weight, model23_cv220_conv_bias, model23_cv220_conv_output,
          Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
          Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(256), Constant(16)}, 0))
    return false;
  if (!graph.Silu(model23_cv220_conv_output, model23_cv220_conv_activation, 0)) return false;
  if (!graph.Convolution(
          model23_cv220_conv_activation, model23_cv221_conv_weight, model23_cv221_conv_bias,
          model23_cv221_conv_output, Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
          Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(16), Constant(16)}, 0))
    return false;
  if (!graph.Silu(model23_cv221_conv_output, model23_cv221_conv_activation, 0)) return false;
  if (!graph.Convolution(
          model23_cv221_conv_activation, model23_cv222_weight, model23_cv222_bias,
          model23_cv222_output, Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
          Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(1), Constant(16), Constant(4)}, 0))
    return false;
  if (!graph.Transpose(model23_cv222_output, transpose_5_output,
                       {Constant(0), Constant(3), Constant(1), Constant(2)}, 0))
    return false;
  if (!graph.Reshape(transpose_5_output, reshape_8_output, {Constant(1), Constant(4), Constant(0)},
                     0))
    return false;
  if (!graph.Concatenate({reshape_1_output, reshape_2_output, reshape_8_output},
                         box_regression_output, Constant(2), 0))
    return false;
  return graph.Finish();
}

// Produces class logits at all three scales, concatenates them with decoded box
// coordinates, and converts the combined head output to the public layout.
bool BuildClassificationAndDecode(const GraphContext& context) {
  FixedGraphBuilder graph(context, GraphId::kClassificationAndDecode,
                          ClassificationAndDecodeBindings::kCount);
  if (!graph.Begin()) return false;
  [[maybe_unused]] const GraphValue reserved_value_0 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(64), ResolutionDivided(8), ResolutionDivided(8)});
  const GraphValue p3_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(64)},
      ClassificationAndDecodeBindings::kP3Input);
  [[maybe_unused]] const GraphValue reserved_value_2 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(128), ResolutionDivided(16), ResolutionDivided(16)});
  const GraphValue p4_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)},
      ClassificationAndDecodeBindings::kP4Input);
  [[maybe_unused]] const GraphValue reserved_value_4 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), Constant(256), ResolutionDivided(32), ResolutionDivided(32)});
  const GraphValue p5_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)},
      ClassificationAndDecodeBindings::kP5Input);
  const GraphValue model23_cv3000_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(1), Constant(3), Constant(3), Constant(64)},
                         WeightSlot::kModel23Cv3000ConvWeight);
  const GraphValue model23_cv3000_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(64)});
  const GraphValue model23_cv3000_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(64)}, WeightSlot::kModel23Cv3000ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_9 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(64)});
  const GraphValue model23_cv3000_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), Constant(64)});
  const GraphValue model23_cv3001_conv_weight = graph.DefineWeight(
      ValueType::kFp32, {ClassificationWidth(), Constant(1), Constant(1), Constant(64)},
      WeightSlot::kModel23Cv3001ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_12 = graph.DefineIntermediate(
      ValueType::kFp16,
      {Constant(1), ResolutionDivided(8), ResolutionDivided(8), ClassificationWidth()});
  const GraphValue model23_cv3001_conv_bias = graph.DefineWeight(
      ValueType::kFp32, {ClassificationWidth()}, WeightSlot::kModel23Cv3001ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_14 = graph.DefineIntermediate(
      ValueType::kFp16,
      {Constant(1), ResolutionDivided(8), ResolutionDivided(8), ClassificationWidth()});
  const GraphValue model23_cv3001_conv_output = graph.DefineIntermediate(
      ValueType::kFp16,
      {Constant(1), ResolutionDivided(8), ResolutionDivided(8), ClassificationWidth()});
  const GraphValue model23_cv3010_conv_weight = graph.DefineWeight(
      ValueType::kFp32, {Constant(1), Constant(3), Constant(3), ClassificationWidth()},
      WeightSlot::kModel23Cv3010ConvWeight);
  const GraphValue model23_cv3010_conv_output = graph.DefineIntermediate(
      ValueType::kFp16,
      {Constant(1), ResolutionDivided(8), ResolutionDivided(8), ClassificationWidth()});
  const GraphValue model23_cv3010_conv_bias = graph.DefineWeight(
      ValueType::kFp32, {ClassificationWidth()}, WeightSlot::kModel23Cv3010ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_19 = graph.DefineIntermediate(
      ValueType::kFp16,
      {Constant(1), ResolutionDivided(8), ResolutionDivided(8), ClassificationWidth()});
  const GraphValue model23_cv3010_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16,
      {Constant(1), ResolutionDivided(8), ResolutionDivided(8), ClassificationWidth()});
  const GraphValue model23_cv3011_conv_weight = graph.DefineWeight(
      ValueType::kFp32, {ClassificationWidth(), Constant(1), Constant(1), ClassificationWidth()},
      WeightSlot::kModel23Cv3011ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_22 = graph.DefineIntermediate(
      ValueType::kFp16,
      {Constant(1), ResolutionDivided(8), ResolutionDivided(8), ClassificationWidth()});
  const GraphValue model23_cv3011_conv_bias = graph.DefineWeight(
      ValueType::kFp32, {ClassificationWidth()}, WeightSlot::kModel23Cv3011ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_24 = graph.DefineIntermediate(
      ValueType::kFp16,
      {Constant(1), ResolutionDivided(8), ResolutionDivided(8), ClassificationWidth()});
  const GraphValue model23_cv3011_conv_output = graph.DefineIntermediate(
      ValueType::kFp16,
      {Constant(1), ResolutionDivided(8), ResolutionDivided(8), ClassificationWidth()});
  const GraphValue model23_cv302_weight = graph.DefineWeight(
      ValueType::kFp32, {ClassCount(), Constant(1), Constant(1), ClassificationWidth()},
      WeightSlot::kModel23Cv302Weight);
  const GraphValue model23_cv302_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(8), ResolutionDivided(8), ClassCount()});
  const GraphValue model23_cv302_bias =
      graph.DefineWeight(ValueType::kFp32, {ClassCount()}, WeightSlot::kModel23Cv302Bias);
  const GraphValue transpose_1_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ClassCount(), ResolutionDivided(8), ResolutionDivided(8)});
  const GraphValue reshape_1_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ClassCount(), ResolutionDividedSquared(8)});
  const GraphValue model23_cv3100_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(1), Constant(3), Constant(3), Constant(128)},
                         WeightSlot::kModel23Cv3100ConvWeight);
  const GraphValue model23_cv3100_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)});
  const GraphValue model23_cv3100_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(128)}, WeightSlot::kModel23Cv3100ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_34 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)});
  const GraphValue model23_cv3100_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), Constant(128)});
  const GraphValue model23_cv3101_conv_weight = graph.DefineWeight(
      ValueType::kFp32, {ClassificationWidth(), Constant(1), Constant(1), Constant(128)},
      WeightSlot::kModel23Cv3101ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_37 = graph.DefineIntermediate(
      ValueType::kFp16,
      {Constant(1), ResolutionDivided(16), ResolutionDivided(16), ClassificationWidth()});
  const GraphValue model23_cv3101_conv_bias = graph.DefineWeight(
      ValueType::kFp32, {ClassificationWidth()}, WeightSlot::kModel23Cv3101ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_39 = graph.DefineIntermediate(
      ValueType::kFp16,
      {Constant(1), ResolutionDivided(16), ResolutionDivided(16), ClassificationWidth()});
  const GraphValue model23_cv3101_conv_output = graph.DefineIntermediate(
      ValueType::kFp16,
      {Constant(1), ResolutionDivided(16), ResolutionDivided(16), ClassificationWidth()});
  const GraphValue model23_cv3110_conv_weight = graph.DefineWeight(
      ValueType::kFp32, {Constant(1), Constant(3), Constant(3), ClassificationWidth()},
      WeightSlot::kModel23Cv3110ConvWeight);
  const GraphValue model23_cv3110_conv_output = graph.DefineIntermediate(
      ValueType::kFp16,
      {Constant(1), ResolutionDivided(16), ResolutionDivided(16), ClassificationWidth()});
  const GraphValue model23_cv3110_conv_bias = graph.DefineWeight(
      ValueType::kFp32, {ClassificationWidth()}, WeightSlot::kModel23Cv3110ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_44 = graph.DefineIntermediate(
      ValueType::kFp16,
      {Constant(1), ResolutionDivided(16), ResolutionDivided(16), ClassificationWidth()});
  const GraphValue model23_cv3110_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16,
      {Constant(1), ResolutionDivided(16), ResolutionDivided(16), ClassificationWidth()});
  const GraphValue model23_cv3111_conv_weight = graph.DefineWeight(
      ValueType::kFp32, {ClassificationWidth(), Constant(1), Constant(1), ClassificationWidth()},
      WeightSlot::kModel23Cv3111ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_47 = graph.DefineIntermediate(
      ValueType::kFp16,
      {Constant(1), ResolutionDivided(16), ResolutionDivided(16), ClassificationWidth()});
  const GraphValue model23_cv3111_conv_bias = graph.DefineWeight(
      ValueType::kFp32, {ClassificationWidth()}, WeightSlot::kModel23Cv3111ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_49 = graph.DefineIntermediate(
      ValueType::kFp16,
      {Constant(1), ResolutionDivided(16), ResolutionDivided(16), ClassificationWidth()});
  const GraphValue model23_cv3111_conv_output = graph.DefineIntermediate(
      ValueType::kFp16,
      {Constant(1), ResolutionDivided(16), ResolutionDivided(16), ClassificationWidth()});
  const GraphValue model23_cv312_weight = graph.DefineWeight(
      ValueType::kFp32, {ClassCount(), Constant(1), Constant(1), ClassificationWidth()},
      WeightSlot::kModel23Cv312Weight);
  const GraphValue model23_cv312_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(16), ResolutionDivided(16), ClassCount()});
  const GraphValue model23_cv312_bias =
      graph.DefineWeight(ValueType::kFp32, {ClassCount()}, WeightSlot::kModel23Cv312Bias);
  const GraphValue transpose_2_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ClassCount(), ResolutionDivided(16), ResolutionDivided(16)});
  const GraphValue reshape_2_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ClassCount(), ResolutionDividedSquared(16)});
  const GraphValue model23_cv3200_conv_weight =
      graph.DefineWeight(ValueType::kFp32, {Constant(1), Constant(3), Constant(3), Constant(256)},
                         WeightSlot::kModel23Cv3200ConvWeight);
  const GraphValue model23_cv3200_conv_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue model23_cv3200_conv_bias =
      graph.DefineWeight(ValueType::kFp32, {Constant(256)}, WeightSlot::kModel23Cv3200ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_59 = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue model23_cv3200_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), Constant(256)});
  const GraphValue model23_cv3201_conv_weight = graph.DefineWeight(
      ValueType::kFp32, {ClassificationWidth(), Constant(1), Constant(1), Constant(256)},
      WeightSlot::kModel23Cv3201ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_62 = graph.DefineIntermediate(
      ValueType::kFp16,
      {Constant(1), ResolutionDivided(32), ResolutionDivided(32), ClassificationWidth()});
  const GraphValue model23_cv3201_conv_bias = graph.DefineWeight(
      ValueType::kFp32, {ClassificationWidth()}, WeightSlot::kModel23Cv3201ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_64 = graph.DefineIntermediate(
      ValueType::kFp16,
      {Constant(1), ResolutionDivided(32), ResolutionDivided(32), ClassificationWidth()});
  const GraphValue model23_cv3201_conv_output = graph.DefineIntermediate(
      ValueType::kFp16,
      {Constant(1), ResolutionDivided(32), ResolutionDivided(32), ClassificationWidth()});
  const GraphValue model23_cv3210_conv_weight = graph.DefineWeight(
      ValueType::kFp32, {Constant(1), Constant(3), Constant(3), ClassificationWidth()},
      WeightSlot::kModel23Cv3210ConvWeight);
  const GraphValue model23_cv3210_conv_output = graph.DefineIntermediate(
      ValueType::kFp16,
      {Constant(1), ResolutionDivided(32), ResolutionDivided(32), ClassificationWidth()});
  const GraphValue model23_cv3210_conv_bias = graph.DefineWeight(
      ValueType::kFp32, {ClassificationWidth()}, WeightSlot::kModel23Cv3210ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_69 = graph.DefineIntermediate(
      ValueType::kFp16,
      {Constant(1), ResolutionDivided(32), ResolutionDivided(32), ClassificationWidth()});
  const GraphValue model23_cv3210_conv_activation = graph.DefineIntermediate(
      ValueType::kFp16,
      {Constant(1), ResolutionDivided(32), ResolutionDivided(32), ClassificationWidth()});
  const GraphValue model23_cv3211_conv_weight = graph.DefineWeight(
      ValueType::kFp32, {ClassificationWidth(), Constant(1), Constant(1), ClassificationWidth()},
      WeightSlot::kModel23Cv3211ConvWeight);
  [[maybe_unused]] const GraphValue reserved_value_72 = graph.DefineIntermediate(
      ValueType::kFp16,
      {Constant(1), ResolutionDivided(32), ResolutionDivided(32), ClassificationWidth()});
  const GraphValue model23_cv3211_conv_bias = graph.DefineWeight(
      ValueType::kFp32, {ClassificationWidth()}, WeightSlot::kModel23Cv3211ConvBias);
  [[maybe_unused]] const GraphValue reserved_value_74 = graph.DefineIntermediate(
      ValueType::kFp16,
      {Constant(1), ResolutionDivided(32), ResolutionDivided(32), ClassificationWidth()});
  const GraphValue model23_cv3211_conv_output = graph.DefineIntermediate(
      ValueType::kFp16,
      {Constant(1), ResolutionDivided(32), ResolutionDivided(32), ClassificationWidth()});
  const GraphValue model23_cv322_weight = graph.DefineWeight(
      ValueType::kFp32, {ClassCount(), Constant(1), Constant(1), ClassificationWidth()},
      WeightSlot::kModel23Cv322Weight);
  const GraphValue model23_cv322_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ResolutionDivided(32), ResolutionDivided(32), ClassCount()});
  const GraphValue model23_cv322_bias =
      graph.DefineWeight(ValueType::kFp32, {ClassCount()}, WeightSlot::kModel23Cv322Bias);
  const GraphValue transpose_3_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ClassCount(), ResolutionDivided(32), ResolutionDivided(32)});
  const GraphValue reshape_3_output = graph.DefineIntermediate(
      ValueType::kFp16, {Constant(1), ClassCount(), ResolutionDividedSquared(32)});
  const GraphValue concatenation_1 =
      graph.DefineIntermediate(ValueType::kFp16, {Constant(1), ClassCount(), DetectionPositions()});
  const GraphValue anchors_input =
      graph.DefineInput(ValueType::kFp16, {Constant(1), Constant(2), DetectionPositions()},
                        ClassificationAndDecodeBindings::kAnchorsInput);
  const GraphValue box_lower_input =
      graph.DefineInput(ValueType::kFp16, {Constant(1), Constant(2), DetectionPositions()},
                        ClassificationAndDecodeBindings::kBoxLowerInput);
  const GraphValue difference_1 =
      graph.DefineIntermediate(ValueType::kFp16, {Constant(1), Constant(2), DetectionPositions()});
  const GraphValue box_upper_input =
      graph.DefineInput(ValueType::kFp16, {Constant(1), Constant(2), DetectionPositions()},
                        ClassificationAndDecodeBindings::kBoxUpperInput);
  const GraphValue residual_sum_1 =
      graph.DefineIntermediate(ValueType::kFp16, {Constant(1), Constant(2), DetectionPositions()});
  const GraphValue residual_sum_2 =
      graph.DefineIntermediate(ValueType::kFp16, {Constant(1), Constant(2), DetectionPositions()});
  const GraphValue quotient_1 =
      graph.DefineIntermediate(ValueType::kFp16, {Constant(1), Constant(2), DetectionPositions()});
  const GraphValue center_divisor_input = graph.DefineInput(
      ValueType::kFp16, {Constant(1)}, ClassificationAndDecodeBindings::kCenterDivisorInput);
  const GraphValue difference_2 =
      graph.DefineIntermediate(ValueType::kFp16, {Constant(1), Constant(2), DetectionPositions()});
  const GraphValue concatenation_2 =
      graph.DefineIntermediate(ValueType::kFp16, {Constant(1), Constant(4), DetectionPositions()});
  const GraphValue strides_input =
      graph.DefineInput(ValueType::kFp16, {Constant(1), DetectionPositions()},
                        ClassificationAndDecodeBindings::kStridesInput);
  const GraphValue product_1 =
      graph.DefineIntermediate(ValueType::kFp16, {Constant(1), Constant(4), DetectionPositions()});
  const GraphValue sigmoid_1_output =
      graph.DefineIntermediate(ValueType::kFp16, {Constant(1), ClassCount(), DetectionPositions()});
  const GraphValue detections_output =
      graph.DefineOutput(ValueType::kFp16, {Constant(1), DetectionChannels(), DetectionPositions()},
                         ClassificationAndDecodeBindings::kDetectionsOutput);
  if (!graph.DefinitionsValid()) return false;
  if (!graph.DepthwiseConvolution(
          p3_input, model23_cv3000_conv_weight, model23_cv3000_conv_bias,
          model23_cv3000_conv_output, Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
          Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(64), Constant(1), Constant(1)}, 0))
    return false;
  if (!graph.Silu(model23_cv3000_conv_output, model23_cv3000_conv_activation, 0)) return false;
  if (!graph.Convolution(model23_cv3000_conv_activation, model23_cv3001_conv_weight,
                         model23_cv3001_conv_bias, model23_cv3001_conv_output,
                         Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
                         Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), Constant(64), ClassificationWidth()},
                         XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.DepthwiseConvolution(
          model23_cv3001_conv_output, model23_cv3010_conv_weight, model23_cv3010_conv_bias,
          model23_cv3010_conv_output, Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
          Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)},
          Groups{ClassificationWidth(), Constant(1), Constant(1)}, 0))
    return false;
  if (!graph.Silu(model23_cv3010_conv_output, model23_cv3010_conv_activation, 0)) return false;
  if (!graph.Convolution(model23_cv3010_conv_activation, model23_cv3011_conv_weight,
                         model23_cv3011_conv_bias, model23_cv3011_conv_output,
                         Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
                         Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), ClassificationWidth(), ClassificationWidth()},
                         XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Convolution(model23_cv3011_conv_output, model23_cv302_weight, model23_cv302_bias,
                         model23_cv302_output,
                         Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
                         Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), ClassificationWidth(), ClassCount()}, 0))
    return false;
  if (!graph.Transpose(model23_cv302_output, transpose_1_output,
                       {Constant(0), Constant(3), Constant(1), Constant(2)}, 0))
    return false;
  if (!graph.Reshape(transpose_1_output, reshape_1_output, {Constant(1), ClassCount(), Constant(0)},
                     0))
    return false;
  if (!graph.DepthwiseConvolution(
          p4_input, model23_cv3100_conv_weight, model23_cv3100_conv_bias,
          model23_cv3100_conv_output, Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
          Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(128), Constant(1), Constant(1)}, 0))
    return false;
  if (!graph.Silu(model23_cv3100_conv_output, model23_cv3100_conv_activation, 0)) return false;
  if (!graph.Convolution(model23_cv3100_conv_activation, model23_cv3101_conv_weight,
                         model23_cv3101_conv_bias, model23_cv3101_conv_output,
                         Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
                         Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), Constant(128), ClassificationWidth()},
                         XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.DepthwiseConvolution(
          model23_cv3101_conv_output, model23_cv3110_conv_weight, model23_cv3110_conv_bias,
          model23_cv3110_conv_output, Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
          Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)},
          Groups{ClassificationWidth(), Constant(1), Constant(1)}, 0))
    return false;
  if (!graph.Silu(model23_cv3110_conv_output, model23_cv3110_conv_activation, 0)) return false;
  if (!graph.Convolution(model23_cv3110_conv_activation, model23_cv3111_conv_weight,
                         model23_cv3111_conv_bias, model23_cv3111_conv_output,
                         Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
                         Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), ClassificationWidth(), ClassificationWidth()},
                         XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Convolution(model23_cv3111_conv_output, model23_cv312_weight, model23_cv312_bias,
                         model23_cv312_output,
                         Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
                         Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), ClassificationWidth(), ClassCount()}, 0))
    return false;
  if (!graph.Transpose(model23_cv312_output, transpose_2_output,
                       {Constant(0), Constant(3), Constant(1), Constant(2)}, 0))
    return false;
  if (!graph.Reshape(transpose_2_output, reshape_2_output, {Constant(1), ClassCount(), Constant(0)},
                     0))
    return false;
  if (!graph.DepthwiseConvolution(
          p5_input, model23_cv3200_conv_weight, model23_cv3200_conv_bias,
          model23_cv3200_conv_output, Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
          Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)}, Groups{Constant(256), Constant(1), Constant(1)}, 0))
    return false;
  if (!graph.Silu(model23_cv3200_conv_output, model23_cv3200_conv_activation, 0)) return false;
  if (!graph.Convolution(model23_cv3200_conv_activation, model23_cv3201_conv_weight,
                         model23_cv3201_conv_bias, model23_cv3201_conv_output,
                         Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
                         Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), Constant(256), ClassificationWidth()},
                         XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.DepthwiseConvolution(
          model23_cv3201_conv_output, model23_cv3210_conv_weight, model23_cv3210_conv_bias,
          model23_cv3210_conv_output, Padding{Constant(1), Constant(1), Constant(1), Constant(1)},
          Kernel{Constant(3), Constant(3)}, Stride{Constant(1), Constant(1)},
          Dilation{Constant(1), Constant(1)},
          Groups{ClassificationWidth(), Constant(1), Constant(1)}, 0))
    return false;
  if (!graph.Silu(model23_cv3210_conv_output, model23_cv3210_conv_activation, 0)) return false;
  if (!graph.Convolution(model23_cv3210_conv_activation, model23_cv3211_conv_weight,
                         model23_cv3211_conv_bias, model23_cv3211_conv_output,
                         Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
                         Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), ClassificationWidth(), ClassificationWidth()},
                         XNN_FLAG_YOLO26_GEMM_SILU_FINAL_STORE))
    return false;
  if (!graph.Convolution(model23_cv3211_conv_output, model23_cv322_weight, model23_cv322_bias,
                         model23_cv322_output,
                         Padding{Constant(0), Constant(0), Constant(0), Constant(0)},
                         Kernel{Constant(1), Constant(1)}, Stride{Constant(1), Constant(1)},
                         Dilation{Constant(1), Constant(1)},
                         Groups{Constant(1), ClassificationWidth(), ClassCount()}, 0))
    return false;
  if (!graph.Transpose(model23_cv322_output, transpose_3_output,
                       {Constant(0), Constant(3), Constant(1), Constant(2)}, 0))
    return false;
  if (!graph.Reshape(transpose_3_output, reshape_3_output, {Constant(1), ClassCount(), Constant(0)},
                     0))
    return false;
  if (!graph.Concatenate({reshape_1_output, reshape_2_output, reshape_3_output}, concatenation_1,
                         Constant(2), 0))
    return false;
  if (!graph.Subtract(anchors_input, box_lower_input, difference_1, 0)) return false;
  if (!graph.Add(anchors_input, box_upper_input, residual_sum_1, 0)) return false;
  if (!graph.Add(difference_1, residual_sum_1, residual_sum_2, 0)) return false;
  if (!graph.Divide(residual_sum_2, center_divisor_input, quotient_1, 0)) return false;
  if (!graph.Subtract(residual_sum_1, difference_1, difference_2, 0)) return false;
  if (!graph.Concatenate({quotient_1, difference_2}, concatenation_2, Constant(1), 0)) return false;
  if (!graph.Multiply(concatenation_2, strides_input, product_1, 0)) return false;
  if (!graph.Sigmoid(concatenation_1, sigmoid_1_output, 0)) return false;
  if (!graph.Concatenate({product_1, sigmoid_1_output}, detections_output, Constant(1), 0))
    return false;
  return graph.Finish();
}

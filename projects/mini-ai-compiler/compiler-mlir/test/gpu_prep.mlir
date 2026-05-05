module {
  func.func @prep_linear_relu(%arg0: tensor<2x4xf32>) -> tensor<2x8xf32> {
    %w = "mini.constant"() {value = dense<0.0> : tensor<8x4xf32>} : () -> tensor<8x4xf32>
    %b = "mini.constant"() {value = dense<0.0> : tensor<8xf32>} : () -> tensor<8xf32>
    %0 = "mini.linear"(%arg0, %w, %b) : (tensor<2x4xf32>, tensor<8x4xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    %1 = "mini.relu"(%0) : (tensor<2x8xf32>) -> tensor<2x8xf32>
    return %1 : tensor<2x8xf32>
  }
}

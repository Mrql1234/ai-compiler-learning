module {
  func.func @compute(%arg0: tensor<2x4xf32>) -> tensor<2x8xf32> {
    %w = "mini.constant"() {value = dense<0.0> : tensor<8x4xf32>} : () -> tensor<8x4xf32>
    %b = "mini.constant"() {value = dense<0.0> : tensor<8xf32>} : () -> tensor<8xf32>
    %0 = "mini.linear"(%arg0, %w, %b) : (tensor<2x4xf32>, tensor<8x4xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    %1 = "mini.relu"(%0) : (tensor<2x8xf32>) -> tensor<2x8xf32>
    return %1 : tensor<2x8xf32>
  }

  func.func @run() -> f32 attributes {llvm.emit_c_interface} {
    %input = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>
    %result = call @compute(%input) : (tensor<2x4xf32>) -> tensor<2x8xf32>
    %c0 = arith.constant 0 : index
    %first = tensor.extract %result[%c0, %c0] : tensor<2x8xf32>
    return %first : f32
  }
}

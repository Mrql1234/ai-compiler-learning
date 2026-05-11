module {
  func.func @compute(%arg0: tensor<2x3xf32>) -> tensor<2x2xf32> {
    %weight = "mini.constant"() {value = dense<[
      [1.0, -2.0, 0.5],
      [0.25, 0.0, -0.25]
    ]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
    %bias = "mini.constant"() {value = dense<[0.5, -0.5]> : tensor<2xf32>} : () -> tensor<2xf32>
    %0 = "mini.linear"(%arg0, %weight, %bias) : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }

  func.func @run() -> f32 attributes {llvm.emit_c_interface} {
    %input = arith.constant dense<[[1.0, 2.0, -1.0], [0.0, 0.0, 0.0]]> : tensor<2x3xf32>
    %result = call @compute(%input) : (tensor<2x3xf32>) -> tensor<2x2xf32>
    %c0 = arith.constant 0 : index
    %first = tensor.extract %result[%c0, %c0] : tensor<2x2xf32>
    return %first : f32
  }
}

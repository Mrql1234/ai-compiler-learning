// RUN: %mini_compiler_opt --mini-lower-to-linalg %s | FileCheck %s

module {
  func.func @lower_linear_relu(%arg0: tensor<2x4xf32>) -> tensor<2x8xf32> {
    %w = "mini.constant"() {value = dense<0.0> : tensor<8x4xf32>} : () -> tensor<8x4xf32>
    %b = "mini.constant"() {value = dense<0.0> : tensor<8xf32>} : () -> tensor<8xf32>
    %0 = "mini.linear"(%arg0, %w, %b) : (tensor<2x4xf32>, tensor<8x4xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    %1 = "mini.relu"(%0) : (tensor<2x8xf32>) -> tensor<2x8xf32>
    return %1 : tensor<2x8xf32>
  }
}

// CHECK: arith.constant
// CHECK: tensor.empty
// CHECK: linalg.fill
// CHECK: linalg.matmul_transpose_b
// CHECK: linalg.generic
// CHECK-NOT: "mini.linear"
// CHECK-NOT: "mini.relu"

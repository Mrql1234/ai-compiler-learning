// RUN: %mini_compiler_opt --mini-fusion %s | FileCheck %s

module {
  func.func @fuse_linear_relu(%arg0: tensor<2x4xf32>) -> tensor<2x8xf32> {
    %w = "mini.constant"() {value = dense<0.0> : tensor<8x4xf32>} : () -> tensor<8x4xf32>
    %b = "mini.constant"() {value = dense<0.0> : tensor<8xf32>} : () -> tensor<8xf32>
    %0 = "mini.linear"(%arg0, %w, %b) : (tensor<2x4xf32>, tensor<8x4xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    %1 = "mini.relu"(%0) : (tensor<2x8xf32>) -> tensor<2x8xf32>
    return %1 : tensor<2x8xf32>
  }
}

// CHECK: func.func @fuse_linear_relu
// CHECK: "mini.fused_linear_relu"
// CHECK-NOT: "mini.linear"
// CHECK-NOT: "mini.relu"

// RUN: %mini_compiler_opt --mini-lower-to-linalg --one-shot-bufferize %s | FileCheck %s

module {
  func.func @bufferize_linear_relu(%arg0: tensor<2x4xf32>) -> tensor<2x8xf32> {
    %w = "mini.constant"() {value = dense<0.0> : tensor<8x4xf32>} : () -> tensor<8x4xf32>
    %b = "mini.constant"() {value = dense<0.0> : tensor<8xf32>} : () -> tensor<8xf32>
    %0 = "mini.linear"(%arg0, %w, %b) : (tensor<2x4xf32>, tensor<8x4xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    %1 = "mini.relu"(%0) : (tensor<2x8xf32>) -> tensor<2x8xf32>
    return %1 : tensor<2x8xf32>
  }
}

// CHECK: memref.global
// CHECK: bufferization.to_buffer
// CHECK: memref.alloc
// CHECK: linalg.matmul
// CHECK: bufferization.to_tensor
// CHECK-NOT: "mini.linear"
// CHECK-NOT: "mini.relu"
